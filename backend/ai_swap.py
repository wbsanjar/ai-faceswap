"""Deep-learning face swap using InsightFace's inswapper_128 + ArcFace.

Runs fully on CPU via onnxruntime. Models are downloaded once from the
facefusion-assets GitHub release if they are missing. The preprocessing,
the identity-latent transform (emap) and the diff-based paste-back faithfully
mirror InsightFace's reference implementation.
"""

import os
import shutil
import threading
import time
import traceback

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from faceswap import (
    FaceSwapError, _encode_base64, _ensure_model, _mediapipe_landmarks,
    _read_image, _runtime_model_dir, _umeyama_similarity, _yunet_face,
)

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_SWAPPER_FN = "inswapper_128.onnx"
_ARCFACE_FN = "arcface_w600k_r50.onnx"
_EMAP_FN = "inswapper_128.emap.npy"

_BASE_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
_SWAPPER_URL = f"{_BASE_URL}/{_SWAPPER_FN}"
_ARCFACE_URL = f"{_BASE_URL}/{_ARCFACE_FN}"

# ArcFace canonical 5-point template for a 112px crop.
_ARCFACE_DST = np.float32(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]]
)

_LOCK = threading.RLock()


def _load_emap(emap_path, swapper_path):
    """Resolve the swapper's learned latent map.

    Prefers the packaged ``.npy`` (copied from the bundle when running in a
    read-only serverless filesystem) and only extracts it from the swapper's
    graph initializers as a last resort.
    """
    if os.path.exists(emap_path) and os.path.getsize(emap_path) > 0:
        return np.load(emap_path)
    bundled = os.path.join(_DEFAULT_MODEL_DIR, _EMAP_FN)
    if os.path.exists(bundled) and os.path.getsize(bundled) > 0:
        os.makedirs(os.path.dirname(emap_path), exist_ok=True)
        shutil.copyfile(bundled, emap_path)
        return np.load(emap_path)
    try:
        import onnx  # noqa: WPS433
    except Exception as exc:
        raise FaceSwapError(
            "The identity latent map is missing; install 'onnx' once to build it."
        ) from exc
    model = onnx.load(swapper_path)
    emap = onnx.numpy_helper.to_array(model.graph.initializer[-1])
    os.makedirs(os.path.dirname(emap_path), exist_ok=True)
    np.save(emap_path, emap)
    return emap


class _Sessions:
    """Lazy-loaded onnxruntime sessions (guarded by our own lock)."""

    _swapper = None
    _arcface = None
    _emap = None

    @classmethod
    def _loaded(cls):
        return cls._swapper is not None and cls._arcface is not None and cls._emap is not None

    @classmethod
    def _load(cls):
        if cls._loaded():
            return
        if ort is None:
            raise FaceSwapError("onnxruntime is not installed.")
        model_dir = _runtime_model_dir()
        swapper = os.path.join(model_dir, _SWAPPER_FN)
        arcface = os.path.join(model_dir, _ARCFACE_FN)
        emap = os.path.join(model_dir, _EMAP_FN)
        _ensure_model(_SWAPPER_URL, swapper)
        _ensure_model(_ARCFACE_URL, arcface)
        swapper_sess = ort.InferenceSession(
            swapper, providers=["CPUExecutionProvider"]
        )
        try:
            arcface_sess = ort.InferenceSession(
                arcface, providers=["CPUExecutionProvider"]
            )
            emap_arr = _load_emap(emap, swapper)
        except BaseException:
            del swapper_sess
            raise
        cls._swapper = swapper_sess
        cls._arcface = arcface_sess
        cls._emap = emap_arr

    @classmethod
    def get(cls):
        with _LOCK:
            if not cls._loaded():
                cls._load()
            return cls._swapper, cls._arcface, cls._emap


# Warm-up state. The heavy model download/load runs off the request path so a
# cold function responds fast with "warming" instead of hanging the browser.
_WARM_STATE = {"status": "cold", "error": None}  # cold | warming | ready | failed
_WARM_LOCK = threading.Lock()
_WARM_LAST_ATTEMPT = [0.0]
_RETRY_COOLDOWN_S = 30.0


def warm_status():
    """Current warm-up state plus the last failure reason, if any."""
    with _WARM_LOCK:
        return dict(_WARM_STATE)


def ai_ready():
    with _WARM_LOCK:
        return _WARM_STATE["status"] == "ready"


def ensure_warm(force=False):
    """Start (or restart) the background model warm-up exactly once.

    A failed attempt is not retried more than once every `_RETRY_COOLDOWN_S`
    seconds, so a broken download cannot make every request re-allocate the
    ~1 GB of model weights.
    """
    with _WARM_LOCK:
        if _WARM_STATE["status"] in ("warming", "ready"):
            return dict(_WARM_STATE)
        if (not force and _WARM_STATE["status"] == "failed"
                and time.monotonic() - _WARM_LAST_ATTEMPT[0] < _RETRY_COOLDOWN_S):
            return dict(_WARM_STATE)
        _WARM_STATE["status"] = "warming"
        _WARM_STATE["error"] = None
        _WARM_LAST_ATTEMPT[0] = time.monotonic()

    def _run():
        try:
            _Sessions.get()
            with _WARM_LOCK:
                _WARM_STATE["status"] = "ready"
        except BaseException as exc:
            with _WARM_LOCK:
                _WARM_STATE["status"] = "failed"
                _WARM_STATE["error"] = "%s: %s" % (type(exc).__name__, exc)
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()
    return dict(_WARM_STATE)


def warm_up():
    """Compatibility helper — starts the background warm-up thread."""
    ensure_warm()


def _align_crop(img, kps, size):
    """Align `kps` onto the ArcFace template at `size` and crop."""
    dst = _ARCFACE_DST * (size / 112.0)
    rot, trans = _umeyama_similarity(np.float32(kps), dst)
    m = np.hstack([rot, trans.reshape(-1, 1)])
    crop = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR)
    return crop, m


# MediaPipe mesh indices mapped onto the canonical 5-point ArcFace order:
# [left eye, right eye, nose tip, left mouth corner, right mouth corner].
_MP_EYE_LEFT = (33, 133)
_MP_EYE_RIGHT = (362, 263)
_MP_NOSE = 1
_MP_MOUTH_LEFT = 61
_MP_MOUTH_RIGHT = 291


def _mediapipe_kps(img):
    """5-point arcface-ordered keypoints from the MediaPipe face mesh."""
    pts = _mediapipe_landmarks(img)
    if pts is None or len(pts) < 292:
        return None
    mid = lambda i, j: ((pts[i][0] + pts[j][0]) / 2.0, (pts[i][1] + pts[j][1]) / 2.0)
    return np.float32([
        mid(_MP_EYE_LEFT[0], _MP_EYE_LEFT[1]),
        mid(_MP_EYE_RIGHT[0], _MP_EYE_RIGHT[1]),
        pts[_MP_NOSE],
        pts[_MP_MOUTH_LEFT],
        pts[_MP_MOUTH_RIGHT],
    ])


def _detect_kps(img):
    """Highest-precision 5-point landmarks available (MediaPipe, else YuNet)."""
    kps = _mediapipe_kps(img)
    if kps is not None:
        return kps
    face = _yunet_face(img)
    return np.float32(face[1]) if face else None


def _normed_embedding(img, kps):
    """512-d ArcFace embedding for the aligned face, L2-normalized."""
    crop, _ = _align_crop(img, kps, 112)
    blob = ((crop[..., ::-1] - 127.5) / 127.5).transpose(2, 0, 1)
    blob = blob[None].astype(np.float32)
    _, arcface, _ = _Sessions.get()
    emb = arcface.run(None, {"input": blob})[0][0]
    return (emb / max(np.linalg.norm(emb), 1e-9)).astype(np.float32)


def _swapper_latent(source_embedding):
    """Map the arcface identity into the swapper's latent space (emap)."""
    _, _, emap = _Sessions.get()
    latent = np.dot(source_embedding.reshape(1, -1), emap)
    return (latent / max(np.linalg.norm(latent), 1e-9)).astype(np.float32)


def _sharpen(img, amount=0.3, radius=1.5):
    """Mild unsharp mask; compensates for the model's low-res 128px output."""
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _match_color(img, bgr_fake, mask, strength=0.6):
    """Skew the swapped patch's colour/lighting towards the target face.

    Reinhard-style transfer computed on the eroded core of the blend mask,
    then blended with the original patch by `strength` to avoid over-matching.
    """
    core = cv2.erode((mask * 255).astype(np.uint8), np.ones((9, 9), np.uint8),
                     iterations=2)
    sel = core >= 200
    if sel.sum() < 300:
        return bgr_fake

    lab_t = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_f = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2LAB).astype(np.float32)

    t = lab_t[sel]
    f = lab_f[sel]
    mu_t, mu_f = t.mean(0), f.mean(0)
    sd_t, sd_f = t.std(0) + 1e-6, f.std(0) + 1e-6

    remapped = (lab_f - mu_f) / sd_f * sd_t + mu_t
    corr = lab_f * (1 - strength) + remapped * strength
    corr = np.clip(corr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corr, cv2.COLOR_LAB2BGR)


def _execute_swap(src_img, dst_img, src_kps, dst_kps):
    """Run the swap model on the aligned target crop."""
    with _LOCK:
        swapper, _, _ = _Sessions.get()

        aimg, m128 = _align_crop(dst_img, dst_kps, 128)
        blob = ((aimg[..., ::-1] / 255.0)).transpose(2, 0, 1)  # RGB, [0, 1]
        blob = blob[None].astype(np.float32)

        latent = _swapper_latent(_normed_embedding(src_img, src_kps))

        out = swapper.run(None, {"target": blob, "source": latent})[0][0]
        bgr_fake = np.clip(255 * out, 0, 255).astype(np.uint8)
        bgr_fake = bgr_fake.transpose(1, 2, 0)[..., ::-1]  # CHW RGB -> HWC BGR
        bgr_fake = _sharpen(bgr_fake, amount=0.35, radius=1.5)
        return bgr_fake, aimg, m128


def _paste_back(img, aimg, bgr_fake, m128, color_strength=0.6):
    """Feathered coverage-mask paste back with colour/lighting transfer.

    The aligned 128px patch is warped back with LANCZOS, blended through an
    eroded + blurred coverage mask (hair/background outside the face stays
    untouched), and a Reinhard colour/lighting match pulls the patch's tone
    towards the target face.
    """
    inv = cv2.invertAffineTransform(m128)
    h, w = img.shape[:2]
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)

    bgr_fake = cv2.warpAffine(bgr_fake, inv, (w, h),
                              flags=cv2.INTER_LANCZOS4, borderValue=0.0)
    img_white = cv2.warpAffine(img_white, inv, (w, h), borderValue=0.0)

    img_white[img_white > 20] = 255
    img_mask = img_white.copy()

    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) == 0:
        return img
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))

    k = max(mask_size // 10, 10)
    img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)

    k = max(mask_size // 20, 5)
    k = k + 1 if k % 2 == 0 else k
    img_mask = cv2.GaussianBlur(img_mask, (k, k), 0)

    if color_strength > 0:
        bgr_fake = _match_color(img, bgr_fake, img_mask, color_strength)

    img_mask = (img_mask / 255.0).reshape(h, w, 1)
    merged = img_mask * bgr_fake.astype(np.float32) + \
        (1 - img_mask) * img.astype(np.float32)
    return merged.astype(np.uint8)


def ai_face_swap(src_img, dst_img):
    """Swap the face of `src_img` onto `dst_img`. Returns a BGR image."""
    src_kps = _detect_kps(src_img)
    dst_kps = _detect_kps(dst_img)
    if src_kps is None or dst_kps is None:
        return None

    bgr_fake, aimg, m128 = _execute_swap(src_img, dst_img, src_kps, dst_kps)
    return _paste_back(dst_img, aimg, bgr_fake, m128)


def swap_images(source_bytes, target_bytes):
    """AI swap entry point; raises FaceSwapError if it cannot run."""
    src = _read_image(source_bytes)
    dst = _read_image(target_bytes)
    result = ai_face_swap(src, dst)
    if result is None:
        raise FaceSwapError(
            "Could not detect a face in one of the images with the AI detector."
        )
    return _encode_base64(result)