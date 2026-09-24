"""Face swap engine.

Pipeline:
  1. MediaPipe FaceLandmarker for 478 landmarks (full image).
  2. If that fails, YuNet FaceDetectorYN finds the face box, the region is
     cropped and upscaled, and landmarks are re-estimated on the crop.
  3. Eye-aligned source + Delaunay mesh warp + feathered blend.
  4. If no landmarks at all, a YuNet-box elliptical paste is used so the
     tool still produces a result for hard images.
"""

import os
import tempfile

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover
    mp = None
    mp_vision = None
    BaseOptions = None

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_LANDMARK_MODEL = os.path.join(_DEFAULT_MODEL_DIR, "face_landmarker.task")
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceSwapError(Exception):
    """Raised when a swap cannot be performed."""


def _download(url, dest):
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def _runtime_model_dir():
    """Writable model directory.

    Locally this is the bundled ``backend/models`` folder. On serverless
    platforms (e.g. Vercel) the bundle is read-only, so we fall back to the
    per-instance temp directory and download the models on demand.
    """
    try:
        os.makedirs(_DEFAULT_MODEL_DIR, exist_ok=True)
        probe = os.path.join(_DEFAULT_MODEL_DIR, ".write_probe")
        with open(probe, "w") as handle:
            handle.write("")
        os.remove(probe)
        return _DEFAULT_MODEL_DIR
    except OSError:
        writable = os.path.join(tempfile.gettempdir(), "faceswap-models")
        os.makedirs(writable, exist_ok=True)
        return writable


def _ensure_model(url, dest):
    """Download `url` to `dest` when missing or empty; raises on failure."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    _download(url, dest)
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise FaceSwapError(
            "Failed to download the model '{}'.".format(os.path.basename(dest))
        )


class _Landmarker:
    """Lazy, process-wide FaceLandmarker singleton."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            if mp_vision is None:
                return None
            if not os.path.exists(_LANDMARK_MODEL):
                try:
                    _download(
                        "https://storage.googleapis.com/mediapipe-models/"
                        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                        _LANDMARK_MODEL,
                    )
                except Exception:
                    return None
            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_LANDMARK_MODEL),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            try:
                cls._instance = mp_vision.FaceLandmarker.create_from_options(options)
            except Exception:
                cls._instance = None
        return cls._instance


class _Yunet:
    """Lazy YuNet face detector singleton."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            yunet_path = os.path.join(_runtime_model_dir(), "face_detection_yunet.onnx")
            if not os.path.exists(yunet_path):
                _ensure_model(_YUNET_URL, yunet_path)
            cls._instance = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320), score_threshold=0.6,
                nms_threshold=0.3, top_k=5000,
            )
        return cls._instance


# Landmark indices used for face alignment (MediaPipe mesh).


def _read_image(image_bytes):
    data = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FaceSwapError("Could not read the uploaded image.")
    if max(img.shape[:2]) > 1600:
        scale = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _encode_base64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise FaceSwapError("Failed to encode the result.")
    import base64

    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Face detection / landmarks
# ---------------------------------------------------------------------------

def _mediapipe_landmarks(img):
    landmarker = _Landmarker.get()
    if landmarker is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    h, w = img.shape[:2]
    result = landmarker.detect(frame)
    if not result.face_landmarks:
        return None
    return [(int(p.x * w), int(p.y * h)) for p in result.face_landmarks[0]]


def _yunet_face(img):
    """Returns ((x, y, w, h), 5 landmarks) for the largest detected face."""
    detector = _Yunet.get()
    if detector is None:
        return None
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, wf, hf = best[:4]
    landmarks = best[4:14].reshape(5, 2)
    return (int(x), int(y), int(wf), int(hf)), landmarks


def _yunet_box(img):
    face = _yunet_face(img)
    return face[0] if face else None


def _landmarks_robust(img):
    """Landmarks via MediaPipe; retries on a YuNet crop if the first pass fails."""
    pts = _mediapipe_landmarks(img)
    if pts is not None:
        return pts

    box = _yunet_box(img)
    if box is None:
        return None

    fx, fy, fw, fh = box
    pad_x, pad_y = int(fw * 0.35), int(fh * 0.35)
    h_img, w_img = img.shape[:2]
    x0, y0 = max(0, fx - pad_x), max(0, fy - pad_y)
    x1, y1 = min(w_img, fx + fw + pad_x), min(h_img, fy + fh + pad_y)

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    scale = 320 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    crop_pts = _mediapipe_landmarks(crop)
    if crop_pts is None:
        return None

    pts = [(int(px / scale) + x0, int(py / scale) + y0) for px, py in crop_pts]
    return pts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _umeyama_similarity(a, b):
    """Similarity transform (rotation + uniform scale + translation) mapping
    point set `a` onto point set `b` in the least-squares sense (Umeyama)."""
    a = np.float32(a)
    b = np.float32(b)
    n = a.shape[0]
    a_mean = a.mean(axis=0, keepdims=True)
    b_mean = b.mean(axis=0, keepdims=True)
    a_c = a - a_mean
    b_c = b - b_mean
    cov = b_c.T @ a_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, d]) @ vt
    var = float((np.linalg.norm(a_c) ** 2) / n)
    scale = (s.sum() / var) if var > 1e-9 else 1.0
    rot *= scale
    trans = (b_mean.T - rot @ a_mean.T).ravel()
    return rot, trans


_ANCHORS = (33, 133, 362, 263, 1, 4, 61, 291, 152)


def _anchors(points):
    """Stable facial anchor points: eyes, nose, mouth corners, chin."""
    return [np.float32(points[i]) for i in _ANCHORS]


def _rect_boundary_points(points, expand=0.25):
    """Hull bounding rect corner + edge midpoints, expanded a touch."""
    hull = cv2.convexHull(np.int32(points))
    x, y, w, h = cv2.boundingRect(hull)
    pad_x, pad_y = int(w * expand), int(h * expand)
    left, top = x - pad_x, y - pad_y
    right, bottom = x + w + pad_x, y + h + pad_y
    mid_x, mid_y = (left + right) // 2, (top + bottom) // 2
    return [(left, top), (mid_x, top), (right, top), (right, mid_y),
            (right, bottom), (mid_x, bottom), (left, bottom), (left, mid_y)]


def _triangulate(points):
    """Delaunay triangles over `points` via cv2.Subdiv2D."""
    pts = np.int32(points)
    x, y, w, h = cv2.boundingRect(pts)
    rect = (x - w, y - h, x + 2 * w, y + 2 * h)
    subdiv = cv2.Subdiv2D(rect)
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))
    lookup = {tuple(int(v) for v in p): i for i, p in enumerate(points)}
    triangles = []
    for t in subdiv.getTriangleList():
        idx = []
        for cx, cy in ((t[0], t[1]), (t[2], t[3]), (t[4], t[5])):
            best, best_d = None, 1e18
            for (px, py), i in lookup.items():
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                break
            idx.append(best)
        if len(idx) == 3:
            triangles.append(tuple(idx))
    return triangles


def _warp_triangle(src, dst, t1, t2):
    """Warp source triangle t1 into destination triangle t2, accumulating in dst."""
    r = cv2.boundingRect(np.float32([t2]))
    xs, ys, w, h = r
    t2_rel = [(p[0] - xs, p[1] - ys) for p in t2]
    m = cv2.getAffineTransform(np.float32(t1), np.float32(t2_rel))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rel), 1.0)
    warped = cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    roi = dst[ys:ys + h, xs:xs + w]
    mask3 = cv2.merge([mask, mask, mask])
    roi[:] = roi * (1.0 - mask3) + warped * mask3


# ---------------------------------------------------------------------------
# Swap implementations
# ---------------------------------------------------------------------------

def _match_face_color(src, dst, src_pts, dst_pts):
    """Shift the source face's LAB color distribution toward the target's."""
    def face_labs(img, pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.int32(pts)), 255)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab, mask

    src_lab, src_mask = face_labs(src, src_pts)
    dst_lab, dst_mask = face_labs(dst, dst_pts)

    src_vals = [src_lab[..., c][src_mask > 0].mean() for c in range(3)]
    dst_vals = [dst_lab[..., c][dst_mask > 0].mean() for c in range(3)]
    shift = np.float32(dst_vals) - np.float32(src_vals)

    soft = cv2.GaussianBlur((src_mask > 0).astype(np.float32), (31, 31), 0)
    soft = soft[..., None]
    adjusted = np.clip(src_lab + shift, 0, 255)
    merged = src_lab * (1.0 - soft) + adjusted * soft
    return cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _swap_mesh(src_aligned, dst, src_pts, dst_pts):
    """Triangulated mesh warp + Poisson blending.

    `src_aligned` is the color-matched source pre-warped onto `dst`.
    """
    boundary = _rect_boundary_points(dst_pts)
    tgt_list = list(dst_pts) + boundary
    src_list = list(src_pts) + boundary
    triangles = _triangulate(tgt_list)

    out = dst.astype(np.float32).copy()
    for t1i, t2i, t3i in triangles:
        _warp_triangle(src_aligned, out,
                       [src_list[t1i], src_list[t2i], src_list[t3i]],
                       [tgt_list[t1i], tgt_list[t2i], tgt_list[t3i]])

    warped = np.clip(out, 0, 255).astype(np.uint8)

    hull = cv2.convexHull(np.int32(dst_pts))
    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    pts = np.int32(dst_pts)
    cx = int(np.clip(np.mean(pts[:, 0]), 0, dst.shape[1] - 1))
    cy = int(np.clip(np.mean(pts[:, 1]), 0, dst.shape[0] - 1))

    return cv2.seamlessClone(warped, dst, mask, (cx, cy), cv2.NORMAL_CLONE)


def _swap_ellipse(src, dst, src_face, dst_face):
    """Landmark-free fallback: align with YuNet's 5 facial landmarks and
    finish with a feathered Poisson blend."""
    (_, _, _w1, _h1), src_lms = src_face
    (bx, by, dw, dh), dst_lms = dst_face
    dw, dh = max(dw, 1), max(dh, 1)

    rot, trans = _umeyama_similarity(np.float32(src_lms), np.float32(dst_lms))
    m = np.hstack([rot, trans.reshape(-1, 1)])
    h, w = dst.shape[:2]
    warped = cv2.warpAffine(src, m, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    render = _match_face_color(warped, dst, src_lms.astype(int), dst_lms.astype(int))

    pad_w, pad_h = int(dw * 0.3), int(dh * 0.3)
    rel = np.float32([[0.0, 0.0], [1.0, 0.0], [1.2, 0.4], [1.2, 0.9],
                      [1.0, 1.2], [0.0, 1.2], [-0.2, 0.9], [-0.2, 0.4]])
    box = np.int32(rel * [dw, dh] + [bx - pad_w, by - pad_h])
    poly = np.vstack([np.int32(dst_lms), box]).reshape(-1, 1, 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    cx = int(np.clip(np.mean(dst_lms[:, 0]), 0, w - 1))
    cy = int(np.clip(np.mean(dst_lms[:, 1]), 0, h - 1))
    return cv2.seamlessClone(render, dst, mask, (cx, cy), cv2.NORMAL_CLONE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def swap_images(source_bytes, target_bytes, prefer_low_level=False):
    """Swap the face from `source_bytes` onto the face in `target_bytes`.

    Returns PNG-encoded bytes.
    """
    src = _read_image(source_bytes)
    dst = _read_image(target_bytes)

    src_pts = None if prefer_low_level else _landmarks_robust(src)
    dst_pts = None if prefer_low_level else _landmarks_robust(dst)

    if src_pts and dst_pts:
        # Align the source face onto the target (multi-point least squares),
        # match skin tone, then run the mesh swap.
        rot, trans = _umeyama_similarity(_anchors(src_pts), _anchors(dst_pts))
        h, w = dst.shape[:2]
        m = np.hstack([rot, trans.reshape(-1, 1)])
        aligned = cv2.warpAffine(src, m, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        aligned_pts = _landmarks_robust(aligned)
        if aligned_pts is None:
            aligned_pts = cv2.transform(
                np.float32(src_pts).reshape(-1, 1, 2), m
            ).reshape(-1, 2).astype(int).tolist()
        render = _match_face_color(aligned, dst, aligned_pts, dst_pts)
        result = _swap_mesh(render, dst, aligned_pts, dst_pts)
        if result is not None:
            return _encode_base64(result)

    src_box = _yunet_box(src)
    dst_box = _yunet_box(dst)
    if src_box and dst_box:
        return _encode_base64(_swap_ellipse(src, dst, _yunet_face(src), _yunet_face(dst)))

    missing = []
    if _yunet_box(src) is None:
        missing.append("source")
    if _yunet_box(dst) is None:
        missing.append("target")
    if missing:
        raise FaceSwapError(
            "Could not detect a face in the {} image. "
            "Try a clearer, front-facing photo.".format(" and ".join(missing))
        )
    raise FaceSwapError("Face detection failed. Try different photos.")