# FaceSwap Studio

A polished, self-hosted face-swap website. Upload two photos, click **Swap faces**, and get a photorealistic **AI** face swap in a couple of seconds. Everything runs locally on your machine — no account, no watermark, no uploads to third parties.

![Stack](https://img.shields.io/badge/Flask-3-blue) ![Stack](https://img.shields.io/badge/InsightFace%20inswapper-128-green) ![Stack](https://img.shields.io/badge/MediaPipe-1.0-blue) ![Stack](https://img.shields.io/badge/OpenCV-5-blue)

## How the swap works

The core is **InsightFace's `inswapper_128`** deep-learning face-swap model (the same technique used by popular deepfake tools):

1. **Detect** — MediaPipe's face mesh first (precise eye/mouth/chin points); YuNet falls back for hard photos.
2. **Embed** — ArcFace (`arcface_w600k_r50`) turns the aligned source face into a 512-d identity vector.
3. **Swap** — the identity vector is mapped through the model's latent `emap` and the encoder-decoder re-renders the target face with the source identity.
4. **Paste** — a diff-based mask blends the result back without visible seams.

Running on CPU, a swap takes about **1–2 seconds** after warm-up (first request loads the models).

## Face swap core logic

The conceptual pipeline behind any face-swap system, from detection to the final image:

```
Source Image (Face A)
        ↓
Face Detection
        ↓
Face Landmarks
(eyes, nose, mouth, jaw)
        ↓
Face Alignment
        ↓
Face Embedding / Identity Features
        ↓
Target Image (Face B)
        ↓
Target Face Detection + Landmarks
        ↓
Face Transformation / Swap
        ↓
Mask + Seamless Blending
        ↓
Color / Lighting Matching
        ↓
Final High-Quality Image
```

### 1. Face Detection

Locate the face in the image first.

Popular options: **MediaPipe Face Detection**, **RetinaFace**, **InsightFace**.

### 2. Face Landmarks

Detect the key facial points — usually the eyes, nose, mouth, and jawline:

```
      •       •
   eyes     eyes
        \   /
        •       ← nose
      •   •     ← mouth
    •       •
```

### 3. Face Alignment

Transform the source and target faces to the same orientation and scale so the swap fits properly:

```
Source Face          Target Face
    ↗                     ↖
     ↓                     ↓
  Alignment            Alignment
        \               /
         → Same geometry
```

### 4. Identity Preservation

This part decides how recognizable the swapped identity stays. A face-recognition model extracts an identity representation (embedding) from the source face. **InsightFace / ArcFace-family** models are commonly used for this.

### 5. Actual Face Swap

Here the AI model synthesizes the source identity onto the target's facial-region geometry and pose. Two common approaches:

**A. Traditional** — landmarks → affine transform → face mask → blend.

**B. AI-based** — source identity + target pose/expression → face-swap model → generated face.

For natural results, the AI-based approach is generally better.

### 6. Blending — very important

Simply pasting the face on top looks obvious because of the visible boundary. Instead, use:

```
AI Face → Mask → Color matching → Poisson/seamless blending → Natural transition
```

### 7. Final Enhancement

Finish with color/skin-tone matching, lighting matching, sharpening, and resolution restoration.

## Quick start

### Windows

Double-click `run.bat`. It creates a virtual environment on first run, installs dependencies, and starts the server.

To keep the server running **without a console window**, use `start-bg.bat` once (it runs in the background and writes logs to `server.log`); stop it later with `stop.bat`.

> Your uploaded photos and the last result are saved in the browser (IndexedDB),
> so refreshing the page — or reopening the site later — keeps everything. No re-uploading.

### macOS / Linux

```bash
./run.sh
```

### Manual

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-local.txt     # Windows
.venv/bin/pip install -r requirements-local.txt          # macOS/Linux
.venv\Scripts\python backend\server.py                   # Windows
.venv/bin/python backend/server.py                       # macOS/Linux
```

Then open **http://localhost:5000**.

> **First run downloads the models (~730 MB)** into `backend/models/`:
> `inswapper_128.onnx` (555 MB), `arcface_w600k_r50.onnx` (174 MB),
> the MediaPipe landmarker and YuNet detector. The identity latent map
> (`inswapper_128.emap.npy`) is extracted and cached automatically.

## Project layout

```
.
├── backend/
│   ├── server.py        # Flask app: serves the site + /api/swap
│   ├── ai_swap.py       # deep-learning swap (inswapper + ArcFace) — primary
│   ├── faceswap.py      # geometric mesh-swap fallback engine
│   └── models/          # auto-downloaded ONNX/TFLite models (gitignored)
├── frontend/
│   ├── index.html       # landing page + studio UI
│   ├── style.css        # dark, glassy responsive theme
│   └── app.js           # drag-drop uploads, fetch + download handling
├── app.py               # Vercel entrypoint (re-exports the Flask app)
├── pyproject.toml       # Vercel dependencies (no mediapipe; see Deploy)
├── vercel.json          # Vercel function settings (300 s, 2 GB)
├── .vercelignore        # keeps ~700 MB of models out of the bundle
├── requirements-local.txt  # full local stack (mediapipe + onnx)
├── run.bat / run.sh     # one-click launchers (foreground)
├── start-bg.bat         # Windows: start server in the background (no console)
├── stop.bat             # Windows: stop the background server
├── background.vbs       # hidden-process helper used by start-bg.bat
└── README.md
```

## API

`POST /api/swap` with multipart fields `source` and `target`:

```bash
curl -F "source=@myface.jpg" -F "target=@friend.jpg" http://localhost:5000/api/swap
```

Returns `{"image": "<base64 png>", "mime": "image/png", "encoding": "base64"}`.
If the AI engine cannot run, the server transparently falls back to the
geometric engine.

## Responsible use

Only swap faces you own or have permission to edit. Don't use this to impersonate people,
spread misinformation, or create non-consensual content. Processing happens in memory and
is discarded when each request finishes.

## Deploying to Vercel

This app needs ~700 MB of models, so a Vercel function needs special handling:

* The function bundle is capped at **500 MB** (standard) / **5 GB** (Large Functions),
  and the Hobby source upload is capped at **100 MB** — the `.onnx` models can't ride
  inside the deployment.
* In production, the code **downloads the models into the function's writable `/tmp`
  on cold start** (`_runtime_model_dir` in `backend/faceswap.py`) and reuses them while
  the instance is warm. Each new instance re-downloads, so cold starts take minutes.
* `pyproject.toml` deliberately lists **no mediapipe/onnx** — those are huge and their
  absence just drops to the YuNet detector. The tiny identity latent map
  (`inswapper_128.emap.npy`) IS tracked and shipped, so no extraction is needed at runtime.

Settings (`vercel.json`): `maxDuration: 300`, `memory: 2048` (Hobby caps both).

```bash
vercel login
vercel link
vercel deploy --prod
```

Warm the function once it's live, otherwise your first visitor waits minutes:

```bash
curl -X POST https://YOUR-APP.vercel.app/api/swap \
  -F "source=@photoA.jpg" -F "target=@photoB.jpg"
```

Trade-offs to know:

* **Cold starts are slow** (a one-off ~30–180 s hit to fetch 700 MB); keeping a stub
  WARMUP timer/ping helps.
* **Hobby disk**: `/tmp` holds the models between requests, but instances scale to zero.
* **Concurrency**: parallel requests spin up more instances, each re-downloading — keep
  traffic low or move the models into a Large Function + `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`
  on a Pro plan (then remove the `.onnx`/`.task` lines from `.vercelignore`).
* This is a genuinely heavy ML workload; a long-lived container (Hugging Face Spaces,
  Render, Fly.io, or a VPS) is faster and cheaper per request if Vercel's cold starts
  get annoying.