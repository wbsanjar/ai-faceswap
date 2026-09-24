"""Flask server: serves the frontend and exposes the /api/swap endpoint."""

import os
import sys
import traceback

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
import ai_swap
from ai_swap import swap_images as ai_swap_images
from faceswap import FaceSwapError, swap_images as geometric_swap

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
MAX_UPLOAD = 16 * 1024 * 1024

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

# Warm the AI models at import time (fires during the function's init phase
# on Vercel, and at startup locally) so swaps don't stall on the first request.
ai_swap.ensure_warm()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "ai": ai_swap.ai_ready()})


@app.post("/api/swap")
def api_swap():
    source = request.files.get("source")
    target = request.files.get("target")
    if not source or not target:
        return jsonify({"error": "Both a source and a target image are required."}), 400

    source_data = source.read()
    target_data = target.read()

    if not ai_swap.ai_ready():
        ai_swap.ensure_warm()
        return jsonify({
            "status": "warming",
            "message": "AI models are loading (first launch can take a few minutes). "
                       "Retrying automatically…",
        }), 202

    png_data = _try_swap(ai_swap_images, source_data, target_data)
    if png_data is None:
        png_data = _try_swap(geometric_swap, source_data, target_data)
    if png_data is None:
        return jsonify({
            "error": "No usable face swap could be produced. Try clearer, "
                     "front-facing photos and reload the page if the models "
                     "were still downloading."
        }), 422

    return jsonify({"image": png_data, "encoding": "base64", "mime": "image/png"})


def _try_swap(fn, source_data, target_data):
    """Run a swap engine; return base64 PNG or None on any failure."""
    try:
        return fn(source_data, target_data)
    except FaceSwapError as exc:
        print(f"[swap] {fn.__module__}: {exc}")
        return None
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        return None


@app.errorhandler(413)
def too_large(_err):
    return jsonify({"error": "Image is too large (max 16 MB)."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)