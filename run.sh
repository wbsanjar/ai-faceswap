#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  echo "Installing dependencies (first run only)..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-local.txt
fi

echo "Starting FaceSwap Studio at http://localhost:5000 (Ctrl+C to stop)"

# Stop any old/stuck server still holding port 5000.
if command -v fuser >/dev/null 2>&1; then
  fuser -k 5000/tcp 2>/dev/null || true
fi

exec .venv/bin/python backend/server.py