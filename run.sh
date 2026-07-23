#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="venv"
APP_MODULE="app.main:app"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

PYTHON="$VENV_DIR/bin/python3"

# Apple Silicon requires x86_64 for certain native extensions (e.g. XGBoost)
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    exec arch -x86_64 "$PYTHON" -m uvicorn "$APP_MODULE" --reload "$@"
else
    exec "$PYTHON" -m uvicorn "$APP_MODULE" --reload "$@"
fi