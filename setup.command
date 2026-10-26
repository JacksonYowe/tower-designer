#!/bin/bash
# One-time setup for Tower Designer. Double-click this file in Finder.
# The Python environment lives in ~/.towerdesigner (Qt can't load its plugins from inside ~/Documents).
set -e
cd "$(dirname "$0")"
VENV="$HOME/.towerdesigner/venv"
echo "Setting up Tower Designer in $VENV ..."
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" -r requirements.txt
else
  PY=$(command -v python3.12 || command -v python3)
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements.txt
fi
echo
echo "Done. Open 'Tower Designer.app'."
