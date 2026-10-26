#!/bin/bash
# Build a standalone "Tower Designer.app" (no Python needed on the target Mac).
# Usage: ./packaging/build_mac_app.sh   (from the repository root, with the requirements installed)
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -m pip install pyinstaller >/dev/null 2>&1 || true
"$PY" -m PyInstaller --noconfirm --clean --windowed \
  --name "Tower Designer" \
  --icon "Tower Designer.app/Contents/Resources/AppIcon.icns" \
  --osx-bundle-identifier io.github.towerdesigner \
  --add-data "towerdesigner/examples:towerdesigner/examples" \
  --collect-submodules towerdesigner \
  --collect-all cadquery --collect-all OCP --collect-all pyqtgraph --collect-all ezdxf \
  --exclude-module tkinter --exclude-module matplotlib \
  --distpath dist --workpath build \
  packaging/run_towerdesigner.py
echo "Built: dist/Tower Designer.app"
