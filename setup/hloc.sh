#!/usr/bin/env bash
set -euo pipefail

VENDOR_DIR="$(dirname "$0")/third_party/hloc"
PIP="/opt/venv/reconstruction/bin/pip"
PYTHON="/opt/venv/reconstruction/bin/python"

echo "==> hloc setup"

if [ -d "$VENDOR_DIR" ]; then
    echo "    third_party/hloc/ already exists — skipping clone"
else
    echo "    Cloning Hierarchical-Localization with submodules..."
    git clone --recursive https://github.com/cvg/Hierarchical-Localization "$VENDOR_DIR"
fi

echo "    Installing hloc into reconstruction env..."
"$PIP" install -e "$VENDOR_DIR"

echo "    Verifying install..."
"$PYTHON" -c "import hloc; print('hloc', hloc.__version__, 'installed OK')"

echo "==> Done."
