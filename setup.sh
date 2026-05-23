#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/opt/conda/envs/nerfstudio/bin/python
PIP=/opt/conda/envs/nerfstudio/bin/pip
export PIP_NO_BUILD_ISOLATION=1

echo "=== Step 1: install local nerfstudio ==="
bash "$SCRIPT_DIR/setup_nerfstudio.sh"

echo "=== Step 2: install collab-splats ==="
$PIP install -e "$SCRIPT_DIR"

echo "=== Step 3: install collab-data (private) ==="
$PIP install git+https://github.com/BasisResearch/collab-data.git

echo "=== Step 4: install co3d eval dependency (--no-deps required) ==="
$PIP install git+https://github.com/facebookresearch/co3d.git --no-deps
