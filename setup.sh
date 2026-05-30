#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP=/opt/venv/reconstruction/bin/pip
export PIP_ROOT_USER_ACTION=ignore

echo "=== Pre-step a: install cuDSS (must land on disk before bae builds; bae find_cudss_root() scans site-packages/nvidia/cu12 at build time, so it cannot share bae's pip transaction) ==="
$PIP install --no-build-isolation "nvidia-cudss-cu12==0.6.0.5"

echo "=== Pre-step b: build bae CUDA extension (--no-build-isolation required; PIP_NO_BUILD_ISOLATION=1 does not propagate to dependency builds) ==="
CUDA_HOME=/usr/local/cuda \
    $PIP install --no-build-isolation \
        "bae @ git+https://github.com/pypose/bae.git@0.2.4"

echo "=== Step 1: install collab-splats ==="
$PIP install -e "$SCRIPT_DIR"

echo "=== Step 2: install collab-data (private) ==="
$PIP install git+https://github.com/BasisResearch/collab-data.git

echo "=== Step 3: install co3d eval dependency (--no-deps required) ==="
$PIP install git+https://github.com/facebookresearch/co3d.git --no-deps

echo "=== Step 4: install feedforward deps (vggt-x, mapanything, evo, gtsam, salad, vggt-omega) ==="
bash "$SCRIPT_DIR/setup/feedforward.sh"

echo "=== Step 5: install dashboard deps (panel, param) ==="
$PIP install -e "$SCRIPT_DIR[dashboard]"
