#!/bin/bash
# Install feedforward deps: vggt-x + mapanything
#
# Must run with nerfstudio env active, or use the full python path.
# Guards: torch 2.4.0+cu121, torchvision 0.19.0+cu121 — nothing upgrades these.
#
# Env constraints enforced:
#   torch==2.4.0+cu121    — CUDA 12.1 gsplat-rade kernels must not change
#   torchvision==0.19.0+cu121
#
# timm>=0.9: pyproject declares timm>=0.9,<2.0; nerfstudio patched to timm>=0.6.7.
# uniception's perception_encoder imports timm.layers (added in 0.9).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/opt/conda/envs/nerfstudio/bin/python
PIP=/opt/conda/envs/nerfstudio/bin/pip
# Prevent pip build isolation from using system Python 3.13 instead of env Python
export PIP_NO_BUILD_ISOLATION=1

# Ensure VGGT-X submodule is initialized (guards against clone without --recurse-submodules)
git -C "$SCRIPT_DIR" submodule update --init third_party/VGGT-X 2>/dev/null || true
CONSTRAINTS="$SCRIPT_DIR/constraints_feedforward.txt"

echo "=== feedforward install: vggt-x + mapanything ==="
echo "Using constraints: $CONSTRAINTS"

# vggt-x
# --no-deps: avoids torch==2.3.1 / torchvision==0.18.1 upgrade
$PIP install "$SCRIPT_DIR/third_party/VGGT-X" --no-deps -q
$PIP install "viser==0.2.23" evo pyliblzfse safetensors roma kornia \
    -c "$CONSTRAINTS" -q

# mapanything
# --no-deps: avoids opencv-python-headless==4.10.0.84 conflict with opencv-python
$PIP install 'git+https://github.com/facebookresearch/map-anything.git' --no-deps -q
$PIP install huggingface_hub hydra-core natsort orjson pillow-heif plyfile \
    python-box requests tensorboard tqdm \
    -c "$CONSTRAINTS" -q
# uniception==0.1.7 is mapanything's core model dep
$PIP install 'uniception==0.1.7' -c "$CONSTRAINTS" -q
# colmap extra: needed for MapAnythingCreator reconstruction pipeline
$PIP install 'lightglue @ git+https://github.com/cvg/LightGlue.git' --no-deps -q
$PIP install open3d -c "$CONSTRAINTS" -q

# loop closure deps
# gtsam-develop: PyPI 4.3a1; has SL4/PriorFactorSL4/BetweenFactorSL4 required by VGGT-SLAM 2.0
$PIP install gtsam-develop -q
# salad: Dominic101/salad is a pip-installable fork of serizba/salad with a proper
# Python package structure (salad.models_salad.*). pytorch-metric-learning is its
# only non-torch runtime dep.
$PIP install -q git+https://github.com/Dominic101/salad.git
$PIP install pytorch-metric-learning -q

# vggt-omega
# --no-deps: skips numpy<2 metadata constraint; env has numpy 2.4.x (same bypass as VGGT-X)
echo "=== Installing vggt-omega ==="
if [ ! -f "$SCRIPT_DIR/third_party/vggt-omega/pyproject.toml" ]; then
    echo "ERROR: third_party/vggt-omega submodule not initialized"
    echo "       Run: git submodule update --init third_party/vggt-omega"
    exit 1
fi
$PIP install --no-deps -e "$SCRIPT_DIR/third_party/vggt-omega" -q

# collab-splats feedforward extras (non-git deps declared in pyproject.toml)
$PIP install -e '.[feedforward]' --no-deps -q

# Invariant checks
$PYTHON - << 'PYEOF'
import torch, timm
assert "12.1" in torch.version.cuda, f"FAIL: CUDA changed to {torch.version.cuda}"
assert torch.__version__.startswith("2.4"), f"FAIL: torch upgraded to {torch.__version__}"
print(f"[OK] torch={torch.__version__}  cuda={torch.version.cuda}  timm={timm.__version__}")
PYEOF

# Smoke test
$PYTHON -c "
from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
print('[OK] VGGTXCreator, MapAnythingCreator, and VGGTOmegaCreator import successfully')
"

echo ""
echo "=== feedforward install complete ==="
