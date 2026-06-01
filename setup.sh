#!/bin/bash
# Single source of truth for env setup. Runs in the Docker build AND standalone.
# Floor required from the host/image: gcc/g++ (build-essential) + (at runtime) NVIDIA driver.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/opt/venv/reconstruction
PYTHON="$VENV/bin/python"
export PIP_ROOT_USER_ACTION=ignore
export UV_PROJECT_ENVIRONMENT="$VENV"

# Precondition: vendored third_party packages must exist (editable path deps vggt / vggt-omega).
# They are NOT git submodules; an onboarding/clone step must have populated them.
for d in third_party/VGGT-X third_party/vggt-omega; do
    if [ ! -f "$SCRIPT_DIR/$d/pyproject.toml" ]; then
        echo "ERROR: $d is missing (vendored package, not a submodule)."
        echo "       Populate third_party/ before running setup.sh."
        exit 1
    fi
done

# CUDA build environment — nvcc + headers come from the pip cuda-toolkit wheels (Option A).
# These dirs are populated during `uv sync` (cuda-toolkit is a wheel, installed before the
# no-build-isolation source builds); env vars resolve at compile time, after install.
NV="$VENV/lib/python3.11/site-packages/nvidia"
export CUDA_HOME="$NV/cuda_nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$NV/cuda_runtime/include:$NV/cuda_cccl/include:$NV/cuda_nvcc/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NV/cuda_runtime/lib:$NV/cuda_nvcc/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NV/cuda_runtime/lib:$NV/cuda_nvcc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;8.9;8.6;8.0;7.5;7.0}"

echo "=== uv sync: full env (all extras incl. gpu toolkit + feedforward) ==="
cd "$SCRIPT_DIR"
/root/.local/bin/uv sync --all-extras

# collab-data: private repo, kept out of the locked graph; installed explicitly post-sync.
echo "=== install collab-data (private) ==="
/root/.local/bin/uv pip install --python "$PYTHON" "git+https://github.com/BasisResearch/collab-data.git"

# Smoke test
"$PYTHON" - << 'PYEOF'
import torch
assert "12.1" in torch.version.cuda, f"FAIL: cuda={torch.version.cuda}"
assert torch.__version__.startswith("2.5"), f"FAIL: torch={torch.__version__}"
from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
print(f"[OK] torch={torch.__version__} cuda={torch.version.cuda}; creators import")
PYEOF
echo "=== setup complete ==="
