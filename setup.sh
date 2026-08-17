#!/bin/bash
# Single source of truth for env setup. Runs in the Docker build AND standalone.
# Floor required from the host/image: gcc/g++ (build-essential) + (at runtime) NVIDIA driver.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/opt/venv/reconstruction
PYTHON="$VENV/bin/python"
export PIP_ROOT_USER_ACTION=ignore
export UV_PROJECT_ENVIRONMENT="$VENV"

# CUDA build environment for the source-compiled extensions (bae, gsplat).
# Prefer a complete SYSTEM toolkit: the nvidia/cuda:*-devel image ships nvcc + all headers +
# libs under one root at /usr/local/cuda — exactly the unified layout torch's cpp_extension
# expects. Fall back to the pip cuda-toolkit wheels on bare machines with no system CUDA,
# wiring the scattered site-packages/nvidia/* dirs via CPATH/LIBRARY_PATH.
if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
else
    NV="$VENV/lib/python3.11/site-packages/nvidia"
    export CUDA_HOME="$NV/cuda_nvcc"
    export PATH="$CUDA_HOME/bin:$PATH"
    export CPATH="$NV/cuda_runtime/include:$NV/cuda_cccl/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$NV/cuda_runtime/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$NV/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;8.9;8.6;8.0;7.5;7.0}"
# Cap parallel compile jobs. torch's cpp_extension defaults to one job per CPU; each cc1plus
# for a torch C++/CUDA file needs ~2-4 GB, so the default OOM-kills the compiler in a memory-
# constrained Docker VM. 2 keeps peak RAM sane; raise via MAX_JOBS if the build host has more.
export MAX_JOBS="${MAX_JOBS:-2}"

echo "=== uv sync: full env (all extras incl. gpu toolkit + feedforward) ==="
cd "$SCRIPT_DIR"
/root/.local/bin/uv sync --all-extras

# collab-data: private repo, kept out of the locked graph; installed explicitly post-sync.
# Best-effort: a credential-less Docker build (e.g. on a Mac) can't auth to the private repo —
# don't fail the build. Re-run setup.sh at deploy (where git creds exist) to install it.
echo "=== install collab-data (private) ==="
/root/.local/bin/uv pip install --python "$PYTHON" "git+https://github.com/BasisResearch/collab-data.git" \
    || echo "WARN: collab-data not installed (no git auth in this environment) — re-run setup.sh at deploy."

# Pre-fetch vismatch default-model weights so remote/tmux runs never download mid-run.
# Best-effort like collab-data: a build stage without network/system libs skips it and
# the weights download lazily on first use instead.
"$PYTHON" - <<'EOF' || echo "WARN: vismatch weights pre-fetch failed — weights will download on first use."
import vismatch
for name in ("disk-lightglue",):  # extend when configs reference more models
    vismatch.get_matcher(name, device="cpu")
    print(f"vismatch weights cached: {name}")
EOF

# Smoke test — mandatory: torch + the extensions this script compiled (bae, gsplat).
# The full creator chain pulls cv2/open3d, which need GUI/X11 system libs absent in a Docker
# BUILD stage but present at runtime — so import it best-effort here (verified for real in the
# runtime image). Keeps the build from failing on runtime-only system libs.
"$PYTHON" - << 'PYEOF'
import torch, bae, gsplat
assert "12.1" in torch.version.cuda, f"FAIL: cuda={torch.version.cuda}"
assert torch.__version__.startswith("2.5"), f"FAIL: torch={torch.__version__}"
print(f"[OK] torch={torch.__version__} cuda={torch.version.cuda}; bae + gsplat compiled")
try:
    from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
    print("[OK] full creator chain imports")
except ImportError as e:
    print(f"[WARN] creator import deferred to runtime (system lib absent in build stage): {e}")
PYEOF
echo "=== setup complete ==="
