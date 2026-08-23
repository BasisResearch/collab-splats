#!/bin/bash
# Single source of truth for env setup. Runs in the Docker build AND standalone.
# Floor required from the host/image: gcc/g++ (build-essential) + (at runtime) NVIDIA driver.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/opt/venv/reconstruction
PYTHON="$VENV/bin/python"
export PIP_ROOT_USER_ACTION=ignore
export UV_PROJECT_ENVIRONMENT="$VENV"

# CUDA build environment for the source-compiled extensions (bae, gsplat, fused-ssim).
# Prefer a complete SYSTEM toolkit: the nvidia/cuda:*-devel image ships nvcc + all headers +
# libs under one root at /usr/local/cuda — exactly the unified layout torch's cpp_extension
# expects. There is no pip fallback: every nvidia-cuda-nvcc-cu12 wheel (12.1 … 12.8) ships
# only ptxas, so a bare host must provide the toolkit itself — fail fast with the recipe.
if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
else
    cat >&2 <<'MSG'
setup.sh: no nvcc at /usr/local/cuda/bin/nvcc — bae and gsplat build from source and need it.
Use the nvidia/cuda:12.1.1-devel image, or on a bare host build the toolkit with micromamba
(gsplat d2f5c0f needs CCCL >= 2.2 for <cuda/std/optional>; verified 2026-08-22):
  micromamba create -p /opt/cuda-nvcc-12.1 -c nvidia -c conda-forge \
      cuda-version=12.1 cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1
  ln -sfn libcudart.so.12 /opt/cuda-nvcc-12.1/lib/libcudart.so   # solver leaves it dangling
  ln -sfn lib /opt/cuda-nvcc-12.1/lib64 && ln -sfn /opt/cuda-nvcc-12.1 /usr/local/cuda
MSG
    exit 1
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

# --- InstantSfM backend (optional, CC-BY-NC-4.0 — research use) ------------------
# Their pyproject pins numpy==1.26.4; --no-deps is load-bearing (we run numpy 2.x).
# NOTE: plain `uv sync` prunes these (outside the lock) — rerun this block after any sync.
# Best-effort like collab-data/vismatch: the backend is optional, and scikit-sparse builds against
# libsuitesparse-dev (in the Dockerfile apt lists; `apt-get install -y libsuitesparse-dev` on a bare host).
# easydict is VDA's only hard runtime dep missing from the lock (dpt_temporal.py); xformers is
# optional upstream (falls back to plain attention) and decord is only used by its video reader.
echo "=== install InstantSfM backend (instantsfm --no-deps, pyceres, scikit-sparse, easydict) ==="
{
    /root/.local/bin/uv pip install --python "$PYTHON" --no-deps \
        'git+https://github.com/cre185/InstantSfM@d3e599e1a42b4c5a806a84d9f383e1005d25f61b' \
    && /root/.local/bin/uv pip install --python "$PYTHON" pyceres==2.3 scikit-sparse==0.4.15 easydict==1.13
} || echo "WARN: InstantSfM backend not installed (optional; needs libsuitesparse-dev) — re-run setup.sh to retry."

# --- Video Depth Anything (metric) — clone + checkpoint, not pip-installable -----
# Imported from the clone root via sys.path (collab_splats/pointcloud/sfm.py:generate_vda_depth).
# The pin is re-applied on every run (idempotent), so an existing clone cannot drift. NOTE: a
# dangling third_party/Video-Depth-Anything symlink (worktree layouts pointing at an absent
# main-checkout clone) makes `git clone` fail — fix the link target first.
# The ~1.5 GB checkpoint is best-effort like the vismatch pre-fetch: a no-network build stage
# skips it and the first SfM run fails fast with an actionable FileNotFoundError. Download to a
# .part file so an interrupted transfer never leaves a truncated .pth the guard would then skip.
VDA_DIR="$SCRIPT_DIR/third_party/Video-Depth-Anything"
VDA_COMMIT=4f5ae23172ba60fd7bc11ef671cca678842c7072
if [ ! -d "$VDA_DIR/video_depth_anything" ]; then
    git clone https://github.com/DepthAnything/Video-Depth-Anything "$VDA_DIR"
fi
git -C "$VDA_DIR" cat-file -e "$VDA_COMMIT^{commit}" 2>/dev/null || git -C "$VDA_DIR" fetch --quiet
git -C "$VDA_DIR" checkout --quiet "$VDA_COMMIT"
mkdir -p "$VDA_DIR/checkpoints"
VDA_CKPT="$VDA_DIR/checkpoints/metric_video_depth_anything_vitl.pth"
if [ ! -f "$VDA_CKPT" ]; then
    wget -nv -O "$VDA_CKPT.part" \
        "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth" \
        && mv "$VDA_CKPT.part" "$VDA_CKPT" \
        || { rm -f "$VDA_CKPT.part"; echo "WARN: VDA metric checkpoint download failed — re-run setup.sh before using pointcloud.backend: instantsfm."; }
fi

# Smoke test — mandatory: torch + the extensions this script compiled (bae, gsplat, fused-ssim).
# The full creator chain pulls cv2/open3d, which need GUI/X11 system libs absent in a Docker
# BUILD stage but present at runtime — so import it best-effort here (verified for real in the
# runtime image). Keeps the build from failing on runtime-only system libs.
"$PYTHON" - << 'PYEOF'
import torch, bae, gsplat, fused_ssim
assert "12.1" in torch.version.cuda, f"FAIL: cuda={torch.version.cuda}"
assert torch.__version__.startswith("2.5"), f"FAIL: torch={torch.__version__}"
print(f"[OK] torch={torch.__version__} cuda={torch.version.cuda}; bae + gsplat + fused-ssim compiled")
try:
    from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
    print("[OK] full creator chain imports")
except ImportError as e:
    print(f"[WARN] creator import deferred to runtime (system lib absent in build stage): {e}")
PYEOF
echo "=== setup complete ==="
