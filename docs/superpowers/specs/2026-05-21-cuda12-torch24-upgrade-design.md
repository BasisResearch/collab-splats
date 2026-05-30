# CUDA 12.1 + torch 2.4 Upgrade — Design Spec

**Date:** 2026-05-21  
**Branch:** new files only — existing `Dockerfile` and `env.yml` untouched

---

## Context

Current Docker image pins `nvidia/cuda:11.8.0` and `torch==2.1.2+cu118`. These pins exist because `gsplat-rade` CUDA kernels were compiled against them. The downstream cost:

- **BAE** (Bundle Adjustment in Eager Mode): 3 custom patches in `patches/bae-torch21-compat.patch` bridging its torch 2.3+ target down to 2.1.x
- **MapAnything 1.1**: runtime monkey-patch `_patch_mapanything_torch_compat()` in `feedforward.py` because `tensor.any(dim=(1,3))` syntax requires torch 2.4+
- **GPU coverage**: CUDA 11.8 excludes native support for H100/H200 workloads on RunPod

Upgrading to **CUDA 12.1 + torch 2.4** removes all compat debt and enables native BAE + MapAnything. Existing `Dockerfile` and `env.yml` stay untouched so the current environment keeps working during development.

---

## Architecture

Same multi-stage structure as the current Dockerfile. No structural redesign.

```
Stage 1: builder (nvidia/cuda:12.1.0-devel-ubuntu22.04)
  ├── conda env "nerfstudio" created
  ├── torch 2.4.0+cu121 installed
  ├── gsplat-rade compiled (TORCH_CUDA_ARCH_LIST SM70–90)
  ├── hloc v1.4 installed
  └── nerfstudio fork installed (torch 2.4 compat branch)

Stage 2: conda-source (continuumio/miniconda3:latest)
  └── provides /opt/conda base

Stage 3: colmap-source (colmap/colmap:20240213.23)
  └── provides colmap binary compiled against CUDA 12.x

Stage 4: runtime (nvidia/cuda:12.1.0-runtime-ubuntu22.04)
  ├── system packages
  ├── /opt/conda copied from conda-source
  ├── /opt/conda/envs/nerfstudio copied from builder
  ├── /opt/hloc, /opt/nerfstudio copied from builder
  ├── colmap binary copied from colmap-source
  └── SSH + bashrc + RunPod CMD
```

---

## New Files

### `Dockerfile.cu121`

Diff from current `Dockerfile`:

| Line / section | Current | New |
|---|---|---|
| `ARG NVIDIA_CUDA_VERSION` | `11.8.0` | `12.1.0` |
| `ARG CUDA_ARCHITECTURES` | `"90;89;86;80;75;70;61"` | `"90;89;86;80;75;70"` (drop SM61 Pascal) |
| torch install | `torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url .../whl/cu118` | `torch==2.4.0+cu121 torchvision==0.19.0+cu121 --extra-index-url .../whl/cu121` |
| cuda-toolkit conda channel | `nvidia/label/cuda-11.8.0` | `nvidia/label/cuda-12.1.0` |
| tiny-cuda-nn install block | present (lines 97–101) | **removed** (not needed) |
| gsplat-rade install | `pip install git+https://github.com/brian-xu/gsplat-rade.git` | same URL — verify torch 2.4 compat before building (see Risks) |
| nerfstudio install | `github.com/nerfstudio-project/nerfstudio` (upstream) | fork URL — torch 2.4 compat branch |
| colmap-source stage | `FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5` | `FROM colmap/colmap:20240213.23` (CUDA 12.3, runs on 12.1) |
| rclone + api-key data download | present | **keep as-is** (no change) |

Everything else (SSH config, bashrc, runtime system packages, env vars) is unchanged.

### `env.cu121.yml`

Mirrors `env.yml` (currently just: `python=3.10`, `wheel`, `ninja`, `git`). No substantive change needed — torch is installed via pip in the Dockerfile, not via conda. Keep the file minimal, rename for clarity.

---

## Code Changes (separate from new Docker files)

These are independent cleanup tasks, gated on the new image working:

### 1. Remove `_patch_mapanything_torch_compat` monkey-patch

**File:** `collab_splats/pointcloud/feedforward.py`

The patch exists because torch 2.1.x doesn't support `tensor.any(dim=(1,3))`. With torch 2.4 this is native. Remove the function definition and its call site. Memory note: `project_mapanything_torch_bridge.md` — "intentionally temporary, sunset on torch>=2.4."

### 2. Remove BAE torch 2.1 compat patches

**File:** `setup_bundle_adjustment.sh`  
**File:** `patches/bae-torch21-compat.patch`

Three shims in the patch file:
- `torch._C.TensorBase` → `_TensorBase` (torch <2.3)
- `torch.utils._triton` guard (torch <2.3)
- `conversion.so` import guard (cuSPARSE ABI mismatch on CUDA 11.8)

All three are irrelevant on torch 2.4 + CUDA 12.1. Remove the `git apply` call in `setup_bundle_adjustment.sh`. Archive or delete `patches/bae-torch21-compat.patch`.

Also: `USE_CUDSS=0` flag in `setup_bundle_adjustment.sh` was required because cuDSS is unavailable on CUDA 11.8. CUDA 12.1 includes cuDSS — remove this flag (or verify BAE actually uses it and benefits).

---

## Risks & Pre-Implementation Checks

1. **gsplat-rade torch 2.4 compat**: `brian-xu/gsplat-rade` is a fork; verify it builds cleanly against torch 2.4.0+cu121 before investing in a full image build. Quick check: try `pip install git+https://github.com/brian-xu/gsplat-rade.git` in a torch 2.4 env and run its tests.

2. **nerfstudio fork torch 2.4 compat**: The upstream nerfstudio likely supports torch 2.4; confirm the fork hasn't diverged on torch version guards. If it has, update the fork first.

3. **pycolmap 0.4.0 + CUDA 12.x**: pycolmap 0.4.0 is old. It links against system COLMAP libs, not CUDA directly. With the colmap binary coming from `colmap/colmap:20240213.23` (CUDA 12.3), the system libs will differ. May need to upgrade pycolmap to 0.6.x or later — test hloc localization end-to-end.

4. **colmap binary ABI**: `colmap/colmap:20240213.23` was compiled for CUDA 12.3 runtime. Running on a CUDA 12.1 host: CUDA minor-version forward-compat should work, but confirm on actual RunPod hardware.

5. **CUDA arch SM61 removal**: GTX 1080Ti (SM61) dropped from arch list. If any RunPod instances use Pascal-era GPUs, they'll fail. Accept this tradeoff — H100/A100 is the target.

---

## Verification

After `docker build -f Dockerfile.cu121 -t collab-env:cu121 .`:

```bash
# CUDA and torch
docker run --gpus all collab-env:cu121 bash -c "
  source /opt/conda/etc/profile.d/conda.sh && conda activate nerfstudio &&
  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
"
# Expected: 2.4.0+cu121 True

# gsplat-rade
docker run --gpus all collab-env:cu121 bash -c "
  source /opt/conda/etc/profile.d/conda.sh && conda activate nerfstudio &&
  python -c 'import gsplat; print(gsplat.__version__)'
"

# nerfstudio
docker run --gpus all collab-env:cu121 bash -c "
  source /opt/conda/etc/profile.d/conda.sh && conda activate nerfstudio &&
  ns-train --help
"

# colmap
docker run collab-env:cu121 colmap --version

# After setup/feedforward.sh (MapAnything without monkey-patch)
docker run --gpus all collab-env:cu121 bash -c "
  source /opt/conda/etc/profile.d/conda.sh && conda activate nerfstudio &&
  bash /workspace/collab-splats/setup/feedforward.sh &&
  python -c 'from collab_splats.pointcloud.feedforward import MapAnythingCreator; print(\"ok\")'
"

# After setup_bundle_adjustment.sh (BAE without patches)
docker run --gpus all collab-env:cu121 bash -c "
  source /opt/conda/etc/profile.d/conda.sh && conda activate nerfstudio &&
  bash /workspace/collab-splats/setup_bundle_adjustment.sh &&
  python -c 'import bae; print(\"ok\")'
"
```

Full pipeline test: run the feedforward tutorial notebook (`docs/pointcloud/feedforward_exploration.ipynb`) end-to-end in the new container.
