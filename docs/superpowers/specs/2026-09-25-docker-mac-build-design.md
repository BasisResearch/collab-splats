# Docker build on Apple Silicon — design

**Date:** 2026-09-25 · **Status:** implemented, first Mac build pending

## Problem

`docker build --platform=linux/amd64` on an Apple Silicon Mac never finished: gsplat was still
compiling after a night.

- every nvcc/cicc call runs under amd64 emulation
- bae took ~70 min (log: `Built bae` at 4194.7 s); gsplat is the large one and never finished
- uv built all 4 CUDA packages at once, each with `MAX_JOBS` jobs: swap pressure on Docker's VM
- 6 SASS arches (7.0 … 9.0) = 6 full compiles per kernel

## Constraints

- no GCP permissions (Cloud Build, VMs); no repo admin (Actions secrets for private collab-data)
- the RunPod dev pod is an unprivileged container: no daemon, no `CAP_SYS_ADMIN`, `unshare` denied
- so the Mac is the only builder; the design cuts what the emulator has to compile

## Changes

1. **Arch list `8.0+PTX`** (`Dockerfile` ARG, `setup.sh` default)
   - sm_80 SASS runs natively on every 8.x GPU; PTX JITs forward to sm_90
   - ~1.5 compiles of work instead of 6
2. **fused-ssim arches pinned** (`CUDA_ARCHITECTURES="80;90"` in `setup.sh`)
   - its setup.py ignores `TORCH_CUDA_ARCH_LIST`; GPU-less fallback was `75;80;89`, no sm_90
3. **`UV_CONCURRENT_BUILDS=1`** (`setup.sh`): one CUDA package at a time, peak RAM = `MAX_JOBS` × 7.3 GB
4. **Two-pass builder** (`Dockerfile` + `SETUP_DEPS_ONLY=1` in `setup.sh`)
   - pass 1 copies only `pyproject.toml uv.lock README.md LICENSE setup.sh` and runs
     `uv sync --locked --all-extras --no-install-project`
   - source edits no longer invalidate the CUDA layer
5. **`CUDA_CACHE_MAXSIZE=4294967296`** (runtime ENV): Hopper's PTX JIT result fits the driver cache
6. **Mac recipe in the Dockerfile header**: Rosetta on, Docker memory up, `MAX_JOBS` = memory GB / 8

## GPU support

| Family | Examples | Status |
|---|---|---|
| Ampere / Ada (sm_80/86/89) | A100, A40, A6000, A10, L4, L40, RTX 3090/4090 | native |
| Hopper (sm_90) | H100, H200 | PTX JIT on first launch (fused-ssim native) |
| Volta / Turing (sm_70/75) | V100, T4, RTX 2080 | not supported — add `7.0;7.5` |
| Blackwell (sm_100/120) | B200, RTX 5090 | not supported — torch 2.5.1+cu121 has no kernels |

Measured: `torch.cuda.get_arch_list()` = `sm_50 … sm_90`, no PTX.

## Verification

- pass 1 dry-run from the 5 files alone: installs bae/gsplat/fused-ssim/nvdiffrast, not the project
- owed: full Mac build completes; runtime smoke test (AOT `.so` present) passes
- owed: `docker run --gpus all … python -c "import bae, gsplat, fused_ssim"` on an A40; H100 if available

## Out of scope

CI builders, prebuilt wheelhouse, torch/CUDA upgrade for Blackwell.
