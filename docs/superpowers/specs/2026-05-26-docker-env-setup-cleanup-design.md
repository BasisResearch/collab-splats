# Docker Env Setup Cleanup — Design

**Date:** 2026-05-26  
**Status:** Draft  
**Branch:** refactor/cu121

## Problem

First-run of the cu121 container fails at three points:

1. `setup_nerfstudio.sh` — open3d resolution error ("from versions: none") caused by `--no-cache-dir` forcing PyPI re-fetch on nerfstudio install; script also applies 5 patches that are already in the nerfstudio fork.
2. `setup/feedforward.sh` — `constraints_feedforward.txt` pins `torch==2.4.0+cu121` (stale from intermediate upgrade); env has `2.5.1`. Constraint forces torch downgrade → cascades to old numpy source build → `distutils.msvccompiler` missing → crash.
3. `bash setup_bundle_adjustment.sh` — script intentionally deleted in cu121 migration but `vendor/README.md` still references it.

Additional bugs found during investigation:
- `setup/hloc.sh` hardcodes conda env `nerfstudio` (old name); env is now `reconstruction`.
- `pyproject.toml` bae comment says "Pre-installed by Dockerfile.cu121 builder stage" — wrong; bae is installed by `setup.sh` Step 2, not the Dockerfile.

## Dry-Run Verification (all confirmed before spec written)

| Test | Result |
|------|--------|
| `pip install --dry-run 'nerfstudio @ git+https://github.com/BasisResearch/nerfstudio.git@d99c8cd7' 'bae @ git+https://github.com/pypose/bae.git@0.2.4' 'numpy>=2'` (fresh-container simulation, no vggt in env) | `Would install bae-0.2.4 nerfstudio-1.1.5 nuscenes-devkit-1.1.9 protobuf-4.25.9` — no numpy downgrade, no conflicts |
| `pip install --dry-run salad pytorch-metric-learning gtsam-develop` (no constraints) | torch 2.5.1 + numpy 2.4.6 both "already satisfied" — no downgrade |
| bae git tags | `0.2`, `0.2.1`, `0.2.2`, `0.2.3`, `0.2.4` confirmed (no `v` prefix) |
| numpy downgrade concern | Only surfaces in dirty env (vggt already installed). On fresh container, setup.sh runs before setup/feedforward.sh — vggt absent, no conflict. |

## Design

### Change 1 — Delete `setup_nerfstudio.sh`, add nerfstudio git URL to `pyproject.toml`

The 5 patches setup_nerfstudio.sh applied are already committed in `/workspace/nerfstudio/pyproject.toml`:
- `gsplat>=1.4.0` ✓, `timm>=0.6.7` ✓, `viser>=0.2.0` ✓, `nerfacc>=0.5.2` ✓, `opencv-python>=4.8.0` ✓ (headless removed)

Add to `pyproject.toml` dependencies:

```toml
"nerfstudio @ git+https://github.com/BasisResearch/nerfstudio.git@d99c8cd7",
```

Delete `setup_nerfstudio.sh`. Remove `bash setup_nerfstudio.sh` from `setup.sh` entirely — no replacement line needed. Nerfstudio is now installed as part of `pip install -e .` (Change 3 step).

**No numpy conflict:** on a fresh container, setup.sh runs before setup/feedforward.sh, so vggt (the source of `numpy<2` metadata) is not yet installed. Dry run with `numpy>=2` confirms clean resolution.

### Change 2 — Delete `constraints_feedforward.txt`, update `setup/feedforward.sh`

Feedforward packages with `--no-deps` (vggt-x, mapanything, vggt-omega) never touch torch.  
Packages without `--no-deps` (pytorch-metric-learning, salad, gtsam-develop) declare only floor bounds:
- `pytorch-metric-learning`: `torch>=1.6.0` — no ceiling
- `gtsam-develop`: `numpy>=1.11.0` — no torch dep at all

Dry run confirmed: all "already satisfied" with no constraint file. Constraint file was causing the problem (stale `==2.4.0` forced downgrade); removing it is strictly better.

Delete `constraints_feedforward.txt`.  
In `setup/feedforward.sh`: remove `CONSTRAINTS=` variable and all `--constraint $CONSTRAINTS` flags.

### Change 3 — Move bae into pyproject.toml git URL, delete setup.sh bae step

Replace `"bae>=0.2.4"` in pyproject.toml with a pinned git URL:

```toml
# pypose BA CUDA extension. Pinned git URL prevents PyPI 'bae' (2.0.x) package collision.
# Built inline during `pip install -e .` (Step 2 below). Requires:
#   PIP_NO_BUILD_ISOLATION=1 (exported at top of setup.sh — bae setup.py imports torch)
#   CUDA_HOME set (Step 2 below — nvcc not in PATH without conda activate)
#   nvidia-cudss-cu12 installed (declared below; pip installs wheels before source builds)
# USE_CUDSS defaults to "1" in bae 0.2.4 — no env var needed.
"bae @ git+https://github.com/pypose/bae.git@0.2.4",
```

Tag `0.2.4` confirmed to exist (no `v` prefix). pypose and nvidia-cudss-cu12 remain declared in pyproject.toml as runtime deps — pip installs them (pure wheels) before building the bae source package.

Delete the entire `setup.sh` Step 2 (bae install block) and Step 1 (nerfstudio). setup.sh becomes a **single install step**:

```bash
echo "=== Step 1: install collab-splats (builds nerfstudio + bae CUDA extension inline) ==="
CUDA_HOME=/opt/conda/envs/reconstruction \
    $PIP install -e "$SCRIPT_DIR"
```

- `USE_CUDSS=1` dropped: bae 0.2.4 defaults to `USE_CUDSS="1"`.
- `CUDA_HOME` kept: nvcc not in PATH without `conda activate`; torch's CUDAExtension needs it.
- `PIP_NO_BUILD_ISOLATION=1` already exported by setup.sh.
- nvidia-cudss-cu12 pre-install removed: pip installs wheels before source packages in the same batch, so nvidia-cudss-cu12 is in site-packages when bae's `find_cudss_root()` runs at build time.

### Change 5 — Fix `setup/hloc.sh` conda env name

```bash
# Before
PIP="/opt/conda/envs/nerfstudio/bin/pip"
PYTHON="/opt/conda/envs/nerfstudio/bin/python"

# After
PIP="/opt/conda/envs/reconstruction/bin/pip"
PYTHON="/opt/conda/envs/reconstruction/bin/python"
```

### Change 6 — Update `vendor/README.md`

Remove three references to `setup_bundle_adjustment.sh` (lines 30, 40, 53). Script was deleted in commit `e7fa1c1` ("bae pre-installed in cu121"). Replace with note that BA deps are installed by `setup.sh`.

## Files Changed

| File | Action | Reason |
|------|--------|--------|
| `setup_nerfstudio.sh` | Delete | Patches already in nerfstudio fork; nerfstudio moved to pyproject git URL |
| `constraints_feedforward.txt` | Delete | Stale torch pin caused numpy downgrade; all feedforward deps floor-only, no constraint needed |
| `setup.sh` | Modify | Delete nerfstudio + bae steps; single `pip install -e .` with CUDA_HOME |
| `setup/feedforward.sh` | Modify | Remove `--constraint` flag and `CONSTRAINTS` variable |
| `pyproject.toml` | Modify | Add nerfstudio git URL dep; replace bae loose pin with pinned git URL |
| `setup/hloc.sh` | Modify | Fix conda env name: `nerfstudio` → `reconstruction` |
| `vendor/README.md` | Modify | Remove setup_bundle_adjustment.sh references |

## Out of Scope

- numpy<2 constraint: env has numpy 2.4.6; vggt-x/vggt-omega bypass with `--no-deps`; no downgrade needed
- hloc pycolmap compatibility: already tracked as separate blocker in STATE.md
- Dockerfile changes: all fixes are in post-build setup scripts
