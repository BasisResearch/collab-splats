# Feedforward Install Design: vggt-x + mapanything

**Date:** 2026-04-22
**Branch:** refactor/core-modules
**Status:** Implemented

## Context

`VGGTXCreator` and `MapAnythingCreator` are implemented in `collab_splats/pointcloud/feedforward.py`. The feedforward exploration notebook (`docs/pointcloud/feedforward_exploration.ipynb`) imports both. Neither package is installable without version conflicts against the nerfstudio environment.

The goal: install vggt-x and mapanything so the feedforward notebook runs, without breaking nerfstudio (ns-train, imports, CUDA kernels). Both coexist in one Python process since nerfstudio training follows point cloud creation.

## Environment (verified)

| Package | Installed | Constraint | Status |
|---|---|---|---|
| torch | 2.1.2+cu118 | Must not change — CUDA 11.8 kernels | ✅ unchanged |
| torchvision | 0.16.2+cu118 | Must not change | ✅ unchanged |
| numpy | 1.26.4 | Must stay 1.26.x | ✅ unchanged |
| timm | 0.6.7 | nerfstudio pins `==0.6.7` | ✅ unchanged — upgrade not needed |
| tyro | 0.9.35 | nerfstudio ≥0.9.8 | ✅ |
| nerfstudio | 1.1.5 | Editable at /workspace/nerfstudio | NOT modified |

## Key Findings from Audit

**timm upgrade not required.** mapanything's `timm` dep is only in the `radio` optional extra (RADIO backbone). Core `MapAnythingCreator` code and `uniception==0.1.7` both work with timm 0.6.7.

**nerfstudio not modified.** nerfstudio pins `timm==0.6.7` but doesn't import timm in its code. The pip check warning "nerfstudio requires timm==0.6.7" remains cosmetic and safe.

**Python path split.** `python` in this environment points to base conda (Python 3.13, `/opt/conda/`). All feedforward packages install into the nerfstudio env (Python 3.10, `/opt/conda/envs/nerfstudio/`). Use `/opt/conda/envs/nerfstudio/bin/python` explicitly, or activate the nerfstudio conda env before running.

**rerun-sdk excluded.** `rerun-sdk==0.24.1` (mapanything dep) requires `numpy>=2.0`, which conflicts with the `numpy<2.0` constraint. It is only used in mapanything's visualization/dataset code (`viz.py`, `datasets/wai/`), not in the reconstruction pipeline. Excluded from install.

## Package Conflicts (resolved)

### vggt-x
- Pins `torch==2.3.1`, `torchvision==0.18.1` → conflict → install `--no-deps`
- No timm requirement
- vggt namespace package (no `__init__.py`) — importable as namespace package in Python 3.10 ✓

### mapanything
- `opencv-python-headless==4.10.0.84` conflicts with installed `opencv-python` → `--no-deps`
- No torch/numpy pin in core deps
- `rerun-sdk~=0.24.1` excluded (numpy>=2.0 conflict)
- `timm` only in `radio` optional extra — not needed for core usage

## Install Contract (setup_feedforward.sh)

```bash
# vggt-x: --no-deps avoids torch==2.3.1 upgrade
pip install 'git+https://github.com/Linketic/VGGT-X.git' --no-deps
pip install viser==0.2.23 evo pyliblzfse safetensors roma kornia -c constraints_feedforward.txt

# mapanything: --no-deps avoids opencv-python-headless conflict
pip install 'git+https://github.com/facebookresearch/map-anything.git' --no-deps
pip install huggingface_hub hydra-core natsort orjson pillow-heif plyfile \
    python-box requests tensorboard tqdm -c constraints_feedforward.txt
pip install 'uniception==0.1.7' -c constraints_feedforward.txt
pip install 'lightglue @ git+https://github.com/cvg/LightGlue.git' --no-deps
pip install open3d -c constraints_feedforward.txt

# collab-splats feedforward extras
pip install -e '.[feedforward]' --no-deps
```

## §Discovered Deps (from execution audit)

| Package | Source | Notes |
|---|---|---|
| viser==0.2.23 | vggt-x requirements.txt | visualization server |
| evo | vggt-x requirements.txt | trajectory evaluation |
| pyliblzfse | vggt-x requirements.txt | compression lib |
| safetensors | vggt-x requirements.txt | model weight loading |
| roma | vggt-x requirements.txt (already in collab-splats core) | rotation math |
| kornia | vggt-x requirements.txt (already in collab-splats core) | image processing |
| uniception==0.1.7 | mapanything core dep | core model backbone |
| huggingface_hub | mapanything core (already in collab-splats core) | model hub |
| hydra-core | mapanything core (already in collab-splats core) | config |
| natsort | mapanything core | natural sort |
| orjson | mapanything core | fast JSON |
| pillow-heif | mapanything core | HEIF image support |
| plyfile | mapanything core | PLY 3D format |
| python-box | mapanything core | dict → dotaccess |
| tensorboard | mapanything core | logging |
| lightglue | mapanything colmap extra | feature matching |
| open3d | mapanything colmap extra | 3D processing |
| **rerun-sdk** | mapanything core dep | **EXCLUDED** — requires numpy>=2.0 |

## Files Modified

| File | Change |
|---|---|
| `pyproject.toml` | Add `[feedforward]` optional extras |
| `setup_feedforward.sh` | New — full install contract |
| `constraints_feedforward.txt` | New — pip guards (torch, torchvision, numpy) |

**nerfstudio NOT modified.**

## Verification Results

- `import nerfstudio` → OK
- `from collab_splats.pointcloud import VGGTXCreator, MapAnythingCreator` → OK
- `import vggt; import mapanything` → OK
- numpy=1.26.4, torch=2.1.2+cu118, cuda=11.8 — all unchanged ✓
- pytest (non-GPU, non-nerfstudio, non-dashboard): 137 pass, 7 fail — all 7 are pre-existing API mismatches (`pycolmap.Rig` missing, `softmax_temp` signature change), unrelated to feedforward install

## Remaining Work

- Run `docs/pointcloud/feedforward_exploration.ipynb` with nerfstudio kernel to verify end-to-end
- `pip check` will show: `nerfstudio 1.1.5 requires timm==0.6.7, but you have timm 0.6.7` — no actual conflict; this is cosmetic
