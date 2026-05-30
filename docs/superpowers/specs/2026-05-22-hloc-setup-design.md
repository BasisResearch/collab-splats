# hloc Setup Design

**Date:** 2026-05-22
**Status:** approved

## Goal

Install [hloc (Hierarchical-Localization)](https://github.com/cvg/Hierarchical-Localization) into the nerfstudio conda env for SfM map building. hloc is an alternative path to VGGT-X — used to produce COLMAP-format camera poses + sparse point cloud for nerfstudio gaussian splatting. No code integration into collab_splats; standalone script usage only.

## Environment Context

- Env: `/opt/conda/envs/nerfstudio` (Python 3.11, torch 2.4.0+cu121, CUDA 12.1)
- All hloc hard deps already present: torch, torchvision, numpy, opencv-python, h5py, pycolmap 4.0.4, kornia 0.8.3, lightglue 0.0, scipy, tqdm, matplotlib, gdown
- Only `plotly` may be missing (hloc dep, non-critical for core pipeline)

## Design

### New file: `setup_hloc.sh`

- Clone `https://github.com/cvg/Hierarchical-Localization` with `--recursive` into `third_party/hloc/`
- Guard: if `third_party/hloc/` already exists, skip clone (re-runnable)
- Install editable via `/opt/conda/envs/nerfstudio/bin/pip install -e third_party/hloc/`
- Verify: print `hloc.__version__` to confirm install

### `.gitignore` change

Add `third_party/hloc/` so the cloned repo is not tracked.

## Out of Scope

- No `BasePointcloudCreator` subclass for hloc
- No changes to `pyproject.toml`
- No changes to `setup_feedforward.sh` or `setup.sh`
- Submodule presence enables SuperPoint/R2D2/D2-Net extractors but their weights download on first use — no pre-download step

## Typical Usage (post-install)

```bash
# Extract features
/opt/conda/envs/nerfstudio/bin/python -m hloc.extract_features \
    --conf superpoint_aachen --image_dir images/ --export_dir outputs/

# Build image pairs via retrieval
/opt/conda/envs/nerfstudio/bin/python -m hloc.pairs_from_retrieval \
    --descriptors outputs/global-feats-netvlad.h5 --output outputs/pairs.txt

# Match features
/opt/conda/envs/nerfstudio/bin/python -m hloc.match_features \
    --conf superglue --pairs outputs/pairs.txt \
    --features outputs/feats-superpoint.h5 --matches outputs/matches.h5

# SfM reconstruction
/opt/conda/envs/nerfstudio/bin/python -m hloc.reconstruction \
    --sfm_dir outputs/sfm/ --image_dir images/ \
    --pairs outputs/pairs.txt --features outputs/feats-superpoint.h5 \
    --matches outputs/matches.h5
```
