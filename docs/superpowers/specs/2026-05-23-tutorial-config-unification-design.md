# Tutorial Config Unification — Design Spec

**Date:** 2026-05-23  
**Branch:** refactor/cu121  
**Status:** Approved

## Goal

All tutorial notebooks share a single config source that controls dataset, frame cap, and derived paths. Change one file to update all tutorials. Limit tutorial runs to 30 images so feedforward inference is fast enough to run as a light example.

## New File: `docs/source/tutorials/tutorial_config.py`

```python
from pathlib import Path
from collab_splats.utils.paths import get_cache_dir

DATASET    = "birds_c0043"
MAX_FRAMES = 30          # cap for light tutorial runs

CACHE_DIR  = get_cache_dir(DATASET)
IMAGES     = CACHE_DIR / "images"
```

Single edit point for dataset and frame cap. All 15 notebooks sit one level below the tutorials root, so `%run ../tutorial_config.py` works uniformly.

## Notebook Changes

### Pattern applied to every notebook

Replace the local config block:
```python
# before
DATASET   = "birds_c0043"
CACHE_DIR = get_cache_dir(DATASET)
IMAGES    = CACHE_DIR / "images"
```

With:
```python
# after
%run ../tutorial_config.py   # injects DATASET, MAX_FRAMES, CACHE_DIR, IMAGES
```

Notebook-specific variables (`METHOD`, `VARIANT`, `BACKEND`, `OUTPUT_DIR`, `device`, etc.) stay in the local config block below the `%run` line.

### Per-notebook specifics

| Notebook | Additional change |
|---|---|
| `01_preprocessing/keyframe_extraction.ipynb` | Pass `max_frames=MAX_FRAMES` to `sample_frames_optical_flow` call |
| `02_pointcloud/slam_loop_closure.ipynb` | Drop `SCENE_DIR = /workspace/bicycle/images_4` → use `IMAGES` from shared config |
| all others | Remove local `DATASET`/`CACHE_DIR`/`IMAGES` lines only |

### Bundle adjustment exception

`bundle_adjustment.ipynb` has a self-contained inline demo using canned 7-Scenes data (`CANNED_SOURCE`, `N_FRAMES = 5`). That section stays unchanged — it demonstrates BA mechanics on known-good data, independent of the shared dataset. The shared config is still applied for its `DATASET`/`CACHE_DIR`/`IMAGES` config block at the top.

## What Does NOT Change

- Library code (`setup_inference`, `BaseFeedforwardCreator`, `frame_sampling.py`) — no API changes
- Notebooks `03_splats` through `07_localization` — they load from zarr/reconstruction outputs, not raw images; they still get `%run ../tutorial_config.py` to remove the duplicated DATASET/CACHE_DIR/IMAGES lines
- `evals/ground_truth_evals.ipynb` — same `%run ../tutorial_config.py` treatment

## How MAX_FRAMES Flows

```
tutorial_config.py
  └─ MAX_FRAMES = 30
       └─ keyframe_extraction.ipynb
            sample_frames_optical_flow(..., max_frames=MAX_FRAMES)
            → writes ≤30 frames to IMAGES dir
                 └─ feedforward_methods.ipynb  ← reads IMAGES, gets ≤30
                 └─ slam_loop_closure.ipynb    ← reads IMAGES, gets ≤30
                 └─ colmap_sfm.ipynb           ← reads IMAGES, gets ≤30
                 └─ ... all downstream         ← same
```

## Out of Scope

- Adding `max_frames` to `setup_inference` / `BaseFeedforwardCreator` (library unchanged)
- Changing the dataset from birds_c0043
- Modifying eval scripts (`evals/eval_gt.py`) — not a tutorial
