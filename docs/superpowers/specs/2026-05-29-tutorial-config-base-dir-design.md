# Tutorial Config: BASE_DIR + Notebook Wiring

**Date:** 2026-05-29  
**Status:** Approved

## Problem

`tutorial_config.py` hardcodes paths via `get_cache_dir()`, which resolves to `docs/source/.cache/<dataset>`. No way to point notebooks at a different data root without editing the cache utility. `VIDEO_PATH` and `OUTPUT_DIR` are set inline per-notebook rather than centrally.

## Goal

Single file to edit (`tutorial_config.py`) — set `BASE_DIR` and `DATASET`, all notebooks work out of the box.

## Design

### `tutorial_config.py`

Replace `get_cache_dir` import with explicit `BASE_DIR`-derived paths:

```python
from pathlib import Path

# ── Edit these two variables to point at your data ───────────────────────────
BASE_DIR   = Path("/workspace/outputs")
DATASET    = "birds_c0043"
MAX_FRAMES = 30

# ── Derived paths (do not edit) ───────────────────────────────────────────────
VIDEO_PATH = BASE_DIR / DATASET / "video.mp4"
OUTPUT_DIR = BASE_DIR / DATASET
CACHE_DIR  = OUTPUT_DIR          # alias — notebooks use CACHE_DIR
IMAGES     = OUTPUT_DIR / "images"
```

- `get_cache_dir` import removed; `collab_splats.utils.paths` no longer needed here.
- `CACHE_DIR` kept as alias so existing notebook cells that reference it continue to work unchanged.
- `DATASET` kept so notebooks can display it in headers/logs.

### Notebooks touched

| Notebook | Change |
|---|---|
| `tutorial_config.py` | Replace `get_cache_dir` with `BASE_DIR` block |
| `01_preprocessing/keyframe_extraction.ipynb` | Remove inline `VIDEO_PATH` assignment; it now comes from `%run ../tutorial_config.py` |
| `02_pointcloud/slam_loop_closure.ipynb` | Remove inline `OUTPUT_DIR` assignment; it now comes from config |

All other notebooks — zero changes. They already consume `CACHE_DIR` and `IMAGES`.

### No YAML, no example file

Config lives in Python. One file, two variables. No file-copy step for new users.

## Out of scope

- Per-notebook override cells (can be added later if needed)
- Dashboard integration (separate concern)
- Renaming `CACHE_DIR → OUTPUT_DIR` across all notebooks (can be done as follow-up)
