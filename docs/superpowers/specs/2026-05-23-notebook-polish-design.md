# Notebook Polish — Design Spec

**Date:** 2026-05-23  
**Branch:** `refactor/cu121`  
**Prerequisite:** All in-flight agent work (vggtx-feedforward-refactor, cu121 migration) must land before execution begins.

---

## Goal

Make all 15 tutorial notebooks consistent in style, properly wired into the zarr cache chain, and fully executed so rendered outputs are visible on next open — without any manual steps from the user.

---

## Approach

Approach C: fix + execute each notebook in order before moving to the next. Errors surface immediately and are resolved in context before downstream notebooks consume stale/missing cache.

---

## Execution Order

Notebooks are processed in dependency order. Chain notebooks write zarr artifacts consumed by later notebooks. Standalone notebooks are self-contained.

| # | Notebook | Tier | Cache writes | Cache reads |
|---|----------|------|-------------|-------------|
| 1 | `01_preprocessing/keyframe_extraction` | chain | `images/`, `frame_scores.json` | — |
| 2 | `02_pointcloud/feedforward_methods` | chain | `vggtx/reconstruction.zarr`, `mapanything/reconstruction.zarr` | `images/` |
| 3 | `02_pointcloud/bundle_adjustment` | standalone | — | `images/` (7-scenes canned) |
| 4 | `02_pointcloud/slam_loop_closure` | standalone | — | — |
| 5 | `02_pointcloud/feedforward_mesh` | standalone | — | — |
| 6 | `02_pointcloud/colmap_sfm` | stub | — | — |
| 7 | `03_splats/derive_splats` | standalone | — | — |
| 8 | `03_splats/visualization` | standalone | — | — |
| 9 | `04_semantics/feature_extraction` | ported | `semantics/features.zarr` | `images/` |
| 10 | `04_semantics/segmentation` | ported | — | `images/` |
| 11 | `04_semantics/maskclip_vs_talk2dino` | standalone | — | — |
| 12 | `05_lifting/semantic_lifting` | chain | `vggtx/ba/lifted.zarr` | `vggtx/reconstruction.zarr` |
| 13 | `06_mesh/create_mesh` | standalone | — | — |
| 14 | `07_localization/localization` | chain | — | `vggtx/reconstruction.zarr` |
| 15 | `evals/ground_truth_evals` | standalone | — | — |

Cache root: `docs/.cache/{DATASET}/` where `DATASET = "birds_c0043"` for the main chain.

---

## Canonical Cell Structure

The following three-cell opening is the canonical style for every notebook. All notebooks are updated to match this exactly.

**Cell 1 — autoreload:**
```python
%load_ext autoreload
%autoreload 2
```

**Cell 2 — all imports (one cell, at the top, no exceptions):**
```python
import os
import time
import numpy as np
from pathlib import Path

import torch
import pyvista as pv
import matplotlib
matplotlib.use("Agg") if os.environ.get("PYVISTA_OFF_SCREEN") else None
%matplotlib inline

# PyVista: interactive (trame) when running live; static PNG when headless (nbconvert)
pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

from collab_splats.pointcloud.feedforward import VGGTXCreator
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
```

Notebooks are executed headlessly with `PYVISTA_OFF_SCREEN=true jupyter nbconvert --execute --inplace ...`. Live Jupyter sessions get trame (interactive 3D); headless execution gets static inline PNGs.

Every stdlib, third-party, and `collab_splats` import lives here. No imports anywhere else in the notebook — not in config cells, not mid-notebook, not inside functions. Exception: optional heavy deps that would crash the module on missing install may be imported inside the function that needs them, with a clear `ImportError` message (CLAUDE.md rule).

**Cell 3 — config block (pure variable assignments, zero imports):**
```python
# ── Configuration ─────────────────────────────────────────────────────────────
DATASET  = "birds_c0043"
METHOD   = "vggtx"       # "vggtx" | "mapanything"
VARIANT  = "ba"          # "ba" | "lc" | "" (raw baseline)

CACHE  = Path("../../.cache") / DATASET
IMAGES = CACHE / "images"
RECON  = CACHE / METHOD / VARIANT if VARIANT else CACHE / METHOD
```

This is the style to emulate. Config is readable, imports are predictable, and the notebook can be scanned top-to-bottom without hunting for where dependencies are introduced.

Standalone notebooks that don't consume cache still declare `CACHE`/`IMAGES` for consistency and to enable future chain wiring.

---

## Per-Notebook Fix Protocol

Every notebook receives the same treatment before execution.

### 4. Section dividers

Major sections separated by `########` dividers in code cells. Section headers in markdown use `## §N — Title` format throughout (e.g., `## §1 — Load Reconstruction`).

### 5. Markdown explanatory cells

Every code section is preceded by a markdown cell explaining:
- What the code executes
- What it produces
- Why (its role in the pipeline)

1–3 sentences. Not just a header — actual prose. Example:

> *Runs VGGT-X on the extracted keyframes. Produces a `FeedforwardResult` with world-space 3D points, per-point colors, and camera poses. Result is saved to zarr cache so downstream notebooks (semantic lifting, localization) skip this step.*

### 6. Cache-or-run guards

For every expensive computation that writes to cache:

```python
if _cache_path.exists():
    result = load_from_cache(_cache_path)
    print(f"Loaded from cache: {_cache_path}")
else:
    result = run_computation(...)
    save_to_cache(result, _cache_path)
    print(f"Saved to cache: {_cache_path}")
```

### 7. Inline block comments

Each logical code block gets one short inline comment explaining what it does (not how). Follows CLAUDE.md standard.

### 8. Kernelspec

All notebooks: `python3` display name, Python 3.11 kernelspec (`/opt/conda/envs/nerfstudio/`).

### 9. Logging

`print()` only acceptable in notebooks for user-visible status messages. No `print()` inside imported library code.

---

## Ported Notebooks

`feature_extraction` and `segmentation` currently use the old `Splatter` / `ConfigLoader` / `sample_frames_fps` pattern. Both are ported to:

- Drop `Splatter`, `ConfigLoader` imports
- Load first frame directly from `IMAGES` (first `.jpg` in cache dir)
- Add standard `DATASET`/`CACHE`/`IMAGES` config block
- `feature_extraction`: add `semantics/features.zarr` cache-or-run guard around `extract_and_cache`

---

## Execution Protocol

For each notebook in order:

1. Apply all fixes (style, cache, imports, markdown cells)
2. Execute: `PYVISTA_OFF_SCREEN=true jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 <notebook.ipynb>`
3. Verify: check cell output count > 0, no error tracebacks in outputs
4. Fix any errors, re-execute
5. Proceed to next notebook

Timeout: 1800s (30 min) per notebook — covers heavy GPU notebooks.

Stub notebooks (`colmap_sfm`): fix style only, no execution (no compute defined yet).

---

## Additional Functions List (Post-Pass Deliverable)

During the fix pass, candidates for shared helper functions are collected and delivered as a list (not implemented). Expected candidates include:

- `load_frame_from_cache(images_dir) → np.ndarray` — loads first image from `IMAGES/` as numpy HxWx3; replaces repeated `cv2.imread` / `Image.open` patterns across notebooks
- `notebook_config_block(dataset, method, variant) → dict` — typed config helper that validates paths exist before returning, with clear error if cache is missing
- `display_stats_table(results: dict)` — shared tabular summary (pts raw/filtered, conf mean/std, timing) currently duplicated across feedforward_methods and bundle_adjustment
- `assert_cache_exists(path, notebook_name)` — raises informative error pointing to prerequisite notebook when a cache dependency is missing, rather than a cryptic FileNotFoundError
- `save_notebook_figure(fig, name, cache_dir)` — saves matplotlib/pyvista figures to `.cache/{dataset}/figures/` for cross-notebook reuse in comparison cells
- `print_section_header(title)` — consistent section banner for notebook stdout (replaces ad-hoc `print("=" * 40)` patterns)

---

## Success Criteria

- All 15 notebooks have > 0 rendered output cells after execution pass (except `colmap_sfm` stub)
- All imports are at top of first code cell
- All notebooks use `## §N — Title` section header format
- Every code section has a preceding markdown explanation cell
- `feature_extraction` and `segmentation` no longer reference `Splatter` or `ConfigLoader`
- Cache chain notebooks load from zarr on re-open without re-running GPU compute
- User opens any notebook cold and sees rendered results
