# Notebook Reorganization & Zarr Cache Design

**Date:** 2026-05-23
**Status:** Approved

## Problem

Tutorial notebooks live in two places: canonical sources in `docs/{pointcloud,semantics,splats}/`
and symlinks in `docs/source/tutorials/` pointing back to them. This makes the docs source tree
confusing to navigate, organizes notebooks by subsystem rather than pipeline stage (breaking the
GIGO mental model), and leaves the cross-notebook cache chain inconsistent and partially broken.

## Goals

1. Single canonical location for all notebooks: `docs/source/tutorials/`
2. Pipeline-stage folder taxonomy matching the GIGO pipeline order
3. Consistent zarr-based cache with explicit lineage — no re-derivation across notebooks
4. Cache structure follows existing `evals/results/{dataset}/{variant}/` convention

## Non-Goals

- Rewriting notebook content beyond cache I/O wiring
- Fixing Splatter to accept pre-extracted frames (next task)
- Adding new notebook content beyond stub shells for new notebooks

---

## Part 1 — Notebook Structure

### Remove symlinks

All symlinks in `docs/source/tutorials/` are deleted. Canonical notebooks move from
`docs/{pointcloud,semantics,splats}/` into `docs/source/tutorials/`. The old directories are
deleted after migration.

One notebook (`keyframe_extraction.ipynb`) already lives directly in `docs/source/tutorials/preprocessing/` — no action needed.

### New folder taxonomy

```
docs/source/tutorials/
  01_preprocessing/
    keyframe_extraction.ipynb          ← exists, move

  02_pointcloud/
    feedforward_methods.ipynb          ← rename from feedforward_exploration
                                          (VGGT-X vs MapAnything side-by-side comparison)
    bundle_adjustment.ipynb            ← exists, move
    slam_loop_closure.ipynb            ← rename from loop_closure_eval
    colmap_sfm.ipynb                   ← NEW stub
    feedforward_mesh.ipynb             ← exists, move (end-to-end FF→mesh demo)

  03_splats/
    derive_splats.ipynb                ← exists, move
    visualization.ipynb                ← exists, move

  04_semantics/
    feature_extraction.ipynb           ← exists, move
    segmentation.ipynb                 ← exists, move
    maskclip_vs_talk2dino.ipynb        ← rename from maskclip_reference_comparison

  05_lifting/
    semantic_lifting.ipynb             ← exists, move

  06_mesh/
    create_mesh.ipynb                  ← exists, move (splat → TSDF mesh, primary path)

  07_localization/
    localization.ipynb                 ← exists, move

  evals/
    ground_truth_evals.ipynb           ← exists, move
```

### Updated toctree (`docs/source/tutorials/index.rst`)

Sections renamed to match numbered folders. `localization.ipynb` added (was missing from toctree).
`feedforward_mesh` and `ground-truth-evals` moved to correct sections.

---

## Part 2 — Cache Structure

### Format

- **zarr** (`zarr>=2.16` + `numcodecs`): all large array artifacts (point clouds, features, lifted features, per-vertex mesh features)
- **JSON**: `frame_scores.json` — small, human-readable, stays JSON
- **PLY**: mesh geometry — Open3D/PyVista native format, stays PLY
- **PT**: `decoder.pt` — small MLP weights, stays torch

### Root

```
docs/.cache/{dataset}/
```

`dataset` is a flat string encoding scene identity, e.g. `birds_c0043`, `chess_seq01`.
Mirrors `evals/results/{dataset}/` convention. No deeper nesting of dataset path.

### Full structure

```
docs/.cache/{dataset}/
  images/                              ← sequential JPGs: 000001.jpg, 000002.jpg, …
                                          (COLMAP convention, written by keyframe_extraction)
  frame_scores.json                    ← optical-flow scores (written by keyframe_extraction)

  semantics/
    features.zarr                      ← (N, H, W, D) float32, chunks=(1, H, W, D)
                                          pure 2D per-frame features, method-agnostic
                                          written by feature_extraction

  vggtx/
    colmap/sparse/0/                   ← cameras.bin, images.bin, points3D.bin
    reconstruction.zarr                ← arrays: pts3d (P,3), extrinsics (N,4,4),
                                          intrinsics (N,3,3), pixel_indices (P,3),
                                          colors (P,3), world_points (N,H,W,3)[chunked by N],
                                          images (N,3,H,W)[chunked by N]
                                          attrs: method, video_path, timestamp
    lifted.zarr                        ← (P, D) features lifted into raw vggtx pts
    mesh/
      mesh.ply                         ← FF→direct mesh (feedforward_mesh demo only)
    ba/
      reconstruction.zarr              ← BA-corrected poses/points
      lifted.zarr                      ← (P, D) lifted into BA-corrected pts (primary path)
    lc/
      reconstruction.zarr              ← LC-corrected poses/points
      lifted.zarr                      ← (P, D) lifted into LC-corrected pts

  mapanything/
    colmap/sparse/0/
    reconstruction.zarr
    lifted.zarr
    mesh/
      mesh.ply
    ba/
      reconstruction.zarr
      lifted.zarr
    lc/
      reconstruction.zarr
      lifted.zarr

  colmap/
    sparse/0/                          ← pure COLMAP SfM output (colmap_sfm.ipynb writes here)

  splats/                              ← GS outputs separated from reconstruction tree
    {recon_variant}/{gs_method}/       ← recon_variant: vggtx_ba, vggtx_lc, vggtx,
                                          mapanything_ba, mapanything, colmap
      …nerfstudio output…
      mesh/
        mesh.ply
        features.zarr                  ← (V, D) per-vertex semantic features
        decoder.pt                     ← MLP decoder weights
```

### Notebook config convention

Every notebook opens with two config cells:

```python
# Cell 1 — identity
DATASET = "birds_c0043"            # change per scene
METHOD  = "vggtx"                  # or "mapanything", "colmap"
VARIANT = "ba"                     # or "lc", "" for raw baseline
GS_METHOD = "rade-features"        # nerfstudio method

# Cell 2 — derived paths (do not edit)
from pathlib import Path
CACHE    = Path("../../.cache") / DATASET
IMAGES   = CACHE / "images"
RECON    = CACHE / METHOD / (VARIANT if VARIANT else "")
SPLAT    = CACHE / "splats" / f"{METHOD}_{VARIANT}" / GS_METHOD if VARIANT else \
           CACHE / "splats" / METHOD / GS_METHOD
```

---

## Part 3 — Cache Chain (Notebook I/O)

```
01_preprocessing/keyframe_extraction
  IN:  VIDEO_PATH
  OUT: images/        ← FIX: currently missing — add extract_video_frames call
       frame_scores.json

02_pointcloud/feedforward_methods
  IN:  images/
  OUT: vggtx/reconstruction.zarr  (if exists: load, else run+save)
       vggtx/colmap/sparse/0/
       mapanything/reconstruction.zarr  (if exists: load, else run+save)
       mapanything/colmap/sparse/0/

02_pointcloud/bundle_adjustment
  IN:  RECON/reconstruction.zarr  (default METHOD=vggtx, VARIANT="")
  OUT: vggtx/ba/reconstruction.zarr
  NOTE: verify exact I/O during implementation

02_pointcloud/slam_loop_closure
  IN:  RECON/reconstruction.zarr
  OUT: vggtx/lc/reconstruction.zarr

02_pointcloud/colmap_sfm            ← NEW stub
  IN:  images/
  OUT: colmap/sparse/0/

02_pointcloud/feedforward_mesh      ← end-to-end demo, no downstream
  IN:  mapanything/reconstruction.zarr
  OUT: mapanything/mesh/mesh.ply

03_splats/derive_splats
  IN:  images/ + RECON/reconstruction.zarr  (after Splatter fix — see known gaps)
  OUT: splats/{METHOD}_{VARIANT}/{GS_METHOD}/
  CURRENT: reads video directly (known gap)

03_splats/visualization
  IN:  splats/{METHOD}_{VARIANT}/{GS_METHOD}/
  OUT: visualization only

04_semantics/feature_extraction
  IN:  images/
  OUT: semantics/features.zarr  (if exists: load, else run+save)

04_semantics/segmentation
  IN:  images/
  OUT: visualization only

04_semantics/maskclip_vs_talk2dino
  IN:  images/ + semantics/features.zarr
  OUT: comparison visualization only

05_lifting/semantic_lifting
  IN:  RECON/reconstruction.zarr + semantics/features.zarr
  OUT: RECON/lifted.zarr
  FIX: currently re-runs VGGT-X from scratch; must load reconstruction from cache

06_mesh/create_mesh
  IN:  splats/{METHOD}_{VARIANT}/{GS_METHOD}/
  OUT: splats/{METHOD}_{VARIANT}/{GS_METHOD}/mesh/

07_localization/localization
  IN:  vggtx/reconstruction.zarr + images/
  OUT: visualization only (no cache write)

evals/ground_truth_evals
  IN:  vggtx/ba/reconstruction.zarr
       vggtx/lc/reconstruction.zarr
  OUT: metrics (no cache write)
```

---

## Part 4 — Code Changes Required

### 4.1 Add zarr to dependencies

`setup.py` (or `pyproject.toml`):
```
zarr>=2.16
numcodecs
```

### 4.2 `FeedforwardResult.save_zarr()` / `load_zarr()`

File: `collab_splats/pointcloud/feedforward/base.py`

Add two class methods alongside existing `save()`/`load()`:

- `save_zarr(path: Path)` — writes all arrays including `world_points` and `images` (chunked
  by frame), stores metadata in `.zattrs`
- `load_zarr(path: Path) -> FeedforwardResult` — reads from zarr store; supports partial reads
  (e.g. load only `pts3d` + `extrinsics` without pulling full `images` tensor)

Existing `.npz` `save()`/`load()` remain for backward compatibility during transition.

### 4.3 `keyframe_extraction.ipynb` — save images/

Add `extract_video_frames(video_path, selected_indices, IMAGES)` call after frame scoring.
Currently saves `frame_scores.json` but does not write frames to disk.

### 4.4 `semantic_lifting.ipynb` — load from cache

Replace re-run of `VGGTXCreator` with `FeedforwardResult.load_zarr(RECON / "reconstruction.zarr")`.
Replace `sample_frames_fps` frame extraction with load from `IMAGES/`.

---

## Part 5 — Known Gaps

| Gap | Resolution |
|-----|------------|
| Splatter reads video directly, not `images/` cache | Next task: extend `SplatterConfig` to accept pre-extracted image dir |
| `bundle_adjustment.ipynb` I/O not fully verified | Confirm source/output arrays during implementation |
| `world_points` / `images` excluded from current `.npz` save | Resolved by zarr backend (Part 4.2) |
| `colmap_sfm.ipynb` is a stub with no content | Stub only in this task; content filled separately |

---

## Primary Tutorial Path (GIGO)

```
VIDEO → images/ + frame_scores.json           [keyframe_extraction]
      → vggtx/reconstruction.zarr             [feedforward_methods]
      → vggtx/ba/reconstruction.zarr          [bundle_adjustment]
      → splats/vggtx_ba/rade-features/        [derive_splats]
      → splats/vggtx_ba/rade-features/mesh/   [create_mesh]

images/ → semantics/features.zarr             [feature_extraction]
        → vggtx/ba/lifted.zarr                [semantic_lifting]
```
