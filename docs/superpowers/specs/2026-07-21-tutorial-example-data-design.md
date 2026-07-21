# Tutorial example data + self-contained frames.zarr chain — design

**Date:** 2026-07-21
**Status:** Draft (spec review)
**Scope:** Make the tutorial runnable end-to-end by a new user who has no gcloud access, from a single committed example video, and clean up the tutorial in the process.

## Problem

A new user cannot run the tutorial today:

1. **No local data.** `tutorial_config.py` hardcodes `BASE_DIR = /workspace/outputs` and infers the video from a gcloud-scene layout (`run_config.yaml` / `*.MP4` glob). The dashboard pulls scenes from GCS via rclone; a fresh clone has none of it.
2. **The notebook chain is not self-contained.** `01_preprocessing` writes a FrameStore to `TUTORIAL_CACHE/keyframes.zarr`, but `02`/`04` read keyframes from a jpg `FRAMES` dir (`OUTPUT_DIR/frames`) and `07` reads `OUTPUT_DIR/frames.zarr` — both **pipeline-owned** artifacts written by the dashboard, never by the notebooks. Running 01→07 from just a video breaks at 02 (missing frames).
3. **Clutter.** `DATASET` session/stem scoping, a `_infer_video_path` inference helper, a jpg-dir keyframe path that duplicates `frames.zarr`, per-notebook duplicated pyvista/backend boilerplate, dead imports.

## Goal

`git clone` → run notebooks 01→07 in order → full reconstruction + external-frame localization, with **no gcloud, no pre-baked artifacts, no GPU-optional shortcuts** (the user reruns every stage). Reuse of the existing pre-computed 6 GB outputs is explicitly **not** a goal — the tutorial regenerates everything from the committed video.

## Design

### 1. Committed example assets

Two files, committed to the repo under `data/tutorial/` with generic, dataset-agnostic names:

| file | size | source | role |
|---|---|---|---|
| `data/tutorial/tutorial_example-video.mp4` | ~75 MB | re-encode of `2024_02_06/C0043/C0043.MP4` | builds the reconstruction (nb 01→06) |
| `data/tutorial/tutorial_example-frame.jpg` | 2.2 MB | `2024_02_06/C0043/localized_frames/GX010119_f000000.jpg` | external localization query (nb 07) |

**Re-encode** (fits under GitHub's 100 MB/file cap; audio dropped, unused):

```bash
ffmpeg -i C0043.MP4 -c:v libx264 -b:v 6M -maxrate 6M -bufsize 12M -an \
       data/tutorial/tutorial_example-video.mp4
```

Native 1080p is retained; ~6 Mbps is ample for the ≤30 keyframes the tutorial samples. `data/tutorial/README.md` records the exact command and the query-frame provenance (GoPro `GX010119`, frame 0 — a *different* video from the reconstruction) so both assets are reproducible.

No Git LFS (quota/tooling burden for one asset). No external host / fetch helper (reintroduces the network dependency we are removing).

### 2. `tutorial_config.py` — decluttered, repo-relative

Replaces the current 37-line yaml-inference version:

```python
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[3]   # docs/source/tutorials/ -> repo root
_TUT_DATA   = REPO_ROOT / "data" / "tutorial"
MAX_FRAMES  = 30

# Committed inputs (read-only)
VIDEO_PATH  = _TUT_DATA / "tutorial_example-video.mp4"
QUERY_IMAGE = _TUT_DATA / "tutorial_example-frame.jpg"

# Generated outputs (gitignored; notebooks write here)
OUTPUT_DIR     = REPO_ROOT / "data" / "outputs"
FRAMES_ZARR    = OUTPUT_DIR / "frames.zarr"
CACHE_DIR      = OUTPUT_DIR
TUTORIAL_CACHE = OUTPUT_DIR / "tutorial_cache"
```

Removed: `BASE_DIR`, `DATASET` (only 1 notebook reference), `_infer_video_path` + `yaml` import, and the jpg `FRAMES` dir var. A missing `VIDEO_PATH`/`QUERY_IMAGE` raises a clear error pointing at `data/tutorial/README.md`.

`.gitignore`: add `data/outputs/`.

### 3. Self-contained chain on `frames.zarr`

Every keyframe read/write goes through the canonical `FrameStore` at `OUTPUT_DIR/frames.zarr` (`FrameStore.open`, `.image_by_frame_idx`, `.images`, `.frame_indices`, `.frame_idx_from_path`). This is the store the dashboard/`run_pipeline` already produce, and the pattern `07_localization` already uses.

- **01_preprocessing** writes the canonical `OUTPUT_DIR/frames.zarr` (rename from `TUTORIAL_CACHE/keyframes.zarr`).
- **02** (`feedforward_methods`) and **04** (`feature_extraction`, `maskclip_vs_talk2dino`, `segmentation`) replace `FRAMES.glob("*.jpg")` with `FrameStore.open(FRAMES_ZARR)` reads. (`02/bundle_adjustment` and `02/slam_loop_closure` are out of scope — see below.)
- **05** (`semantic_lifting`) reads images via `FrameStore` rather than the path-locked `FeedforwardResult.image_paths` jpgs (use `.image_by_frame_idx` keyed off `frame_idx_from_path`, mirroring 07).
- Each stage persists its canonical artifact to `OUTPUT_DIR` so the next notebook consumes it (02 → `feedforward.zarr`, 04/05 → semantics). The jpg `FRAMES` dir is no longer produced or read.

### 4. `07_localization` — external-frame query

Replace the in-video `QUERY_IDX`/reference-trim logic with the dashboard's external-query path:

```python
from collab_splats.localization import CameraLocalizer, LomaExtractor, estimate_intrinsics
query_image = cv2.cvtColor(cv2.imread(str(QUERY_IMAGE)), cv2.COLOR_BGR2RGB)
K = estimate_intrinsics(query_image)                 # experimental; refined by PnP focal refinement
localizer = CameraLocalizer.from_feedforward(result, images=..., ids=..., extractor=LomaExtractor())
loc = localizer.localize(query_image, K)             # full reconstruction; no trim (query not in map)
```

Viz unchanged in spirit: correspondences (external query ↔ top ranked ref) + localized pose (red) in the point cloud.

### 5. `notebook_utils.py` (beside `tutorial_config.py`)

New `docs/source/tutorials/notebook_utils.py` holds tutorial-presentation helpers, co-located with `tutorial_config.py` (notebooks already import from that dir). Kept out of `collab_splats/` — these are notebook-only concerns with no non-tutorial consumer, so they add no product-suite test burden.

- `set_notebook_backend()` — the pyvista static/trame backend selection duplicated across 8 notebooks.
- `load_keyframes(...)` — thin `FrameStore` open + image accessor wrapper, replacing the FRAMES-glob boilerplate repeated in 5 notebooks.

### 6. Dead-code removal

- Remove unused `import sys` (`03/visualization.ipynb`).

## Implementation principles

- **Reuse:** lean on existing `FrameStore` and `estimate_intrinsics` (the dashboard's own paths); add no parallel machinery.
- **Retire:** delete `_infer_video_path`, `DATASET`, `BASE_DIR`, the jpg `FRAMES` var/dir dependency, `keyframes.zarr` naming, and the duplicated boilerplate the helpers replace.
- **Minimal:** `notebook_utils.py` gets only the two helpers that remove real duplication — no speculative abstraction.

## Verification

- Smoke-run notebooks 01→07 in order against `tutorial_example-video.mp4` on a clean `data/outputs/`, confirming every stage handoff regenerates (frames.zarr → feedforward.zarr → semantics → mesh → localization) with no missing-artifact error.
- 07 localizes the external `tutorial_example-frame.jpg` and renders a plausible pose.
- Committed video is < 100 MB; `git clone` yields both assets with no LFS.
- Existing dashboard/rclone scene flow is untouched (tutorial and dashboard data paths are independent).

## Out of scope

- `02/bundle_adjustment` and `02/slam_loop_closure` notebooks (frames.zarr migration + dead-import cleanup deferred for both).
- Reusing the pre-computed 6 GB outputs (the tutorial regenerates from the video).
- Any change to the dashboard's gcloud/rclone scene pull.
- Re-chunking or footprint work on the generated zarr stores.
