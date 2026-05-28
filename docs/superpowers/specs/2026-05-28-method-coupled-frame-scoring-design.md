# Method-Coupled Frame Scoring — Design Spec

**Date:** 2026-05-28  
**Status:** Draft  
**Scope:** `collab_splats/utils/frame_sampling.py`, `collab_splats/dashboard/panes/preprocess.py`, `collab_splats/wrapper/reconstructor.py`, `collab_splats/wrapper/splatter.py`

---

## Problem

`_run_extraction` in `PreprocessPane` decodes the video **twice**:

1. `score_all_frames()` — full sequential decode of every frame + Lucas-Kanade OF scoring → metrics chart
2. `sample_frames_fps()` or `sample_frames_optical_flow()` — decode again → actual frames

This means:
- OF scoring runs even when the user chose FPS mode (unnecessary).
- Every video is decoded at least twice regardless of method.
- `score_all_frames` computes OF between consecutive full-video frames; in FPS mode the chart is never used.

---

## Design

### 1. Change `sample_frames_optical_flow` return type

**Before:** `-> list[np.ndarray]`  
**After:** `-> tuple[list[np.ndarray], list[dict]]`

The OF loop already computes `should_select, score, components` per frame — scores are currently discarded (`_, _`). Capture and return them:

```python
# Each score dict:
{
    "frame_idx": int,       # index into the selected frames list (0, 1, 2, ...)
    "disparity": float,
    "rotation": float,
    "histogram_similarity": float,
    "score": float,
    "selected": bool,       # always True (only selected frames are returned)
}
```

### 2. Remove `score_all_frames` from `_run_extraction`

`score_all_frames` is no longer called during dashboard extraction. It remains as a standalone utility function (used in offline analysis, tests, notebooks).

### 3. `_run_extraction` — method-coupled flow

**FPS mode:**
```
sample_frames_fps(fps, max_frames) → (frames, indices)
_render_fps_raster(indices, total_frames) → PNG bytes for chart panel
```

**OF mode:**
```
sample_frames_optical_flow(...) → (frames, scores)
_render_metrics_figure(scores, selected_indices, total_frames) → PNG bytes (unchanged)
```

Progress: single 0→100% pass. Remove the 0–50 / 50–100 split.

### 4. `_render_fps_raster` — new pure helper in `preprocess.py`

Single-panel bar chart:
- x-axis: video frame index (0 → `total_frames`)
- Vertical bars at each sampled index (bar height = 1, width = 1px)
- Same dark background style as existing metrics chart
- Returns PNG bytes (same interface as `_render_metrics_figure`)

```python
def _render_fps_raster(
    sampled_indices: list[int],
    total_frames: int,
) -> bytes:
    """Render a frame-position raster for FPS-sampled frames."""
```

### 5. Update callers of `sample_frames_optical_flow`

All callers that only need frames must unpack `(frames, _)`:

| File | Line | Change |
|------|------|--------|
| `collab_splats/dashboard/panes/preprocess.py` | ~346 | unpack `(frames, scores)` |
| `collab_splats/wrapper/reconstructor.py` | ~68 | unpack `(frame_arrays, _)` |
| `collab_splats/wrapper/splatter.py` | ~311 | unpack `(sampled_frames, _)` |
| `collab_splats/utils/__init__.py` | re-export | no change |

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/utils/frame_sampling.py` | Change `sample_frames_optical_flow` return type; update docstring |
| `collab_splats/dashboard/panes/preprocess.py` | Remove `score_all_frames` call; add `_render_fps_raster`; update FPS/OF paths; flatten progress |
| `collab_splats/wrapper/reconstructor.py` | Unpack `(frame_arrays, _)` |
| `collab_splats/wrapper/splatter.py` | Unpack `(sampled_frames, _)` |
| `tests/utils/test_frame_sampling.py` | Update `sample_frames_optical_flow` tests to unpack tuple |
| `tests/dashboard/test_preprocess.py` | Add `_render_fps_raster` tests |

---

## Non-Goals

- Changing `score_all_frames` behavior — stays as-is, no longer called from dashboard by default.
- OF metrics for FPS mode — OF between non-adjacent frames is misleading; raster only.
- Changing `sample_frames_fps` interface — unchanged.
- Caching scores to disk — out of scope.
