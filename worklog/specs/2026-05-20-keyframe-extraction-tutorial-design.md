# Keyframe Extraction Tutorial — Design Spec

**Date:** 2026-05-20  
**Status:** Approved  
**Section:** preprocessing (new, first in tutorial series)

---

## Context

First tutorial in an ordered series: `preprocessing → pointcloud → splats → semantics`.  
Teaches users why frame selection matters before they ever touch COLMAP or Gaussian splatting.

---

## Files Changed

| Action | Path |
|--------|------|
| New | `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb` |
| Modify | `collab_splats/utils/frame_sampling.py` |

---

## `frame_sampling.py` Changes

### Core section — new functions

```python
def get_video_info(video_path: str) -> dict:
    """Return basic video metadata.

    Keys: total_frames (int), fps (float), duration_s (float)
    Keeps cv2 out of the notebook.
    """

def score_all_frames(
    video_path: str,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Score every frame in a video using OpticalFlowFrameSelector.

    Returns one dict per frame with keys:
        frame_idx, disparity, rotation, hist_similarity, score, selected
    Does not cap frame count — processes the full video.
    """
```

`score_all_frames` drives the OF timeseries visualization without requiring the notebook to own a cv2 loop. `get_video_info` keeps all cv2 calls out of the notebook.

### Visualization section — appended behind divider

```
##############################################################################
# Visualization
##############################################################################
```

Four functions (all return `matplotlib.figure.Figure`):

| Function | Signature | Purpose |
|----------|-----------|---------|
| `plot_frame_grid` | `(frames: list[np.ndarray], title: str, n_cols: int = 6) -> Figure` | Display a grid of RGB frames |
| `plot_selection` | `(total_frames: int, fps_indices: list[int] \| None = None, of_indices: list[int] \| None = None) -> Figure` | One set → single timeline; both sets → two stacked timelines for comparison |
| `plot_frame_scores` | `(frame_scores: list[dict]) -> Figure` | 3-panel plot: disparity / rotation / histogram similarity over frame index, with selection markers |
| `plot_disparity_sensitivity` | `(frame_scores: list[dict], disparity_values: list[float]) -> Figure` | Line plot: N selected frames vs min_disparity threshold |

`plot_selection` replaces both `plot_selection_timeline` and `plot_selection_comparison` — single/dual panel driven by which args are passed. `plot_disparity_sensitivity` reapplies thresholds to precomputed `frame_scores` — avoids re-decoding the video.

---

## Notebook Structure

**File:** `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb`

Notebook imports only — no inline plot functions, no lambda helpers, no cv2 calls.

```python
# Standard imports block (top of notebook)
from collab_splats.utils.frame_sampling import (
    get_video_info,
    sample_frames_fps,
    sample_frames_optical_flow,
    score_all_frames,
    plot_frame_grid,
    plot_selection,
    plot_frame_scores,
    plot_disparity_sensitivity,
)
```

### Phase 0 — Why Frame Selection Matters (markdown)

Explains:
- Frames fed to COLMAP determine pointcloud density and spatial distribution
- Even/FPS sampling: uniform in time but biased toward regions filmed slowly
- Optical flow sampling: uniform in scene content — only selects frames where the scene actually changes
- Key insight: a slow pan produces dozens of near-identical frames → oversampled region, undersampled elsewhere

### Phase 1 — Load Video

```python
video_path = "path/to/your/video.mp4"  # user substitutes

info = get_video_info(video_path)
print(f"Frames: {info['total_frames']}  FPS: {info['fps']:.1f}  Duration: {info['duration_s']:.1f}s")

preview_frames = sample_frames_fps(video_path, fps=1.0)
plot_frame_grid(preview_frames, title="Video preview (1 fps)")
```

Preview grid samples every `total_frames // 12` frame using `sample_frames_fps`.

### Phase 2 — Even Sampling (FPS)

```python
fps_frames = sample_frames_fps(video_path, fps=target_fps)
fps_indices = list(range(0, info['total_frames'], info['total_frames'] // len(fps_frames)))
plot_selection(info['total_frames'], fps_indices=fps_indices)
plot_frame_grid(fps_frames, title=f"FPS-sampled frames (n={len(fps_frames)})")
```

### Phase 3 — Optical Flow Score Timeseries

```python
frame_scores = score_all_frames(video_path)
plot_frame_scores(frame_scores)
```

3-panel plot shows disparity, rotation, and histogram similarity over time. Spikes indicate high scene-change frames. Horizontal threshold line at `min_disparity` default.  
Markdown cell annotates: "Spikes = camera moved or scene changed. Flat regions = redundant frames."

### Phase 4 — OF Selection Overlay

```python
of_frames = sample_frames_optical_flow(video_path)
of_indices = [d["frame_idx"] for d in frame_scores if d["selected"]]
plot_selection(info['total_frames'], of_indices=of_indices)
```

### Phase 5 — Side-by-Side Comparison

```python
plot_selection(info['total_frames'], fps_indices=fps_indices, of_indices=of_indices)
plot_frame_grid(of_frames, title=f"OF-sampled frames (n={len(of_frames)})")
```

Markdown: "FPS distributes uniformly in time. OF distributes uniformly in content."

### Phase 6 — Parameter Sensitivity

```python
disparity_values = [10, 25, 50, 75, 100, 150, 200]
plot_disparity_sensitivity(frame_scores, disparity_values)
```

Uses precomputed `frame_scores` — no re-decoding. Shows frame count vs threshold.  
Markdown: guidance on tuning — higher disparity = fewer, more distinct frames; lower = denser coverage at cost of redundancy.

### Phase 7 — Guidance (markdown)

| Scenario | Recommended method | Notes |
|----------|--------------------|-------|
| Slow camera pans | Optical flow | Avoids oversampling stationary/slow regions |
| Handheld walk-through | Either | FPS simpler; OF marginally better |
| Fast action / drone | FPS | OF may miss fast transients below disparity window |
| Limited frame budget | Optical flow | Better spatial coverage per frame |

---

## Ordering in Docs

```
docs/source/tutorials/
  preprocessing/         ← new section, listed first in index.rst
    keyframe_extraction.ipynb
  pointcloud/
  splats/
  semantics/
```

`index.rst` updated to include `tutorials/preprocessing/keyframe_extraction` as first tutorial entry.

---

## Out of Scope (V1)

- Downstream COLMAP/pointcloud quality comparison
- Interactive widgets (ipywidgets sliders)
- Side-by-side rendered video output
- Automated sample video download
