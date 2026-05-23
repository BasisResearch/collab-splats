# Semantics Dashboard — Fixes, UX Redesign & Optical Flow Sampling

**Date:** 2026-04-17
**Branch strategy:** Two PRs — PR 1 on `refactor/dashboard`, PR 2 on a new branch off `refactor/dashboard`.

---

## Context

The Panel-based `SemanticsDashboard` (`collab_splats/dashboard/semantics.py`) was recently implemented. Six issues surfaced during use:

1. `transformers` (needed by `Talk2DinoExtractor`) is missing from `pyproject.toml`.
2. Video frames display upside-down — cv2 ignores rotation metadata embedded in video files.
3. Frame previews are unconstrained in size (hardcoded `width=700`, no max height).
4. No clear visual indicator of whether the Training or Explore tab is active.
5. The Explore tab is a single long vertical column — four sections require heavy scrolling.
6. Frame extraction uses only FPS-based sampling; optical flow sampling (more efficient, better scene coverage) is not available.

Issues 1–5 land in PR 1. Issue 6 is a new capability and lands in PR 2 on its own branch.

---

## PR 1: Dashboard Bug Fixes + UX Redesign (`refactor/dashboard`)

### 1. Missing `transformers` dependency

**File:** `pyproject.toml`

Add `transformers` to `[project.dependencies]`. It is imported unconditionally by `Talk2DinoExtractor` (a registered extractor) and is not optional.

Also add a defensive `ImportError` wrapper to each extractor's `__init__` for any dep that could be missing (e.g. `maskclip_onnx`, `transformers`), surfacing a clear `pip install` hint rather than a bare `ModuleNotFoundError`.

### 2. Upside-down video frames

**File:** `collab_splats/dashboard/semantics.py` — `_load_frames_from_video()`

After `cap = cv2.VideoCapture(video_path)`, read the rotation metadata:

```python
rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
```

Apply the appropriate `cv2.rotate()` call to each frame before appending:

| Metadata value | Rotation applied |
|---------------|-----------------|
| 90            | `cv2.ROTATE_90_COUNTERCLOCKWISE` |
| 180           | `cv2.ROTATE_180` |
| 270           | `cv2.ROTATE_90_CLOCKWISE` |
| 0 / other     | no-op |

### 3. Image size constraints

**File:** `collab_splats/dashboard/semantics.py`

Replace all `pn.pane.PNG(None, width=700)` with:

```python
pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
```

This constrains both dimensions while preserving aspect ratio.

### 4. Active tab indicator

**File:** `collab_splats/dashboard/semantics.py`

Two changes:

**a) CSS override** — pass to `pn.extension(raw_css=[...])` in `build_app()` to strengthen the active tab underline:

```css
.bk-tab.bk-active {
    border-bottom: 4px solid #2596be !important;
    font-weight: 700 !important;
    color: #2596be !important;
}
.bk-tab {
    color: #aaa;
    font-weight: 400;
}
```

**b) Mode badge** — add a `pn.pane.HTML` badge at the top of each tab's content column:

```python
_BADGE_STYLE = (
    "display:inline-flex;align-items:center;gap:6px;"
    "background:#e8f4fd;border:1px solid #2596be;"
    "border-radius:20px;padding:4px 14px;margin-bottom:12px"
)
_DOT = "<span style='width:8px;height:8px;border-radius:50%;background:#2596be;display:inline-block'></span>"
_LABEL_STYLE = "font-size:12px;font-weight:700;color:#2596be;text-transform:uppercase;letter-spacing:0.5px"

training_badge = pn.pane.HTML(
    f"<div style='{_BADGE_STYLE}'>{_DOT}<span style='{_LABEL_STYLE}'>Training Mode</span></div>"
)
explore_badge = pn.pane.HTML(
    f"<div style='{_BADGE_STYLE}'>{_DOT}<span style='{_LABEL_STYLE}'>Explore Mode</span></div>"
)
```

### 5. Explore tab restructure — sub-tabs with two-column layout

**File:** `collab_splats/dashboard/semantics.py`

Replace the single `explore_tab = pn.Column(...)` with an inner `pn.Tabs(dynamic=True)` containing four sub-tabs. Each sub-tab uses a `pn.Row(controls, image_pane)` layout:

```
pn.Tabs(
    ("① Frames",       pn.Row(frames_controls,    current_frame_pane)),
    ("② Features",     pn.Row(features_controls,  feature_overlay_pane)),
    ("③ Segmentation", pn.Row(seg_controls,        seg_output_pane)),
    ("④ Query",        pn.Row(query_controls,      query_gallery)),
    dynamic=True,
)
```

Controls column is `width=300`. Image/output pane uses `max_width=640, max_height=480, sizing_mode="scale_both"`.

**Frames controls column:**
```
fps_slider
sampling_mode_dd  ← new widget: Select("FPS" / "Optical Flow"), disabled=True until PR 2
extract_frames_btn
frame_count_txt
frame_slider
```

**Features controls column:**
```
extractor_dd
device_dd
extract_features_btn
```

**Segmentation controls column:**
```
seg_strategy_dd
seg_device_dd
seg_btn
seg_count_txt
```

**Query controls column:**
```
hf_model_dd
query_device_dd
text_pairs_input
method_dd
temp_slider
query_btn
```

---

## PR 2: Optical Flow Frame Sampling (new branch off `refactor/dashboard`)

### New module: `collab_splats/semantics/frame_sampling.py`

Extract the FPS logic from `semantics.py` (including the cv2 rotation correction introduced in PR 1 fix #2) and add an optical flow sampler. PR 2 removes `_load_frames_from_video` from `semantics.py` entirely — the dashboard's `_extract_frames` dispatches via this module instead.

```python
def sample_frames_fps(video_path: str, fps: float) -> list[np.ndarray]:
    """Extract frames at a fixed FPS rate. Refactored from _load_frames_from_video."""
    ...

def sample_frames_optical_flow(
    video_path: str,
    flow_threshold: float = 500.0,
    max_frames: int = 200,
) -> list[np.ndarray]:
    """
    Select frames where scene content has changed significantly.

    Accumulates dense optical flow magnitude (Farneback) between consecutive
    frames. Emits a frame when cumulative flow exceeds flow_threshold, then
    resets the accumulator. Caps output at max_frames.
    """
    ...
```

Implementation: `cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)`, then `np.sum(np.sqrt(flow[...,0]**2 + flow[...,1]**2))` as the per-frame magnitude scalar.

Both functions apply the same cv2 rotation correction from fix #2.

### Dashboard integration

`_load_frames_from_video` in `semantics.py` is replaced by a dispatch:

```python
from collab_splats.semantics.frame_sampling import sample_frames_fps, sample_frames_optical_flow

def _extract_frames(self, event):
    if self.sampling_mode_dd.value == "Optical Flow":
        frames = sample_frames_optical_flow(
            str(video_path),
            flow_threshold=self.flow_threshold_slider.value,
            max_frames=200,
        )
    else:
        frames = sample_frames_fps(str(video_path), self.fps_slider.value)
```

The Frames sub-tab gains a `flow_threshold_slider` (FloatSlider, 100–2000, default 500) that is shown/hidden based on `sampling_mode_dd.value`.

### Wrapper integration

**File:** `collab_splats/wrapper/splatter.py` (and `SplatterConfig` TypedDict)

Add `frame_selection: Literal["fps", "optical_flow"] = "fps"` to `SplatterConfig`. `Splatter.preprocess()` dispatches to the appropriate sampler when extracting training frames.

---

## File map

| File | Change |
|------|--------|
| `pyproject.toml` | Add `transformers` to deps |
| `collab_splats/dashboard/semantics.py` | Fixes 2–5; sampling_mode widget (disabled); PR 2 enables it |
| `collab_splats/semantics/frame_sampling.py` | **New** — `sample_frames_fps`, `sample_frames_optical_flow` |
| `collab_splats/wrapper/splatter.py` | Add `frame_selection` config field + dispatch |

---

## Out of scope

- DINO/CLIP — `maskclip_onnx` already installs CLIP transitively; `torch.hub` handles DINOv2 at runtime. No additional install needed.
- Optical flow GPU acceleration — Farneback on CPU is sufficient for fieldwork video lengths.
- Changing the `MaterialTemplate` theme color or sidebar width.
