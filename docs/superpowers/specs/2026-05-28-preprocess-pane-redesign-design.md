# PreprocessPane Redesign — Design Spec

**Date:** 2026-05-28
**Status:** Draft — pending user review
**Scope:** Tab 1 (`PreprocessPane`) layout + interaction redesign; `AppState` schema update; no changes to other tabs

---

## Overview

Refine Phase 1 dashboard (PreprocessPane) to:

1. Show video on left half immediately when a config/video is loaded
2. Gate extraction controls visibility on video being loaded
3. Move frame extraction controls into a collapsible `pn.Card` (right half)
4. After extraction: show interactive Bokeh metric plots + horizontal frame strip in right half
5. Tap on metric plot → seek video + scroll frame strip to that frame
6. Write extracted frames to zarr (lazy) instead of holding `list[np.ndarray]` in RAM
7. Fix config-load bug where `state.video_path` was not set from sidebar

---

## AppState Schema Change

```python
class AppState(param.Parameterized):
    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames_zarr_path = param.Parameter(default=None)   # NEW: Path to frames.zarr
    selected_indices = param.List(default=[])           # NEW: list[int] of selected frame indices
    # frames = param.List(...)                          # REMOVED — was list[np.ndarray]
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
```

**Zarr layout:**
- Path: `<output_dir>/frames.zarr`
- Array name: `"frames"`, dtype uint8, shape `(N, H, W, 3)`
- Chunks: `(1, H, W, 3)` — one chunk per frame for O(1) random access
- Compressor: Blosc lz4

**Downstream access pattern (any pane needing a frame):**
```python
import zarr
store = zarr.open(str(state.frames_zarr_path), mode="r")
frame = store["frames"][i]  # shape (H, W, 3), uint8
```

**Tab 3 note:** `ReconstructPane` gates on `state.output_dir`, not frames — unaffected by this change. `state.frames` no longer exists; any reference to it must be updated.

---

## Layout

```
┌────────────────────┬─────────────────────────────────┐
│                    │  ▼ Frame Extraction  [Card]       │
│  Video Player      │    Method / n_frames / window     │
│  (~55% width)      │    Extract button                 │
│                    │                                  │
│  [hidden until     │─────────────────────────────────  │
│   video loaded]    │  Metrics (3 Bokeh plots)          │
│                    │  [click → seek video + jump]      │
│                    │─────────────────────────────────  │
│                    │  ←── Frame Strip (horiz scroll) ──│
│                    │  [th][th][th][th][th]...          │
└────────────────────┴─────────────────────────────────┘
```

Right panel is a `pn.Column(sizing_mode="stretch_both")` — stacks Card → metrics → frame strip vertically. No full-width bottom section.

---

## State Machine

| `video_path` | extraction done | Controls Card | Metrics | Frame Strip |
|---|---|---|---|---|
| None | — | hidden | hidden | hidden |
| set | no | visible, expanded | hidden | hidden |
| set | yes | visible, collapsed | visible | visible |

Controls Card can be re-expanded and extraction re-run at any time. Each re-run updates metrics and frame strip in place.

---

## Components

### Video Player (left)

- `pn.pane.Video`, `sizing_mode="stretch_width"`, `height=400`
- `visible=False` until `state.video_path` set
- `time` property used for seeking (set from tap callback)
- Info strip below: filename · total frames · fps · duration

### Controls Card (right, top)

`pn.Card(collapsed=False, title="Frame Extraction", ...)` — collapses automatically after successful extraction. Re-expandable.

Contents (unchanged from current):
- Method dropdown: `fps` | `optical_flow`
- FPS slider (fps only) / Min disparity slider (optical_flow only)
- Target frames slider
- Window start/end sliders
- **Extract Frames** button
- Status HTML (frame count / error message)

Visibility gated on `state.video_path is not None` via `param.watch`.

### Zarr Write (extraction thread)

After `sample_frames_*` returns frames:

```python
import zarr
from numcodecs import Blosc

path = state.output_dir / "frames.zarr"
store = zarr.open(str(path), mode="w")
store.create_dataset(
    "frames",
    data=np.stack(frames),          # (N, H, W, 3) uint8
    chunks=(1, *frames[0].shape),
    compressor=Blosc(cname="lz4", clevel=3),
    overwrite=True,
)
state.frames_zarr_path = path
state.selected_indices = list(range(len(frames)))
```

### Metrics Panel (right, middle)

Replace `_render_metrics_figure()` (matplotlib PNG) with `_build_metrics_panel()` returning `pn.Column` of Bokeh figures.

**Three figures, shared x-axis (frame index):**
- Optical Flow / Disparity — `#7ec8e3`
- Rotation (°) — `#f0a500`
- Hist. Similarity — `#d090e0`

Each figure:
- `height=120`, `sizing_mode="stretch_width"`
- Green `Span` overlays at each selected frame index
- `TapTool` enabled — on tap, compute nearest selected frame index:

```python
def _on_plot_tap(self, attr, old, new):
    indices = new["index"]
    if not indices:
        return
    x = new["x"][0]
    nearest = min(self._selected_indices, key=lambda i: abs(i - x))
    self._seek_to_frame(nearest)
```

`_seek_to_frame(idx)`:
1. Compute timestamp: `t = idx / total_fps`
2. Set `self._video_pane.time = t`
3. Update `self._active_frame_idx = idx`
4. Trigger frame strip highlight + scroll (see below)

### Frame Strip (right, bottom)

Rendered as `pn.pane.HTML` — full HTML string with embedded base64 thumbnails.

Each thumbnail:
```html
<img id="frame-{i}" src="data:image/png;base64,..." 
     style="width:120px;height:90px;cursor:pointer;
            border: 2px solid {active_color};"
     onclick="panelSeekFrame({i})" />
```

Active frame gets `border-color: #50c050`; others `#333`.

**Auto-scroll on tap:** After updating `_active_frame_idx`, append a JS `<script>` into a dedicated `pn.pane.HTML` script pane:

```python
self._scroll_script.object = (
    f"<script>document.getElementById('frame-{idx}')"
    f".scrollIntoView({{behavior:'smooth',inline:'center'}});</script>"
)
```

The script pane re-renders on each assignment, executing the scroll.

**Lazy thumbnail loading:** Thumbnails generated from zarr on demand:

```python
store = zarr.open(str(self._state.frames_zarr_path), mode="r")
frames_arr = store["frames"]
thumbnails = _frames_to_thumbnails([frames_arr[i] for i in selected_indices[:100]])
```

Cap at 100 thumbnails for render performance (same as current).

---

## Config Load Bug Fix

Location: `app.py` or sidebar session loader.

When user loads a `run_config.yaml`:
- Parse `video_path` field from config YAML
- Set `state.video_path = Path(config["video_path"])` if field present and file exists
- `PreprocessPane._on_video_path_change` fires automatically, calls `_load_video()`

No change needed in `preprocess.py` — the watch is already wired.

---

## Files Changed

| File | Change |
|---|---|
| `collab_splats/dashboard/state.py` | Add `frames_zarr_path`, `selected_indices`; remove `frames` |
| `collab_splats/dashboard/panes/preprocess.py` | Layout, Bokeh metrics, zarr write, frame strip HTML, scroll JS |
| `collab_splats/dashboard/app.py` | Sidebar config loader sets `state.video_path` |
| `tests/dashboard/test_preprocess.py` | Update tests: no `state.frames`, add zarr path assertions |

---

## Out of Scope

- Reverse video sync (video play position → highlight frame in strip) — deferred
- Frame strip > 100 thumbnails — deferred (virtual scroll)
- Metric computation outside of `score_all_frames()` — no change
