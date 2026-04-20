# Dashboard Fixes — Design Spec

**Date:** 2026-04-20  
**Branch:** `refactor/dashboard-complete`  
**Scope:** Bug fixes and UX improvements to `collab_splats/dashboard/semantics.py` and `collab_splats/utils/frame_sampling.py`

---

## 1. Image Orientation Fix

**Problem:** Frames display upside down. Root cause: `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)` in both `sample_frames_fps` and `sample_frames_optical_flow` disables OpenCV's built-in rotation correction. The manual fallback (`CAP_PROP_ORIENTATION_META` + `_apply_rotation`) fails when rotation metadata is absent or zero despite the video being physically inverted.

**Fix:** Remove `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)` from both functions. OpenCV defaults to auto-rotation (value=1), correctly handling rotation metadata. Also remove the now-unnecessary `rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))` lines and `_apply_rotation` calls within those functions. The `_apply_rotation` helper and `_rotation_map` can be deleted entirely.

**Files:** `collab_splats/utils/frame_sampling.py`

---

## 2. Frame Slider Layout Fix

**Problem:** `frame_slider` has `width=700` but lives inside `pn.Column(width=300)` (`frames_controls`), causing it to overflow and overlap the image pane on the right.

**Fix:** Move `frame_slider` out of `frames_controls`. Restructure the Frames inner tab layout:

```
Before:
  pn.Row(frames_controls[..., frame_slider], current_frame_pane)

After:
  pn.Column(
      pn.Row(frames_controls, current_frame_pane),
      frame_slider,          # full-width scrubber below both
  )
```

Set `frame_slider` to `sizing_mode="stretch_width"` (drop hardcoded `width=700`). Remove `width=700` from `frame_slider` widget definition.

**Files:** `collab_splats/dashboard/semantics.py`

---

## 3. Determinate Progress Bar for Frame Extraction

**Problem:** `_extract_frames` shows a static fullscreen overlay with no progress feedback. Extraction can take tens of seconds; users have no indication of progress.

**Design:**

- Add `pn.widgets.Progress` widget (`name="Extracting frames"`, `value=0`, `max=100`, `bar_color="info"`, `sizing_mode="stretch_width"`, `visible=False`) to the dashboard.
- Add `pn.pane.HTML` status label next to the bar for "X / Y frames".
- Get total extractable frame count upfront: `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` — O(1), no decoding.
- Run extraction in a background thread. Pass an `on_progress(current: int, total: int)` callback into the sampling functions; callback updates a `threading.Event`-protected counter tuple `(current, total)` on the dashboard.
- Poll at 200ms via `pn.state.add_periodic_callback`: read the counter, update `progress_bar.value` and label.
- Show bar on extraction start, hide on completion/error.
- Replace `_show_loading` / `_hide_loading` calls in `_extract_frames` with progress bar show/hide.

**Sampling function signature change (`frame_sampling.py`):**

```python
def sample_frames_fps(
    video_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]: ...

def sample_frames_optical_flow(
    video_path: str,
    ...,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]: ...
```

Callbacks are optional — existing callers unaffected.

**Files:** `collab_splats/dashboard/semantics.py`, `collab_splats/utils/frame_sampling.py`

---

## 4. Frame Extraction Speed Improvements

### 4a. FPS Mode — Seek-Based Extraction

**Problem:** `sample_frames_fps` calls `cap.read()` for every frame, decoding frames that are immediately discarded. At 1 fps from 30 fps video, 29 out of 30 frames are decoded and thrown away.

**Fix:** Use direct seeking:

```python
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0
while frame_num < total:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if on_progress:
        on_progress(frame_num, total)
    frame_num += interval
```

Seeking skips codec decoding for non-keyframe-aligned seeks, giving significant speedup at low fps ratios (e.g. ~10–20x at 1fps/30fps for H.264).

### 4b. Optical Flow Mode — Low-Resolution Analysis

**Problem:** OF analysis on full-res frames (e.g. 4K: 3840×2160) is slow. Selected frames are the only ones needing full resolution.

**Fix:** Resize each frame to max 480px wide before OF analysis; keep original `frame` for appending to results:

```python
scale = min(1.0, 480 / frame.shape[1])
small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
should_select, _, _ = selector.should_select_frame(small)
if should_select:
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # full-res
    selector.accept_frame(small)                            # low-res state
```

Expected speedup: ~4–9x on 4K footage; no impact on 1080p and below (scale=1.0, no resize).

Progress callback in OF mode fires on every decoded frame (same semantics as FPS mode), using `total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` as denominator. Add a decoded-frame counter inside the while loop alongside the existing `len(frames) < max_frames` guard.

**Files:** `collab_splats/utils/frame_sampling.py`

---

## 5. Branch Cleanup

Delete local branches that are fully absorbed into `refactor/dashboard-complete`:

| Branch | Status | Action |
|--------|--------|--------|
| `dashboard` | CUDA auto-detect cherry-picked into PR2 | `git branch -d dashboard` |
| `refactor/dashboard-optical-flow` | Full dashboard snapshot absorbed into PR2 | `git branch -d refactor/dashboard-optical-flow` |
| `tlb-grouping-segmentation` | Parked — separate plan | Leave |
| `tlb-improve-mesh` | Parked — separate plan | Leave |
| `tlb-improve-splatter` | Parked — separate plan | Leave |

No remote branches to delete (remotes for `dashboard` and `refactor/dashboard-optical-flow` don't appear in remote listing).

---

## Affected Files Summary

| File | Changes |
|------|---------|
| `collab_splats/utils/frame_sampling.py` | Remove auto-rotation disable; seek-based FPS extraction; low-res OF analysis; `on_progress` callbacks |
| `collab_splats/dashboard/semantics.py` | Progress bar widget; threaded extraction; frame slider layout fix |

## Out of Scope

- Optical flow total frame count accuracy (using `max_frames` as denominator is an acceptable approximation)
- Remote branch deletion (remotes for these branches not present)
- `tlb-*` branch work (separate plans)
