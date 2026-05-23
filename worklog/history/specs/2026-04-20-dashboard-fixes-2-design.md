# Dashboard Fixes (Session 3) — Design Spec

**Date:** 2026-04-20  
**Approach:** A — targeted fixes, no refactoring

Changes split by branch: `frame_sampling.py` lives in `collab_splats/utils/` (PR1 scope); dashboard UI lives in `collab_splats/dashboard/` (PR2 scope).

---

## PR1 — `refactor/core-modules`

### A. Slow Frame Extraction (FPS mode)

**Problem:** `sample_frames_fps` uses `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)` + `cap.read()` per extracted frame. For H.264 video, each seek forces a full keyframe-decode restart; consecutive forward seeks are O(n²) in the worst case. Typical GOP size of 60 frames means every seek at interval=6 re-decodes ~60 frames from the nearest I-frame.

**Fix:** Sequential read — iterate every frame with `cap.read()`, keep every `interval`-th frame, fire `on_progress(frame_num, total)` on every decoded frame. One linear pass, no seeks.

```python
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % interval == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    if on_progress is not None:
        on_progress(frame_num, total)
    frame_num += 1
```

**File:** `collab_splats/utils/frame_sampling.py`

### B. Max Frames Parameter

**Problem:** FPS mode produces frame counts proportional to video length — no video-length-agnostic cap exists.

**Fix:** Add `max_frames: int | None = None` to `sample_frames_fps` signature (integrated into sequential loop above). `None` = no cap; existing callers unaffected. `sample_frames_optical_flow` already has `max_frames`; no change needed there.

**File:** `collab_splats/utils/frame_sampling.py`

### C. Image Displayed Upside Down

**Problem:** Orientation fix was specced in `2026-04-20-dashboard-fixes-design.md` but never committed. Both sampling functions open `cv2.VideoCapture` without setting orientation auto-correction, so container rotation metadata is ignored.

**Fix:** Add `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)` immediately after `cv2.VideoCapture(video_path)` in both `sample_frames_fps` and `sample_frames_optical_flow`.

**File:** `collab_splats/utils/frame_sampling.py`

---

## PR2 — `refactor/dashboard-complete`

### D. E-1021 Slider start==end

**Problem:** `frame_slider` initialised with `start=0, end=0`. Bokeh fires `E-1021 (EQUAL_SLIDER_START_END)` immediately on page load.

**Fix:** Initialise `end=1`. In `_poll_extraction_progress`, set `end=max(1, n-1)`. The slider is visually harmless with `end=1` before any frames are loaded.

**File:** `collab_splats/dashboard/semantics.py`

### E. Progress Bar Width

**Problem:** `pn.Row(self.progress_bar, self.progress_label)` sits at the top of the Frames tab at full tab width (`sizing_mode="stretch_width"`).

**Fix:** Remove the row from the top-level `pn.Column`. Insert both widgets as the *first* items in `frames_controls` (before `fps_slider`). The `frames_controls` column has `width=300`, which constrains the bar naturally. Keep `sizing_mode="stretch_width"` on the bar — it stretches within `width=300`.

**File:** `collab_splats/dashboard/semantics.py`

### F. Max Frames UI

**Fix:**
- Add `max_frames_slider = pn.widgets.IntSlider(name="Max frames", value=200, start=10, end=2000, step=10, width=350)` to dashboard `__init__`.
- Place it in `frames_controls` below `fps_slider`.
- Wire `self.max_frames_slider.value` to both `sample_frames_fps` and `sample_frames_optical_flow` calls in `_extract_frames`.

**File:** `collab_splats/dashboard/semantics.py`

---

## Affected Files Summary

| Branch | File | Changes |
|--------|------|---------|
| `refactor/core-modules` | `collab_splats/utils/frame_sampling.py` | Sequential read; `max_frames` param; `CAP_PROP_ORIENTATION_AUTO=1` |
| `refactor/dashboard-complete` | `collab_splats/dashboard/semantics.py` | `frame_slider` init `end=1`; progress bar into `frames_controls`; `max_frames_slider` widget + wiring |

## Out of Scope

- Optical flow sequential read (already sequential; no seek used)
- Seek-based extraction for very-low FPS (< 0.1 fps) — not a use case here
- Remote branch deletion
