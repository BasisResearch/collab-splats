# Spec: Frame Extraction — Orientation Fix & Performance

**Date:** 2026-04-19  
**Branch:** `refactor/dashboard-optical-flow`

## Problem

1. **Orientation**: extracted frames display sideways (90° CCW from correct). Root cause: rotation map in `frame_sampling.py` uses wrong direction (`90 → ROTATE_90_COUNTERCLOCKWISE` should be `ROTATE_90_CLOCKWISE`). Additionally, OpenCV may auto-apply rotation via `CAP_PROP_ORIENTATION_AUTO`, causing double-rotation that cancels the manual correction.

2. **Performance**: `sample_frames_optical_flow` runs Farneback dense optical flow on full-resolution frames and applies rotation to every decoded frame (not just selected ones). For 1080p+ video this is significantly slower than necessary — flow computation cost scales with pixel count.

3. **Bug**: `splatter.py` calls `sample_frames_optical_flow(..., n_samples=n_samples)` but the function signature uses `max_frames`, not `n_samples`. This raises `TypeError` in the Splatter preprocessing path.

4. **Dead code**: `_load_frames_from_video()` in `semantics.py` duplicates sampler logic and is never called.

5. **UX**: Dashboard tab order puts Training before Explore, but Explore is the primary interactive workflow.

## Changes

### Commit 1 — Fix frame orientation (`frame_sampling.py`)

- Add `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)` in both `sample_frames_fps` and `sample_frames_optical_flow` immediately after `cv2.VideoCapture(...)`. Disables OpenCV auto-rotation so manual correction has exclusive control.
- Fix `_rotation_map()`: swap `90 → cv2.ROTATE_90_CLOCKWISE` and `270 → cv2.ROTATE_90_COUNTERCLOCKWISE`.

**Files:** `collab_splats/semantics/frame_sampling.py`  
**Lines changed:** ~8

### Commit 2 — Optimize optical flow (`frame_sampling.py`)

- Add `downsample_factor: float = 0.25` parameter to `sample_frames_optical_flow`. Default 0.25 gives 1/16 pixel count = ~16× faster Farneback at any input resolution. Aspect ratio preserved.
- Resize gray frame by `downsample_factor` before Farneback: `cv2.resize(gray, (0, 0), fx=downsample_factor, fy=downsample_factor)`.
- Keep full-res frame reference for saving; only resize the gray copy used for flow.
- Replace `np.sum(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))` with `np.mean(np.hypot(flow[..., 0], flow[..., 1]))`. Mean is scale-invariant — threshold values remain valid regardless of `downsample_factor`.
- Move `_apply_rotation(frame, rotation)` inside the save branch (`accumulated_flow >= flow_threshold`) so rotation is only applied to selected frames, not every decoded frame.

**Files:** `collab_splats/semantics/frame_sampling.py`  
**Lines changed:** ~15

### Commit 3 — Bug fix, dead code removal, tab reorder (`splatter.py`, `semantics.py`)

- `splatter.py`: fix `sample_frames_optical_flow(..., n_samples=n_samples)` → `max_frames=n_samples`.
- `semantics.py`: delete `_load_frames_from_video()` (~25 lines removed).
- `semantics.py`: swap tab order in `create_layout()` → `("Explore", explore_tab)` before `("Training", training_tab)`.

**Files:** `collab_splats/wrapper/splatter.py`, `collab_splats/dashboard/semantics.py`  
**Lines changed:** ~10 net

## Non-Goals

- No changes to FPS sampler performance (already efficient — rotation only applied to selected frames).
- No changes to Farneback parameters (`pyr_scale`, `levels`, `winsize`) — `downsample_factor` is sufficient.
- No new dependencies.

## Testing

- Extract frames from a portrait-orientation phone video; verify upright display.
- Extract frames with `sampling_mode=Optical Flow`; verify no `TypeError` in Splatter wrapper.
- Verify Explore tab appears first in dashboard.
