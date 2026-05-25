# Video Decode & Frame Sampling Speedup

**Date:** 2026-05-21
**Branch:** refactor/core-modules

## Problem

`score_all_frames` and `sample_frames_optical_flow` decode every frame at native fps (e.g., 18 000 frames for a 10-min 30fps video) then run Lucas-Kanade optical flow on each. `sample_frames_fps` also decodes every frame sequentially even when it only needs 1 in N.

## Goals

- Reduce optical flow compute in scoring paths ~5-7x
- Eliminate sequential decode waste in fps sampling path
- Add terminal progress bars to all three public functions
- Zero new dependencies, minimal code change

## Non-Goals

- Swapping video decode library (PyAV, FFmpeg subprocess, torchcodec)
- Changing `OpticalFlowFrameSelector` internals
- Changing `min_disparity=50` default (must stay aligned with VGGT-SLAM)

## Approach: cv2-only, ~15 lines changed

### `score_all_frames` + `sample_frames_optical_flow`

Add `stride: int = 5` parameter. Use `cap.grab()` for skipped frames, `cap.read()` only for frames that will be processed.

- `stride=5` on 30fps source → effective 6fps scoring rate → ~5x OF compute reduction
- `cap.grab()` reads compressed packet, skips YUV→BGR decode + numpy allocation
- `frame_idx` in results always reflects original video frame index (not stride counter)
- `on_progress` callback still fires per frame for external callers
- tqdm wraps the loop, gated by `verbose: bool = True`

```python
def score_all_frames(
    video_path: str,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
    stride: int = 5,
    verbose: bool = True,
) -> list[dict]:
```

Loop pattern:
```python
with tqdm(total=total, desc="scoring frames", unit="frame", disable=not verbose) as pbar:
    while True:
        if frame_idx % stride == 0:
            ret, frame = cap.read()
            if not ret:
                break
            # existing OF logic — unchanged
        else:
            if not cap.grab():
                break
        pbar.update(1)
        if on_progress:
            on_progress(frame_idx, total)
        frame_idx += 1
```

### `sample_frames_fps`

Replace sequential decode-and-discard loop with `cap.set(CAP_PROP_POS_FRAMES)` seek per target frame.

- Seeks to nearest I-frame before target — accuracy ±GOP size (~2s at 30fps)
- Acceptable for reconstruction fps sampling (approximate regular intervals)
- `on_progress` semantic changes: fires per extracted frame (not per decoded frame)

```python
def sample_frames_fps(
    video_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
    verbose: bool = True,
) -> list[np.ndarray]:
```

Loop pattern:
```python
target_frame_indices = [round(i * interval) for i in range(n_targets)]
for i, target_idx in tqdm(
    enumerate(target_frame_indices),
    total=len(target_frame_indices),
    desc="extracting frames",
    unit="frame",
    disable=not verbose,
):
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
    if on_progress:
        on_progress(i, len(target_frame_indices))
    if max_frames is not None and len(frames) >= max_frames:
        break
```

## Robustness Notes

- `cap.grab()` within H.264/HEVC GOP: reads packet but does not allocate RGB frame. Does not save codec decode within a GOP — saves YUV→BGR + numpy allocation only. OF compute is the primary bottleneck.
- `cap.set(CAP_PROP_POS_FRAMES)` seek: lands on nearest keyframe before target. Max error = GOP size (typically 60 frames = 2s at 30fps). Acceptable for reconstruction use.
- `frame_idx` correctness: must increment for every frame (grabbed or read) so results index original video positions.
- `OpticalFlowFrameSelector` tracks disparity from last *accepted* keyframe, not last decoded frame — striding does not corrupt selector state.

## Files Changed

- `collab_splats/utils/frame_sampling.py` — only file touched

## Testing

- `tests/utils/test_frame_sampling.py` — existing tests must pass with default params
- Add: test `stride=5` produces subset of `stride=1` frame_idxs (monotone, original indices)
- Add: test `sample_frames_fps` seek returns correct frame count for short test video
- Add: test `verbose=False` suppresses output (mock tqdm or capture stderr)
