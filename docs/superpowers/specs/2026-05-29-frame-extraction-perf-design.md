# Frame Extraction Performance & Correctness

**Date:** 2026-05-29  
**Status:** approved  
**Scope:** `collab_splats/utils/frame_sampling.py`, `collab_splats/dashboard/panes/preprocess.py`

---

## Problem

Two bugs in the balanced frame extraction path (`sample_frames_fps` → `_decode_fps_ffmpeg`):

1. **Corruption** — ffmpeg auto-rotates video by default. `_decode_fps_ffmpeg` computes `frame_size` and reshapes output using native cv2 dimensions, which are pre-rotation. For 90°/270° rotated videos (e.g. GoPro portrait), the output dims are swapped → every byte lands in the wrong pixel position → garbled thumbnails with wrong aspect ratio.

2. **Speed** — `_decode_fps_ffmpeg` spawns one ffmpeg subprocess per frame (up to 8 workers). For 200 frames: 25 rounds of 8 concurrent process launches + seek + decode + pipe. A single ffmpeg pass with a select filter reads the video once and is substantially faster.

3. **Memory** — `_write_frames_zarr` calls `np.stack(frames)` before writing, materialising all frames in RAM simultaneously. For 200 4K frames this is ~5 GB.

---

## Design

### Change 1 — Rotation-aware reshape in `_decode_fps_ffmpeg`

ffmpeg applies rotation from container metadata automatically (default behaviour). The output pixel dimensions after rotation differ from native cv2 dimensions for 90°/270° rotated videos.

Fix: call `_get_rotation_degrees(video_path)` once per call, then:

```python
rotation = _get_rotation_degrees(video_path)
out_h = width if rotation in (90, 270) else height
out_w = height if rotation in (90, 270) else width
frame_size = out_w * out_h * 3
# ...
np.frombuffer(raw, np.uint8).reshape(out_h, out_w, 3).copy()
```

No change to the ffmpeg command — auto-rotation is the correct behaviour; we just reshape to match the actual output.

### Change 2 — Single-pass ffmpeg replaces parallel-seek

`sample_frames_fps` always produces evenly-spaced targets (`range(0, total, interval)`). For this pattern, one ffmpeg invocation with a select filter is faster than N parallel seeks because it reads the video file once.

Replace `_decode_fps_ffmpeg` (parallel per-frame seeks) with a single-pass implementation:

```
ffmpeg -i <video> \
  -vf "select='not(mod(n\,INTERVAL))',setpts=N/FRAME_RATE/TB" \
  -frames:v <N_TARGETS> -vsync 0 \
  -f rawvideo -pix_fmt rgb24 -an pipe:1
```

Read `N_TARGETS × frame_size` bytes from the pipe, slice into frames. Fire `on_progress(i, N_TARGETS)` after each frame is sliced. Apply rotation-aware reshape (Change 1).

The `interval` is computed as `targets[1] - targets[0]` when `len(targets) > 1`, else `1`.

**Remove** the old parallel-seek `_decode_fps_ffmpeg`. The torchcodec and cv2 fallback paths in `sample_frames_fps` are unchanged.

### Change 3 — Streaming zarr write + resolution cap

Replace `_write_frames_zarr` in `preprocess.py`:

**Before:**
```python
arr = np.stack(frames)          # all frames in RAM at once
store["frames"][:] = arr
```

**After:**
- Pre-allocate zarr array using shape from first frame (after optional resize)
- Write frames one-by-one: `store["frames"][i] = frame`
- Cap width at 1920px before writing (resize if source wider); height scaled proportionally

Resolution cap rationale: zarr frames are used only by the dashboard UI (320×240 thumbnails, semantics pane display). The feedforward pipeline reads directly from video, not from zarr. Capping at 1920px wide reduces zarr size ~4× for 4K source with no downstream impact.

---

## Affected Files

| File | Change |
|------|--------|
| `collab_splats/utils/frame_sampling.py` | Replace `_decode_fps_ffmpeg` with single-pass impl; add rotation-aware reshape |
| `collab_splats/dashboard/panes/preprocess.py` | Replace `_write_frames_zarr` with streaming + resize |

## Not Affected

- `sample_frames_optical_flow` — uses `_iter_decoded_frames` (cv2 fallback with manual rotation), correct already
- `load_video_frames`, `extract_video_frames` — use cv2 with `_apply_rotation`, correct already
- Keyframe extraction notebook — calls `sample_frames_fps` / `sample_frames_optical_flow` by name; single-pass is a backend swap, transparent to callers
- `_iter_decoded_frames` ffmpeg path — used only by `sample_frames_optical_flow` which decodes all frames sequentially; rotation bug there too (uses same width/height) but that function is on a different code path and out of scope here

## Testing

- Unit test for rotation-aware reshape: mock `_get_rotation_degrees` returning 0/90/180/270; verify `out_h`, `out_w` are correct for each
- Unit test for `_write_frames_zarr` streaming: verify zarr shape matches resized dims; verify no frame exceeds 1920px wide
- Integration: manual run on C0043.MP4 — thumbnails should show correct orientation and aspect ratio; extraction time should be < 20s
