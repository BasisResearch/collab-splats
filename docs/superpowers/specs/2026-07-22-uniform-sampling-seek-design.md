# Uniform frame sampling: seek-based + exact count — design

**Date:** 2026-07-22
**Status:** Approved, ready for planning

## Problem

Two user-reported problems with uniform frame sampling (the dashboard's default "Max frames" mode; there is no literal "balanced" mode — the method is `"uniform"`):

1. **It decodes every frame before selecting.** `_sample_uniform` (`collab_splats/preproc/sampling.py:400`) loops `_iter_frames` (`sampling.py:152`) — one sequential ffmpeg rawvideo pipe that decodes and blur-scores **every** frame. `interval = total // max_frames` (line 418) spreads windows across the whole video, so the early-exit at line 440 almost never fires before EOF. Slow on long videos.

2. **The returned count is fewer than requested (e.g. 100 → fewer).** The quality gate (`sampling.py:431-432`) keeps only the sharpest *gate-passing* frame per window; a window in which no frame passes the blur/exposure gate yields **zero** frames (line 436). `max_frames` is therefore a ceiling that the gate pushes below target. Integer-floor `interval` also skews the window count off target.

### Ruled out: the overwrite hypothesis

The count shortfall is **not** a stale-store / overwrite bug. `FrameStore.create` (`collab_splats/preproc/frame_store.py:42`) opens with `zarr.open(mode="w")`; zarr v3.1.6 fully replaces the group (deletes the prior `images` array and all record columns) before writing. The dashboard's displayed count is the in-memory `len(frames)` (`collab_splats/dashboard/pipeline.py:243`), never a re-read of the store. A re-run with a new target fully replaces `frames.zarr`; no stale frames survive.

## Goals

- Uniform sampling seeks to evenly-spaced positions (O(1) input-seek, no full decode).
- Returns exactly `min(max_frames, total)` frames.

## Non-goals

- `_sample_optical_flow` is unchanged — full decode is inherent to its motion scoring.
- No sharpest-within-window selection: dropped intentionally for speed (see tradeoffs).

## Approach

- **Seek-based uniform:** replace the sequential full decode with O(1) input-seeks at evenly-spaced indices.
- **Best-effort fallback:** every target position yields a frame (the seeked frame, even if it fails the quality gate), so the count is exact.

Combined, the returned count is exactly `min(max_frames, total)`.

## Design

### 1. Probe-once seek helper (`sampling.py`)

`extract_frame` (lines 561-600) currently calls `get_video_info` (an ffprobe subprocess) on **every** call. Seeking 100 times would spawn 100 ffprobes. Split the seek body out so callers can probe once:

```
def _seek_frame(video_path, frame_idx, *, fps, w, h) -> np.ndarray
    # the existing ffmpeg -ss input-seek body, verbatim; no get_video_info

def extract_frame(video_path, frame_idx) -> np.ndarray
    # probe via get_video_info, then delegate to _seek_frame
```

`extract_frame`'s public contract is unchanged (probe + seek, same `-ss` timestamp math `seek_s = max(frame_idx-0.5, 0)/fps`, rgb24 pipe, same range/`ValueError` checks).

### 2. Rewrite `_sample_uniform` (lines 400-448)

- Probe once (`get_video_info`): native `fps`, `w`, `h`, `total`. Return `[], []` if `total == 0`.
- Compute the target indices:
  - `fps` arg given → `step = max(1, round(native_fps / fps))`, indices `= range(0, total, step)`.
  - elif `max_frames` → `n = min(max_frames, total)`, indices `= np.unique(np.linspace(0, total - 1, n).round().astype(int))`.
  - else → fall back to 2.0 fps (`step = round(native_fps / 2.0)`), as today.
- For each target index: `_seek_frame` decode (RGB), `_analysis_gray` + `compute_blur_score` + `check_frame_quality`. **Best-effort: always append the frame**; record `{"frame_idx": idx, "blur_score": blur}`. The gate result no longer drops the frame (it may be recorded as a `usable` flag if free, but the frame is kept so the count stays exact).
- Progress (`_progress_reporter`) reports over `n` (small), not `total`.
- Return exactly `len(indices)` frames.

**Reuse (all already in the module):** `get_video_info`, `_analysis_gray`, `compute_blur_score`, `check_frame_quality`, `_progress_reporter`.

**Record keys** stay `frame_idx` (source video index) and `blur_score` — consumed by `FrameStore.create` columns and `pipeline.py:242`.

### 3. Dashboard / pipeline — no change

`pipeline.py:160-185` already passes `max_frames`; the displayed count is already `len(frames)`. The fix is entirely inside `_sample_uniform`.

## Tradeoffs

- Pure seek lands on exact evenly-spaced frames; a target frame that happens to be motion-blurred is kept (best-effort). Sharpest-within-window selection is intentionally dropped for speed.
- VFR sources: `_seek_frame` may land ±1 frame near keyframes (the caveat `extract_frame` already documents).
- `total < max_frames`: returns all `total` frames — can't exceed the source.

## Testing

- Unit (`tests/preproc/test_sampling.py`):
  - `len(sample_frames(video, method="uniform", max_frames=N)) == min(N, total)`.
  - A frame is still returned per position when the gate would reject every frame (best-effort).
  - Seek count ≈ N, not `total` (patch `_seek_frame` / count calls) — proves no full decode.
  - Update or remove any existing test asserting the old window/interval behavior.
- Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -q`
- Dashboard smoke gate (mandatory pre-commit per CLAUDE.md): `python -m collab_splats.dashboard --smoke` → `SMOKE PASS`.
- Manual: dashboard Max frames = 100 on the tutorial video → exactly 100 frames, no whole-video decode.
- Format: `black . && isort .`
