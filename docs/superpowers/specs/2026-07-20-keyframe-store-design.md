# Keyframe Store — Design

**Date:** 2026-07-20
**Status:** draft
**Related:** [windowed-streaming-reconstruction](2026-07-20-windowed-streaming-reconstruction-design.md) (depends on this)

## Context

Frame decoding is a package-wide inefficiency. `sample_frames` (`preproc/sampling.py`)
picks keyframes in a single streamed decode pass and returns in-memory arrays, but the
selected frames are **not persisted in a canonical form**. Every downstream consumer that
later needs those frames re-decodes the whole video by index:

- `preproc/viz.py:135` — `load_frames`
- `dashboard/pipeline.py:165`
- `wrapper/reconstructor.py:72,84`
- `wrapper/splatter.py:311`
- `webapp/routers/preprocess.py:89`
- feedforward `_preprocess` reloads frames from an image dir

`load_frames` / `extract_frame` re-stream the video via `_iter_frames_at` up to the max
requested index — a second (near-)full decode pass. For a >1000-frame video this is a
large, repeated cost, and it blocks the windowed-streaming reconstruction goal
(decode-once, read windows from disk).

The fix is a **canonical keyframe store**: the preprocess stage decodes once, writes
keyframes + records + provenance to a dedicated artifact, and all stages read from it
cheaply. This is upstream of feedforward and independent of loop closure.

## Goals

- Decode a source video **exactly once**; persist selected keyframes to a canonical store.
- A cheap **partial-read accessor**: fetch a keyframe (or subset) by index without decoding.
- Migrate **all** current re-decode callers onto the accessor.
- Store carries enough **provenance** to detect staleness (video path, fps, sampling params).

## Non-goals

- Changing the keyframe *selection* algorithm (`optical_flow` / `uniform` gates stay).
- Windowed reconstruction, submap spill, viewer (that is Spec 2).
- Caching arbitrary intermediate tensors (only decoded keyframes + their records).

## Design

### Store: `keyframes.zarr`

A dedicated preproc artifact written next to other stage outputs (sibling of
`feedforward.zarr`, not inside it — keyframes are upstream of the pointcloud stage).

```
keyframes.zarr/
  images        (N, H, W, 3) uint8, chunked per-frame, Blosc-compressed
  frame_idx     (N,) int      # source frame index in the original video
  blur_score    (N,) float
  score         (N,) float    # optical_flow composite score (NaN for uniform)
  selected      (N,) bool
  disparity     (N,) float    # optical_flow only, else NaN
  rotation      (N,) float
  histogram_similarity (N,) float
  attrs: {video_path, video_sha_or_mtime, fps, method, max_frames,
          min_disparity, motion_weight, coverage_weight, blur_threshold,
          native_fps, model_version}
```

- Chunk `images` one frame per chunk so a consumer reads a single keyframe without
  touching the rest. Blosc codec per repo convention (`compressors=[BloscCodec(...)]`,
  `store.create_array`; zarr 3.x — see project zarr-v3 note).
- Records mirror the dicts `sample_frames` already returns, stored columnar.

### Accessor

New module `preproc/keyframe_store.py`:

```python
class KeyframeStore:
    @classmethod
    def create(cls, path, frames, records, *, provenance) -> "KeyframeStore"
    @classmethod
    def open(cls, path) -> "KeyframeStore"

    def __len__(self) -> int
    def image(self, i) -> np.ndarray            # (H,W,3) uint8, partial read
    def images(self, idxs=None) -> np.ndarray   # subset or all
    def record(self, i) -> dict
    def records(self) -> list[dict]
    def frame_indices(self) -> np.ndarray
    def is_stale(self, video_path, params) -> bool   # provenance mismatch
```

- `is_stale` compares stored provenance against the current request so a stage can decide
  to reuse vs re-sample. Keeps the store honest without a rebuild-every-run cost.

### Producer: preprocess stage

`wrapper/reconstructor.py` preprocess stage calls `sample_frames` once, then
`KeyframeStore.create(...)` writes `keyframes.zarr`. The stage output becomes "the store
exists," not "frames returned in memory." Reuse existing store if `is_stale` is False.

### Consumer migration

Replace every re-decode call site with `KeyframeStore.open(...).image(i)` /
`.images(idxs)`:

- feedforward `_preprocess` reads keyframes from the store (also unblocks Spec 2's
  per-window `_preprocess_window(paths_or_idxs)`).
- `preproc/viz.py`, `dashboard/pipeline.py`, `wrapper/reconstructor.py`,
  `wrapper/splatter.py`, `webapp/routers/preprocess.py` — swap `load_frames` /
  `extract_frame*` for accessor reads.
- Keep `extract_frames` / `load_frames` as thin fallbacks only where a raw video with no
  store is a legitimate input; otherwise route through the store.

## Testing

- **Round-trip:** `create` then `open`; `image(i)` equals the source keyframe; records match.
- **Partial read:** reading one frame does not materialize the whole `images` array
  (assert via chunk access / memory).
- **Provenance/staleness:** `is_stale` True on changed video/params, False on match.
- **Consumer parity:** each migrated caller produces the same output reading from the store
  as it did re-decoding (golden compare on a short clip).
- **No-double-decode:** end-to-end preprocess→pointcloud on a short video decodes the video
  once (assert `_iter_frames` invoked a single pass, e.g. via a decode counter).

## Verification

Run preprocess + pointcloud on a short clip through `docs/examples/run_scenes.py`; confirm
`keyframes.zarr` is written, the pointcloud stage reads it (no re-decode), and outputs are
unchanged vs the pre-migration baseline. Run the test suite.
