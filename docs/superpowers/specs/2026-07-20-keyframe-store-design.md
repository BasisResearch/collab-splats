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

The fix is a **canonical frame store**: the preprocess stage decodes once, writes
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

### Store: `frames.zarr`

A dedicated preproc artifact written next to other stage outputs (sibling of
`feedforward.zarr`, not inside it — keyframes are upstream of the pointcloud stage). It is
the **sole persistent RAW-frame source**: it replaces the old `output_path/images/` JPG dir.
(It does NOT replace `feedforward.zarr`'s `images` array — that is the model-resolution
depth-aligned tensor, a distinct artifact, kept. See the correction note below.) Consumers with
path-locked APIs (VGGT-X / MapAnything model preprocessing, external extractors) get a
**transient** `export(dir)` deleted after use — a derived copy, not persistent duplication.
COLMAP (names + arrays), nerfstudio and splatter (raw video) never touch it.

```
frames.zarr/
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

New module `preproc/frame_store.py`:

```python
class FrameStore:
    @classmethod
    def create(cls, path, frames, records, *, provenance) -> "FrameStore"
    @classmethod
    def open(cls, path) -> "FrameStore"

    def __len__(self) -> int
    def image(self, i) -> np.ndarray            # (H,W,3) uint8, partial read
    def images(self, idxs=None) -> np.ndarray   # subset or all
    def record(self, i) -> dict
    def records(self) -> list[dict]
    def image_by_frame_idx(self, frame_idx) -> np.ndarray   # lookup by SOURCE video index
    def frame_indices(self) -> np.ndarray
    def is_stale(self, provenance) -> bool                   # provenance mismatch
    def export(self, out_dir, *, ext="jpg") -> list[Path]    # transient dir for path-locked consumers
```

- `is_stale` compares stored provenance against the current request so a stage can decide
  to reuse vs re-sample. Keeps the store honest without a rebuild-every-run cost.

### Producer: preprocess stage

`wrapper/reconstructor.py` preprocess stage calls `sample_frames` once, then
`FrameStore.create(...)` writes `frames.zarr`. The stage output becomes "the store
exists," not "frames returned in memory." Reuse existing store if `is_stale` is False.

### Consumer migration

Replace every re-decode call site with `FrameStore.open(...).image(i)` /
`.images(idxs)`:

- feedforward `_preprocess` reads keyframes from the store (also unblocks Spec 2's
  per-window `_preprocess_window(paths_or_idxs)`).
- `preproc/viz.py`, `dashboard/pipeline.py`, `wrapper/reconstructor.py`,
  `wrapper/splatter.py`, `webapp/routers/preprocess.py` — swap `load_frames` /
  `extract_frame*` for accessor reads.
- Keep `extract_frames` / `load_frames` as thin fallbacks only where a raw video with no
  store is a legitimate input; otherwise route through the store.

### `images` in feedforward.zarr is NOT a duplicate — keep it

Correction (verified during implementation): `FeedforwardResult.images` (`base.py:70`) is the
**model-resolution, center-cropped, channel-first tensor pixel-aligned with the depth map** —
produced by `load_and_preprocess_images(mode="crop")` / MapAnything `img_no_norm`, at
`model_width×model_height`. It is NOT the same as `frames.zarr`'s raw full-res HWC keyframes
(different resolution, FOV, layout, dtype). It is consumed as pixels by **TSDF meshing, BA
track extraction, and per-point colors** (`reconstructor.py:285`, `bundle_adjustment.py:101/
173/209`, `vggtx.py:122`), which need pixels aligned to the model-res depth — a `frames.zarr`
read cannot substitute. **Keep the `images` array in feedforward.zarr.** The only cleanup: the
lift path (`_lift_and_save`) loads it via `load_images=True` even though `lift_features` never
reads `.images` — drop that flag to avoid loading the tensor when unused (a memory win, no
behavior change).

Future footprint work (separate spec): `images` is a pure resize+crop of `frames.zarr` and
`world_points` is unproject(depth, intrinsics, extrinsics) — both are derivable, so a later
optimization could store derivation params and regenerate on load, cutting the largest arrays.
That needs the pre-existing VGGT `[0,255]` vs MapAnything `[0,1]` image-scale inconsistency
fixed first, so it is deferred, not part of this refactor.

## Cleanup scope (preproc audit)

Fold these into the same change — the store refactor is the moment to make `preproc`
production-clean. Each is verified against repo callers.

**Dead code — delete:**
- `OpticalFlowFrameSelector.stats` (`sampling.py:255`, appended 280-282) — written every
  frame, **read nowhere**. Delete the dict + the three appends.

**Redundant params — remove (always default, no caller varies them):**
- `motion_weight` / `coverage_weight` threaded `sample_frames` → `_sample_optical_flow` →
  `OpticalFlowFrameSelector` → `_combine_scores` (`sampling.py:211,239-241,374-375,509`).
  No caller ever sets them. Hardcode 0.6 / 0.4, drop the weight-validation branch (244-247).
  Keep `min_disparity` — it *is* varied.

**Inefficiency — fix:**
- **Evals double-decode** (`evals/datasets.py:251,253`): `sample_frames(uniform)` decodes the
  whole video, discards the frames, then `extract_frames` **re-decodes** it to write JPEGs.
  Resolve by reading persisted frames from `frames.zarr` instead.
- **Full-file probe for two integers**: `_iter_frames` → `get_video_info` (`sampling.py:80`)
  always runs `ffprobe -count_packets` (demuxes the entire file) but only needs W/H. Split a
  cheap W/H-only probe from the packet-count path so every decode stops paying a full demux.

**Style:**
- `extract_frame` / `extract_frame_fast` annotate `"str | Path"` as **string literals**
  (`sampling.py:595,608`) despite `from __future__ import annotations` — make them bare
  `str | Path`.

**Do NOT trim** `check_frame_quality` / `reject_reason` from `__all__`: they were
deliberately promoted to public for the tutorial (`project_tutorial_keyframe_rework`); the
audit's zero-hit grep undercounts because several tutorial notebooks are currently
broken/pending a sweep. Keep public.

**Retirement folds into the store (from §Consumer migration above):**
- Delete `load_frames` (`592`), `extract_frames` (`652`), `_iter_frames_at` (`143`) — pure
  re-decode-by-index helpers the store replaces.
- `extract_frame` (`595`, exact-provenance re-decode) → its callers (`dashboard/pipeline.py`,
  localize) become store lookups; delete. This frees the plain name.
- **Keep** `extract_frame_fast` (`608`) — seek-based scrub preview for arbitrary
  (non-keyframe) indices the store does not hold — and **rename it `extract_frame`** (the
  O(N) original is gone, so the seek-based reader takes the canonical name; no `_fast`
  suffix qualifier). Update its two callers.

## Implementation principles

- **Reuse, don't rewrite.** Decode via the existing `_iter_frames` generator — no new decode
  path. Store the record dicts `sample_frames` already returns; reuse the Blosc/zarr helper
  pattern used elsewhere in the package. The accessor is a thin wrapper over `zarr`, not a
  new abstraction layer.
- **Retire dead code.** Once all callers read from the store, the re-decode helpers become
  dead: audit and remove `load_frames`, `extract_frame`, `extract_frame_fast`, and
  `_iter_frames_at` (`preproc/sampling.py`) unless a raw-video-with-no-store path genuinely
  still needs them — if so, keep exactly one, delete the rest. Do not leave both a store
  read and a re-decode fallback "just in case."
- **Minimal surface.** `FrameStore` gets only the methods listed; no speculative params
  (compression level, formats) until a caller needs them.
- **Inline block comments** on each logical block (schema write, provenance check, accessor
  read) per repo code style; one-line docstrings on public methods.

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
`frames.zarr` is written, the pointcloud stage reads it (no re-decode), and outputs are
unchanged vs the pre-migration baseline. Run the test suite.
