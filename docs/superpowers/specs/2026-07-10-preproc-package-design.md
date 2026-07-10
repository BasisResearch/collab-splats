# Preproc Package Design

**Date:** 2026-07-10
**Status:** Approved (brainstorm complete)
**Scope:** Promote frame sampling to a top-level `collab_splats/preproc/` package; simplify to a single ffmpeg decode path; add a per-frame quality gate (blur + exposure); unify the sampler API; delete dead code.

## Motivation

`collab_splats/utils/frame_sampling.py` (869 lines) mixes five concerns in one
file: video decode backends, rotation metadata, keyframe selection, frame I/O,
and matplotlib visualization. Problems:

- **Three decode backends** (torchcodec / ffmpeg / cv2) × two access patterns
  (stream, seek), each with its own rotation handling. The file itself
  documents torchcodec hangs and cv2 seek inaccuracy; the fps path already
  prefers ffmpeg.
- **matplotlib imported at module top** — every pipeline consumer (webapp,
  wrapper, dashboard) pays the import for notebook-only plot functions.
- **Duplicated loops**: the scoring loop appears twice
  (`score_all_frames` / `sample_frames_optical_flow`); the cv2 seek loop
  appears three times.
- **Dead knobs** verified unused: `reset()`, homography branch in
  `_estimate_rotation`, `return_components`, `stride`, `native_fps`,
  always-default ctor params.
- **No frame-quality filtering** — motion-blurred and blown-out frames pass
  selection today.
- Callers (webapp `routers/preprocess.py`, dashboard `pipeline.py`) branch
  manually on a method string that should be library dispatch.

Precedent: `localization/` was promoted to a top-level stage package
(a4055c7, hard cut, no shim).

## Package Structure

```
collab_splats/preproc/
  __init__.py   # public API re-exports (~10 lines)
  sampling.py   # ~440 lines: ffmpeg/ffprobe decode + video info, quality gate,
                # OpticalFlowFrameSelector, sample_frames dispatcher,
                # frame I/O, score I/O
  viz.py        # ~110 lines: plot_frame_grid, plot_selection,
                # plot_frame_scores, plot_disparity_sensitivity
```

- `viz.py` is **not** re-exported from `__init__` — matplotlib loads only on
  explicit `from collab_splats.preproc.viz import ...` (notebooks).
- No `video.py`/`quality.py` split: no independent consumers; revisit only if
  future capabilities (dynamic masks, depth-consistency filters) push
  `sampling.py` past ~900 lines.

## Public API (`__init__.py`)

```python
sample_frames(video_path, method="uniform" | "optical_flow", ...)
get_video_info(video_path)
load_video_frames(video_path, frame_indices)
extract_video_frames(video_path, frame_indices, output_dir)
score_all_frames(video_path, ...)
save_frame_scores(scores, path) / load_frame_scores(path)
```

`score_all_frames` and score I/O survive because the
`01_preprocessing/keyframe_extraction.ipynb` tutorial consumes them.

### `sample_frames` dispatcher

```python
def sample_frames(
    video_path: str,
    *,                                  # all params keyword-only
    method: str = "uniform",            # "uniform" | "optical_flow"
    max_frames: int | None = None,
    fps: float = 2.0,                   # uniform only
    min_disparity: float = 50.0,        # optical_flow only
    motion_weight: float = 0.6,         # optical_flow only
    coverage_weight: float = 0.4,       # optical_flow only
    blur_threshold: float = 50.0,       # quality gate, both methods
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
```

Rationale for one entry point: both methods share ~80% of the body (decode
loop, quality gate, `max_frames`, progress); webapp and dashboard already
receive the method as a user-config string and branch manually — dispatch
moves into the library once. Old `sample_frames_fps` /
`sample_frames_optical_flow` become internals. Unknown method raises
`ValueError`. If a third method ever lands, switch to a registry then.

Explicit keyword params, deliberately not `**kwargs` passthrough: at 8 params
the flat signature is self-documenting; `**kwargs` would silently swallow
typos and kill autocomplete. Params are keyword-only (`*,`) to prevent
positional misuse. Revisit (per-method config or registry) only when a third
method brings its own params.

**Return contract (both methods):** `(frames, records)` where `frames` is
RGB arrays and `records` has one dict per selected frame, always containing
`frame_idx` (source video index — fixes the current optical-flow bug where
`frame_idx` is the output list position, not the source index) and
`blur_score`; optical-flow adds `disparity`, `rotation`,
`histogram_similarity`, `score`. Callers needing bare indices use
`[r["frame_idx"] for r in records]`.

## Decode: ffmpeg-only

- Delete `_get_decoder_backend`, all torchcodec paths, all cv2 decode paths.
- Streaming decode: single ffmpeg rawvideo pipe (existing `_iter_decoded_frames`
  ffmpeg branch, now the only branch).
- Index-based extraction: ffmpeg `select` filter (existing `_decode_fps_ffmpeg`
  generalized to arbitrary sorted index lists so `load_video_frames` /
  `extract_video_frames` share it).
- `get_video_info` moves from cv2 to ffprobe — one probe returns
  frames/fps/dims/rotation together (rotation logic from
  `_get_rotation_degrees` folds in).
- ffmpeg applies rotation metadata itself → `_apply_rotation` and cv2
  orientation flags die; `_ffmpeg_output_dims` remains the single rotation
  touchpoint.
- cv2 stays a dependency for **image ops only** (Laplacian, resize,
  histograms, LK flow, JPEG write).
- Missing ffmpeg → `RuntimeError` with install hint at first use.

## Quality gate (new)

One pure function used by both methods, checked on the 480px grayscale
analysis frame before any flow computation:

```python
def _frame_ok(gray: np.ndarray, blur_threshold: float) -> bool:
    # Reject blur: Laplacian variance below threshold.
    # Reject exposure blowouts: mean outside [20, 235] or std < 10.
```

- Hard reject only — no soft "sharpness score" component, no weight
  renormalization (deliberately rejected as over-built).
- Exposure bounds are module constants, not params, until someone needs to
  tune them.
- A standalone `compute_blur_score(gray) -> float` is exposed for direct use
  (webapp preview).

## Sampling methods

- **`uniform`** — stride from `fps` param, plus **sharpest-in-window**: within
  each stride window, pick the frame with max Laplacian variance instead of
  blindly taking the stride-aligned frame. Avoids coverage holes when the
  gate rejects a frame. (~15 lines, reuses gate scores.)
- **`optical_flow`** — existing `OpticalFlowFrameSelector` behavior: LK sparse
  flow disparity + RANSAC-affine rotation (motion) and histogram similarity
  (coverage), weighted score ≥ 0.5 selects; gate applied first.

## Selector simplifications (all verified unused in production)

| Change | Lines |
|---|---|
| Delete `reset()` (fresh instance per video; update the one test) | −10 |
| Delete `_estimate_rotation` `method` param + homography branch | −10 |
| Merge `compute_frame_score`/`should_select_frame` → `score_frame(frame) -> (score, components)`; threshold at call site | −20 |
| `max_features`, `rotation_threshold`, LK/feature param dicts → module constants; ctor keeps `min_disparity`, weights | −15 |
| Delete `stride` param from `score_all_frames` | −5 |
| Delete unused `native_fps` param from ffmpeg fps decoder | −2 |

Class itself stays: streaming keyframe state (`last_keyframe_*`) and
accumulated stats are legitimately instance state; the public API is already
functional.

## Deduplication refactors

- `_iter_scored_frames(video_path, selector, max_frames, on_progress)` —
  single generator yielding `(idx, frame, selected, score, components)`;
  `score_all_frames` and the optical-flow method become thin consumers.
- Index-extraction loop shared by `load_video_frames` /
  `extract_video_frames` (ffmpeg select filter, above).
- Single progress mechanism: internal hook with tqdm as the default
  `on_progress`; the parallel `verbose` tqdm plumbing goes away.
- `combine_scores(disparity, hist_similarity, weights)` pure function used by
  both the selector and `plot_disparity_sensitivity` — kills the hardcoded
  `0.6/0.4` formula duplicate in the plot.

## Migration (hard cut, no shim)

Full downstream surface (verified by grep 2026-07-10):

- Delete `collab_splats/utils/frame_sampling.py` and
  `collab_splats/semantics/frame_sampling.py` re-export (no importers of the
  latter).
- `collab_splats/utils/__init__.py` — remove `OpticalFlowFrameSelector`,
  `sample_frames_fps`, `sample_frames_optical_flow` re-exports.
- Update call sites to `collab_splats.preproc`:
  `webapp/routers/preprocess.py`, `dashboard/app.py`,
  `dashboard/pipeline.py`, `wrapper/reconstructor.py`,
  `wrapper/splatter.py`, `evals/datasets.py` — webapp/dashboard/wrapper
  method branching replaced by `sample_frames(method=...)`;
  `evals/datasets.py` keeps `extract_video_frames` + uniform sampling via the
  new API.
- Tests: `tests/utils/test_frame_sampling.py` → `tests/preproc/`
  (flat functions), updated for API changes; new tests below.
  `tests/dashboard/test_pipeline.py` mocks/patch targets updated to
  `collab_splats.preproc`. `tests/test_cu121_migration.py:133` module-path
  string list updated from `collab_splats.utils.frame_sampling` to
  `collab_splats.preproc.sampling`.
- Tutorial notebook `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`
  updated to new import path and `sample_frames` API.

## Testing

- Existing coverage relocates with import/API updates.
- New: `compute_blur_score` sharp vs Gaussian-blurred synthetic frames;
  `_frame_ok` rejects blurred / near-black / near-white frames;
  `sample_frames(method="uniform")` picks the sharpest frame in a window;
  dispatcher raises on unknown method; ffmpeg-missing path raises
  `RuntimeError` (monkeypatched `shutil.which`).

## External repo analysis (jashshah999/vggt-factor-refinement)

Adopted (ideas only — implementations are crude/magic-constant-laden):
blur filtering via Laplacian variance (`keyframe_selection.py`) → quality
gate above.

Deferred to separate future specs: multi-view depth-consistency filtering
(`depth_fusion.py`, belongs in feedforward-mesh work), dynamic-object
masking (`dynamic_mask.py`, concept sound, impl crude), adaptive chunking
vs fixed `--submap_size` (benchmark first).

Rejected: factor-graph/iSAM2/point-BA/photometric-BA stack (duplicates LM BA
+ Sim3 LC, heavy gtsam dep), covisibility-graph LC candidates (retrieval LC
already calibrated per model), trajectory smoothing (cosmetic, masks real
error), uncertainty estimation (hardcoded pseudo-calibration), VGGT+MASt3R
ensemble (scope), loaders/exporters (have equivalents).

## Out of scope

- Dynamic-object masking, depth-consistency filtering, adaptive chunking
  (future specs).
- Embedding-based (DINO/CLIP) frame diversity — histogram coverage
  approximates it at near-zero cost.
- Any change to reconstruction pipeline behavior beyond frame selection
  inputs.

## Size estimate

869 lines / 1 file → ~550 lines / 2 files + `__init__`, with quality gate,
sharpest-in-window, and unified dispatcher included.
