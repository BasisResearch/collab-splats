# fps frame sampling — design

**Date:** 2026-08-13
**Status:** approved, not implemented
**Touches:** `collab_splats/preproc/sampling.py`, `collab_splats/wrapper/reconstructor.py`, `collab_splats/dashboard/{app,config,pipeline}.py`, `configs/{base,loop_closure}.yaml`, `configs/README.md`

## Problem

`preprocessing` has no way to say "sample a frame every T seconds." The knobs it does have —
`frame_proportion`, `min_frames`, `max_frames` — control **how many** frames, never **how far apart**.

Count is the wrong contract for reconstruction. `frame_proportion: 0.1` on a 1-minute video and on a
20-minute video both land in `[min_frames, max_frames]`, so both yield a similar count at wildly
different baselines between consecutive frames — ~2.5 fps vs ~0.25 fps. Baseline is what feedforward
registration and loop-closure submap hinges actually depend on; the fraction of a video is not a
physical quantity.

Second defect, invisible today: `max_frames` does two jobs at once. For `optical_flow` it is a pure
ceiling (`reconstructor.py:107`). For `uniform` it is the *target count* being spread
(`reconstructor.py:113-119`). One key, two meanings — which is why there is nowhere for a density
knob to live.

## What already exists

`sample_frames` already accepts `fps` and `_sample_uniform` already branches on it
(`sampling.py:471-480`). It is unreachable: no config key, `Reconstructor._extract_frames` never
passes it, the dashboard never passes it. `configs/README.md:224` already documents
`frame_selection: fps` — a stale claim this design makes true.

Two latent bugs in that unreachable branch:

- `sampling.py:482` — `targets = targets[:max_frames]` **truncates the tail**. A 10-minute video at
  2 fps with `max_frames: 300` covers only the first 150 seconds; the rest is never seen.
- `sampling.py:479` — a hidden `2.0` fps literal fires when neither `fps` nor `max_frames` is set,
  contradicting "`base.yaml` is the SINGLE SOURCE OF DEFAULTS."

## Design

### One density knob per method; `max_frames` is the frame budget

| `frame_selection` | density knob | `min_frames` | `max_frames` |
|---|---|---|---|
| `fps` | `fps` (samples/second) | floor | ceiling |
| `uniform` | — | inert | **the target count** |
| `optical_flow` | `min_disparity` | inert | ceiling |

`max_frames` reads consistently as "the most frames I will take." `uniform` means "spend the budget,
spread it evenly," so its target and the ceiling coincide by definition — there is no case for
"uniform, 100 frames, ceiling 300" that is not better written as `max_frames: 100`.

`min_frames` is meaningful only for `fps`, where the count floats with video length. `uniform` is
pinned at the ceiling, and `optical_flow` cannot be forced to yield frames its selector rejected.

`frame_proportion` is **deleted**. With `fps` available it controls nothing physical, and once
`min_frames` defaults to `null` it decides the count for no video at all.

### Sampler structure

`_sample_uniform` and `_sample_optical_flow` are already siblings in `sampling.py`. `_sample_fps`
joins them as a third — the smallest structural change available, not the largest.

```
sample_frames(method="fps" | "uniform" | "optical_flow", ...)
  ├─ _sample_fps(fps, min_frames, max_frames, ...)  → _fps_targets     ─┬→ _sample_positions
  ├─ _sample_uniform(max_frames, ...)               → _uniform_targets ─┘
  └─ _sample_optical_flow(...)                      → streaming, untouched
```

`fps` and `uniform` are genuinely different contracts, not two spellings of one:

- **`uniform`** — "N frames spanning the video." N is the contract, spacing falls out.
  Endpoint-anchored: `np.unique(np.linspace(0, total-1, n).round())`.
- **`fps`** — "a frame every T seconds." Spacing is the contract, N falls out. Stride-anchored:
  `range(0, total, round(native_fps / fps))`, deliberately not stretched to hit the last frame.

They coincide only while the band does not bind. When it binds the contracts conflict, and which one
gives way matters — so they stay separate functions rather than one function with a mode flag.

`_sample_positions` is today's `_sample_uniform` **body with the target block lifted out**: validation
window radius (`_VALID_PROBE_MAX`), the single ffmpeg `select=` pass, the quality gate, and the
sharpest-usable-neighbour substitution. Unmoved and still covered by the existing tests.

### Performance — unchanged

Uniform sampling was moved to a single ffmpeg `select=`-filter pass in 14fdc42 (~178s → 20s in
tests). That path is `_iter_selected_frames` (`sampling.py:408`) and **nothing here touches it**. Both
target-list methods build a list of source indices and hand it to the same single pass — same decode
cost, same one subprocess. `optical_flow` remains the expensive method (it must decode every frame to
score it).

ffmpeg still demuxes the whole file in C either way; `select` drops frames after decode. Wall-clock
scales with video length, not sample count — true today, true after.

### Band behaviour for `fps`

- **Over ceiling** — decimate the fps target list evenly (via `_uniform_targets`) so it still spans
  the video. **Not** truncation. Truncation hands the reconstructor a video that stops halfway, which
  fails worse than reduced density; reduced density is just `uniform`, which already works.
- **Under floor** — re-spread at `min_frames` via `_uniform_targets`.
- Either way, `logger.warning` the **effective fps** so a bound band is never silent.

### Validation

Wrong knob for the method raises `ValueError`, both directions — the combination this design exists to
eliminate:

- `method="uniform"` with `fps` set → raise
- `method="fps"` with `fps=None` → raise (this is what removes the hidden `2.0` at `sampling.py:479`)
- `method="uniform"` with `max_frames=None` → raise (uniform has no count without it)

Validation fires **before** the empty-video path: a missing knob is a caller bug regardless of whether
the file decodes.

### Config

```yaml
# configs/base.yaml
preprocessing:
  frame_selection: fps   # fps | uniform | optical_flow
  fps: 1.0               # fps method — samples/second
  min_frames: null       # fps floor; null = fps is honoured literally
  max_frames: 300        # frame budget (vggt_omega OOMs above ~300 on 44 GB GPU)
```

`fps` is the only new key. `frame_selection` gains a value and flips its default;
`frame_proportion` is deleted; `min_frames` becomes fps-only and defaults to `null`.

`min_frames: null` is deliberate: it makes the default mean literally 1 fps. A floor would push short
videos to a higher effective fps, turning `fps: 1.0` into a hint rather than a setting.

`max_frames: 300` stays — it is an empirically measured hardware limit, not a stylistic default.
`null` remains available per-run.

```yaml
# configs/loop_closure.yaml
preprocessing:
  frame_selection: fps
  fps: 4.0
  min_frames: 300        # the floor LC measurably needs
  max_frames: 600
```

LC chains submaps through a shared overlap frame, so it needs steady frame-to-frame overlap — which
`fps` provides more honestly than a count-driven fraction. It keeps an explicit `min_frames: 300`
because its existing comment ("RAISE to actually feed more frames — otherwise submaps just re-chunk
the same 200 keyframes") is a floor requirement, and `fps` alone has no floor by construction.

**Owed:** the LC calibrations (`submap_size: 64`, `submap_overlap: 1`,
`lc_retrieval_threshold: 0.95`, per-backbone target layers) and the 4-backbone ATE numbers in
`project_lc_vggtslam_alignment` were all measured under `uniform` spacing. The *mechanism* is
unaffected, but the numbers should be re-verified on `fps` spacing before they are quoted again.

### Reconstructor

`_extract_frames` (`reconstructor.py:65-140`) gains `fps`, drops `frame_proportion`, and drops its own
`max(min_frames, ...)` derivation — the band moved into `_sample_fps`. Three-way branch on
`frame_selection`. The `frames.zarr` provenance dict gains `fps` so a processed scene records how it
was sampled.

### Dashboard

- `_SAMPLERS = ["fps", "uniform", "optical_flow"]` (`app.py:53`)
- new `fps` `FloatInput`, visibility bound on `== "fps"` — mirroring the existing `min_disparity`
  pattern at `app.py:243`
- `RunConfig.fps` (`config.py:20`); `_sample` and `_write_frames_zarr` thread it
  (`pipeline.py:346-355`, `pipeline.py:55-63`)
- saved settings read `s.get("fps", 1.0)`

`python -m collab_splats.dashboard --smoke` must print `SMOKE PASS` before any dashboard commit.

## Testing

Behaviour-defining:

- **Ceiling decimates, does not truncate** — assert the last `frame_idx` lands near end-of-video.
  This is the test that would have caught `targets[:max_frames]`.
- Floor re-spreads at `min_frames`.
- Both `ValueError` directions, plus `uniform` with no `max_frames`.
- `_fps_targets` / `_uniform_targets` as units.
- Reconstructor three-way branch.

Migration: 9 call sites in `tests/preproc/test_sampling.py` (lines 213, 221, 227, 233, 244, 263, 270,
353) are mechanical renames. `test_sampling.py:353` (`method="uniform"` with no count, expecting
`([], [])`) needs `max_frames=` added, since validation now fires before the empty-video path.
`tests/wrapper/test_reconstructor.py` and `test_reconstructor_preprocess.py` need their
`frame_proportion` assertions dropped.

## Out of scope

- **`splatter.py`** — `splatter.py:47` types `frame_selection: Literal["fps", "optical_flow"]`, but
  its code branches only on `optical_flow` (`splatter.py:295`), and its `"fps"` means "let
  `ns-process-data` decode," *not* this sampler. It also carries its own `frame_proportion`. Both
  names now collide with ours. The nerfstudio path is untouched; the collision is recorded here, not
  fixed.
- **Tutorial notebooks** — `03_splats/derive_splats.ipynb` and `06_mesh/create_mesh.ipynb` pass
  `frame_proportion=0.25` to **`Splatter`**, a different code path that never reaches `sample_frames`.
  Unaffected. Only `docs/source/tutorials/03_splats/configs/base.yaml` (a `Reconstructor` config)
  loses its `frame_proportion` line.
- Broader `preproc` refactor. `sampling.py` is 697 lines but cohesive, with the section dividers and
  docstrings the style guide asks for. The conflation described above is its one real defect.

## Implementation principles

- **Reuse.** `_sample_positions` is the existing `_sample_uniform` body, moved not rewritten. The
  single-pass decode, window radius, quality gate, and neighbour substitution are untouched.
- **Retire.** `frame_proportion` is deleted, not deprecated. The hidden `2.0` fps literal
  (`sampling.py:479`) is deleted. The `targets[:max_frames]` truncation is deleted. No compatibility
  shims — this repo takes hard breaks.
- **Minimal.** One new config key (`fps`). One new sampler function following the shape already in the
  file. No new abstraction layer, no band helper beyond what `_sample_fps` needs inline.
