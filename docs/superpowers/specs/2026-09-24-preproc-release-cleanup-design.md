# Preproc release cleanup — design

Date: 2026-09-24 · Branch: `clean/preproc-release` off `clean/final` · Status: approved in brainstorm

## Goal

Make `collab_splats/preproc/` release-ready: brief docs, no over-engineering, no buried
tunables, loud failures. A naive user reads a docstring and knows how to call the function.

## Decisions (from brainstorm)

- **API may break.** Nothing external depends on it yet; configs, callers, tests and the
  tutorial 01 notebook are updated in the same commits.
- **Tunables are function inputs, not module constants.** A number someone could tune
  becomes a kwarg with a default on the function that uses it, threaded up to the public
  entry point where a caller needs it. Fixed facts stay module-level: lookup tables and
  format identifiers (`IMAGE_EXTS`, `_ROTATE_CODES`, `SCHEMA_VERSION`, `_MANIFEST_NAME`).
- **Silent fallbacks become errors.** No `fps or 30.0`, no zeros-on-failure probe, no
  `{"available": False}` report.
- **Measurement lore leaves the code.** It already lives in `docs/superpowers/specs/*`
  and git history; a non-obvious *why* keeps 1-3 lines, plus a spec path when one exists.
- **Scope:** `__init__`, `video`, `qa`, `sampling`, `frames`, `undistort`, `viz`, the
  `configs/base.yaml` preproc block, `scripts/migrate_frames_zarr.py`, and every caller
  or test broken by the API changes. No readability rewrite of `tests/preproc/`.
- **Isolation:** worktree `.worktrees/preproc-release`; every gate runs as
  `cd <wt> && PYTHONPATH=<wt> python ...` and prints `collab_splats.__file__`.

## Round 1 — prose only

No code change. Proof: AST equal after deleting every docstring statement on both sides
and stripping comments, plus one sanity mutation showing the check can fail.

### Rules

- Module docstring: summary line + 2-4 bullets on what it holds. No architecture history.
- Function docstring: summary line, bullets only for caller-facing contract (units,
  color order, raises), then `Args:` / `Returns:`. No measurements, speedups, dataset
  names (GH010229, tutorial), or "replaces the old X".
- Comments say what the block does; a *why* stays only where the code would otherwise
  be "fixed" wrongly (thread pin, float64 histogram, nan-not-0, stride `>= 1`).
- `configs/base.yaml` preproc block: one comment line per key — meaning and direction.

### Per file

- `__init__.py` — docstring to 3 bullets.
- `video.py` — fix false "grid lives here" claim; drop ffmpeg-pipe / legacy
  `tags.rotate` / "seek this replaces" history in `_rotation_degrees`, `_frame_index`,
  `_upright`, `iter_frames`; unopenable-path note once, not twice.
- `qa.py` — `compute_pair_motion` docstring to contract (analysis-grid px, nan =
  unmeasurable, parallax in [0, 1], one crossCheck caveat line); speedup numbers out of
  `detect_orb`, `compute_exposure`, `compute_frame_quality`; HYPOTHESIS bullet and
  `workers` timings out of `compute_video_quality`; thread-pin and seek-check comments
  to 1-2 lines.
- `sampling.py` — drop "no cv2 equivalent", mutable-default lecture, GH010229 figures;
  fix the VDA claim on `context_indices`.
- `frames.py` — module docstring trimmed; PNG compression comment to one line.
- `undistort.py` — module docstring to 2 fact bullets; SIFT-threads comment to 1 line.
- `viz.py` — drop "five duplicates beat a helper" note, "ffmpeg seek" wording,
  collab-data DPI reference.
- `configs/base.yaml` — one line per key; fix stale `undistort` "alpha=0 crop" (the
  canvas grows, it does not crop).

## Round 2 — code, one commit per logical change

### sampling.py

- Delete `_split_quality` and `SLOT_POLICIES`; `on_empty_slot` becomes a
  `sample_fps(..., on_empty_slot="rescue")` kwarg, validated there. Config moves the key
  out of `preproc.quality` to `preproc.on_empty_slot`; reconstructor passes it.
- `_eligible` stays as a 3-line helper (mask -> flatnonzero -> raise if empty; 3 callers).
- Inline `context_indices` into `sample_fps`; delete it and its `info=` param.
- Delete the dead `pool.size == 0` branch in `sample_fps`.
- Extract `_spread(pool, n)` (shared by `sample_uniform` and the `sample_fps` re-spread)
  and `_sharpest(candidates, target, laplacian)` (normal slot and rescue both use it).
- `fps or 30.0` removed; a zero fps raises.
- `OpticalFlowFrameSelector` -> private `_OpticalFlowSelector`: `combine` folded into
  `score_frame`; `lk_params` / `feature_params` dicts replaced by explicit `__init__`
  kwargs (LK window, levels, max corners, min inliers, hist bins, motion weight);
  `sample_optical_flow` exposes `min_disparity`, `select_threshold`,
  `rotation_threshold_deg`.
- Fix double seeding: first frame is accepted once, and `analysis_gray` runs once per frame.
- `_estimate_rotation` catches `cv2.error`, not `Exception`.
- Drop the constant `"selected": True` record field.
- `_decode_selection` raises when a requested frame is not decoded, instead of skipping.

### qa.py

- `analysis_width`, `blur_h_size`, `n_features`, `ransac_thresh_px` thread from
  `compute_video_quality` through `_measure_photometry_and_motion` to the per-frame and
  per-pair functions; no hardcoded 480.
- `compute_video_quality` loses `output_path`; `load_video_quality` owns write and reuse.
- Unreadable or empty video raises instead of returning `{"available": False}`; the
  `available` key leaves the report (no reader outside `qa.py`).
- Extract `_ranges(total, workers, stride)`; build the pair columns in a loop.

### video.py

- `get_video_info` raises `FileNotFoundError` (missing path) or `ValueError`
  (unprobeable) instead of returning zeros; `iter_frames` raises on an unopenable path.
- Callers swept: `dashboard/localize.py:479`, `dashboard/app.py:384`,
  `wrapper/reconstructor.py:292`, plus samplers and tests that relied on zeros/empty.

### frames.py

- `write_frames(..., png_compression=1)` replaces `_PNG_COMPRESSION`.
- Delete the `frames.zarr` hint in `read_manifest` and `scripts/migrate_frames_zarr.py`
  (pre-release format; store is `images/` + `frames.json`).

### undistort.py

- `calibrate_camera(..., num_threads=8, min_registered_frac=0.6, min_images=8)` replaces
  the three module constants.

### viz.py

- `_save(fig, out_dir, name, dpi)` and `_mark_selected(axes, selected, fps)` replace the
  duplicated title/save/close and overlay blocks.
- `plot_selection(total_frames, selections: dict[str, list])` — fixes the fps set being
  labeled "Uniform" and covers all three samplers.
- `plot_frame_scores` stops reading the dropped `selected` field.
- `dpi=` kwarg per plotter replaces `_PNG_DPI` / `_THUMB_DPI`; `_HIST_BINS` and figure
  sizes become kwargs or inline values.

## Testing

- Gate per commit: `tests/preproc tests/wrapper tests/dashboard` in the worktree, with
  the `__file__` proof line and a SKIP count compared to control (third_party symlinks).
- Tests that exercise deleted code are deleted; each behavior change (raises, kwarg
  move, dropped field, `plot_selection` signature) gets a test.
- Tutorial 01 notebook re-executed once at the end.
- `tests/test_docstring_contract.py` keeps passing for `preproc`.

## Consistency across modules

Preproc is the first of several module cleanups. Consistency comes from three layers, each
holding the rules it can hold; a prompt alone drifts by the third module.

| Layer | Holds | Enforced by |
|---|---|---|
| Rules doc `docs/superpowers/decisions/017-release-cleanup-rules.md` | module-agnostic rules from this spec's Decisions + Round 1 Rules | every module spec links it, never restates it (016 is taken on another branch) |
| Project skill `.claude/skills/release-cleanup/SKILL.md` | the process: read all files -> findings + verdict table -> user approval -> spec -> worktree -> round 1 -> round 2 -> gate | same steps, same output shape per agent |
| `tests/test_docstring_contract.py` | the mechanical subset of the rules | failing test |

### Contract test additions

Run over every package in `PACKAGES`:

- no module-level UPPER_CASE name bound to a bare number; allowlist for fixed facts
  (`SCHEMA_VERSION`, lookup tables)
- banned words in docstrings and comments: `measured`, `GH010229`, `HYPOTHESIS`,
  `x faster`, `ffmpeg pipe`, `replaces the old`
- caps: at most 6 bullets per docstring, at most 4 lines per comment run
- no silent fallback idioms: `or <float literal>` on a probe value, bare
  `except Exception:` without re-raise

Each check is validated against a known-bad fixture before it is trusted. Preproc is the
first package to pass all of them. Existing entries (`semantics`, `pointcloud`) are listed
as known failures until their own cleanup, not silently exempted.

### Verdict table (not lintable)

Every module spec carries the table from Round 2 above: one row per function and
module-level name, verdict in {keep, merge, inline, delete, make-kwarg, make-private,
raise}, one-line why. The user approves the table before any code change.

### Order

1. Write `017-release-cleanup-rules.md` from this spec; replace this spec's Decisions and
   Round 1 Rules with a link to it.
2. Add the contract checks; run them on preproc before round 1 to measure the noise.
3. Write the skill; preproc round 2 is its first real run.

### Prompt for the next module

```
Run the release-cleanup skill on collab_splats/<module>/.
Rules: docs/superpowers/decisions/017-release-cleanup-rules.md — do not restate or reinterpret.
Reference: docs/superpowers/specs/2026-09-24-preproc-release-cleanup-design.md.
Stop for approval after (1) findings + verdict table and (2) the spec.
Then worktree .worktrees/<module>-release off clean/final; round 1 prose-only (docstring-stripped
AST equality + one sanity mutation), round 2 one commit per change, gate prints collab_splats.__file__.
Add <module> to PACKAGES in tests/test_docstring_contract.py; it must pass.
Do not touch other packages. Verify callers yourself; do not trust claims in this prompt.
```

## Out of scope

- Readability rewrite of `tests/preproc/`.
- Any other package.
