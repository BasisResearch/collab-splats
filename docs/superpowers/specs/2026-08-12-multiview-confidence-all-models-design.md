# Multiview confidence for all feedforward backbones — implementation handoff

**Date:** 2026-08-12
**Status:** handoff — ready to implement
**Owner:** implementing agent (fresh session)
**Out of scope:** the gsplat trainer port (separate spec, in discussion), continuous-weight
loss supervision (noted as optional extension only).

## Goal

MapAnything already filters its point cloud with geometric cross-view depth
consistency (`use_multiview_confidence=True` by default). Bring the other three
backbones — VGGT-X, VGGT-Omega, VGGT-Spark — to the same standard: calibrated,
default-on multiview confidence, with the computed confidence map persisted to
`feedforward.zarr` so downstream consumers (future splat-trainer depth
supervision) can use it.

## Current state (verified 2026-08-12)

All code paths already exist; this is calibration + wiring + persistence, not new algorithms.

- **Shared implementation:** `compute_multiview_depth_confidence(depth, intrinsics,
  extrinsics, depth_masks=None, abs_thresh=0.0, rel_thresh=0.05, device="cuda")`
  at `collab_splats/pointcloud/feedforward/base.py:378`. Projects each source
  pixel into all other frames; inlier when reprojected and sampled depth agree
  within `abs_thresh + rel_thresh * depth`. Returns per-pixel **inlier ratio in
  [0, 1]** of overlapping views, shape (N, H, W). O(N²) frame pairs, GPU loop.
- **MapAnything** (`mapanything.py`): default **on**. `mv_conf_abs_thresh=0.02`
  (metres — valid because MapAnything depth is metric; calibrated),
  `mv_conf_threshold=0.0` (keep any pixel with ≥1 agreeing view). Call site
  ~line 448: mask combined with the learned-confidence mask. The long docstring
  (lines 115–146) explains why `confidence_percentile` is bypassed when mv is on
  — percentile thresholds collapse on the quantized k/N ratio. Read it before
  changing threshold semantics.
- **VGGT-X** (`vggtx.py`): field `use_multiview_confidence: bool = False` (line
  194), `mv_conf_threshold: float = 0.0`. Call site ~line 334: `abs_thresh=0.0,
  rel_thresh=0.05` hardcoded; `mv_mask = mv_conf > self.mv_conf_threshold`
  passed as `extra_mask` to `unproject_and_filter_points`.
- **VGGT-Omega** (`vggt_omega.py`): same pattern — flag at line 151, call site
  ~line 245. **Caution:** omega passes model-res `intrinsic` to the mv call but
  `raw_outputs["intrinsics_downsampled"]` to `unproject_and_filter_points`. The
  mv confidence grid and the unprojection grid must stay pixel-aligned — verify
  the depth array resolution matches the intrinsics used in each call before and
  after your changes (omega's zarr stores model-res intrinsics; see memory/CLAUDE.md
  on the model-res vs original-res intrinsics trap that caused the mesh regression).
- **VGGT-Spark** (`vggt_spark_creator.py`): `VGGTSPARKCreator` is a dataclass
  subclass of `VGGTXCreator`, so it inherits the flag and the call path. Nothing
  to wire unless an override bypasses the VGGT-X section — verify, don't assume.
- **Tests:** `tests/pointcloud/test_mv_conf.py` exists, plus per-creator tests in
  `tests/pointcloud/`.
- **Config:** `configs/base.yaml` currently exposes **no** mv keys. Check how the
  `pointcloud` section forwards creator kwargs in `wrapper/reconstructor.py` and
  follow that pattern.

## Tasks

### T1 — Calibrate rel_thresh and mv_conf_threshold for the three VGGT backbones

`abs_thresh` stays 0.0 for all three (non-metric depth — a fixed metre tolerance is
meaningless; this is why MapAnything's 0.02 m does not transfer). Sweep
`rel_thresh` (suggest 0.02–0.10) × `mv_conf_threshold` (suggest 0.0–0.5) per
backbone.

Calibration method (mirrors the LC layer-calibration discipline: sweep with
positives AND clean negatives, not vibes):

- **Quantitative:** 7-Scenes has GT depth and the loader exists
  (`evals/datasets.py`). Metric: depth error of *retained* pixels vs GT +
  retention fraction, compared against the current learned-confidence percentile
  baseline. A win = lower retained-pixel error at comparable retention.
- **Qualitative gate:** run one local scene (`data/outputs/`) end-to-end and
  compare sparse cloud + mesh vertex counts before/after. The mesh baseline
  post-intrinsics-fix is 711,079 verts at `depth_trunc: 2.0` — large deviations
  need explaining.
- Run sweeps via `evals/scripts/eval.py` conventions: CLI/tmux only, never
  notebooks; results under `evals/results/` (gitignored). Record the chosen
  values and the sweep table in the completion notes.

### T2 — Verify Spark inheritance

Confirm `VGGTSPARKCreator` actually executes the mv path (no override skips it).
Trap documented in the file itself: a previously import-cached VGGT-X `vggt`
package silently shadows the SPARK tree — run Spark checks in a **fresh process**
(`_assert_loaded_from_spark` guards this).

### T3 — Persist mv_conf to feedforward.zarr

Store the computed (N, H, W) mv confidence beside the existing learned
`confidence` array (e.g. dataset name `mv_confidence`), chunked by frame like
its neighbours (`base.py` `to_zarr` section, ~line 121+). Keep the existing
transient-mask filtering behaviour unchanged — persistence is additive.
Zarr is v3 (3.1.5): `compressors=[BloscCodec(...)]`, not `codecs=`.
Only write when mv was computed; no backfill of old scenes.

### T4 — Expose config knobs

Add `use_multiview_confidence`, `rel_thresh` (promote the hardcoded 0.05 to a
field like MapAnything's `mv_conf_abs_thresh`), and `mv_conf_threshold` to the
creators' config surface, wired through however `configs/base.yaml`'s
`pointcloud` section reaches creator kwargs today. Follow the existing pattern
exactly; no new config layers.

### T5 — Flip defaults

Only after T1 shows a win per backbone: set `use_multiview_confidence=True` with
the calibrated values as field defaults, mirroring MapAnything. If a backbone
shows no win, leave it off and record why.

### T6 — Tests

Extend `tests/pointcloud/test_mv_conf.py` and creator tests: flag on/off paths,
mask-resolution alignment (omega), zarr round-trip of `mv_confidence`, spark
inheritance. Flat test functions, no classes. Any test touching semantics write
paths must respect the artifact-naming contract (see CLAUDE.md).

### Performance note (measure, don't pre-fix)

The mv computation is O(N²) view pairs. Fine at tutorial scale; measure at
~200–1000 frames before enabling by default for long sequences. If it's a
blocker, a frame-window limit is the obvious cut — but that is a scope decision
to surface, not silently implement.

## Acceptance criteria

1. All four backbones share one mv code path (`base.py`) with per-backbone
   calibrated defaults; sweep table recorded.
2. Spark verified to execute the path in a fresh process.
3. `mv_confidence` persisted in `feedforward.zarr` when computed; existing
   filtering unchanged.
4. Config knobs exposed per existing pattern.
5. Test suite green: `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly`
   (pytest-randomly is installed; disable it). Known failures listed in
   `docs/known-test-failures.md` are exempt.

## Environment traps (inherited from project memory — real, all measured)

- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Base-shell `python` may be 3.13.
- Never run repo-wide `black .` — venv black 26.5.1 is newer than repo formatting. Format only files you touched.
- Concurrent sessions may edit `configs/base.yaml` mid-run; if the suite suddenly shows ~60 unrelated failures, re-run before believing it.
- Commit with explicit pathspec (parallel sessions share the index). `docs/superpowers/` needs `git add -f`.
- Heavy eval runs: tmux, no parallel heavy side-shells (cgroup cap 46.6 GB).
- Conventional commits, e.g. `feat(pointcloud): calibrate multiview confidence for VGGT backbones`.
