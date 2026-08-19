# BA track-quality parity — design

**Date:** 2026-08-19
**Status:** approved (brainstormed in-session)
**Follows:** `2026-08-18-ba-pipeline-wiring-design.md` (§Validation measured BA hurting sub-cm baselines)

## Problem

The 2026-08-19 chess/seq-01 sweep showed BA *worsens* ATE on every sub-cm baseline
(vggtx/omega/loger +28–72%) and helps only the weakest (mapanything −10%). A line-by-line
audit against upstream `zitongzhan/vggt demo_colmap.py` (the reference bae usage) found the
LM optimizer core matches exactly — the divergence is entirely in **track quality and
observation filtering**. BA faithfully converges onto noisier observations than upstream
ever feeds it.

## Audit findings (ranked)

1. **No visibility threshold.** `_optimize` does `vis = vis_scores.astype(bool)` — any
   score > 0 passes. Upstream gates `pred_vis_scores > 0.2` before BA. Occluded/drifted
   tracks with reproj error < 4 px pull poses systematically.
2. **`fine_tracking` forced off.** `_extract_tracks_vggsfm` hardcodes `fine_tracking=False`.
   Upstream (and the installed `predict_tracks` default) always runs the fine refinement
   stage. Coarse-only tracks carry ~1–2 px error vs ~0.3–0.5 px fine; at f≈300 px, Z≈2 m,
   1 px correlated error ≈ 6–7 mm pose error — the size of the measured regression.
3. **Track density.** Ours 2048 query pts / 5 query frames; upstream demo 4096 / 8.
4. **Per-frame focal is model noise, not signal.** Measured fx spread across frames of one
   video: vggtx 1.2%, omega **11%**, loger 0%. All scenes are one physical camera.
   `shared_camera=False` (current default) lets BA absorb per-frame pose error into
   per-frame focal. `shared_camera=True` is physically correct and upstream-supported.
5. **Filter-order gap.** We check the ≥2-observation landmark rule *before* dropping
   under-inlier frames; upstream drops frames first, then counts. Our order can retain
   single-observation landmarks (depth-unconstrained along the ray).

Out of scope (deliberate): 1024-square track extraction from original frames (chess source
is 640×480 — upstream would *upsample*; revisit for high-res scenes), focal freezing
(departs from upstream; confounds the parity test), windowed BA (`images` gap, known).

## Design

All changes in `collab_splats/geometry/bundle_adjustment.py`. Config surface stays
`BundleAdjustmentConfig` dataclass defaults — pipeline (`reconstructor.py:927`) and eval
(`eval.py` `ba` condition) both construct it bare, so defaults flow everywhere. No YAML
changes; `pointcloud.bundle_adjustment` remains one boolean.

### Config changes (`BundleAdjustmentConfig`)

| field | old | new |
|---|---|---|
| `vis_thresh` | — (implicit ~0) | `0.2` (new field) |
| `fine_tracking` | — (hardcoded False) | `True` (new field) |
| `max_query_pts` | 2048 | 4096 |
| `query_frame_num` | 5 | 8 |
| `shared_camera` | False | True |

### Code changes

- `_optimize`: `vis = vis_scores > cfg.vis_thresh` replaces `astype(bool)`. `vis_thresh=0.0`
  reproduces old behaviour for any score > 0.
- `_load_or_extract_tracks` threads `cfg.fine_tracking` into `_extract_tracks_vggsfm`.
- Filter order in `_optimize` reordered to upstream: (a) reproj + vis mask →
  (b) drop frames under `min_inliers_per_frame` → (c) drop landmarks with <2 surviving
  observations or out of range. Single-observation landmarks after (b) become impossible.
- **Track cache key** (`_compute_tracks_cache_key`) gains `fine_tracking`. Without it, a
  stale coarse-track cache would silently defeat the fix. `vis_thresh` and `shared_camera`
  are post-extraction knobs — deliberately *not* in the key, so ablations reuse the cache.

### Validation (chess/seq-01, 100 frames, single-pass — same protocol as the 2026-08-19 sweep)

Runs, serial in tmux:

1. `vggtx` (worst regression) — baseline vs ba, full new defaults.
2. `mapanything` (the one BA helped) — regression guard in the other direction.

Conditional third run — built only if the gate below demands attribution (YAGNI: no
committed ablation surface unless the combined result is ambiguous): `vggtx` with
`shared_camera=False` via an eval condition `ba_percam` added at that point, isolating
the shared-camera contribution.

**Correction (measured 2026-08-19):** this run is *not* cheap, contrary to the original claim
that it would reuse the track cache. `tracks_cache_dir` defaults to `None` (= always extract) and
`eval.py` constructs `BundleAdjustmentConfig()` bare, so **the eval path never uses the track
cache at all** — every eval BA condition re-extracts. Excluding `shared_camera` from the cache key
is still correct design (it is a post-extraction knob), but it buys nothing here. Budget a full
run, ~14 min, not ~3.

Decision gate:

- vggtx ba ATE ≤ baseline → land; update spec §Validation table + CLAUDE.md verdict.
- Improved but still > baseline → follow-on: 1024/original-res track extraction (option B).
- No improvement → follow-on: focal freeze / pose-only BA (option C).
- mapanything must not regress below its measured BA gain (ATE 0.0120); if it does, ablate
  which new knob costs it.

### Measured results (chess/seq-01, 100 frames, 2026-08-19)

Baselines reproduced the 2026-08-19 sweep exactly on both backbones — a valid control.

**vggtx** (worst prior regression):

| metric | baseline | ba (parity) | ba (old code) |
|---|---|---|---|
| ATE rmse | 0.008185 | 0.008836 | 0.0141 |
| ATE median | 0.007134 | 0.007844 | — |
| ATE max | 0.018124 | 0.019880 | — |
| RPE-t | 0.008521 | 0.008548 | 0.0087 |
| RPE-rot | 0.1858° | 0.2012° | 0.198° |
| AUC@5/15/30 | 20.37 / 64.48 / 80.54 | 8.85 / 58.73 / 77.67 | 13.1 / — / — |
| time | 133.3 s | 836.6 s | 129 s |

**mapanything** (the one BA already helped):

| metric | baseline | ba (parity) | ba (old code) |
|---|---|---|---|
| ATE rmse | 0.013267 | 0.008926 | 0.0120 |
| ATE median | 0.010740 | 0.007255 | — |
| ATE max | 0.044837 | 0.017234 | — |
| RPE-t | 0.018773 | 0.013850 | 0.0141 |
| RPE-rot | 0.2130° | 0.2201° | 0.201° |
| AUC@5/15/30 | 8.33 / 54.22 / 73.83 | 11.53 / 57.06 / 76.02 | 19.2 / — / — |
| time | 162.6 s | 637.4 s | 189 s |

Gate outcome:

- **mapanything passes outright.** ATE −32.7% vs baseline (was −9.8% under the old code) and below
  the old BA result of 0.0120. ATE max more than halves (44.8 mm → 17.2 mm) — the parity filters
  are killing worst-case frames, exactly what the visibility gate was meant to do.
- **vggtx improved but stays marginally above baseline** (+7.96%, was +72%). Per the gate this
  triggers the shared-camera attribution run (`ba_percam`), then option B (1024/original-res track
  extraction).
- Net: the parity fix removed ~89% of the vggtx regression and turned mapanything's modest gain
  into a large one. BA is no longer harmful on a sub-cm baseline — but it is not free, at 5–6×
  runtime.

Two observations the design did not predict:

1. **Rotation regresses slightly on both backbones while translation improves** (vggtx RPE-rot
   0.186°→0.201°; mapanything 0.213°→0.220° even as its ATE drops a third). `shared_camera=True`
   removes per-frame focal as a free parameter, and the pose solve appears to absorb some of that
   in rotation. This is what the `ba_percam` ablation isolates.
2. **AUC@5 is a poor instrument on this sequence.** Consecutive chess/seq-01 frames sit ~10–20 mm
   apart, so `auc_at_threshold`'s translation term — a *bearing* angle between normalized relative
   translations — is ill-conditioned: a sub-millimetre perpendicular shift on a 15 mm baseline
   swings the bearing past the 1° bins. Hence AUC@5 falling while AUC@30 barely moves and ATE
   improves. Prefer ATE/RPE on short-baseline indoor sequences.

Track extraction is effectively deterministic: `predict_tracks` selects query points by ALIKED
top-K (`max_num_keypoints=max_query_pts`) *before* the unseeded `torch.randperm` at
`track_predict.py:173`, which shuffles order (hence batch grouping) only. Run-to-run variation is
float-level, so these single-run numbers are not sampling noise.

**Upstream parity note (verified 2026-08-19):** BA runs *once* — a single `LM` +
`StopOnPlateau(steps=40, patience=3, decreasing=1e-3)` optimization — matching upstream
`demo_colmap.py` exactly, including `reject=10`, `TrustRegion(up=2.0, down=0.5**4)` and `PCG()`.
Upstream carries its own `# TODO: add iterative BA`: observations are filtered once, up front,
against the *initial* poses, so tracks that only become outliers after the poses move are never
dropped. Iterative BA (BA → refilter → BA) is a follow-on option, ranked alongside option B.

### Testing

- `tests/geometry/test_bundle_adjustment.py` (25 green) adapts: new-field defaults, vis
  threshold actually filters (score 0.1 observation dropped at 0.2, kept at 0.0),
  filter-order property (no landmark with <2 observations survives after frame drop),
  cache-key invalidation on `fine_tracking` change, cache-hit on `vis_thresh` change.
- No new test files; extend the existing module.

## Implementation principles

Reuse existing seams — every change lands inside functions that already exist; no new
modules, no wrappers. Delete nothing except the hardcoded `fine_tracking=False` and the
`astype(bool)` line. Eval protocol and scripts unchanged.
