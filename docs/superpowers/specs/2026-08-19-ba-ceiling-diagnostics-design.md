# BA Ceiling Diagnostics — Design

**Date:** 2026-08-19
**Status:** measured — **ceiling confirmed**. Results:
[ba-ceiling-measured-report](2026-08-19-ba-ceiling-measured-report.md)
**Predecessors:** [ba-track-quality-parity](2026-08-19-ba-track-quality-parity-design.md), [ba-lm-convergence](2026-08-19-ba-lm-convergence-design.md)
**Handoff:** `docs/superpowers/handoffs/2026-08-19-ba-omega-no-improvement-handoff.md`

## Problem

On 7-Scenes chess/seq-01, 100 frames, `vggt_omega`: bundle adjustment cuts the objective
**49%** (converged loss `4.083486e+05`) while making pose accuracy **worse** — ATE
`0.007976 → 0.008481` (+6.3%), RPE-rot +5.3%, RPE-t −0.4%. Filtering is not the culprit:
100/100 frames and 37248/37890 points survive, 1,387,977 observations enter the solve.
The optimizer is doing exactly what it was asked and the answer gets worse, so the fault is
in the objective, not the solver.

Track-quality parity (2026-08-19) already moved `mapanything` from −9.8% to **−32.7%** ATE
on the same scene while `vggtx` went **+8.0%**. Whatever remains is specific to sequences
whose baseline poses are already sub-centimetre.

## Question

Only one: **is BA's objective minimised at the true poses?** Everything else is attribution
and can wait for the answer.

- If the truth scores *worse* than BA's solution, BA is converging correctly to the wrong
  answer. The objective is wrong and the optimizer is exonerated.
- If the truth scores *better*, BA failed to reach a reachable optimum and this becomes an
  optimizer problem.

## Verdict under test

**BA has hit a structure-conditioning ceiling on this sequence.** The tracks that carry
enough baseline to improve on the model's poses are the long ones, and the long ones are
the ones most likely to have drifted. The information you need and the information you can
trust are disjoint.

Supporting arithmetic, from measured geometry (see the census below — measured, not assumed).
**Units matter here: the measured Sim(3) scale to metres is 0.385156, so 1 reconstruction
unit = 2.596 m.** Everything below is stated in recon units and then converted.

- camera-centre extent **0.3075** (0.798 m), median landmark depth **1.31** (3.40 m)
- σ_Z ≈ Z²·σ_px / (f·B), with f ≈ 548 px, σ_px ≈ 0.5
  - best case, full extent B = 0.3075 → σ_Z ≈ 0.0051 units = **13 mm**
  - typical, median triangulation angle 3.98° → B ≈ 0.091 units → σ_Z ≈ 0.0172 = **45 mm**
- baseline ATE = **7.98 mm**

Structure uncertainty straddles the pose error BA is meant to remove. Under plain least
squares that structure noise is transferred into the poses.

## Method — one experiment

`evals/scripts/ba_start_at_gt.py`. Run the package's own `BundleAdjustment` **twice** on one
reconstruction, changing nothing but the starting poses:

- **A — model start.** Reproduces the shipping `ba` condition.
- **B — GT start.** Ground-truth poses mapped into the reconstruction frame by
  `umeyama_sim3` on camera centres. Reprojection is invariant to a global similarity, so
  this changes only the frame, never how well the tracks are explained.

Readouts per run: starting ATE, converged ATE, LM loss curve, mean camera-centre movement.

The falsifier is B's ATE trajectory. B starts at ATE ≈ 0 by construction. If it *rises*
while the loss falls, BA is walking away from the truth to satisfy its objective — ceiling
confirmed, and the distance it walks is the size of the ceiling. If B stays near 0 and its
converged loss sits well below A's, the truth is the better optimum and A merely failed to
find it — ceiling refuted, and this becomes an optimizer spec.

**No behaviour change to `collab_splats/geometry/bundle_adjustment.py`.** Both runs are
plain `ba.refine` on a `replace(result, extrinsics=...)`.

### Known asymmetry — closed by a second measurement

`pts3d_tracks` are seeded from `result.world_points` — the model's frame — and are shared
between both runs through the track cache. Run B therefore starts from GT poses with a
model-frame landmark initialisation, so its starting loss is "truth with mismatched
structure", not "loss at truth".

`evals/scripts/refit_at_fixed_poses.py` closes this: poses held fixed, every landmark
re-solved at each pose set independently, then the losses compared. **Measured: model poses
8.891822e+06 (2.1265 px RMS) vs GT poses 2.490652e+07 (3.5591 px RMS).** The truth stays 2.8×
worse with the structure fitted to it, so the verdict does not rest on the seeding.

Its own correctness gate: the loss it computes at the seeded structure is byte-identical to
the package's `_reproject_shared` (3.609295e+07 vs 3.609296e+07).

### Cache finding

`evals/scripts/eval.py:180 _prepare_image_dir` symlinks into `tempfile.mkdtemp()`, and
`_compute_tracks_cache_key` hashes `image_paths`. **A `--tracks_cache_dir` can therefore
only be hit within a single eval invocation — never across runs.** The existing
`evals/results/ba_convergence_chess/cache/vggt_omega/tracks.zarr` is unreachable from any
new process. This script uses a stable image dir so its two BA runs share one extraction.

## Supporting census (already measured)

Per-landmark max pairwise angle between viewing rays from the cameras that observe it
(vis > `vis_thresh`). Pure geometry — camera centres and existing landmark positions. No
triangulation is performed and no BA is run. `evals/scripts/tri_angle_census.py`.

**Measured 2026-08-19** — 37257 landmarks with ≥2 observations at vis>0.2, 1,966,262
observations, `pred_ba` camera centres:

percentiles (deg): p1 0.12, p5 0.23, p25 1.07, **p50 3.98**, p75 7.01, p95 10.23, p99 11.26

| gate | points dropped | obs dropped | points kept |
|------|----------------|-------------|-------------|
| 0.5° | 6761 (18.1%) | 153952 (7.8%) | 30496 |
| 1.0° | 9043 (24.3%) | 234181 (11.9%) | 28214 |
| **1.5°** (COLMAP filter) | **10684 (28.7%)** | **291384 (14.8%)** | **26573** |
| 2.0° | 11915 (32.0%) | 335602 (17.1%) | 25342 |
| 3.0° | 15035 (40.4%) | 460702 (23.4%) | 22222 |
| 6.0° (COLMAP local BA) | 25626 (68.8%) | 997632 (50.7%) | 11631 |
| 16.0° (COLMAP init) | 37257 (100%) | 1966262 (100%) | **0** |

(COLMAP thresholds verified on installed pycolmap 4.0.4: `filter_min_tri_angle=1.5`,
`ba_local_min_tri_angle=6.0`, `init_min_tri_angle=16.0`.)

Three readings:

1. **The whole population tops out near 11°** (p99 = 11.26). No landmark in the scene
   reaches COLMAP's 16° initialization bar — COLMAP would refuse to seed a reconstruction
   here at all. Its answer to this sequence is to decline it, not to solve it better.
2. **A 1.5° gate is asymmetric in our favour**: 28.7% of the points but only 14.8% of the
   observations. Degenerate landmarks are seen in fewer, closer frames, so removing them
   costs less residual mass than their count suggests.
3. **No frame starvation at any candidate gate.** Minimum surviving observations per frame
   is 10816 at 1.5° and 7142 at 6.0°, against `min_inliers_per_frame=64`. The gate cannot
   drop a frame, so it cannot change which poses BA solves for.

## Findings that shape the design

Established before this spec; recorded so the experiment does not re-litigate them.

- **Zero-pad hypothesis is dead.** Model res ≈ 672×504 padded to 672². Track `y_max =
  494.31 < 504`, so nothing lands in the pad band. `min ‖pts3d‖ = 0.773`, zero landmarks
  below 1e-3.
- **Gauge freedom is not the issue.** `ate_translation`
  (`collab_splats/geometry/loop_closure/eval.py:139`) aligns with Umeyama **Sim(3)**, which
  absorbs all 7 DoF. (`evals/metrics.py` defaults to `se3` but is a different call path.)
- **Solver and trust region are not divergences.** We run `CuDirectSparseSolver` where
  upstream runs `PCG()` — ours is stronger. bae's `schur.TrustRegion` is numerically
  equivalent to `pp.optim.strategy.TrustRegion.update`, which handles our sparse `J`.
- **Robust kernels are unreachable through bae's API.** `LM.step` computes the step from
  raw residuals and applies the kernel only in `self.loss(...)`, so `kernel=` changes the
  accept/reject test and nothing else; `weight=` is assigned and never used. Handoff items
  A and D are reachable only by folding √w into the residual inside `_BAModel.forward`.
- **Upstream runs BA once too.** `zitongzhan/vggt demo_colmap.py` carries its own
  `# TODO: add iterative BA`. Single-shot filtering is parity, not our deviation.
- **Tracks are long.** After the vis>0.2 gate: mean length 51.9/100 frames, median 45,
  41.4% of tracks in >50 frames, p95 = 100. Visibility is bimodal (p25 0.0, p50 0.293,
  p75 0.996), so the 0.2 gate cuts inside the low mode.
- **`ba/colmap/sparse/0` carries no observations.** `num_observations = 0`, so the written
  model cannot be used to measure BA's own reprojection residuals; the loss curve is the
  only record of them.
- **`_last_loss_history[0]` is the loss *after* LM step 1, not the initial loss.** Measured
  initial losses are 3.609296e+07 (model poses) and 5.349418e+08 (GT poses); the first LM
  step alone pulls 4.28 px RMS down to ~1 px. Do not read history[0] as a starting point.
- **Track coordinates exceed the model grid only in the low-visibility mode.** Ungated,
  tracks span x −25.4…678.8 against a 592×448 model grid; after the vis > 0.2 gate they span
  −4.9…593.7 by −5.3…455.7. There is no track/intrinsics resolution mismatch — an earlier
  ungated extent measurement suggested one.

## Out of scope

Recorded as follow-ons, not measured here:

- min-triangulation-angle gate in `_filter_observations` (a fix, and only if the ceiling
  verdict holds)
- `ba_percam` rotation-vs-translation attribution (owed from track-quality parity)
- multiview-confidence-on ablation for BA landmark quality
- robust kernel / observation weighting via `_BAModel.forward`
- incremental-frame inclusion. `BundleAdjustmentConfig.increment_size` /
  `_refine_incremental` already exist but **discard refined points every increment**, so
  they are a pose-only warm-start chain, not incremental BA. Their sweep numbers in the
  config comment are stale on two counts (measured with the point-discard behaviour *and*
  under the pre-`b27a411` truncated solver). Do not cite them.

## Testing

No unit tests. One-off measurement script. Its correctness gate is internal: GT mapped into
the recon frame must score ATE ≈ 0 against GT before either BA run starts, which catches a
wrong Sim(3) or pose convention immediately.

## Deliverable

A measured report at `docs/superpowers/specs/2026-08-19-ba-ceiling-measured-report.md`
(following the multiview-confidence precedent) carrying the numbers and one of two verdicts:

- **ceiling confirmed** — record the quantified limit, close the "make BA help everywhere"
  line of work, keep `bundle_adjustment: false` as default, and promote the min-tri-angle
  gate to the next spec
- **ceiling refuted** — the run names what to fix instead, and this becomes an optimizer spec

## Environment

- `/opt/venv/reconstruction/bin/python` (base `python` is the wrong 3.13)
- GPU jobs serially, in tmux, never two at once
- Memory: read `rss` from `/sys/fs/cgroup/memory/memory.stat`, **not**
  `memory.usage_in_bytes` — the latter counts page cache and reads ~48/50 GB on an idle
  box whose real anonymous usage is 1.5 GB. Earlier handoffs treat that counter as a
  blocker; it is not one.
- stage named files only; `git add -f` for anything under `docs/superpowers/`
