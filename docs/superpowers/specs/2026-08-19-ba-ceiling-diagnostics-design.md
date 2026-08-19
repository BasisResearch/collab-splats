# BA Ceiling Diagnostics — Design

**Date:** 2026-08-19
**Status:** design, awaiting review
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

## Verdict under test

**BA has hit a structure-conditioning ceiling on this sequence.** The tracks that carry
enough baseline to improve on the model's poses are the long ones, and the long ones are
the ones most likely to have drifted. The information you need and the information you can
trust are disjoint.

Supporting arithmetic, from measured geometry (not assumed):

- camera-centre extent **0.304 m**, median landmark depth **1.03** (recon units)
- σ_Z ≈ Z²·σ_px / (f·B), with f ≈ 550 px, σ_px ≈ 0.5
  - full extent B = 0.304 m → σ_Z ≈ **3 mm**
  - median triangulation angle 4.24° → B ≈ 76 mm → σ_Z ≈ **13 mm**
- baseline ATE = **7.98 mm**

Structure uncertainty straddles the pose error BA is meant to remove. Under plain least
squares that structure noise is transferred into the poses.

**Falsifier:** point-only BA with poses frozen at ground truth. If the converged loss lands
near `4.083486e+05`, the tracks are consistent with the *correct* poses and the ceiling
verdict fails — the fault would be in the optimizer or the parameterization instead.

## Scope

Diagnose only. **No behaviour change to `collab_splats/geometry/bundle_adjustment.py`.**
One standalone script owning its own BA re-runs. `vggt_omega` only.

Out of scope, recorded as follow-ons:

- min-triangulation-angle gate in `_filter_observations` (a fix)
- `ba_percam` rotation-vs-translation attribution (owed from track-quality parity)
- multiview-confidence-on ablation for BA landmark quality
- robust kernel / observation weighting via `_BAModel.forward` (bae's `LM.step` ignores
  both `kernel=` and `weight=` on the step direction — see Findings)

## Diagnostics

| id | question | BA run? | cost |
|----|----------|---------|------|
| D1 | are the tracks consistent with the *true* poses? | yes, points-only | ~15 min GPU |
| D2 | what does the converged residual field look like? | no, re-uses D1 state | seconds |
| D3 | does BA recover poses from *clean* tracks? | yes, synthetic | ~15 min GPU |
| D4 | does incremental frame inclusion help? | yes, 20 + 50 frame windows | ~10 min GPU |
| D5 | how degenerate is the structure geometrically? | no | seconds |

### D1 — point-only BA at frozen ground-truth poses

Freeze extrinsics at 7-Scenes GT (Sim(3)-aligned into the recon frame), optimize landmarks
only, same filter settings, same LM schedule. Report converged loss against
`4.083486e+05`.

- loss **≪** baseline → tracks are fine, poses were the problem → verdict fails, look at
  the optimizer
- loss **≈ or >** baseline → tracks cannot be reconciled with the true poses → ceiling
  confirmed, and its magnitude is quantified

The single decisive measurement. Everything else is attribution.

### D2 — converged residual field

At the converged state, with no further optimization:

- reprojection-residual histogram, per-observation, plus p50/p90/p99
- cheirality violations (landmarks behind a camera that observes them)
- **re-filter flip count**: how many observations `_filter_observations` would drop if
  re-applied at `max_reproj_error` ∈ {4, 2, 1} px

The flip count directly tests whether one-shot filtering costs us — it is what COLMAP's
`ba_global_max_refinements=5` loop would keep removing.

### D3 — synthetic negative control

Generate perfect tracks by projecting GT-posed landmarks into GT poses, then add isotropic
pixel noise at σ ∈ {0, 0.5, 1, 2} px. Same track topology (length distribution, visibility
pattern) as the real cache, so only correspondence quality changes.

- σ=0 must recover GT poses to numerical precision. If it does not, the bug is ours and
  every other diagnostic is void — **this is the harness's own self-test**.
- the σ at which ATE degradation matches the observed +6.3% is a calibrated estimate of our
  effective track noise.

### D4 — window sweep

BA on frames [0,20) and [0,50), ATE scored on the same window. Tests the incremental-frame-
inclusion hypothesis at ~5% of the cost of implementing it.

`BundleAdjustmentConfig.increment_size` / `_refine_incremental` already exist but **discard
refined points every increment**, so they are a pose-only warm-start chain, not incremental
BA. Their sweep numbers in the config comment are stale on two counts (measured with the
point-discard behaviour *and* under the pre-`b27a411` truncated solver). Do not cite them.

### D5 — triangulation-angle census

For each landmark, the max pairwise angle between viewing rays from the cameras that observe
it (vis > `vis_thresh`). Pure geometry — camera centres and existing landmark positions. No
triangulation is performed and no BA is run.

Report the distribution and the fraction of points *and* observations below candidate gates,
including COLMAP's own thresholds for reference (`filter_min_tri_angle=1.5°`,
`ba_local_min_tri_angle=6.0°`, `init_min_tri_angle=16.0°`; verified on installed
pycolmap 4.0.4).

Sizes the prospective fix before anyone writes it.

## Findings that shape the design

Established before this spec; recorded so the diagnostics do not re-litigate them.

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

## Structure

One script, `evals/scripts/diagnose_ba_ceiling.py`, with a subcommand per diagnostic so any
one can run alone:

```
diagnose_ba_ceiling.py {angles,frozen-gt,residuals,synthetic,windows} [--out DIR]
```

- reads the warm track cache at
  `evals/results/ba_convergence_chess/cache/vggt_omega/tracks.zarr` (the only fine-tracking
  cache on disk) — never re-extracts
- reads GT + BA poses from `evals/results/ba_convergence_chess/vggt_omega/trajectories.npz`
  (keys `gt`, `pred_ba`, `ate_per_frame_ba`)
- imports `BundleAdjustment` read-only; each BA variant is set up in the script via config
  overrides and local pose freezing, never by editing the module
- writes one JSON per subcommand plus a combined report to `--out`
  (default `evals/results/ba_ceiling_diag/`, gitignored)

Baseline (pre-BA) poses are not in `trajectories.npz`. `angles` uses `pred_ba` centres as a
proxy; the two trajectories differ by <1 mm ATE against a 0.304 m extent, so the angle
distribution is insensitive to the choice. Stated in the output, not assumed silently.

## Error handling

- refuse to run if the track cache is missing, rather than triggering a 25 GB re-extraction
- assert `tracks`/`vis_scores`/`pts3d_tracks` frame counts agree with the pose arrays
- `synthetic` σ=0 asserts GT recovery and aborts the whole run if it fails

## Testing

No unit tests. This is a one-off measurement script and its correctness gate is D3 σ=0 — a
self-test with more teeth than any fixture, since it exercises the full BA path and demands
an exactly-known answer.

## Deliverable

A measured report at `docs/superpowers/specs/2026-08-19-ba-ceiling-measured-report.md`
(following the multiview-confidence precedent) carrying the numbers and one of two verdicts:

- **ceiling confirmed** — record the quantified limit, close the "make BA help everywhere"
  line of work, keep `bundle_adjustment: false` as default, and promote the min-tri-angle
  gate to the next spec
- **ceiling refuted** — D1 names what to fix instead, and this becomes an optimizer spec

## Environment

- `/opt/venv/reconstruction/bin/python` (base `python` is the wrong 3.13)
- GPU jobs serially, in tmux, never two at once
- Memory: read `rss` from `/sys/fs/cgroup/memory/memory.stat`, **not**
  `memory.usage_in_bytes` — the latter counts page cache and reads ~48/50 GB on an idle
  box whose real anonymous usage is 1.5 GB. Earlier handoffs treat that counter as a
  blocker; it is not one.
- stage named files only; `git add -f` for anything under `docs/superpowers/`
