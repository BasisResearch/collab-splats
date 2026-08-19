# BA Ceiling — Measured Report

**Date:** 2026-08-19
**Status:** measured
**Design:** [ba-ceiling-diagnostics](2026-08-19-ba-ceiling-diagnostics-design.md)
**Scene:** 7-Scenes `chess/seq-01`, first 100 frames, `vggt_omega`
**Scripts:** `evals/scripts/ba_start_at_gt.py`, `evals/scripts/refit_at_fixed_poses.py`,
`evals/scripts/tri_angle_census.py`

## Verdict

**Ceiling confirmed. BA's objective is not minimised at the true poses on this sequence.**
The optimizer is working; the objective it is given prefers the model's answer to the truth.
No amount of solver work fixes that — only better observations or a better-conditioned
objective would.

Consequences:

- `bundle_adjustment: false` stays the default. Nothing found here changes it.
- "Make BA help on sub-centimetre baselines" is closed as a line of work.
- The remaining levers are **observation quality and weighting**, not the solver.

## Experiment 1 — start BA at the truth

The package's own `BundleAdjustment` run twice on one reconstruction, changing nothing but
the starting poses. Run B starts from ground truth mapped into the reconstruction frame by
`umeyama_sim3` on camera centres (reprojection is invariant to a global similarity, so this
changes the frame and nothing else). `max_reproj_error=None` on both runs, so both solve the
**identical** problem: 100/100 frames, 37,257/37,890 points, 1,966,262 observations.

| run | ATE start → end | LM loss (after step 1) → end | mean camera-centre movement |
|---|---|---|---|
| A — model start | 0.007976 → 0.008625 | 1.948468e+06 → 1.223338e+06 | 0.042195 |
| B — GT start | **0.000002 → 0.011641** | 4.660833e+07 → 1.493592e+06 | 0.097450 |

Sanity gate passed: GT expressed in the reconstruction frame scores ATE 2.2e-06 against GT
before the solve, so the Sim(3) and the pose convention are right.

Two readings:

1. **Started exactly at the truth, BA walks 11.6 mm away from it** while cutting its loss
   31×. It does not stall near the truth; it actively leaves.
2. **A's converged loss (1.223e6) is below B's (1.494e6).** The answer reachable from the
   model's poses is a *better* optimum of this objective than anything reachable from the
   truth.

Run A reproduced the shipping `ba` condition exactly in the filtered configuration
(loss 4.083486e+05, ATE 0.008481), so the harness is not a different pipeline.

## Experiment 2 — loss at the truth after the structure refits

Experiment 1 leaves one confound: `pts3d_tracks` are seeded from `result.world_points`,
fitted to the *model's* poses, so B's starting loss is "truth with mismatched structure".

`refit_at_fixed_poses.py` closes it. Poses held fixed, every landmark re-solved by
per-landmark LM triangulation (initialised from both the seed and the ray midpoint, cheaper
kept, so a refit can never score worse than its seed), shared focal held at 547.6871, same
observation set, same loss definition.

| poses (fixed) | structure = seeded | **structure refit to those poses** |
|---|---|---|
| model | 3.609296e+07 (4.2844 px RMS) | **8.891822e+06 (2.1265 px RMS)** |
| GT | 5.349418e+08 (16.4943 px RMS) | **2.490652e+07 (3.5591 px RMS)** |

**The truth is a 2.8× worse explanation of these tracks than the model's poses, with the
structure fitted to it.** The verdict is not an artifact of how the landmarks were seeded.

Both figures are upper bounds on their pose set's true best loss — poses are frozen and the
focal is not refit — but the treatment is identical for both, and the 2.8× gap is far beyond
what one shared focal DoF could close against 111,771 already-refit point DoF.

What 3.56 px RMS at the true poses means physically: 3.56 px at f = 547.7 is 6.5 mrad, which
at the median landmark depth is ~22 mm of lateral disagreement — **2.8× the 7.98 mm ATE that
BA is being asked to remove**. The tracks disagree with the truth by more than the error.

### Harness correctness

- Loss computed here is byte-identical to the package's own `_reproject_shared`:
  3.609295e+07 vs 3.609296e+07 at the model poses.
- **`_last_loss_history[0]` is the loss *after* LM step 1, not the initial loss.** The true
  initial losses are 3.609296e+07 (model) and 5.349418e+08 (GT); LM's first step already
  pulls 4.28 px RMS to ~1 px. Any reading of the loss curve must start from that.

## Why the objective prefers the wrong answer

Measured, in order of how much each contributes:

1. **The structure is under-conditioned for the error being corrected.** Sim(3) scale to
   metres is 0.385156, so 1 reconstruction unit = 2.596 m. With camera-centre extent 0.3075
   (0.798 m), median landmark depth 1.31 (3.40 m), f ≈ 548 px, σ_px ≈ 0.5:
   - best case, full 0.3075 baseline → σ_Z ≈ 0.0051 units = **13 mm**
   - typical, at the median 3.98° triangulation angle → B ≈ 0.091 units, σ_Z ≈ 0.0172 units
     = **45 mm**

   against a baseline ATE of **7.98 mm**. Depth uncertainty is 2–6× the pose error. Under
   plain least squares that structure noise transfers straight into the poses.

2. **The whole landmark population tops out near 11° of triangulation angle** (p99 = 11.26,
   p50 = 3.98). No landmark reaches COLMAP's `init_min_tri_angle=16.0`. COLMAP's answer to
   this sequence is to refuse to seed a reconstruction from it, not to solve it better.

3. **`_filter_observations` is self-confirming.** It selects observations by how well they
   agree with the model's *own* poses and points, so it keeps precisely the evidence that
   confirms the starting answer and discards evidence for a distant one — and it runs once,
   before the only solve. COLMAP escapes this by re-filtering across
   `ba_global_max_refinements=5`; upstream `zitongzhan/vggt` does not, and carries its own
   `# TODO: add iterative BA`. This is parity with upstream, not a deviation, but it is a
   real mechanism and it is why a filtered run cannot recover from a bad start.

4. **Robust weighting is unreachable through bae's API.** `LM.step` computes the step from
   raw residuals and applies the kernel only inside `self.loss(...)`, so `kernel=` changes
   the accept/reject test and nothing else; `weight=` is assigned and never used. Any
   down-weighting of long, drifted tracks has to fold √w into the residual inside
   `_BAModel.forward`.

## Supporting census

Per-landmark max pairwise angle between viewing rays from the cameras that observe it
(vis > 0.2). Pure geometry on existing camera centres and landmarks; no triangulation, no BA.
37,257 landmarks, 1,966,262 observations.

percentiles (deg): p1 0.12, p5 0.23, p25 1.07, **p50 3.98**, p75 7.01, p95 10.23, p99 11.26

| gate | points dropped | obs dropped | points kept |
|------|----------------|-------------|-------------|
| 0.5° | 6761 (18.1%) | 153952 (7.8%) | 30496 |
| 1.0° | 9043 (24.3%) | 234181 (11.9%) | 28214 |
| **1.5°** (COLMAP `filter_min_tri_angle`) | **10684 (28.7%)** | **291384 (14.8%)** | **26573** |
| 2.0° | 11915 (32.0%) | 335602 (17.1%) | 25342 |
| 3.0° | 15035 (40.4%) | 460702 (23.4%) | 22222 |
| 6.0° (COLMAP `ba_local_min_tri_angle`) | 25626 (68.8%) | 997632 (50.7%) | 11631 |
| 16.0° (COLMAP `init_min_tri_angle`) | 37257 (100%) | 1966262 (100%) | **0** |

A 1.5° gate is asymmetric in our favour — 28.7% of points but only 14.8% of observations —
and starves no frame (minimum surviving observations per frame 10,816 at 1.5°, 7,142 at 6.0°,
against `min_inliers_per_frame=64`). Thresholds verified on installed pycolmap 4.0.4.

## What is left worth trying

Ranked by expected value against this evidence. None is authorised by this report; each needs
its own spec.

1. **Min-triangulation-angle gate in `_filter_observations`** (1.5°). The only cheap change
   that attacks cause 1 directly, and the census says it costs 14.8% of the residual mass and
   cannot drop a frame. Expect it to *reduce harm* on sub-cm sequences, not to produce gains.
2. **Residual weighting by conditioning** — fold √w into `_BAModel.forward`, w falling with
   triangulation angle or track length. Strictly more information than a hard gate, and the
   only route to a robust kernel through bae.
3. **Iterative BA / re-filtering** (COLMAP's 5 refinements). Attacks cause 3. Larger change,
   and on this sequence the ceiling limits the upside.
4. **Anchor landmarks to the network's depth** with a prior term, so structure cannot absorb
   pose error. Biggest change, most likely to actually move the ceiling, least certain.

Explicitly *not* worth more work on this sequence: solver choice, trust region, LM step
count, initialisation. Experiment 1 rules all of them out — the optimizer reaches a better
optimum than the truth.

## Scope

Single scene, single backbone (`vggt_omega`), 100 frames. The ceiling arithmetic is a
property of the *baseline-to-depth ratio*, so it should hold on any sequence with
sub-centimetre inter-frame motion and metre-scale depths, and should not hold on wide-baseline
outdoor capture. Untested there.
