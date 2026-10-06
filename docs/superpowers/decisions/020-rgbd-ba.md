# 020 — RGB-D bundle adjustment: carry `joint` forward

Date: 2026-09-30 · Status: accepted · Branch: `feat/rgbd-ba` ([spec](../specs/2026-09-30-rgbd-ba-design.md))

## Context

- spec added depth (C1), depth scale (C2) and photometric (bae `rgbd` stage D port) residuals to
  track BA, plus a post-solve gauge fix; question: does any term set improve VGGT-Omega poses
- refine-only runs: one Omega pointcloud per scene, each BA condition on it, tracks cached
- 7-Scenes chess, fire, office, seq-01, 200 frames

## Scoring

- 7-Scenes D-SLAM GT is itself inaccurate (Brachmann et al., ICCV'21): it scores every BA
  condition worse than raw Omega, including ones that halve epipolar error
- primary reference: the SfM pseudo-GT from that paper (ATE, RPE, AUC@5)
- reference-free check: median Sampson px over 600 SIFT + MAGSAC pairs per scene (gaps 1/2/4/8),
  at each run's own refined focal

## Results

pGT ATE mm / pGT AUC@5 / Sampson px:

| Condition | chess | fire | office |
|---|---|---|---|
| omega (no BA) | 13.9 / 42.8 / 0.426 | 13.2 / 37.4 / 0.495 | 19.2 / 65.1 / 0.383 |
| reproj | 14.6 / 71.2 / 0.175 | 20.5 / 39.7 / 0.255 | 6.7 / 83.7 / 0.241 |
| photometric | 6.7 / 69.2 / 0.145 | 16.0 / 43.7 / 0.204 | 10.5 / 63.1 / 0.194 |
| **joint** (reproj + photometric + depth) | **6.2 / 85.4** / 0.154 | 14.6 / **43.7** / 0.251 | 8.3 / **89.9** / 0.219 |

- joint: best AUC@5 on all three scenes, best ATE on chess, 35-54 s per scene
- fire: joint gains AUC@5 and Sampson but ATE is 1.4 mm worse than raw Omega
- photometric alone gives the lowest Sampson but never moves focal and trails joint on AUC@5

## Rejected: per-frame depth correction grid

- per-frame 6x8 log-depth grid, bilinear, prior σ 0.05, jointly with poses
- photometric_grid: Sampson -0.001 to -0.004 px vs photometric, pose metrics worse on chess and fire
- joint_grid: worse everywhere; chess collapses (focal 556 → 450, pGT ATE 126 mm)
- a free grid inside the track depth rows ran out of GPU memory in J^T J (30 GB request);
  a grid frozen per relinearization fit but scored above
- code removed; not re-proposed without a fix for the scale / focal drift

## Deviations from the spec

- `fine_tracking: false`: VGGSfM fine tracking peaks at 34 GB RSS on 50 frames
- photometric samples 128 per view, not rgbd's 2048: 256 faults on a shared 46 GB card
- matrix-free LM (bae 0.2.5 `matrix_free_normal`, PCG), not CuDSS: 0.2.4 CuDSS goes NaN after the
  pattern changes, and forming J^T J cost 26.3 GB vs 9.7 GB on chess N=100 at equal ATE
- bae `TrustRegion` (damping floor 1e-6) + undo of a loss-raising step: pypose's scored a
  negative predicted drop as good and diverged (chess N=50, loss 4.4e5 -> 9.5e61)
- `torch.cuda.empty_cache()` before each solve and each photometric scale: warp allocates the Jacobian outside torch's pool
- gauge rotation from the chordal mean of orientation changes: a center-only Sim(3) cannot see a
  roll about a near-collinear trajectory

## Decision

- carry `joint` forward: `use_photometric` and `use_depth` default to true, `fine_tracking` to false,
  in both `configs/base.yaml` and `BundleAdjustmentConfig`
- BA itself stays off by default; `bundle_adjustment: {enabled: true}` now runs joint
- removed, not carried: `fit_depth_scale` (per-frame depth scale, C2) and `use_reprojection: false`
  (photometric-only, pose-only refinement); both scored below joint (depth scale on chess only,
  pre-gauge-fix pass: pGT ATE 10.7 mm / AUC@5 63.7 vs joint 4.5 / 83.7). Reprojection is always on
- `refine()` raises if depth or photometric is on and no depth is passed
