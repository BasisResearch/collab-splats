# Depth alignment x 2dgs surface normals — grid results

**Scene:** GH010229, 300 keyframes, InstantSfM backend, undistorted (`GH010229_undist_r7`).
**Training:** 2dgs, 12k steps, `pose_opt` + `appearance_opt` on, `normalize_scene` on,
`grow_grad2d 2e-4`, depth loss 0.01 -> 0.001, distortion 0.01 from step 3000.
**Held fixed:** every cell reuses cell 1's reconstruction. SfM is never re-run, so the only
variables are the depth alignment model and the normal-consistency `depth_ratio`.

## PSNR / SSIM

| cell | alignment | normals | PSNR | SSIM | gaussians | train s |
|---|---|---|---|---|---|---|
| 1 (baseline) | scale | mean | 20.8049 | 0.6776 | 1,875,613 | — |
| 2 | affine | mean | 20.7293 | 0.6798 | 1,829,766 | 922 |
| 3 | affine | median 0.6 | 20.8762 | **0.6854** | 1,833,964 | 838 |
| 3b | scale | median 0.6 | **20.9080** | 0.6831 | 1,871,864 | 841 |

Deltas against cell 1:

- median normals alone (3b): **+0.103 dB / +0.0055 SSIM**
- affine alone (2): **-0.076 dB** / +0.0022 SSIM
- affine on top of median (3): **-0.032 dB** against 3b

The two levers are anti-additive — the same shape every prior lever in this repo has shown
(appearance + decay + radius, 2026-08-26). Cell 3b is the best PSNR; cell 3 is the best SSIM by
0.0023 while costing 0.032 dB, and it carries the affine defect below.

**Verdict: ship median normals, drop affine.** Affine loses on PSNR both alone and stacked, and
its only win is an SSIM margin cell 3b nearly matches without it.

## Why affine is dropped rather than parked

The affine model's near-pole blow-up (Task 6/7 review, finding I1) was real: `d / (a + b*d)`
has a pole at `d = -a/b` for a `b < 0` fit, and the apply path masks only *at or past* that pole.
A fit whose pole sat just beyond the frame's far end therefore passed the positivity-only guard
and mapped far pixels to many times their scale-only depth while staying inside every mask.
Dropping affine from the splat path removes that exposure from splats, but not from the metric
path, where the aligned depth is re-unprojected into `world_points` and reaches `sparse_pc.ply`
and the pointcloud-sourced TSDF.

Fixed at the source instead: the absolute `AFFINE_EPS = 1e-9` positivity floor is replaced by
`AFFINE_MIN_FAR_DISPARITY_FRAC = 0.5` — the fitted disparity at the frame's far end (p99 of the
map, or the furthest inlier, whichever is larger) must keep at least half of what the scale-only
mapping would give there, which caps the pole's distortion at 2x. Frames that fail fall back to
scale-only, the same mapping the shipped default uses.

Measured on GH010229 (300 frames, all frames clear the observation floor and none invert):

| far-end floor | frames fitted | fell back |
|---|---|---|
| positivity only | 273 | 27 |
| 0.25 (4x cap) | 243 | 57 |
| **0.5 (2x cap, shipped)** | **202** | **98** |

So a third of the frames carried a fit that blew up past 2x at its own far end. They now take the
scale path. Anyone re-measuring affine later should read the fitted count first: at 202/300 the
affine mode is much closer to the scale mode than the grid above tested it at.

## Cell 4 (8 FPS contiguous VDA context stream) — not run

Skipped on user decision after the mechanism was found refuted in-tree. `sfm.py:270-273` records
that `metric=True` loads the metric head *and* disables `infer_video_depth`'s cross-window
scale-and-shift chaining (`video_depth.py:135`), so consecutive windows are stitched on the
head's own absolute output rather than fitted to each other. Measured 2026-08-26: a full-video
pass moved contiguity by -0.7% CV and -0.9% residual — the wrong direction.

A control would also have cost a second full pipeline rather than a re-alignment: InstantSfM
consumes the VDA maps off disk (`sfm.py:306`), so depth cannot be swapped under a finished
reconstruction.

The code ships tested (Tasks 1-4, 11). The **relative** (non-metric) VDA path is untested and
remains the only version of this idea with an intact mechanism; it needs a relative checkpoint
plus a scale-and-shift fit against the reconstruction.

## Mesh comparison

Cell 1 vs cell 3b — one variable, mean vs median depth/normals — crossed with the meshing-time
`mesh.splat_depth` knob (Task 10), which selects which rendered depth TSDF fuses. These are two
distinct levers that were being conflated: `depth_ratio` shapes training, `splat_depth` shapes
fusion.

Both cells' `splats.zarr` were rebuilt from `ckpt.pt` rather than retrained, since splat training
is unseeded and a retrain would not reproduce the graded PSNR. The rebuild reproduces cell 1's
recorded metrics exactly (PSNR delta 0.0000, SSIM delta 0.0000), which is asserted in the driver.

*(numbers pending)*
