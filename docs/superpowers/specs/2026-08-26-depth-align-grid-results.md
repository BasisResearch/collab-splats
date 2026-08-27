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

`align_depth_affine` now also reports how well the accepted fits explain their own inliers —
per-frame median relative disparity residual, p10/p50/p90. On the same scene it moves 0.0208 /
0.0454 / 0.1020 (positivity only) to 0.0193 / 0.0414 / 0.0897 at the shipped floor: the frames
the guard rejects were the worse-fitting ones, not a random third.

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

Cell 1: `normal_consistency` without `depth_ratio` (mean normals) — PSNR 20.8049 / SSIM 0.6776,
1,875,613 gaussians. Cell 3b: the same run with `depth_ratio: 0.6` — PSNR 20.9080 / SSIM 0.6831,
1,871,864 gaussians. Component counts are post-`clean_repair`; "speckle" counts components of 10
triangles or fewer.

| cell | splat_depth | voxel | verts | components | main frac | main area | speckle | ply | fuse |
|---|---|---|---|---|---|---|---|---|---|
| 1 | expected | 0.2 | 1,967,467 | 92,271 | 0.6338 | 0.7403 | 71,134 | 96 MB | 249 s |
| 1 | median | 0.2 | 1,510,864 | 23,386 | 0.6082 | 0.6376 | 16,661 | 76 MB | 191 s |
| 1 | expected | 0.1 | 8,592,585 | 428,028 | 0.5729 | 0.6838 | 318,576 | 423 MB | 700 s |
| 1 | median | 0.1 | 5,726,475 | 88,713 | 0.5469 | 0.5802 | 61,305 | 289 MB | 593 s |
| 3b | expected | 0.2 | 2,196,254 | 100,286 | 0.5895 | 0.7024 | 72,816 | 107 MB | 225 s |
| 3b | median | 0.2 | 1,703,976 | 30,206 | 0.5826 | 0.6174 | 21,094 | 85 MB | 194 s |
| 3b | expected | 0.1 | 9,546,593 | 493,405 | 0.5053 | 0.6000 | 377,800 | 467 MB | 666 s |
| 3b | median | 0.1 | 4,430,759 | 76,314 | 0.5471 | 0.5884 | 51,541 | 223 MB | 475 s |

**`splat_depth` is the larger lever, and it is a meshing-time one.** Fusing the median depth
instead of the alpha-weighted expected depth cuts component count 3.3-6.5x across both cells
(92,271 -> 23,386 at cell 1 / voxel 0.2) for 22-54% fewer vertices. Alpha-weighted depth places a
sample at the opacity centroid along each ray, which for a semi-transparent gaussian sitting in
front of a surface lands in empty space — those are the components that vanish. Nothing about the
trained field changed; this is one config key at fuse time.

**`depth_ratio: 0.6` in training does not transfer to the mesh the way it does to PSNR.** It is
worth +0.10 dB / +0.0055 SSIM, and on the expected-depth path it fuses ~12% MORE geometry, but
the extra geometry is fragments: main fraction falls 0.6338 -> 0.5895 at voxel 0.2 and
0.5729 -> 0.5053 at 0.1, with 19% more speckle at 0.1. On the median path the sign flips — cell
3b fuses 23% FEWER vertices than cell 1 (4.43M vs 5.73M at voxel 0.1) and 16% less speckle. A
field trained toward its median surface fuses a thinner shell when the median is what TSDF reads,
and a noisier one when it is not: the training knob and the fusion knob want to agree.

**Halving the voxel buys resolution and noise together.** 0.2 -> 0.1 multiplies vertices 2.6-4.4x
and speckle 2.4-5.2x while main fraction drops 0.04-0.08 in every cell. There is no setting here
where fine voxels are free.

### What the renders show

Every mesh was rendered from the scene's own cameras at views 0 / 75 / 150 / 225 / 299,
1296x972, one open3d process per mesh (`t13_render_mesh.py`); the four variants of each cell are
stacked as `t13_montage_view{000,150,225}.jpg`. The counts above cannot tell detail from speckle,
so these are the half of the verdict that decides the voxel.

**Distant sloppiness is voxel-bound, not depth-mode-bound.** At voxel 0.2 both paths render the
far hedge as smooth pale blobs with no leaf structure (view 225) — median does not sharpen them.
What median removes at distance is the bridging: the sheets that expected depth stretches between
the tree trunk and the wall behind it become honest holes. Leaf-scale structure only appears at
voxel 0.1, and only the median mesh also opens the dark gaps between leaves, which is what makes
the hedge read as foliage instead of a cliff face.

**Near-field detail arrives with the voxel too, and median is what keeps it usable.** At 0.2 the
near ground is a smooth blur (view 0) and the truck's side panel is featureless (view 150); at 0.1
individual cobbles and grass tufts resolve, the truck's door ribs appear, and the railing spindles
separate. `expected` at 0.1 buys that same detail buried under 318,576 speckle components;
`median` at 0.1 keeps it at 61,305 — 5.2x fewer — for 33% fewer vertices. This is the setting the
"fine detail up close" goal asks for.

**Median's cost is holes, not blur.** Every median tile carries black gaps where the expected mesh
has geometry: grazing surfaces and thin structures never accumulate consistent median votes and
drop out entirely. Nothing that survives is softer than its expected twin. Prefer expected only
where coverage matters more than correctness.

**Cell 3b is not visually separable from cell 1 at matched settings.** That confirms the table:
`depth_ratio: 0.6` earns its +0.10 dB in renders of the splats, not in the mesh.

### Verdict

- Set `mesh.splat_depth: median`. It is free at training time, cuts components 3.3-6.5x, and
  costs no visible sharpness — only coverage on grazing surfaces.
- Use voxel 0.1 when the mesh is the deliverable (5.7M verts, 289 MB, ~10 min fuse); keep 0.2 for
  preview meshes, where the distant surfaces are blobs either way.
- Leave `normal_consistency.depth_ratio` at its default. Revisit it only if median fusion becomes
  the default, since its sign flips favourable on that path.
