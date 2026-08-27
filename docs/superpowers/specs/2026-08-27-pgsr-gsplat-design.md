# PGSR on gsplat — design

**Status:** design approved for implementation (2026-08-27)
**Reference implementation:** `yanxian-ll/GS-SR` (`gssr/scene/pgsr_scene.py`, `gssr/gaussian/pgsr_gaussian.py`,
`gssr/dataloader/pgsr_dataloader.py`, `gssr/utils/{point_utils,graphics_utils,mvsnet_utils}.py`) and the
plane rasterizer it calls, `zju3dv/PGSR` `submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu`.
Neither ships a license, so **nothing is vendored** — every piece below is reimplemented against our own
tensors and cites the upstream file and line at the site, per the repo's attribution rule.

## Goal

Add PGSR (Planar-based Gaussian Splatting Reconstruction) as a third `splats.primitive`, matching GS-SR's
`method_configs["pgsr"]` in render math, loss set, and constants — so it composes with
`representation: scaffold` the way GS-SR's `scaffold-pgsr` does, and so the mesh stage gets a
plane-consistent depth instead of the alpha-expected one.

## Why it fits gsplat without a custom kernel

PGSR's rasterizer is vanilla 3DGS plus a 5-channel alpha-weighted side buffer, and its unbiased depth is a
per-pixel closed form over that buffer — not a per-Gaussian accumulation:

```
out_plane_depth[pix] = All_map[4] / -(All_map[0]*ray.x + All_map[1]*ray.y + All_map[2] + 1e-8)
ray = ((x - cx)/fx, (y - cy)/fy)                 # forward.cu:404, ray at forward.cu:304
All_map[0:3] = sum_i alpha_i T_i * n_i           # camera-frame normal, oriented toward the camera
All_map[3]   = sum_i alpha_i T_i                 # == the accumulated alpha gsplat already returns
All_map[4]   = sum_i alpha_i T_i * |n_i . p_i|   # camera-frame plane distance of Gaussian i
```

So the side buffer is four channels we do not already have (normal 3 + distance 1) — exactly the width of
the `extra_signals` path our 3DGS branch uses today for normals (`rendering.py:162-170`, 4 channels because
the compiled kernel takes 8 total but not 7). Both `All_map[0:3]` and `All_map[4]` are un-normalised
alpha-weighted sums, and the depth formula is a ratio of them, so the missing alpha division cancels — the
same reason the plane-induced homography below can use the raw accumulated `n` and `d`.

**Deviations we accept, all documented at the call sites:**
- our 3DGS kernel runs `rasterize_mode="antialiased"`; upstream's plane rasterizer is the plain 3DGS kernel;
- no `out_observe` (upstream counts pixels where a Gaussian contributed while `T > 0.5`, and uses it only to
  gate `max_radii2D` updates during densification) — we gate on `radii > 0` instead;
- densification is gsplat's `DefaultStrategy` with `absgrad=True`, not upstream's dual-threshold
  (grad **and** abs-grad) split with its `max_all_points` / `abs_split_radii2D_threshold` caps.

## Components

### 1. Render (`collab_splats/splats/rendering.py`)

`primitive == "pgsr"` takes the 3DGS branch with `extra_signals = [normal_cam(3), distance(1)]`:

- `normal_cam` is the existing `gaussian_normals_in_camera_frame` — it already flips each normal to face the
  camera (`n . p_cam <= 0`), which is upstream's `get_normal` (`pgsr_scene.py:252-258`) and the reason the
  depth formula carries a leading minus.
- `distance = (normal_cam * means_cam).sum(-1).abs()` (`pgsr_scene.py:297`).
- `plane_depth` is the closed form above; `render["depth"]` stays the alpha-expected depth so the depth prior,
  the zarr writer and the mesh stage keep reading what they read today, and `plane_depth` is added beside it.
- `depth_normal` = `depth_to_normal(plane_depth, identity, K) * alpha.detach()` — upstream finite-differences
  the plane depth in **camera** space (`point_utils.depth2point_world` returns camera points; its world
  transform is commented out) and scales by the detached alpha (`pgsr_scene.py:319`). gsplat's
  `depth_to_normal` cross product has the same handedness as upstream's `depth_pcd2normal`.
- `normal` is the raw accumulated normal, **not** normalised — upstream compares and warps with the
  accumulated one.

### 2. Losses (`collab_splats/splats/losses.py`, helpers in `collab_splats/splats/pgsr.py`)

Three registry entries, all gated at step 7000 upstream (`pgsr_scene.py:108,114`):

| ours | upstream | weight |
|---|---|---|
| `pgsr_normal` | single-view planar loss | `lambda_normal = 0.015` |
| `pgsr_geo` | multi-view geometric loss | `lambda_geo = 0.03` |
| `pgsr_ncc` | multi-view photometric loss | `lambda_ncc = 0.15` |

- **Single view:** `w = (1 - grad_weight(gt))^5`, eroded with a 5×5 min-filter, times
  `|depth_normal - normal|.sum(channels)`, meaned. `grad_weight` is the max of the mean absolute
  horizontal/vertical colour differences, min-max normalised, padded with 1.0 (`pgsr_scene.py:29-45`).
- **Multi view:** unproject this view's `plane_depth` to world points, project into the neighbour, read its
  `plane_depth` bilinearly, re-project back, and take the pixel round-trip error `pixel_noise`. The mask is
  `in-frame and z > 0.1 and pixel_noise < 1.0`; the weight is `exp(-pixel_noise).detach()`. `pgsr_geo` is
  `(weight * pixel_noise)[mask].mean()`.
- **NCC:** sample up to `num_sample = 102400` masked pixels, build 7×7 patches (`patch_size = 3`), warp them
  into the neighbour through the plane-induced homography
  `H = K_near (R_rel - t_rel n^T / d) K_ref^-1` built from the **accumulated** `n`, `d` (scale cancels),
  and take `1 - NCC^2` clamped to [0, 2], keeping patches with `ncc < 0.9`, weighted by the same weights.

### 3. Neighbour views (`collab_splats/splats/pgsr.py`)

GS-SR prefers its COLMAP path (`pgsr_dataloader.py:31-46`): the MVSNet-style covisibility score
`sum over co-visible points of exp(-(theta - 5)^2 / (2 sigma^2))`, `sigma = 1` below 5° and `10` above,
top `num_multi_view = 5` per view. We have no per-image track ids at the trainer boundary, so the score is
computed over the seed point cloud we already pass in, with co-visibility approximated by "projects inside
both frames with positive depth". The pose-only fallback (angle < 30°, distance in [0.01, 1.5]) is not
reimplemented: its distance thresholds are raw-COLMAP-unit values that mean nothing in our normalised frame.

### 4. Trainer (`collab_splats/splats/trainer.py`)

- `pgsr` renders a second view — one of the current view's neighbours, sampled uniformly — once any
  multi-view term is live, at the same downscale factor and through the same pose refiner, with gradient
  (upstream backprops through both renders, `pgsr_scene.py:211-217`).
- The neighbour render, the two cameras and the gray images ride in `target["pgsr"]`, so the loss registry
  signature stays `(render, target, gaussians, scene_scale, spec)`.
- Gray is ITU-R 601-2 luma (`torchvision.transforms.Grayscale`, GS-SR `cameras/__init__.py:64`),
  computed on the GPU target at the step's resolution; GS-SR keeps `ncc_scale = 1.0`.
- Strategy: `DefaultStrategy(absgrad=True, grow_grad2d=8e-4, grow_scale3d=0.001, prune_opa=0.005,
  refine_start_iter=500, refine_stop_iter=15000, refine_every=100, reset_every=3000)` — upstream's
  `PGSRGaussianConfig` numbers (`pgsr_gaussian.py:12-18`, `vanilla_gaussian.py:41-46`) mapped onto the
  knobs gsplat exposes.

### 5. Config

```yaml
splats:
  primitive: pgsr
  losses:
    pgsr_normal: {weight: 0.015, start: 7000}
    pgsr_geo:    {weight: 0.03,  start: 7000}
    pgsr_ncc:    {weight: 0.15,  start: 7000}
```

`normal_consistency` and `distortion` are 2DGS-only and stay off under `pgsr`; validation rejects them, the
way `depth_ratio` is already rejected off 2DGS.

## Testing

Unit tests in `tests/splats/test_pgsr.py`, all CPU except the two marked `@cuda`:
- plane depth of a fronto-parallel plane equals its distance, and of a slanted plane equals the ray-plane
  intersection (closed form, no rasterizer);
- the gradient weight is 1 on a flat image and drops on an edge; erosion shrinks the mask;
- `lncc` is 0 for identical patches, and invariant to per-patch gain/bias (that is what NCC buys);
- the homography maps the reference pixel of a plane onto its true neighbour pixel, checked against an
  explicit projection of a known 3D plane;
- neighbour selection prefers a co-visible view at ~5° over a co-visible view at 60° and over one that sees
  nothing in common;
- `@cuda`: a 60-step `primitive: pgsr` run writes the same artifact set as 3dgs, and the same with
  `representation: scaffold` (GS-SR's `scaffold-pgsr`).

## Measurement

One 12k-step run on `GH010229_undist` against the logged 3DGS baseline (21.606 / 0.7169 / 1,000,000 / 601 s)
and the scaffold runs, reported the way Task 13 reports the others. PGSR's claim is mesh quality, not PSNR,
so the mesh from `plane_depth` is the second half of that comparison.
