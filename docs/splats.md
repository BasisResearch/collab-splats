# Splats Module

`collab_splats.splats` trains Gaussian splats on upstream [gsplat](https://github.com/nerfstudio-project/gsplat)
(pinned at `GSPLAT_COMMIT`) from an existing pointcloud stage. Training runs in the COLMAP camera
convention: poses are `world_to_cam`, intrinsics are at frame resolution, and the seed points come
from `pointcloud.zarr`.

---

## Module layout

| File | Contents |
|------|----------|
| `trainer.py` | `SplatsConfig` and `train()`: setup, loop, finish. No representation branches. |
| `gaussian.py` | `Gaussians` — vanilla 3DGS/2DGS parameters, `make_strategy`, `SH_C0`. |
| `scaffold.py` | `Scaffold` — Scaffold-GS anchors, the three MLP heads, `AnchorStrategy`. |
| `rendering.py` | `render_gaussians` (the one call into gsplat), plus `render_views` / `write_outputs` / `load_checkpoint`. |
| `losses.py` | the loss schedule and the optional-loss registry. |
| `cameras.py` | `CameraOpt` — per-camera pose deltas and per-image color affine, each half optional — and `rotation_6d_to_matrix`. |
| `pgsr.py` | PGSR planar geometry, neighbor selection, `render_neighbor`. |
| `utils.py` | scene normalization, coarse-to-fine downscaling, view order, target preparation. |

`train()` builds one model — `Gaussians` or `Scaffold`, chosen by `representation` — and one
`CameraOpt`. The two model classes expose the same members, so the loop is
representation-agnostic; `CameraOpt` is one module whose two halves — pose delta and color
affine — are selected independently by `pose_opt` / `appearance_opt` and pass their input
through untouched when off, so the loop does not branch on those either: refine the camera,
render, apply the color affine, composite over a random background, compute the losses,
backward, step every optimizer and scheduler, then let the model densify itself. At the end the
model and cameras are put back in the world frame and `write_outputs` writes `splats.ply`,
`ckpt.pt` and `splats_quality_report.json`. The checkpoint is self-contained: `load_checkpoint`
plus `render_views` reproduce every training-view render, which is how the mesh stage and the
eval scripts consume the model.

---

## Quickstart

The pipeline runs this stage for you (`splats:` in the yaml, `--stages splats`). Direct use:

```python
from pathlib import Path

from collab_splats.splats import SplatsConfig, train

cfg = SplatsConfig(primitive="3dgs", max_steps=30000)
train(
    cfg,
    images=images,              # (n_views, H, W, 3) uint8
    world_to_cam=world_to_cam,  # (n_views, 4, 4), COLMAP convention
    intrinsics=intrinsics,      # (n_views, 3, 3) at frame resolution
    points=points,              # (P, 3) float32 seed points
    colors=colors,              # (P, 3) uint8
    out_dir=Path("scene/splats"),
    depth_targets=depth,        # optional (n_views, h, w) float32, 0 = no target
)
```

`SplatsConfig` mirrors the `splats:` yaml block minus `enabled`; `configs/base.yaml` documents every
knob with its default.

---

## Two axes: primitive and representation

The two are independent — every combination trains.

### `primitive` — which rasterizer

- `3dgs` (default): `gsplat.rasterization`, antialiased. Per-Gaussian normals ride along as an extra
  signal. Densifies with MCMC under a `cap_max` budget.
- `2dgs`: `gsplat.rasterization_2dgs`. The rasterizer returns rendered normals, a distortion map and
  the RaDe-GS median depth. Densifies with `DefaultStrategy` under `grow_grad2d`.

Better surfaces come from `2dgs`; better thin structure and speed from `3dgs`.

### `representation` — what the parameters are

- `vanilla` (default): one set of parameters per Gaussian — means, quats, log-scales, logit-opacities
  and SH bands. `sh_degree` unlocks one band every `sh_degree_interval` steps.
- `scaffold`: anchors, each carrying a feature vector and `n_offsets` learned offsets. Nothing is
  stored per Gaussian; small MLP heads decode opacity, covariance and color **per view** from the
  anchor feature and the viewing direction, and only the decoded Gaussians reach the rasterizer.
  Densification grows and prunes *anchors* (voxel-quantized, on accumulated 2D gradient) rather than
  Gaussians. Color comes from `mlp_color`, so `sh_degree` / `sh_degree_interval` are
  vanilla-only: left at their defaults they are ignored, set deliberately they raise.

```yaml
splats:
  primitive: 2dgs
  representation: scaffold
  scaffold:
    n_offsets: 10
    feat_dim: 32
    voxel_multiplier: 1.0     # x median kNN spacing of the seed points
    appearance_dim: 32        # Scaffold's per-image embedding; 0 = off
```

`scaffold.appearance_dim` and the top-level `appearance_opt` are two different appearance models —
a per-image embedding fed into the MLP heads, versus a per-image affine color transform on the
render. They are independent toggles; either, both, or neither.

---

## Outputs

`train` writes three files into `out_dir`:

| file | contents |
|---|---|
| `splats.ply` | binary Gaussian ply. Under `scaffold` it is **baked**: each anchor offset is decoded once at that anchor's mean observed view direction, so the ply is a fixed-view snapshot rather than the model. |
| `ckpt.pt` | `splats`, `config`, `cam_to_world`, `intrinsics`, `image_ids`, `image_size`, `appearance` — and, under `scaffold`, the MLP heads and voxel size. The **pose refiner is not stored**: its deltas are folded into `cam_to_world` before the write, so reloading cannot double-apply them. `appearance` is the color half of `CameraOpt`. This is the reloadable model. |
| `splats_quality_report.json` | mean and per-frame PSNR / SSIM, wall time, the final-step loss snapshot, and `n_gaussians` — the Gaussian count under `vanilla`, the **anchor** count under `scaffold` (the decoded count is per view, so there is no single number). |

There is no stored render. `splats.zarr` — per-view `rgb` / `alpha` / `depth` / `normal` — was
retired on 2026-09-06: `ckpt.pt` is self-contained (model, cameras, image size, config), so
`collab_splats.splats.rendering.load_checkpoint` + `render_views` reproduce every one of those
views on demand, and a stored copy could only go stale against the checkpoint beside it.

---

## Losses

`losses:` is a schedule, not a list of flags: `name: {weight[, start, end, end_weight]}`. A loss
contributes when its weight at the step is > 0; with `end` the weight decays log-linearly from
`weight` at `start` to `end_weight` at `end` and holds there. Photometric (0.8 L1 + 0.2 (1 - SSIM))
is always on and not configurable.

Registered: `depth`, `normal_consistency`, `distortion`, `opacity_reg`, `scale_reg`,
`appearance_reg`, `pgsr_normal`, `pgsr_multiview` — the last two are the PGSR planar terms in
`pgsr.py`, and both need a plane render (`render_plane`), which only the 3dgs kernel produces.
Defaults differ per primitive — `opacity_reg` / `scale_reg` for 3dgs (MCMC needs them), `distortion`
for 2dgs. Under `scaffold` the two regularizers read the *decoded* opacities and log-scales off the
render, since there is no per-Gaussian parameter to regularize.
