# Splats module — gsplat trainer replacing the nerfstudio submodule

**Date:** 2026-08-22
**Status:** approved design, awaiting plan
**Branch:** refactor/cu121-uv-migration

## Goal

Train Gaussian splats from an existing pointcloud-stage output (COLMAP poses + sparse points,
frames from `frames.zarr`) using upstream gsplat, so that rendered RGB/depth/normals feed the
mesh stage. Replace `collab_splats/nerfstudio/` (gsplat-rade fork + nerfstudio fork) with a
minimal `collab_splats/splats/` module. Four user requirements:

1. Losses (photometric, depth, normal consistency, distortion) integrated in one trainer.
2. Loss scheduling — each optional loss has a `start` step.
3. Camera pose optimization during training.
4. A seam for other rasterizers (PAGaS) later.

Overall aim: better rendering → better mesh.

## Implementation principles

- Reuse gsplat's installed API (`rasterization`, `rasterization_2dgs`, `gsplat.losses`,
  `DefaultStrategy`, `export_splats`, `load_ply_to_splats`). Nothing re-derived that gsplat ships.
- Copy specific upstream *functions* only (not example files), each attributed with
  repo + commit + file + lines at the site.
- Three files. One trainer for both primitives. No result class, no dataset class, no
  config module, no render adapter module.
- Upstream-proven defaults hardcoded; yaml exposes only what a user would change.
- Report-only sidecar JSON, same convention as `verification.json` and
  `reconstruction_quality_report.json`.
- Delete what this obsoletes: `collab_splats/nerfstudio/` (2165 LOC), `wrapper/splatter.py`
  (747 LOC), both fork pins, nerfstudio entry points, their tests.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| gsplat version | git source `nerfstudio-project/gsplat` at commit `90d7b4b` (main, 2026-08-20, version string 1.6.0), `no-build-isolation` | No v1.6.0 release tag exists; latest tag v1.5.3 (2025-07-04) lacks `gsplat.losses`, fast 3DGS kernel (+31k CUDA lines since), `GaussianScene`. Pin is a SHA; bump is one line. |
| Primitive | `primitive: 3dgs \| 2dgs`, default `3dgs` | 3DGS has the fast kernel (Jun-2026 perf pass, ~30%); normals for 3DGS are rendered via `extra_signals` (below), so normal consistency works on both. 2DGS kept for its distortion loss + native surfel normals. One trainer, branch only at the rasterizer call. |
| Losses | photometric (always on, `0.8·L1 + 0.2·SSIM`), optional `depth`, `normal_consistency`, `distortion`, each a plain `{weight, start}` dict | `depth`/`distortion` upstream-proven. `normal_consistency` on 3DGS is composed from package pieces (not upstream-validated — owed measurement). `distortion` is 2DGS-only → `ValueError` under 3dgs. No `end`; no ramps. |
| 3DGS normals | per-Gaussian normal = shortest scale axis rotated by quat (what `rade_gs.py:74` did), rendered as 3 `extra_signals` channels of `rasterization()` (`meta["render_extra_signals"]`); depth normal via `gsplat.utils.depth_to_normal(depth, c2w, K)`; loss `normal_cosine_loss(render_normal, depth_normal * alpha.detach())` | No fused kernel (unlike RaDe-GS); every piece is in the installed package. 2DGS uses `normals_rend` / `normals_surf` from the rasterizer directly. |
| TV loss | not included | In nerfstudio/NeRF land TV regularizes grid-shaped *parameters* (hash grid, planes, bilateral grid); Gaussians have no grid. Only splat use is the bilateral grid (skipped — view-dependent exposure breaks RGB/depth handoff) or edge-aware depth TV (not upstream-validated). Do not re-propose without a measured need. |
| Depth targets | COLMAP `points3D` projected through tracks (upstream `datasets/colmap.py` logic), `gsplat.losses.depth_l1_loss` | Every backend writes points3D via `build_colmap`; one init + one depth-target path. Dense feedforward depth prior deferred. |
| Pose opt | vendored `CameraOptModule` (`examples/utils.py`, not in installed package; verified no package-native pose opt at main), `pose_opt: bool`, lr 1e-5, wd 1e-6 | Refined poses written to `splats.zarr`; pointcloud COLMAP untouched (no-invalidate contract like `refine`). |
| Intrinsics | read from COLMAP; one camera → shared K automatically; not optimized | gsplat has no differentiable K. |
| Quality flags | `antialiased=True`, `absgrad=True` (`grow_grad2d=8e-4` per upstream note), `random_bkgd=True`, `sh_degree=3` — hardcoded | Upstream-proven improvements; not yaml knobs. |
| Off | `app_opt` (view-dependent exposure breaks RGB/depth handoff), MCMC, bilateral grid + TV, compression, distributed, holdout split, LPIPS | Not needed for train-view → mesh use case. Ablation tooling belongs in a later `evals/scripts/eval_splats.py`. |
| Eval separation | none in stage; PSNR/SSIM over train views only | Pipeline only renders train views. Caveat: train PSNR cannot detect pose-opt drift — eval script catches it. |
| Output handoff | `mesh.source: feedforward \| splats` | `splats` feeds rendered depth+RGB+poses+K from `splats.zarr` (all mutually consistent, native res). |
| PAGaS | follow-on spec | Needs gsplat pinned at `bd64a47` (=1.4.0) + 865-line CUDA patch; incompatible in-process with main. Seam: `_render()` in `trainer.py`. |
| Feature distillation (rade_features) | follow-on spec | Out of scope for the trainer. |

## Module layout

```
collab_splats/splats/
  __init__.py
  trainer.py    # SplatsConfig + train(cfg, scene_dir, out_dir) -> dict   (~300 LOC)
  losses.py     # LossSpec + compute_losses(step, out, batch, specs)     (~60 LOC)
  cameras.py    # CameraOptModule, vendored + attributed                  (~40 LOC)
```

### `trainer.py`

Sections (`########` dividers):

- `SplatsConfig` dataclass: `primitive`, `max_steps`, `pose_opt`, `losses: dict[str, dict]` (each `{weight, start?}`, straight from yaml).
  Upstream defaults as module constants (per-param lrs, `DefaultStrategy(refine_start_iter=500,
  refine_stop_iter=15000, reset_every=3000, refine_every=100, prune_opa=0.05,
  grow_grad2d=8e-4, absgrad=True)`, `sh_degree_interval=1000`, `ExponentialLR` on means
  to 0.01× over `max_steps`). `__post_init__` validates: unknown loss name, 2dgs-only loss
  under 3dgs.
- `# Load scene` block inside `train()` (no function — existing accessors do the work):
  `PointcloudResult.extrinsics/.intrinsics/.points/.colors`, `FrameStore.images()`; new code is
  only per-frame sparse `(pixel_xy, depth)` targets from point2D↔point3D tracks (~15 LOC) and
  upstream world-space normalization (similarity `T_norm`, stored, inverted at export, ~10 LOC).
  All tensors resident on GPU (scenes ≤ ~300 frames).
- `_create_splats_with_optimizers(points, rgbs, scene_scale)`: copied from
  `examples/simple_trainer.py` (attributed), knn scale init, SH0 from RGB, Adam per param
  group.
- `_render(primitive, splats, c2w, K, H, W, sh_degree)`: 3dgs →
  `rasterization(render_mode="RGB+ED", antialiased=True, absgrad=True,
  extra_signals=gaussian_normals)` then `depth_to_normal`; 2dgs → `rasterization_2dgs(...)`.
  Returns `(rgb, alpha, depth, normal, depth_normal, distort|None, info)`. Only function
  touching rasterizers.
- `train(cfg, scene_dir, out_dir) -> dict[str, Tensor]`: upstream loop — random frame,
  pose delta if `pose_opt`, render, `compute_losses`, `strategy.step_pre_backward` /
  `step_post_backward`, optimizer steps, schedulers. `progress()` wrapper. Logs active-loss
  set when a window opens. Ends with `_save`.
- `_save(out_dir, splats, pose_module, ...)`: renders every train view at native resolution,
  un-normalizes, writes the outputs below.

### `losses.py`

```python
def compute_losses(step, out, batch, losses: dict[str, dict], scene_scale) -> tuple[Tensor, dict[str, float]]
```
Photometric always (`0.8·l1_loss + 0.2·ssim_loss`). Optional loss active iff
`name in losses and weight > 0 and step >= start` (`start` defaults 0). `depth`: `depth_l1_loss`
on sparse targets (disparity space, `scene_scale`). `normal_consistency`:
`normal_cosine_loss(normal, depth_normal * alpha.detach())` — same form for both primitives.
`distortion`: `distort.mean()`. Returns total + per-loss floats for logging/report.

### `cameras.py`

`CameraOptModule` + `rotation_6d_to_matrix`, copied from
`nerfstudio-project/gsplat@90d7b4b examples/utils.py` (line range cited in file).
`nn.Embedding(n, 9)` zero-init, `c2w @ delta`.

## Config surface

```yaml
splats:
  enabled: false
  primitive: 3dgs            # 3dgs | 2dgs
  max_steps: 30000
  pose_opt: false
  losses:                    # photometric (0.8 L1 + 0.2 SSIM) always on; entries optional, {weight, start}
    depth: {weight: 0.01}                              # sparse COLMAP points, depth_l1_loss
    normal_consistency: {weight: 0.05, start: 7000}    # rendered normal vs depth_to_normal
    distortion: {weight: 0.0, start: 3000}             # 2dgs only
mesh:
  source: feedforward        # feedforward | splats
```

Unlisted loss = off. `nerfstudio:` block deleted from `configs/base.yaml` and
`configs/README.md`.

## Outputs — `<backend>/splats/`

```
splats.ply                   export_splats(format="ply"), COLMAP world frame
ckpt.pt                      {"splats": params, "pose_adjust": state|None, "config": asdict(cfg), "step"}
splats.zarr
  rgb     (N,H,W,3) uint8    rendered train views, native resolution
  depth   (N,H,W)   float32  expected depth (3dgs) / median depth (2dgs), COLMAP scale
  normal  (N,H,W,3) float32  rendered Gaussian normal (both primitives)
  alpha   (N,H,W)   float32
  c2w     (N,4,4)   float32  refined if pose_opt, else COLMAP verbatim
  K       (N,3,3)   float32
  attrs: image_ids, primitive, pose_opt, gsplat_commit, config
splats_quality_report.json
  summary:   psnr, ssim, n_gaussians, steps, seconds, final per-loss values, config
  per_frame: psnr, ssim          (nan → null)
```

## Pipeline wiring (`wrapper/reconstructor.py`)

- `_STAGE_DEPS["splats"] = ["pointcloud"]` — leaf stage, re-runnable from
  `environments-processed` via `--stages splats`. Marker: `splats/splats.zarr`.
- `Reconstructor.splats(overwrite)` (~40 LOC, mirrors `verify`): resolves result, calls
  `train`, returns `Path`.
- `mesh.source`: `feedforward` keeps today's path. `splats` → `_run_tsdf_mesh` reads
  `(depth, rgb, c2w, K)` from `splats.zarr`; `conf_percentile` masks on `alpha`;
  `native_resolution` ignored (already native) with a log line. Mesh deps stay
  `["pointcloud"]`; `source: splats` with no `splats.zarr` → `ValueError`, never auto-run.
- `_VALID_METHODS` → `{"feedforward", "sfm"}`.
- `pyproject.toml`: gsplat source → upstream SHA; nerfstudio dependency, fork source, and
  `rade_gs_method` / `rade_features_method` entry points removed; `no-build-isolation-package`
  keeps `gsplat`.
- GCS push: `splats/` included by default (ply + zarr are the deliverables).

## Errors

- Unknown yaml key / loss name → `ValueError` at config.
- `primitive: 3dgs` with `distortion` weight > 0 → `ValueError`.
- `points3D` fewer than 100 → `ValueError` (pointcloud-stage knob, not ours).
- frames.zarr length ≠ COLMAP image count → `ValueError`.
- `mesh.source: splats` without `splats.zarr` → `ValueError`.
- `pose_opt` with BA/LC: allowed (poses are inputs; nothing written back).

## Testing

- `tests/splats/test_losses.py` — CPU, synthetic tensors: window gate (`start`), weight 0
  disables, 3dgs rejection, per-loss log dict.
- `tests/splats/test_trainer.py` — CUDA: synthetic 8-frame 64×64 scene with 200 points,
  `max_steps=50`, both primitives. Asserts all outputs exist, zarr schema, depth back in COLMAP
  scale, 3dgs rendered normals unit-length where alpha>0.5, `pose_opt: true` changes `c2w`, `pose_opt: false`
  leaves it byte-equal, `load_ply_to_splats(splats.ply)` round-trips.
- `tests/wrapper/test_splats_stage.py` — stage deps / LEAF_STAGES membership; `source: splats`
  missing-zarr error.
- Retirement: no `nerfstudio` import remains (`tests/test_cu121_migration.py`,
  `tests/test_models.py`, `tests/nerfstudio_methods/` deleted or rewritten);
  `python -m collab_splats.dashboard --smoke` still passes.

## Owed measurements (human-run, appended to a measured report)

- Tutorial scene (`data/tutorial/`): 3dgs vs 2dgs step time and `splats_quality_report` PSNR.
- 3DGS `normal_consistency` on/off: mesh visual + rendered-depth error (not upstream-validated).
- `mesh.source: splats` vs `feedforward` mesh on the same scene — visual + the 7-Scenes
  rendered-depth error via the existing out10/p90 harness (gate design deferred).
- `pose_opt` effect on BA-clean poses.

## Follow-ons (separate specs)

- `evals/scripts/eval_splats.py` — holdout split, LPIPS, config ablations.
- PAGaS depth refinement — separate env or kernel port; plugs at `_render`.
- Feature distillation (replaces `rade_features`).
- Dense feedforward depth / mono normal priors.
