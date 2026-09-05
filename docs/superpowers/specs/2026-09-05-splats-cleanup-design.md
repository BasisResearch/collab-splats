# Splats module cleanup — design

**Date:** 2026-09-05
**Scope:** `collab_splats/splats/`, `tests/splats/`, `tests/test_cu121_migration.py`, `docs/splats.md`, `CLAUDE.md` tree line, `configs/base.yaml` comments; plus the `splats.zarr` retirement, which touches `collab_splats/mesh/utils.py`, `collab_splats/wrapper/reconstructor.py`, `evals/scripts/analyze_splats.py`, `tests/mesh/test_splats_adapter.py`, two tutorial notebooks and `configs/README.md`.
**Branch:** `refactor/cu121-uv-migration`.
**Status:** approved design, awaiting plan.

## Goal

Reduce overengineering in the splats module: delete dead code, flatten the trainer, put each
concern in one file, and make every docstring say what a function does, what it takes, and
what it returns. Behaviour is preserved: any yaml that trains today trains the same model
afterwards (numerically equivalent, see Verification).

Frozen public surface: `SplatsConfig`, `train`, the `splats:` yaml block, `splats.ply` and
`splats_quality_report.json`. `ckpt.pt` becomes self-contained (gains cameras, drops
`pose_adjust`) and `splats.zarr` is retired — see "Retire splats.zarr". Everything else in the
package may be renamed or moved.

## Audit findings that drive the design

| # | Finding | Resolution |
|---|---------|-----------|
| 1 | `collab_splats/nerfstudio/` and `tests/nerfstudio_methods/` hold only `__pycache__`; zero tracked files | `rm -rf` both |
| 2 | `trainer.py` module docstring is 20 lines of dated measurements | 4-line docstring; measurements move to the CHANGELOG entry |
| 3 | Tunable literals scattered as module constants (`SCALE_CAP`, `THETA0`, `SIGMA_*`, `MIN_RAY_COSINE`, `MIN_DEPTH`) and inline (`0.8*l1 + 0.2*ssim`, `n_neighbors=4`, `eps=1e-15`, `weight_decay=1e-6`, `0.01 ** (1/max_steps)`, `1.1`, `seed=42`, `prune_opa=0.1`, `prune_scale3d=0.5`, `refine_scale2d_stop_iter=4000`, `min_points=100`) | Each becomes a keyword argument with a default on the function that uses it |
| 4 | Loss validation (`LOSS_SPEC_KEYS`, decay/range checks) lives in `trainer.py`, not `losses.py`; PGSR defaults repeated at call sites | Validation and defaults move into `losses.py`; `SplatsConfig.from_dict` delegates |
| 5 | `train()` is ~330 lines with 8 `if anchor_field is not None` branches, 4 optimizer groups stepped by hand, 2 near-duplicate denormalize helpers | Two model classes with one shared interface; trainer loop has zero representation branches |
| 6 | Scene geometry helpers, downscale, sampler, target prep sit in `trainer.py` | Move to `utils.py` |
| 7 | `losses.py` has `compute_losses` at the bottom, hardcoded l1/ssim weights, stale worktree path in a comment | Reorder: schedule + validation + `compute_losses` on top, loss functions below, registry at bottom |
| 8 | `cameras.py` (88 lines) and `appearance.py` (28 lines) are both per-camera learned corrections | `AppearanceModule` moves into `cameras.py`; `appearance.py` deleted |
| 9 | `rendering.py` and `outputs.py` split one concern (render a view / render all views and write) and `outputs.py` repeats the vanilla-vs-scaffold render branch | Merge into `rendering.py`; the model interface removes the branch |
| 10 | `scaffold.py` inherits gsplat `Strategy` for nothing, carries a `verbose` flag, a CPU frustum fallback, a passed-around `state` dict, its own lr-schedule code, and leaks `log_scales`/`visible_ids` through the render dict | Drop base class, flag, fallback, state dict, custom schedule; `Scaffold.render` puts regulariser inputs in `render` and densifier inputs in `info` |
| 11 | `splats.zarr` is a render cache of the model that `ckpt.pt` already holds: ~70 lines of writer, ~60 of reader in mesh, a cross-check in the reconstructor, a symlink trap, and a second place a stale model can hide | Retire it; ckpt carries cameras; mesh and evals render from the checkpoint |

## File layout

```
collab_splats/splats/
  __init__.py   GSPLAT_COMMIT; re-exports SplatsConfig, train
  trainer.py    SplatsConfig (+ from_dict) and train()                     843 → ~250 lines
  gaussian.py   Gaussians: vanilla 3dgs/2dgs primitives                    new, ~200
  scaffold.py   ScaffoldConfig, ScaffoldMLPs, Scaffold, AnchorStrategy     774 → ~550
  losses.py     schedule, validation, compute_losses; loss fns; registry   ~350
  pgsr.py       plane / NCC helpers, select_near_views, render_neighbour   484 → ~420
  rendering.py  render_gaussians, render_views, write_outputs, load_checkpoint   rendering + outputs
  cameras.py    CameraOptModule, AppearanceModule, PoseAndAppearance             cameras + appearance
  utils.py      scene scale/normalisation, downscale, target prep, sampler new, ~120
```

Deleted: `collab_splats/splats/outputs.py`, `collab_splats/splats/appearance.py`,
`collab_splats/nerfstudio/`, `tests/nerfstudio_methods/`.

Nine files today, nine after. The difference is that no file mixes concerns: the trainer holds
config and the loop, each representation holds its own initialisation, optimisation, rendering
and densification, and every helper has one home.

## Model interface

`Gaussians` (gaussian.py) and `Scaffold` (scaffold.py) expose the same attributes and methods.
No base class; the trainer calls them identically and tests assert the interface directly.

```python
params: torch.nn.ParameterDict        # tensors densification grows/prunes; saved as ckpt["splats"]
optimizers: list[torch.optim.Optimizer]
schedulers: list[torch.optim.lr_scheduler.LRScheduler]
n_primitives: int                     # len(params["means"]) | len(params["anchors"])

def render(self, cam_to_world, intrinsics, width, height, camera_id, step=None,
           render_normals=True, render_plane=False) -> tuple[dict, dict]
def pre_backward(self, step, info) -> None
def post_backward(self, step, info) -> None
def denormalize(self, center, scale) -> None
def export_gaussians(self, cam_to_world, intrinsics) -> dict
def checkpoint(self) -> dict

@classmethod
def from_checkpoint(cls, ckpt, device) -> Self
```

Six methods plus one constructor, three attributes. `pre_backward` / `post_backward` mirror
gsplat's `Strategy.step_pre_backward` / `step_post_backward` by name, so `render` stays a pure
render and the densifier's two hooks sit where an upstream reader expects them.

| Method | Gaussians | Scaffold |
|--------|-----------|----------|
| `__init__(cfg, points, colors, scene_scale, n_views, device, *, knn=4, adam_eps=1e-15, lr_decay=0.01)` | kNN log-scales, SH DC colour, random quats, `logit(init_opacity)`; one Adam per tensor; means lr × scene_scale with `ExponentialLR(gamma=lr_decay ** (1/max_steps))`; `make_strategy(cfg, n_views, *, prune_opa=0.1, prune_scale3d=0.5, refine_scale2d_stop_iter=4000)` | voxelise seed points, anchors/offsets/feat/scaling/rotation, per-tensor Adams, MLP heads; every group's schedule is a `LambdaLR` lambda `lr_final/lr_init ** min(step/lr_max_steps, 1)` (replaces `expon_lr`, `lr_schedule`, `update_learning_rate`); `knn`/`adam_eps`/`lr_decay` are not taken — ScaffoldConfig owns its rates |
| `render(...)` | activate params; `sh_degree = min(step // sh_degree_interval, sh_degree)`, `None` → full degree; `absgrad` from strategy | decode visible anchors through MLPs, render with `sh_degree=None`; writes `render["log_scales"]` and `render["opacities"]` for the regularisers; adds `decode_index`, `visible_ids`, `opacities` to `info`; `step` ignored |
| `pre_backward(step, info)` | `strategy.step_pre_backward(...)` for DefaultStrategy (retains the 2D-means gradient); nothing for MCMC, which has no pre-hook | `info[key_for_gradient].retain_grad()` |
| `post_backward(step, info)` | `strategy.step_post_backward(...)`, passing `lr=` the current means lr for MCMC, `packed=False` for Default | `strategy.accumulate(info)` (returns early outside the statistics window), then `strategy.refine(self, step)` (today's `step_post_backward`) |
| `denormalize(center, scale)` | `means / scale + center`, `scales -= log(scale)` | `anchors / scale + center`, `scaling -= log(scale)`; offsets are scale-free |
| `export_gaussians(cam_to_world, intrinsics)` | returns `params` (arguments unused) | today's `bake_anchor_gaussians`: decode under the mean observed view direction, unseen anchors from the nearest camera |
| `checkpoint()` | `{"splats": params}` | `{"splats": params, "mlps": state_dict, "voxel_size": float}` |
| `from_checkpoint(ckpt, device)` | rebuild `params` from `ckpt["splats"]`; no optimizers or strategy (render-only) | rebuild anchors and MLP heads from `ckpt["splats"]`, `ckpt["mlps"]`, `ckpt["voxel_size"]`; render-only |

**Ordering note for Scaffold.** Today `accumulate` runs between `backward()` and the optimizer
step, behind a comment saying the optimizers would zero the gradient. They would not: it reads
`info[key].grad` on a retained non-leaf tensor plus `info["radii"]`, `decode_index`, detached
opacities and `visible_ids`. `zero_grad(set_to_none=True)` touches parameter gradients only.
Moving `accumulate` after the optimizer step into `post_backward` is therefore order-safe and
lets both representations share one hook.

## Module details

### trainer.py

- `SplatsConfig`: fields, defaults and `__post_init__` unchanged. `from_dict` keeps the
  top-level key check, primitive/representation validation, the scaffold-block and
  sh-under-scaffold checks, and calls `losses.validate_schedule(losses, primitive)` for the rest.
- `train(cfg, images, world_to_cam, intrinsics, points, colors, out_dir, depth_targets=None, *, min_points=100, lr_decay=0.01)`:
  1. Setup: point-count and per-view checks; `compute_scene_scale`; optional
     `scene_normalization` of cameras, points, depth targets and the distortion weight
     (3-line dict update, inline as today); `model = Gaussians(...)` or `Scaffold(...)`;
     `refine = PoseAndAppearance.from_config(...)`; `near_ids = select_near_views(...)` when a PGSR
     loss is active (3dgs only, validated in losses).
  2. Loop per step: `next(views)` → `downscale_factor` / `downscale_view` / `prepare_target` →
     `c2w = refine.camera(cam_to_world[id], id)` → `render, info = model.render(...)` →
     `render["rgb"] = refine.colour(render["rgb"], id)` and `render["appearance"]` when
     appearance is on → random-background composite → when PGSR is active, downscale and
     pose-correct the neighbour view with the same two calls and set
     `render["pgsr_neighbour"] = render_neighbour(model, image, c2w, K)` →
     `model.pre_backward(step, info)` → `compute_losses` → `loss.backward()` → step every
     optimizer in `model.optimizers + refine.optimizers` and `zero_grad(set_to_none=True)` →
     step every scheduler → `model.post_backward(step, info)` → log every `log_every` steps
     with `type(model).__name__` and `model.n_primitives`.
  3. Finish: if normalised, `denormalize_cameras(cam_to_world, center, scale)`,
     `model.denormalize(center, scale)`, `refine.pose.translation.weight /= scale` when a
     pose module exists; then `write_outputs(...)`.
- Module docstring: four lines — what the module trains, the two axes
  (`primitive` × `representation`), inputs, outputs. The dated measurements move to the
  CHANGELOG entry for this cleanup.

### gaussian.py

- `SH_C0 = 0.5 / math.sqrt(math.pi)` with a block comment stating the convention
  `rgb = SH_C0 * sh0 + 0.5` (zeroth-order spherical-harmonic basis; a mathematical constant,
  not an option). Replaces `SH_DC_NORMALISER`; scaffold.py imports it.
- `make_strategy` moves here from trainer.py. 3dgs → `MCMCStrategy(cap_max)`; 2dgs →
  `DefaultStrategy` with the same arguments as today, the three literals now keyword arguments.
- `Gaussians.activate()` replaces `rendering.activate_vanilla`; `Gaussians.render` replaces
  `rendering.render_view`.

### scaffold.py

- `ScaffoldConfig` fields unchanged (yaml frozen); each field comment cut to one line.
- `VIEW_DIM` deleted; MLP input width written as `feat_dim + 3` with the block comment
  `# anchor feature + unit view direction (xyz)`.
- `Scaffold` (renamed from `AnchorField`) gains the interface methods above.
  `visible_anchors` uses `fully_fused_projection` only; `_frustum_anchors` is deleted and the
  tests that exercised the CPU path are marked `cuda`.
- `AnchorStrategy` no longer subclasses gsplat `Strategy` and drops `verbose`. With no base
  class there is no reason to pass a `state` dict around: the strategy owns its accumulators
  (`grad_accum`, `denom`, `opacity_accum`, `visit_count`) as attributes, so `initialize_state`
  and `should_accumulate` are deleted (the window check is an early return in `accumulate`).
  Methods: `accumulate(info)`, `refine(scaffold, step)` (grow + prune on the refine cadence),
  `grow`, `prune(*, scale_cap=0.05)`, `_candidate_cells`, `_append_anchors`. `grow` is split
  into `_candidate_cells` (per-level voxel candidates not already occupied) and
  `_append_anchors`; numerics unchanged.
- `Scaffold.export_gaussians` is today's `outputs.bake_anchor_gaussians`.

### losses.py (top to bottom)

1. `default_losses(primitive)` — moved from trainer.py unchanged.
2. `validate_schedule(losses, primitive)` — unknown loss name, unknown spec key, missing
   weight, weight/end_weight positive and numeric (bool rejected), `end > start`, distortion
   2dgs-only, `depth_ratio` numeric in [0, 1] and 2dgs-only, PGSR losses 3dgs-only.
3. `loss_weight`, `loss_active` — unchanged.
4. `compute_losses(...)` keeps its current positional arguments and adds `*, l1_weight=0.8, ssim_weight=0.2` replacing the inline `0.8 * l1 + 0.2 * ssim`.
5. `########` divider, then the eight loss functions with their current names and signature
   `(render, target, gaussians, scene_scale, spec)`. Yaml-tunable options stay as
   `spec.get("geo", 0.03)` at the use site, listed in the `Args` entry for `spec`.
6. `########` divider, then `OPTIONAL_LOSSES` and `LOSS_SPEC_KEYS` as two plain dicts,
   moved from trainer.py. They follow the function definitions; `validate_schedule` and
   `compute_losses` look them up at call time.
7. The comment citing `.worktrees/streaming/.../rade_gs.py:254` is replaced by a citation
   of the upstream RaDe-GS repository file and line.

### rendering.py (rendering + outputs)

- Keep `gaussian_normals_in_camera_frame` and `render_gaussians(primitive, decoded, cam_to_world, intrinsics, width, height, sh_degree, absgrad, render_normals, render_plane)`.
- `render_views(model, appearance, cam_to_world, intrinsics, height, width)` — a generator
  yielding one render dict per view (`rgb`, `depth`, `alpha`, `normal`, `median_depth` for
  2dgs) from `model.render(step=None)` with the appearance affine applied. Cameras are
  already pose-corrected. Used at train end for the quality report and by mesh / evals after
  `load_checkpoint`.
- `write_outputs(cfg, model, refine, images, image_ids, cam_to_world, intrinsics, out_dir, seconds, final_losses)`
  — ply via `gsplat.export_splats(**model.export_gaussians(...))`; `ckpt.pt` as below;
  `splats_quality_report.json` schema identical to today, its per-frame psnr/ssim computed
  from `render_views` in memory.
- `load_checkpoint(path, device) -> tuple[model, appearance | None, cam_to_world, intrinsics, image_ids, (height, width)]`
  — `torch.load`, `SplatsConfig.from_dict(ckpt["config"])`, `Gaussians.from_checkpoint` or
  `Scaffold.from_checkpoint` by `config["representation"]`, `AppearanceModule` from its state
  when present.

`ckpt.pt` keys after this change:

| Key | Value |
|-----|-------|
| `splats` | `model.params`, cpu tensors |
| `config` | `SplatsConfig` as a yaml-shaped dict |
| `cam_to_world` | `(N, 4, 4)` float32, pose-corrected, world units |
| `intrinsics` | `(N, 3, 3)` float32 |
| `image_ids` | `list[str]` |
| `image_size` | `(height, width)` |
| `appearance` | `AppearanceModule.state_dict()` or `None` |
| `mlps`, `voxel_size` | scaffold only |

`pose_adjust` is dropped: its deltas are folded into `cam_to_world`, and nothing resumes
training from a checkpoint.

### cameras.py (cameras + appearance)

- `rotation_6d_to_matrix`, `CameraOptModule`, `AppearanceModule` unchanged, citations kept.
- `@dataclass PoseAndAppearance`: fields `pose: CameraOptModule | None`,
  `appearance: AppearanceModule | None`, `optimizers: list`, `schedulers: list`. Methods
  `camera(cam_to_world, ids)` (identity passthrough when `pose` is None) and
  `colour(rgb, ids)` (passthrough when `appearance` is None). Nothing else. It exists to
  replace eight `if x is not None` sites and four extra function parameters with one object.
- `PoseAndAppearance.from_config(cfg, n_views, world_extent, scene_scale, lr_gamma, device, *, weight_decay=1e-6)`
  builds the pose module (rotation lr × world_extent, translation lr × scene_scale, Adam with
  `weight_decay`, `ExponentialLR(lr_gamma)`) and the appearance module as today.

### utils.py

`compute_scene_scale(cam_to_world, *, margin=1.1)`, `scene_normalization(cam_to_world)`,
`denormalize_cameras(cam_to_world, center, scale)`, `downscale_factor(step, num_downscales, resolution_schedule)`,
`downscale_view(image, intrinsics, factor)`, `prepare_target(image, depth, device)`,
`view_order(n_views, *, seed=42)` — a generator replacing the `ViewSampler` class: shuffle
`range(n_views)` with `random.Random(seed)`, `yield from reversed(order)`, repeat; the
sequence is identical to today's shuffle-and-pop. All moved from trainer.py; `denormalize_outputs` and
`denormalize_anchors` are replaced by `denormalize_cameras` + `model.denormalize` + the
one-line pose fix-up in the trainer.

### pgsr.py

- Module constants become keyword arguments:
  `select_near_views(..., *, theta0=5.0, sigma_below=1.0, sigma_above=10.0)`,
  `plane_depth(..., *, min_cosine=1e-4)`, `project(..., *, min_depth=1e-6)`.
- `pixel_rays` builds on `pixel_grid`; `_normalise_pixels` is inlined into `sample_at_pixels`
  so one function normalises pixel coordinates.
- New `render_neighbour(model, image, cam_to_world, intrinsics) -> dict` returning
  `{"plane_depth", "gray", "world_to_cam", "intrinsics"}` for one already-downscaled,
  pose-corrected neighbour view — today's ~50-line block in the trainer loop. The trainer
  also sets `render["world_to_cam"]` and `render["intrinsics"]` for the current view as it
  does now.

## Retire splats.zarr

`splats.zarr` stored one render per training view (`rgb`, `depth`, `normal`, `alpha`,
`median_depth`, `c2w`, `K`) so the mesh stage could fuse without touching the model. The
checkpoint already holds the model; with cameras added it holds everything the zarr did, and
rendering 300 views takes seconds on the GPU the splats stage already needs. Two artifacts
that describe one model is one too many, and the zarr is where a stale render could disagree
with the checkpoint it came from.

Changes:

- **splats**: `write_outputs` stops writing the store; `ckpt.pt` gains `cam_to_world`,
  `intrinsics`, `image_ids`, `image_size` and drops `pose_adjust`. `render_views` and
  `load_checkpoint` (rendering.py) are the public way to get renders back.
- **mesh** (`collab_splats/mesh/utils.py::_splats_to_tsdf_inputs`): takes the `ckpt.pt` path,
  calls `load_checkpoint` then `render_views`, and builds the same `(depths, rgbs, c2w, intrinsics)`
  tuple it builds today, picking `depth` or `median_depth` from each render dict by
  `mesh.splat_depth` and using render `alpha` where it used the zarr `alpha`. Streams one view
  at a time; never holds all renders.
- **reconstructor**: `mesh.source: splats` looks for `splats/ckpt.pt`; the view-count
  cross-check against `pointcloud.zarr` is deleted (the checkpoint carries its own cameras).
  `run_splats` returns the checkpoint path where it returned the zarr path.
- **evals** (`evals/scripts/analyze_splats.py`): `analyze_normals` renders from the checkpoint
  with the same two calls instead of reading arrays.
- **docs**: `docs/splats.md` output table; tutorials `03_splats/train_splats.ipynb` and
  `06_mesh/splats_mesh.ipynb` cells that read or describe the store; `configs/README.md`
  artifact list.
- **tests**: see the Tests table.

Not changed: `splats.ply` (external viewers), `splats_quality_report.json`, TSDF fusion itself.

## Constants rule

- A literal that tunes behaviour is a keyword argument with a default on the function that
  uses it, documented under `Args`. No module-level tunables, no defaults dicts, no registries
  of defaults. Full list: `train(min_points=100, lr_decay=0.01)`;
  `Gaussians(knn=4, adam_eps=1e-15)`; `make_strategy(prune_opa=0.1, prune_scale3d=0.5, refine_scale2d_stop_iter=4000)`;
  `PoseAndAppearance.from_config(weight_decay=1e-6)`; `AnchorStrategy.prune(scale_cap=0.05)`;
  `select_near_views(theta0=5.0, sigma_below=1.0, sigma_above=10.0)`;
  `plane_depth(min_cosine=1e-4)`; `project(min_depth=1e-6)`; `compute_scene_scale(margin=1.1)`;
  `view_order(seed=42)`; `compute_losses(l1_weight=0.8, ssim_weight=0.2)`.
- Mathematical facts stay named at module top and are derived, not typed: `SH_C0`. `VIEW_DIM`
  is deleted (it is the size of an xyz vector, inlined with a comment).
- `SplatsConfig` and `configs/base.yaml` gain no new keys.

## Docstring and comment rules

Applied to every public and private function and class in the nine files.

- `"""` on its own line; summary on the next line; blank line; `Args:` with one line per
  argument giving shape/dtype/unit where relevant; `Returns:` with one line per returned value.
  Where a function was ported, one final line: `Port of <repo>@<commit> <file>:<lines>`.
- Block comments state what a block does, with a short "why" clause only where the reason is
  not obvious from the code (e.g. `# rasterizer returns alpha-weighted normals: divide by alpha`).
  Dates, decibel numbers and experiment narratives are removed from code; the ones worth
  keeping go into the CHANGELOG entry.
- `########` dividers separate constants, helpers, classes and public entry points in each file.

## Tests

`tests/splats/` mirrors the new code tree. Assertions are unchanged unless the thing they
asserted was deleted; only imports and file placement move.

| Today | After |
|-------|-------|
| `test_trainer.py` | config/`train` tests stay; scene-scale, normalisation, downscale, sampler, target tests → `test_utils.py`; init, strategy, denormalize tests → `test_gaussian.py` |
| `test_cameras.py` + `test_appearance.py` | `test_cameras.py` (adds `PoseAndAppearance` passthrough tests) |
| `test_outputs.py` + `test_rendering.py` | `test_rendering.py`; zarr-layout tests replaced by ckpt-key and `load_checkpoint` round-trip tests (render from checkpoint matches render from the live model) |
| `tests/mesh/test_splats_adapter.py` | fixtures write a small `ckpt.pt` instead of a zarr store |
| `test_scaffold.py` | same file; `_frustum_anchors` tests deleted; visibility tests marked `cuda`; `expon_lr` tests become `LambdaLR` value checks at steps 0, mid, end |
| `test_losses.py` | same file; validation tests move here from `test_trainer.py` |
| `test_pgsr.py` | same file; add a `render_neighbour` key/shape test |
| `tests/test_cu121_migration.py` | module list → `splats`, `cameras`, `gaussian`, `losses`, `pgsr`, `rendering`, `scaffold`, `trainer`, `utils` |
| `tests/wrapper/*`, `tests/mesh/test_absent_confidence.py` | unchanged (`collab_splats.splats.trainer.train` patch target survives) |

New interface test: one parametrised test builds `Gaussians` and `Scaffold` on the synthetic
scene and asserts both expose the three attributes and six methods in the interface table.

## Docs and config

- `docs/splats.md`: file table and data-flow paragraph rewritten for the new layout.
- `CLAUDE.md` architecture tree line:
  `splats/ # gsplat training: trainer, gaussian, scaffold, losses, rendering, cameras, utils`.
- `configs/base.yaml` `splats:` block: comments trimmed to one line per key; no key changes.
- `docs/superpowers/CHANGELOG.md`: one entry linking this spec and the plan, carrying the
  measured numbers removed from the trainer docstring (pose lr split, appearance_opt gains,
  distortion rescale, 2dgs absgrad).

## Error handling

Every validation error raised today (`ValueError` from `from_dict` / `validate_schedule`,
point-count and per-view count checks in `train`, coincident cameras in `scene_normalization`)
is raised with the same message from its new home. Mesh errors move with the artifact: a
missing `ckpt.pt` raises the "run the splats stage first" `FileNotFoundError`;
`mesh.splat_depth: median` on a 3dgs checkpoint raises the same `ValueError` as today's
missing-array check, decided from `config["primitive"]` instead of a zarr key.

## Verification

1. **Parity baseline before any code change.** A scratchpad script trains 3dgs vanilla, 2dgs
   vanilla and 3dgs scaffold on `tests/splats/synthetic.py::make_scene` for 300 steps with a
   fixed seed and stores `splats_quality_report.json` and the ply means. After the refactor the
   same script runs again: PSNR within 1e-3 dB, means `allclose(rtol=1e-4)`. The only expected
   delta is ulp-level drift from `LambdaLR` replacing `expon_lr` (pow vs exp/log).
2. `/opt/venv/reconstruction/bin/python -m pytest tests/splats tests/wrapper/test_splats_stage.py tests/test_cu121_migration.py`
   green on the A40.
3. `black` and `isort` on touched files only; `graphify update .` after the last code change.

## Phases (one commit each)

0. Delete `collab_splats/nerfstudio/` and `tests/nerfstudio_methods/`; capture parity baseline.
1. `utils.py` and `cameras.py` merge — mechanical moves, tests re-pointed.
2. `losses.py` reorder, validation move, `compute_losses` kwargs.
3. `gaussian.py`, `Scaffold` interface, trainer loop rewrite, `rendering.py` merge.
4. Scaffold simplification: `LambdaLR`, drop CPU fallback / `Strategy` base / `verbose`, `grow` split.
5. `pgsr.py` tidy and `render_neighbour`.
6. Docstring and comment sweep; remaining literals to kwargs.
7. Tests, `docs/splats.md`, `CLAUDE.md`, `base.yaml` comments, CHANGELOG; graphify update.
8. Retire `splats.zarr` (section below): ckpt schema, `load_checkpoint`, mesh adapter,
   reconstructor, `analyze_splats.py`, notebooks, `configs/README.md`, tests.

Parity check (Verification 1) runs after phases 3, 4, 7 and 8 (phase 8 compares the mesh
adapter's `(depths, rgbs, c2w, intrinsics)` tuple from the checkpoint against the tuple the
zarr path produced from the same run, captured before the phase).

## Out of scope

- Any change to `SplatsConfig` keys, defaults, or the yaml schema.
- Any change to `splats.ply` or `splats_quality_report.json`.
- Tuning: no default changes, no new losses or strategies.
- `.worktrees/streaming/collab_splats/nerfstudio/` — a different branch, untouched.
- The other in-flight modules (preproc, semantics, pointcloud) have their own specs.

## Risks

- **Concurrent sessions on the same branch.** Commit with explicit paths
  (`git add -f docs/superpowers/...`, `git commit --only <paths>`) so foreign uncommitted work
  in the main tree is never swept in.
- **Hidden order dependence in Scaffold.accumulate.** Covered by the ordering note and by the
  parity check after phase 3.
- **Test imports outside `tests/splats/`.** `grep -rn "collab_splats.splats\." tests evals docs`
  runs in phase 7 to catch any import of a moved symbol.
