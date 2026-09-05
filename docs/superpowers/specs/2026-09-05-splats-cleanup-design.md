# Splats module cleanup — design

**Date:** 2026-09-05
**Scope:** `collab_splats/splats/`, `tests/splats/`, `tests/test_cu121_migration.py`, `docs/splats.md`, `CLAUDE.md` tree line, `configs/base.yaml` comments.
**Branch:** `refactor/cu121-uv-migration`.
**Status:** approved design, awaiting plan.

## Goal

Reduce overengineering in the splats module: delete dead code, flatten the trainer, put each
concern in one file, and make every docstring say what a function does, what it takes, and
what it returns. Behaviour is preserved: any yaml that trains today trains the same model
afterwards (numerically equivalent, see Verification).

Frozen public surface: `SplatsConfig`, `train`, the `splats:` yaml block, and every artifact
(`splats.ply`, `ckpt.pt`, `splats.zarr`, `splats_quality_report.json`) byte-layout and key set.
Everything else in the package may be renamed or moved.

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
| 10 | `scaffold.py` inherits gsplat `Strategy` for nothing, carries a `verbose` flag, a CPU frustum fallback, its own lr-schedule code, and leaks `log_scales`/`visible_ids` through the render dict | Drop base class, flag, fallback, custom schedule; stash regulariser inputs inside `Scaffold.render` |

## File layout

```
collab_splats/splats/
  __init__.py   GSPLAT_COMMIT; re-exports SplatsConfig, train
  trainer.py    SplatsConfig (+ from_dict) and train()                     843 → ~250 lines
  gaussian.py   Gaussians: vanilla 3dgs/2dgs primitives                    new, ~200
  scaffold.py   ScaffoldConfig, ScaffoldMLPs, Scaffold, AnchorStrategy     774 → ~550
  losses.py     schedule, validation, compute_losses; loss fns; registry   ~350
  pgsr.py       plane / NCC helpers, select_near_views, render_neighbour   484 → ~420
  rendering.py  render_gaussians, render_all_views, write_outputs          rendering + outputs
  cameras.py    CameraOptModule, AppearanceModule, Corrections             cameras + appearance
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
unit: str                             # "gaussians" | "anchors", for the log line
n_primitives: int                     # len(params["means"]) | len(params["anchors"])

def render(self, cam_to_world, intrinsics, width, height, camera_id, step=None,
           render_normals=True, render_plane=False) -> tuple[dict, dict]
def pre_backward(self, step, info) -> None
def post_backward(self, step, info) -> None
def denormalize(self, center, scale) -> None
def export_gaussians(self, cam_to_world, intrinsics) -> dict
def checkpoint_extras(self) -> dict
```

| Method | Gaussians | Scaffold |
|--------|-----------|----------|
| `__init__(cfg, points, colors, scene_scale, n_views, device, *, knn=4, adam_eps=1e-15, lr_decay=0.01)` | kNN log-scales, SH DC colour, random quats, `logit(init_opacity)`; one Adam per tensor; means lr × scene_scale with `ExponentialLR(gamma=lr_decay ** (1/max_steps))`; `make_strategy(cfg, n_views, *, prune_opa=0.1, prune_scale3d=0.5, refine_scale2d_stop_iter=4000)` | voxelise seed points, anchors/offsets/feat/scaling/rotation, per-tensor Adams, MLP heads; every group's schedule is a `LambdaLR` lambda `lr_final/lr_init ** min(step/lr_max_steps, 1)` (replaces `expon_lr`, `lr_schedule`, `update_learning_rate`); `knn`/`adam_eps`/`lr_decay` are not taken — ScaffoldConfig owns its rates |
| `render(...)` | activate params; `sh_degree = min(step // sh_degree_interval, sh_degree)`, `None` → full degree; `absgrad` from strategy | decode visible anchors through MLPs, render with `sh_degree=None`; writes `render["log_scales"]` and `render["opacities"]` for the regularisers; keeps `decode_index`, `visible_ids`, decoded opacities on `self` for `post_backward`; `step` ignored |
| `pre_backward(step, info)` | `strategy.step_pre_backward(...)` (DefaultStrategy retains the 2D-means gradient; MCMC no-op) | `info[key_for_gradient].retain_grad()` |
| `post_backward(step, info)` | `strategy.step_post_backward(...)`, passing `lr=` the current means lr for MCMC, `packed=False` for Default | `strategy.accumulate(...)` if inside the statistics window, then `strategy.refine(step)` (today's `step_post_backward`) |
| `denormalize(center, scale)` | `means / scale + center`, `scales -= log(scale)` | `anchors / scale + center`, `scaling -= log(scale)`; offsets are scale-free |
| `export_gaussians(cam_to_world, intrinsics)` | returns `params` (arguments unused) | today's `bake_anchor_gaussians`: decode under the mean observed view direction, unseen anchors from the nearest camera |
| `checkpoint_extras()` | `{}` | `{"mlps": state_dict, "voxel_size": float}` |

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
     `corrections = make_corrections(...)`; `near_ids = select_near_views(...)` when a PGSR
     loss is active (3dgs only, validated in losses).
  2. Loop per step: `ViewSampler` → `downscale_factor` / `downscale_view` / `prepare_target` →
     `c2w = corrections.camera(cam_to_world[id], id)` → `render, info = model.render(...)` →
     `render["rgb"] = corrections.colour(render["rgb"], id)` and `render["appearance"]` when
     appearance is on → random-background composite → `render["pgsr_neighbour"] =
     render_neighbour(...)` when PGSR is active → `model.pre_backward` → `compute_losses` →
     `loss.backward()` → step every optimizer in `model.optimizers + corrections.optimizers`
     and `zero_grad(set_to_none=True)` → step every scheduler → `model.post_backward` → log
     every `log_every` steps using `model.unit` / `model.n_primitives`.
  3. Finish: if normalised, `denormalize_cameras(cam_to_world, center, scale)`,
     `model.denormalize(center, scale)`, `corrections.pose.translation.weight /= scale` when a
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
- `AnchorStrategy` no longer subclasses gsplat `Strategy` and drops `verbose`. Methods:
  `initialize_state`, `should_accumulate`, `accumulate`, `refine(step)` (grow + prune on the
  refine cadence), `grow`, `prune(*, scale_cap=0.05)`, `_candidate_cells`, `_append_anchors`.
  `grow` is split into `_candidate_cells` (per-level voxel candidates not already occupied)
  and `_append_anchors`; numerics unchanged.
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
- `render_all_views(model, corrections, images, cam_to_world, intrinsics, store)` — one loop
  over views calling `model.render(step=None)` and `corrections.camera`/`colour`. Zarr arrays
  (`rgb` uint8, `depth`, `normal`, `alpha`, `median_depth` for 2dgs), chunking, `c2w`, `K`,
  per-frame psnr/ssim, and `n_decoded` are unchanged — `collab_splats/mesh/utils.py` reads
  this store.
- `write_outputs(cfg, model, corrections, images, cam_to_world, intrinsics, out_dir, seconds, final_losses)`
  — ply via `gsplat.export_splats(**model.export_gaussians(...))`; `ckpt.pt` with keys
  `splats`, `pose_adjust`, `appearance`, `config` plus `model.checkpoint_extras()`; zarr attrs
  and `splats_quality_report.json` schema identical to today.

### cameras.py (cameras + appearance)

- `rotation_6d_to_matrix`, `CameraOptModule`, `AppearanceModule` unchanged, citations kept.
- `@dataclass Corrections`: fields `pose: CameraOptModule | None`,
  `appearance: AppearanceModule | None`, `optimizers: list`, `schedulers: list`. Methods
  `camera(cam_to_world, ids)` (identity passthrough when `pose` is None) and
  `colour(rgb, ids)` (passthrough when `appearance` is None). Nothing else.
- `make_corrections(cfg, n_views, world_extent, scene_scale, lr_gamma, device, *, weight_decay=1e-6) -> Corrections`
  builds the pose module (rotation lr × world_extent, translation lr × scene_scale, Adam with
  `weight_decay`, `ExponentialLR(lr_gamma)`) and the appearance module as today.

### utils.py

`compute_scene_scale(cam_to_world, *, margin=1.1)`, `scene_normalization(cam_to_world)`,
`denormalize_cameras(cam_to_world, center, scale)`, `downscale_factor(step, num_downscales, resolution_schedule)`,
`downscale_view(image, intrinsics, factor)`, `prepare_target(image, depth, device)`,
`ViewSampler(n_views, *, seed=42)`. All moved from trainer.py; `denormalize_outputs` and
`denormalize_anchors` are replaced by `denormalize_cameras` + `model.denormalize` + the
one-line pose fix-up in the trainer.

### pgsr.py

- Module constants become keyword arguments:
  `select_near_views(..., *, theta0=5.0, sigma_below=1.0, sigma_above=10.0)`,
  `plane_depth(..., *, min_cosine=1e-4)`, `project(..., *, min_depth=1e-6)`.
- `pixel_rays` builds on `pixel_grid`; `_normalise_pixels` is inlined into `sample_at_pixels`
  so one function normalises pixel coordinates.
- New `render_neighbour(model, corrections, images, view_id, cam_to_world, intrinsics, factor, step) -> dict`
  returning `{"plane_depth", "gray", "world_to_cam", "intrinsics"}` — today's ~50-line block
  in the trainer loop. The trainer also sets `render["world_to_cam"]` and
  `render["intrinsics"]` for the current view as it does now.

## Constants rule

- A literal that tunes behaviour is a keyword argument with a default on the function that
  uses it, documented under `Args`. No module-level tunables, no defaults dicts, no registries
  of defaults. Full list: `train(min_points=100, lr_decay=0.01)`;
  `Gaussians(knn=4, adam_eps=1e-15)`; `make_strategy(prune_opa=0.1, prune_scale3d=0.5, refine_scale2d_stop_iter=4000)`;
  `make_corrections(weight_decay=1e-6)`; `AnchorStrategy.prune(scale_cap=0.05)`;
  `select_near_views(theta0=5.0, sigma_below=1.0, sigma_above=10.0)`;
  `plane_depth(min_cosine=1e-4)`; `project(min_depth=1e-6)`; `compute_scene_scale(margin=1.1)`;
  `ViewSampler(seed=42)`; `compute_losses(l1_weight=0.8, ssim_weight=0.2)`.
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
| `test_cameras.py` + `test_appearance.py` | `test_cameras.py` (adds `Corrections` passthrough tests) |
| `test_outputs.py` + `test_rendering.py` | `test_rendering.py` |
| `test_scaffold.py` | same file; `_frustum_anchors` tests deleted; visibility tests marked `cuda`; `expon_lr` tests become `LambdaLR` value checks at steps 0, mid, end |
| `test_losses.py` | same file; validation tests move here from `test_trainer.py` |
| `test_pgsr.py` | same file; add a `render_neighbour` key/shape test |
| `tests/test_cu121_migration.py` | module list → `splats`, `cameras`, `gaussian`, `losses`, `pgsr`, `rendering`, `scaffold`, `trainer`, `utils` |
| `tests/wrapper/*`, `tests/mesh/test_absent_confidence.py` | unchanged (`collab_splats.splats.trainer.train` patch target survives) |

New interface test: one parametrised test builds `Gaussians` and `Scaffold` on the synthetic
scene and asserts both expose every attribute and method in the interface table.

## Docs and config

- `docs/splats.md`: file table and data-flow paragraph rewritten for the new layout.
- `CLAUDE.md` architecture tree line:
  `splats/ # gsplat training: trainer, gaussian, scaffold, losses, rendering, cameras, utils`.
- `configs/base.yaml` `splats:` block: comments trimmed to one line per key; no key changes.
- `docs/superpowers/CHANGELOG.md`: one entry linking this spec and the plan, carrying the
  measured numbers removed from the trainer docstring (pose lr split, appearance_opt gains,
  distortion rescale, 2dgs absgrad).

## Error handling

Unchanged: every validation error raised today (`ValueError` from `from_dict` /
`validate_schedule`, point-count and per-view count checks in `train`, coincident cameras in
`scene_normalization`) is raised with the same message from its new home. No new error paths.

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

Parity check (Verification 1) runs after phases 3, 4 and 7.

## Out of scope

- Any change to `SplatsConfig` keys, defaults, or the yaml schema.
- Any change to artifact formats read downstream (`splats.zarr` by `mesh/utils.py`).
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
