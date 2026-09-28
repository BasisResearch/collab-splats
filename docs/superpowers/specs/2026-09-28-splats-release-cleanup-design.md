# Splats release cleanup — design

Date: 2026-09-28 · Branch: `clean/splats-release` off `clean/final` · Status: approved in brainstorm

Rules: [017-release-cleanup-rules.md](../decisions/017-release-cleanup-rules.md) — linked, not restated.
Reference run: [preproc release cleanup](2026-09-24-preproc-release-cleanup-design.md).

## Goal

Make `collab_splats/splats/` release-ready: delete the never-validated PGSR path, reuse package
helpers instead of local copies, split `rendering.py` by layer so no function imports inside
its body, and bring docs and `configs/base.yaml` to 017.

## Decisions (from brainstorm)

- **PGSR is deleted.** gsplat has no native plane rasterizer, so `render_plane` rebuilt plane
  depth from the rendered normals after the fact (`pgsr.plane_depth`); the gradient was an
  approximation, results looked poor, and no config ever enabled it. Re-entry is its own
  effort: native plane rasterization, GS-SR `566359be` parity, then a mesh A/B against the
  shipped 2dgs baseline. Backup ref `refs/backup/splats-release/pgsr` before the delete.
- **Checkpoint I/O becomes its own layer.** New `splats/checkpoint.py` holds `MODEL_CLASSES`,
  `REPRESENTATIONS`, `write_outputs`, `load_checkpoint`. Import order is one-way:
  `rendering` <- `gaussian`/`scaffold`/`cameras` <- `checkpoint` <- `trainer`.
- **Callers import from the package.** `from collab_splats.splats import load_checkpoint,
  render_views` — the module a name lives in is internal.
- **Reuse over local copies:** `geometry.transforms.invert_poses`,
  `geometry.transforms.rescale_intrinsics`, one kNN-spacing helper, one Scaffold decode helper.
- **Kept deliberately:**
  - `device = "cuda"` in `train` — gsplat is CUDA-only, and `get_device()` would fall back to CPU silently
  - `compute_scene_scale` — camera spread (gsplat parity), not `mesh.clean.get_scene_scale`'s vertex extent
  - `_frustum_anchors` — the documented CPU path for `visible_anchors`
  - `AnchorStrategy` `0.5` factors — cited upstream constants
- **No checkpoint break.** No state-dict key or config key is renamed. A checkpoint whose
  `config["losses"]` holds `pgsr_*` keys stops loading; none exist in `configs/`.
- **Scope:** every file in `collab_splats/splats/`, the `configs/base.yaml` `splats:` block,
  and every caller or test broken by the changes. No readability rewrite of `tests/splats/`.
- **Isolation:** worktree `.worktrees/splats-release`, `third_party/*` symlinked; every gate
  runs as `cd <wt> && PYTHONPATH=<wt> python ...` and prints `collab_splats.__file__`.

## Round 1 — prose only

No code change. Proof: AST equal after deleting every docstring statement on both sides and
stripping comments, plus one sanity mutation showing the check can fail. `base.yaml` edits are
comment-only; proof is `yaml.safe_load` equality.

### Per file

- `trainer.py` — module docstring to summary + bullets, measurement lore out
  ("~79x weaker pose opt", "2.3x over-densification", "65x … -1.6 dB"); `grow_grad2d` comment
  loses "GH010229 160k vs 1.4M, -1.26 dB".
- `gaussian.py` — `SH_C0` comment loses the "0 ULP vs the retired copy" history;
  `make_strategy` comment loses "measured-good".
- `rendering.py` — `render_views` docstring loses "~23 B/px, ~14 GB"; module docstring loses
  GS-SR history lines.
- `scaffold.py` — `_frustum_anchors` bullet to one fragment line.
- `cameras.py`, `utils.py`, `losses.py`, `__init__.py` — 017 pass (bullet caps, comment runs,
  header-then-bullets); no content change expected.
- `configs/base.yaml` `splats:` block — one comment line per key; `pose_opt` loses
  "+2.2 dB … +0.44 dB … +5% time"; `grow_grad2d` loses "8e-4 starved densification"; the
  5-line loss-schedule preamble to one line plus the `end`/`end_weight` example.

PGSR prose (the `base.yaml` recipe block, pgsr docstrings) is not edited in round 1; it leaves
with the code in round 2.

## Round 2 — code, one commit per logical change

### 1. Delete PGSR

- `pgsr.py` deleted (all 19 defs).
- `losses.py` — delete `neighbor_selection`, `pgsr_normal_loss`, `pgsr_multiview_loss`, their
  `OPTIONAL_LOSSES` entries and `LOSS_SPEC_KEYS` entries.
- `trainer.py` — delete the `select_near_views` / `render_neighbor` imports, `near_ids`, the
  neighbor pick (`random`), the `render["world_to_cam" | "intrinsics" | "pgsr_neighbor"]`
  wiring and the pgsr-under-2dgs rejection.
- `rendering.py` — delete `render_plane` from `render_gaussians`; the 4th extra channel is the
  zero pad; `gaussian_normals_in_camera_frame` drops the pgsr-only `means_cam` return.
- `gaussian.py`, `scaffold.py` — delete `render_plane` from `render`.
- `configs/base.yaml` — delete the PGSR recipe block.
- Tests: delete `tests/splats/test_pgsr.py` and the pgsr cases in `test_rendering`,
  `test_scaffold`, `test_gaussian`, `test_losses`, `test_trainer`; drop
  `collab_splats.splats.pgsr` from `tests/test_cu121_migration.py`.

### 1b. One 3dgs rasterization call

- With `render_plane` gone, the two 3dgs branches in `render_gaussians` (plain, normals) become
  one `rasterization` call with `extra_signals=None` when `render_normals` is off. Test: both
  settings return the same `rgb` / `depth` as before on a seeded scene.
- Considered and skipped (saves <= 3 lines each, or couples the models): shared per-param Adam
  builder, shared `denormalize`, shared `from_checkpoint` ParameterDict rebuild, a model base
  class — the interface stays duck-typed, pinned by `test_model_interface.py`.

### 2. Delete `absgrad`

- Always `False` at both callers: delete the param from `render_gaussians`, the literal in
  `Gaussians.render` and `Scaffold.render`. The strategy's `absgrad=False` stays (gsplat arg).

### 3. `checkpoint.py` split

- Move `MODEL_CLASSES`, `REPRESENTATIONS` (from `trainer.py`) and `write_outputs`,
  `load_checkpoint` (from `rendering.py`) into `splats/checkpoint.py`; the inline import in
  `load_checkpoint` goes.
- `rendering.py` keeps `render_gaussians`, `gaussian_normals_in_camera_frame`, `render_views`.
- `__init__.py` re-exports from `checkpoint`.
- Callers switch to `from collab_splats.splats import ...`:
  - `collab_splats/mesh/io.py:179` (and the `sys.modules` stub target in `tests/mesh/test_io.py`)
  - `evals/scripts/eval_splats.py:39`, `evals/scripts/analyze_splats.py:32`
  - `docs/source/tutorials/03_splats/train_splats.ipynb`
  - `tests/splats/test_trainer.py`, `test_rendering.py`, `test_model_interface.py`

### 4. Reuse `invert_poses`

- `trainer.py:244` `np.linalg.inv(world_to_cam)` -> `geometry.transforms.invert_poses`.

### 5. Reuse `rescale_intrinsics`

- `downscale_view` no longer scales K. `train` computes K once per downscale factor with
  `rescale_intrinsics(K, (H, W), (H // f, W // f))`, then moves it to the GPU.
- Behavior change: when H or W is not divisible by f, K now matches the resized image exactly
  instead of dividing by f. Test: non-divisible size, K principal point equals the
  resized-grid center.

### 6. One kNN-spacing helper

- `utils._knn_spacing(points, k)` returns per-point RMS distance to the k nearest neighbors.
  `Gaussians.__init__` uses it directly; `Scaffold` takes its median. `median_knn_spacing`
  deleted. Test: both callers bit-equal to the old code on a seeded cloud.

### 7. One Scaffold decode helper

- `Scaffold._offsets_to_gaussians(anchors, offsets, scaling, cov, keep)` returns means,
  scales, quats; `decode` and `export_gaussians` both call it. Test: `export_gaussians` output
  unchanged on a seeded model.

### 8. `Scaffold` `adam_eps` kwarg

- `Scaffold.__init__(..., adam_eps=1e-15)` replaces the hardcoded `eps=1e-15`, matching
  `Gaussians`.

### Verdict table

| name | verdict | why |
|---|---|---|
| `GSPLAT_COMMIT` | keep | pin, asserted by `test_cu121_migration` |
| `utils.compute_scene_scale` | keep | 2 callers; gsplat scene-extent proxy |
| `utils.scene_normalization` | keep | trainer; pairs with `denormalize_cameras` |
| `utils.denormalize_cameras` | keep | trainer + test; in-place inverse not obvious |
| `utils.downscale_factor` | keep | 2 callers |
| `utils.downscale_view` | merge | K scaling -> `rescale_intrinsics` |
| `utils.prepare_target` | keep | trainer + 2 external test callers |
| `utils.view_order` | keep | trainer |
| `utils._knn_spacing` (new) | merge | absorbs Gaussians + Scaffold kNN copies |
| `cameras.rotation_6d_to_matrix` | keep | vendored, attributed, readable unit |
| `cameras.CameraOpt` (all methods) | keep | model interface, all called |
| `trainer.PRIMITIVES` | keep | validation allow-list |
| `trainer.SplatsConfig` | keep | stage entry point |
| `trainer.train` | keep | pgsr wiring deleted |
| `checkpoint.MODEL_CLASSES`, `REPRESENTATIONS` | merge | moved from trainer |
| `checkpoint.write_outputs`, `load_checkpoint` | merge | moved from rendering; inline import gone |
| `gaussian.SH_C0` | keep | 2 modules |
| `gaussian.make_strategy` | keep | 1 caller, readable unit; `prune_*` documented kwargs |
| `gaussian.Gaussians` | keep | `render_plane`, `absgrad` literal deleted |
| `rendering.gaussian_normals_in_camera_frame` | keep | `means_cam` return deleted |
| `rendering.render_gaussians` | keep | `absgrad`, `render_plane` params deleted; 3dgs branches merged |
| `rendering.render_views` | keep | external callers |
| `losses.default_losses` … `appearance_reg_loss`, `OPTIONAL_LOSSES` | keep | live |
| `losses.LOSS_SPEC_KEYS` | keep | pgsr entries deleted |
| `losses.neighbor_selection`, `pgsr_normal_loss`, `pgsr_multiview_loss` | delete | pgsr only |
| `pgsr.*` (19 defs) | delete | never validated, never enabled |
| `scaffold.ScaffoldConfig` | keep | config |
| `scaffold.ScaffoldMLPs` | keep | state_dict keys in checkpoints |
| `scaffold._decay_lambda` | keep | 2 callers |
| `scaffold.median_knn_spacing` | merge | into `utils._knn_spacing` |
| `scaffold.voxelize` | keep | readable unit |
| `scaffold.Scaffold.__init__` | make-kwarg | `adam_eps` |
| `scaffold.Scaffold.visible_anchors`, `_frustum_anchors` | keep | documented CPU path |
| `scaffold.Scaffold.decode`, `export_gaussians` | merge | shared `_offsets_to_gaussians` |
| `scaffold.Scaffold.render` | keep | `render_plane`, `absgrad` deleted |
| `scaffold.AnchorStrategy` | keep | tests import it; 0.5 factors cited upstream |
| `base.yaml` PGSR block | delete | pgsr |
| `base.yaml` multi-line key comments | inline | one line per key |

## Caller sweep

Grepped over `collab_splats configs scripts tests evals docs/source`:

- `collab_splats/wrapper/reconstructor.py` — `SplatsConfig`, `train` from `splats.trainer`;
  unchanged (both stay in `trainer`).
- `collab_splats/mesh/io.py`, `evals/scripts/eval_splats.py`, `evals/scripts/analyze_splats.py`,
  tutorial `03_splats` — `load_checkpoint`, `render_views`; switch to the package import.
- `tests/wrapper/test_splats_stage.py`, `tests/evals/test_eval_splats.py` — `prepare_target`,
  `SplatsConfig`, `splats.trainer.train` patch target; unchanged.
- `tests/mesh/test_io.py` — stubs `collab_splats.splats.rendering` in `sys.modules`; retarget.

### Sibling branches (land order)

- `clean/evals-release` deletes `eval_splats.py` and `analyze_splats.py`; whichever lands second
  drops the edits to them.
- `clean/mesh-release` dissolves `mesh/io.py` and adds `render_tsdf_inputs` to
  `splats/rendering.py`, which calls `load_checkpoint`. Whichever lands second places
  `render_tsdf_inputs` in `splats/checkpoint.py`; in `rendering.py` it would re-create the cycle.
- `clean/tutorials` rewrites `train_splats.ipynb`; it adopts the package import on rebase.

## Testing

- Baseline gate recorded in the worktree before any edit (pass/fail/skip).
- Gate per commit, with the `__file__` proof line:
  `tests/splats tests/wrapper tests/mesh tests/semantics tests/evals tests/test_cu121_migration.py`.
  Never piped through `tail`, never `--tb=no`.
- Deleted-code tests are deleted; each behavior change (rescale K, shared kNN helper, shared
  decode helper) gets a test named above.
- `tests/test_docstring_contract.py` gains `splats` in `PACKAGES` and passes.
- Tutorial `03_splats` re-executed once at the end.
- `graphify update .` at the end. No merge; the user decides.

## Out of scope

- Readability rewrite of `tests/splats/` (dropped in the previous splats cleanup too).
- The `_field` -> `_scaffold` test rename and shrinking `test_trainer.py` (declined before).
- Any change to training behavior beyond the K rescale fix; no new tunables in `SplatsConfig`.
- `reconstructor.py` comments outside `splats/`.
- Re-implementing PGSR.
