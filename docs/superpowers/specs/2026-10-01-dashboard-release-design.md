# Dashboard release — run the dashboard over `Reconstructor`

Date: 2026-10-01
Branch: `clean/dashboard-release` (worktree `.worktrees/dashboard-release`), rebased onto the ocr-lens work once it sits on `clean/final`
Follows: [semantics store cleanup](2026-10-01-semantics-store-cleanup-design.md) — its "dashboard follow-up" section is this spec

## Goal

- dashboard reads and runs the nested `Reconstructor` layout, nothing else
- compare backends on one preproc: `vggt_omega` vs `instantsfm` over the same `<scene>/images/`, switch in the local `SplitViewer`
- delete every dashboard copy of a pipeline stage; call existing functions, rewrite none
- new code is one 3-line helper (`_backend_config`) and a backend dropdown; everything else is deletion or a call swap

## Containment rule

- every edit lands in `collab_splats/dashboard/` or `tests/dashboard/`
- zero edits to `reconstructor.py`, `__main__.py`, `remote.py`, `semantics/`, `configs/` or any other package module
- the dashboard only calls public names: `Reconstructor` (+ `run_config_path`, `done`, `outputs`, `result`, `run`), the creator registries, `SceneSource`, `semantics.store.load_point_features`, `semantics.lifting.transfer_features`
- no private import from another package (`collab_splats.__main__._run_scene` etc.)

## Non-goals

- flat `{scene}/pointcloud.zarr` scenes — unsupported, no legacy checks
- localization tab — removed with `localize.py`
- viser `collab_splats/viewer.py` (ocr-lens mesh click/labels) — separate tool
- adding `dashboard` to the docstring-contract `PACKAGES` tuple

## Base branch

- needs `semantics/store.py`, `transfer_features`, the AE-inside-lifted-store layout — all on ocr-lens only
- `rebase/ocr-lens` (worktree `.worktrees/ocr-lens-rebase`) is ocr-lens rebased onto `clean/final`; gate 1761 passed / 0 failed
- `clean/dashboard-release` (this spec commit only) rebases onto it before implementation starts

## Layout read (from `Reconstructor`, never spelled in the dashboard)

- shared: `<out>/images/`, `<out>/semantics/<extractor>.zarr` (2D cache, local only — `PUSH_EXCLUDES` drops `/semantics/**`)
- per backend: `rec.pointcloud_zarr`, `rec.colmap_model_dir`, `rec.outputs["mesh"]`, `rec.outputs["semantics"]` (`<extractor>_lifted.zarr`, AE inside), `Reconstructor.run_config_path(out, backend)`
- `rec.done(stage)` is the only "exists" check

## Section 1 — deletions

| File | Fate |
|---|---|
| `dashboard/localize.py` | delete; `SceneCache` moves verbatim into `app.py` |
| `dashboard/shell.py` | delete; `run_app` serves `SplatsApp.view()` directly |
| `dashboard/config.py` | delete; `RunConfig`/`LocalizationConfig` go, `PULL_EXCLUDES` moves into `app.py` (its only reader) |
| `dashboard/pipeline.py` | keep `_push_async` only; delete `_write_images_dir`, `_build_creator`, `_extract_semantics`, `AutoencoderPolicy`, `semantics_ae_policy`, `resolve_latent_dim`, `resolve_semantics_dir`, `_lift_and_compress`, `_transfer_mesh_features`, `_sample`, `run_pipeline`, the whole localization block (`LocalizationRunOutput` … `run_localization`) |
| `dashboard/viewer.py` | delete `lift_point_features`, `_save_point_features`, `load_mesh_vertex_features` and their dead `semantics.utils` imports |
| `dashboard/app.py` | delete `_MODEL_CONF_DEFAULTS`, `_on_env_model`, `_current_config`, `_LIFT_MEMBERS`, `_ensure_lift_inputs`, `_cleanup_lift_inputs`, the `SplatsPage` alias, localization warm-up in `_warm_heavy_stack`; `_scan_output_dirs` gates on `rec.done("preproc")` instead of the flat `run_config.yaml` |
| tests | delete `test_localize_page.py`, `test_run_localization.py`, `test_localization_config.py`, `test_shell.py`, `test_config.py`, `test_viewer_lift.py`, `test_semantics_layout.py` |


## Section 2 — run path

The existing `_start_run` job keeps its shape; only its `run_pipeline(...)` call is replaced:

1. inputs: `self._ensure_local_video(scene)` as today; a pulled scene already has `images/` from load
2. config: `_backend_config(scene_dir, backend)` (Section 3) merged under `overrides` and `{input_path, output_path: base_dir/scene, pointcloud: {method, backend}}`; method is whichever registry holds `backend`
3. run, same three steps as the CLI:
   - `rec = Reconstructor(config)`
   - dump `rec.config` to `Reconstructor.run_config_path(output_path, backend)` (pushed scenes keep their config, as CLI runs do)
   - `rec.run(stages, overwrite=overwrite)`
4. push: `_push_async` as today
5. `on_done`: existing load dispatch, now for `(scene, backend)`

Widgets:

- scene dropdown (existing)
- backend dropdown: feedforward + sfm registry names
- stages multiselect: `STAGES` keys
- overrides textbox: a YAML mapping, `yaml.safe_load` → nested dict; no override parser
- force checkbox → `overwrite`

Notes:

- semantics off in overrides unless the run is for semantics

## Section 3 — load and viewer

Config for an existing backend (`_backend_config(scene_dir, backend)`, `app.py`):

- `Reconstructor.run_config_path(scene_dir, backend)` exists → its YAML; else `{}` (base.yaml defaults)
- reason: `outputs["semantics"]` is spelled from `semantics.extractor`; default config would point at the wrong lifted store when the run used another extractor

Backend selection (no discovery pass):

- backend dropdown lists every registered backend
- selecting one builds its `Reconstructor` (cheap: base.yaml merge + validation, no mkdir); loads when `rec.done("pointcloud")`, else status "`<backend>` not run — press Run"
- `done("pointcloud")` needs `pointcloud.zarr` + `colmap/sparse/0`; pulls carry both

Remote pull:

- `PULL_EXCLUDES` patterns get a `*/` prefix: dense zarr members skipped under every `<backend>/`
- `colmap/`, `<backend>/semantics/`, `<backend>/run_config.yaml`, `mesh.ply` still pulled
- pulled scene: lifted features viewable, new backends runnable (`images/` pulled); re-lifting needs a local semantics run (2D cache never travels)

Load:

- `rec.result` (lazy lean `load_zarr`), `rec.outputs["mesh"]`
- `SceneCache` key `(scene, backend)`
- point features: `rec.done("semantics")` → `load_point_features(rec.outputs["semantics"])`; else status "no lifted features for `<backend>` — run the semantics stage"
- `ensure_mesh_features` stays on demand; calls `transfer_features`

## Section 4 — error handling

- `GpuWorker` catches, `OperationLog` shows; `Reconstructor` errors pass through unchanged (`validate_config`, leaf already done)
- leaf-already-done message suggests the force checkbox
- overrides textbox not a YAML mapping → status line, run not started
- scene with no done backend: empty backend dropdown, status "no pointcloud — run the pointcloud stage"
- query without semantics: status line, not an exception
- remote missing: existing `SceneSource.check_available` disables pull/push
- no flat-layout detection, no shims

## Section 5 — testing

- fixture: fake scene tree — `<scene>/images/`, per backend `<backend>/pointcloud.zarr` + `colmap/sparse/0`; reuse reconstructor test helpers where present
- flat tests:
  - selection: done backend loads; not-run backend shows the status line
  - scene list: `_scan_output_dirs` keeps dirs where `done("preproc")`
  - `_backend_config`: recorded run_config wins, so `outputs["semantics"]` names the extractor that ran
  - `PULL_EXCLUDES`: dense members skipped under `<backend>/`; `colmap/`, `<backend>/semantics/`, `run_config.yaml` kept
  - `_start_run`: merged config carries `output_path`, `method`, `backend`; run_config written; `Reconstructor.run` monkeypatched
  - load: cache key `(scene, backend)`; features iff `done("semantics")`
- existing `test_app.py`, `test_viewer.py`, `test_pipeline.py` move onto the fixture, not rewritten
- containment gate: `git diff --name-only <base>..HEAD` lists only `collab_splats/dashboard/`, `tests/dashboard/`, docs
- gates: `pytest tests/dashboard`, `python -m collab_splats.dashboard --smoke` (mandatory pre-commit), `tests/test_import_style.py`
- manual acceptance on `/workspace/outputs/2026_07_15-Goprosplat-GH010229`:
  1. load `vggt_omega`
  2. run `instantsfm`, stages `pointcloud`, same `images/`
  3. switch dropdown, both render in `SplitViewer`
  4. semantics stage on one backend, query works
