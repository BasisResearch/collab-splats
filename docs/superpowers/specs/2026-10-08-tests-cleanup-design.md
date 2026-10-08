# Tests cleanup (conservative) — design

Date: 2026-10-08 · Branch: `clean/final` · Scope: A (conservative)

## Goal

Remove tests that no longer protect live behavior and consolidate satellites, with **zero loss of
live coverage**. Baseline: ~150 files, 44.5k lines, 4282 collected tests.

A test is removed only if it is one of:

- **dead target** — asserts on removed code / a removed field ("stays gone", tombstones, legacy-format reads)
- **finished migration gate** — pins a refactor or env migration that has landed (frozen pre-refactor copies, cu121 gates)
- **exact duplicate** — same function, same contract, covered by a named surviving test
- **vacuous / tautological** — cannot fail (golden computed by the code under test, `or True`, import of an empty module)

Everything else stays. Satellite files merge into their module's test file; misplaced files move to
mirror `collab_splats/`. Moved tests keep their function names.

## Non-goals (deferred follow-ups, scope B/C)

- trainer branch-check plant tests and scaffold name/constant plant tests (~63 cases)
- keyword-only signature tests (~14)
- `sfm/test_creator_args.py` collapse to one backend (−50)
- collapsing per-file params in `test_docstring_contract.py` (802) / `test_import_style.py` (538)
- `strict=True` on docstring-contract xfails
- insid3 private-helper trim; preproc overlap trims; `test_metrics.py:929` GPU-sync AST test;
  `test_metrics_controls.py` fold; BA seed reduction; shape-only / trivial-constructor tests
- building `tests/reconstructor/_stubs.py::_stub_reconstructor` from `configs/base.yaml`

## Per-deletion gate

Each cut below was found by reading code, not by coverage. Before deleting, re-grep: dead targets
must have zero hits in `collab_splats/`; duplicates must be compared against the named surviving
test. A candidate that fails the check **stays** and is reported.

## Commits (in order, each `git commit --only <paths>`)

### 1. `chore(tests): register gpu mark, deselect by default`

- `pyproject.toml` `[tool.pytest.ini_options]`: add `gpu` marker; `addopts` gains `-m "not gpu"`
- mark `tests/pointcloud/feedforward/test_mapanything_creator.py:304` (real-model download),
  `:321`, `:481` (need absent `/workspace/bicycle`) with `@pytest.mark.gpu`
- opt in with `-m gpu`; any explicit `-m` on the command line replaces the default (last `-m` wins),
  so `-m "not slow"` alone re-includes gpu tests — document in the marker description

### 2. `test(docs-contract): pointcloud joins RELEASED`

- `tests/test_docstring_contract.py:210` — add `"pointcloud"`; measured 80/80 XPASS, 0 failures

### 3. `test(geometry): …`

Delete:
- `loop_closure/test_pose_graph_incremental.py` — golden from `_helpers.drive_pose_graph` runs the
  same `add_submap`+`optimize` loop (tautology); move `:122` to `test_graph.py`
- `loop_closure/test_hw_formula.py` — pinned to one past bug; `:40` calls no `collab_splats` code
- `loop_closure/test_closure_split.py` + `tests/geometry/conftest.py` (only user) — dup of `test_graph.py:175`
- `loop_closure/test_loop_closure_integration.py` — each test duplicated (`test_loop_closure.py:52`,
  `test_graph.py:132`, `tests/pointcloud/test_feedforward_shared.py:56+`)
- `loop_closure/test_pgo_parity.py` — VGGT-SLAM parity (removed); move `:77,:83,:93,:186`
  (`calculate_pairwise_frame_scale`) to `test_graph.py`
- `test_tracks.py:637` + `tests/geometry/data/match_tracks_prototype.py.txt` — prototype parity
- `loop_closure/test_wrapper.py:119,:126` (`use_ba` absent), `:74,:82` (dup of BA config defaults)
- `test_bundle_adjustment.py:348` (dup of `:314`); `loop_closure/test_verify_threshold_resolution.py:16`
  (dup of `test_loop_closure.py:31`)

Move:
- `loop_closure/test_graph.py:20,:31,:41,:53,:193` (`decompose_camera`) → `test_transforms.py`
- `test_metrics.py:725,:731` (stage tables) → `tests/reconstructor/test_run.py`
- `loop_closure/test_wrapper.py:62` → `tests/pointcloud/test_base.py`; `:93` → `test_vggtx_creator.py`

### 4. `test(pointcloud): …`

Delete:
- `tests/integration/` — cu121 migration gate, every test covered elsewhere
- `feedforward/test_center_crop_coords.py` frozen "@ 1142edf8" copies (`:16-120`) and the
  `MAPANYTHING_GRIDS` sweep; keep `:163,:168`
- stays-gone checks: `test_vggtx_preproc.py:18,:24`, `test_vggtx_creator.py:166`,
  `test_feedforward_shared.py:39,:195`, `feedforward/test_mapanything_creator.py:474`,
  `test_feedforward_density.py:30`
- `test_registry.py:12,:16,:20` (dups of `test_*_is_feedforward_creator`, `test_loger_creator.py:775`)

Merge / move:
- → `test_base.py`: `test_zarr_attrs.py`, `feedforward/test_load_zarr_flags.py`,
  `test_feedforward_zarr.py`, `test_feedforward_reproject.py`
- → `test_feedforward_shared.py`: rest of `test_registry.py`, `test_mv_creator_wiring.py`,
  `test_feedforward_preprocess_store.py`
- → `test_vggtx_creator.py`: `test_cuda_guard.py`, rest of `test_vggtx_preproc.py`,
  rest of `test_feedforward_density.py`
- → `feedforward/test_mapanything_creator.py`: `test_lc_collate_window.py`
- → `test_loger_creator.py`: `feedforward/test_loger_load_guard.py`
- → `test_pointcloud_utils.py`: `test_utils_subsample.py`
- `test_pose_extraction.py` → `tests/geometry/loop_closure/test_submap.py`
- flat `test_{vggtx_creator,vggt_omega_creator,loger_creator,feedforward_*}.py` → `tests/pointcloud/feedforward/`

### 5. `test(splats): …`

- `test_trainer.py:240-290` depth_ratio cases — dups of `test_losses.py:430-449`; keep one wiring test
- `test_scaffold.py:1319`, `:1238` — dups of `test_model_interface.py:197`, `:153`; `test_gaussian.py:149` — dup of `:197`
- `splats.zarr`-absent asserts: `test_rendering.py:369`, `test_trainer.py:313`

### 6. `test(semantics,localization,preproc,utils): …`

Delete:
- `semantics/test_features_guards.py` — all dups (`utils/test_image.py:20-48`,
  `utils/test_torch_utils.py:21`, `semantics/test_semantics_utils.py:13-39`) or assertion-free
- `utils/test_utils_import_light.py` — vacuous; real check is `test_io.py:209`
- dead-target checks: `localization/test_retrieval.py:34`, `semantics/test_semantics_utils.py:98,:117`,
  `preproc/test_video.py:83`, `semantics/test_sky_segmentation.py:290`, `preproc/test_qa.py:697`

Merge:
- → `localization/test_localization_cache.py`: `test_decoupling_parity.py`, `test_provenance.py`,
  `test_reference_alignment.py`
- → `localization/test_localizer.py`: `test_intrinsics_seed.py`, `test_localization_result.py`
- `localization/test_viz_correspondences.py` + `test_viz_distribution.py` → `localization/test_viz.py`
- `semantics/test_features.py` → `test_query_api.py`
- `semantics/features/test_extract_from_zarr.py` → `test_extractor_preprocessing.py`

### 7. `test(reconstructor): …` (plus root tests and empty dirs)

Delete:
- `tests/test_bae_smoke.py` — subset of env gate
- `tests/test_semantics_logging.py` — log text only, has `or True` tautology (`:39`); behavior in
  `semantics/test_query_api.py:67,:75`
- tombstones `reconstructor/test_reconstructor.py:194`, `:234`
- run-order tests in `test_reconstructor.py:914-~1000` that duplicate `test_run.py`
- `tests/wrapper/`, `tests/nerfstudio_methods/`, `tests/examples/` (pycache only)

Merge / move:
- `tests/test_cu121_migration.py` → `tests/test_env.py`: keep gsplat pin (`:189,:209`), bae+cudss
  (`:61,:229,:284`), pycolmap API (`:237`), numpy≥2 (`:39`), torch/cuda (`:25,:32`); drop migration
  phase checks, version floors and the rotted MODULES sweep (`:88-135`, `:164`)
- `tests/test_visualization.py` + `tests/test_heatmap.py` → `tests/utils/test_visualization.py`
- `tests/test_feedforward_logging.py` → `tests/pointcloud/feedforward/test_logging.py`
- `reconstructor/test_reconstructor_mv_config.py` + `test_reconstructor_loger_kwargs.py` +
  `test_reconstructor_export.py` → `reconstructor/test_pointcloud_stage.py`
- `reconstructor/test_reconstructor_preprocess.py` → `test_preproc_stage.py`

### Wrap-up

- `docs/known-test-failures.md`: update paths touched by moves/deletes
- `docs/superpowers/CHANGELOG.md` entry; CLAUDE.md in-flight line added at start, removed at end

## Merge rules

- moved tests keep names; collisions keep the better name, noted in commit body
- imports lifted to the top (fixes inline-import violations in touched files)
- duplicated helpers fold into target file / existing conftest
- `isort` + `black` on touched files only, never repo-wide

## Verification

- **Baseline** before commit 1: full `pytest tests/` in tmux, logged; `--collect-only -q` ID list saved
- **Per commit:**
  - collected-ID diff = intended removals only; moves show the same function name under a new path
  - touched dirs pass, except entries in `docs/known-test-failures.md`
- **End:** full suite vs baseline — same failure set, no new failures
- expected: ~−115 tests, ~−27 files

## Risks

- duplicate claims from reading, not coverage → per-deletion gate above
- `clean/final` moves fast; other sessions share the index → `git commit --only`, land each commit promptly
- merges change fixture scope → run each merged file alone and in its directory
