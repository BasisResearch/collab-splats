# Evals release cleanup — design

Date: 2026-09-28 · Branch: `clean/evals-release` off `clean/final` · Status: approved in brainstorm

Rules: [017-release-cleanup-rules](../decisions/017-release-cleanup-rules.md) — linked, not restated.
Reference run: [preproc release cleanup](2026-09-24-preproc-release-cleanup-design.md).

## Goal

Cut `evals/` to what the ground-truth harness needs. Most files are one-offs from the
loop-closure, BA-ceiling and multiview-confidence investigations, which are closed. The
package now does the orchestration those scripts rebuilt by hand.

## Decisions (from brainstorm)

- **The pipeline is the pipeline.** `evals` never rebuilds creators, LC, BA or InstantSfM.
  It stages frames and runs `Reconstructor` with config overrides.
- **An A/B is a config path, not a script.** A condition is a named override block in the
  grid YAML (`{pointcloud: {loop_closure: true}}`, `{splats: {primitive: 2dgs}}`,
  `{mesh: {mask_sky: true}}`). No per-experiment scripts, no LC/BA CLI flags.
- **Reference-free metrics live in the package.** Each stage writes its own report
  (`video_quality_report`, `reconstruction_quality_report`, `splats_quality_report`); `evals`
  collects them. `evals` owns only ground-truth metrics: ATE, RPE, AUC@k and GT-depth error.
- **Mesh quality report is deferred.** It belongs to the mesh work in progress and gets its
  own brainstorm (e.g. reprojection PSNR). Until then a grid cell with the mesh stage reports
  no mesh numbers.
- **Finished investigations are docs, not scripts.** Results and a replication recipe live
  in `docs/`; the script is deleted and the doc cites the last commit that had it.
- **Datasets:** 7-Scenes, TUM RGB-D, CO3Dv2. KITTI, Waymo, bicycle and video go.
- **evo for ATE/RPE**, in memory (no TUM files); AUC@k stays ours (evo has none).
- **No `scripts/` dir.** One runner at `evals/eval.py`, run as `python -m evals.eval`;
  plain `from evals.x import ...` imports, no `sys.path.insert`.
- **Downloads are a README recipe**, not code: 7-Scenes and TUM are `wget` + `unzip`/`tar`,
  CO3Dv2 uses the official `facebookresearch/co3d` downloader.

## Target layout

```
evals/
  __init__.py
  README.md          # get data, run a grid, read the table (~40 lines)
  configs/*.yaml     # one per experiment: datasets x conditions (+ stages)
  datasets.py        # 7scenes | tum | co3dv2 -> EvalDataset (frames, GT poses, 7-Scenes GT depth)
  gt_metrics.py      # ate / rpe (evo), auc_at_threshold, depth_error
  eval.py            # grid runner
```

## Runner design (`evals/eval.py`)

- **Config:** `name`, `output_dir`, `stages` (default `[pointcloud]`), `datasets`
  (`{name, type, seq_dir, max_frames}`), `conditions` (`{label: override dict}`). The
  current `backbones` axis becomes an override (`{pointcloud: {backend: vggtx}}`).
- **Per cell** (dataset x condition), in its own subprocess (OOM isolation, as today):
  1. load the dataset, symlink its frames into `<cell>/input/` in GT order
  2. `Reconstructor(deep_merge(base, {input_path, output_path}, override)).run_pipeline(stages)`
  3. read predicted poses from `pointcloud.zarr`; map them to GT order through `frames.json`
  4. write `metrics.json`: GT metrics + a copy of every stage report found in the cell
- **Resume:** a cell with `metrics.json` is skipped (unchanged behavior).
- **Aggregate:** one `comparison.md` + `comparison.json` over all cells.
- **Plots:** trajectory and per-frame ATE PNGs stay (cheap, read by the GT notebook).
- **Fails loud:** unknown dataset type, empty condition set, a pose count that does not
  match the GT count, a stage that wrote no report the table expects.

## Round 1 — prose only

Only `datasets.py`, `configs/*.yaml` and `README.md` survive as files; the rest are deleted
or rewritten in round 2. Round 1 trims `datasets.py` docstrings/comments to 017, fixes the
stale `eval_suite.sh` comments in `configs/7scenes.yaml`, and rewrites `README.md`
(interpreter path, `from scripts.X`, Investigations B/C/D history, baselines tables all go).
Proof as 017: docstring-stripped AST equal for `datasets.py`, plus one sanity mutation.

## Round 2 — code, one commit per logical change

1. **Delete one-offs:** `scripts/{tri_angle_census, ba_start_at_gt, refit_at_fixed_poses,
   eval_run_backend, eval_multiview_conf, eval_similarity_calibration,
   eval_localization_parity}.py`, `pose_graph_diagnostics.py`, `envs/`,
   `data/extract_waymo.py`, and their tests.
2. **Untrack data:** `git rm --cached -r evals/results/chess_seq01 evals/baselines/` (already
   gitignored; files stay on disk).
3. **Datasets:** drop `_load_kitti`, `_load_waymo`, `_load_bicycle`, `_load_video`;
   `_load_co3dv2` raises instead of falling back to all annotations; hoist inline imports;
   `get_dataset` names the valid types on a miss.
4. **`gt_metrics.py`:** new; `ate`/`rpe` wrap evo `APE`/`RPE` on in-memory `PoseTrajectory3D`
   with Sim3 alignment; `auc_at_threshold` moved from `trajectory_metrics`; `depth_error`
   = median-scale alignment + error stats (from `eval_multiview_conf`'s helpers).
   Delete `trajectory_metrics.py`, `metrics.py`, `trajectory_io.py`.
5. **Runner:** rewrite `eval.py` per the design above at `evals/eval.py`; fold the
   `eval_compare` grid-table helpers in; delete `scripts/` (incl. `eval_compare.py`).
   Rewrite `configs/7scenes.yaml` and `cross_model_chess.yaml` to the override form.
6. **Splats / mesh scripts:** delete `eval_splats.py`, `analyze_splats.py`,
   `eval_sky_mask.py`. Splat PSNR/SSIM comes from `splats_quality_report`; GT depth from
   `gt_metrics.depth_error`; sky-mask A/B is `configs/sky_mask.yaml`.
7. **Download tooling:** delete `data/download_datasets.py`; README gains the recipe.

### Verdict table

Kept files, one row per name:

| name | verdict | why |
|---|---|---|
| `datasets.EvalDataset` | keep | loader contract; gains optional `gt_depth` |
| `datasets._load_7scenes` | keep | GT poses + depth |
| `datasets._read_tum_assoc` | keep | TUM loader helper |
| `datasets._read_tum_groundtruth` | keep | TUM loader helper |
| `datasets._load_tum` | keep | — |
| `datasets._load_co3dv2` | raise | silent fallback to all annotations |
| `datasets._load_kitti` | delete | dataset dropped |
| `datasets._load_waymo` | delete | dataset dropped |
| `datasets._load_bicycle` | delete | all-zero GT, ATE meaningless |
| `datasets._load_video` | delete | all-zero GT, ATE meaningless |
| `datasets._REGISTRY` | keep | 3 entries |
| `datasets.get_dataset` | keep | raise names valid types |
| `trajectory_metrics.umeyama_align` | delete | test-only; SE3 wrong for monocular |
| `trajectory_metrics.ate_translation` | delete | evo `APE` |
| `trajectory_metrics.rpe` | delete | evo `RPE` |
| `trajectory_metrics.auc_at_threshold` | merge | into `gt_metrics` |
| `eval_multiview_conf.load_7scenes_depth` | merge | into `datasets._load_7scenes` |
| `eval_multiview_conf.median_align`, `retained_error` | merge | into `gt_metrics.depth_error` |
| `eval._INSTANTSFM_CONDITIONS`, `_FIXED_CONDITIONS`, `_validate_condition` | delete | conditions are overrides |
| `eval._BACKBONE_PREFIX`, `_make_creator`, `_run_instantsfm`, `_run_condition` | delete | `Reconstructor` |
| `eval._cam_positions`, `_write_tum` | delete | evo / no TUM output |
| `eval._COLORS` | delete | colors by cycle, conditions are open-ended |
| `eval.EvalCell`, `EvalConfig`, `load_eval_config`, `build_grid` | keep | reshaped to dataset x override |
| `eval._default_output_dir` | delete | grid config names `output_dir` |
| `eval._prepare_image_dir` | keep | stage frames into `<cell>/input/` |
| `eval._save_outputs`, `_plot_trajectory`, `_plot_ate_per_frame` | keep | metrics + plots |
| `eval._build_parser`, `_subprocess_mode`, `_build_cell_command`, `_run_grid`, `main` | keep | `--config`, `--dry_run`, internal cell mode only |
| `eval_compare.collect_grid_metrics`, `format_markdown_rows`, `_metric` | merge | into `eval.py` |
| `eval_compare` (rest), `metrics.py`, `trajectory_io.py` | delete | standalone TUM mode unused |

Deleted whole files (no per-name rows): the seven one-off scripts in step 1,
`pose_graph_diagnostics.py`, `eval_splats.py`, `analyze_splats.py`, `eval_sky_mask.py`,
`download_datasets.py`, `extract_waymo.py`, `envs/vggt_long.yml`.

## Caller sweep

| caller | change |
|---|---|
| `CLAUDE.md`, `README.md` | `evals/scripts/eval.py` -> `python -m evals.eval` |
| `collab_splats/pointcloud/vda.py:89` | comment path |
| `docs/parity.md` | dead script refs -> commit citations |
| `docs/known-test-failures.md` | drop `pose_graph_diagnostics` and deleted-test entries |
| `docs/source/tutorials/evals/ground_truth_evals.ipynb` | `eval_gt.py` -> `evals.eval`; new result layout |
| `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` | `eval_gt.py` refs |
| `docs/source/tutorials/03_splats/train_splats.ipynb` | stop importing `eval_splats`; train via splats API (one cell) |
| `tests/pointcloud/{conftest, test_co3dv2_loader}.py` | import `evals.datasets` |
| `tests/geometry/conftest.py`, `tests/conftest.py` | drop the `evals` shadowing workaround if dead |
| sky-mask plan | A/B via `configs/sky_mask.yaml`; mesh numbers wait for the mesh report |

## Testing

- Baseline gate before any edit: `tests/evals tests/pointcloud tests/geometry tests/wrapper`,
  counts recorded; worktree `__file__` proof line.
- Tests of deleted code are deleted. New tests: `gt_metrics` (ATE zero on identical poses,
  Sim3-invariance, AUC on a known pair set, depth_error scale recovery), `datasets` (three
  loaders on fixtures, co3dv2 raise), runner (grid expansion, override merge, resume skip,
  frame-order mapping, pose-count raise) with `Reconstructor` stubbed.
- One live cell (7-Scenes chess, `vggt_omega`, baseline) in tmux; ATE compared to the last
  recorded number for that cell.
- `tests/test_docstring_contract.py` covers `evals/*.py` (it scans `collab_splats/` only
  today; add a root-relative source list).

## Out of scope

- Mesh quality report (separate brainstorm with the mesh work).
- Held-out split and LPIPS in `splats_quality_report`.
- `docs/benchmarks/scripts/` one-offs.
- Deleting untracked `evals/results/` (16 GB) and `evals/baselines/` (6.2 GB) on disk.
