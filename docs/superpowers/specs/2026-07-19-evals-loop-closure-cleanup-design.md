# Design: Evals Suite + Loop-Closure Aggressive Cleanup

- **Status:** Approved (supersedes the earlier conservative 7-section version of this file, 2026-07-19)
- **Date:** 2026-07-19
- **Branch:** `refactor/cu121-uv-migration` (source of truth). `runners/`→`scripts/` reorg included.

## Context

`evals/` grew to ~5,200 LOC. Three sweep drivers (`run_cross_model_benchmark.py`, `run_disparity_sweep.py`, `run_lc_parity.py`) independently reimplement one subprocess-matrix shape; two table-builders duplicate a glob→json→markdown render; library-grade code (SE3 inversion, LC diagnostics) sits in `evals/` with no eval-specific content; and a `bash` suite driver bundles the same `eval_gt` calls a YAML config could declare. Loop-closure validation is complete (`project_lc_parity_harness`: LC improves all backbones, precision 1.0), so the validation-era scaffolding can retire — its evidence is frozen in `evals/baselines/` + decision/findings docs. Separately, `collab_splats/geometry/loop_closure/closure.py` (894 lines) bundles three concerns VGGT-SLAM keeps in three files, and carries dead branches with zero production callers.

**Goal (user's converged intent):** one config-driven `eval.py` that runs a model with given parameters, compares across models, and compares across datasets; VGGT-SLAM parity documented (not a primary feature); loop-closure code cleaned and split so a user can read it. `pytest tests/` stays green throughout.

## The 4 target capabilities

1. **VGGT-SLAM parity** — historical, README-documented, re-runnable via one merged `run_vggt_slam.py`; its output TUM drops into a results dir and shows up as a comparison row. Not a bespoke harness.
2. **Run one model + params → metrics** — `eval.py --config <yaml>` (single cell).
3. **Compare across models** — same config, `backbones: [...]`.
4. **Compare across datasets** — same config, `datasets: [...]`.

## Non-goals

- **`GraphMap` persistence** — deferred, [ADR 003](../decisions/003-defer-graphmap.md).
- **Full `Solver`-class port** — borrow VGGT-SLAM method *names* for the per-window step, not its class structure.
- **Decoupling `LoopClosure` from feedforward creators** — only FF creators use it; LC is intrinsically FF-shaped. Revisit only if a non-FF creator needs LC.
- **`bundle_adjustment.py`** — separate module, out of scope.
- **Promoting `metrics.py`/`trajectory_io.py` into `collab_splats`** — evaluation tooling, correct where it is (evals/); zero `collab_splats` consumers.
- **Download output-dir convention unification** (`data/7scenes/` vs `evals/data/<dataset>/`) — changing a documented path is a behavior change; consolidate the *scripts*, not the paths.

## Guardrails for all new code

1. Grep for an existing helper before writing one.
2. Every new function: one-line docstring + block comments on the non-obvious parts (repo convention).
3. No abstraction without a second concrete use case. Config parse stays one flat dataclass — **no `SweepConfig` hierarchy**.
4. `evo` is the metric source of truth. `metrics.py` already wraps `evo` (APE/RPE) — do not re-handwrite; the only non-evo compute is `compute_auc` (keep).

---

## 1. `eval.py` — the single config-driven runner

Rename `evals/eval_gt.py` → `evals/scripts/eval.py`. It already runs one backbone × conditions in clean per-condition subprocesses and writes `metrics.json`. Add:

- **YAML config** (`evals/configs/<experiment>.yaml`): flat, human-readable, one file per experiment. `yaml.safe_load` (yaml 6.0.3 present) → flat `EvalConfig` dataclass → `itertools.product` over declared axes (`datasets` × `backbones` × `conditions` × swept params) → existing per-condition subprocess loop → **resume** (skip a cell whose `metrics.json` exists) → `--dry_run`.
- **Post-grid aggregation**: after the grid, collect every cell's `metrics.json` → one comparison table (markdown + json) with rows keyed by `dataset/backbone/condition`. Reuses `eval_compare.py`'s row-collect + markdown formatter (see §2). Generic table — **no bespoke `delta_vs_spark`/`delta_vs_slam` columns**; a VGGT-SLAM row is just another row when its TUM is present.
- CLI keeps the existing single-cell path (`--backbone --seq_dir --conditions ...`) for ad-hoc runs; `--config` is the grid path. Both share the same per-condition leaf.

Example config:

```yaml
name: cross_model_chess
datasets:
  - {name: 7scenes, seq_dir: evals/data/7scenes/chess/chess/seq-01, keyframe_list: null}
backbones: [vggt_omega, vggt_spark, mapanything]
conditions: [baseline, lc]
submap_size: 50
max_frames: 200
lc_layer: null            # optional per-backbone LC verify-layer override
output_dir: evals/results/cross_model_chess
```

Preset configs to author (replace retired bash drivers): `7scenes.yaml` (chess/fire/office, 6 conditions — replaces `eval_suite.sh`), `cross_model_chess.yaml` (replaces `run_cross_model_benchmark.py`).

## 2. `eval_compare.py` → aggregation module

**Format note:** `.tum` (`timestamp tx ty tz qx qy qz qw`) is the *single* canonical trajectory format for every dataset — ours, GT, and baselines (VGGT-SLAM/VGGT-Long publish via evo/TUM). Dataset-specific logic lives only at the GT-loader boundary (raw 7-Scenes/CO3D/KITTI poses → `.tum` via `trajectory_io`); everything downstream is format-uniform. Outputs are exactly two types, both dataset-agnostic: `.tum` (trajectory per method/condition/baseline + GT) and `metrics.json` (ATE/RPE/AUC). `eval_compare.py`'s `_scan_results_dir` ingests any `.tum` in a results dir as a method row; `_format_markdown` renders the table. Convert to an imported module (`evals/scripts/eval_compare.py` or fold into `eval.py`): keep the row-collect + formatter functions, drop the standalone CLI framing. `eval.py` imports them for §1 aggregation. `test_eval_compare.py` repointed to the module functions **only after** a new test asserts identical table output on the same fixtures (equivalence-first).

## 3. Retire the validation sweep drivers + analysis

Each retired script's test is deleted with it (allowed: deleting a script deletes its covering test).

| File | LOC | Fate | Reason |
|---|---|---|---|
| `runners/run_cross_model_benchmark.py` + `test_cross_model_runner.py` | 131 | retire | backbone×frameset = `eval.py` grid + preset config |
| `runners/run_disparity_sweep.py` (no test) | 288 | retire | ours-vs-SLAM gating was validation; parity now a comparison row |
| `runners/run_lc_parity.py` + `test_run_lc_parity.py` | 236 | retire | LC parity frozen in `baselines/` + specs |
| `runners/build_benchmark_table.py` + `test_build_benchmark_table.py` | 78 | retire | replaced by `eval.py` generic aggregation |
| `runners/build_parity_table.py` + `test_build_parity_table.py` | 229 | retire | parity gate table, historical |
| `runners/lc_loop_pr.py` + `test_lc_loop_pr.py` | 294 | retire | GT-only research metric (needs covisibility labels); validation done (precision 1.0); not user-facing |
| `runners/visualize_lc_correction.py` + `test_visualize_lc_correction.py` | 255 | retire | validation-era loop-chord viz; `evo_traj` covers routine trajectory plots for users |
| `evals/ate_utils.py` + `test_ate_utils_tum.py` | ~130 | retire | `compute_ate_rmse` duplicates `metrics.compute_ate` (2nd evo path); `_load_gt_as_tum_trajectory` unused once it dies |
| `evals/eval_suite.sh` | — | retire | bash bundle of `eval_gt` calls → `7scenes.yaml` + `eval.py` |

**Keep:** `eval_multiview_conf.py`, `eval_similarity_calibration.py` (measure point-cloud confidence / retrieval similarity — distinct signals, not ATE, evo can't produce them; the latter already has a `--layer_index` sweep).

## 4. Merge the VGGT-SLAM wrappers → one `run_vggt_slam.py`

`run_vggt_slam.py` (dense/no-LC, referenced by `setup/vggt_slam.sh` + README) and `run_vggt_slam_lc.py` (dense+LC, TUM + ATE + `selected_frames.txt`) overlap; `run_vggt_slam_lc.py` with `--max_loops 0` **is** the no-LC baseline (verified: `run_vggt_slam_lc.py` does not import `run_vggt_slam.py`). Merge into one `evals/scripts/run_vggt_slam.py`:

- Absorb the LC path (TUM + ATE + `selected_frames.txt`) into `run_vggt_slam.py`; LC controlled by `--max_loops` (0 = published baseline).
- **Equivalence-first:** before deleting `run_vggt_slam_lc.py`, confirm the merged script reproduces both prior outputs (no-LC baseline TUM; LC TUM + `selected_frames.txt`) on a fixture.
- **Repoint references:** `setup/vggt_slam.sh` documented example command, `evals/README.md`. `eval_suite.sh` reference vanishes with its retirement (§3).
- Keep `setup/vggt_slam.sh` itself (isolated-venv installer, not a driver — orthogonal).
- Tests: keep/repoint one merged test; retire the redundant one after equivalence.

## 5. Dissolve `lc_parity_common.py`

Only two live consumers survive the §3 retirements: `eval.py` (`_serialize_lc_decisions`) and merged `run_vggt_slam.py` (`write_tum_allowed_frames` + frame helpers `collect_frames`/`list_scene_images`/`filter_images_to_list`). Move:

- `_serialize_lc_decisions` → `eval.py`.
- `write_tum_allowed_frames` + frame helpers → `evals/datasets.py` (GT/dataset concern).
- Drop parity-gate machinery: `SceneSpec`/`SCENES`, `check_gates`, `check_scaling_gate`, `check_lc_harmless`, `slice_keyframes`, `slam_max_frames_for_prefix`, `PREFIX_FRACTIONS`, `ATE_*_TOL`.
- Delete the file. Repoint `test_lc_parity_common.py` / `test_lc_decisions.py` to the moved functions (equivalence-first) or delete the parts covering dropped machinery.

## 6. Library extraction (evals → collab_splats)

- **`trajectory_io._invert_se3`** duplicates `geometry.transforms.invert_poses` (strict superset). Delete `_invert_se3`; swap its 4 call sites. `trajectory_io.py` **stays** in evals/ (zero consumers elsewhere). Keep `_check_poses`.
- **`reconstruction_quality.py`** → `collab_splats/geometry/loop_closure/diagnostics.py` (reads private `_lc_*` attrs off `LoopClosure.base` — cohesion). Update its `eval_gt.py:405`→`eval.py` inline import.
- **`compare_loop_edges.py`** helpers (`compose_slam_chain`, `edge_divergence`) → `collab_splats/geometry/loop_closure/` (used by live geometry test `test_loop_edge_chain.py`, not just an evals test). Repoint `test_loop_edge_chain.py`; retire `test_compare_loop_edges.py` (or repoint, equivalence-first).

## 7. Consolidate downloads → `evals/data/download_datasets.py`

Fold `download_7scenes.py` + `download_7scenes.sh` + `download_co3dv2.sh` + `download_kitti.sh` + `download_tum.sh` + `download_waymo.sh` + `data/download_parity_scenes.sh` (which re-hardcodes the same URLs a 3rd time) → one `download_datasets.py` (function per dataset, subcommand CLI). Move `runners/extract_waymo.py` → `evals/data/`. Fix `.gitignore` allowlist: `!evals/data/download_datasets.py` (+ `!evals/data/extract_waymo.py`) so the new files aren't swallowed by the `evals/data/*` ignore.

## 8. `runners/` → `scripts/` reorg

`evals/runners/` → `evals/scripts/`. After §3–§7, `scripts/` holds: `eval.py`, `eval_compare.py`, `eval_multiview_conf.py`, `eval_similarity_calibration.py`, `run_vggt_slam.py`. (`extract_waymo.py`→`data/`; `lc_parity_common.py`/`compare_loop_edges.py` dissolved/moved.)

- **Repoint:** the ~10 `tests/evals/*.py` files doing `sys.path.insert(.../"evals")` + `from runners.X` → point at `scripts/`. `CLAUDE.md` 2 literal `evals/eval_gt.py` refs → `evals/scripts/eval.py`. `evals/README.md` script table.
- No `eval_suite.sh` path arithmetic (it's retired). `setup/vggt_slam.sh` example command → `evals/scripts/run_vggt_slam.py`.
- Library modules (`metrics.py`, `trajectory_io.py`, `datasets.py`) stay at `evals/` top; `scripts/` = executable entrypoints.

## 9. Loop-closure codebase cleanup (carries from conservative spec §5–§7, verified)

### 9a. Dead code
- **`_assemble_precorrection_extrinsics`** (`wrapper.py:54-65`): single caller (`wrapper.py:385`), pure passthrough to `dedup_overlap`. Inline + delete.
- **`normalize_to_sl4`** (`graph.py:71-80`): zero prod callers; `PoseGraph.add_node` already SL4-normalizes (`gtsam.SL4(H)`). Delete; test callers inline the trivial det-normalize or pass raw.
- **`manifold="se3"`** (`PoseGraph.__init__` + `run_pose_graph_optimization` param + `LoopClosureConfig.manifold`): zero prod callers. Delete branch + `_pose3` helper + its tests.
- **`scale_method="none"` — KEEP** (not dead: `eval_gt.py --lc_scale_method` exposes it).

### 9b. Split `closure.py` (894L → 3 files, mirrors VGGT-SLAM)
- `matching.py` (new): `LoopMatch`, `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check`.
- `graph.py` (extend): `run_pose_graph_optimization` + scale/align helpers (`_estimate_scale_pairwise_dist`, `umeyama_se3`, `umeyama_sim3`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`, `_MIN_CONF_POINTS`, `_RNG`).
- `merge.py` (new): `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs`.
- `LoopClosureConfig` → `wrapper.py`.

**Exhaustive import fallout (grepped, verified):**
- `.matching`: `test_loop_closure.py:156`, `test_translation_jump.py:8`.
- `.merge`: `test_alignment_dedup.py:3`, `test_loop_closure_eval.py`.
- `.graph`: `test_closure_split.py`, `test_hw_formula.py`, `test_loop_ablation.py`, `test_graph.py:155` (collapse 2nd import), `test_loop_edge_chain.py:22`+`:188` (`_SL4PoseGraph` alias → re-alias `PoseGraph` locally) + logger-name assertion `~:355` (`...closure`→`...graph`), `test_pgo_parity.py:16`+`:20`.
- **`collab_splats/geometry/loop_closure/eval.py:14`** `from .closure import umeyama_se3, umeyama_sim3` → `.graph`. (package-internal importer — easy to miss.)
- **Mock patches** in `test_feedforward_lc_state.py:54-55` patch `"...closure.find_loop_closures"` by string path → move to patching `wrapper.<name>` (the pattern the file's other patches use), not a mechanical `.closure`→`.matching` rename.
- **Lazy `__getattr__` (BOTH files):** `LoopClosureConfig` is eager-imported in `loop_closure/__init__.py` AND `geometry/__init__.py`. Once it lives in `wrapper.py` (which pulls `pointcloud`→cycles through `geometry.transforms`), those eager imports deadlock. Add a `LoopClosureConfig` branch to BOTH `__getattr__` hooks (mirror the existing `LoopClosure` case); drop from both eager import lines; keep in both `__all__`.
- **`visualize_lc_correction.py:23`** imports `umeyama_sim3` from `.closure` — moot, that script retires (§3).
- **Deferred, leave alone:** `docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb` imports from `.closure` (out of scope per CLAUDE.md in-flight list).

### 9c. `_run_lc_loop` → VGGT-SLAM `Solver` parity rename
- `run_predictions(window, submaps, ...)` → forward pass + build Submap + retrieval + verify + jump-check. Returns `(submap, lc_submaps: list[Submap], loop_matches: list[LoopMatch])` — **not** a singular `lc_submap_or_None` (`max_loops_per_submap` defaults 5; multiple accepted matches per window is common). `loop_matches` (all candidates, accepted+rejected) lets the caller drive `all_loop_candidates` + pbar counters.
- `add_points(...)` → `submaps.append` + `lc_submaps.extend` (bookkeeping only; PGO deferred to the batch call, unlike `Solver.add_points`).
- Outer sweep stays a private method (pbar, `end >= N` break) — not exposed. One calling convention (BaseFeedforwardCreator template method).

## 10. Test-equivalence protocol (hard constraint)

- **Deleting a script deletes its covering test** — allowed.
- **Moving/repointing a test:** the new test must assert the same behavior as the old, both green in one commit, *then* remove the old. Applies to `eval_compare`, `lc_parity_common`, `compare_loop_edges`, merged `run_vggt_slam`, all LC-split import repoints.
- `pytest tests/` green per commit; baseline pass count vs `docs/known-test-failures.md`.

## 11. Commit order (separable, bisectable)

1. LC dead code (§9a) — smallest, isolated.
2. Library extraction (§6: `_invert_se3`, `reconstruction_quality`→diagnostics, `compare_loop_edges` helpers→geometry).
3. `closure.py` split + `LoopClosureConfig` move + `__getattr__` fixes + import fallout (§9b).
4. `_run_lc_loop` rename (§9c).
5. `eval.py` config runner + `eval_compare` module + preset configs (§1, §2).
6. Merge VGGT-SLAM wrappers + repoint (§4).
7. Dissolve `lc_parity_common` (§5).
8. Retire sweep drivers/analysis/`ate_utils`/`eval_suite.sh` + their tests (§3).
9. Downloads consolidation + `.gitignore` (§7).
10. `runners/`→`scripts/` reorg + test/CLAUDE.md/README repoints (§8).

Order rationale: LC-internal changes first (self-contained), then the eval.py capability that replaces the drivers, then retirements (so nothing references a deleted file mid-sequence), reorg last (pure path churn).

## Environment / house rules

- Python: `/opt/venv/reconstruction/bin/python` (py3.11).
- Test: `/opt/venv/reconstruction/bin/python -m pytest tests/`.
- Format before every commit: `black . && isort .`.
- Commit style: `refactor(evals):`, `refactor(geometry):`, `docs(specs):`; trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `docs/superpowers/**` may need `git add -f`.
