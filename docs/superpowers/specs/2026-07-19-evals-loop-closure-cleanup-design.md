# Design: Evals Suite + Loop-Closure Cleanup

- **Status:** Approved
- **Date:** 2026-07-19

## Context

`evals/` has grown to 5,198 LOC across `baselines/`, `data/`, `envs/`, `results/`, and a dozen scripts, with three runner scripts (`run_cross_model_benchmark.py`, `run_disparity_sweep.py`, `run_lc_parity.py`) independently reimplementing the same subprocess-matrix-sweep shape, two table-builders duplicating the same glob→json→markdown render shape, and library-grade code (trajectory I/O, LC diagnostics) sitting in `evals/` despite having no eval-specific content. `collab_splats/geometry/loop_closure/` was ported from the vendored `third_party/VGGT-SLAM/` reference and picked up dead branches and abstractions along the way (`_assemble_precorrection_extrinsics`, `normalize_to_sl4`, `scale_method="none"`/`manifold="se3"`) that have zero production callers today. Separately, `closure.py` (894 lines) bundles three concerns — retrieval/matching, pose-graph merge, submap-output merge — that VGGT-SLAM keeps in three separate files (`loop_closure.py`, `graph.py`, `map.py`), and `LoopClosure`'s core per-window loop (`_run_lc_loop`) has no naming correspondence to VGGT-SLAM's `run_predictions`/`add_points` split, making it hard to cross-reference against the reference implementation.

Goal: reduce LOC and consolidate duplicated logic while keeping `pytest tests/` green throughout (dead-code removal takes its covering tests with it; everything else must reproduce existing behavior exactly).

## Non-goals

- **`GraphMap` persistence** — already deferred, see [ADR 003](../decisions/003-defer-graphmap.md); unaffected by this pass.
- **Full `Solver`-class port** — `LoopClosure` (`wrapper.py`) already plays the orchestrator role; we borrow VGGT-SLAM's method *names* for the per-window step, not its class structure.
- **Decoupling `LoopClosure` from feedforward creators.** Only `VGGTXCreator`/`MapAnythingCreator` (both feedforward) ever use it, and the LC mechanism is intrinsically FF-shaped (windowed model forward-passes producing world-points+poses in one shot — a classical SfM backend has no equivalent to decouple *into*). VGGT-SLAM's own `Solver` is likewise written directly against its VGGT model call, not abstracted over "any SLAM backbone." Defer, same shape as ADR 003: revisit only if a real non-FF creator needs LC.
- **`bundle_adjustment.py`** (`_refine_allonce` inline candidate) — separate module from loop-closure, skip to stay focused.
- **Unifying download output-dir conventions** (`data/7scenes/` vs `evals/data/<dataset>/`) — flagged in Section 4, not fixed; changing a documented default path is a behavior change.

## Guardrails for all new code in this pass

1. Before writing a new helper, grep for an existing one that already does the job.
2. Every new function gets a short inline block comment explaining the non-obvious part (existing repo convention) — not a restatement of the signature.
3. No abstraction without a second concrete use case driving it (applies throughout — see Non-goals).

## 1. Evals dead files/dirs

- **Keep `evals/runners/run_vggt_slam.py`** (relocates to `evals/scripts/` per Section 4, doesn't delete). Earlier research called this dead; that was wrong — it's called by `eval_suite.sh`, documented in `evals/README.md`, referenced by `setup/vggt_slam.sh`, and covered by `tests/evals/test_run_vggt_slam.py`. It's the dense/no-LC subprocess wrapper (published-matching anchor); `run_vggt_slam_lc.py` is a separate dense+LC path — not a superset.
- Delete `evals/notebooks/` (empty).
- Delete `evals/envs/` entirely — its one file (`vggt_long.yml`) is unreferenced by any script or README, and once removed the directory is empty.
- Flatten `evals/baselines/{lc_parity,lc_parity_d5,lc_parity_d5_postfix,lc_parity_matrix}/` into `evals/baselines/lc_parity/<config>/` — these are parameter variants of one sweep, not distinct baselines.

## 2. Evals library extraction → `collab_splats`

One `evals/` root module turns out to be genuine library code; a second looked generic but isn't, once checked against Guardrail #3 (no abstraction without a second concrete use case):

- **`trajectory_io.py` stays in `evals/`.** TUM/KITTI trajectory I/O (`read_tum`, `write_tum`, `kitti_file_to_w2c`, `kitti_3x4_flat_to_w2c`, `w2c_to_kitti_3x4_flat`) is content-generic — no eval-specific concept in it — but confirmed via repo-wide grep it has **zero consumers outside `evals/`** (`eval_gt.py`, `metrics.py`, `datasets.py`, `build_parity_table.py`, plus its own tests; nothing in `collab_splats/` or `dashboard/` imports it, and no equivalent TUM/KITTI I/O exists elsewhere to dedupe against). Moving it to `geometry/` on genericity grounds alone, with no second consumer driving it, is exactly the abstraction-without-a-use-case Guardrail #3 rules out. Leave the file where it is.
  - **Still fix: delete `trajectory_io._invert_se3`.** It hand-rolls the same batched SE(3) inverse (`R^T`, `-R^T@t`) as `geometry/transforms.py`'s `invert_poses`, just restricted to the `(N,4,4)` case — `invert_poses` already handles arbitrary leading batch dims, so it's a strict superset. All four call sites in `trajectory_io.py` (`write_tum`, `read_tum`, `kitti_3x4_flat_to_w2c`, `w2c_to_kitti_3x4_flat`) switch to `from collab_splats.geometry.transforms import invert_poses`. This is not a new cross-package dependency direction — `evals/` already imports from `collab_splats` elsewhere (e.g. `metrics.py`'s `auc_at_threshold`). `_check_poses` stays (no equivalent shape-validation helper exists in `transforms.py`).
- **`reconstruction_quality.py` → `collab_splats/geometry/loop_closure/diagnostics.py`.** Different justification from the above, so the move still holds despite the same "no other consumer" fact pattern: this one isn't about genericity, it's about cohesion with the class it inspects. Its aggregator, `compute_alignment_metrics(lc_creator)`, reads `_lc_submaps` / `_lc_all_matches` / `_lc_precorrection_extrinsics` / `_lc_corrected_extrinsics` directly off a post-run `LoopClosure`'s `base` — these are private attributes set inside `wrapper.py` itself (lines 381-398). This isn't an eval-comparison function operating on file pairs like `metrics.py`; it's a diagnostics extension of `geometry/loop_closure` that happens to have been placed in `evals/`. Its three public metrics (`loop_match_residual`, `submap_boundary_gap`, `pointcloud_chamfer`) and internal helpers (`_cam_positions`, `_symmetric_chamfer`, `_global_frame`) move as-is. `eval_gt.py`'s inline import (line 405, already inline because it's optional/heavy-path) updates to `from collab_splats.geometry.loop_closure.diagnostics import compute_alignment_metrics`.
- **`ate_utils.py` dissolves — no direct replacement file.** It bundles two unrelated things:
  1. `_load_gt_as_tum_trajectory(seq_dir, selected_frames)` — GT loading from raw 7-Scenes/TUM dataset directories. This is eval-specific (only meaningful for benchmark evaluation) and belongs beside `datasets.py`'s other GT loaders (`_load_7scenes`, `_load_tum`, etc.) — moves there.
  2. `compute_ate_rmse(...)` — duplicates `metrics.py`'s `compute_ate`: both wrap evo APE with alignment, just through two different evo entry points (`evo.main_ape.ape` here vs. `metrics.APE` there). Its only non-duplicate value is that it accepts a `seq_dir` (raw GT) instead of a pre-materialized GT TUM path. Fix: its one caller (`run_vggt_slam_lc.py:179`) writes the loaded GT trajectory through `trajectory_io.write_tum` and calls `metrics.compute_ate` against it, collapsing to one evo call path instead of two.
  - Side effect: kills the `ate_utils`-specific `importlib.util.spec_from_file_location` workaround in `run_vggt_slam_lc.py:51` (there to dodge a name collision with an installed package) — the file it pointed at no longer exists.
- **Test relocation** (mirrors `tests/` ↔ `collab_splats/` convention): `tests/evals/test_trajectory_io.py` stays put (just gains a case for the `invert_poses` swap); `tests/evals/test_reconstruction_quality.py` → `tests/geometry/loop_closure/test_diagnostics.py`; `tests/evals/test_ate_utils_tum.py`'s cases merge into the existing `tests/evals/test_datasets.py` (testing the relocated GT loader), then the file is deleted. `test_metrics.py`/`test_eval_compare.py` are unaffected — `trajectory_io`'s import path doesn't change.

## 3. Evals runner/table consolidation

`run_cross_model_benchmark.py`, `run_disparity_sweep.py`, and `run_lc_parity.py` each independently implement: build a subprocess argv matrix over some axes → `--dry_run` flag → skip a combination if its `metrics.json` already exists (resume) → serial `subprocess.run` loop. Extract this into one shared driver function (new module, `evals/scripts/sweep_driver.py`, see Section 4 for the `runners/` → `scripts/` rename), following the same shared-module pattern `lc_parity_common.py` already establishes for scene registries. Each of the three runners shrinks to: its axes definition + target command template, calling the shared driver.

Same treatment for `build_benchmark_table.py`/`build_parity_table.py`: shared glob→load-json→compute-delta→render-markdown function, parameterized by column schema; each builder becomes a thin schema definition.

Minor: `visualize_lc_correction.py`'s hand-rolled `sim3_align`/`apply_sim3` duplicates the align/apply pattern already available via `closure.py`'s `umeyama_sim3` (post-split: `graph.py`, Section 6) — reuse instead of reimplementing.

## 4. Evals directory reorg — `scripts/` + `data/`

Answering "what does `runners/` actually do, and do we need `runners/`/`notebooks/` as separate directories": `runners/` is sweep-orchestration (subprocess-matrix drivers) plus post-hoc metrics aggregation on top of `eval_gt.py`, mixed in with several one-off utilities (`compare_loop_edges.py`, `visualize_lc_correction.py`) and a shared lib (`lc_parity_common.py`). Real function, name doesn't say so. `notebooks/` is empty (deleted, Section 1); `envs/` is now empty too (deleted, Section 1).

**Rename `evals/runners/` → `evals/scripts/`, and fold in the top-level CLI entry points** so everything that's *run* (as opposed to imported as a library) lives in one place:

```
evals/scripts/
    eval_gt.py  eval_compare.py  eval_multiview_conf.py  eval_similarity_calibration.py  eval_suite.sh
    run_cross_model_benchmark.py  run_disparity_sweep.py  run_lc_parity.py
    run_vggt_slam.py  run_vggt_slam_lc.py
    build_benchmark_table.py  build_parity_table.py
    sweep_driver.py  lc_parity_common.py          # shared libs for the above
    lc_loop_pr.py  compare_loop_edges.py  visualize_lc_correction.py
```

Library modules (`datasets.py`, `metrics.py`) stay at `evals/` root — nothing runs them directly, only the `scripts/` entry points import them.

This is a real interface change, not a pure move: `eval_suite.sh`'s `REPO_ROOT` computation (`cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd`) goes up one directory today; after the move into `scripts/` it must go up two. `CLAUDE.md`'s two literal `evals/eval_gt.py` references need updating to `evals/scripts/eval_gt.py`. Any `evals.eval_gt`/`evals.datasets`-style imports in tests need checking against the new `evals.scripts.*` namespace (`datasets.py`/`metrics.py` imports are unaffected, they're not moving).

**Downloads consolidate into `evals/data/download.py`.** Six download scripts currently exist: `download_7scenes.py`, `download_7scenes.sh` (exact duplicate — deleted, Section 1 already covers the case for it, restated here for completeness), `download_co3dv2.sh`, `download_tum.sh`, `download_waymo.sh`, `download_kitti.sh`, plus `evals/data/download_parity_scenes.sh` — a batch downloader for LC-parity scenes that re-hardcodes its own copies of the 7-Scenes CDN URL and TUM base-URL/path mechanics rather than calling the other scripts (its own header comment admits manually cross-checking for consistency instead). That's the same download logic duplicated a third time.

**Delete `download_parity_scenes.sh` outright rather than refactor it.** It's a convenience wrapper around downloading a fixed list of scenes that already-existing per-dataset downloads handle; no unique capability, and folding it into a shared module just to keep a thin batch-loop is unwarranted — a one-line note in `evals/README.md` listing which scenes/sequences the LC-parity suite needs (fire/heads/office/pumpkin/redkitchen/stairs; 4 named TUM freiburg sequences) covers the same ground without a script to maintain. Nothing calls it programmatically (confirmed via repo-wide grep — only documentation cross-references it), so deleting it has zero blast radius beyond updating those docs.

Consolidate the remaining 5 into one module, `evals/data/download.py`: one function per dataset (`download_7scenes()`, `download_co3dv2()`, `download_tum()`, and guide-only `download_waymo()`/`download_kitti()` verifiers — ported bodies, not reinvented), one CLI with subcommands (`python evals/data/download.py 7scenes --scenes fire office`). `extract_waymo.py` (waymo tfrecord → eval-layout extraction, a data-prep sidecar rather than an eval computation) moves alongside it into `evals/data/`.

- **`.gitignore` fix required in the same commit:** `evals/data/*` is ignored except for one explicit exception, `!evals/data/download_parity_scenes.sh`. Once that file is deleted and replaced by `download.py`, this line must become `!evals/data/download.py` (and cover `extract_waymo.py`) or the new tracked file is silently swallowed by the blanket ignore.
- **Flag, don't fix:** default output-dir convention is inconsistent — `download_7scenes.py` defaults to top-level `data/7scenes/`, while the others default to `evals/data/<dataset>/`. Leave as-is this pass (changing a documented default path is a behavior change); note it for a future pass.

## 5. Loop-closure dead-code removal

- **`_assemble_precorrection_extrinsics`** (`wrapper.py:54-65`): single caller (`wrapper.py:385`), pure passthrough to `dedup_overlap`. Inline the `dedup_overlap(...)` call directly at the call site; delete the wrapper function.
- **`normalize_to_sl4`** (`graph.py`): zero production callers; dead upstream too (VGGT-SLAM defines it but every call site is commented out). Delete, along with its two test-only callers (`test_graph.py`, `test_pose_extraction.py`) — inline whatever determinant-normalization assertion those tests actually need, if any.
- **`scale_method="none"` / `manifold="se3"`** (`LoopClosureConfig`, `run_pose_graph_optimization`): zero production callers, no CLI/config surface ever sets them. Unlike VGGT-SLAM's analogous `VGGT_SLAM_SCALE_SE3` env-var toggle (a live, reachable feature there), ours is only reachable through tests. Delete both branches and their dedicated test coverage (`test_graph.py`, `test_loop_edge_chain.py` — verify no other test depends on the branch before removing).

## 6. `closure.py` split (mirrors VGGT-SLAM's `loop_closure.py` / `graph.py` / `map.py` seams)

`closure.py` dissolves into:

- **`matching.py`** (new) — `LoopMatch`, `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check`. Mirrors VGGT-SLAM's `loop_closure.py`.
- **`graph.py`** (existing, extended) — gains `run_pose_graph_optimization` and its scale/alignment helpers (`_estimate_scale_pairwise_dist`, `umeyama_se3`, `umeyama_sim3`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`). Consolidates with the `PoseGraph` class already there instead of adding a fourth file.
- **`merge.py`** (new) — `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs`. Mirrors VGGT-SLAM's `map.py`.
- **`LoopClosureConfig`** moves into `wrapper.py` (its only consumer).
- `loop_closure/__init__.py` exports updated accordingly; the existing lazy `__getattr__` for `LoopClosure` (documented circular-import workaround) is kept as-is. The new `diagnostics.py` (Section 2) needs no lazy treatment — it takes `lc_creator` as a duck-typed argument, no import of `LoopClosure` itself.

## 7. `LoopClosure` wrapper (`wrapper.py`) renames

`_run_lc_loop`'s per-window body (forward pass → retrieval → loop-candidate detect → verify → append) currently has no naming correspondence to VGGT-SLAM's `Solver`. Split it into:

- **`run_predictions(window, submaps, ...)`** — forward pass + retrieval + loop-candidate detection + verification; returns `(submap, lc_submap_or_None)`. Direct analog of `Solver.run_predictions`.
- **`add_points(submap, lc_submap, submaps, lc_submaps)`** — appends to the running submap/lc_submap lists (bookkeeping only; we defer pose-graph optimization to the batch `run_pose_graph_optimization` call rather than wiring edges incrementally like `Solver.add_points` does — naming matches the role, not the implementation).

The outer sweep loop stays a private method inside `LoopClosure`, precoded and internal — **not** exposed to callers. `LoopClosure` has exactly one calling convention (the `BaseFeedforwardCreator` template method via `run_inference()`), unlike VGGT-SLAM where the sliding-window loop lives in caller-side `main.py` because `Solver` is driven flexibly from custom scripts. There is no caller here who would ever want to supply their own loop, so it isn't split out. `run_inference`, `load_model`, `setup_inference`, `postprocess`, `build_colmap`, `reconstruct`, `run`, `reproject` are unchanged — fixed by the template-method contract, not renamed.

## Verification

- `pytest tests/` green after each numbered section — recommended landing order: Section 1 (dead files/dirs) → Section 2 (library extraction) → Section 3 (runner/table consolidation) → Section 4 (scripts/data reorg) → Section 5 (LC dead-code removal) → Section 6 (`closure.py` split) → Section 7 (wrapper rename). Land as separable commits so a regression is easy to bisect.
- `black . && isort .` before each commit.
- No behavior change anywhere except the deleted dead branches (`normalize_to_sl4`, `scale_method="none"`, `manifold="se3"`), their removed tests, and the consolidated `ate_utils`/`metrics` evo call path (numerically identical: same evo APE computation, one fewer entry point).
