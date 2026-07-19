# Design: Evals Suite + Loop-Closure Cleanup

- **Status:** Approved
- **Date:** 2026-07-19

## Context

`evals/` has grown to 5,198 LOC across `baselines/`, `data/`, `envs/`, `results/`, and a dozen scripts, with three runner scripts (`run_cross_model_benchmark.py`, `run_disparity_sweep.py`, `run_lc_parity.py`) independently reimplementing the same subprocess-matrix-sweep shape, and two table-builders duplicating the same glob→json→markdown render shape. `collab_splats/geometry/loop_closure/` was ported from the vendored `third_party/VGGT-SLAM/` reference and picked up dead branches and abstractions along the way (`_assemble_precorrection_extrinsics`, `normalize_to_sl4`, `scale_method="none"`/`manifold="se3"`) that have zero production callers today. Separately, `closure.py` (894 lines) bundles three concerns — retrieval/matching, pose-graph merge, submap-output merge — that VGGT-SLAM keeps in three separate files (`loop_closure.py`, `graph.py`, `map.py`), and `LoopClosure`'s core per-window loop (`_run_lc_loop`) has no naming correspondence to VGGT-SLAM's `run_predictions`/`add_points` split, making it hard to cross-reference against the reference implementation.

Goal: reduce LOC and consolidate duplicated logic while keeping `pytest tests/` green throughout (dead-code removal takes its covering tests with it; everything else must reproduce existing behavior exactly).

## Non-goals

- **`GraphMap` persistence** — already deferred, see [ADR 003](../decisions/003-defer-graphmap.md); unaffected by this pass.
- **Full `Solver`-class port** — `LoopClosure` (`wrapper.py`) already plays the orchestrator role; we borrow VGGT-SLAM's method *names* for the per-window step, not its class structure.
- **Decoupling `LoopClosure` from feedforward creators.** Only `VGGTXCreator`/`MapAnythingCreator` (both feedforward) ever use it, and the LC mechanism is intrinsically FF-shaped (windowed model forward-passes producing world-points+poses in one shot — a classical SfM backend has no equivalent to decouple *into*). VGGT-SLAM's own `Solver` is likewise written directly against its VGGT model call, not abstracted over "any SLAM backbone." Defer, same shape as ADR 003: revisit only if a real non-FF creator needs LC.
- **`bundle_adjustment.py`** (`_refine_allonce` inline candidate) — separate module from loop-closure, skip to stay focused.

## Guardrails for all new code in this pass

1. Before writing a new helper, grep for an existing one that already does the job (e.g. reuse `eval.py`'s `umeyama_align`-style helpers instead of hand-rolling `apply_sim3`; confirmed no existing subprocess-matrix-sweep driver exists anywhere outside `evals/runners/` before adding one).
2. Every new function gets a short inline block comment explaining the non-obvious part (existing repo convention) — not a restatement of the signature.
3. No abstraction without a second concrete use case driving it (applies throughout — see Non-goals).

## 1. Evals dead files/dirs

- **Keep `evals/runners/run_vggt_slam.py`.** Earlier research called this dead; that was wrong — it's called by `eval_suite.sh:83`, documented in `evals/README.md`, referenced by `setup/vggt_slam.sh`, and covered by `tests/evals/test_run_vggt_slam.py`. It's the dense/no-LC subprocess wrapper (published-matching anchor); `run_vggt_slam_lc.py` is a separate dense+LC path — not a superset. No deletion here.
- Delete `evals/notebooks/` (empty).
- Delete `evals/envs/vggt_long.yml` (unreferenced by any script or README).
- Flatten `evals/baselines/{lc_parity,lc_parity_d5,lc_parity_d5_postfix,lc_parity_matrix}/` into `evals/baselines/lc_parity/<config>/` — these are parameter variants of one sweep, not distinct baselines.

## 2. Evals runner/table consolidation

`run_cross_model_benchmark.py`, `run_disparity_sweep.py`, and `run_lc_parity.py` each independently implement: build a subprocess argv matrix over some axes → `--dry_run` flag → skip a combination if its `metrics.json` already exists (resume) → serial `subprocess.run` loop. Extract this into one shared driver function (new module, e.g. `evals/runners/sweep_driver.py`), following the same shared-module pattern `lc_parity_common.py` already establishes for scene registries. Each of the three runners shrinks to: its axes definition + target command template, calling the shared driver.

Same treatment for `build_benchmark_table.py`/`build_parity_table.py`: shared glob→load-json→compute-delta→render-markdown function, parameterized by column schema; each builder becomes a thin schema definition.

Minor: `runners/visualize_lc_correction.py`'s hand-rolled `sim3_align`/`apply_sim3` duplicates the align/apply pattern already available via `closure.py`'s `umeyama_sim3` (post-split: `graph.py`) — reuse instead of reimplementing.

## 3. Loop-closure dead-code removal

- **`_assemble_precorrection_extrinsics`** (`wrapper.py:54-65`): single caller (`wrapper.py:385`), pure passthrough to `dedup_overlap`. Inline the `dedup_overlap(...)` call directly at the call site; delete the wrapper function.
- **`normalize_to_sl4`** (`graph.py`): zero production callers; dead upstream too (VGGT-SLAM defines it but every call site is commented out). Delete, along with its two test-only callers (`test_graph.py`, `test_pose_extraction.py`) — inline whatever determinant-normalization assertion those tests actually need, if any.
- **`scale_method="none"` / `manifold="se3"`** (`LoopClosureConfig`, `run_pose_graph_optimization`): zero production callers, no CLI/config surface ever sets them. Unlike VGGT-SLAM's analogous `VGGT_SLAM_SCALE_SE3` env-var toggle (a live, reachable feature there), ours is only reachable through tests. Delete both branches and their dedicated test coverage (`test_graph.py`, `test_loop_edge_chain.py` — verify no other test depends on the branch before removing).

## 4. Evals script organization — download scripts

`evals/` has 5 dataset-fetch scripts loose at its top level: `download_7scenes.py`, `download_7scenes.sh`, `download_co3dv2.sh`, `download_tum.sh`, `download_waymo.sh`, `download_kitti.sh`.

- **Delete `download_7scenes.sh`.** It's an exact functional duplicate of `download_7scenes.py` — both implement the same nested-zip-flatten trick (Microsoft's archive nests as `<scene>/<scene>/seq-NN.zip`), once in bash once in Python. `eval_suite.sh` already calls only the `.py` version, which is also strictly more complete (`--list`, `--force`, progress bar, multi-scene batch download). No unique capability in the `.sh` version to preserve.
- **Do not merge the remaining 4 scripts into one generic downloader.** `download_co3dv2.sh` (co3d pip package + its own bundled downloader script), `download_tum.sh` (`wget`+`tar` over a direct URL), and `download_waymo.sh`/`download_kitti.sh` (license-gated: no scriptable bulk fetch exists, these only verify local layout and print manual instructions) are four genuinely different acquisition mechanisms, not duplicated logic. A unified interface would just branch per-dataset internally — same LOC, more indirection, no consolidation win. Keep as separate scripts.
- **Move all 5 remaining dataset-fetch scripts into a new `evals/download/` subdirectory** (`download_7scenes.py`, `download_co3dv2.sh`, `download_tum.sh`, `download_waymo.sh`, `download_kitti.sh`). This is a pure location change to declutter `evals/`'s flat top level. Update the one caller (`eval_suite.sh`) and any README references.
- **Flag, don't fix:** default output-dir convention is inconsistent — `download_7scenes.py` defaults to top-level `data/7scenes/`, while `download_co3dv2.sh`/`download_tum.sh`/`download_kitti.sh` default to `evals/data/<dataset>/`. Changing a script's documented default output path is a behavior change, so leave as-is this pass; note it in the moved README for a future pass to unify.

## 5. `closure.py` split (mirrors VGGT-SLAM's `loop_closure.py` / `graph.py` / `map.py` seams)

`closure.py` dissolves into:

- **`matching.py`** (new) — `LoopMatch`, `LoopMatchQueue`, `find_loop_closures`, `translation_jump_check`. Mirrors VGGT-SLAM's `loop_closure.py`.
- **`graph.py`** (existing, extended) — gains `run_pose_graph_optimization` and its scale/alignment helpers (`_estimate_scale_pairwise_dist`, `umeyama_se3`, `umeyama_sim3`, `_cam_local_points`, `_lc_anchor_scale`, `_loop_chain_relatives`). Consolidates with the `PoseGraph` class already there instead of adding a fourth file.
- **`merge.py`** (new) — `dedup_overlap`, `_resolve_frame_node`, `merge_submap_outputs`. Mirrors VGGT-SLAM's `map.py`.
- **`LoopClosureConfig`** moves into `wrapper.py` (its only consumer).
- `loop_closure/__init__.py` exports updated accordingly; the existing lazy `__getattr__` for `LoopClosure` (documented circular-import workaround) is kept as-is.

## 6. `LoopClosure` wrapper (`wrapper.py`) renames

`_run_lc_loop`'s per-window body (forward pass → retrieval → loop-candidate detect → verify → append) currently has no naming correspondence to VGGT-SLAM's `Solver`. Split it into:

- **`run_predictions(window, submaps, ...)`** — forward pass + retrieval + loop-candidate detection + verification; returns `(submap, lc_submap_or_None)`. Direct analog of `Solver.run_predictions`.
- **`add_points(submap, lc_submap, submaps, lc_submaps)`** — appends to the running submap/lc_submap lists (bookkeeping only; we defer pose-graph optimization to the batch `run_pose_graph_optimization` call rather than wiring edges incrementally like `Solver.add_points` does — naming matches the role, not the implementation).

The outer sweep loop stays a private method inside `LoopClosure`, precoded and internal — **not** exposed to callers. `LoopClosure` has exactly one calling convention (the `BaseFeedforwardCreator` template method via `run_inference()`), unlike VGGT-SLAM where the sliding-window loop lives in caller-side `main.py` because `Solver` is driven flexibly from custom scripts. There is no caller here who would ever want to supply their own loop, so it isn't split out. `run_inference`, `load_model`, `setup_inference`, `postprocess`, `build_colmap`, `reconstruct`, `run`, `reproject` are unchanged — fixed by the template-method contract, not renamed.

## Verification

- `pytest tests/` green after each numbered section (evals cleanup + script org first, then loop-closure dead-code removal, then the `closure.py` split, then the wrapper rename) — land as separable commits so a regression is easy to bisect.
- `black . && isort .` before each commit.
- No behavior change anywhere except the deleted dead branches (`normalize_to_sl4`, `scale_method="none"`, `manifold="se3"`) and their removed tests.
