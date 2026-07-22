# 015 — Port loop_closure to VGGT-SLAM's architecture (submap-centric, GraphMap-owned)

**Date:** 2026-07-21
**Status:** accepted
**Context:** [windowed-streaming spec](../specs/2026-07-20-windowed-streaming-reconstruction-design.md)

## Decision

The long-scene loop-closure goal is achieved by **porting our `geometry/loop_closure/` package to
mirror VGGT-SLAM's clean architecture**, not by bolting streaming infrastructure onto the current
scattered design. VGGT-SLAM's codebase is already clean; we align to it directly where sensible.
Streaming behavior (lean RAM, long scenes) **falls out of** the aligned structure.

### Target architecture (VGGT-SLAM-aligned)

| File / class | Action | Notes |
|---|---|---|
| `submap.py` `Submap` | **Fatten to match VGGT-SLAM.** Store dense local `points` + `colors` + `conf` (+ `conf_masks`, `conf_threshold`); add `get_points_in_world_frame(graph)` (graph-corrected + conf-masked + colored), `get_all_poses_world(graph)`, `filter_data_by_confidence`. | Reuses the Phase-A `get_world_points(H)` + `graph.get_homography`. |
| `map.py` `GraphMap` (**new**) | **Port VGGT-SLAM `map.py`.** Owns the submaps dict + the graph; `get_frames_from_loops`, `ordered_submaps_by_key`, `get_all_cam_matricies`, world-pointcloud assembly. **Merge lives here** (VGGT-SLAM's location). | Replaces `merge.py::merge_submap_outputs`, which is **deleted** (no shim). |
| `graph.py` `PoseGraph` | **Keep our decomposition (Phase A)** — `add_submap`/`add_loop_edge`/`extract_extrinsics` are cleaner than VGGT-SLAM's in-`Solver` `add_edge`. **Rename gtsam-wrapper internals to VGGT-SLAM names** (`add_homography`, `add_between_factor`, `add_prior_factor`, `get_homography`). | `run_pose_graph_optimization` shim is **removed**; `GraphMap` drives `PoseGraph` directly. |
| `wrapper.py` `LoopClosure` | **Match `Solver`'s structure** (hold `{map: GraphMap, graph: PoseGraph, retrieval, viewer, current_working_submap}`; `run_predictions`/`add_points`). **Keep the name `LoopClosure`** (do not rename to `Solver`). | Graph construction stays in `PoseGraph` (our deliberate divergence from VGGT-SLAM, which puts it in `Solver`). |
| `matching.py` | **Keep.** Already a direct port of VGGT-SLAM `loop_closure.py` (`LoopMatch`, `LoopMatchQueue`, `find_loop_closures`) **plus our improvements** (`translation_jump_check`, NMS, `max_jump_ratio`). The retrieval *extractor* stays in `localization/retrieval.py` — intentional (localization reuses it). | No change needed. |

### Rules

- **No shims.** The refactored code is the ground-truth codebase. `merge_submap_outputs` and the
  `run_pose_graph_optimization` batch shim are **deleted**, not thinned — all callers migrate to
  `GraphMap` / `PoseGraph`. Batch and streaming **converge to one path**.
- **Port headers.** Files ported from VGGT-SLAM (`submap.py`, `map.py`, `matching.py`) carry a
  top-of-file note: `# Ported from VGGT-SLAM (github.com/.../vggt_slam), adapted for collab-splats.`
- **Lean RAM (streaming falls out).** Submaps live in `GraphMap`'s RAM dict (VGGT-SLAM style), held
  **lean** — finished local `points`/`colors`/`conf`/`poses`/`intrinsics`/`descriptors`, **no
  `frames`, no `raw_outputs`**. Loop-verify reads the detected frame from `frames.zarr` by
  `frame_idx` (our one deviation, since we have the decode-once store). Correction is graph-derived
  at read; nothing corrected persisted. RAM is `O(scene)` but lean → low-thousands of keyframes;
  true `O(window)` disk spill is a deliberately-deferred later extension.

## Validation shift (consequence of "no shims")

Parity moves from **unit-level** (the old `run_pose_graph_optimization` 1e-6 vs monolith, with ~12
direct-call tests) to **system-level**: the unified VGGT-SLAM-aligned path must reproduce the known
LC ATE **across all four backbones** (vggtx / vggt_omega / mapanything / vggt_spark) vs the frozen
baselines, and preserve the per-model calibrations (`project_lc_layer_calibration`). The
unit tests that called the deleted shim are rewritten against `GraphMap` / `Submap` / `PoseGraph`.
This is coarser than the 1e-6 anchor but is how VGGT-SLAM itself is validated, and is the explicit
cost of making the refactor the ground truth rather than carrying shims.

## Long-term direction

VGGT-SLAM's `Submap` is a **fat, central domain object**; ours *was* thin, with responsibilities
scattered across `_postprocess`, `merge.py`, and `graph.py`. This refactor centralizes per-submap
logic onto `Submap` and scene-level logic onto `GraphMap`, mirroring VGGT-SLAM. We do **not** port
VGGT-SLAM extras we don't need: its `voxelized_points` cache (we have `subsample_points`) and
`semantic_vectors` (we have a separate semantics path) stay out.

## Rejected alternatives

- **`SubmapStore` / `submaps.zarr` disk spill (explored, reverted B1–B4).** A whole store subsystem
  VGGT-SLAM lacks; overengineered for "scale via LC now." Revive deliberately only if `O(window)` is
  later required.
- **A′ (spill `raw_outputs`-minus-images).** Preserved exact batch parity but kept the heavy
  raw-outputs coupling. Heavier than replicating VGGT-SLAM.
- **A new `SubmapMap` wrapper class.** Rejected — `GraphMap` is the collection owner; no second class.
- **Keeping batch shims for parity.** Rejected per the no-shims rule above.
