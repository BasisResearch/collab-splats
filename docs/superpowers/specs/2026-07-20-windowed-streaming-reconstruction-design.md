# Windowed Streaming Reconstruction — Design

**Date:** 2026-07-20
**Status:** draft
**Depends on:** [frame-store](2026-07-20-keyframe-store-design.md) (Spec 1)
**Related:** [scene-viewer](2026-07-19-scene-viewer-design.md) (viewer + deferred LC hooks),
[LC parity harness](../../../CLAUDE.md) memory `project_lc_parity_harness`

## Context

The reconstruction pipeline is **batch**: `setup_inference` preprocesses ALL N frames up
front (`self.base.views` holds everything), and loop closure is a post-process — loops are
detected per submap window, but the SL(4) pose graph is solved **once** at the end over the
whole `list[Submap]` held in RAM. GPU memory is already bounded by `submap_size` (VGGT
global attention per window), so the scaling wall is **CPU-side accumulation**: the submap
list (frame tensors + world_points) and the merged cloud grow linearly with scene length.
On a >1000-keyframe scene this exhausts the 46.6 GB container cap.

The goal is to process arbitrarily long videos: sample keyframes as needed, build the scene
**window by window**, apply cross-window loop closure with global optimization, and
visualize the growing scene in real time — with memory bounded by *window size, not scene
length*.

Since the video already exists on disk, a true streaming/online ingest (producer/consumer +
online incremental PGO) buys little over a windowed sequential pass and costs the hardest
part (online solver). We take the **windowed sequential** approach, recognizing that the
existing LC loop already iterates submap-by-submap — so `window = submap`, and the work is
to make that loop **memory-streaming, disk-backed, incrementally-optimized, and
viewer-wired**.

## Goals

- RAM bounded by window (submap) size, not scene length.
- Cross-window loop closure with **global PGO** for a globally consistent long scene.
- Real-time viser visualization of the scene as it is built and corrected.
- Final output **identical** to the batch path (parity gate).
- Decode-once keyframe I/O (via Spec 1's `FrameStore`).

## Non-goals

- True online/streaming ingest during capture (frames are on disk).
- Online *incremental* PGO (iSAM). Global PGO fires on loop events + final; that is
  sufficient and keeps parity with the batch final solve.
- Changing keyframe selection or the VGGT/MapAnything backends' inference.

## Design

### window = submap

Reuse existing `submap_size` / `submap_overlap` (`LoopClosureConfig`,
`geometry/loop_closure/wrapper.py`). The change is to `_run_lc_loop`: instead of
preprocessing all frames then iterating in-memory submaps and retaining the whole list,
process one submap at a time, spill it to disk, and retain only compact state.

### Per-submap streaming loop

For each window (submap) of keyframes:

1. **Preprocess only this window** — `_preprocess_window(idxs)` on
   `BaseFeedforwardCreator`, reading keyframes from `FrameStore` (Spec 1). Replaces the
   load-everything `_preprocess`.
2. **Forward** — unchanged; GPU bounded by `submap_size`.
3. **Spill to disk** — write the submap to
   `submaps.zarr/submap_NNN/{points, colors, poses, intrinsics, descriptors}`
   (Blosc-compressed, chunked). Then **free** the window's frames / world_points.
4. **Graph update** — add a sequential SL(4) edge to a **persistent** `PoseGraph`
   (`geometry/loop_closure/graph.py`); compose forward for the viewer pose. Retain the
   submap's **retrieval descriptors** in RAM (small vectors).
5. **Loop closure** — detect loops vs retained descriptors. On a candidate, **reload that
   one submap** from `submaps.zarr` (cheap) to run `_verify_loop_candidate`.
6. **On verified loop** — add the loop edge, run **global PGO** over the compact node
   graph, and have the viewer **re-upload corrected** windows.
7. **Viewer push** — subsample and `add_points` + frusta for the new submap (drifting until
   a loop corrects it).

### Retained-in-RAM state

Across the whole run only: retrieval descriptors + `PoseGraph` nodes (SL(4) 4×4 matrices).
Everything heavy lives in `submaps.zarr`. → **bounded by window, not scene length.**

### submaps.zarr — checkpoint store (own lifecycle)

Submaps spill to a **dedicated `submaps.zarr`** (sibling of `feedforward.zarr`), not into the
final artifact. Rationale: submaps are not throwaway scratch — they are a **resume /
re-optimize checkpoint**. A >1k-frame run is hours of GPU inference; retained submaps let a
crashed run resume without re-inferring earlier windows, and let LC/PGO be retuned
(re-solve from stored points + descriptors + local poses) **without re-running inference**.
Pre-PGO local poses + the node graph also serve drift debugging.

```
submaps.zarr/
  submap_000/ points, colors, poses(local), intrinsics, descriptors, frame_idx
  submap_001/ ...
  nodes/      pose-graph state (SL(4) node values, edges, loop edges)
  attrs: {schema_version, submap_size, submap_overlap, n_submaps, complete}
```

- A separate store sidesteps the `mode="w"` clobber hazard of `FeedforwardResult.save_zarr`
  entirely — the final artifact writer never touches `submaps.zarr`.
- **Kept by default**; `keep_submaps=false` deletes `submaps.zarr` after the final merge.
- The `complete` attr + per-submap presence gives a trivial **resume** check (skip windows
  already spilled).

### PGO cadence — on-loop + final (VGGT-SLAM style)

Appending a submap adds a *sequential* edge — a pure chain with a trivial forward-composed
solution, so **no global re-solve** is needed between loops. A **loop closure** is the only
event that adds a cycle constraint that redistributes error, so global PGO fires **on
verified loop** (for live viewer correction) and once **at the end** (authoritative). This
matches VGGT-SLAM, which re-optimizes the GTSAM graph on loop detection plus a final global
optimization — versus the repo's current single end-of-run `pg.optimize()`.

### Merge from disk + final export

At the end: final global PGO over the full node graph → `merge_submap_outputs` **streams
submap groups from `submaps.zarr`** (not from a RAM list) → `build_colmap` →
`PointcloudResult` → `save_zarr` writes the final `feedforward.zarr` (global merged arrays,
**no `images` copy** — references `frames.zarr` by `frame_idx`, see Spec 1). Final
full-scene COLMAP export preserved. If `keep_submaps=false`, delete `submaps.zarr`.

### Viewer wiring

Instantiate `Viewer` (`viewer.py`) in the driver when `viz.enabled`. Wire the three
deferred hook sites from the scene-viewer spec:

1. **After submap append** — re-base to the global frame, `subsample_points`
   (`viz_max_points_per_submap`, conf percentile), `add_points` + frusta.
2. **On accepted loop** — `add_lines` between the two submaps' camera centers.
3. **After PGO** — re-upload all (or affected) submaps with corrected extrinsics.

Hook sites guarded `if self.viz is not None:` with try/except (per scene-viewer spec).
Add a **keep-alive** so the viser server survives pipeline end: a
`Viewer.serve_forever()` / driver block, exposed as `--keep-viewer` in the example runner
(the server thread otherwise dies with the process).

## Parity (de-risked)

The authoritative output is the **final** global PGO over the full edge set — identical
edges to the batch path → identical poses and merged cloud. The on-loop solves affect only
the live viewer, not the final artifact. The only new numerical surface is the zarr float
round-trip on submap spill/reload, negligible at a consistent dtype (store float32/float64
matching the in-memory dtype). This preserves the per-model LC calibrations from
`project_lc_parity_harness`. A short-sequence **windowed-vs-batch pose parity test** is the
correctness gate.

## Components / files

- `pointcloud/feedforward/base.py` — `_preprocess_window(idxs)`.
- `geometry/loop_closure/wrapper.py` `_run_lc_loop` — streaming / spill / on-loop-PGO /
  viewer refactor; persistent `PoseGraph`; reload-submap-on-verify; merge-from-disk; resume.
- `submaps.zarr` schema + spill/reload/resume helpers (Blosc, zarr-v3 API).
- `viewer.py` — keep-alive helper.
- `docs/examples/run_scenes.py` + a config — example for a >1k-frame video.

## Example config (knobs)

```yaml
pointcloud:
  loop_closure: true
  streaming: true                 # per-submap spill + on-loop PGO + merge-from-disk
  submap_size: 20
  submap_overlap: 1
  spill_store: submaps.zarr        # dedicated checkpoint store
  keep_submaps: true              # keep for resume/re-opt; false deletes after merge
  viz:
    enabled: true
    port: 8080
    max_points_per_submap: 50000
    conf_percentile: 20.0
preprocess:
  method: optical_flow            # keyframe selector (Spec 1 store)
  max_frames: 400                 # bound keyframes drawn from the >1k-frame video
```

## Implementation principles

- **Refactor the existing loop, don't add a framework.** The streaming path is `_run_lc_loop`
  restructured, not a new streaming/producer-consumer subsystem. Reuse `Submap`, `PoseGraph`
  (+ `add_sequential_edge` / loop edges / `optimize`), `run_pose_graph_optimization`,
  `find_loop_closures`, `_verify_loop_candidate`, `merge_submap_outputs`, `subsample_points`,
  the `Viewer`, and `build_pycolmap_reconstruction` / `build_colmap`. No new solver, no new
  viewer, no online iSAM.
- **Reuse Spec 1.** `_preprocess_window` reads from `FrameStore` — it does not add its own
  decode/IO path.
- **Retire dead code.** `graph.py` notes per-submap optimize "is possible but currently runs
  once" — the on-loop path realizes that; remove any now-dead single-shot-only stub or
  commented scaffolding it leaves behind. If the batch (load-all) path is fully subsumed,
  collapse it rather than keeping two parallel code paths; if it must stay for short scenes,
  factor the shared steps so there is one implementation, not a fork.
- **Don't over-build the viewer hooks.** Guarded `if self.viz is not None:` call sites only;
  no incremental-PGO, mesh preview, or dashboard integration (explicit non-goals of the
  scene-viewer spec).
- **Inline block comments** on each logical block (window preprocess, spill, free, graph
  update, loop verify, PGO, viewer push); one-line docstrings on new public methods.

## Testing

- **Unit:** `_preprocess_window` loads only the window's frames; submap zarr round-trip;
  free-after-spill drops frame/world_point refs (RAM drop); on-loop PGO fires only on
  verified loops (not on plain sequential appends); merge-from-disk == in-memory merge;
  **resume** — a run interrupted mid-way and restarted skips already-spilled submaps and
  produces the same final result; `keep_submaps=false` removes `submaps.zarr` after merge.
- **Parity:** windowed poses ≈ batch poses on a short sequence (within tolerance) — the
  correctness gate; preserves per-model LC calibrations.
- **Smoke:** short video end-to-end with viewer stubbed.
- **Memory:** >1k-frame run — RSS bounded by window, not scene length (assert peak RSS
  roughly flat across windows vs the batch path's linear growth / OOM).

## Verification

Run `docs/examples/run_scenes.py` with the streaming config on a >1000-frame video in
tmux (per repo memory-eval guidance). Confirm: `frames.zarr` decoded once (Spec 1),
per-submap groups appear in `submaps.zarr`, RSS stays bounded, the viser page shows the
scene growing and snapping on loop closures, and the final merged cloud + COLMAP match the
batch result on a truncated parity run.
