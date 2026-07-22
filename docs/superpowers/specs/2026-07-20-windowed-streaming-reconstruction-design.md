# Windowed Streaming Reconstruction — Design

**Date:** 2026-07-20
**Status:** SUPERSEDED IN PART — see decision 015
**Depends on:** [frame-store](2026-07-20-keyframe-store-design.md) (Spec 1)
**Related:** [scene-viewer](2026-07-19-scene-viewer-design.md) (viewer + deferred LC hooks),
[LC parity harness](../../../CLAUDE.md) memory `project_lc_parity_harness`

> **⚠ REVISION (2026-07-21) — read [decision 015](../decisions/015-streaming-submap-centric-vs-scattered.md) first.**
> The **disk-spill / `submaps.zarr` / `SubmapStore`** approach described throughout this spec is
> **replaced** by a **minimal replication of VGGT-SLAM's in-RAM pipeline** (decision 015):
> - **No disk store.** Submaps live in a RAM dict (VGGT-SLAM style), held **lean** — only finished
>   local `points` + `colors` + `conf` + local `poses` + `intrinsics` + `descriptors`. **No
>   `frames`, no `raw_outputs`** retained (the two heavy OOM terms).
> - **Loop-verify reads frames from `frames.zarr`** by `frame_idx` (our one deviation from
>   VGGT-SLAM, since we already have the decode-once store) — this is what removes the frame RAM.
> - **Correction is graph-derived at read**, nothing corrected persisted: points via
>   `Submap.get_world_points(H=graph.get_homography(frame_start))`, extrinsics via
>   `PoseGraph.extract_extrinsics` (Phase A). Merge = iterate in-RAM submaps → conf-mask → concat,
>   exactly VGGT-SLAM's `write_points_to_file`.
> - **RAM is O(scene) but lean** (drops frames+raw_outputs) → handles low-thousands of keyframes;
>   true O(window) disk spill is a deliberately-deferred later extension.
> - **Parity gate shifts:** the streaming cloud is produced the VGGT-SLAM way (H-on-points), **not
>   bitwise-identical** to the current batch `_postprocess`. Gate = windowed == a batch-D reference,
>   validated **across all four backbones**, plus a documented tolerance vs current-batch.
>
> Wherever the sections below say "spill to `submaps.zarr`", "`SubmapStore`", "free the window",
> "merge-from-disk", or "resume", read them per decision 015: in-RAM lean submaps, frames from
> `frames.zarr`, in-RAM merge. The **PGO cadence, loop-edge A/B gate, viewer wiring, `_preprocess`
> refactor, and config** sections remain valid as written.

## Context

The reconstruction pipeline is **batch**: `setup_inference` preprocesses ALL N frames up
front (`self.base.views` holds everything), and the whole `list[Submap]` is held in RAM for
the duration. GPU memory is already bounded by `submap_size` (VGGT global attention per
window) and the **pose graph** is cheap (SL(4) 4×4 nodes, ~KB/frame). The scaling wall is
**CPU-side point accumulation**: every submap's dense per-pixel `world_points` `(S,H,W,3)` +
colors + conf (repo also keeps `frames` + `raw_outputs`) is retained for the *entire* run,
never evicted — same as VGGT-SLAM's `map.submaps` dict (`map.py:11,20`), which also keeps
every submap resident. Upstream applies conf-masking and voxel-downsampling **on read only**
(`submap.py:170-238`), not to the stored arrays, so the resident footprint is the full dense
grid regardless.

This growth is **linear and unbounded** — RAM ∝ scene length. The exact OOM point is
resolution/config-dependent (order hundreds of submaps ≈ thousands of keyframes at the
46.6 GB cap; repo's extra `raw_outputs`/`frames` per `Submap` pull it lower), but the
mechanism guarantees OOM on a long enough scene. There is no fixed submap count that is
"safe"; the batch path simply does not scale.

The goal is to process arbitrarily long videos: sample keyframes as needed, build the scene
**window by window**, apply cross-window loop closure with **global** PGO, and visualize the
growing scene in real time — with memory bounded by *window size, not scene length*. The
immediate goal is **scaling long-scene loop closure** (spill dense points to disk → O(window)
RAM). **Point-count / storage optimization is explicitly deferred** — submaps spill as full
dense arrays (parity-exact); voxel/conf reduction of the *stored* payload is a later concern,
and the merged-cloud output reductions stay on-read exactly as upstream.

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
- Online *incremental* PGO (iSAM). PGO re-solves per submap (VGGT-SLAM cadence, see below);
  that is sufficient and keeps exact parity with the batch path. **Online is a strict
  superset** of windowed (windowed = online minus live ingest minus iSAM); switching later is
  a localized swap of the frame source (behind `FrameStore`) + the solver (behind
  `PoseGraph`), with spill/merge/graph-build unchanged. Those seams **already exist** — we do
  not build them speculatively now; we simply do not cross them.
- Changing keyframe selection or the VGGT/MapAnything backends' inference.
- **Reducing the stored point count.** Submaps spill as full dense arrays. Voxel/conf
  downsampling of the *stored* payload (vs the on-read output reductions that already exist)
  is deferred — the immediate goal is scaling loop closure, not shrinking `submaps.zarr`.

## Design

### window = submap

Reuse existing `submap_size` / `submap_overlap` (`LoopClosureConfig`,
`geometry/loop_closure/wrapper.py`). The change is to `_run_lc_loop`: instead of
preprocessing all frames then iterating in-memory submaps and retaining the whole list,
process one submap at a time, spill it to disk, and retain only compact state.

### Per-submap streaming loop

For each window (submap) of keyframes:

1. **Preprocess only this window** — the caller (driver / `_run_lc_loop`) pulls the window's
   frames from `FrameStore` (`store.read(idxs)`) and passes the **in-memory frame arrays**
   into `_preprocess(frames)`. `_preprocess` is refactored from `_preprocess(image_dir: Path)`
   to accept decoded frames, so `FrameStore` is the **sole I/O path** and decode-once (Spec 1)
   is honored. This **deletes the `_preprocess_from_store` + `_frame_export` temp-JPG
   round-trip** (`base.py:805-812`), which currently re-encodes zarr → JPG → re-reads —
   fighting the frame store. Per-model crop/resize stays in each creator's `_preprocess`
   (it differs per backend); only the *source* moves out. `image_paths` become `frame_idx`
   labels (Spec 1 references frames by index, not filename).
2. **Forward** — unchanged; GPU bounded by `submap_size`.
3. **Spill to disk** — write the submap to
   `submaps.zarr/submap_NNN/{points, colors, poses, intrinsics, descriptors}`
   (Blosc-compressed, chunked). Then **free** the window's frames / world_points.
4. **Graph update** — add the submap's nodes + sequential SL(4) edges to a **persistent**
   `PoseGraph` (`geometry/loop_closure/graph.py`) and call `pg.optimize()` — matching
   VGGT-SLAM, which optimizes every submap (`main.py:120`), and the repo's current
   per-submap `pg.optimize()` (`graph.py:575`). Retain in RAM: the submap's **retrieval
   descriptors** and its **overlap-frame world_points + conf** (the `O` boundary frames the
   next submap's inter-submap scale estimation reads — see Retained-in-RAM state).
5. **Loop closure** — detect loops vs retained descriptors. On a candidate, **reload that
   one submap** from `submaps.zarr` (cheap) to run `_verify_loop_candidate` and to supply
   the loop-anchor world_points for scale reconciliation.
6. **On verified loop** — stash the `lc_submap` (2 frames, small) for the loop edge; draw the
   viewer loop **line** on accept. **When** the edge enters the graph is set by the A/B gate:
   *deferred* → applied with all loop edges at the final solve; *live* → inserted now,
   absorbed by the next per-submap `optimize()` (VGGT-SLAM `solver.py:294-295`). Either way
   loop detection triggers a **full-scene viewer re-upload** (all submaps' corrected
   extrinsics) vs the latest-only push on a plain append — VGGT-SLAM's `update_all_submap_vis`
   vs `update_latest_submap_vis` (`main.py:122-127`), a viewer-scope choice, **not** an extra
   solve.
7. **Viewer push** — subsample and `add_points` + frusta for the new submap (drifting until
   a loop corrects it).

### Retained-in-RAM state

Across the whole run only:
- **Retrieval descriptors** per submap (small `(k, D)` vectors) — the loop-detection index.
- **`PoseGraph` nodes** (SL(4) 4×4 matrices) + edges — the graph is built and solved
  incrementally, so its node/edge state is the persistent object.
- **Overlap-frame world_points + conf** per submap: the `O` boundary frames only, not the
  full submap. `run_pose_graph_optimization` reads `world_points[:O]` / `[-O:]` and their
  conf to estimate the inter-submap scale (`graph.py:482-535`); these must survive the free.
  `O` is tiny (default 1), so this is negligible.

Loop-anchor scale (`graph.py:594-595`) needs the *full-frame* world_points at the two
matched frames — supplied by the **submap reload on loop verify** (step 5), not retained.
Everything else heavy (full points, colors, frames, per-frame raw_outputs) lives only in
`submaps.zarr`. → **RAM bounded by window, not scene length.**

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

### PGO cadence — per-submap optimize (VGGT-SLAM style)

**Verified against upstream.** VGGT-SLAM calls `solver.graph.optimize()` **every submap,
unconditionally**, inside its per-submap loop (`main.py:120`) — not gated on loop detection,
with no separate final solve (the last iteration's optimize is final). Loop detection there
gates only the **viewer refresh scope** (`update_all_submap_vis` vs
`update_latest_submap_vis`, `main.py:122-127`) — a visualization choice, not an extra solve.
The repo already mirrors this: `run_pose_graph_optimization` optimizes per submap
(`graph.py:575`) plus a final solve (`graph.py:626`). **Repo == upstream today.**

So the streaming loop **keeps per-submap `pg.optimize()`** — realized by feeding the
persistent `PoseGraph` one submap at a time (the graph build is already inherently
incremental: submap *s+1*'s first node inits from the *optimized* `H_overlap` of *s*,
`graph.py:544-546`). This is the exact cadence of both upstream and the current batch path,
so the final poses are identical — **zero cadence divergence, zero parity risk**. A loop
closure adds its edge (absorbed by the next per-submap optimize, as VGGT-SLAM's `add_edge`
does) and additionally triggers a full-scene viewer re-upload.

### Merge from disk + final export

At the end: final global PGO over the full node graph → `merge_submap_outputs` **streams
submap groups from `submaps.zarr`** (not from a RAM list) → `build_colmap` →
`PointcloudResult` → `save_zarr` writes the final `feedforward.zarr` (global merged arrays,
**no `images` copy** — references `frames.zarr` by `frame_idx`, see Spec 1). Final
full-scene COLMAP export preserved. If `keep_submaps=false`, delete `submaps.zarr`.

### Viewer wiring

Instantiate `Viewer` (`viewer.py`) in the driver when `viz.enabled`. Wire the three
deferred hook sites from the scene-viewer spec:

1. **After submap append + optimize** — re-base to the global frame, `subsample_points`
   (`viz_max_points_per_submap`, conf percentile), `add_points` + frusta for the **new**
   submap using its freshly optimized extrinsics (latest-only push, upstream
   `update_latest_submap_vis`).
2. **On accepted loop** — `add_lines` between the two submaps' camera centers, and
   **re-upload all** submaps with their corrected extrinsics (upstream
   `update_all_submap_vis`, `main.py:124`) — the loop is what redistributes error across the
   whole scene.

Hook sites guarded `if self.viz is not None:` with try/except (per scene-viewer spec).
Add a **keep-alive** so the viser server survives pipeline end: a
`Viewer.serve_forever()` / driver block, exposed as `--keep-viewer` in the example runner
(the server thread otherwise dies with the process).

## Parity (de-risked)

**Sequential cadence — settled.** The per-submap `optimize()` cadence matches VGGT-SLAM
(`main.py:120`) and the current repo (`graph.py:575`); commit `1372ac2` validated the
sequential forward path **bitwise-identical to SLAM** ("d50/d30/d20 stay exact"). Streaming
feeds the same submaps' sequential edges into the same persistent graph under the same
cadence → identical. The only new numerical surface is the zarr float round-trip on submap
spill/reload, negligible at a consistent dtype (store float32/float64 matching the in-memory
dtype) — and it touches only points/conf, never the pose graph. The **windowed-vs-batch pose
parity test** (short sequence) is the correctness gate for the sequential path.

**Loop-edge timing — OPEN, decided by A/B (see below).** A pre-existing divergence from
upstream lives in the *batch* path already: VGGT-SLAM inserts each loop edge **live**
(`solver.py:294-295`, absorbed by that submap's next `optimize()`), whereas the repo
**defers all loop edges to a single final solve** (`graph.py:581-624` after the submap loop,
then `:626`). This was never a deliberate "prefer deferred" choice — the parity harness
validated the sequential forward bitwise and validated that loops *improve ATE*
(`1b7bc84`), but loop-edge timing was never A/B'd. Both paths add the **identical final
factor set**; they differ only in the **initialization** fed to the final LM solve, so on
non-convex SL(4) the final poses *may* differ slightly. The per-model LC calibrations
(`spark 0.95, vggtx L10/1.17, omega L13/1.55, mapanything L4/1.46`) are **retrieval/verify
gates — independent of edge timing** (they decide which loops are accepted, not optimizer
convergence), so switching timing does **not** invalidate them.

### A/B gate — loop-edge timing (deferred vs live)

The deciding experiment, run **before** the streaming loop-edge insertion + viewer
loop-snap cadence are finalized:

1. Add a `loop_edge_timing: {deferred, live}` flag to `run_pose_graph_optimization`
   (`live` = insert the loop edge inside the per-submap loop, matching `solver.py:294-295`;
   `deferred` = current behavior).
2. Eval matrix: chess d5 (21 loops) + TUM fr3 (longest scene) × 4 backbones — the scenes
   `1b7bc84` used. Compare ATE + delta-vs-SLAM.
3. Decision rule: **live ≤ deferred ATE and ≈ SLAM → adopt live** (converge repo to
   VGGT-SLAM; streaming gets live viewer snap for free). **Live regresses → keep deferred**
   (streaming defers too; viewer snaps at final solve only).

This A/B is **headless geometry + eval**, run in its **own disposable worktree** — only the
*decision* feeds the streaming build. Whichever timing wins becomes the batch reference and
the streaming path matches it → windowed-vs-batch parity holds by construction under the
chosen timing.

## Components / files

- `pointcloud/feedforward/base.py` — refactor `_preprocess(image_dir: Path)` →
  `_preprocess(frames)` accepting decoded in-memory frames (per-model crop stays); caller
  slices the window from `FrameStore`. **Deletes** `_preprocess_from_store` + `_frame_export`
  temp-JPG hack (`base.py:805-812`). Touches all 3 creators' `_preprocess` (bounded; net
  removes code).
- `geometry/loop_closure/graph.py` — **`PoseGraph` gains incremental methods**
  `add_submap(submap, overlap_frames, ...)` (hoist `:445-575`: nodes, sequential edges,
  inter-submap `H_w`, scale estimation) and `add_loop_edge(lc, ...)` (hoist `:581-624`: loop
  3-edge chain, anchor scales), plus `extract_extrinsics(total_frames)`. The parity math
  gets **one home** (these methods) — it is *relocated, not deleted*.
  `run_pose_graph_optimization` becomes a **thin batch shim** over them (loop `add_submap` +
  `optimize` per submap → `add_loop_edge` all → final `optimize` → `extract_extrinsics`),
  preserving its signature and all ~12 direct test callers + the parity harness
  (`test_closure_split` asserts unchanged signature). Streaming calls the *same* incremental
  methods across windows → one implementation, no fork. Carries the `loop_edge_timing` flag
  (A/B gate).
- `geometry/loop_closure/wrapper.py` `_run_lc_loop` — streaming / spill / per-submap-PGO
  (persistent `PoseGraph` fed incrementally) / viewer refactor; reload-submap-on-verify;
  merge-from-disk; resume.
- `submaps.zarr` schema + spill/reload/resume helpers (Blosc, zarr-v3 API).
- `viewer.py` — keep-alive helper.
- `docs/examples/run_scenes.py` + a config — example for a >1k-frame video.

## Work parallelization

Two tracks run concurrently; they share only one coupled decision (loop-edge timing),
isolated to a single sync point. ~90% of the work is timing-independent.

### Track 1 — A/B gate (disposable worktree, headless)

The `loop_edge_timing` flag + the chess-d5 / TUM-fr3 × 4-backbone eval (§A/B gate). Pure
geometry + eval, **no viewer code**. Runs in its own `.worktrees/` copy (symlink
`third_party/`) so its throwaway measurement code never collides with Track 2's edits to
`graph.py` / `wrapper.py`. **Output: the deferred-vs-live decision** — nothing else survives.
~1 hr tmux eval.

### Track 2 — streaming spine + viewer (main worktree, timing-independent)

Everything that does **not** depend on loop-edge timing, built in parallel:
- `PoseGraph.add_submap` / `add_loop_edge` / `extract_extrinsics` + the shim refactor.
- `submaps.zarr` schema + spill / reload / resume / free-after-spill.
- `_preprocess_window(idxs)` reading `FrameStore`.
- `_run_lc_loop` streaming restructure: per-submap preprocess → forward → spill → free →
  `add_submap` + `optimize` (sequential cadence is **already settled** — matches upstream).
- Viewer: instantiate + after-submap push (`subsample_points` + `add_points` + frusta) +
  loop **line** on accept (cosmetic, fires regardless of edge timing) + keep-alive /
  `--keep-viewer`.

### Sync point (last, small)

The one coupled piece: **loop-edge insertion + viewer pose-correction re-upload cadence**.
- A/B → **deferred**: loop edges applied at final solve; viewer re-uploads corrected poses
  **once** after the final `optimize()`.
- A/B → **live**: loop edge inserted in the per-submap loop; viewer re-uploads **per
  accepted loop** (upstream `update_all_submap_vis`).

A single conditional at one hook site. Keep it **parameterized and last** — do not hardcode a
timing until the A/B returns, or a wrong guess forces a re-do of the snap cadence.

## Example config (extends base.yaml — do NOT re-invent the schema)

`configs/base.yaml` is the SOLE default source (strict access, no inline `.get` defaults —
`test_guard` in `tests/wrapper/test_reconstructor.py`). The streaming path reuses the
existing `LoopClosureConfig` fields (`submap_size`, `submap_overlap`, `scale_method`,
`conf_threshold` — already `wrapper.py:52-71`) and adds **only 4 keys**, pruned to avoid
duplication and speculative knobs:

```yaml
# configs/base.yaml — schema uses `preprocessing:` + `frame_selection:` (NOT `preprocess:`/`method:`)
preprocessing:
  frame_selection: optical_flow   # existing key; streaming wants motion-based selection
  max_frames: 200                 # existing cap — see note below; streaming can lift it

pointcloud:
  loop_closure: true
  loop_closure:                   # LoopClosureConfig (submap_size/overlap already live here)
    streaming: true               # NEW — spill+merge-from-disk path; false = current batch
    keep_submaps: true            # NEW — keep submaps.zarr for resume/re-opt; false deletes after merge
  viz:
    enabled: false                # NEW — instantiate the viser Viewer
    port: 8080                    # NEW — viser port
```

**Deliberately NOT added** (reuse / no-overengineering):
- `submap_size` / `submap_overlap` — already in `LoopClosureConfig`.
- `spill_store` — derived path `<output_path>/submaps.zarr`, nothing to configure.
- `loop_edge_timing` — A/B-disposable; the winner is **hardcoded**, not a shipped toggle
  (shipping it = two code paths).
- `viz.max_points_per_submap` / `conf_percentile` — `subsample_points` already defaults them;
  expose only if tuning proves necessary.

**base.yaml deltas to record for later:**
- `preprocessing.max_frames: 200` comment says "vggt_omega OOMs above ~300 on GPU". Under
  streaming, GPU is bounded by `submap_size`, **not** `max_frames` — so the cap can lift for
  streaming runs. Leave `200` as the default (safe for batch); note that a streaming run may
  override to a large value / null.

## Implementation principles

- **Refactor the existing loop, don't add a framework.** The streaming path is `_run_lc_loop`
  restructured, not a new streaming/producer-consumer subsystem. Reuse `Submap`, `PoseGraph`
  (+ `add_sequential_edge` / loop edges / `optimize`), `run_pose_graph_optimization`,
  `find_loop_closures`, `_verify_loop_candidate`, `merge_submap_outputs`, `subsample_points`,
  the `Viewer`, and `build_pycolmap_reconstruction` / `build_colmap`. No new solver, no new
  viewer, no online iSAM.
- **Reuse Spec 1.** `_preprocess_window` reads from `FrameStore` — it does not add its own
  decode/IO path.
- **Relocate, don't delete, the PGO math.** The refactor hoists the per-submap body of
  `run_pose_graph_optimization` (`graph.py:445-575`) into `PoseGraph.add_submap` and the loop
  chain (`:581-624`) into `PoseGraph.add_loop_edge`; `run_pose_graph_optimization` becomes a
  thin shim over them (signature preserved — all ~12 test callers + the parity harness stay
  green). The streaming loop and the batch shim both call the *same* methods → one
  implementation. The dead code to remove is the **full-materialize batch path**:
  `setup_inference` preprocessing all frames up front (`self.base.views` holding everything)
  and the RAM `list[Submap]` accumulator in `_run_lc_loop` — not any graph.py stub (none
  exists). If the load-all path is fully subsumed, collapse it; if it must stay for short
  scenes (`< submap_size` frames, the existing `_enough_frames` branch), it shares the same
  `PoseGraph` methods, not a fork.
- **Don't over-build the viewer hooks.** Guarded `if self.viz is not None:` call sites only;
  no incremental-PGO, mesh preview, or dashboard integration (explicit non-goals of the
  scene-viewer spec).
- **Explore the package before writing anything.** Every task starts by finding the existing
  function/config/helper to reuse (`graphify query` / grep the package) — reuse over new
  code. Reused this pass: `LoopClosureConfig` fields (no new `submap_*` keys), `subsample_points`
  defaults (no new viz knobs), `FrameStore.read` (no new IO), `PoseGraph` (methods added, not a
  new solver). If new code is genuinely needed, pick the **simplest** thing that works — no
  premature abstraction, no toggle for a mode that should just be the behavior (e.g.
  `loop_edge_timing` is hardcoded to the A/B winner, not shipped).
- **Inline block comments** on each logical block (window preprocess, spill, free, graph
  update, loop verify, PGO, viewer push) so the code reads top-to-bottom for a human;
  one-line docstrings on new public methods.

## Testing

- **Unit:** `_preprocess_window` loads only the window's frames; submap zarr round-trip;
  free-after-spill drops full frame/world_point refs while retaining overlap world_points +
  descriptors (RAM drop); `pg.optimize()` fires **every submap** (matching batch cadence),
  and a verified loop additionally triggers the full-scene viewer re-upload (not a plain
  append); merge-from-disk == in-memory merge;
  **resume** — a run interrupted mid-way and restarted skips already-spilled submaps and
  produces the same final result; `keep_submaps=false` removes `submaps.zarr` after merge.
- **Parity:** windowed poses ≈ batch poses on a short sequence (within tolerance) — the
  correctness gate, run **under the A/B-chosen `loop_edge_timing`**; preserves per-model LC
  calibrations (which are timing-independent verify gates).
- **A/B gate (Track 1):** `loop_edge_timing={deferred,live}` on chess d5 + TUM fr3 × 4
  backbones → ATE + delta-vs-SLAM table; the decision rule (§A/B gate) picks the timing.
  Headless, disposable worktree.
- **Smoke:** short video end-to-end with viewer stubbed.
- **Memory:** >1k-frame run — RSS bounded by window, not scene length (assert peak RSS
  roughly flat across windows vs the batch path's linear growth / OOM).

## Verification

Run `docs/examples/run_scenes.py` with the streaming config on a >1000-frame video in
tmux (per repo memory-eval guidance). Confirm: `frames.zarr` decoded once (Spec 1),
per-submap groups appear in `submaps.zarr`, RSS stays bounded, the viser page shows the
scene growing and snapping on loop closures, and the final merged cloud + COLMAP match the
batch result on a truncated parity run.
