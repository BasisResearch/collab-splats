# Loop-Closure → VGGT-SLAM Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Port `geometry/loop_closure/` to mirror VGGT-SLAM's clean architecture (fat `Submap`, new `GraphMap` owning submaps + merge, `PoseGraph` kept but renamed to VGGT-SLAM primitives, `LoopClosure` matching `Solver`'s structure), unifying the batch + streaming paths into one VGGT-SLAM-style pipeline. Streaming (lean-RAM long scenes) falls out of the aligned structure.

**Architecture:** See [decision 015](../decisions/015-streaming-submap-centric-vs-scattered.md). Submaps hold finished local dense points+colors+conf in a `GraphMap` RAM dict; correction is graph-derived at read; merge lives in `GraphMap`; no frames/raw_outputs retained (frames from `frames.zarr` on verify). **No shims** — deleted functions' callers migrate; the refactor is the ground truth.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), gtsam SL(4) PGO, viser, pytest. Ports from `third_party/VGGT-SLAM/vggt_slam/{submap,map,graph,solver,loop_closure}.py`.

---

## Environment / conventions
- Python: `/opt/venv/reconstruction/bin/python`. Tests: `... -m pytest tests/ -q`. Format: `black . && isort .`.
- Worktree: `/workspace/collab-splats/.worktrees/streaming` (branch `feat/windowed-streaming`). `third_party` is a symlink (re-`ln -s /workspace/collab-splats/third_party third_party` if a reset drops it). Stage specific files; never `git add -A` (leave the `third_party` symlink + deleted tracked `third_party/README.md` unstaged).
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Port-header rule:** files ported from VGGT-SLAM get a top note: `# Ported from VGGT-SLAM (github.com/MIT-SPARK/VGGT-SLAM), adapted for collab-splats.`
- **No-shim rule:** delete superseded functions; migrate callers; rewrite their tests against the new API.
- Config: `configs/base.yaml` is the SOLE default source (strict access; `test_reconstructor.py::test_guard`).

## Prior work reused
- **Phase A (committed, `ba4472d`):** `PoseGraph.add_submap` / `add_loop_edge` / `extract_extrinsics` — the incremental graph API. These SURVIVE; only the `run_pose_graph_optimization` shim wrapper is removed in P3.

## Validation model (decision 015)
Parity is **system-level**: the unified path reproduces known LC ATE across **all 4 backbones** (vggtx / vggt_omega / mapanything / vggt_spark) vs frozen baselines, preserving per-model calibrations (`project_lc_layer_calibration`). Unit tests that called deleted shims are rewritten against `GraphMap`/`Submap`/`PoseGraph`.

---

## Phase P1 — Fatten `Submap` to VGGT-SLAM

**Reference:** `third_party/VGGT-SLAM/vggt_slam/submap.py`. **Target file:** `collab_splats/geometry/loop_closure/submap.py`, test `tests/geometry/loop_closure/test_submap.py`.

### Task P1.1: dense points + colors + conf fields

- [ ] **Step 1 — failing test** (`test_submap.py`, new): construct a `Submap` with dense `points (S,H,W,3)`, `colors (S,H,W,3) uint8`, `conf (S,H,W)`; assert fields round-trip and `conf_threshold` is set from a percentile.

```python
"""Fat Submap (ported from VGGT-SLAM) — dense points/colors/conf + world-frame reads."""
import numpy as np
from collab_splats.geometry.loop_closure.submap import Submap

def _dense_submap(sid=0, S=2, H=4, W=5):
    rng = np.random.default_rng(sid)
    return Submap(
        submap_id=sid,
        frames=None,
        poses=np.tile(np.eye(4, dtype=np.float32), (S, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (S, 1, 1)),
        retrieval_vectors=rng.standard_normal((S, 8)).astype(np.float32),
        image_paths=[f"f{i}.jpg" for i in range(S)],
        points=rng.standard_normal((S, H, W, 3)).astype(np.float32),
        colors=(rng.random((S, H, W, 3)) * 255).astype(np.uint8),
        conf=rng.random((S, H, W)).astype(np.float32),
        frame_start=sid * S,
    )

def test_dense_fields_and_conf_threshold():
    s = _dense_submap()
    assert s.points.shape == (2, 4, 5, 3)
    assert s.colors.dtype == np.uint8
    assert s.conf_threshold is not None   # percentile-derived
```

- [ ] **Step 2 — run, confirm FAIL** (`TypeError`/`AttributeError` on the new fields): `... -m pytest tests/geometry/loop_closure/test_submap.py -q`.
- [ ] **Step 3 — implement.** Add the port header. Add fields matching VGGT-SLAM (`points`, `colors`, `conf`, `conf_masks`, `conf_threshold`) alongside existing ones. Keep back-compat: `world_points`/`world_points_conf` stay for LC scale-estimation (they read overlap points); the new `points` is the dense finished cloud. Set `conf_threshold = np.percentile(conf, pct) + 1e-6` in a post-init or an `add_all_points`-style method (mirror VGGT-SLAM `add_all_points`). Prefer a method `add_all_points(points, colors, conf, conf_threshold_percentile, intrinsics)` over the raw dataclass to match VGGT-SLAM; if staying a dataclass, add a `__post_init__` computing `conf_threshold` when `conf` is given.

**Note to implementer:** read the current `Submap` dataclass first — it is frozen-ish with required `frames`. Make `frames` Optional (already passed None on reload paths). Do NOT break existing `Submap(...)` call sites (grep `Submap(` across `collab_splats/` + `tests/`); add new fields with defaults so old constructions still work, updating them in later tasks.

- [ ] **Step 4 — run, confirm PASS.**
- [ ] **Step 5 — verify no existing breakage:** `... -m pytest tests/geometry/loop_closure/ -q` (Phase-A + existing submap users green).
- [ ] **Step 6 — commit:** `feat(loop_closure): fatten Submap with dense points/colors/conf (port VGGT-SLAM)`.

### Task P1.2: graph-corrected + conf-masked reads

**Reference:** VGGT-SLAM `submap.py:170-210` (`filter_data_by_confidence`, `get_points_in_world_frame`), `:114-130` (`get_all_poses_world`).

- [ ] **Step 1 — failing test** (append `test_submap.py`): build a `Submap` + a `PoseGraph` with that submap added; assert `get_points_in_world_frame(graph)` returns conf-masked points in world frame (shape ≤ S·H·W, correction applied) and `get_points_colors()` is index-aligned.

```python
from collab_splats.geometry.loop_closure.graph import PoseGraph

def test_get_points_in_world_frame_conf_masked():
    s = _dense_submap()
    pg = PoseGraph()
    pg.add_submap(s, overlap_frames=1)
    pg.optimize()
    pts = s.get_points_in_world_frame(pg)
    cols = s.get_points_colors()
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert cols.shape[0] == pts.shape[0]        # index-aligned
    assert pts.shape[0] <= 2 * 4 * 5            # conf mask may drop points
```

- [ ] **Step 2 — run, confirm FAIL.**
- [ ] **Step 3 — implement** (port from VGGT-SLAM, adapt to our `graph.get_homography`):
  - `filter_data_by_confidence(data)` → `data[self.conf > self.conf_threshold]`.
  - `get_points_in_world_frame(graph)` → per-frame `H = graph.get_homography(self.frame_start + i)`, apply projective transform to `self.points[i]` (dehomogenize), conf-mask, concat. Mirror VGGT-SLAM `:192-210` but use OUR `graph.get_homography` (returns the SL(4) 4×4). Reuse the existing `get_world_points(H)` transform math where possible.
  - `get_points_colors()` → `filter_data_by_confidence(self.colors).reshape(-1, 3)`.
  - `get_all_poses_world(graph)` → per-frame `decompose_camera(K @ inv(graph.get_homography(node)))` → (S,4,4). Reuse `decompose_camera` from `graph.py`.
- [ ] **Step 4 — run, confirm PASS.** **Step 5 — LC suite green.** **Step 6 — commit:** `feat(loop_closure): Submap graph-corrected conf-masked world-frame reads (port VGGT-SLAM)`.

---

## Phase P2 — `GraphMap` (new, port VGGT-SLAM `map.py`); merge moves here

**Reference:** `third_party/VGGT-SLAM/vggt_slam/map.py`. **New file:** `collab_splats/geometry/loop_closure/map.py` (`GraphMap`), test `tests/geometry/loop_closure/test_map.py`.

### Task P2.1: GraphMap collection + submap access

- [ ] **Step 1 — failing test** (`test_map.py`): add submaps; assert `add_submap`, `get_submap(id)`, `ordered_submaps_by_key()`, `get_largest_key()`, `__len__` behave like VGGT-SLAM's `GraphMap`.
- [ ] **Step 2 — run FAIL** (no module). 
- [ ] **Step 3 — implement `GraphMap`** with the port header. Port VGGT-SLAM `map.py` collection methods: `self.submaps = dict()`, `add_submap`, `get_submap`, `ordered_submaps_by_key`, `get_largest_key`, `get_latest_submap`, `__len__`, `get_frames_from_loops` (adapt: read frames from `frames.zarr` by `frame_idx` instead of submap-resident frames — pass a `FrameStore`). Keep it minimal — only methods our pipeline uses.
- [ ] **Step 4 PASS. Step 5 commit:** `feat(loop_closure): GraphMap submap collection (port VGGT-SLAM)`.

### Task P2.2: ADD `GraphMap.get_world_pointcloud` + `get_corrected_extrinsics` (no deletion yet)

**Ordering note:** `merge_submap_outputs` is still called by `wrapper.py:433` until P4 rewrites `_run_lc_loop`. Deleting it here would break the suite mid-refactor. So P2.2 only ADDS the new `GraphMap` methods; the DELETION of `merge_submap_outputs`/`dedup_overlap` happens **atomically in P4.3** together with the wrapper migration (never a shim, never a broken state). Do NOT compare against `merge_submap_outputs` — it builds a `raw_outputs` dict for `_postprocess`, whereas the new methods build the final `(points, colors)` cloud directly (different outputs, not comparable).

- [ ] **Step 1 — failing test** (`test_map.py`): with dense submaps + a `PoseGraph`, `map.get_world_pointcloud(graph)` returns `(points (M,3), colors (M,3))` = concat of each non-LC submap's `get_points_in_world_frame`/`get_points_colors`, index-aligned; `map.get_corrected_extrinsics(graph, total_frames)` == `graph.extract_extrinsics(total_frames)`. Assert against hand-built expected (concat of the submap reads), NOT against `merge_submap_outputs`.
- [ ] **Step 2 — run FAIL.**
- [ ] **Step 3 — implement** on `GraphMap` (port VGGT-SLAM `write_points_to_file`/`get_all_cam_matricies` logic as return-values, not file writes):
  - `get_world_pointcloud(graph)` → iterate `ordered_submaps_by_key()`, skip `is_lc_submap`, concat each submap's `get_points_in_world_frame(graph)` + `get_points_colors()` → `(points, colors)`.
  - `get_corrected_extrinsics(graph, total_frames)` → `graph.extract_extrinsics(total_frames)` (Phase A trusted path).
- [ ] **Step 4 PASS. Step 5 — full LC suite green** (nothing deleted, so all still passes). **Step 6 — commit:** `feat(loop_closure): GraphMap.get_world_pointcloud + get_corrected_extrinsics (port VGGT-SLAM)`.

---

## Phase P3 — `PoseGraph` rename internals to VGGT-SLAM

**Reference:** `third_party/VGGT-SLAM/vggt_slam/graph.py`.

### Task P3.1: rename gtsam-wrapper primitives to VGGT-SLAM names

- [ ] **Step 1 — failing test:** add `tests/geometry/loop_closure/test_graph_naming.py` asserting `PoseGraph` exposes `add_homography`, `add_between_factor`, `add_prior_factor`, `get_homography`.
- [ ] **Step 2 — run FAIL.**
- [ ] **Step 3 — rename** (mechanical, internal): `add_node`→`add_homography`, `add_sequential_edge`→`add_between_factor`, `add_prior`→`add_prior_factor` (keep `get_homography`, `optimize`). Update ALL internal call sites in `graph.py` (incl. Phase-A `add_submap`/`add_loop_edge`) + any test/caller. Grep `add_node\|add_sequential_edge\|add_prior\b` across `collab_splats/` + `tests/`.
- [ ] **Step 4 — run the incremental + LC suite green** (`test_pose_graph_incremental.py` still 1e-6 — pure rename).
- [ ] **Step 5 — commit:** `refactor(loop_closure): PoseGraph primitives renamed to VGGT-SLAM (add_homography/between_factor/prior_factor)`.

### Task P3.2: ~~delete run_pose_graph_optimization shim~~ — FOLDED INTO P4.3

**Moved (ordering fix).** `run_pose_graph_optimization` is still called by `wrapper.py:433` (`_run_lc_loop`) until P4.3 rewrites the loop. Deleting it before that migration would break the wrapper; migrating its ~12 parity tests off it *while it is still live in production* would strip coverage from a used function. So the deletion + test migration happen **atomically in P4.3**, together with the `_run_lc_loop` rewrite and the `merge_submap_outputs` deletion — no shim removed before its last caller is gone, no broken intermediate state. No separate "GraphMap drives PoseGraph" helper is needed: the orchestrator drives `PoseGraph.add_submap`/`optimize` incrementally per-window (P4.2), which fully replaces the batch shim.

---

## Phase P4 — `LoopClosure` matches `Solver` structure; dense unproject in `add_points`

**Reference:** `third_party/VGGT-SLAM/vggt_slam/solver.py` (`__init__`, `run_predictions`, `add_points`). **Target:** `wrapper.py` (keep the class name `LoopClosure`).

### Task P4.1: LoopClosure holds `{map: GraphMap, graph: PoseGraph, retrieval, viewer}`

- [ ] **Step 1 — failing test** (`test_wrapper.py`): assert `LoopClosure` instance exposes `self.map` (GraphMap), `self.graph` (PoseGraph) after init.
- [ ] **Step 2 FAIL. Step 3** — refactor `LoopClosure.__init__` to instantiate/hold `map` + `graph` (mirroring `Solver.__init__`, minus `add_edge` which stays in `PoseGraph`). Keep the class name + existing public entry (`run`).
- [ ] **Step 4 PASS. Step 5 commit:** `refactor(loop_closure): LoopClosure holds GraphMap+PoseGraph (match Solver structure)`.

### Task P4.2: `add_points` populates dense points+colors into the submap + `self.map` (DATA ONLY — no graph driving)

**Scope split (clean commits):** P4.2 only POPULATES the fat-submap data (dense points/colors/conf) and adds the submap to `self.map`. It does NOT change graph driving or output — the existing `_run_lc_loop` tail (still calling `run_pose_graph_optimization` on its submap list) keeps producing the output unchanged. Graph driving + GraphMap output + shim deletion are P4.3 (atomic). This avoids a redundant double-PGO intermediate state.

- [ ] **Step 1 — failing test:** a stubbed forward → the per-window submap gets dense `points` (via `unproject_depth_map_to_point_map`) + `colors` (from images·255 uint8) + `conf`, AND `self.map` contains that submap after the window is processed. Assert on the populated submap fields + `len(self.map) > 0`.
- [ ] **Step 2 FAIL. Step 3** — in `run_predictions`/`add_points` (`wrapper.py`), port VGGT-SLAM `add_points` (`solver.py:230-255`) DATA steps: pull `depth`/`extrinsic`/`intrinsic` from the window's `raw_lc`; `dense_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)`; `colors = (images·255).astype(uint8)`; store on the `Submap` via its dense fields (`points`, `colors`, `conf`) — use `Submap.add_all_points(...)` if present, else set the fields directly so `conf_threshold` is derived; `self.map.add_submap(submap)`. Reuse the existing forward — do NOT re-run inference. Do NOT touch `self.graph`, do NOT change the loop's PGO/output tail.
- [ ] **Step 4 PASS. Step 5 — full suite green** (old tail still drives output). **Step 6 commit:** `feat(loop_closure): add_points populates dense points+colors + self.map (port VGGT-SLAM data path)`.

**Carried into P4.3 (do here):** reset `self.map`/`self.graph` at the START of `_run_lc_loop` (they are created once at `__init__`; a second run on the same instance would otherwise reuse accumulated submaps/edges — review note from P4.1). Drive `self.graph.add_submap`/`add_loop_edge` **always passing `scale_method=cfg.scale_method` explicitly** (never the method default — the old shim defaulted `se3`, the methods default `rotation_only`; production is `rotation_only`).

### Task P4.3 — split into a/b/c (large, behavior-changing; reviewable commits)

#### P4.3a — drive `self.graph` incrementally (additive; parity anchor)
- [ ] **Step 1 — failing test:** after a stubbed LC run, `self.graph.extract_extrinsics(N)` equals the old `run_pose_graph_optimization(submaps, lc_submaps, N, overlap, scale_method=cfg.scale_method)` at atol=1e-5 (proves the incrementally-driven `self.graph` == the batch shim).
- [ ] **Step 2 FAIL. Step 3** — in `_run_lc_loop`/`add_points`: per window `self.graph.add_submap(submap, cfg.submap_overlap, scale_method=cfg.scale_method)` + `self.graph.optimize()`; after the window loop, for each verified `lc` submap `self.graph.add_loop_edge(lc, submaps_list, scale_method=cfg.scale_method)` then final `self.graph.optimize()` (deferred loop-edge timing = current repo default; the A/B in P7.2 may switch to live). **Always pass `scale_method=cfg.scale_method` explicitly.** Do NOT change the output tail yet — `run_pose_graph_optimization`/`merge_submap_outputs` still produce the output (self.graph is populated but unused for output this commit; brief redundant PGO is acceptable for one commit).
- [ ] **Step 4 PASS** (existing output tests unchanged — output still from old tail). **Step 5 commit:** `feat(loop_closure): drive self.graph incrementally per window (== batch shim, 1e-6)`.

#### P4.3b — flip output to GraphMap (behavior change)
- [ ] **Step 1 — failing test:** an LC run sets `self.base.outputs` (a `FeedforwardResult`) whose `points`/`colors` come from `self.map.get_world_pointcloud(self.graph)` and `extrinsics` from `self.map.get_corrected_extrinsics(self.graph, N)`.
- [ ] **Step 2 FAIL. Step 3** — assemble `FeedforwardResult` directly in the LC path from `GraphMap`: `points, colors = self.map.get_world_pointcloud(self.graph)`; `extrinsics = self.map.get_corrected_extrinsics(self.graph, N)`; gather `intrinsics` (dedup per-frame from submaps), `image_paths`, `original_coords`, `model_width/height` (from submap point grid / stored dims); set `self.base.outputs` and make the LC `run()`/`postprocess()` use this assembled result **instead of** `_postprocess(raw_outputs)`. Remove the `run_pose_graph_optimization` + `merge_submap_outputs` CALLS from `_run_lc_loop` (leave the functions defined for P4.3c). **`get_world_pointcloud` must tolerate `points=None` submaps** (skip them — MapAnything degraded path, P4.2 finding). Update the `_dedup_rows` handling in `run()` (GraphMap output is already N-frame, so the M→N remap likely drops — verify + remove if so). **Behavior change:** existing LC integration/output tests (`test_loop_closure_integration`, etc.) will churn — rewrite their assertions to the new GraphMap-sourced output; keep them meaningful (assert cloud non-empty, extrinsics shape N, colors aligned), not just loosened.
- [ ] **Step 4 PASS. Step 5** — full suite; fix churned tests. **Step 6 commit:** `refactor(loop_closure): LC output assembled from GraphMap dense cloud (VGGT-SLAM, decision 015 numerics)`.
- [ ] **Step 7 — single-backbone ATE sanity (tmux):** run vggtx on chess d5 through the new path; confirm ATE is in the ballpark of the frozen baseline (not a gross regression). Record the number. This is the fail-fast gate before P5/P6 — NOT the full P7 matrix.

#### P4.3c — delete the now-unused shims + dissolve merge.py (no-shim)
`merge.py` currently holds three functions: `merge_submap_outputs` (dead after P4.3b), plus `dedup_overlap` + `_resolve_frame_node` — the latter two are used **only by `graph.py`** (`extract_extrinsics` calls `dedup_overlap`; `add_loop_edge` calls `_resolve_frame_node`). No other consumer. So `merge.py` becomes pure indirection to a single caller → dissolve it into `graph.py` (also more VGGT-SLAM-faithful: upstream has no `merge.py`, pose reconstruction lives in the graph/map layer).
- [ ] **Step 1** — DELETE `run_pose_graph_optimization` (from `graph.py`) + `merge_submap_outputs` (from `merge.py`); both unused after P4.3b.
- [ ] **Step 2** — MOVE `dedup_overlap` + `_resolve_frame_node` from `merge.py` into `graph.py` (their sole caller), then DELETE `merge.py`. Update `graph.py`'s `from .merge import ...` (remove — now local). Update `__init__.py`: drop `merge_submap_outputs`; keep exporting `dedup_overlap` if desired (now from `graph`). Update `tests/geometry/loop_closure/test_alignment_dedup.py` import → `from collab_splats.geometry.loop_closure.graph import dedup_overlap`.
- [ ] **Step 3 — migrate/rewrite every direct-caller test** of the DELETED functions (`test_pgo_parity`, `test_hw_formula`, `test_loop_edge_chain`, `test_closure_split`, `test_graph`, `test_pose_extraction`, any `test_*merge*` testing `merge_submap_outputs`) to drive the incremental API directly (`pg.add_submap`/`add_loop_edge`/`optimize`/`extract_extrinsics`), preserving numerical assertions (incremental == old shim at 1e-6). `dedup_overlap` tests stay — just repoint the import. Grep `run_pose_graph_optimization\|merge_submap_outputs\|from .merge\|import merge` across `collab_splats/` + `tests/` → ZERO stragglers.
- [ ] **Step 4 — full suite green.** **Step 5 commit:** `refactor(loop_closure): delete run_pose_graph_optimization + merge_submap_outputs; dissolve merge.py into graph.py (no shim)`.

---

## Phase P5 — Lean RAM (streaming falls out)

**Order: P5.2 BEFORE P5.1.** Verify re-preprocesses the detected frame from `frames.zarr` on loop-verify; the current `_preprocess` takes file paths, so doing that without P5.2's in-memory `_preprocess(frames)` would reintroduce the temp-JPG hack P5.2 deletes. Re-preprocess cost is negligible (crop/resize ~ms/frame; verify already runs a model forward that dwarfs it; ~38 verifies/scene → ~0.1s total).

### Task P5.1a (was P5.2): `_preprocess(frames)` refactor + delete temp-JPG export hack (from old plan C1)

- [ ] Port the old-plan Task C1: `_preprocess(image_dir: Path)` → `_preprocess(frames, frame_idxs)` accepting in-memory decoded frames; caller slices the window from `FrameStore`; delete `_preprocess_from_store` + `_frame_export` temp-dir hack. Update all 3 creators (vggtx/vggt_omega/mapanything) — replace the `load_and_preprocess_images(paths)` file-path preamble with an in-memory-array path (keep the per-model crop/resize math). `image_paths` become `frame_idx` labels. Test `tests/pointcloud/feedforward/test_preprocess_frames.py`. Commit: `refactor(pointcloud): _preprocess takes decoded frames; delete temp-JPG hack`.
- [ ] Expose a reusable single-/multi-frame preprocess entry the wrapper can call for verify (e.g. `creator._preprocess(frame_array, frame_idxs)` works for 1 frame too) so P5.1b re-preprocesses in-memory.

### Task P5.1b: free per-submap `raw_outputs` after unproject (VGGT-SLAM memory profile)

**Scope (user decision A):** match VGGT-SLAM's actual memory profile — `add_points` immediately unprojects depth→dense points/colors/conf and DISCARDS the raw prediction dict; frames stay RETAINED (VGGT-SLAM keeps `submap.frames` for loop verify). So this task ONLY frees `raw_outputs`; it does NOT free frames, add a re-preprocess-on-verify path, or a store-conditional branch (those were beyond upstream + marginal RAM — dropped). `raw_outputs` (full depth+images+conf per frame) is the OOM driver; freeing it is ~all the win.

- [ ] **Step 1 — failing test:** after a stubbed LC run, every retained `Submap` in `self.map` has `raw_outputs is None` (freed post-unproject); `frames` remain non-None (retained for verify); the output (from GraphMap) is unchanged.
- [ ] **Step 2 FAIL. Step 3** — in `run_predictions`/`add_points`, after the dense unproject (P4.2) has extracted points/colors/conf from `raw_lc`, set `submap.raw_outputs = None` (and drop any transient `self.base.raw_outputs` reference the LC path no longer needs). VERIFY nothing reads `submap.raw_outputs` after unproject on the LC path (P4.3b assembles output from GraphMap, `_postprocess` is bypassed, `merge_submap_outputs` + `_dedup_rows` are gone). Keep `submap.frames` intact.
- [ ] **Step 4 PASS. Step 5** — full suite green; the omega-chess ATE sanity MUST still be 0.0267 / 0.0152 / 38 loops (freeing raw_outputs changes nothing about the geometry — it's freed only after everything that needs it has run). **Step 6 commit:** `feat(loop_closure): free per-submap raw_outputs after unproject (VGGT-SLAM memory profile)`.

---

## Phase P6 — Viewer wiring + config + example

- [ ] **P6.1 `Viewer.serve_forever` keep-alive** (old-plan E1) + guarded hooks in `_run_lc_loop` reusing `subsample_points`/`Viewer.add_points`/`add_frustum`/`add_lines` (old-plan E2). Now sourced from `GraphMap`/`Submap` reads.
- [ ] **P6.2 base.yaml keys** (`viz.enabled`, `viz.port`; `loop_closure` stays a bool + existing `LoopClosureConfig` fields) + strict plumbing (old-plan F1). NOTE: no `streaming`/`keep_submaps` keys (no disk store).
- [ ] **P6.3 `docs/examples/run_scenes.py`** runner with `--keep-viewer` (old-plan F2).

---

## Phase P7 — Validation across 4 backbones + A/B loop-edge timing

- [ ] **P7.1 System parity:** run vggtx / vggt_omega / mapanything / vggt_spark through the unified LC path on chess d5 (+ TUM fr3); assert ATE reproduces the frozen baselines within tolerance; confirm per-model calibrations intact (`project_lc_layer_calibration`). tmux (heavy). Record numbers in a decision-doc appendix.
- [ ] **P7.2 A/B loop-edge timing** (deferred vs live) — the disposable-worktree eval from the old plan Phase G; feed the verdict into the loop-edge insertion point in `add_loop_edge` application.
- [ ] **P7.3 Memory check:** >1k-keyframe run, assert lean RAM (no frames/raw_outputs resident) stays well under cap. tmux.
- [ ] **P7.4 Docs/memory:** update `CLAUDE.md` In-Flight + a `project_lc_vggtslam_alignment` memory (unified path, GraphMap, no-shim, validation model, A/B verdict).

---

## Self-review notes
- **Order matters:** P1 (Submap) → P2 (GraphMap+merge delete) → P3 (PoseGraph rename + shim delete) → P4 (LoopClosure/Solver shape) → P5 (lean RAM) → P6 (viewer/config) → P7 (validate). Each phase keeps the LC suite green (P2/P3 rewrite the deleted-shim tests in the same commit).
- **No-shim discipline:** P2.2 and P3.2 delete functions and migrate/rewrite callers in the same task — never leave a thin wrapper.
- **Phase A reused:** `add_submap`/`add_loop_edge`/`extract_extrinsics` survive; only the `run_pose_graph_optimization` wrapper is deleted (P3.2).
- **Risk:** validation is system-level (ATE × 4 backbones), not 1e-6 unit — per decision 015.
