# Windowed Streaming Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process arbitrarily long videos window-by-window with cross-window loop closure, RAM bounded by window size (not scene length), and real-time viser visualization — final output identical to the current batch path.

**Architecture:** Refactor the existing loop-closure loop (`_run_lc_loop`) into a memory-streaming, disk-backed path. Decompose the monolithic `run_pose_graph_optimization` into incremental `PoseGraph` methods (persistent graph fed one submap at a time), spill each submap's dense payload to a dedicated `submaps.zarr`, retain only descriptors + overlap points in RAM, and wire the existing `Viewer`. No new solver, no new viewer, no online iSAM.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), zarr v3 (`BloscCodec`), GTSAM (SL(4) PGO), viser, pytest. Reuses `FrameStore`, `PoseGraph`, `Submap`, `merge_submap_outputs`, `subsample_points`, `Viewer`, `find_loop_closures`, `_verify_loop_candidate`.

**Spec:** `docs/superpowers/specs/2026-07-20-windowed-streaming-reconstruction-design.md`

---

## Environment / conventions (read once)

- **Python:** `/opt/venv/reconstruction/bin/python` (py3.11, editable install — no PYTHONPATH).
- **Tests:** `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
- **Format:** `black . && isort .` before every commit.
- **Heavy inference/eval:** tmux only (46.6 GB cap; OOM risk). Never in a notebook or a side shell during an eval.
- **Do NOT stage** `collab_splats/geometry/loop_closure/merge.py` (concurrent session owns an uncommitted change there). Task B4 modifies `merge.py` — coordinate: only stage the specific hunk this plan adds, or rebase onto their commit first. If `merge.py` still shows uncommitted foreign changes at Task B4, STOP and ask.
- **Worktrees:** `.worktrees/` is gitignored. `third_party/` is gitignored — after creating a worktree, symlink it in or tests won't collect (see Task 0).
- **zarr v3 API:** `store.create_array(...)`, `compressors=[BloscCodec(cname="lz4")]` (NOT `codecs=`). See memory `feedback_zarr_v3`.
- **Config:** `configs/base.yaml` is the SOLE default source. Strict access — no inline `.get(key, default)`. `tests/wrapper/test_reconstructor.py` has a `test_guard` that fails on defaults added in Python.

---

## File Structure

**New files:**
- `collab_splats/geometry/loop_closure/submap_store.py` — `submaps.zarr` spill / reload / resume helpers. One responsibility: submap ↔ disk.
- `docs/examples/run_scenes.py` — example runner for a >1k-frame video (streaming config + `--keep-viewer`).
- `tests/geometry/loop_closure/test_submap_store.py`
- `tests/geometry/loop_closure/test_pose_graph_incremental.py`
- `tests/geometry/loop_closure/test_streaming_parity.py`
- `tests/pointcloud/feedforward/test_preprocess_frames.py`

**Modified files:**
- `collab_splats/geometry/loop_closure/graph.py` — add `PoseGraph.add_submap` / `add_loop_edge` / `extract_extrinsics`; `run_pose_graph_optimization` becomes a thin shim over them.
- `collab_splats/geometry/loop_closure/wrapper.py` — `_run_lc_loop` streaming restructure; `LoopClosureConfig` gains `streaming` + `keep_submaps`; viewer wiring.
- `collab_splats/geometry/loop_closure/merge.py` — `merge_submap_outputs` gains a disk-streaming variant (see foreign-change caveat above).
- `collab_splats/pointcloud/feedforward/base.py` — `_preprocess(frames)` refactor; delete `_preprocess_from_store` + `_frame_export`.
- `collab_splats/pointcloud/feedforward/{vggtx,vggt_omega,mapanything}.py` — `_preprocess` signature `image_dir: Path` → `frames`.
- `collab_splats/viewer.py` — `serve_forever()` keep-alive helper.
- `configs/base.yaml` — 4 new keys (`loop_closure.streaming`, `loop_closure.keep_submaps`, `viz.enabled`, `viz.port`).

**Disposable (Track 1, own worktree — never merged):**
- A `loop_edge_timing` param on `run_pose_graph_optimization` + an eval driver. Produces the deferred-vs-live decision only.

---

## Phasing & parallelism

Two tracks (spec §Work parallelization):

- **Track 2 (main worktree)** — Phases A–F below. Timing-independent. The bulk of the work.
- **Track 1 (disposable worktree)** — Phase G. The A/B gate. Runs concurrently; only its *decision* feeds Task D8.

**Sync point:** Task D8 (loop-edge insertion cadence) consumes the Phase G decision. Keep Task D8 last and parameterized. Everything before it is timing-independent.

Recommended order: A → B → C → (D1–D7, E, F in any order) → [await G] → D8.

---

## Phase 0: Worktree setup

### Task 0: Create the Track 2 worktree

**Files:** none (git plumbing)

- [ ] **Step 1: Create the worktree + branch off the current branch**

```bash
cd /workspace/collab-splats
git worktree add .worktrees/streaming -b feat/windowed-streaming
```

- [ ] **Step 2: Symlink gitignored third_party + configs deps so tests collect**

```bash
cd /workspace/collab-splats/.worktrees/streaming
ln -s /workspace/collab-splats/third_party third_party
# graphify-out is gitignored but read by hooks; symlink so graphify works
ln -s /workspace/collab-splats/graphify-out graphify-out 2>/dev/null || true
```

- [ ] **Step 3: Verify the suite collects + is green from this worktree**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q --co -q | tail -5`
Expected: tests collect with no import errors (localization/creator tests need `third_party`).

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q`
Expected: PASS (baseline green before any change).

---

## Phase A: PoseGraph decomposition (pure refactor — parity-preserving)

**Goal:** Split the monolith `run_pose_graph_optimization` (`graph.py:417-626`) into incremental `PoseGraph` methods, then rewrite the monolith as a thin shim. Behavior byte-identical — all ~12 existing callers + `test_closure_split` (asserts unchanged signature) stay green. This is the foundation streaming reuses.

### Task A1: Characterization test — lock current PGO output before refactoring

**Files:**
- Test: `tests/geometry/loop_closure/test_pose_graph_incremental.py` (Create)

- [ ] **Step 1: Write a golden-output test on the existing monolith**

```python
"""Incremental PoseGraph == monolith run_pose_graph_optimization (parity lock)."""
import numpy as np
import pytest

from collab_splats.geometry.loop_closure.graph import run_pose_graph_optimization
from collab_splats.geometry.loop_closure.submap import Submap


def _regular_submap(sid: int, k: int, seed: int) -> Submap:
    """Deterministic submap: identity-ish poses + random dense points."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    # small forward translation per frame so inter-frame relatives are non-trivial
    for i in range(k):
        poses[i, :3, 3] = [0.0, 0.0, 0.1 * (sid * k + i)]
    P = 64
    return Submap(
        submap_id=sid,
        frames=np.zeros((k, 3, 4, 4), dtype=np.float32),
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=rng.standard_normal((k, 8)).astype(np.float32),
        image_paths=[f"s{sid}_f{i}.jpg" for i in range(k)],
        world_points=rng.standard_normal((k, P, 3)).astype(np.float32),
        world_points_conf=np.full((k, P), 50.0, dtype=np.float32),
        frame_start=sid * k,
    )


@pytest.fixture
def two_submaps():
    return [_regular_submap(0, 4, 1), _regular_submap(1, 4, 2)]


def test_monolith_golden_shape(two_submaps):
    out = run_pose_graph_optimization(
        two_submaps, lc_submaps=[], total_frames=8, overlap_frames=1
    )
    assert out.shape == (8, 4, 4)
    # first node is identity (prior)
    np.testing.assert_allclose(out[0], np.eye(4), atol=1e-5)
```

- [ ] **Step 2: Run — verify it passes on the CURRENT monolith**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_pose_graph_incremental.py -q`
Expected: PASS (this characterizes existing behavior; it must stay green through Task A4).

- [ ] **Step 3: Commit**

```bash
black . && isort .
git add tests/geometry/loop_closure/test_pose_graph_incremental.py
git commit -m "test(geometry): characterize run_pose_graph_optimization before refactor"
```

### Task A2: Add `PoseGraph.add_submap` (hoist graph.py:445-575)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/graph.py` (add method to `PoseGraph` class, ~line 154 after `get_homography`)
- Test: `tests/geometry/loop_closure/test_pose_graph_incremental.py`

- [ ] **Step 1: Write the failing test**

Append to `test_pose_graph_incremental.py`:

```python
from collab_splats.geometry.loop_closure.graph import PoseGraph


def test_add_submap_matches_monolith_sequential(two_submaps):
    """Incremental add_submap + optimize per submap == monolith (no loops)."""
    golden = run_pose_graph_optimization(
        two_submaps, lc_submaps=[], total_frames=8, overlap_frames=1
    )

    pg = PoseGraph()
    for s in two_submaps:
        pg.add_submap(s, overlap_frames=1, conf_threshold=25.0, scale_method="rotation_only")
        pg.optimize()
    incremental = pg.extract_extrinsics(total_frames=8)

    np.testing.assert_allclose(incremental, golden, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_pose_graph_incremental.py::test_add_submap_matches_monolith_sequential -q`
Expected: FAIL — `AttributeError: 'PoseGraph' object has no attribute 'add_submap'`.

- [ ] **Step 3: Add per-submap state + `add_submap` to `PoseGraph.__init__` and the class**

In `graph.py`, extend `PoseGraph.__init__` (line 105) to hold the incremental bookkeeping the monolith kept in locals:

```python
    def __init__(self) -> None:
        # ... existing gtsam graph/values init ...
        self._global_node_id = 0
        self._frame_to_node: dict[tuple[int, int], int] = {}
        self._submap_node_ids: dict[int, list[int]] = {}
        self._submaps_seen: list[Submap] = []  # for inter-submap overlap lookups
```

Add the method (body is `graph.py:445-575` per-submap block, with `s_idx == 0` replaced by "first submap seen" and `submaps[s_idx-1]` replaced by `self._submaps_seen[-1]`):

```python
    def add_submap(
        self,
        submap: "Submap",
        overlap_frames: int,
        conf_threshold: float = 25.0,
        scale_method: str = "rotation_only",
    ) -> None:
        """Add one submap's nodes + sequential SL(4) edges to the persistent graph.

        Incremental form of run_pose_graph_optimization's per-submap block
        (graph.py:445-575). Inter-submap scale reads the previous submap's overlap
        world_points; first submap gets the identity prior.
        """
        # ... hoisted per-submap body: build K_4x4, first-submap prior branch vs
        # inter-submap H_w (scale estimation over overlap world_points), inner
        # sequential edges. Uses self._global_node_id / _frame_to_node /
        # _submap_node_ids / _submaps_seen instead of the monolith's locals. ...
        self._submaps_seen.append(submap)
```

**Note to implementer:** copy the monolith block verbatim, then mechanically substitute: `global_node_id` → `self._global_node_id`, `frame_to_node` → `self._frame_to_node`, `submap_node_ids` → `self._submap_node_ids`, `s_idx == 0` → `not self._submaps_seen`, `submaps[s_idx - 1]` → `self._submaps_seen[-1]`. Do NOT call `pg.optimize()` inside — the caller drives cadence.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_pose_graph_incremental.py -q`
Expected: PASS (both tests — sequential parity holds).

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/graph.py tests/geometry/loop_closure/test_pose_graph_incremental.py
git commit -m "feat(geometry): PoseGraph.add_submap — incremental per-submap graph build"
```

### Task A3: Add `PoseGraph.add_loop_edge` + `extract_extrinsics` (hoist graph.py:581-624)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/graph.py`
- Test: `tests/geometry/loop_closure/test_pose_graph_incremental.py`

- [ ] **Step 1: Write the failing test (with a loop)**

Append:

```python
def _lc_submap(sid: int, q_path: str, d_path: str) -> Submap:
    """2-frame loop-closure submap tying q_path→d_path."""
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    return Submap(
        submap_id=sid,
        frames=np.zeros((2, 3, 4, 4), dtype=np.float32),
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        retrieval_vectors=np.zeros((2, 8), dtype=np.float32),
        image_paths=[q_path, d_path],
        is_lc_submap=True,
        world_points=np.zeros((2, 64, 3), dtype=np.float32),
        world_points_conf=np.full((2, 64), 50.0, dtype=np.float32),
    )


def test_add_loop_edge_matches_monolith(two_submaps):
    """Incremental build + deferred loop edges == monolith with the same loop."""
    lc = _lc_submap(99, "s1_f1.jpg", "s0_f1.jpg")
    golden = run_pose_graph_optimization(
        two_submaps, lc_submaps=[lc], total_frames=8, overlap_frames=1
    )

    pg = PoseGraph()
    for s in two_submaps:
        pg.add_submap(s, overlap_frames=1)
        pg.optimize()
    pg.add_loop_edge(lc, self_submaps=two_submaps, conf_threshold=25.0, scale_method="rotation_only")
    pg.optimize()
    incremental = pg.extract_extrinsics(total_frames=8)

    np.testing.assert_allclose(incremental, golden, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_pose_graph_incremental.py::test_add_loop_edge_matches_monolith -q`
Expected: FAIL — `AttributeError: ... 'add_loop_edge'`.

- [ ] **Step 3: Add `add_loop_edge` + `extract_extrinsics`**

`add_loop_edge` body = the per-`lc` block from `graph.py:581-624`. It needs the resolved frame→node map (`self._frame_to_node`) and the submap list for `_resolve_frame_node` / `_lc_anchor_scale` — pass the caller's submap list as `self_submaps`:

```python
    def add_loop_edge(
        self,
        lc: "Submap",
        self_submaps: list["Submap"],
        conf_threshold: float = 25.0,
        scale_method: str = "rotation_only",
    ) -> None:
        """Add one verified loop closure's 3-edge SL(4) chain (graph.py:581-624)."""
        # ... hoisted per-lc block: resolve q/d nodes via self._frame_to_node,
        # anchor scales, _loop_chain_relatives, two graph-only LC nodes, three
        # sequential edges. Uses self._global_node_id / _frame_to_node. ...

    def extract_extrinsics(self, total_frames: int) -> np.ndarray:
        """Read optimized per-frame w2c homographies → (total_frames, 4, 4)."""
        # ... the monolith's tail: for each (submap_id, local_i) in frame_to_node,
        # decompose get_homography(nid) → w2c pose. Mirror graph.py output convention. ...
```

**Note:** if `total_frames == 0` or no submaps were added, return `np.tile(np.eye(4), (total_frames, 1, 1))` (matches monolith `graph.py:437-438`).

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_pose_graph_incremental.py -q`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/graph.py tests/geometry/loop_closure/test_pose_graph_incremental.py
git commit -m "feat(geometry): PoseGraph.add_loop_edge + extract_extrinsics"
```

### Task A4: Rewrite `run_pose_graph_optimization` as a thin shim

**Files:**
- Modify: `collab_splats/geometry/loop_closure/graph.py:417-626`
- Test: existing suite (`test_graph`, `test_pgo_parity`, `test_hw_formula`, `test_loop_edge_chain`, `test_closure_split`, `test_pose_extraction`)

- [ ] **Step 1: Replace the monolith body with the shim**

```python
def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
    conf_threshold: float = 25.0,
    scale_method: Literal["se3", "rotation_only", "pairwise_dist", "none"] = "rotation_only",
    debug_out: list | None = None,
) -> np.ndarray:
    """Build + optimize per-frame SL(4) pose graph; return (total_frames, 4, 4).

    Thin batch shim over the incremental PoseGraph API: adds each submap +
    optimizes (VGGT-SLAM per-submap cadence), then all loop edges, then a final
    solve. Signature preserved for existing callers + the parity harness.
    """
    if not submaps:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

    pg = PoseGraph()
    for submap in submaps:
        pg.add_submap(submap, overlap_frames, conf_threshold, scale_method, debug_out=debug_out)
        pg.optimize()
    for lc in lc_submaps:
        pg.add_loop_edge(lc, submaps, conf_threshold, scale_method)
    pg.optimize()
    return pg.extract_extrinsics(total_frames)
```

**Note:** thread `debug_out` through `add_submap` (the monolith appended per-submap debug dicts at `graph.py:548-557`) so `test_hw_formula` / debug callers keep working. Add `debug_out: list | None = None` param to `add_submap`.

- [ ] **Step 2: Run the full loop-closure + pose suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ tests/pointcloud/test_pose_extraction.py -q`
Expected: PASS — all pre-existing tests (parity, hw_formula, loop_edge_chain, closure_split signature guard) green. This proves the refactor is behavior-preserving.

- [ ] **Step 3: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/graph.py
git commit -m "refactor(geometry): run_pose_graph_optimization → thin shim over PoseGraph methods"
```

---

## Phase B: submaps.zarr spill / reload / resume

**Goal:** A dedicated `submaps.zarr` store: spill a submap's dense payload, reload one submap, list present submaps for resume, delete the store. Own file, no dependency on the LC loop.

### Task B1: `SubmapStore.create` + `spill` + round-trip

**Files:**
- Create: `collab_splats/geometry/loop_closure/submap_store.py`
- Test: `tests/geometry/loop_closure/test_submap_store.py`

- [ ] **Step 1: Write the failing test**

```python
"""submaps.zarr spill / reload / resume round-trip."""
import numpy as np
import pytest

from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.loop_closure.submap_store import SubmapStore


def _submap(sid: int, k: int = 3, P: int = 32) -> Submap:
    rng = np.random.default_rng(sid)
    return Submap(
        submap_id=sid,
        frames=np.zeros((k, 3, 4, 4), dtype=np.float32),
        poses=np.tile(np.eye(4, dtype=np.float32), (k, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=rng.standard_normal((k, 8)).astype(np.float32),
        image_paths=[f"f{sid}_{i}.jpg" for i in range(k)],
        world_points=rng.standard_normal((k, P, 3)).astype(np.float32),
        world_points_conf=np.full((k, P), 42.0, dtype=np.float32),
        frame_start=sid * k,
    )


def test_spill_reload_roundtrip(tmp_path):
    store = SubmapStore.create(tmp_path / "submaps.zarr", submap_size=3, submap_overlap=1)
    s = _submap(0)
    store.spill(s)

    back = store.reload(0)
    np.testing.assert_allclose(back.poses, s.poses)
    np.testing.assert_allclose(back.world_points, s.world_points)
    np.testing.assert_allclose(back.world_points_conf, s.world_points_conf)
    np.testing.assert_allclose(back.intrinsics, s.intrinsics)
    assert back.submap_id == 0
    assert back.frame_start == 0
    assert list(back.image_paths) == s.image_paths
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_submap_store.py -q`
Expected: FAIL — `ModuleNotFoundError: ... submap_store`.

- [ ] **Step 3: Implement `SubmapStore` (create + spill + reload)**

```python
"""Dedicated submaps.zarr checkpoint store: spill/reload/resume for streaming LC.

Sibling of frames.zarr / feedforward.zarr. Keeps each submap's dense payload on
disk so the streaming loop retains only descriptors + overlap points in RAM.
Kept by default as a resume/re-opt checkpoint (see spec §submaps.zarr).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .submap import Submap

_SCHEMA_VERSION = 1
_ARRAYS = ("poses", "intrinsics", "world_points", "world_points_conf")


class SubmapStore:
    """Read/write per-submap dense payloads in a dedicated submaps.zarr group."""

    def __init__(self, path: Path, root) -> None:
        self.path = Path(path)
        self._root = root

    @classmethod
    def create(cls, path, submap_size: int, submap_overlap: int) -> "SubmapStore":
        """Create an empty submaps.zarr with run metadata; overwrites any existing."""
        path = Path(path)
        root = zarr.open_group(str(path), mode="w")
        root.attrs["schema_version"] = _SCHEMA_VERSION
        root.attrs["submap_size"] = int(submap_size)
        root.attrs["submap_overlap"] = int(submap_overlap)
        root.attrs["complete"] = False
        return cls(path, root)

    @classmethod
    def open(cls, path) -> "SubmapStore":
        """Open an existing submaps.zarr (resume)."""
        path = Path(path)
        root = zarr.open_group(str(path), mode="a")
        return cls(path, root)

    def spill(self, submap: Submap) -> None:
        """Write one submap's dense arrays + metadata to submaps.zarr/submap_NNN."""
        name = f"submap_{submap.submap_id:04d}"
        g = self._root.create_group(name, overwrite=True)
        comp = [BloscCodec(cname="lz4")]
        for key in _ARRAYS:
            arr = getattr(submap, key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            z = g.create_array(key, shape=arr.shape, dtype=arr.dtype, compressors=comp)
            z[:] = arr
        # Retrieval vectors are small; keep them for resume-time re-detection.
        rv = np.asarray(submap.retrieval_vectors)
        z = g.create_array("retrieval_vectors", shape=rv.shape, dtype=rv.dtype, compressors=comp)
        z[:] = rv
        g.attrs["submap_id"] = int(submap.submap_id)
        g.attrs["frame_start"] = int(submap.frame_start)
        g.attrs["image_paths"] = [str(p) for p in submap.image_paths]
        g.attrs["is_lc_submap"] = bool(getattr(submap, "is_lc_submap", False))

    def reload(self, submap_id: int) -> Submap:
        """Read one submap back from disk into a Submap (frames NOT restored)."""
        g = self._root[f"submap_{submap_id:04d}"]
        data = {k: g[k][:] for k in _ARRAYS if k in g}
        return Submap(
            submap_id=int(g.attrs["submap_id"]),
            frames=None,
            poses=data["poses"],
            intrinsics=data["intrinsics"],
            retrieval_vectors=g["retrieval_vectors"][:],
            image_paths=list(g.attrs["image_paths"]),
            world_points=data.get("world_points"),
            world_points_conf=data.get("world_points_conf"),
            frame_start=int(g.attrs["frame_start"]),
            is_lc_submap=bool(g.attrs["is_lc_submap"]),
        )
```

**Note:** confirm the `Submap` dataclass accepts `frames=None` (reload does not restore frames — the graph/merge paths read poses/points, not frames). If `frames` is non-optional, make it `Optional` in `submap.py` with a one-line comment "None after disk reload — frames live only in frames.zarr".

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_submap_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/submap_store.py tests/geometry/loop_closure/test_submap_store.py
git commit -m "feat(geometry): SubmapStore — submaps.zarr spill/reload round-trip"
```

### Task B2: Resume support — `present_ids` + `mark_complete` + `delete`

**Files:**
- Modify: `collab_splats/geometry/loop_closure/submap_store.py`
- Test: `tests/geometry/loop_closure/test_submap_store.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_present_ids_and_resume(tmp_path):
    p = tmp_path / "submaps.zarr"
    store = SubmapStore.create(p, submap_size=3, submap_overlap=1)
    store.spill(_submap(0))
    store.spill(_submap(1))
    assert store.present_ids() == [0, 1]
    assert not store.is_complete()

    store.mark_complete()
    # reopen (simulate a fresh process resuming)
    store2 = SubmapStore.open(p)
    assert store2.present_ids() == [0, 1]
    assert store2.is_complete()


def test_delete_removes_store(tmp_path):
    p = tmp_path / "submaps.zarr"
    store = SubmapStore.create(p, submap_size=3, submap_overlap=1)
    store.spill(_submap(0))
    store.delete()
    assert not p.exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_submap_store.py -k "resume or delete" -q`
Expected: FAIL — `AttributeError: ... 'present_ids'`.

- [ ] **Step 3: Implement**

```python
    def present_ids(self) -> list[int]:
        """Sorted submap ids already spilled (resume: skip these windows)."""
        ids = [
            int(k.split("_")[1])
            for k in self._root.keys()
            if k.startswith("submap_")
        ]
        return sorted(ids)

    def is_complete(self) -> bool:
        """True once mark_complete ran — the run finished spilling all windows."""
        return bool(self._root.attrs.get("complete", False))

    def mark_complete(self) -> None:
        """Flag the store as fully spilled (final-merge / resume checkpoint)."""
        self._root.attrs["complete"] = True

    def delete(self) -> None:
        """Remove submaps.zarr from disk (keep_submaps=False after merge)."""
        shutil.rmtree(self.path, ignore_errors=True)
```

**Note:** `is_complete` is the one place an inline `.get` default is acceptable — it's zarr attrs, not the base.yaml config guarded by `test_reconstructor`. Leave a comment saying so.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_submap_store.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/submap_store.py tests/geometry/loop_closure/test_submap_store.py
git commit -m "feat(geometry): SubmapStore resume (present_ids/complete) + delete"
```

### Task B3: Register `SubmapStore` in the package `__init__`

**Files:**
- Modify: `collab_splats/geometry/loop_closure/__init__.py`

- [ ] **Step 1: Add the export**

Add `SubmapStore` to the imports + `__all__` in `collab_splats/geometry/loop_closure/__init__.py` (mirror how `run_pose_graph_optimization` is exported at `__init__.py:6,35`).

- [ ] **Step 2: Verify import**

Run: `/opt/venv/reconstruction/bin/python -c "from collab_splats.geometry.loop_closure import SubmapStore; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/geometry/loop_closure/__init__.py
git commit -m "feat(geometry): export SubmapStore"
```

### Task B4: `merge_submap_outputs` — stream submaps from disk

**Files:**
- Modify: `collab_splats/geometry/loop_closure/merge.py:70` (⚠ foreign-change caveat — see Environment)
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Confirm merge.py has no foreign uncommitted change**

Run: `git status --short collab_splats/geometry/loop_closure/merge.py`
Expected: clean (empty). If it shows `M`, STOP — the concurrent session owns it; ask the user before proceeding.

- [ ] **Step 2: Write the failing test — disk merge == in-memory merge**

```python
"""Merge-from-disk == in-memory merge (streaming parity building block)."""
import numpy as np

from collab_splats.geometry.loop_closure.merge import merge_submap_outputs, merge_submap_outputs_from_store
from collab_splats.geometry.loop_closure.submap_store import SubmapStore
from tests.geometry.loop_closure.test_submap_store import _submap  # reuse factory


def test_merge_from_disk_equals_in_memory(tmp_path):
    submaps = [_submap(0), _submap(1)]
    corrected = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))

    in_mem = merge_submap_outputs(submaps, corrected)

    store = SubmapStore.create(tmp_path / "submaps.zarr", submap_size=3, submap_overlap=1)
    for s in submaps:
        store.spill(s)
    from_disk = merge_submap_outputs_from_store(store, [0, 1], corrected)

    for key in in_mem:
        if isinstance(in_mem[key], np.ndarray):
            np.testing.assert_allclose(from_disk[key], in_mem[key], atol=1e-6)
```

- [ ] **Step 3: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py::test_merge_from_disk_equals_in_memory -q`
Expected: FAIL — `ImportError: cannot import name 'merge_submap_outputs_from_store'`.

- [ ] **Step 4: Add the disk-streaming wrapper (reuses the in-memory core)**

In `merge.py`, add a thin wrapper that reloads submaps one at a time and delegates to the existing `merge_submap_outputs` (do NOT duplicate the merge math):

```python
def merge_submap_outputs_from_store(store, submap_ids, corrected_extrinsics, graph=None):
    """merge_submap_outputs, streaming submaps from a SubmapStore instead of RAM.

    Reloads one submap at a time (bounded RAM), then reuses the in-memory merge.
    Identical output to merge_submap_outputs given the same submaps + poses.
    """
    submaps = [store.reload(sid) for sid in submap_ids]
    return merge_submap_outputs(submaps, corrected_extrinsics, graph=graph)
```

**Note:** if the in-memory `merge_submap_outputs` needs `frames` (it should not — it merges points/poses), and reload sets `frames=None`, verify no `frames` access. If it does touch frames, that is a real coupling — surface it, do not paper over.

- [ ] **Step 5: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py::test_merge_from_disk_equals_in_memory -q`
Expected: PASS.

- [ ] **Step 6: Commit (ONLY the merge.py hunk this task added)**

```bash
black . && isort .
git add -p collab_splats/geometry/loop_closure/merge.py   # stage only the new function
git add tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "feat(geometry): merge_submap_outputs_from_store — stream submaps from disk"
```

---

## Phase C: `_preprocess(frames)` refactor + delete export hack

**Goal:** `_preprocess` accepts decoded frames instead of a directory; caller reads windows from `FrameStore`. Deletes `_preprocess_from_store` + `_frame_export` temp-JPG round-trip.

### Task C1: Refactor `BaseFeedforwardCreator._preprocess` contract to frames

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`_preprocess` abstract sig line 870; `setup_inference` 792-803; delete `_preprocess_from_store` 805-812; drop `_frame_export` field 753)
- Modify: `collab_splats/pointcloud/feedforward/{vggtx,vggt_omega,mapanything}.py` (`_preprocess` impls)
- Test: `tests/pointcloud/feedforward/test_preprocess_frames.py`

- [ ] **Step 1: Write the failing test (frames in, views + idx labels out)**

```python
"""_preprocess accepts decoded frames; FrameStore is the sole IO path."""
import numpy as np

from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator


def test_preprocess_accepts_frames_array():
    creator = VGGTXCreator()
    # 2 synthetic RGB frames (H, W, 3) uint8 — no disk involved
    frames = np.zeros((2, 64, 96, 3), dtype=np.uint8)
    frame_idxs = [10, 20]
    views, labels, coords = creator._preprocess(frames, frame_idxs=frame_idxs)
    assert views.shape[0] == 2               # one view per frame
    assert [int(x) for x in labels] == [10, 20]  # frame_idx labels, not paths
    assert coords.shape == (2, 6)            # original_coords unchanged contract
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_preprocess_frames.py -q`
Expected: FAIL — current `_preprocess(image_dir)` signature rejects an array / `frame_idxs` kwarg.

- [ ] **Step 3: Change the abstract contract + `setup_inference`**

In `base.py`:
- Change abstract sig (line 870) to `def _preprocess(self, frames, frame_idxs) -> tuple[Any, list, np.ndarray]: ...` — `frames` is `(N, H, W, 3)` uint8, `frame_idxs` are the source video indices used as labels.
- Rewrite `setup_inference` (792-803) to read from `FrameStore` in the caller and pass arrays:

```python
    def setup_inference(self, source: FrameStore | Path) -> None:
        """Load frames (from a FrameStore/zarr, or a legacy image dir) into self.views."""
        t0 = time.perf_counter()
        console.log("Preprocessing images...")
        store = self._as_frame_store(source)          # FrameStore | None
        if store is not None:
            frames = store.images()                    # (N, H, W, 3) — decode-once
            frame_idxs = list(store.frame_indices())
        else:
            frames, frame_idxs = _load_dir_as_frames(Path(source))  # legacy dir fallback
        self.views, self.image_paths, self.original_coords = self._preprocess(frames, frame_idxs)
        console.log(f"  → {len(self.image_paths)} images  done in {time.perf_counter() - t0:.1f}s")
```

- Add a small `_as_frame_store(source)` helper (FrameStore passthrough / `.zarr` → `FrameStore.open` / else None) and a `_load_dir_as_frames(dir)` helper (decode a legacy image dir to an array + integer labels) so per-model `_preprocess` never touches disk.
- **Delete** `_preprocess_from_store` (805-812) and the `_frame_export` field (753) + any `tempfile` import left unused.

- [ ] **Step 4: Update the 3 creator `_preprocess` impls**

For each of `vggtx.py`, `vggt_omega.py`, `mapanything.py`: change `_preprocess(self, image_dir: Path)` to `_preprocess(self, frames, frame_idxs)`. Replace the "collect + sort + read image files" preamble with "iterate the given `frames` array"; keep the per-model crop/resize math verbatim. Return `image_paths` as the `frame_idxs` labels (wrap as `str` if downstream expects path-like, but prefer int labels per Spec 1).

**Note:** `build_colmap` uses `p.name for p in o.image_paths` (`base.py:842`). Update it to accept integer/`frame_idx` labels — format as `f"frame_{idx:06d}"` so COLMAP image names stay stable. Check `_write_transforms` similarly.

- [ ] **Step 5: Run the focused test + the creator suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/ -q`
Expected: PASS (new frames test + existing creator tests; the latter need `third_party` symlinked).

- [ ] **Step 6: Commit**

```bash
black . && isort .
git add collab_splats/pointcloud/feedforward/ tests/pointcloud/feedforward/test_preprocess_frames.py
git commit -m "refactor(pointcloud): _preprocess takes decoded frames; delete temp-JPG export hack"
```

---

## Phase D: `_run_lc_loop` streaming restructure

**Goal:** Rewrite `_run_lc_loop` (`wrapper.py:369-437`) to preprocess-per-window, spill to `SubmapStore`, free the window, feed the persistent `PoseGraph`, and merge-from-disk. Add `streaming` + `keep_submaps` to `LoopClosureConfig`. Resume support. Loop-edge cadence (Task D8) is last, gated on Phase G.

### Task D1: Add `streaming` + `keep_submaps` to `LoopClosureConfig`

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py:50-71`
- Test: `tests/geometry/loop_closure/test_wrapper.py`

- [ ] **Step 1: Write the failing test**

```python
def test_loopclosure_config_streaming_defaults():
    from collab_splats.geometry.loop_closure.wrapper import LoopClosureConfig
    cfg = LoopClosureConfig()
    assert cfg.streaming is True
    assert cfg.keep_submaps is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_wrapper.py::test_loopclosure_config_streaming_defaults -q`
Expected: FAIL — `AttributeError: ... 'streaming'`.

- [ ] **Step 3: Add the fields**

In `LoopClosureConfig` (after `conf_threshold`, line 71):

```python
    # Streaming: spill each submap to submaps.zarr + merge-from-disk (RAM bounded by
    # window). False = the legacy batch path (whole list[Submap] held in RAM).
    streaming: bool = True
    # Keep submaps.zarr after merge as a resume/re-opt checkpoint; False deletes it.
    keep_submaps: bool = True
```

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_wrapper.py::test_loopclosure_config_streaming_defaults -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_wrapper.py
git commit -m "feat(geometry): LoopClosureConfig.streaming + keep_submaps"
```

### Task D2: `_preprocess_window(idxs)` — per-window forward from FrameStore

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py` (LoopClosure holds the FrameStore; add window preprocess)
- Test: `tests/geometry/loop_closure/test_wrapper.py`

- [ ] **Step 1: Write the failing test — window preprocess loads only the window**

```python
def test_preprocess_window_loads_only_window(monkeypatch, tmp_path):
    """_preprocess_window reads exactly the requested rows from the FrameStore."""
    # Build a tiny FrameStore of 10 frames; assert reading a 3-window touches 3 rows.
    # (Use a fake creator whose _preprocess records how many frames it received.)
    ...
```

**Note to implementer:** construct a minimal `FrameStore` via `FrameStore.create` with 10 synthetic frames; wrap a stub creator whose `_preprocess(frames, frame_idxs)` asserts `len(frames) == len(idxs)`. Assert the window read returns 3 views for `idxs=[0,1,2]`.

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_wrapper.py::test_preprocess_window_loads_only_window -q`
Expected: FAIL — no `_preprocess_window`.

- [ ] **Step 3: Implement `_preprocess_window`**

Add to `LoopClosure` (the wrapper holds `self._frame_store` set in `setup_inference`):

```python
    def _preprocess_window(self, idxs: list[int]):
        """Decode + preprocess exactly the window's frames from the FrameStore."""
        frames = self._frame_store.images(idxs)          # (k, H, W, 3) — only these rows
        frame_idxs = [int(self._frame_store.frame_indices()[i]) for i in idxs]
        return self.base._preprocess(frames, frame_idxs)  # (views, labels, coords)
```

**Note:** the wrapper must capture the `FrameStore` at `setup_inference` time. If `setup_inference` currently exports+forgets, add `self._frame_store = store` there.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_wrapper.py::test_preprocess_window_loads_only_window -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_wrapper.py
git commit -m "feat(geometry): _preprocess_window — decode only the window from FrameStore"
```

### Task D3: Streaming loop body — spill + free + persistent graph (no loops yet)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py:369-437` (`_run_lc_loop`)
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Write the failing test — free-after-spill drops refs**

```python
def test_streaming_frees_window_after_spill(tmp_path, monkeypatch):
    """After spilling submap i, its frames/world_points are not retained in RAM."""
    # Run the streaming loop on a stubbed creator over ~4 windows; assert the
    # in-RAM submap objects retained carry retrieval_vectors + overlap points only
    # (frames is None / world_points trimmed to overlap), and SubmapStore has all 4.
    ...
```

**Note:** stub `self.base._forward` to return deterministic tiny outputs; assert `store.present_ids()` grows per window and that retained RAM submaps have `frames is None`.

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py::test_streaming_frees_window_after_spill -q`
Expected: FAIL.

- [ ] **Step 3: Rewrite `_run_lc_loop` streaming path**

Restructure into: create `SubmapStore` → per window: `_preprocess_window` → forward → build `Submap` → `store.spill(submap)` → `pg.add_submap` + `pg.optimize()` → retain a **lightweight** submap (descriptors + overlap `world_points[:O]`/`[-O:]` + metadata; `frames=None`, drop full points) → free the window. Keep `find_loop_closures` detection using retained descriptors (loop-edge *application* deferred to D8). Gate on `cfg.streaming`; when `False`, keep the existing batch body (factor the shared per-window forward into `run_predictions`, already present).

```python
    def _run_lc_loop(self, **kwargs):
        cfg = self.config
        if not cfg.streaming:
            return self._run_lc_loop_batch(**kwargs)   # existing body, renamed
        # ... streaming body: SubmapStore.create → window loop (preprocess/forward/
        #     spill/free/add_submap/optimize/detect) → retain lightweight submaps +
        #     stashed lc_submaps → mark_complete → merge-from-disk (Task D5) ...
```

**Note:** move the current `_run_lc_loop` body into `_run_lc_loop_batch` verbatim first (one commit), then add the streaming branch — keeps the batch path (and its parity tests) untouched.

- [ ] **Step 4: Run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "feat(geometry): streaming _run_lc_loop — spill/free/persistent-graph (no loop edges yet)"
```

### Task D4: Reload-on-verify — loop candidate verification from disk

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py`
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Write the failing test**

Assert that when a loop candidate is detected, the streaming path calls `store.reload(detected_id)` to fetch the detected submap's frames/points for `_verify_loop_candidate` + anchor scale, rather than reading a retained full submap. Stub `_verify_loop_candidate` to record the submap it received.

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -k reload -q`
Expected: FAIL.

- [ ] **Step 3: Implement reload-on-verify**

In the streaming detection path, when `find_loop_closures` returns a candidate against submap `d_id`, `reload` that submap from the store to obtain the detected frame + world_points for `_verify_loop_candidate` and `_lc_anchor_scale`. Keep the verified `lc_submap` in a small RAM list (`lc_submaps`), exactly as the batch path accumulates it.

- [ ] **Step 4: Run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "feat(geometry): reload-on-verify loop candidates from submaps.zarr"
```

### Task D5: Merge-from-disk + final export + keep_submaps cleanup

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py`
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_streaming_merge_equals_batch(tmp_path):
    """Full streaming run == batch run (same poses + merged outputs) on a short seq."""
    ...  # run both paths on the same stubbed creator; compare raw_outputs arrays

def test_keep_submaps_false_deletes_store(tmp_path):
    """keep_submaps=False removes submaps.zarr after the final merge."""
    ...  # streaming run with keep_submaps=False → assert not (out/'submaps.zarr').exists()
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -k "merge_equals_batch or keep_submaps" -q`
Expected: FAIL.

- [ ] **Step 3: Implement the tail of the streaming loop**

After the window loop: `store.mark_complete()` → final `pg.optimize()` → `corrected = pg.extract_extrinsics(N)` → `self.base.raw_outputs = merge_submap_outputs_from_store(store, store.present_ids(), corrected)` → if `not cfg.keep_submaps: store.delete()`. The `submaps.zarr` path is `Path(output_dir)/"submaps.zarr"` (sibling of `feedforward.zarr`).

- [ ] **Step 4: Run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "feat(geometry): merge-from-disk final export + keep_submaps cleanup"
```

### Task D6: Resume — skip already-spilled windows

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py`
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resume_skips_spilled_windows(tmp_path):
    """A run interrupted after k windows, restarted, skips those k + matches full run."""
    # 1. Run streaming but interrupt after 2 windows (raise inside the loop on window 2).
    # 2. Re-run: assert windows 0,1 are NOT re-forwarded (reloaded from store) and the
    #    final poses equal an uninterrupted full run.
    ...
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py::test_resume_skips_spilled_windows -q`
Expected: FAIL.

- [ ] **Step 3: Implement resume**

At streaming start: if `submaps.zarr` exists and not complete, `SubmapStore.open` it; `already = store.present_ids()`. For each window, if its submap_id ∈ `already`, `reload` it (rebuild the persistent graph via `add_submap` + `optimize` from the stored submap) instead of re-forwarding. Rebuild retained descriptors from the reloaded submaps. Continue forwarding from the first missing window.

**Note:** the graph must be rebuilt deterministically from stored submaps so resume ≡ uninterrupted. Reload in submap_id order and `add_submap`/`optimize` each, matching the original cadence.

- [ ] **Step 4: Run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "feat(geometry): streaming resume — skip already-spilled windows"
```

### Task D7: Windowed-vs-batch pose parity test (the correctness gate)

**Files:**
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Write the parity test on a short real-ish sequence**

```python
def test_windowed_equals_batch_poses(tmp_path):
    """Streaming poses == batch poses within tolerance on a short multi-submap seq."""
    # Same stubbed deterministic creator, ≥3 submaps, ≥1 loop. Run streaming and
    # batch; assert np.allclose(streaming_extrinsics, batch_extrinsics, atol=1e-5).
    ...
```

- [ ] **Step 2: Run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py::test_windowed_equals_batch_poses -q`
Expected: PASS — with `loop_edge_timing=deferred` (default until Phase G). This is the correctness gate.

- [ ] **Step 3: Commit**

```bash
git add tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "test(geometry): windowed-vs-batch pose parity gate"
```

### Task D8: Loop-edge insertion cadence (SYNC POINT — needs Phase G decision)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py` (+ `graph.py` if `live`)

- [ ] **Step 1: Read the Phase G A/B result**

Confirm the deferred-vs-live decision from Task G3. If Phase G says **deferred**, the current streaming path (loop edges applied at final solve via the shim ordering) is already correct — mark this task done, no code change. If **live**, proceed.

- [ ] **Step 2 (only if live): Insert loop edges during the window loop**

Change the streaming path so an accepted loop calls `pg.add_loop_edge(lc, retained_submaps)` immediately (before the next window's `optimize`), matching VGGT-SLAM `solver.py:294-295`. Update `_run_lc_loop_batch` identically so batch stays the parity reference. Re-run `test_windowed_equals_batch_poses` — it must still pass (both paths now live).

- [ ] **Step 3: Run the full LC suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/
git commit -m "feat(geometry): finalize loop-edge cadence per A/B decision (<deferred|live>)"
```

---

## Phase E: Viewer wiring + keep-alive

**Goal:** Instantiate `Viewer` when `viz.enabled`; push submaps/frusta after append; loop line on accept; corrected re-upload; keep-alive. All guarded `if self.viz is not None:`.

### Task E1: `Viewer.serve_forever` keep-alive

**Files:**
- Modify: `collab_splats/viewer.py`
- Test: `tests/test_viewer.py` (Create or extend)

- [ ] **Step 1: Write the failing test**

```python
def test_viewer_serve_forever_blocks_until_stop(monkeypatch):
    """serve_forever returns promptly when a stop event is pre-set (no real sleep loop)."""
    from collab_splats.viewer import Viewer
    v = Viewer.__new__(Viewer)          # avoid binding a real port
    import threading
    v._stop = threading.Event()
    v._stop.set()
    v.serve_forever(poll=0.01)          # should return immediately
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py::test_viewer_serve_forever_blocks_until_stop -q`
Expected: FAIL — no `serve_forever`.

- [ ] **Step 3: Implement**

```python
    def serve_forever(self, poll: float = 0.5) -> None:
        """Block so the viser server thread survives pipeline end (Ctrl-C / _stop to exit)."""
        if not hasattr(self, "_stop"):
            self._stop = threading.Event()
        try:
            while not self._stop.is_set():
                self._stop.wait(poll)
        except KeyboardInterrupt:
            pass
```

Add `import threading` at the top if absent.

- [ ] **Step 4: Run + commit**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_viewer.py -q`
Expected: PASS.

```bash
black . && isort .
git add collab_splats/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): serve_forever keep-alive"
```

### Task E2: Wire the Viewer into the streaming loop (guarded hooks)

**Files:**
- Modify: `collab_splats/geometry/loop_closure/wrapper.py`
- Test: `tests/geometry/loop_closure/test_streaming_parity.py`

- [ ] **Step 1: Write the failing test — viewer hooks fire, stubbed**

```python
def test_streaming_viewer_hooks_called(tmp_path):
    """With a stub Viewer, streaming calls add_points+add_frustum per submap and
    add_lines on an accepted loop; parity unaffected when viz is None."""
    # Inject a recording stub as self.viz; run a short streaming pass with ≥1 loop.
    # Assert add_points called n_submaps times, add_lines ≥1.
    ...
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py::test_streaming_viewer_hooks_called -q`
Expected: FAIL.

- [ ] **Step 3: Add guarded hook sites**

Three `if self.viz is not None:` blocks (try/except per scene-viewer spec):
1. **After submap append + optimize** — `pts, cols = subsample_points(world_points, colors, conf, max_points, conf_percentile)`; `self.viz.add_points(f"submap_{sid}", pts, cols)`; loop frusta via `self.viz.add_frustum(f"submap_{sid}/cams/frame_{i}", pose, K)`.
2. **On accepted loop** — `self.viz.add_lines(f"loop_{q}_{d}", segments)` between the two camera centers; full-scene re-upload of corrected extrinsics.
3. **After final PGO** — re-upload all submaps' corrected poses (for deferred; for live also per-loop, per Task D8).

`self.viz` is set by the driver (Task F2); default `None` → all hooks skip → parity tests unaffected.

- [ ] **Step 4: Run**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_streaming_parity.py -q`
Expected: PASS (hooks fire with stub; parity tests with `viz=None` still green).

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add collab_splats/geometry/loop_closure/wrapper.py tests/geometry/loop_closure/test_streaming_parity.py
git commit -m "feat(viewer): guarded streaming hooks — submap push, loop line, corrected re-upload"
```

---

## Phase F: Config + example runner

### Task F1: Add the 4 base.yaml keys + strict-access plumbing

**Files:**
- Modify: `configs/base.yaml`
- Modify: `collab_splats/wrapper/reconstructor.py` (map config → `LoopClosureConfig` / viewer)
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

```python
def test_streaming_and_viz_config_plumbed():
    """base.yaml streaming/keep_submaps/viz keys reach the LC config + viewer flag."""
    # Load base.yaml via Reconstructor; assert loop_closure.streaming True,
    # keep_submaps True, viz.enabled False, viz.port 8080 are read (strict access).
    ...
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -k "streaming_and_viz" -q`
Expected: FAIL.

- [ ] **Step 3: Add the keys + plumbing**

In `configs/base.yaml`, under `pointcloud`:

```yaml
  loop_closure:
    streaming: true          # spill+merge-from-disk; false = batch path
    keep_submaps: true       # keep submaps.zarr for resume/re-opt; false deletes after merge
  viz:
    enabled: false           # instantiate the viser Viewer
    port: 8080               # viser port
```

**Note:** `pointcloud.loop_closure` is currently a bool (`loop_closure: false`). Reconcile: keep the bool as the on/off switch and read `streaming`/`keep_submaps` from the `LoopClosureConfig` merge path the Reconstructor already uses (`submap_size` etc. flow through there). If the YAML shape forces a choice, put `streaming`/`keep_submaps` alongside the existing LC fields the Reconstructor maps, NOT under the bool. Match the existing merge in `reconstructor.py` (strict access — no `.get` defaults; the `test_guard` enforces this).

- [ ] **Step 4: Run the reconstructor suite (incl. the guard)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -q`
Expected: PASS (new test + `test_guard` still green — no Python-side defaults).

- [ ] **Step 5: Commit**

```bash
black . && isort .
git add configs/base.yaml collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(config): base.yaml streaming/keep_submaps/viz keys + strict plumbing"
```

### Task F2: `docs/examples/run_scenes.py` — long-scene runner with `--keep-viewer`

**Files:**
- Create: `docs/examples/run_scenes.py`

- [ ] **Step 1: Write the runner (no test — it's an example driver; smoke-covered in Phase H)**

A CLI that: loads a streaming config, samples keyframes to `frames.zarr` (Spec 1 `FrameStore` / existing preproc), instantiates `Viewer(port)` when `viz.enabled`, sets `loop_closure.viz = viewer`, runs the streaming reconstruction, and — if `--keep-viewer` — calls `viewer.serve_forever()` at the end. All imports at top (project style). Inline block comments per logical section. One-line module docstring.

```python
"""Example: windowed streaming reconstruction on a long (>1k-frame) video.

Usage:
    python docs/examples/run_scenes.py --input scene.mp4 --output out/ [--keep-viewer]
"""
# ... argparse (input, output, --config, --keep-viewer) → build Reconstructor from
#     configs/base.yaml (streaming defaults) → run → optional viewer.serve_forever() ...
```

- [ ] **Step 2: Byte-compile check**

Run: `/opt/venv/reconstruction/bin/python -m py_compile docs/examples/run_scenes.py`
Expected: no output (compiles).

- [ ] **Step 3: Commit**

```bash
black . && isort .
git add docs/examples/run_scenes.py
git commit -m "docs(examples): run_scenes.py streaming runner with --keep-viewer"
```

---

## Phase G: A/B gate — loop-edge timing (Track 1, DISPOSABLE worktree)

**Goal:** Decide deferred-vs-live loop-edge timing from ATE, not guesswork. Runs in its OWN worktree; only the decision (a sentence) feeds Task D8. **No code from this phase is merged.**

### Task G1: Create the disposable A/B worktree

- [ ] **Step 1: Branch off the SAME base as Track 2 (pre-streaming is fine — A/B tests the batch shim)**

```bash
cd /workspace/collab-splats
git worktree add .worktrees/ab-timing -b throwaway/ab-loop-timing
cd .worktrees/ab-timing && ln -s /workspace/collab-splats/third_party third_party
```

### Task G2: Add `loop_edge_timing` param to `run_pose_graph_optimization`

**Files (in the A/B worktree only):**
- Modify: `collab_splats/geometry/loop_closure/graph.py`

- [ ] **Step 1: Add `loop_edge_timing: Literal["deferred","live"] = "deferred"`**

`deferred` = current shim order (all loop edges after the submap loop). `live` = interleave: inside the submap loop, after adding submap *s* and optimizing, apply any loop edges whose query resolves to *s* (mirrors `solver.py:294-295`). Reuse `add_submap`/`add_loop_edge`/`optimize` — no new math.

- [ ] **Step 2: Sanity — both timings run, deferred matches HEAD**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_pgo_parity.py -q`
Expected: PASS with `deferred` (default). `live` is exercised by the eval, not unit parity.

### Task G3: Run the A/B eval + record the decision

**Files:** none merged — output is a decision note.

- [ ] **Step 1: Run the eval matrix in tmux**

On chess d5 + TUM fr3 × 4 backbones (the scenes from `1b7bc84`), run each backbone twice (`loop_edge_timing=deferred` vs `live`) via the existing eval driver (`evals/scripts/eval.py`), collecting ATE + Δ-vs-SLAM. Heavy → tmux, per repo memory guidance.

```bash
# in tmux, from the ab-timing worktree
/opt/venv/reconstruction/bin/python evals/scripts/eval.py --help   # confirm flags
# run deferred and live variants; capture ATE per scene/backbone
```

- [ ] **Step 2: Apply the decision rule + record it**

Decision rule (spec §A/B gate): **live ≤ deferred ATE and ≈ SLAM → adopt live**; else **keep deferred**. Write the ATE table + the one-line verdict into `docs/superpowers/decisions/` (create `NNN-loop-edge-timing.md`, sequential number) **in the Track 2 worktree** so it's committed (the A/B worktree itself is thrown away).

- [ ] **Step 3: Tear down the A/B worktree**

```bash
cd /workspace/collab-splats
git worktree remove .worktrees/ab-timing --force
git branch -D throwaway/ab-loop-timing
```

**→ Feed the verdict into Task D8.**

---

## Phase H: Verification

### Task H1: Full suite green + smoke

**Files:** none (verification)

- [ ] **Step 1: Full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: PASS (excluding the 2 pre-existing `run_pipeline` failures + known xfails in `docs/known-test-failures.md` — confirm no NEW failures).

- [ ] **Step 2: Short-video end-to-end smoke (viewer stubbed)**

Run `docs/examples/run_scenes.py` on a short clip with `viz.enabled=false`, small `max_frames`. Confirm: `frames.zarr` decoded once, per-submap groups appear in `submaps.zarr`, final `feedforward.zarr` + COLMAP written, `submaps.zarr` retained (`keep_submaps=true`).

- [ ] **Step 3: Memory bound check (tmux)**

On a >1k-frame video in tmux: sample RSS per window (e.g. log `psutil.Process().memory_info().rss` each window). Assert peak RSS stays roughly flat across windows (bounded by window), not linear — contrast a batch run's growth. Record numbers in the decision doc.

- [ ] **Step 4: Update CLAUDE.md In-Flight + memory**

Add the streaming feature to `collab_splats/CLAUDE.md` "Recently completed" with spec/plan links; write/refresh a memory file (`project_windowed_streaming.md`) capturing: streaming default on, `submaps.zarr` checkpoint semantics, the A/B verdict, parity gate location.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "docs: record windowed-streaming completion + A/B verdict + memory"
```

---

## Self-review notes (author)

- **Spec coverage:** window=submap (D2–D3), spill/free (D3), retained state descriptors+overlap (D3), reload-on-verify (D4), per-submap PGO cadence (A2/A4/D3), merge-from-disk (B4/D5), keep_submaps (D5), resume (D6), parity gate (D7/A1), loop-edge A/B (G, D8), viewer 3 hooks + keep-alive (E1/E2), config 4 keys (F1), example runner (F2), preprocess-takes-frames + hack deletion (C1), online-seam (design-only, no task — correct, it's a non-goal). All spec sections map to a task.
- **Parity discipline:** Phase A is a pure refactor locked by A1's characterization test; batch body preserved as `_run_lc_loop_batch` (D3) so existing parity tests never move until D8 changes both paths together.
- **Foreign-change guard:** merge.py (B4) has an explicit stop-check.
