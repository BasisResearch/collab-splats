# Loop Closure Evaluation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eval module + tests + narrative notebook that demonstrate and measure loop closure's effect on feedforward camera poses (pose-graph residual reduction with per-edge breakdown), built so future GT-pose metrics drop in cleanly.

**Architecture:** Split `closure.run_pose_graph_optimization` into `build_pose_graph` + `optimize_pose_graph` so callers can inspect the pose graph pre-optimization. Save submaps + lc_submaps + overlap as instance attrs on `BaseFeedforwardCreator` after `run_inference()` so the notebook can rebuild a fresh graph for instrumentation. New `loop_closure/eval.py` module exposes `capture_pose_graph_loss(pg) -> dict` (re-runs LM with per-iter cost capture) plus future-GT stubs that raise `NotImplementedError`. Notebook in `docs/pointcloud/loop_closure_eval.ipynb` walks readers through the pipeline using bicycle scene + MapAnything backend; viz reuses `visualize_splat()` from `utils/visualization.py`.

**Tech Stack:** Python 3.10 (nerfstudio conda env), GTSAM 4.x, NumPy, PyTorch, MapAnything, PyVista, matplotlib, pytest.

**Design spec:** `/root/.claude/plans/i-would-like-to-luminous-raccoon.md`

---

## File Structure

**New files:**
- `collab_splats/pointcloud/loop_closure/eval.py` — eval helpers (~150 lines)
- `tests/pointcloud/test_loop_closure_eval.py` — flat pytest functions
- `docs/pointcloud/loop_closure_eval.ipynb` — narrative notebook

**Modified files:**
- `collab_splats/pointcloud/loop_closure/closure.py` — split optimization into build + optimize; existing `run_pose_graph_optimization` keeps signature, calls new helpers internally
- `collab_splats/pointcloud/loop_closure/__init__.py` — export `capture_pose_graph_loss`
- `collab_splats/pointcloud/feedforward.py` — save `_lc_submaps`, `_lc_loop_submaps`, `_lc_overlap_frames` as instance attrs in `_run_loop_closure_inference`

---

## Task 1: Split closure.py into build_pose_graph + optimize

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`
- Test: `tests/pointcloud/test_loop_closure.py` (extend) or new `tests/pointcloud/test_closure_split.py`

Goal: extract pose-graph construction so the eval helper can rebuild the same graph for instrumentation without mutating production state.

- [ ] **Step 1: Write the failing test** at `tests/pointcloud/test_closure_split.py`

```python
import numpy as np
import torch
from pathlib import Path

from collab_splats.pointcloud.loop_closure import PoseGraph, Submap
from collab_splats.pointcloud.loop_closure.closure import (
    build_pose_graph,
    run_pose_graph_optimization,
)


def _identity_submap(submap_id, k=3):
    np.random.seed(0)
    poses = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    poses[:, :3, 3] = np.random.randn(k, 3).astype(np.float32) * 0.01
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 64, 64),
        poses=poses,
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 128),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def test_build_pose_graph_returns_pose_graph_with_factors():
    submaps = [_identity_submap(i, k=3) for i in range(2)]
    pg = build_pose_graph(submaps, lc_submaps=[], overlap_frames=1)
    assert isinstance(pg, PoseGraph)
    # 2 submaps × 2 intra edges + 1 inter edge + 1 prior = 6 factors
    assert pg._graph.size() == 6


def test_build_pose_graph_no_loop_edges_when_lc_submaps_empty():
    submaps = [_identity_submap(i, k=3) for i in range(2)]
    lc = Submap(
        submap_id=99,
        frames=torch.zeros(2, 3, 64, 64),
        poses=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(2, 128),
        image_paths=[Path("s0_f0.jpg"), Path("s1_f0.jpg")],
        is_lc_submap=True,
    )
    pg_no_lc = build_pose_graph(submaps, lc_submaps=[], overlap_frames=1)
    pg_with_lc = build_pose_graph(submaps, lc_submaps=[lc], overlap_frames=1)
    assert pg_with_lc._graph.size() > pg_no_lc._graph.size()


def test_run_pose_graph_optimization_unchanged_signature():
    """Backward compat: existing callers still work."""
    submaps = [_identity_submap(i, k=3) for i in range(2)]
    corrected = run_pose_graph_optimization(
        submaps=submaps,
        lc_submaps=[],
        total_frames=5,  # 2*3 - 1 overlap
        overlap_frames=1,
    )
    assert corrected.shape == (5, 4, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_closure_split.py -v
```

Expected: ImportError on `build_pose_graph` (not yet defined).

- [ ] **Step 3: Refactor `closure.py`** — replace `run_pose_graph_optimization` body

Read current contents at `collab_splats/pointcloud/loop_closure/closure.py`. Replace the function with:

```python
def build_pose_graph(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    overlap_frames: int,
) -> "PoseGraph":
    """Build SE(3) pose graph from submaps + verified loop closures.

    Does not optimize. Returned PoseGraph can be inspected, optimized, or
    passed to eval helpers (e.g. capture_pose_graph_loss).
    """
    from collab_splats.pointcloud.loop_closure import PoseGraph
    pg = PoseGraph()
    pg.add_submaps(submaps, overlap_frames=overlap_frames)
    if lc_submaps:
        pg.add_loop_edges(lc_submaps)
    return pg


def run_pose_graph_optimization(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    total_frames: int,
    overlap_frames: int,
) -> np.ndarray:
    """Build + optimize SE(3) pose graph; return (total_frames, 4, 4) corrected extrinsics."""
    from collab_splats.pointcloud.loop_closure.alignment import dedup_overlap
    pg = build_pose_graph(submaps, lc_submaps, overlap_frames)
    corrected = pg.optimize()
    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected,
        total_frames=total_frames,
    )
```

- [ ] **Step 4: Run new tests + existing closure-related tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_closure_split.py \
  tests/pointcloud/test_loop_closure.py \
  tests/pointcloud/test_loop_closure_integration.py \
  tests/pointcloud/test_pose_graph.py \
  -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/loop_closure/closure.py \
  tests/pointcloud/test_closure_split.py
git commit -m "refactor(lc): split run_pose_graph_optimization into build_pose_graph + optimize

Exposes PoseGraph construction so eval helpers can inspect pre-optimization state.
Existing run_pose_graph_optimization signature unchanged."
```

---

## Task 2: Expose LC intermediate state on `BaseFeedforwardCreator`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py:184` (`_run_loop_closure_inference`)
- Test: `tests/pointcloud/test_feedforward_lc_state.py` (new)

Goal: after `creator.run_inference()` with LC enabled, expose `_lc_submaps`, `_lc_loop_submaps`, `_lc_overlap_frames` as instance attrs so notebook can rebuild PoseGraph for instrumentation.

- [ ] **Step 1: Write the failing test** at `tests/pointcloud/test_feedforward_lc_state.py`

```python
"""Verify LC intermediate state is exposed on creator after run_inference()."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
from collab_splats.pointcloud.loop_closure import LoopClosureConfig, Submap


class _StubCreator(BaseFeedforwardCreator):
    """Minimal creator for unit-testing _run_loop_closure_inference state exposure."""

    def _load_model(self, device):
        m = MagicMock()
        m.parameters.return_value = iter([torch.zeros(1)])
        return m

    def _preprocess(self, image_dir):
        return torch.zeros(40, 3, 224, 224), [None] * 40, np.zeros((40, 6))

    def _forward(self, model, views, **kwargs):
        k = views.shape[0]
        return {
            "extrinsic": np.tile(np.eye(4)[:3], (k, 1, 1)).astype(np.float32),
            "intrinsic": np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        }

    def _verify_loop_candidate(self, frame1, frame2):
        return False  # no loops; tests only need submap exposure

    def _postprocess(self):
        pass

    def build_colmap(self, output_dir):
        pass


def test_lc_state_attrs_set_after_run_inference():
    creator = _StubCreator(
        camera_model="PINHOLE",
        enable_loop_closure=True,
        loop_closure_config=LoopClosureConfig(submap_size=20, submap_overlap=4),
    )
    creator.load_model()
    creator.views = torch.zeros(40, 3, 224, 224)
    creator.image_paths = [None] * 40

    with patch("collab_splats.pointcloud.loop_closure.ImageRetrieval") as MockRetrieval:
        MockRetrieval.return_value.embed_frames.return_value = torch.zeros(20, 128)
        MockRetrieval.return_value.find_loop_closures.return_value = []
        creator.run_inference()

    assert hasattr(creator, "_lc_submaps")
    assert hasattr(creator, "_lc_loop_submaps")
    assert hasattr(creator, "_lc_overlap_frames")
    assert isinstance(creator._lc_submaps, list)
    assert all(isinstance(s, Submap) for s in creator._lc_submaps)
    assert creator._lc_overlap_frames == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_feedforward_lc_state.py -v
```

Expected: FAIL on `hasattr(creator, "_lc_submaps")` — attr not set.

- [ ] **Step 3: Modify `feedforward.py:_run_loop_closure_inference`**

Read the current `_run_loop_closure_inference` body (lines ~184–330). At the end of the method, *before* the call to `closure.run_pose_graph_optimization` and *before* `merge_submap_outputs`, add:

```python
# Expose intermediate state for downstream instrumentation (eval notebooks).
self._lc_submaps = submaps
self._lc_loop_submaps = lc_submaps
self._lc_overlap_frames = O
```

Place these three lines after the submap-construction loop completes and `lc_submaps` is finalized, just before the call to `run_pose_graph_optimization`.

- [ ] **Step 4: Run new test + existing feedforward tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_feedforward_lc_state.py \
  tests/pointcloud/test_feedforward_shared.py \
  tests/pointcloud/test_mapanything_creator.py \
  tests/pointcloud/test_loop_closure_integration.py \
  -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/feedforward.py \
  tests/pointcloud/test_feedforward_lc_state.py
git commit -m "feat(lc): expose submaps + overlap as instance attrs after LC inference

Allows downstream instrumentation (e.g. eval notebook) to rebuild
PoseGraph for analysis without re-running expensive feedforward."
```

---

## Task 3: Implement `_classify_edges` helper

**Files:**
- Create: `collab_splats/pointcloud/loop_closure/eval.py`
- Test: `tests/pointcloud/test_loop_closure_eval.py`

Goal: classify pose-graph factors into `intra` / `inter` / `loop` buckets via noise-model inspection.

- [ ] **Step 1: Create `eval.py` skeleton + write failing test**

Create `collab_splats/pointcloud/loop_closure/eval.py`:

```python
"""Loop-closure pose-graph evaluation helpers.

Provides instrumentation for the LC pose graph (per-iteration loss capture,
per-edge-type residual breakdown) and stubs for future ground-truth-based
metrics (ATE, RPE) that will be implemented when a GT pose dataset is sourced.
"""
from __future__ import annotations

import gtsam
import numpy as np

from .pose_graph import PoseGraph


def _classify_edges(graph: gtsam.NonlinearFactorGraph) -> dict[str, list[int]]:
    """Partition factor indices by edge type.

    Loop edges use Robust(Huber) wrapping. Non-loop edges use Diagonal noise
    with sigma signatures from pose_graph.py:
      intra: sigma_r=0.02 (and sigma_t=0.05)
      inter: sigma_r=0.05 (and sigma_t=0.20)

    Prior factors (anchor) are skipped.
    """
    groups: dict[str, list[int]] = {"intra": [], "inter": [], "loop": []}
    for i in range(graph.size()):
        factor = graph.at(i)
        if not isinstance(factor, gtsam.BetweenFactorPose3):
            continue  # skip PriorFactor
        nm = factor.noiseModel()
        if isinstance(nm, gtsam.noiseModel.Robust):
            groups["loop"].append(i)
            continue
        # Non-robust: classify by rotation sigma
        sigmas = nm.sigmas() if hasattr(nm, "sigmas") else None
        if sigmas is None or len(sigmas) < 1:
            continue
        sigma_r = float(sigmas[0])
        if abs(sigma_r - 0.02) < 1e-6:
            groups["intra"].append(i)
        elif abs(sigma_r - 0.05) < 1e-6:
            groups["inter"].append(i)
    return groups
```

Create `tests/pointcloud/test_loop_closure_eval.py`:

```python
import numpy as np
import pytest
import torch
from pathlib import Path

from collab_splats.pointcloud.loop_closure import PoseGraph, Submap
from collab_splats.pointcloud.loop_closure.eval import _classify_edges


def _identity_submap(submap_id, k=3):
    poses = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    poses[:, :3, 3] = np.random.randn(k, 3).astype(np.float32) * 0.01
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 64, 64),
        poses=poses,
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 128),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def _build_pg_with_loop():
    pg = PoseGraph()
    pg.add_submaps([_identity_submap(i, k=3) for i in range(2)], overlap_frames=1)
    lc = Submap(
        submap_id=2,
        frames=torch.zeros(2, 3, 64, 64),
        poses=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(2, 128),
        image_paths=[Path("s0_f0.jpg"), Path("s1_f0.jpg")],
        is_lc_submap=True,
    )
    pg.add_loop_edges([lc])
    return pg


def test_classify_edges_returns_three_keys():
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    assert set(groups) == {"intra", "inter", "loop"}


def test_classify_edges_intra_count_matches_expected():
    """2 submaps × 2 intra edges = 4 intra."""
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    assert len(groups["intra"]) == 4


def test_classify_edges_inter_count_matches_expected():
    """2 submaps → 1 inter edge."""
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    assert len(groups["inter"]) == 1


def test_classify_edges_loop_count_matches_expected():
    """1 LC submap → 1 loop edge."""
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    assert len(groups["loop"]) == 1


def test_classify_edges_skips_prior_factor():
    """The PriorFactor on frame 0 should not appear in any group."""
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    total_classified = sum(len(v) for v in groups.values())
    # 4 intra + 1 inter + 1 loop = 6; total factors = 7 (incl. prior)
    assert total_classified == pg._graph.size() - 1
```

- [ ] **Step 2: Run tests to verify they fail then pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: PASS (helper already implemented in Step 1).

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/loop_closure/eval.py \
  tests/pointcloud/test_loop_closure_eval.py
git commit -m "feat(lc): add eval._classify_edges for per-edge-type residual breakdown

Classifies BetweenFactorPose3 instances into intra/inter/loop using
noise-model signatures (Robust wrap = loop; sigma_r=0.02 = intra; 0.05 = inter)."
```

---

## Task 4: Implement `_per_edge_error` helper

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/eval.py`
- Test: `tests/pointcloud/test_loop_closure_eval.py` (extend)

- [ ] **Step 1: Add failing test**

Append to `tests/pointcloud/test_loop_closure_eval.py`:

```python
from collab_splats.pointcloud.loop_closure.eval import _per_edge_error


def test_per_edge_error_keys_match_classifier():
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    errors = _per_edge_error(pg._graph, pg._initial, groups)
    assert set(errors) == {"intra", "inter", "loop"}


def test_per_edge_error_returns_floats():
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    errors = _per_edge_error(pg._graph, pg._initial, groups)
    for v in errors.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_per_edge_error_sum_matches_classified_total():
    """Sum of per-edge errors == sum of individual classified factor errors."""
    pg = _build_pg_with_loop()
    groups = _classify_edges(pg._graph)
    errors = _per_edge_error(pg._graph, pg._initial, groups)
    summed = sum(errors.values())
    expected = sum(
        pg._graph.at(i).error(pg._initial)
        for indices in groups.values()
        for i in indices
    )
    assert abs(summed - expected) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: ImportError on `_per_edge_error`.

- [ ] **Step 3: Add `_per_edge_error` to `eval.py`**

Append to `collab_splats/pointcloud/loop_closure/eval.py`:

```python
def _per_edge_error(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    edge_groups: dict[str, list[int]],
) -> dict[str, float]:
    """Sum graph.at(i).error(values) over each edge group."""
    return {
        group_name: float(sum(graph.at(i).error(values) for i in indices))
        for group_name, indices in edge_groups.items()
    }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/loop_closure/eval.py \
  tests/pointcloud/test_loop_closure_eval.py
git commit -m "feat(lc): add eval._per_edge_error for per-group residual sums"
```

---

## Task 5: Implement `capture_pose_graph_loss`

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/eval.py`
- Test: `tests/pointcloud/test_loop_closure_eval.py` (extend)

Goal: re-run LM optimization on a fresh optimizer (no mutation), capture per-iter `optimizer.error()`, return dict with iterations + per-edge initial/final breakdown + optimized values.

- [ ] **Step 1: Add failing test**

Append to `tests/pointcloud/test_loop_closure_eval.py`:

```python
from collab_splats.pointcloud.loop_closure.eval import capture_pose_graph_loss


def test_capture_loss_returns_required_keys():
    pg = _build_pg_with_loop()
    trace = capture_pose_graph_loss(pg)
    assert set(trace) >= {"iterations", "per_edge_initial", "per_edge_final", "optimized_values"}


def test_capture_loss_iterations_is_nonempty_list_of_floats():
    pg = _build_pg_with_loop()
    trace = capture_pose_graph_loss(pg)
    assert isinstance(trace["iterations"], list)
    assert len(trace["iterations"]) >= 1
    assert all(isinstance(x, float) for x in trace["iterations"])


def test_capture_loss_curve_decreases_or_holds():
    """LM should never increase total error step-to-step; final ≤ initial."""
    pg = _build_pg_with_loop()
    trace = capture_pose_graph_loss(pg)
    assert trace["iterations"][-1] <= trace["iterations"][0] + 1e-9


def test_capture_loss_does_not_mutate_input_graph():
    """Capture must not mutate pg._initial."""
    import gtsam
    pg = _build_pg_with_loop()
    initial_keys_before = set(pg._initial.keys())
    initial_pose_before = pg._initial.atPose3(list(initial_keys_before)[0]).matrix().copy()
    capture_pose_graph_loss(pg)
    initial_pose_after = pg._initial.atPose3(list(initial_keys_before)[0]).matrix()
    assert np.allclose(initial_pose_before, initial_pose_after)


def test_capture_loss_per_edge_keys_match_classifier():
    pg = _build_pg_with_loop()
    trace = capture_pose_graph_loss(pg)
    assert set(trace["per_edge_initial"]) == {"intra", "inter", "loop"}
    assert set(trace["per_edge_final"]) == {"intra", "inter", "loop"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: ImportError on `capture_pose_graph_loss`.

- [ ] **Step 3: Implement `capture_pose_graph_loss`**

Append to `collab_splats/pointcloud/loop_closure/eval.py`:

```python
def capture_pose_graph_loss(
    pose_graph: PoseGraph,
    max_iterations: int = 100,
    relative_error_tol: float = 1e-5,
) -> dict:
    """Re-run LM optimization with per-iteration cost capture.

    Does NOT mutate pose_graph. Builds a fresh LM optimizer from
    pose_graph._graph + pose_graph._initial, iterates manually via
    optimizer.iterate(), records optimizer.error() each step.

    Returns:
        iterations: list[float]            graph.error() per LM iter (incl. initial)
        per_edge_initial: dict[str, float] {'intra', 'inter', 'loop'} initial residuals
        per_edge_final: dict[str, float]   {'intra', 'inter', 'loop'} final residuals
        optimized_values: gtsam.Values     optimizer result
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(relative_error_tol)

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        pose_graph._graph, pose_graph._initial, params
    )

    iterations: list[float] = [float(optimizer.error())]
    prev_err = iterations[0]
    for _ in range(max_iterations):
        optimizer.iterate()
        err = float(optimizer.error())
        iterations.append(err)
        if prev_err > 0 and abs(prev_err - err) / max(prev_err, 1e-12) < relative_error_tol:
            break
        prev_err = err

    optimized = optimizer.values()
    groups = _classify_edges(pose_graph._graph)
    per_edge_initial = _per_edge_error(pose_graph._graph, pose_graph._initial, groups)
    per_edge_final = _per_edge_error(pose_graph._graph, optimized, groups)

    return {
        "iterations": iterations,
        "per_edge_initial": per_edge_initial,
        "per_edge_final": per_edge_final,
        "optimized_values": optimized,
    }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/loop_closure/eval.py \
  tests/pointcloud/test_loop_closure_eval.py
git commit -m "feat(lc): add capture_pose_graph_loss for instrumented LM optimization

Re-runs LM iter loop on fresh optimizer (no input mutation), records
per-iteration cost + per-edge-type residual breakdown."
```

---

## Task 6: Add GT-phase stubs (NotImplementedError)

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/eval.py`
- Test: `tests/pointcloud/test_loop_closure_eval.py` (extend)

Goal: future-shaped function signatures for ATE / RPE / Umeyama align that raise NotImplementedError. Implementation lands when a GT pose dataset is sourced.

- [ ] **Step 1: Add failing tests**

Append to `tests/pointcloud/test_loop_closure_eval.py`:

```python
from collab_splats.pointcloud.loop_closure.eval import (
    ate_translation,
    rpe,
    umeyama_align,
)


def test_umeyama_align_raises_not_implemented():
    pred = np.eye(4)[None].astype(np.float32)
    gt = np.eye(4)[None].astype(np.float32)
    with pytest.raises(NotImplementedError, match="GT pose dataset"):
        umeyama_align(pred, gt)


def test_ate_translation_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="GT pose dataset"):
        ate_translation(np.eye(4)[None], np.eye(4)[None])


def test_rpe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="GT pose dataset"):
        rpe(np.eye(4)[None], np.eye(4)[None])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: ImportError on `ate_translation`, `rpe`, `umeyama_align`.

- [ ] **Step 3: Add stubs**

Append to `collab_splats/pointcloud/loop_closure/eval.py`:

```python
# --- GT-phase stubs (implement when GT dataset arrives) ----------------------


def umeyama_align(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align predicted trajectory to ground truth via Umeyama SE(3).

    Wraps alignment.umeyama_se3 once implemented.

    Args:
        pred: (N, 4, 4) predicted world-to-cam poses
        gt:   (N, 4, 4) ground-truth world-to-cam poses

    Returns:
        (aligned_pred, T_align): Umeyama-aligned predicted poses and the SE(3)
        transform applied. Implement once GT pose dataset available.
    """
    raise NotImplementedError(
        "Implement once GT pose dataset available "
        "(reuse alignment.umeyama_se3 from loop_closure.alignment)"
    )


def ate_translation(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Absolute Trajectory Error on translation component.

    Args:
        pred: (N, 4, 4) predicted poses
        gt:   (N, 4, 4) ground-truth poses

    Returns:
        {'rmse': ..., 'mean': ..., 'median': ..., 'max': ...} per-frame
        translation distance after Umeyama alignment. Implement once GT
        pose dataset available.
    """
    raise NotImplementedError("Implement once GT pose dataset available")


def rpe(pred: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Relative Pose Error at frame stride delta.

    Args:
        pred: (N, 4, 4) predicted poses
        gt:   (N, 4, 4) ground-truth poses
        delta: frame stride between compared pose pairs

    Returns:
        {'trans_rmse': ..., 'rot_rmse_deg': ...} relative pose error
        statistics. Implement once GT pose dataset available.
    """
    raise NotImplementedError("Implement once GT pose dataset available")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/loop_closure/eval.py \
  tests/pointcloud/test_loop_closure_eval.py
git commit -m "feat(lc): add GT-phase stubs (umeyama_align, ate_translation, rpe)

Future-shaped signatures raise NotImplementedError. Implement once
KITTI/outdoor GT pose dataset is sourced."
```

---

## Task 7: Export `capture_pose_graph_loss` from `loop_closure/__init__.py`

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/__init__.py`
- Test: `tests/pointcloud/test_loop_closure_eval.py` (extend)

- [ ] **Step 1: Add failing test**

Append to `tests/pointcloud/test_loop_closure_eval.py`:

```python
def test_capture_pose_graph_loss_importable_from_package():
    from collab_splats.pointcloud.loop_closure import capture_pose_graph_loss as cpl
    assert callable(cpl)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py::test_capture_pose_graph_loss_importable_from_package -v
```

Expected: ImportError.

- [ ] **Step 3: Update `__init__.py`**

Read current contents at `collab_splats/pointcloud/loop_closure/__init__.py`. Add the eval import + extend `__all__`:

```python
from .submap import Submap
from .retrieval import ImageRetrieval, LoopClosureConfig, LoopMatch, LoopMatchQueue
from .pose_graph import PoseGraph
from .eval import capture_pose_graph_loss

__all__ = [
    "ImageRetrieval",
    "LoopClosureConfig",
    "LoopMatch",
    "LoopMatchQueue",
    "PoseGraph",
    "Submap",
    "capture_pose_graph_loss",
]
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add \
  collab_splats/pointcloud/loop_closure/__init__.py
git commit -m "feat(lc): export capture_pose_graph_loss from loop_closure package"
```

---

## Task 8: Build notebook scaffold + executable cells

**Files:**
- Create: `docs/pointcloud/loop_closure_eval.ipynb`

Goal: narrative notebook walking readers through LC pipeline assembly, capturing loss, plotting curve + per-edge breakdown + before/after pointcloud+frustum visualization. No automated test (repo convention for notebooks).

**Approach:** write notebook as JSON via Python script (for reproducibility) or directly with the Write tool. Since the file is small + structure is well-defined, write directly as JSON.

- [ ] **Step 1: Create notebook file**

Use the `Write` tool to create `docs/pointcloud/loop_closure_eval.ipynb` as a Jupyter notebook JSON with the following cells (order = §1 → §10 from the design):

**Cell 1 (markdown):**
````markdown
# Loop Closure Evaluation

Demonstrates and measures the effect of loop closure on feedforward camera
pose estimation. Uses the bicycle scene + MapAnything backend by default;
both are configurable.

**Metric (v1):** pose-graph residual reduction. Initial cost = chained-submap
state pre-optimization. Final cost = post-LM-optimization. Per-edge breakdown
(intra / inter / loop) shows where the optimizer makes the biggest difference.

**Future:** ground-truth pose comparison (ATE, RPE) via stubs in
`loop_closure/eval.py` once a GT dataset (KITTI or outdoor with known poses)
is sourced.
````

**Cell 2 (markdown):** `## §1 Setup`

**Cell 3 (code):**
```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collab_splats.wrapper import Splatter, SplatterConfig
from collab_splats.pointcloud.feedforward import MapAnythingCreator
from collab_splats.pointcloud.loop_closure import (
    LoopClosureConfig,
    PoseGraph,
    capture_pose_graph_loss,
)
from collab_splats.pointcloud.loop_closure.closure import build_pose_graph
from collab_splats.utils.visualization import visualize_splat
```

**Cell 4 (markdown):** `## §2 Config`

**Cell 5 (code):**
```python
SCENE_DIR = Path("/workspace/bicycle/images_4")
OUTPUT_DIR = Path("/workspace/bicycle/lc_eval")
BACKEND = "mapanything"  # alt: "vggtx"

splatter_config = SplatterConfig(
    file_path=str(SCENE_DIR),
    method="rade-gs",
    pointcloud_method="feedforward",
    output_path=str(OUTPUT_DIR),
    overwrite=False,
)
splatter = Splatter(splatter_config)
splatter.preprocess()

lc_config = LoopClosureConfig()
print(f"Submap size: {lc_config.submap_size}, overlap: {lc_config.submap_overlap}")
```

**Cell 6 (markdown):** `## §3 Run feedforward with loop closure enabled`

**Cell 7 (code):**
```python
creator = MapAnythingCreator(
    camera_model="PINHOLE",
    enable_loop_closure=True,
    loop_closure_config=lc_config,
)
creator.load_model()
creator.setup_inference(SCENE_DIR)
creator.run_inference()

submaps = creator._lc_submaps
lc_submaps = creator._lc_loop_submaps
overlap = creator._lc_overlap_frames
print(f"Built {len(submaps)} submaps, {len(lc_submaps)} verified loop closures, overlap={overlap}")
```

**Cell 8 (markdown):** `## §4 Build pose graph`

**Cell 9 (code):**
```python
pg = build_pose_graph(submaps, lc_submaps, overlap_frames=overlap)
print(f"Pose graph: {pg._graph.size()} factors, {pg._initial.size()} variables")
```

**Cell 10 (markdown):** `## §5 Capture loss trace`

**Cell 11 (code):**
```python
trace = capture_pose_graph_loss(pg)
print(f"Initial total error:   {trace['iterations'][0]:.4f}")
print(f"Final total error:     {trace['iterations'][-1]:.4f}")
print(f"Reduction:             {(trace['iterations'][0] - trace['iterations'][-1]) / trace['iterations'][0] * 100:.1f}%")
print(f"LM iterations:         {len(trace['iterations'])}")
print()
print("Per-edge initial:", trace["per_edge_initial"])
print("Per-edge final:  ", trace["per_edge_final"])
```

**Cell 12 (markdown):** `## §6 Loss curve`

**Cell 13 (code):**
```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(trace["iterations"], marker="o")
ax.set_xlabel("LM iteration")
ax.set_ylabel("graph.error() (log)")
ax.set_title("Pose-graph total cost during LM optimization")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Cell 14 (markdown):** `## §7 Per-edge residual breakdown`

**Cell 15 (code):**
```python
edge_types = ["intra", "inter", "loop"]
init_vals = [trace["per_edge_initial"][k] for k in edge_types]
final_vals = [trace["per_edge_final"][k] for k in edge_types]

x = np.arange(len(edge_types))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width / 2, init_vals, width, label="initial")
ax.bar(x + width / 2, final_vals, width, label="optimized")
ax.set_xticks(x)
ax.set_xticklabels(edge_types)
ax.set_ylabel("Sum of factor errors")
ax.set_title("Pose-graph residual by edge type")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
```

**Cell 16 (markdown):** `## §8 Pointcloud + camera frustums (before vs after)`

**Cell 17 (code):**
```python
import gtsam

# Initial chained poses (per-submap, world frame)
initial_chained = []
for s in submaps:
    for i in range(s.poses.shape[0]):
        key = gtsam.symbol("x", pg._frame_offset[s.submap_id] + i)
        initial_chained.append(pg._initial.atPose3(key).matrix().astype(np.float32))

# Optimized poses
optimized = []
for s in submaps:
    for i in range(s.poses.shape[0]):
        key = gtsam.symbol("x", pg._frame_offset[s.submap_id] + i)
        optimized.append(trace["optimized_values"].atPose3(key).matrix().astype(np.float32))

# Merged pointcloud across all submaps (downsample if too large)
pts_all = []
for s in submaps:
    if s.world_points is not None:
        pts_all.append(s.world_points.reshape(-1, 3))
pts_merged = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3))
print(f"Merged pointcloud: {pts_merged.shape[0]} points")
```

**Cell 18 (code):**
```python
import pyvista as pv

mesh = pv.PolyData(pts_merged) if pts_merged.shape[0] > 0 else None

if mesh is not None:
    p_before = visualize_splat(
        mesh=mesh,
        aligned_cameras=initial_chained,
        camera_kwargs={"n_poses": 3, "scale": 0.05},
    )
    p_before.show()

    p_after = visualize_splat(
        mesh=mesh,
        aligned_cameras=optimized,
        camera_kwargs={"n_poses": 3, "scale": 0.05},
    )
    p_after.show()
```

**Cell 19 (markdown):**
````markdown
## §9 Future: ground-truth comparison

Once a GT pose dataset is sourced (KITTI, outdoor with known poses), the
section below will quantify pose error against truth. Implementation lives
in `collab_splats/pointcloud/loop_closure/eval.py` (currently
NotImplementedError stubs).

```python
# from collab_splats.pointcloud.loop_closure.eval import (
#     umeyama_align, ate_translation, rpe,
# )
# gt_poses = load_gt(SCENE_DIR)  # (N, 4, 4)
# pred = np.stack(optimized)
# aligned, T = umeyama_align(pred, gt_poses)
# print("ATE:", ate_translation(aligned, gt_poses))
# print("RPE:", rpe(aligned, gt_poses, delta=1))
```
````

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c \
  "import json; json.load(open('docs/pointcloud/loop_closure_eval.ipynb'))"
```

Expected: no output (valid JSON).

- [ ] **Step 3: Verify imports work in nerfstudio env**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c \
  "from collab_splats.wrapper import Splatter, SplatterConfig; \
   from collab_splats.pointcloud.feedforward import MapAnythingCreator; \
   from collab_splats.pointcloud.loop_closure import LoopClosureConfig, PoseGraph, capture_pose_graph_loss; \
   from collab_splats.pointcloud.loop_closure.closure import build_pose_graph; \
   from collab_splats.utils.visualization import visualize_splat; \
   print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git add docs/pointcloud/loop_closure_eval.ipynb
git commit -m "feat(docs): add loop closure evaluation notebook

Walks through LC pipeline assembly on bicycle scene, captures pose-graph
residual reduction (loss curve + per-edge breakdown), visualizes pointcloud
with camera frustums before/after optimization. Future-GT section stubbed."
```

---

## Task 9: End-to-end verification on bicycle scene

**Files:** none (manual verification)

Goal: confirm the notebook actually runs end-to-end and produces the expected outputs on the bicycle scene.

- [ ] **Step 1: Verify bicycle scene exists**

```bash
ls /workspace/bicycle/images_4 | head -5
ls /workspace/bicycle/images_4 | wc -l
```

Expected: ≥40 image files (LC requires ≥submap_size = 20 frames; need at least 2 submaps for inter-submap edges to matter).

- [ ] **Step 2: Run the full unit-test suite to confirm no regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/ -v
```

Expected: all PASS.

- [ ] **Step 3: Execute notebook headlessly with nbconvert**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
  --to notebook --execute docs/pointcloud/loop_closure_eval.ipynb \
  --output loop_closure_eval.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Expected: notebook runs to completion. New file `docs/pointcloud/loop_closure_eval.executed.ipynb` produced.

If PyVista cells fail in headless environment: that's expected — open notebook in VSCode/JupyterLab instead and run cells manually. Document in PR description that PyVista cells require a display.

- [ ] **Step 4: Inspect executed notebook for acceptance criteria**

Open `docs/pointcloud/loop_closure_eval.executed.ipynb` (or run interactively). Verify:
- Cell 11 (§5) prints reduction percentage > 0%
- Cell 13 (§6) renders monotonically descending log-scale loss curve
- Cell 15 (§7) renders bar chart with each edge type's `final < initial`
- Cell 18 (§8) renders two PyVista plots; optimized trajectory visibly smoother than initial chained (manual visual check)
- Cell 19 (§9) renders as markdown only; commented code block visible

- [ ] **Step 5: Clean up the executed notebook artifact (don't commit)**

```bash
cd /workspace/collab-splats && rm -f docs/pointcloud/loop_closure_eval.executed.ipynb
```

- [ ] **Step 6: Final commit if notebook needed any cell adjustments during verification**

If any cell required tweaks during interactive run, commit them:

```bash
cd /workspace/collab-splats && git add docs/pointcloud/loop_closure_eval.ipynb
git commit -m "fix(docs): adjust loop closure notebook cells after end-to-end verification"
```

If no adjustments needed, skip this step.

---

## Self-Review Checklist

- [x] **Spec coverage:** every section of the design spec mapped to a task
  - eval.py module → Tasks 3, 4, 5, 6
  - tests → Tasks 3, 4, 5, 6
  - notebook → Task 8
  - LC state exposure → Task 2
  - closure split → Task 1
  - exports → Task 7
  - verification → Task 9
- [x] **No placeholders:** every code block contains the actual code
- [x] **Type consistency:** function names match across tasks (`capture_pose_graph_loss`, `build_pose_graph`, `_classify_edges`, `_per_edge_error`, `umeyama_align`, `ate_translation`, `rpe`)
- [x] **Repo conventions:** flat test functions; commit prefixes follow existing scope (`refactor(lc)`, `feat(lc)`, `feat(docs)`); no class-based tests; uses nerfstudio env Python explicitly

### Code Review Fixes (commit 63b2d18)

- **Tautological test fixed:** `test_build_pose_graph_no_loop_edges_when_lc_submaps_empty` was passing `lc_submaps=[]` to both branches, making the assertion trivially true. Fixed by constructing a real 2-frame LC submap (`submap_id=99`, `is_lc_submap=True`) and asserting `pg_with_lc._graph.size() > pg_no_lc._graph.size()`.
- **RNG determinism:** Added `np.random.seed(0)` as the first line of `_identity_submap` helper to ensure reproducible pose perturbations across runs.

## Out of Scope (per design spec)

- GT pose comparison (ATE/RPE) — stubs only
- VGGTX backend ablation — `BACKEND` toggle exists in notebook but not exercised in v1 acceptance
- Bundle-adjustment comparison
- C0043 hloc as pseudo-GT
