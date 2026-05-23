# SL(4) Loop Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dead SE(3)/Sim3 pose graph stack with a unified SL(4) PoseGraph (ported from VGGT-SLAM), consolidate 5 loop-closure files into 3, and add a 7-Scenes comparison eval.

**Architecture:** Per-frame SL(4) nodes (one per frame, key = global frame index) built via inner-submap chaining + inter-submap scale estimation from overlapping point clouds. `graph.py` replaces `pose_graph.py`; `closure.py` absorbs `retrieval.py` and `alignment.py`; `submap.py` gains reprojection helpers.

**Tech Stack:** gtsam-develop (SL4/BetweenFactorSL4/PriorFactorSL4), numpy, scipy (RQ decomp for decompose_camera), pytest. Python env: `/opt/conda/envs/nerfstudio/bin/python`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| CREATE | `collab_splats/pointcloud/loop_closure/graph.py` | PoseGraph (SL4+SE3 fallback), decompose_camera, normalize_to_sl4, estimate_scale_pairwise |
| MODIFY | `collab_splats/pointcloud/loop_closure/closure.py` | Absorb LoopClosureConfig/LoopMatch/LoopMatchQueue/find_loop_closures from retrieval.py; absorb dedup_overlap from alignment.py; replace run_sim3_pose_graph_optimization with SL4 version; update merge_submap_outputs |
| MODIFY | `collab_splats/pointcloud/loop_closure/submap.py` | Add get_world_points + get_poses_world methods |
| MODIFY | `collab_splats/pointcloud/loop_closure/__init__.py` | Remove dead exports, add new ones |
| MODIFY | `collab_splats/pointcloud/wrappers.py` | Drop ImageRetrieval import, use closure.find_loop_closures directly, pass graph to merge_submap_outputs |
| DELETE | `collab_splats/pointcloud/loop_closure/pose_graph.py` | Replaced by graph.py |
| DELETE | `collab_splats/pointcloud/loop_closure/retrieval.py` | Absorbed into closure.py |
| DELETE | `collab_splats/pointcloud/loop_closure/alignment.py` | dedup_overlap → closure.py; rest deleted |
| CREATE | `tests/pointcloud/test_graph.py` | graph.py unit tests |
| CREATE | `tests/pointcloud/test_submap_reprojection.py` | submap reprojection method tests |
| MODIFY | `tests/pointcloud/test_alignment_dedup.py` | Update import: alignment → closure |
| DELETE | `tests/pointcloud/test_alignment_umeyama.py` | Functions deleted (only used by dead SE3/Sim3 paths) |
| CREATE | `evals/eval_vggt_slam_comparison.py` | 7-Scenes: baseline / lc_se3 / lc_sl4 / vggt_slam_oob comparison |

---

### Task 1: graph.py module-level helpers

**Files:**
- Create: `collab_splats/pointcloud/loop_closure/graph.py`
- Create: `tests/pointcloud/test_graph.py`

- [ ] **Step 1: Write failing tests for helpers**

```python
# tests/pointcloud/test_graph.py
from __future__ import annotations
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.pointcloud.loop_closure.graph import (
    decompose_camera,
    normalize_to_sl4,
    estimate_scale_pairwise,
)


def test_normalize_to_sl4_det_one():
    H = np.random.default_rng(0).random((4, 4)).astype(np.float64) + np.eye(4)
    H_norm = normalize_to_sl4(H)
    assert abs(np.linalg.det(H_norm) - 1.0) < 1e-9


def test_normalize_to_sl4_singular_raises():
    with pytest.raises(ValueError, match="singular"):
        normalize_to_sl4(np.zeros((4, 4)))


def test_decompose_camera_round_trip():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    R = ScipyR.from_euler("y", 15, degrees=True).as_matrix()
    t = np.array([0.1, -0.2, 0.5])
    P34 = K @ np.hstack([R, t[:, None]])   # (3, 4)
    K_out, R_out, t_out, scale = decompose_camera(P34)
    assert np.allclose(K_out[:3, :3] / K_out[0, 0], K / K[0, 0], atol=1e-6)
    assert np.allclose(np.abs(R_out), np.abs(R), atol=1e-5)


def test_decompose_camera_accepts_4x4():
    K = np.eye(3, dtype=np.float64) * 400.0
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3)
    P34 = K @ np.hstack([R, t[:, None]])
    P44 = np.vstack([P34, [0, 0, 0, 1]])
    K_out, R_out, t_out, scale = decompose_camera(P44)
    assert K_out.shape[0] == 3


def test_estimate_scale_pairwise_known():
    rng = np.random.default_rng(1)
    X = rng.random((50, 3)).astype(np.float64)
    Y = X * 2.5                              # exact scale = 2.5
    scale = estimate_scale_pairwise(X, Y)
    assert abs(scale - 2.5) < 0.01


def test_estimate_scale_pairwise_no_div_zero():
    X = np.zeros((5, 3), dtype=np.float64)   # all at origin
    Y = np.ones((5, 3), dtype=np.float64)
    scale = estimate_scale_pairwise(X, Y)    # should not raise
    assert np.isfinite(scale)
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_graph.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'decompose_camera'`

- [ ] **Step 3: Create graph.py with helpers only**

```python
# collab_splats/pointcloud/loop_closure/graph.py
"""Factor graph over camera projection matrices.

Optimizes camera poses on the SL(4) manifold (or SE(3) fallback) to correct
trajectory drift and close loops. Per-frame nodes — one SL(4) node per frame,
key = global frame index. Ported and adapted from MIT-SPARK/VGGT-SLAM
vggt_slam/graph.py (SL4 backend) and vggt_slam/slam_utils.py (decompose_camera,
normalize_to_sl4) and vggt_slam/scale_solver.py (estimate_scale_pairwise).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import rq

########################################
########## Module-level helpers ########
########################################


def decompose_camera(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RQ decompose 3×4 or 4×4 projection matrix → (K, R, t, scale).

    Source: MIT-SPARK/VGGT-SLAM vggt_slam/slam_utils.py:decompose_camera
    """
    P = np.array(P, dtype=np.float64)
    if P.shape[0] != 3:
        P = P / P[-1, -1]
        P = P[:3, :]
    assert P.shape == (3, 4), f"expected (3,4) after strip, got {P.shape}"

    M = P[:, :3]
    K, R = rq(M)
    # ensure positive diagonal on K
    T = np.diag(np.sign(np.diag(K)).astype(np.float64))
    T[T == 0] = 1.0
    K = K @ T
    R = T @ R
    t = np.linalg.solve(K, P[:, 3])
    scale = float(K[0, 0])
    return K, R, t, scale


def normalize_to_sl4(H: np.ndarray) -> np.ndarray:
    """Normalize 4×4 matrix so det=1 (SL(4) constraint): H / det(H)^(1/4).

    Source: MIT-SPARK/VGGT-SLAM vggt_slam/slam_utils.py:normalize_to_sl4
    """
    H = np.array(H, dtype=np.float64)
    det = np.linalg.det(H)
    if abs(det) < 1e-12:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    return H / (abs(det) ** 0.25)


def estimate_scale_pairwise(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate scale between two point clouds: median(||Y[i]|| / ||X[i]||).

    Used to initialize inter-submap SL(4) edge H with correct scale.
    Source: MIT-SPARK/VGGT-SLAM vggt_slam/scale_solver.py:estimate_scale_pairwise
    """
    assert X.shape == Y.shape
    x_norms = np.linalg.norm(X, axis=1)
    y_norms = np.linalg.norm(Y, axis=1)
    valid = x_norms > 1e-8
    if not np.any(valid):
        return 1.0
    return float(np.median(y_norms[valid] / x_norms[valid]))
```

- [ ] **Step 4: Run helper tests — expect PASS**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_graph.py::test_normalize_to_sl4_det_one tests/pointcloud/test_graph.py::test_normalize_to_sl4_singular_raises tests/pointcloud/test_graph.py::test_decompose_camera_round_trip tests/pointcloud/test_graph.py::test_decompose_camera_accepts_4x4 tests/pointcloud/test_graph.py::test_estimate_scale_pairwise_known tests/pointcloud/test_graph.py::test_estimate_scale_pairwise_no_div_zero -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/graph.py tests/pointcloud/test_graph.py
git commit -m "feat(lc): add graph.py with decompose_camera, normalize_to_sl4, estimate_scale_pairwise"
```

---

### Task 2: PoseGraph class in graph.py

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/graph.py`
- Modify: `tests/pointcloud/test_graph.py`

- [ ] **Step 1: Write failing PoseGraph tests (append to test_graph.py)**

```python
# Append to tests/pointcloud/test_graph.py
import gtsam
from collab_splats.pointcloud.loop_closure.graph import PoseGraph


def _identity_H() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _translate_H(tx: float, ty: float, tz: float) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [tx, ty, tz]
    return H


def test_sl4_add_node_initializes():
    pg = PoseGraph(manifold="sl4")
    pg.add_node(0, _identity_H())
    pg.add_node(1, _translate_H(0.1, 0, 0))
    assert 0 in pg._node_ids
    assert 1 in pg._node_ids


def test_sl4_add_node_duplicate_noop():
    pg = PoseGraph(manifold="sl4")
    pg.add_node(0, _identity_H())
    pg.add_node(0, _translate_H(1, 1, 1))  # duplicate — must not raise or re-insert
    assert len(pg._node_ids) == 1


def test_sl4_sequential_edge_optimize():
    pg = PoseGraph(manifold="sl4")
    H0 = normalize_to_sl4(_identity_H())
    H1 = normalize_to_sl4(_translate_H(0.1, 0, 0))
    pg.add_node(0, H0)
    pg.add_node(1, H1)
    pg.add_prior(0, H0)
    H_rel = normalize_to_sl4(np.linalg.inv(H0) @ H1)
    pg.add_sequential_edge(0, 1, H_rel)
    pg.optimize()
    H0_out = pg.get_homography(0)
    assert H0_out.shape == (4, 4)
    assert np.allclose(H0_out, H0, atol=0.05)


def test_sl4_loop_edge_no_crash():
    pg = PoseGraph(manifold="sl4")
    Hs = [normalize_to_sl4(_translate_H(i * 0.1, 0, 0)) for i in range(3)]
    for i, H in enumerate(Hs):
        pg.add_node(i, H)
    pg.add_prior(0, Hs[0])
    pg.add_sequential_edge(0, 1, normalize_to_sl4(np.linalg.inv(Hs[0]) @ Hs[1]))
    pg.add_sequential_edge(1, 2, normalize_to_sl4(np.linalg.inv(Hs[1]) @ Hs[2]))
    pg.add_loop_edge(2, 0, normalize_to_sl4(np.linalg.inv(Hs[2]) @ Hs[0]), t_norm=0.2)
    pg.optimize()   # must not raise


def test_get_homography_post_optimize():
    pg = PoseGraph(manifold="sl4")
    H0 = normalize_to_sl4(_identity_H())
    pg.add_node(0, H0)
    pg.add_prior(0, H0)
    pg.optimize()
    H_out = pg.get_homography(0)
    assert H_out.shape == (4, 4)
    assert np.isfinite(H_out).all()


def test_se3_fallback_optimize():
    pg = PoseGraph(manifold="se3")
    H0 = _identity_H()
    H1 = _translate_H(0.1, 0, 0)
    pg.add_node(0, H0)
    pg.add_node(1, H1)
    pg.add_prior(0, H0)
    pg.add_sequential_edge(0, 1, np.linalg.inv(H0) @ H1)
    pg.optimize()
    H_out = pg.get_homography(0)
    assert H_out.shape == (4, 4)
```

- [ ] **Step 2: Run — expect ImportError/AttributeError on PoseGraph**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_graph.py -k "PoseGraph or sl4 or se3_fallback or loop_edge or sequential_edge or get_homography" -v 2>&1 | head -20
```

Expected: `ImportError` or `cannot import name 'PoseGraph'`

- [ ] **Step 3: Append PoseGraph class to graph.py**

```python
# Append to collab_splats/pointcloud/loop_closure/graph.py
import logging
from typing import Literal

import gtsam
from gtsam.symbol_shorthand import X

log = logging.getLogger(__name__)

_HUBER_K = 1.0
_LOOP_DOWNWEIGHT_ALPHA = 0.1

########################################
########## SE(3) helpers (fallback) ####
########################################


def _pose3(mat: np.ndarray) -> gtsam.Pose3:
    R = gtsam.Rot3(mat[:3, :3].astype(np.float64))
    t = gtsam.Point3(mat[:3, 3].astype(np.float64))
    return gtsam.Pose3(R, t)


########################################
########## PoseGraph class #############
########################################


class PoseGraph:
    """Per-frame SL(4) (or SE(3)) factor graph for loop closure trajectory correction.

    One node per frame; key = global frame index via gtsam.symbol('x', node_id).
    Ported and adapted from MIT-SPARK/VGGT-SLAM vggt_slam/graph.py.
    Additions over VGGT-SLAM baseline: SE(3) fallback, Huber kernel on loop edges,
    per-edge translation magnitude downweighting.
    """

    def __init__(self, manifold: Literal["sl4", "se3"] = "sl4") -> None:
        self._manifold = manifold
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._node_ids: set[int] = set()

        if manifold == "sl4":
            self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(
                0.05 * np.ones(15, dtype=np.float64)
            )
            self._loop_sigmas = 0.15 * np.ones(15, dtype=np.float64)
            self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.full(15, 1e-6, dtype=np.float64)
            )
        else:
            # SE(3) 6-DOF — existing tuned sigmas from ADR 005
            self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.05, 0.05, 0.05, 0.20, 0.20, 0.20], dtype=np.float64)
            )
            self._loop_sigmas = np.array(
                [0.15, 0.15, 0.15, 0.50, 0.50, 0.50], dtype=np.float64
            )
            self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.full(6, 1e-6, dtype=np.float64)
            )

    # ---- node management ----

    def add_node(self, node_id: int, H: np.ndarray) -> None:
        """Insert per-frame node. H is 4×4; normalize_to_sl4 applied internally."""
        if node_id in self._node_ids:
            return
        key = X(node_id)
        H = np.array(H, dtype=np.float64)
        if self._manifold == "sl4":
            self._initial.insert(key, gtsam.SL4(normalize_to_sl4(H)))
        else:
            self._initial.insert(key, _pose3(H))
        self._node_ids.add(node_id)

    def add_prior(self, node_id: int, H: np.ndarray) -> None:
        """Tight prior anchoring the first frame (σ=1e-6)."""
        key = X(node_id)
        H = np.array(H, dtype=np.float64)
        if self._manifold == "sl4":
            self._graph.add(
                gtsam.PriorFactorSL4(key, gtsam.SL4(normalize_to_sl4(H)), self._anchor_noise)
            )
        else:
            self._graph.add(
                gtsam.PriorFactorPose3(key, _pose3(H), self._anchor_noise)
            )

    # ---- edges ----

    def add_sequential_edge(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """Sequential frame-to-frame odometry constraint."""
        key_i, key_j = X(id_i), X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        if self._manifold == "sl4":
            self._graph.add(
                gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), self._seq_noise)
            )
        else:
            self._graph.add(
                gtsam.BetweenFactorPose3(key_i, key_j, _pose3(H_rel), self._seq_noise)
            )

    def add_loop_edge(
        self, id_i: int, id_j: int, H_rel: np.ndarray, t_norm: float = 0.0
    ) -> None:
        """Loop closure constraint with Huber kernel + translation downweighting."""
        key_i, key_j = X(id_i), X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        scale = 1.0 + _LOOP_DOWNWEIGHT_ALPHA * (t_norm ** 2)
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(_HUBER_K),
            gtsam.noiseModel.Diagonal.Sigmas(self._loop_sigmas * scale),
        )
        if self._manifold == "sl4":
            self._graph.add(gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), robust))
        else:
            self._graph.add(gtsam.BetweenFactorPose3(key_i, key_j, _pose3(H_rel), robust))

    # ---- optimization ----

    def optimize(self) -> None:
        """Levenberg-Marquardt in-place. Updates internal values; use get_homography after."""
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self._graph, self._initial, params
            )
            self._initial = optimizer.optimize()
        except Exception as exc:
            log.warning("GTSAM optimization failed: %s — returning initial values", exc)

    def get_homography(self, node_id: int) -> np.ndarray:
        """Return 4×4 H for a frame node. Call after optimize()."""
        key = X(node_id)
        if self._manifold == "sl4":
            return self._initial.atSL4(key).matrix().astype(np.float64)
        else:
            return self._initial.atPose3(key).matrix().astype(np.float64)
```

- [ ] **Step 4: Run PoseGraph tests — expect PASS**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_graph.py -v
```

Expected: all tests pass (11 total)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/graph.py tests/pointcloud/test_graph.py
git commit -m "feat(lc): add PoseGraph class with SL(4) per-frame nodes and SE(3) fallback"
```

---

### Task 3: Submap reprojection methods

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/submap.py`
- Create: `tests/pointcloud/test_submap_reprojection.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pointcloud/test_submap_reprojection.py
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.loop_closure.submap import Submap


def _make_submap(k: int = 3, n_pts: int = 10) -> Submap:
    rng = np.random.default_rng(42)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, 3] = rng.standard_normal((k, 3)).astype(np.float32) * 0.01
    world_points = rng.standard_normal((k, n_pts, 3)).astype(np.float32)
    return Submap(
        submap_id=0,
        frames=torch.zeros(k, 3, 8, 8),
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=torch.zeros(k, 32),
        image_paths=[Path(f"f{i}.jpg") for i in range(k)],
        world_points=world_points,
    )


def test_get_world_points_none_returns_local():
    sm = _make_submap()
    out = sm.get_world_points(H=None)
    assert out.shape == sm.world_points.shape
    assert np.allclose(out, sm.world_points)


def test_get_world_points_identity_noop():
    sm = _make_submap()
    out = sm.get_world_points(H=np.eye(4))
    assert np.allclose(out, sm.world_points, atol=1e-5)


def test_get_world_points_translation():
    sm = _make_submap(k=2, n_pts=5)
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [1.0, 2.0, 3.0]
    out = sm.get_world_points(H=H)   # shape (k*n_pts, 3) or (k, n_pts, 3)
    local_flat = sm.world_points.reshape(-1, 3).astype(np.float64)
    expected = local_flat + np.array([1.0, 2.0, 3.0])
    assert np.allclose(out.reshape(-1, 3), expected, atol=1e-5)


def test_get_world_points_projective_dehomogenizes():
    sm = _make_submap(k=1, n_pts=4)
    # Non-trivial SL4 H with projective component
    H = np.eye(4, dtype=np.float64)
    H[3, :3] = [0.01, 0.01, 0.01]  # projective row → w != 1
    H /= np.linalg.det(H) ** 0.25
    out = sm.get_world_points(H=H)
    assert out.shape == sm.world_points.shape
    assert np.isfinite(out).all()


def test_get_world_points_no_world_points_raises():
    sm = _make_submap()
    sm.world_points = None
    with pytest.raises(ValueError):
        sm.get_world_points(H=None)


def test_get_poses_world_none_returns_poses():
    sm = _make_submap()
    out = sm.get_poses_world(H=None)
    assert np.allclose(out, sm.poses)


def test_get_poses_world_known_H():
    sm = _make_submap(k=3)
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [5.0, 0.0, 0.0]
    out = sm.get_poses_world(H=H)
    assert out.shape == (3, 4, 4)
    # first pose in output ≈ H @ poses[0]
    expected0 = (H @ sm.poses[0].astype(np.float64)).astype(np.float32)
    assert np.allclose(out[0], expected0, atol=1e-5)
```

- [ ] **Step 2: Run — expect AttributeError**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_submap_reprojection.py -v 2>&1 | head -20
```

Expected: `AttributeError: 'Submap' object has no attribute 'get_world_points'`

- [ ] **Step 3: Add methods to submap.py**

Open `collab_splats/pointcloud/loop_closure/submap.py` and add the two methods to the `Submap` dataclass, before the closing of the class. Insert after the `world_points_conf` field:

```python
    def get_world_points(self, H: np.ndarray | None = None) -> np.ndarray:
        """Return world_points in global frame.

        H=None: return local frame as-is (per-submap visualization).
        H=(4,4): apply SL(4) projective transform, dehomogenize by /w.
        Requires world_points in per-camera local frame (PR #39 constraint).
        """
        if self.world_points is None:
            raise ValueError(f"Submap {self.submap_id} has no world_points")
        pts = self.world_points  # (K, P, 3)
        if H is None:
            return pts
        H = np.array(H, dtype=np.float64)
        k, p, _ = pts.shape
        flat = pts.reshape(-1, 3).astype(np.float64)           # (K*P, 3)
        ones = np.ones((flat.shape[0], 1), dtype=np.float64)
        hom = np.hstack([flat, ones])                          # (K*P, 4)
        out_hom = (H @ hom.T).T                                # (K*P, 4)
        w = out_hom[:, 3:4]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        return (out_hom[:, :3] / w).reshape(k, p, 3).astype(np.float32)

    def get_poses_world(self, H: np.ndarray | None = None) -> np.ndarray:
        """Return (K, 4, 4) poses in global frame.

        H=None: return self.poses unchanged (local, first pose ≈ identity).
        H=(4,4): apply corrected anchor H to each frame's local pose.
        """
        if H is None:
            return self.poses
        H = np.array(H, dtype=np.float64)
        return np.stack([
            (H @ p.astype(np.float64)).astype(np.float32)
            for p in self.poses
        ])
```

- [ ] **Step 4: Run reprojection tests — expect PASS**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_submap_reprojection.py -v
```

Expected: all 7 tests pass

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/submap.py tests/pointcloud/test_submap_reprojection.py
git commit -m "feat(lc): add get_world_points and get_poses_world to Submap"
```

---

### Task 4: Consolidate closure.py

Absorb `LoopClosureConfig`, `LoopMatch`, `LoopMatchQueue`, `find_loop_closures` from `retrieval.py`; absorb `dedup_overlap` from `alignment.py`; replace `run_sim3_pose_graph_optimization` with SL(4) `run_pose_graph_optimization`; update `merge_submap_outputs`.

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`

- [ ] **Step 1: Write failing test for new run_pose_graph_optimization**

```python
# Append to tests/pointcloud/test_graph.py
from pathlib import Path
import torch
from collab_splats.pointcloud.loop_closure.submap import Submap
from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization


def _make_real_submap(submap_id: int, k: int = 4, frame_start: int = 0) -> Submap:
    rng = np.random.default_rng(submap_id)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, 3] = (rng.standard_normal((k, 3)) * 0.05).astype(np.float32)
    intrinsics = np.tile(
        np.diag([400.0, 400.0, 1.0]).astype(np.float32), (k, 1, 1)
    )
    world_points = rng.standard_normal((k, 20, 3)).astype(np.float32)
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 8, 8),
        poses=poses,
        intrinsics=intrinsics,
        retrieval_vectors=torch.zeros(k, 32),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
        frame_start=frame_start,
        world_points=world_points,
    )


def test_run_pose_graph_optimization_returns_correct_shape():
    k = 4
    submaps = [_make_real_submap(0, k=k, frame_start=0),
               _make_real_submap(1, k=k, frame_start=k)]
    result = run_pose_graph_optimization(
        submaps, lc_submaps=[], total_frames=k * 2,
        overlap_frames=1, manifold="sl4",
    )
    assert result.shape == (k * 2, 4, 4)
    assert np.isfinite(result).all()


def test_run_pose_graph_optimization_se3_fallback():
    k = 3
    submaps = [_make_real_submap(0, k=k, frame_start=0)]
    result = run_pose_graph_optimization(
        submaps, lc_submaps=[], total_frames=k,
        overlap_frames=1, manifold="se3",
    )
    assert result.shape == (k, 4, 4)
```

- [ ] **Step 2: Run — expect ImportError on run_pose_graph_optimization**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_graph.py::test_run_pose_graph_optimization_returns_correct_shape -v 2>&1 | head -15
```

Expected: `ImportError` (function doesn't exist yet with new sig) or signature mismatch

- [ ] **Step 3: Add to top of closure.py — absorb LoopClosureConfig, LoopMatch, LoopMatchQueue, find_loop_closures**

Open `collab_splats/pointcloud/loop_closure/closure.py`. The file currently imports from `retrieval.py`. Replace those imports and add the types/NMS queue at the top. The full block to add **before** existing function definitions:

```python
# Absorbed from retrieval.py — LoopClosureConfig, LoopMatch, LoopMatchQueue, find_loop_closures
import heapq
import math
import warnings
from typing import Literal

@dataclass
class LoopMatch:
    """Loop closure candidate produced by DINO-SALAD retrieval."""
    similarity_score: float
    query_submap_id: int
    detected_submap_id: int
    query_frame_idx: int
    detected_frame_idx: int
    accepted: bool = False

@dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 4
    lc_cosine_threshold: float = 0.75
    max_loops_per_submap: int = 5
    verify_match_ratio: float = 0.85
    nms_frame_distance: int = 25
    min_submap_gap: int = 1
    manifold: Literal["sl4", "se3"] = "sl4"
    lc_threshold: float | None = None   # deprecated

    def __post_init__(self) -> None:
        if self.lc_threshold is not None:
            warnings.warn(
                "LoopClosureConfig.lc_threshold is deprecated; use lc_cosine_threshold. "
                f"Equivalent: {1 - self.lc_threshold**2 / 2:.4f}",
                DeprecationWarning, stacklevel=2,
            )

    @property
    def lc_threshold_l2(self) -> float:
        return math.sqrt(2 * (1 - self.lc_cosine_threshold))


class LoopMatchQueue:
    """Max-heap keeping top-k lowest-distance LoopMatch candidates, with NMS."""

    def __init__(self, max_size: int, nms_frame_distance: int = 0) -> None:
        self._max_size = max_size
        self._nms = nms_frame_distance
        self._counter: int = 0
        self._heap: list = []

    def push(self, match: LoopMatch) -> None:
        heapq.heappush(self._heap, (-match.similarity_score, self._counter, match))
        self._counter += 1
        if len(self._heap) > self._max_size:
            heapq.heappop(self._heap)

    def get_matches(self) -> list[LoopMatch]:
        candidates = sorted(
            [m for _, _, m in self._heap], key=lambda m: m.similarity_score
        )
        if self._nms <= 0:
            return candidates
        accepted: list[LoopMatch] = []
        for cand in candidates:
            suppressed = any(
                acc.detected_submap_id == cand.detected_submap_id
                and abs(acc.detected_frame_idx - cand.detected_frame_idx) < self._nms
                for acc in accepted
            )
            if not suppressed:
                accepted.append(cand)
        return accepted


def find_loop_closures(
    query_submap: "Submap",
    past_submaps: list["Submap"],
    lc_threshold: float,
    max_loops: int,
    nms_frame_distance: int = 0,
) -> list[LoopMatch]:
    """Return top-k loop closure candidates using pre-computed retrieval_vectors."""
    import torch
    if not past_submaps:
        return []
    queue = LoopMatchQueue(max_size=max_loops, nms_frame_distance=nms_frame_distance)
    for q_idx in range(query_submap.retrieval_vectors.shape[0]):
        q_vec = query_submap.retrieval_vectors[q_idx]
        for past in past_submaps:
            dists = torch.cdist(q_vec.unsqueeze(0), past.retrieval_vectors).squeeze(0)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])
            if best_dist < lc_threshold:
                queue.push(LoopMatch(
                    similarity_score=best_dist,
                    query_submap_id=query_submap.submap_id,
                    detected_submap_id=past.submap_id,
                    query_frame_idx=q_idx,
                    detected_frame_idx=best_idx,
                ))
    return queue.get_matches()
```

- [ ] **Step 4: Add dedup_overlap (move from alignment.py) to closure.py**

Append to `closure.py` (copy the existing implementation from `alignment.py`):

```python
########################################
########## Absorbed from alignment.py ##
########################################

def dedup_overlap(
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct (total_frames, 4, 4) from per-submap corrected poses, deduplicating overlap."""
    out = np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
    assigned = np.zeros(total_frames, dtype=bool)
    for sid, start in zip(submap_ids, submap_starts):
        poses = corrected[sid]  # (K_i, 4, 4)
        k = poses.shape[0]
        for local_i in range(k):
            global_i = start + local_i
            if 0 <= global_i < total_frames and not assigned[global_i]:
                out[global_i] = poses[local_i]
                assigned[global_i] = True
    return out
```

- [ ] **Step 5: Replace run_sim3_pose_graph_optimization with SL(4) run_pose_graph_optimization**

Find the existing `run_sim3_pose_graph_optimization` function in `closure.py` and replace it (and `build_pose_graph` and `run_pose_graph_optimization` SE3 version) with the new implementation:

```python
########################################
####### SL(4) pose graph optimization ##
########################################

def run_pose_graph_optimization(
    submaps: list["Submap"],
    lc_submaps: list["Submap"],
    total_frames: int,
    overlap_frames: int,
    manifold: Literal["sl4", "se3"] = "sl4",
) -> np.ndarray:
    """Build + optimize per-frame SL(4) pose graph; return (total_frames, 4, 4).

    Per-frame node building mirrors vggt_slam/solver.py:add_edge:
    - Inner frames: H_inner = poses[i-1] @ inv(poses[i]); node chained from prev
    - Inter-submap first frame: scale estimated via estimate_scale_pairwise on
      overlapping world_points, H_w = graph.get_homography(overlap_prev) @ inv(K_prev) @ K_curr @ H_scale
    - Loop edges from lc_submaps (2-frame submaps with verified LC poses)
    """
    from .graph import PoseGraph, decompose_camera, estimate_scale_pairwise, normalize_to_sl4

    pg = PoseGraph(manifold=manifold)
    global_node_id = 0
    # Maps (submap_id, local_frame_idx) → global_node_id for loop edge resolution
    frame_to_node: dict[tuple[int, int], int] = {}
    # Maps submap_id → list of global_node_ids for that submap
    submap_node_ids: dict[int, list[int]] = {}

    for s_idx, submap in enumerate(submaps):
        k = submap.poses.shape[0]
        node_ids_this: list[int] = []

        for local_i in range(k):
            nid = global_node_id + local_i
            frame_to_node[(submap.submap_id, local_i)] = nid
            node_ids_this.append(nid)

        # Build node initial values
        if s_idx == 0:
            # First submap: anchor at identity (poses[0] ≈ identity per assert_world_to_cam)
            H0 = submap.poses[0].astype(np.float64)
            pg.add_node(node_ids_this[0], H0)
            pg.add_prior(node_ids_this[0], H0)
            # Inner frames: chain from previous node
            for local_i in range(1, k):
                H_inner = (
                    submap.poses[local_i - 1].astype(np.float64)
                    @ np.linalg.inv(submap.poses[local_i].astype(np.float64))
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(
                    node_ids_this[local_i - 1], node_ids_this[local_i], H_inner
                )
        else:
            prev_submap = submaps[s_idx - 1]
            O = min(overlap_frames, k, len(prev_submap.poses))

            # Inter-submap scale estimation from overlapping world_points
            scale = 1.0
            if (
                submap.world_points is not None
                and prev_submap.world_points is not None
                and O > 0
            ):
                curr_pts = submap.world_points[:O].reshape(-1, 3).astype(np.float64)
                prev_pts = prev_submap.world_points[-O:].reshape(-1, 3).astype(np.float64)
                # Intrinsic correction: express curr pts in prev's intrinsic space
                K_prev = np.eye(4, dtype=np.float64)
                K_prev[:3, :3] = prev_submap.intrinsics[-1].astype(np.float64)
                K_curr = np.eye(4, dtype=np.float64)
                K_curr[:3, :3] = submap.intrinsics[0].astype(np.float64)
                P_temp = np.linalg.inv(K_prev) @ K_curr
                curr_in_prev = (P_temp[:3, :3] @ curr_pts.T).T
                scale = estimate_scale_pairwise(curr_in_prev, prev_pts)

            H_scale = np.diag([scale, scale, scale, 1.0])
            K_prev4 = np.eye(4, dtype=np.float64)
            K_prev4[:3, :3] = prev_submap.intrinsics[-1].astype(np.float64)
            K_curr4 = np.eye(4, dtype=np.float64)
            K_curr4[:3, :3] = submap.intrinsics[0].astype(np.float64)

            # Node for first frame of new submap
            prev_submap_last_nid = submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = pg.get_homography(prev_submap_last_nid)
            H_w = H_overlap @ np.linalg.inv(K_prev4) @ K_curr4 @ H_scale
            pg.add_node(node_ids_this[0], H_w)

            # Inter-submap sequential edge
            H_rel_inter = (
                np.linalg.inv(pg.get_homography(prev_submap_last_nid)) @ H_w
            )
            pg.add_sequential_edge(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

            # Inner frames of new submap
            for local_i in range(1, k):
                H_inner = (
                    submap.poses[local_i - 1].astype(np.float64)
                    @ np.linalg.inv(submap.poses[local_i].astype(np.float64))
                )
                prev_H = pg.get_homography(node_ids_this[local_i - 1])
                pg.add_node(node_ids_this[local_i], prev_H @ H_inner)
                pg.add_sequential_edge(
                    node_ids_this[local_i - 1], node_ids_this[local_i], H_inner
                )

        submap_node_ids[submap.submap_id] = node_ids_this
        global_node_id += k

    # Loop closure edges
    for lc in lc_submaps:
        if lc.poses.shape[0] != 2:
            continue
        path_q, path_d = lc.image_paths[0], lc.image_paths[1]
        nid_q = _resolve_frame_node(frame_to_node, submaps, path_q)
        nid_d = _resolve_frame_node(frame_to_node, submaps, path_d)
        if nid_q is None or nid_d is None:
            continue
        H_rel_lc = (
            np.linalg.inv(lc.poses[0].astype(np.float64))
            @ lc.poses[1].astype(np.float64)
        )
        t_norm = float(np.linalg.norm(lc.poses[1][:3, 3] - lc.poses[0][:3, 3]))
        pg.add_loop_edge(nid_q, nid_d, H_rel_lc, t_norm=t_norm)

    pg.optimize()

    # Extract (total_frames, 4, 4) corrected extrinsics
    corrected_per_submap: dict[int, np.ndarray] = {}
    for submap in submaps:
        node_ids = submap_node_ids[submap.submap_id]
        k = len(node_ids)
        poses_out = np.zeros((k, 4, 4), dtype=np.float32)
        for local_i, nid in enumerate(node_ids):
            H_opt = pg.get_homography(nid)
            _, R, t, _ = decompose_camera(H_opt)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R.astype(np.float32)
            mat[:3, 3] = t.astype(np.float32)
            poses_out[local_i] = mat
        corrected_per_submap[submap.submap_id] = poses_out

    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected_per_submap,
        total_frames=total_frames,
    )


def _resolve_frame_node(
    frame_to_node: dict[tuple[int, int], int],
    submaps: list["Submap"],
    image_path,
) -> int | None:
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                return frame_to_node.get((submap.submap_id, local_i))
    return None
```

- [ ] **Step 6: Update merge_submap_outputs signature in closure.py**

Find the existing `merge_submap_outputs` function. Change the `sim3_nodes` parameter to `graph`:

```python
def merge_submap_outputs(
    submaps: list["Submap"],
    corrected_extrinsics: np.ndarray,
    graph: "PoseGraph | None" = None,
) -> dict:
    """Assemble unified raw_outputs with corrected poses and globally-consistent world_points.

    graph: if provided (manifold="sl4"), calls submap.get_world_points(graph.get_homography(nid))
    per submap for globally-consistent point clouds.
    """
```

In the function body, replace the existing `sim3_nodes` world_points transformation block with:

```python
    # Apply SL(4) corrected homographies for globally-consistent world_points
    if graph is not None:
        for s in submaps:
            if s.world_points is not None:
                # Use first frame's node homography as submap anchor
                first_nid = s.frame_start
                try:
                    H_corr = graph.get_homography(first_nid)
                    # update in-place for merging
                    s_wp_corrected = s.get_world_points(H=H_corr)
                    # patch into merged["world_points"] below
                except Exception:
                    pass
```

  > **Note:** The exact merge_submap_outputs body is long. Only change: (1) parameter `sim3_nodes` → `graph: "PoseGraph | None" = None`; (2) replace the `if sim3_nodes is not None` block (Sim3 transform) with SL(4) block above. All other concatenation logic is unchanged.

- [ ] **Step 7: Run pose graph optimization tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_graph.py -v
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py
git commit -m "feat(lc): consolidate closure.py — absorb types from retrieval, dedup_overlap from alignment, SL4 run_pose_graph_optimization"
```

---

### Task 5: Delete dead files + fix broken tests + update __init__.py

**Files:**
- Delete: `collab_splats/pointcloud/loop_closure/pose_graph.py`
- Delete: `collab_splats/pointcloud/loop_closure/retrieval.py`
- Delete: `collab_splats/pointcloud/loop_closure/alignment.py`
- Delete: `tests/pointcloud/test_alignment_umeyama.py`
- Modify: `collab_splats/pointcloud/loop_closure/__init__.py`
- Modify: `tests/pointcloud/test_alignment_dedup.py`

- [ ] **Step 1: Check which tests import from deleted modules**

```bash
grep -r "from.*pose_graph\|from.*retrieval\|from.*alignment\|Sim3PoseGraph\|ImageRetrieval\|run_sim3\|umeyama_se3\|umeyama_sim3\|overlap_region_align" tests/ --include="*.py" -l
```

Note which files are listed — those need import updates.

- [ ] **Step 2: Update test_alignment_dedup.py import**

In `tests/pointcloud/test_alignment_dedup.py`, change:

```python
from collab_splats.pointcloud.loop_closure.alignment import dedup_overlap
```

to:

```python
from collab_splats.pointcloud.loop_closure.closure import dedup_overlap
```

- [ ] **Step 3: Delete dead test file**

```bash
rm tests/pointcloud/test_alignment_umeyama.py
```

- [ ] **Step 4: Delete dead source files**

```bash
rm collab_splats/pointcloud/loop_closure/pose_graph.py
rm collab_splats/pointcloud/loop_closure/retrieval.py
rm collab_splats/pointcloud/loop_closure/alignment.py
```

- [ ] **Step 5: Update __init__.py**

Replace `collab_splats/pointcloud/loop_closure/__init__.py` with:

```python
from .submap import Submap, assert_world_to_cam
from .closure import (
    LoopClosureConfig,
    LoopMatch,
    LoopMatchQueue,
    find_loop_closures,
    run_pose_graph_optimization,
    merge_submap_outputs,
    dedup_overlap,
    translation_jump_check,
)
from .graph import PoseGraph, decompose_camera, normalize_to_sl4, estimate_scale_pairwise
from .eval import capture_pose_graph_loss

__all__ = [
    "Submap",
    "assert_world_to_cam",
    "LoopClosureConfig",
    "LoopMatch",
    "LoopMatchQueue",
    "find_loop_closures",
    "run_pose_graph_optimization",
    "merge_submap_outputs",
    "dedup_overlap",
    "translation_jump_check",
    "PoseGraph",
    "decompose_camera",
    "normalize_to_sl4",
    "estimate_scale_pairwise",
    "capture_pose_graph_loss",
]
```

- [ ] **Step 6: Run full pointcloud test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v 2>&1 | tail -30
```

Fix any remaining import errors. Common pattern: any test that did `from collab_splats.pointcloud.loop_closure import PoseGraph` now works (same name, different implementation). Any test that imported `Sim3PoseGraph` or `run_sim3_pose_graph_optimization` — delete those specific test functions.

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "refactor(lc): delete dead pose_graph/retrieval/alignment files, update __init__.py exports"
```

---

### Task 6: Update wrappers.py

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py`

- [ ] **Step 1: Replace ImageRetrieval import in _run_lc_loop**

In `wrappers.py`, find:

```python
from collab_splats.pointcloud.loop_closure import ImageRetrieval, Submap
```

Replace with:

```python
from collab_splats.pointcloud.loop_closure import Submap
from collab_splats.pointcloud.loop_closure.closure import find_loop_closures
from collab_splats.pointcloud.localization import BaseRetrievalExtractor
```

- [ ] **Step 2: Replace ImageRetrieval instantiation**

Find:

```python
retrieval = ImageRetrieval(device=device)
self.base._lc_retrieval = retrieval
```

Replace with:

```python
retrieval = BaseRetrievalExtractor.get("dino-salad")(device=device)
self.base._lc_retrieval = retrieval
```

- [ ] **Step 3: Replace retrieval.find_loop_closures call**

Find:

```python
loop_matches = retrieval.find_loop_closures(
    submap, past_for_lc, cfg.lc_threshold_l2, cfg.max_loops_per_submap,
    nms_frame_distance=cfg.nms_frame_distance,
)
```

Replace with:

```python
loop_matches = find_loop_closures(
    submap, past_for_lc, cfg.lc_threshold_l2, cfg.max_loops_per_submap,
    nms_frame_distance=cfg.nms_frame_distance,
)
```

- [ ] **Step 4: Replace run_sim3_pose_graph_optimization call block**

Find the block:

```python
from collab_splats.pointcloud.loop_closure.closure import (
    merge_submap_outputs,
    run_sim3_pose_graph_optimization,
)
pg_result = run_sim3_pose_graph_optimization(
    submaps, lc_submaps, total_frames=N,
    overlap_frames=cfg.submap_overlap,
    lm_steps=cfg.sim3_lm_steps,
    return_trace=True,
)
...
self.base.raw_outputs = merge_submap_outputs(
    submaps, pg_result["corrected"], sim3_nodes=pg_result["sim3_nodes"]
)
```

Replace with:

```python
from collab_splats.pointcloud.loop_closure.closure import (
    merge_submap_outputs,
    run_pose_graph_optimization,
)
from collab_splats.pointcloud.loop_closure.graph import PoseGraph
corrected = run_pose_graph_optimization(
    submaps, lc_submaps, total_frames=N,
    overlap_frames=cfg.submap_overlap,
    manifold=cfg.manifold,
)
self.base.raw_outputs = merge_submap_outputs(submaps, corrected)
```

- [ ] **Step 5: Run wrappers smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/ tests/pointcloud/ -v 2>&1 | tail -30
```

Expected: all previously passing tests still pass

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py
git commit -m "refactor(lc): update wrappers.py — drop ImageRetrieval, wire SL4 run_pose_graph_optimization"
```

---

### Task 7: Evaluation script

**Files:**
- Create: `evals/eval_vggt_slam_comparison.py`

- [ ] **Step 1: Create eval script**

```python
# evals/eval_vggt_slam_comparison.py
"""7-Scenes comparison: baseline / lc_se3 / lc_sl4 / vggt_slam_oob.

Usage (run in tmux — GPU + memory intensive):
    /opt/conda/envs/nerfstudio/bin/python evals/eval_vggt_slam_comparison.py \
        --scene chess --seq seq-01 --data_root /data/7scenes

Requires evo: pip install evo
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

VGGT_SLAM_ARGS = [
    "--max_loops", "1",
    "--min_disparity", "50",
    "--conf_threshold", "25",
    "--lc_thres", "0.95",
    "--submap_size", "16",
    "--skip_dense_log",
]


def load_7scenes_images(scene_dir: Path) -> list[Path]:
    """Return sorted list of RGB image paths from a 7-Scenes sequence dir."""
    frames = sorted(scene_dir.glob("frame-*.color.png"))
    if not frames:
        raise FileNotFoundError(f"No frame-*.color.png in {scene_dir}")
    return frames


def load_7scenes_gt_poses(scene_dir: Path) -> np.ndarray:
    """Load ground-truth camera-to-world poses from 7-Scenes .pose.txt files.
    Returns (N, 4, 4) float32 world-to-cam (inverted from the stored cam-to-world).
    """
    pose_files = sorted(scene_dir.glob("frame-*.pose.txt"))
    poses = []
    for pf in pose_files:
        mat = np.loadtxt(pf, dtype=np.float64).reshape(4, 4)
        poses.append(np.linalg.inv(mat).astype(np.float32))
    return np.stack(poses)


def compute_ate(est_poses: np.ndarray, gt_poses: np.ndarray, tmp_dir: Path) -> float:
    """Write TUM-format trajectories and call evo_ape; return ATE RMSE (m)."""
    def _to_tum(poses: np.ndarray, path: Path) -> None:
        with open(path, "w") as f:
            for i, P in enumerate(poses):
                t = P[:3, 3]
                from scipy.spatial.transform import Rotation
                q = Rotation.from_matrix(P[:3, :3]).as_quat()  # xyzw
                f.write(f"{i} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    est_path = tmp_dir / "est.txt"
    gt_path = tmp_dir / "gt.txt"
    _to_tum(est_poses, est_path)
    _to_tum(gt_poses, gt_path)

    result = subprocess.run(
        ["evo_ape", "tum", str(gt_path), str(est_path), "--align", "--correct_scale",
         "--no_warnings", "--save_results", str(tmp_dir / "ape.zip")],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "rmse" in line.lower():
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    log.warning("evo_ape output: %s", result.stdout[-500:])
    return float("nan")


def run_baseline(image_paths: list[Path], output_dir: Path) -> np.ndarray:
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    creator = VGGTXCreator(image_paths=image_paths)
    result = creator.reconstruct(image_dir=image_paths[0].parent, output_dir=output_dir)
    return result.extrinsics  # (N, 4, 4)


def run_lc(
    image_paths: list[Path],
    output_dir: Path,
    manifold: str,
) -> np.ndarray:
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure.closure import LoopClosureConfig
    base = VGGTXCreator(image_paths=image_paths)
    cfg = LoopClosureConfig(manifold=manifold)
    lc = LoopClosure(base=base, config=cfg)
    result = lc.reconstruct(image_dir=image_paths[0].parent, output_dir=output_dir)
    return result.extrinsics


def run_vggt_slam_oob(image_paths: list[Path], output_dir: Path) -> np.ndarray:
    """Run VGGT-SLAM out-of-the-box via subprocess; parse TUM log → extrinsics."""
    from collab_splats.pointcloud.loop_closure.graph import decompose_camera

    # Rename images to %06d.png format (issue #43 workaround)
    renamed_dir = output_dir / "renamed_frames"
    renamed_dir.mkdir(exist_ok=True)
    import shutil
    for i, src in enumerate(image_paths):
        dst = renamed_dir / f"{i:06d}.png"
        if not dst.exists():
            shutil.copy(src, dst)

    log_path = output_dir / "vggt_slam_poses.txt"
    slam_main = Path(__file__).parents[1] / "third_party" / "VGGT-SLAM" / "main.py"
    cmd = [
        sys.executable, str(slam_main),
        "--image_folder", str(renamed_dir),
        "--log_results", "--log_path", str(log_path),
    ] + VGGT_SLAM_ARGS
    log.info("Running VGGT-SLAM: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Parse TUM-format log: timestamp tx ty tz qx qy qz qw
    from scipy.spatial.transform import Rotation
    poses = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]])  # xyzw
            R = Rotation.from_quat(q).as_matrix().astype(np.float32)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R
            mat[:3, 3] = t.astype(np.float32)
            poses.append(mat)
    return np.stack(poses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--seq", default="seq-01")
    parser.add_argument("--data_root", type=Path, default=Path("/data/7scenes"))
    parser.add_argument("--output_dir", type=Path, default=Path("evals/results/sl4_comparison"))
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "lc_se3", "lc_sl4", "vggt_slam_oob"])
    args = parser.parse_args()

    scene_dir = args.data_root / args.scene / args.seq
    output_dir = args.output_dir / args.scene / args.seq
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_7scenes_images(scene_dir)
    gt_poses = load_7scenes_gt_poses(scene_dir)
    log.info("Loaded %d frames from %s", len(image_paths), scene_dir)

    results: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if "baseline" in args.conditions:
            log.info("Running baseline...")
            est = run_baseline(image_paths, output_dir / "baseline")
            results["baseline"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("baseline ATE RMSE: %.4f m", results["baseline"])

        if "lc_se3" in args.conditions:
            log.info("Running lc_se3...")
            est = run_lc(image_paths, output_dir / "lc_se3", manifold="se3")
            results["lc_se3"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("lc_se3 ATE RMSE: %.4f m", results["lc_se3"])

        if "lc_sl4" in args.conditions:
            log.info("Running lc_sl4...")
            est = run_lc(image_paths, output_dir / "lc_sl4", manifold="sl4")
            results["lc_sl4"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("lc_sl4 ATE RMSE: %.4f m", results["lc_sl4"])

        if "vggt_slam_oob" in args.conditions:
            log.info("Running vggt_slam_oob...")
            est = run_vggt_slam_oob(image_paths, output_dir / "vggt_slam_oob")
            results["vggt_slam_oob"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("vggt_slam_oob ATE RMSE: %.4f m", results["vggt_slam_oob"])

    results_file = output_dir / "ate_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n=== ATE RMSE (m) — %s/%s ===", args.scene, args.seq)
    for cond, ate in results.items():
        log.info("  %-20s %.4f", cond, ate)
    log.info("Saved to %s", results_file)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without error**

```bash
/opt/conda/envs/nerfstudio/bin/python evals/eval_vggt_slam_comparison.py --help
```

Expected: argparse help printed, no ImportError

- [ ] **Step 3: Commit**

```bash
git add evals/eval_vggt_slam_comparison.py
git commit -m "feat(evals): add eval_vggt_slam_comparison.py — 7-Scenes SL4 vs VGGT-SLAM OOB"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| graph.py + decompose_camera + normalize_to_sl4 | Task 1 |
| estimate_scale_pairwise | Task 1 |
| PoseGraph per-frame SL4 nodes | Task 2 |
| SE(3) fallback manifold="se3" | Task 2 |
| Huber + translation downweighting on loop edges | Task 2 |
| Submap.get_world_points + get_poses_world | Task 3 |
| closure.py absorb LoopClosureConfig/LoopMatch/LoopMatchQueue | Task 4 |
| closure.py absorb dedup_overlap | Task 4 |
| run_pose_graph_optimization with inter-submap scale | Task 4 |
| merge_submap_outputs updated signature | Task 4 |
| Delete pose_graph.py / retrieval.py / alignment.py | Task 5 |
| __init__.py exports updated | Task 5 |
| wrappers.py: drop ImageRetrieval, wire SL4 | Task 6 |
| 7-Scenes eval script (all 4 conditions) | Task 7 |
| Key constraint: per-camera local frame (assert_world_to_cam) | enforced in submap.py (unchanged) |
| Key constraint: gtsam-develop SL4 import | verified pre-plan |
| Key constraint: attribution in graph.py docstring | Task 1 step 3 |

**No placeholders found.** All code blocks contain complete implementations.

**Type consistency check:**
- `PoseGraph.add_node(node_id: int, H: np.ndarray)` — consistent across Tasks 2, 4
- `run_pose_graph_optimization` returns `np.ndarray (total_frames, 4, 4)` — consistent with wrappers.py Task 6 usage
- `merge_submap_outputs(submaps, corrected_extrinsics, graph=None)` — consistent Task 4 def and Task 6 call
- `find_loop_closures` module-level in closure.py — consistent Task 4 def and Task 6 import
- `LoopClosureConfig.manifold` field — consistent Task 4 def and Task 6 `cfg.manifold` usage

---

Plan saved to `docs/superpowers/plans/2026-05-22-sl4-loop-closure.md`.
