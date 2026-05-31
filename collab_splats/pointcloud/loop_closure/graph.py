"""Factor graph over camera projection matrices.

Optimizes camera poses on the SL(4) manifold (or SE(3) fallback) to correct
trajectory drift and close loops. Per-frame nodes — one SL(4) node per frame,
key = global frame index. Ported and adapted from MIT-SPARK/VGGT-SLAM
vggt_slam/graph.py (SL4 backend) and vggt_slam/slam_utils.py (decompose_camera,
normalize_to_sl4) and vggt_slam/scale_solver.py (estimate_scale_pairwise).
"""
from __future__ import annotations

import logging
from typing import Literal

import gtsam
from gtsam.symbol_shorthand import X as _X
import numpy as np
from scipy.linalg import rq

########################################
########## Module-level helpers ########
########################################


def decompose_camera(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RQ decompose 3×4 or 4×4 projection matrix → (K, R, t, scale).

    Source: MIT-SPARK/VGGT-SLAM vggt_slam/slam_utils.py:decompose_camera.

    CONVENTION (critical): this matches VGGT-SLAM's ``no_inverse=True`` branch.
    R is the camera-to-world rotation and ``t = inv(K) @ P[:,3]`` (NOT a
    world-to-cam translation). SLAM's *default* branch returns
    ``t = -R @ inv(K) @ P[:,3]`` and uses it directly as the camera centre
    ``C = -R @ t``; so the world-to-cam pose is ``[R^T | t]``. Callers that
    store a world-to-cam extrinsic MUST transpose R (see closure.py pose
    extraction). Storing R directly yields centre ``-R^T @ t``, which agrees
    only for near-identity rotations (single submap) and bends multi-submap
    trajectories — this was the root cause of the vggt_spark↔VGGT-SLAM ATE
    gap (0.308m → 0.017m once fixed).
    """
    P = np.array(P, dtype=np.float64)
    if P.shape[0] != 3:
        P = P / P[-1, -1]
        P = P[:3, :]
    assert P.shape == (3, 4), f"expected (3,4) after strip, got {P.shape}"

    M = P[:, :3]
    K, R = rq(M)
    # ensure positive diagonal on K (per-column sign fix)
    if K[0, 0] < 0:
        K[:, 0] *= -1
        R[0, :] *= -1
    if K[1, 1] < 0:
        K[:, 1] *= -1
        R[1, :] *= -1
    if K[2, 2] < 0:
        K[:, 2] *= -1
        R[2, :] *= -1
    scale = float(K[2, 2])
    R = np.linalg.inv(R)
    # SVD polar-decomposition snap: enforce R is a proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    # t = inv(K) @ P[:,3] — VGGT-SLAM no_inverse=True value (see docstring).
    # NOT the world-to-cam translation; camera centre is C = -R @ t.
    t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
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
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} vs {Y.shape}")
    x_norms = np.linalg.norm(X, axis=1)
    y_norms = np.linalg.norm(Y, axis=1)
    valid = x_norms > 1e-8
    if not np.any(valid):
        return 1.0
    return float(np.median(y_norms[valid] / x_norms[valid]))


########################################
########## PoseGraph class #############
########################################

log = logging.getLogger(__name__)


def _pose3(mat: np.ndarray) -> gtsam.Pose3:
    """Convert 4×4 matrix to gtsam.Pose3 (SE(3) fallback helper)."""
    R = gtsam.Rot3(mat[:3, :3].astype(np.float64))
    t = gtsam.Point3(mat[:3, 3].astype(np.float64))
    return gtsam.Pose3(R, t)


class PoseGraph:
    """Per-frame SL(4) (or SE(3)) factor graph for loop closure trajectory correction.

    One node per frame; key = global frame index via gtsam.symbol('x', node_id).
    Ported and adapted from MIT-SPARK/VGGT-SLAM vggt_slam/graph.py.
    Noise model matches VGGT-SLAM exactly: σ=0.05 Gaussian for all edges.
    """

    def __init__(self, manifold: Literal["sl4", "se3"] = "sl4") -> None:
        self._manifold = manifold
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._node_ids: set[int] = set()

        if manifold == "sl4":
            # Uniform σ=0.05 for all edges — matches VGGT-SLAM exactly
            self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(
                0.05 * np.ones(15, dtype=np.float64)
            )
            self._loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
                0.05 * np.ones(15, dtype=np.float64)
            )
            self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.full(15, 1e-6, dtype=np.float64)
            )
        else:
            # SE(3) 6-DOF — existing tuned sigmas from ADR 005
            self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.05, 0.05, 0.05, 0.20, 0.20, 0.20], dtype=np.float64)
            )
            self._loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.05, 0.05, 0.05, 0.20, 0.20, 0.20], dtype=np.float64)
            )
            self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.full(6, 1e-6, dtype=np.float64)
            )

    # ---- node management ----

    def add_node(self, node_id: int, H: np.ndarray) -> None:
        """Insert per-frame node. H is 4×4; GTSAM SL4 normalizes internally."""
        if node_id in self._node_ids:
            return
        key = _X(node_id)
        H = np.array(H, dtype=np.float64)
        if self._manifold == "sl4":
            self._initial.insert(key, gtsam.SL4(H))
        else:
            self._initial.insert(key, _pose3(H))
        self._node_ids.add(node_id)

    def add_prior(self, node_id: int, H: np.ndarray) -> None:
        """Tight prior anchoring the first frame (σ=1e-6)."""
        key = _X(node_id)
        H = np.array(H, dtype=np.float64)
        if self._manifold == "sl4":
            self._graph.add(
                gtsam.PriorFactorSL4(key, gtsam.SL4(H), self._anchor_noise)
            )
        else:
            self._graph.add(
                gtsam.PriorFactorPose3(key, _pose3(H), self._anchor_noise)
            )

    # ---- edges ----

    def add_sequential_edge(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """Sequential frame-to-frame odometry constraint."""
        key_i, key_j = _X(id_i), _X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        if self._manifold == "sl4":
            self._graph.add(
                gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), self._seq_noise)
            )
        else:
            self._graph.add(
                gtsam.BetweenFactorPose3(key_i, key_j, _pose3(H_rel), self._seq_noise)
            )

    def add_loop_edge(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """Loop closure constraint — same Gaussian noise as sequential edges (VGGT-SLAM parity)."""
        key_i, key_j = _X(id_i), _X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        if self._manifold == "sl4":
            H_rel = normalize_to_sl4(H_rel)
            self._graph.add(gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), self._loop_noise))
        else:
            self._graph.add(gtsam.BetweenFactorPose3(key_i, key_j, _pose3(H_rel), self._loop_noise))

    # ---- optimization ----

    def optimize(self) -> None:
        """Levenberg-Marquardt in-place. Updates internal values; use get_homography after.

        On GTSAM failure, initial values are preserved unchanged.
        """
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
        key = _X(node_id)
        if self._manifold == "sl4":
            return self._initial.atSL4(key).matrix().astype(np.float64)
        else:
            return self._initial.atPose3(key).matrix().astype(np.float64)
