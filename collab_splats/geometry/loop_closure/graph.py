"""Factor graph over camera projection matrices.

Optimizes camera poses on the SL(4) manifold to correct trajectory drift and
close loops. Per-frame nodes — one SL(4) node per frame, key = global frame
index. Ported and adapted from MIT-SPARK/VGGT-SLAM vggt_slam/graph.py (SL4
backend) and vggt_slam/slam_utils.py (decompose_camera) and
vggt_slam/scale_solver.py (estimate_scale_pairwise).
"""

from __future__ import annotations

import logging

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X as _X
from scipy.linalg import rq

from .submap import Submap

########################################
########## Module-level helpers ########
########################################


def dedup_overlap(
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct (total_frames, 4, 4) from per-submap corrected poses, deduplicating overlap.

    We connect submaps via an overlapping frame (e.g., if submap-0's last frame = 16, then
    submap-1's first frame = 16). This overlap allows us to compute a relative transform
    between the two submaps, but we want to eliminate this duplicated frame in the final output.
    Therefore, we assign the overlapping frame to the first submap (submap-0) and ignore that frame
    in the second submap (submap-1).

    VGGT-SLAM divergence: VGGT-SLAM (https://github.com/MIT-SPARK/VGGT-SLAM) does not perform
    this deduplication -- every frame is written to a file via write_poses_to_file() (map.py:142-162).
    The overlap frame therefore is written twice with the same timestamp. Evo
    (https://github.com/MichaelGrupp/evo) associates by timestamp and keeps the first occurrence
    (from submap-0), so the two pipelines agree on the boundary pose despite SLAM's duplicate row.
    """
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


def _resolve_frame_node(
    frame_to_node: dict[tuple[int, int], int],
    submaps: list[Submap],
    image_path,
) -> tuple[int, Submap, int] | None:
    """Resolve an image path to (node_id, submap, local frame index)."""
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                nid = frame_to_node.get((submap.submap_id, local_i))
                if nid is not None:
                    return nid, submap, local_i
    return None


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


class PoseGraph:
    """Per-frame SL(4) factor graph for loop closure trajectory correction.

    One node per frame; key = global frame index via gtsam.symbol('x', node_id).
    Ported and adapted from MIT-SPARK/VGGT-SLAM vggt_slam/graph.py.
    Noise model matches VGGT-SLAM exactly: σ=0.05 Gaussian for all edges.
    """

    def __init__(self) -> None:
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._node_ids: set[int] = set()

        # Uniform σ=0.05 for all edges (sequential, inter-submap, loop chain)
        # — matches VGGT-SLAM exactly
        self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(0.05 * np.ones(15, dtype=np.float64))
        self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(15, 1e-6, dtype=np.float64))

        # Incremental submap-build bookkeeping (per-submap cadence state)
        self._global_node_id = 0
        self._frame_to_node: dict[tuple[int, int], int] = {}
        self._submap_node_ids: dict[int, list[int]] = {}
        self._submaps_seen: list = []  # prior submaps, for inter-submap overlap lookups

    # ---- node management ----

    def add_homography(self, node_id: int, H: np.ndarray) -> None:
        """Insert per-frame node. H is 4×4; GTSAM SL4 normalizes internally."""
        if node_id in self._node_ids:
            return
        key = _X(node_id)
        H = np.array(H, dtype=np.float64)
        self._initial.insert(key, gtsam.SL4(H))
        self._node_ids.add(node_id)

    def add_prior_factor(self, node_id: int, H: np.ndarray) -> None:
        """Tight prior anchoring the first frame (σ=1e-6)."""
        key = _X(node_id)
        H = np.array(H, dtype=np.float64)
        self._graph.add(gtsam.PriorFactorSL4(key, gtsam.SL4(H), self._anchor_noise))

    # ---- edges ----

    def add_between_factor(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """Sequential frame-to-frame odometry constraint."""
        key_i, key_j = _X(id_i), _X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        self._graph.add(gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), self._seq_noise))

    # ---- optimization ----

    def optimize(self) -> None:
        """Levenberg-Marquardt in-place. Updates internal values; use get_homography after.

        On GTSAM failure, initial values are preserved unchanged.
        """
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial, params)
            self._initial = optimizer.optimize()
        except Exception as exc:
            log.warning("GTSAM optimization failed: %s — returning initial values", exc)

    def get_homography(self, node_id: int) -> np.ndarray:
        """Return 4×4 H for a frame node. Call after optimize()."""
        key = _X(node_id)
        return self._initial.atSL4(key).matrix().astype(np.float64)

    # ---- incremental submap build ----

    def add_submap(
        self,
        submap,
        overlap_frames,
        conf_threshold=25.0,
        scale_method="rotation_only",
        debug_out=None,
    ):
        """Add one submap's nodes + sequential SL(4) edges to the persistent graph."""
        k = submap.poses.shape[0]
        node_ids_this: list[int] = []

        for local_i in range(k):
            nid = self._global_node_id + local_i
            self._frame_to_node[(submap.submap_id, local_i)] = nid
            node_ids_this.append(nid)

        # Build K_4x4 from 3×3 intrinsics — matches VGGT-SLAM's proj_mats = K_4x4.
        # SLAM uses K_4x4 for T computation (inv(K_prev)@K_curr = I for same camera)
        # and for pose extraction (K @ inv(H_opt) → decompose_camera cancels K).
        K_4x4 = np.tile(np.eye(4, dtype=np.float64), (k, 1, 1))
        K_4x4[:, :3, :3] = submap.intrinsics.astype(np.float64)

        if not self._submaps_seen:
            # First node = I, prior = I — matches VGGT-SLAM add_homography(0, I) + add_prior_factor(0, I)
            self.add_homography(node_ids_this[0], np.eye(4))
            self.add_prior_factor(node_ids_this[0], np.eye(4))
            for local_i in range(1, k):
                H_inner = submap.poses[local_i - 1].astype(np.float64) @ np.linalg.inv(
                    submap.poses[local_i].astype(np.float64)
                )
                prev_H = self.get_homography(node_ids_this[local_i - 1])
                self.add_homography(node_ids_this[local_i], prev_H @ H_inner)
                self.add_between_factor(node_ids_this[local_i - 1], node_ids_this[local_i], H_inner)
        else:
            prev_submap = self._submaps_seen[-1]
            prev_K_4x4 = np.tile(np.eye(4, dtype=np.float64), (len(prev_submap.poses), 1, 1))
            prev_K_4x4[:, :3, :3] = prev_submap.intrinsics.astype(np.float64)
            O = min(overlap_frames, k, len(prev_submap.poses))

            # T = inv(K_prev[-1]) @ K_curr[0] — same as VGGT-SLAM's proj_mats-based T.
            # For fixed camera (K_prev = K_curr) this equals I exactly.
            T = np.linalg.inv(prev_K_4x4[-1]) @ K_4x4[0]

            scale = 1.0
            if submap.world_points is not None and prev_submap.world_points is not None and O > 0:
                # curr_pts: world_points[0] = cam_pts[0] (W2C[0]=I for VGGT) — same as SLAM t1.
                curr_pts = submap.world_points[:O].reshape(-1, 3).astype(np.float64)

                # prev_pts: SLAM uses camera-local depth (pointclouds[K-1], camera frame of last frame).
                # Our world_points[-1] = C2W[-1] @ cam_pts[-1] (cam0-frame, not camera-local).
                # Back-transform: W2C[-1] @ world_pts[-1] = cam_pts[-1] (camera-local), matching SLAM.
                prev_wps = prev_submap.world_points[-O:]  # (O, P, 3)
                P_per = prev_wps.shape[1]
                prev_cam_list = []
                for oi in range(O):
                    W2C = prev_submap.poses[-O + oi].astype(np.float64)
                    wh = np.hstack([prev_wps[oi], np.ones((P_per, 1), dtype=np.float64)])
                    prev_cam_list.append((W2C @ wh.T).T[:, :3])
                prev_pts = np.concatenate(prev_cam_list, axis=0)

                n = curr_pts.shape[0]

                # Confidence filtering: match VGGT-SLAM solver.py:132-143
                mask = np.ones(n, dtype=bool)
                if (
                    submap.world_points_conf is not None
                    and prev_submap.world_points_conf is not None
                    and submap.world_points_conf.shape[:2] == submap.world_points.shape[:2]
                    and prev_submap.world_points_conf.shape[:2] == prev_submap.world_points.shape[:2]
                ):
                    curr_conf = submap.world_points_conf[:O].reshape(-1)
                    prev_conf = prev_submap.world_points_conf[-O:].reshape(-1)
                    joint_mask = (curr_conf > conf_threshold) & (prev_conf > conf_threshold)
                    if joint_mask.sum() >= _MIN_CONF_POINTS:
                        mask = joint_mask
                    else:
                        # VGGT-SLAM fallback: prior_conf > thresh only (not OR mask).
                        prior_only_mask = prev_conf > conf_threshold
                        if prior_only_mask.sum() >= _MIN_CONF_POINTS:
                            mask = prior_only_mask

                curr_h = np.hstack([curr_pts, np.ones((n, 1))])
                if scale_method == "none":
                    # No scale estimation — use identity scale (scale=1.0).
                    scale = 1.0
                else:
                    if scale_method == "rotation_only":
                        # VGGT-SLAM style: apply only rotation part of T, drop translation.
                        # norm(R@X) == norm(X), so effectively uses world-frame norms.
                        curr_in_prev = (T[:3, :3] @ curr_pts.T).T
                    else:
                        # "se3" (default): full SE3 — translation shifts anchor, introduces bias
                        curr_in_prev = (T @ curr_h.T).T[:, :3]

                    if scale_method == "pairwise_dist":
                        scale = _estimate_scale_pairwise_dist(curr_in_prev[mask], prev_pts[mask])
                    else:
                        scale = estimate_scale_pairwise(curr_in_prev[mask], prev_pts[mask])

            # H_w = H_overlap @ T @ H_scale — matches VGGT-SLAM solver.py:161-162
            # (H_overlap_prev_node @ inv(K_prev) @ K_curr @ H_scale). Note SLAM's
            # submap.proj_mats stores K (the param is misleadingly named
            # intrinsics_inv but solver.py:256 passes K_4x4), so its
            # inv(proj_mats[-1]) @ proj_mats[0] == our inv(K_prev) @ K_curr == T.
            H_scale = np.diag([scale, scale, scale, 1.0])
            prev_submap_last_nid = self._submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = self.get_homography(prev_submap_last_nid)
            H_w = H_overlap @ T @ H_scale
            self.add_homography(node_ids_this[0], H_w)

            if debug_out is not None:
                debug_out.append(
                    {
                        "submap_id": submap.submap_id,
                        "T": T.copy(),
                        "scale": float(scale),
                        "H_w": H_w.copy(),
                        "H_overlap": H_overlap.copy(),
                    }
                )

            H_rel_inter = np.linalg.inv(self.get_homography(prev_submap_last_nid)) @ H_w
            self.add_between_factor(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

            for local_i in range(1, k):
                H_inner = submap.poses[local_i - 1].astype(np.float64) @ np.linalg.inv(
                    submap.poses[local_i].astype(np.float64)
                )
                prev_H = self.get_homography(node_ids_this[local_i - 1])
                self.add_homography(node_ids_this[local_i], prev_H @ H_inner)
                self.add_between_factor(node_ids_this[local_i - 1], node_ids_this[local_i], H_inner)

        self._submap_node_ids[submap.submap_id] = node_ids_this
        self._global_node_id += k
        self._submaps_seen.append(submap)

    def add_loop_edge(self, lc, self_submaps, conf_threshold=25.0, scale_method="rotation_only"):
        """Add one verified loop closure's 3-edge SL(4) chain (monolith lines ~581-624)."""
        # Only 2-frame LC submaps (query, detected) carry a valid loop constraint.
        if lc.poses.shape[0] != 2:
            return
        path_q, path_d = lc.image_paths[0], lc.image_paths[1]
        loc_q = _resolve_frame_node(self._frame_to_node, self_submaps, path_q)
        loc_d = _resolve_frame_node(self._frame_to_node, self_submaps, path_d)
        if loc_q is None or loc_d is None:
            return
        nid_q, sub_q, qi = loc_q
        nid_d, sub_d, di = loc_d

        # Anchor scales on pixel-aligned identical images; fall back to 1.0 when
        # the LC backend supplies poses only (direction fix still applies).
        s_a = _lc_anchor_scale(lc, 0, sub_q, qi, conf_threshold, scale_method)
        s_b = _lc_anchor_scale(sub_d, di, lc, 1, conf_threshold, scale_method)
        if s_a is None or s_b is None:
            log.warning(
                "Loop submap %d → %d: LC world points unavailable or grid-misaligned; " "using anchor scale 1.0",
                sub_q.submap_id,
                sub_d.submap_id,
            )
            s_a = 1.0 if s_a is None else s_a
            s_b = 1.0 if s_b is None else s_b

        # Per-frame intrinsics as 4×4 for the K change across anchors (I for a shared camera)
        K_q, K_lc0, K_lc1, K_d = (np.eye(4, dtype=np.float64) for _ in range(4))
        K_q[:3, :3] = sub_q.intrinsics[qi].astype(np.float64)
        K_lc0[:3, :3] = lc.intrinsics[0].astype(np.float64)
        K_lc1[:3, :3] = lc.intrinsics[1].astype(np.float64)
        K_d[:3, :3] = sub_d.intrinsics[di].astype(np.float64)

        H_rel_a, H_inner_lc, H_rel_b = _loop_chain_relatives(lc.poses[0], lc.poses[1], s_a, s_b, K_q, K_lc0, K_lc1, K_d)

        # LC nodes chained from the query node's current graph state
        # (upstream solver.py:162-166); they map to no output frame.
        nid_lc0, nid_lc1 = self._global_node_id, self._global_node_id + 1
        self._global_node_id += 2
        H_q_state = self.get_homography(nid_q)
        self.add_homography(nid_lc0, H_q_state @ H_rel_a)
        self.add_homography(nid_lc1, H_q_state @ H_rel_a @ H_inner_lc)

        self.add_between_factor(nid_q, nid_lc0, H_rel_a)
        self.add_between_factor(nid_lc0, nid_lc1, H_inner_lc)
        self.add_between_factor(nid_lc1, nid_d, H_rel_b)

    def extract_extrinsics(self, total_frames: int) -> np.ndarray:
        """Turn optimized homographies into a (total_frames, 4, 4) world-to-cam array."""
        if not self._submaps_seen:
            return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))

        corrected_per_submap: dict[int, np.ndarray] = {}
        for submap in self._submaps_seen:
            node_ids = self._submap_node_ids[submap.submap_id]
            k = len(node_ids)
            poses_out = np.zeros((k, 4, 4), dtype=np.float32)
            # Extraction via K_4x4 — matches VGGT-SLAM: proj_mats[i] @ inv(H_opt[i]).
            # decompose_camera cancels the K factor and returns correct SE3 R, t.
            s_K = np.tile(np.eye(4, dtype=np.float64), (k, 1, 1))
            s_K[:, :3, :3] = submap.intrinsics.astype(np.float64)
            for local_i, nid in enumerate(node_ids):
                H_opt = self.get_homography(nid)
                local_proj = s_K[local_i]  # camera intrinsics as 4×4
                corrected = local_proj @ np.linalg.inv(H_opt)
                _, R, t, _ = decompose_camera(corrected)
                # decompose_camera returns R as the camera-to-world rotation and
                # t = inv(K) @ P[:,3] (VGGT-SLAM's no_inverse=True values). Store R^T
                # so downstream -R_stored^T @ t recovers the camera centre.
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = R.T.astype(np.float32)
                mat[:3, 3] = t.astype(np.float32)
                poses_out[local_i] = mat
            corrected_per_submap[submap.submap_id] = poses_out

        return dedup_overlap(
            submap_ids=[s.submap_id for s in self._submaps_seen],
            submap_starts=[s.frame_start for s in self._submaps_seen],
            corrected=corrected_per_submap,
            total_frames=total_frames,
        )


########################################
####### Scale estimation & alignment ###
########################################

# Minimum points required for a reliable median scale estimate (matches VGGT-SLAM fallback threshold)
_MIN_CONF_POINTS = 100


def _estimate_scale_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> float:
    """Pairwise distance scale estimator — invariant to coordinate origin.

    median(||Y_i - Y_j|| / ||X_i - X_j||). Translation cancels in subtraction,
    so result is unbiased regardless of which camera frame X/Y are expressed in.
    """
    if X.shape[0] < 2:
        return 1.0
    # Per-call seeded RNG — pairwise-dist scale must be deterministic and independent of prior draws (call-order bug).
    rng = np.random.default_rng(0)
    n = min(X.shape[0], 500)
    idx = rng.choice(X.shape[0], (n, 2), replace=True)
    i, j = idx[:, 0], idx[:, 1]
    x_dists = np.linalg.norm(X[i] - X[j], axis=1)
    y_dists = np.linalg.norm(Y[i] - Y[j], axis=1)
    valid = x_dists > 1e-6
    return float(np.median(y_dists[valid] / x_dists[valid])) if valid.any() else 1.0


def umeyama_se3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form SE(3) alignment via SVD (no scale).

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (4, 4) float32 homogeneous T such that target ≈ T @ source.
    """
    M = source.shape[0]
    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return np.eye(4, dtype=np.float32)
    w = w / w_sum
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt
    H = (src_c * w[:, None]).T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_tgt - R @ mu_src
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment via Umeyama (with scale).

    Args:
        source: (M, 3) float32/64 points in source frame.
        target: (M, 3) float32/64 corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (s, R, t): float scale, (3,3) float32 rotation, (3,) float32 translation
                   such that target ≈ s * R @ source + t.
    """
    M = source.shape[0]
    if M < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    w = w / w_sum

    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    scale_src = float(np.sqrt((w * (src_c**2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c**2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t


########################################
####### SL(4) pose graph optimization ##
########################################


def _cam_local_points(pts: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points by a 4x4 world-to-cam pose into camera-local coords."""
    h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1))])
    return (w2c.astype(np.float64) @ h.T).T[:, :3]


def _lc_anchor_scale(
    curr_submap: Submap,
    curr_idx: int,
    prior_submap: Submap,
    prior_idx: int,
    conf_threshold: float,
    scale_method: str,
) -> float | None:
    """Pixel-aligned anchor scale between an LC frame and its identical regular frame.

    Mirrors VGGT-SLAM solver.py:129-151: both frames are the SAME image, so their
    point grids pair pixel-for-pixel. Returns median(||prior_cam|| / ||curr_cam||)
    per `scale_method` (the factor scaling curr-submap units into prior-submap
    units), or None when points are unavailable or the grids cannot be aligned.
    """
    if scale_method == "none":
        return 1.0
    if curr_submap.world_points is None or prior_submap.world_points is None:
        return None

    # Resolution alignment: LC submaps carry full-res (H·W) grids while regular
    # submaps carry subsample-strided grids (arange(0, W, 8) × arange(0, H, 8),
    # row-major over (v, u) — see _raw_to_world_points). Resample the LC side
    # onto the regular grid so points pair pixel-for-pixel.
    lc_side, reg_side = (curr_submap, prior_submap) if curr_submap.is_lc_submap else (prior_submap, curr_submap)
    lc_flat_idx = None
    if lc_side.world_points.shape[1] != reg_side.world_points.shape[1]:
        if lc_side.frames is None:
            return None
        h_img, w_img = int(lc_side.frames.shape[-2]), int(lc_side.frames.shape[-1])
        if lc_side.world_points.shape[1] != h_img * w_img:
            return None
        us = np.arange(0, w_img, 8)  # regular submaps use subsample stride 8 (_raw_to_world_points)
        vs = np.arange(0, h_img, 8)
        if reg_side.world_points.shape[1] != len(us) * len(vs):
            return None
        uu, vv = np.meshgrid(us, vs)
        lc_flat_idx = (vv * w_img + uu).ravel()

    # Gather paired points + confs, resampling the LC side where needed
    def _frame_data(submap: Submap, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        pts = submap.world_points[idx].astype(np.float64)
        conf = None
        if submap.world_points_conf is not None and submap.world_points_conf.shape[:2] == submap.world_points.shape[:2]:
            conf = submap.world_points_conf[idx].astype(np.float64)
        if submap is lc_side and lc_flat_idx is not None:
            pts = pts[lc_flat_idx]
            conf = conf[lc_flat_idx] if conf is not None else None
        return pts, conf

    curr_pts, curr_conf = _frame_data(curr_submap, curr_idx)
    prior_pts, prior_conf = _frame_data(prior_submap, prior_idx)

    # Back-transform both sides to their frame's camera-local coords so norms share
    # an origin (same convention as the sequential-edge prev_pts back-transform).
    curr_cam = _cam_local_points(curr_pts, curr_submap.poses[curr_idx])
    prior_cam = _cam_local_points(prior_pts, prior_submap.poses[prior_idx])

    # Confidence fallback chain (VGGT-SLAM solver.py:132-143): joint > thr,
    # else prior > thr, else prior > 0; missing conf side treated as pass-all.
    n = curr_cam.shape[0]
    mask = np.ones(n, dtype=bool)
    if prior_conf is not None:
        joint = prior_conf > conf_threshold
        if curr_conf is not None:
            joint = joint & (curr_conf > conf_threshold)
        if joint.sum() >= _MIN_CONF_POINTS:
            mask = joint
        elif (prior_conf > conf_threshold).sum() >= _MIN_CONF_POINTS:
            mask = prior_conf > conf_threshold
        else:
            mask = prior_conf > 0

    # T = inv(K_prior) @ K_curr (identity for a shared camera) — mirror the
    # sequential-edge scale_method conventions.
    T = np.eye(4)
    T[:3, :3] = np.linalg.inv(prior_submap.intrinsics[prior_idx].astype(np.float64)) @ curr_submap.intrinsics[
        curr_idx
    ].astype(np.float64)
    if scale_method == "rotation_only":
        curr_in_prior = (T[:3, :3] @ curr_cam.T).T
    else:
        curr_in_prior = _cam_local_points(curr_cam, T)

    if scale_method == "pairwise_dist":
        s = _estimate_scale_pairwise_dist(curr_in_prior[mask], prior_cam[mask])
    else:
        s = estimate_scale_pairwise(curr_in_prior[mask], prior_cam[mask])

    # Guard against degenerate estimates: depth-unprojected LC points can contain
    # NaN/inf (invalid pixels), letting a non-finite or ≤0 median through — which
    # would produce a singular diag(s,s,s,1) between-factor. Fall back to the
    # None path (scale 1.0 + once-per-loop warning at the call site).
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def _loop_chain_relatives(
    P_lc0: np.ndarray,
    P_lc1: np.ndarray,
    s_a: float,
    s_b: float,
    K_q: np.ndarray | None = None,
    K_lc0: np.ndarray | None = None,
    K_lc1: np.ndarray | None = None,
    K_d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three loop-chain relatives in the graph's H_inner convention (H_j = H_i @ M).

    Anchor A (query → LC0, identical image): pure scale fold s_a (LC units →
    query-submap units) plus the K change; anchor B (LC1 → detected) symmetric
    with s_b (detected units → LC units); inner edge carries the LC relative
    P_lc0 @ inv(P_lc1). Identity anchors compose to P_lc0 @ inv(P_lc1), and for
    an LC run at k× scale s_a=1/k, s_b=k cancel to the metric relative
    (VGGT-SLAM solver.py:118-170 chain).
    """
    eye = np.eye(4, dtype=np.float64)
    K_q = eye if K_q is None else K_q
    K_lc0 = eye if K_lc0 is None else K_lc0
    K_lc1 = eye if K_lc1 is None else K_lc1
    K_d = eye if K_d is None else K_d
    # Anchor A: query → LC-frame-0 (identical image) — K change + scale fold s_a.
    H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])
    # Inner edge: the LC pair's own relative pose, no K change.
    H_inner = P_lc0.astype(np.float64) @ np.linalg.inv(P_lc1.astype(np.float64))
    # Anchor B: LC-frame-1 → detected (identical image) — K change + scale fold s_b.
    H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])
    return H_rel_a, H_inner, H_rel_b
