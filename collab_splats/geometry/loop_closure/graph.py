"""
SL(4) factor graph over per-frame camera projection matrices, for loop closure.

- optimizes poses on the SL(4) manifold to correct drift and close loops
- one node per frame, keyed by global frame index
- ported from MIT-SPARK/VGGT-SLAM: vggt_slam/graph.py (SL4 backend),
  vggt_slam/slam_utils.py (decompose_camera), vggt_slam/scale_solver.py (estimate_scale_pairwise)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X as _X
from scipy.linalg import rq

from .submap import Submap

if TYPE_CHECKING:
    from pathlib import Path

# inter-submap scale estimators; "none" pins scale to 1.0
SCALE_METHODS = ("rotation_only", "none")

########################################
########## Module-level helpers ########
########################################


def check_scale_method(scale_method: str) -> None:
    """
    Reject a scale_method outside SCALE_METHODS.

    - retired "se3" and "pairwise_dist" fail loudly instead of running rotation_only

    Args:
        scale_method: the requested inter-submap scale estimator.

    Raises:
        ValueError: `scale_method` is not in SCALE_METHODS.
    """
    if scale_method not in SCALE_METHODS:
        raise ValueError(f"scale_method must be one of {SCALE_METHODS}, got {scale_method!r}")


def dedup_overlap(
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """
    Per-frame poses from per-submap corrected poses, each overlap frame kept once.

    - adjacent submaps share an overlap frame; the earlier submap's pose wins
    - frames no submap covers stay identity
    - VGGT-SLAM writes the overlap frame twice (map.py:142-162); evo (MichaelGrupp/evo) keeps the first
    - keeping the first makes both pipelines agree on the boundary pose

    Args:
        submap_ids: submap ids, in trajectory order.
        submap_starts: global frame index of each submap's first frame.
        corrected: per-submap (K_i, 4, 4) corrected poses, keyed by submap id.
        total_frames: output length.

    Returns:
        (total_frames, 4, 4) float32 poses.
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
    image_path: Path,
) -> tuple[int, Submap, int] | None:
    """
    Resolve an image path to (node_id, submap, local frame index).

    Args:
        frame_to_node: (submap_id, local index) -> graph node id.
        submaps: submaps to search.
        image_path: the path to find.

    Returns:
        (node_id, submap, local index), or None when no node holds the path.
    """
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                nid = frame_to_node.get((submap.submap_id, local_i))
                if nid is not None:
                    return nid, submap, local_i
    return None


def decompose_camera(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    RQ-decompose a 3x4 or 4x4 projection matrix into (K, R, t, scale).

    - port of MIT-SPARK/VGGT-SLAM vggt_slam/slam_utils.py:decompose_camera
    - matches VGGT-SLAM's ``no_inverse=True`` branch: R is camera-to-world, t = inv(K) @ P[:, 3]
    - t = inv(K) @ P[:, 3] is the world-to-cam translation, paired with R^T, not R
    - camera center C = -R @ t
    - storing R directly agrees only near identity rotation and bends multi-submap trajectories

    Args:
        P: 3x4 projection matrix, or 4x4 (divided by P[-1, -1], last row dropped).

    Returns:
        (K with K[2, 2] = 1, camera-to-world R, t, scale = K[2, 2] before normalizing).

    Raises:
        ValueError: P is not (3, 4) after the 4x4 strip.
    """
    P = np.array(P, dtype=np.float64)
    if P.shape[0] != 3:
        P = P / P[-1, -1]
        P = P[:3, :]
    if P.shape != (3, 4):
        raise ValueError(f"expected (3,4) after strip, got {P.shape}")

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
    # t = inv(K) @ P[:,3], VGGT-SLAM's no_inverse=True value
    # - the world-to-cam translation, paired with R^T; camera center C = -R @ t
    t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale
    return K, R, t, scale


def estimate_scale_pairwise(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Scale between two paired point clouds: median(||Y[i]|| / ||X[i]||).

    - initializes the inter-submap SL(4) edge with the right scale
    - port of MIT-SPARK/VGGT-SLAM vggt_slam/scale_solver.py:estimate_scale_pairwise
    - points with ||X[i]|| <= 1e-8 are skipped; 1.0 when none remain

    Args:
        X: (N, 3) source points.
        Y: (N, 3) target points, paired with X.

    Returns:
        Scale taking X's units to Y's.

    Raises:
        ValueError: X and Y differ in shape.
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
    """
    Per-frame SL(4) factor graph for loop-closure trajectory correction.

    - one node per frame, keyed gtsam.symbol('x', node_id) by global frame index
    - port of MIT-SPARK/VGGT-SLAM vggt_slam/graph.py
    - noise matches VGGT-SLAM: sigma 0.05 on every edge, 1e-6 on the first-frame prior

    Args:
        min_conf_points: fewest confident points before the scale fit uses the confidence mask.
    """

    def __init__(self, min_conf_points: int = 100) -> None:
        self.min_conf_points = min_conf_points
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
        """
        Insert a frame node; a node id already present is left unchanged.

        Args:
            node_id: global frame node id.
            H: 4x4 homography; GTSAM's SL4 normalizes it.
        """
        if node_id in self._node_ids:
            return
        key = _X(node_id)
        H = np.array(H, dtype=np.float64)
        self._initial.insert(key, gtsam.SL4(H))
        self._node_ids.add(node_id)

    def add_prior_factor(self, node_id: int, H: np.ndarray) -> None:
        """
        Tight prior (sigma 1e-6) holding a node in place, used on the first frame.

        Args:
            node_id: node to anchor.
            H: 4x4 homography the node is held to.
        """
        key = _X(node_id)
        H = np.array(H, dtype=np.float64)
        self._graph.add(gtsam.PriorFactorSL4(key, gtsam.SL4(H), self._anchor_noise))

    # ---- edges ----

    def add_between_factor(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """
        Relative SL(4) constraint between two nodes: H_j = H_i @ H_rel.

        Args:
            id_i: source node id.
            id_j: target node id.
            H_rel: 4x4 relative homography.
        """
        key_i, key_j = _X(id_i), _X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        self._graph.add(gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), self._seq_noise))

    # ---- optimization ----

    def optimize(self) -> None:
        """
        Run Levenberg-Marquardt in place; read results with get_homography.

        - on a GTSAM RuntimeError the initial values are kept and a warning is logged
        """
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial, params)
            self._initial = optimizer.optimize()
        except RuntimeError as exc:
            log.warning("GTSAM optimization failed: %s — returning initial values", exc)

    def get_homography(self, node_id: int) -> np.ndarray:
        """
        4x4 homography of a frame node; the optimized value once optimize() has run.

        Args:
            node_id: node to read.

        Returns:
            (4, 4) float64 homography.
        """
        key = _X(node_id)
        return self._initial.atSL4(key).matrix().astype(np.float64)

    # ---- incremental submap build ----

    def add_submap(
        self,
        submap: Submap,
        overlap_frames: int,
        conf_threshold: float = 25.0,
        scale_method: str = "rotation_only",
        debug_out: list[dict] | None = None,
    ) -> None:
        """
        Add one submap's nodes and SL(4) edges to the persistent graph.

        - the first submap is anchored at identity with a tight prior
        - later submaps link to the previous submap's last node through the overlap frames
        - the inter-submap scale comes from the overlap points, per `scale_method`

        Args:
            submap: the next Submap in trajectory order.
            overlap_frames: frames shared with the previous submap.
            conf_threshold: point-confidence floor for the scale estimate.
            scale_method: "rotation_only" or "none" (scale 1.0).
            debug_out: list that receives one dict (submap_id, T, scale, H_w, H_overlap) per
                non-first submap, or None.

        Raises:
            ValueError: `scale_method` is not "rotation_only" or "none".
        """
        check_scale_method(scale_method)

        k = submap.poses.shape[0]
        node_ids_this: list[int] = []

        for local_i in range(k):
            nid = self._global_node_id + local_i
            self._frame_to_node[(submap.submap_id, local_i)] = nid
            node_ids_this.append(nid)

        # K_4x4 from 3x3 intrinsics, as VGGT-SLAM's proj_mats
        # - T: inv(K_prev) @ K_curr = I for the same camera
        # - extraction: K @ inv(H_opt), and decompose_camera cancels K
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

                # prev_pts: camera-local points of the previous submap's overlap frames
                # - VGGT-SLAM uses camera-local depth (pointclouds[K-1])
                # - ours are cam0-frame: W2C @ world_pts recovers cam_pts, matching SLAM
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
                    if joint_mask.sum() >= self.min_conf_points:
                        mask = joint_mask
                    else:
                        # VGGT-SLAM fallback: prior_conf > thresh only (not OR mask).
                        prior_only_mask = prev_conf > conf_threshold
                        if prior_only_mask.sum() >= self.min_conf_points:
                            mask = prior_only_mask

                # rotation_only, as VGGT-SLAM: apply the 3x3 block of T, inv(K_prev) @ K_curr
                # - T has zero translation (block-diagonal K), so the 3x3 block is all of T
                # - norm-preserving only for a shared camera (block is I), not in general
                # - "none" keeps scale 1.0
                if scale_method == "rotation_only":
                    curr_in_prev = (T[:3, :3] @ curr_pts.T).T
                    scale = estimate_scale_pairwise(curr_in_prev[mask], prev_pts[mask])

            # H_w = H_overlap @ T @ H_scale, as VGGT-SLAM solver.py:161-162
            # - SLAM: H_overlap_prev_node @ inv(K_prev) @ K_curr @ H_scale
            # - SLAM's proj_mats holds K (param named intrinsics_inv; solver.py:256 passes K_4x4)
            # - so its inv(proj_mats[-1]) @ proj_mats[0] equals our T
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

    def add_loop_edge(
        self,
        lc: Submap,
        self_submaps: list[Submap],
        conf_threshold: float = 25.0,
        scale_method: str = "rotation_only",
    ) -> None:
        """
        Add one verified loop closure as a 3-edge SL(4) chain through two LC nodes.

        - chain: query -> LC frame 0 -> LC frame 1 -> detected; LC nodes map to no output frame
        - skipped unless `lc` holds exactly 2 frames that both resolve to graph nodes
        - anchor scale falls back to 1.0, with a warning, when LC points are unusable

        Args:
            lc: 2-frame loop-closure Submap (query, detected).
            self_submaps: the regular submaps already in the graph.
            conf_threshold: point-confidence floor for the anchor scales.
            scale_method: "rotation_only" or "none" (anchor scales 1.0).

        Raises:
            ValueError: `scale_method` is not "rotation_only" or "none".
        """
        check_scale_method(scale_method)

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
        s_a = _lc_anchor_scale(lc, 0, sub_q, qi, conf_threshold, scale_method, self.min_conf_points)
        s_b = _lc_anchor_scale(sub_d, di, lc, 1, conf_threshold, scale_method, self.min_conf_points)
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
        """
        Optimized homographies as a (total_frames, 4, 4) world-to-cam array.

        - each overlap frame comes from the earlier submap (dedup_overlap)
        - identity for every frame when no submap was added

        Args:
            total_frames: output length.

        Returns:
            (total_frames, 4, 4) float32 world-to-cam poses.
        """
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
                # Store R^T: decompose_camera's R is camera-to-world
                # - t = inv(K) @ P[:, 3], VGGT-SLAM's no_inverse=True value
                # - downstream -R_stored^T @ t then recovers the camera center
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
########## Loop-edge helpers ###########
########################################


def _cam_local_points(pts: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """
    Move (N, 3) points by a 4x4 world-to-cam pose into camera-local coords.

    Args:
        pts: (N, 3) points.
        w2c: 4x4 world-to-cam pose.

    Returns:
        (N, 3) float64 camera-local points.
    """
    h = np.hstack([pts.astype(np.float64), np.ones((pts.shape[0], 1))])
    return (w2c.astype(np.float64) @ h.T).T[:, :3]


def _lc_anchor_scale(
    curr_submap: Submap,
    curr_idx: int,
    prior_submap: Submap,
    prior_idx: int,
    conf_threshold: float,
    scale_method: str,
    min_conf_points: int,
) -> float | None:
    """
    Pixel-aligned anchor scale between an LC frame and its identical regular frame.

    - mirrors VGGT-SLAM solver.py:129-151: same image, so point grids pair pixel-for-pixel
    - median(||prior_cam|| / ||curr_cam||) per `scale_method`: curr units into prior units

    Args:
        curr_submap: submap whose units are scaled.
        curr_idx: frame index in curr_submap.
        prior_submap: submap whose units are the target.
        prior_idx: frame index in prior_submap.
        conf_threshold: point-confidence floor.
        scale_method: "rotation_only" or "none" (always 1.0); validated by the caller.
        min_conf_points: fewest confident points before a confidence mask is used.

    Returns:
        The scale, or None when points are missing, grids cannot be aligned, or the
        estimate is non-finite or <= 0.
    """
    if scale_method == "none":
        return 1.0
    if curr_submap.world_points is None or prior_submap.world_points is None:
        return None

    # Align resolutions: LC grids are full-res, regular grids stride 8
    # - regular: arange(0, W, 8) x arange(0, H, 8), row-major over (v, u); see _raw_to_world_points
    # - resample the LC side onto the regular grid so points pair pixel-for-pixel
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
        if joint.sum() >= min_conf_points:
            mask = joint
        elif (prior_conf > conf_threshold).sum() >= min_conf_points:
            mask = prior_conf > conf_threshold
        else:
            mask = prior_conf > 0

    # rotation_only, as the sequential edge: apply K ratio inv(K_prior) @ K_curr
    # - identity for a shared camera
    K_prior = prior_submap.intrinsics[prior_idx].astype(np.float64)
    K_curr = curr_submap.intrinsics[curr_idx].astype(np.float64)
    T = np.linalg.inv(K_prior) @ K_curr
    curr_in_prior = (T @ curr_cam.T).T
    s = estimate_scale_pairwise(curr_in_prior[mask], prior_cam[mask])

    # Guard degenerate estimates: None for a non-finite or <= 0 scale
    # - depth-unprojected LC points can hold NaN/inf from invalid pixels
    # - a bad median makes the diag(s, s, s, 1) between-factor singular
    # - the caller falls back to scale 1.0 with a warning
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def _loop_chain_relatives(
    P_lc0: np.ndarray,
    P_lc1: np.ndarray,
    s_a: float,
    s_b: float,
    K_q: np.ndarray,
    K_lc0: np.ndarray,
    K_lc1: np.ndarray,
    K_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three loop-chain relatives in the graph's H_inner convention (H_j = H_i @ M).

    - anchor A (query -> LC0, identical image): K change plus scale fold s_a
    - inner edge: the LC relative P_lc0 @ inv(P_lc1), no K change
    - anchor B (LC1 -> detected, identical image): K change plus scale fold s_b
    - an LC run at k-times scale gets s_a = 1/k, s_b = k, which cancel to the metric relative
    - follows the VGGT-SLAM solver.py:118-170 chain

    Args:
        P_lc0: 4x4 pose of LC frame 0.
        P_lc1: 4x4 pose of LC frame 1.
        s_a: scale from LC units to query-submap units.
        s_b: scale from detected-submap units to LC units.
        K_q: 4x4 query intrinsics.
        K_lc0: 4x4 LC frame 0 intrinsics.
        K_lc1: 4x4 LC frame 1 intrinsics.
        K_d: 4x4 detected intrinsics.

    Returns:
        (H_rel_a, H_inner, H_rel_b), each 4x4.
    """
    # Anchor A: query → LC-frame-0 (identical image) — K change + scale fold s_a.
    H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])
    # Inner edge: the LC pair's own relative pose, no K change.
    H_inner = P_lc0.astype(np.float64) @ np.linalg.inv(P_lc1.astype(np.float64))
    # Anchor B: LC-frame-1 → detected (identical image) — K change + scale fold s_b.
    H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])
    return H_rel_a, H_inner, H_rel_b
