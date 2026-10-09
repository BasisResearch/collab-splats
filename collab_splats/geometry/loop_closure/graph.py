"""
SL(4) factor graph over per-frame camera projection matrices, for loop closure.

- optimizes poses on the SL(4) manifold to correct drift and close loops
- one node per frame, keyed by global frame index
- ported from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/graph.py, solver.py, scale_solver.py
- decompose_camera (vggt_slam/slam_utils.py) lives in geometry.transforms
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import gtsam
from gtsam.symbol_shorthand import X as _X

from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.transforms import intrinsics_4x4, transform_points

logger = logging.getLogger(__name__)

########################################################################
# Scale and overlap
########################################################################


def calculate_pairwise_frame_scale(
    curr_submap: Submap,
    curr_idx: int | list[int],
    prior_submap: Submap,
    prior_idx: int | list[int],
    min_conf_points: int,
) -> float | None:
    """
    Scale taking one frame's points into another's units, for two frames of the same image.

    - each frame's points are moved into its own camera first (divergence: decision 027)
    - None when points are missing, grids differ, or the scale is not finite and > 0

    Args:
        curr_submap: submap holding the frames to rescale.
        curr_idx: local frame index, or indices pooled into one fit.
        prior_submap: submap whose units the scale maps into.
        prior_idx: local frame index or indices, paired with curr_idx.
        min_conf_points: fewest confident points before the confidence mask is used.

    Returns:
        Scale from curr_submap's units to prior_submap's, or None.
    """
    # No points to fit the scale on, or grids that do not pair pixel-for-pixel
    if curr_submap.points is None or prior_submap.points is None:
        return None

    if curr_submap.points.shape[1:3] != prior_submap.points.shape[1:3]:
        return None

    # One index or a list of them, paired frame by frame
    curr_idx = list(np.atleast_1d(curr_idx))
    prior_idx = list(np.atleast_1d(prior_idx))

    # Each frame's points in its own camera, rotated by inv(K_prior) @ K_curr (rotation_only)
    curr_pts, prior_pts = [], []

    for ci, pi in zip(curr_idx, prior_idx):
        c_local = curr_submap.points[ci].reshape(-1, 3)
        p_local = prior_submap.points[pi].reshape(-1, 3)
        c = transform_points(c_local, curr_submap.poses[ci].astype(np.float64))
        p = transform_points(p_local, prior_submap.poses[pi].astype(np.float64))
        K_prior = prior_submap.intrinsics[pi].astype(np.float64)
        K_curr = curr_submap.intrinsics[ci].astype(np.float64)
        K_ratio = np.linalg.inv(K_prior) @ K_curr
        curr_pts.append(c @ K_ratio.T)
        prior_pts.append(p)

    # Every pair's points in one array per side
    curr_pts = np.concatenate(curr_pts)
    prior_pts = np.concatenate(prior_pts)

    # Confidence tiers until min_conf_points survive: both > threshold, prior > threshold, prior > 0 (solver.py:129-138)
    mask = np.ones(curr_pts.shape[0], dtype=bool)

    if prior_submap.conf is not None:
        prior_conf = prior_submap.conf[prior_idx].reshape(-1)
        prior_mask = prior_conf > prior_submap.conf_threshold
        joint = prior_mask

        if curr_submap.conf is not None:
            curr_conf = curr_submap.conf[curr_idx].reshape(-1)
            joint = joint & (curr_conf > prior_submap.conf_threshold)

        if joint.sum() >= min_conf_points:
            mask = joint
        elif prior_mask.sum() >= min_conf_points:
            mask = prior_mask
        else:
            mask = prior_conf > 0

    # Median norm ratio over non-degenerate source norms (scale_solver.py:estimate_scale_pairwise); 1.0 when none
    x_norms = np.linalg.norm(curr_pts[mask], axis=1)
    y_norms = np.linalg.norm(prior_pts[mask], axis=1)
    valid = x_norms > 1e-8
    s = float(np.median(y_norms[valid] / x_norms[valid])) if np.any(valid) else 1.0

    # NaN/inf depth pixels can poison the median; a scale <= 0 makes the SL(4) factor singular
    return s if np.isfinite(s) and s > 0 else None


########################################################################
# PoseGraph
########################################################################


class PoseGraph:
    """
    Per-frame SL(4) factor graph for loop-closure trajectory correction.

    - one node per frame, keyed gtsam.symbol('x', node_id) by global frame index
    - port of MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/graph.py:PoseGraph
    - noise matches vggt_slam/graph.py:19-23: sigma 0.05 on every edge, 1e-6 on the first-frame prior

    Args:
        min_conf_points: fewest confident points before the scale fit uses the confidence mask.
    """

    def __init__(self, min_conf_points: int = 100) -> None:
        """
        Empty graph with no nodes, edges or submaps.

        Args:
            min_conf_points: fewest confident points before the scale fit uses the confidence mask.
        """
        # Factor graph and initial values
        self.min_conf_points = min_conf_points
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()

        # Sigma 0.05 on every edge (sequential, inter-submap, loop chain), 1e-6 on the first-frame prior
        self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(
            0.05 * np.ones(15, dtype=np.float64)
        )
        self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(15, 1e-6, dtype=np.float64)
        )

        # Incremental submap-build bookkeeping (per-submap cadence state)
        self._global_node_id = 0
        self._submap_node_ids: dict[int, list[int]] = {}
        self._submaps_seen: list[Submap] = []

    ####################################################################
    # Node management
    ####################################################################

    def add_homography(self, node_id: int, H: np.ndarray) -> None:
        """
        Insert a frame node.

        Args:
            node_id: new global frame node id.
            H: 4x4 homography; GTSAM's SL4 normalizes it.
        """
        # Node initial value as an SL(4) element
        H = np.array(H, dtype=np.float64)
        key = _X(node_id)
        H_sl4 = gtsam.SL4(H)
        self._initial.insert(key, H_sl4)

    ####################################################################
    # Edges
    ####################################################################

    def add_between_factor(self, id_i: int, id_j: int, H_rel: np.ndarray) -> None:
        """
        Relative SL(4) constraint between two nodes: H_j = H_i @ H_rel.

        Args:
            id_i: source node id.
            id_j: target node id.
            H_rel: 4x4 relative homography.
        """
        # Between factor under the sequential noise model
        H_rel = np.array(H_rel, dtype=np.float64)
        key_i = _X(id_i)
        key_j = _X(id_j)
        H_rel_sl4 = gtsam.SL4(H_rel)
        factor = gtsam.BetweenFactorSL4(key_i, key_j, H_rel_sl4, self._seq_noise)
        self._graph.add(factor)

    ####################################################################
    # Optimization
    ####################################################################

    def optimize(self) -> None:
        """
        Run Levenberg-Marquardt in place; read results with get_homography.

        - on a GTSAM RuntimeError the initial values are kept and a warning is logged
        """
        # Levenberg-Marquardt over the whole graph; failures keep the initial values
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self._graph, self._initial, params
            )
            self._initial = optimizer.optimize()
        except RuntimeError as exc:
            logger.warning(
                "GTSAM optimization failed: %s — returning initial values", exc
            )

    def get_homography(self, node_id: int) -> np.ndarray:
        """
        4x4 homography of a frame node; the optimized value once optimize() has run.

        Args:
            node_id: node to read.

        Returns:
            (4, 4) float64 homography.
        """
        # Node value as a float64 matrix
        key = _X(node_id)
        H = self._initial.atSL4(key).matrix()
        return H.astype(np.float64)

    ####################################################################
    # Incremental submap build
    ####################################################################

    def add_submap(self, submap: Submap, overlap_frames: int) -> None:
        """
        Add one submap's nodes and SL(4) edges to the persistent graph.

        - the first submap is anchored at identity with a tight prior
        - later submaps link to the previous submap's last node through the overlap frames
        - the inter-submap scale comes from the overlap points, rotation_only as VGGT-SLAM
        - overlap points on both sides move into their own frame's camera; upstream skips this
        - scale falls back to 1.0, with a warning, when the overlap points are missing or degenerate

        Args:
            submap: the next Submap in trajectory order.
            overlap_frames: frames shared with the previous submap.
        """
        # Allocate consecutive global node ids for this submap's frames
        k = submap.poses.shape[0]
        node_ids_this = list(range(self._global_node_id, self._global_node_id + k))

        # First node = I, prior = I, as VGGT-SLAM @ fd3fd218, vggt_slam/solver.py:169-170
        if not self._submaps_seen:
            self.add_homography(node_ids_this[0], np.eye(4))

            # Tight prior (sigma 1e-6) holding the first frame at identity
            H_identity = gtsam.SL4(np.eye(4))
            key = _X(node_ids_this[0])
            prior = gtsam.PriorFactorSL4(key, H_identity, self._anchor_noise)
            self._graph.add(prior)
        else:
            # Previous submap and the usable overlap length
            prev_submap = self._submaps_seen[-1]
            n_prev = len(prev_submap.poses)
            n_overlap = min(overlap_frames, k, n_prev)

            # T = inv(K_prev[-1]) @ K_curr[0], VGGT-SLAM's P_temp (solver.py:140); see decompose_camera
            K_prev_last = intrinsics_4x4(prev_submap.intrinsics[-1].astype(np.float64))
            K_curr_first = intrinsics_4x4(submap.intrinsics[0].astype(np.float64))
            T = np.linalg.inv(K_prev_last) @ K_curr_first

            # Overlap scale, each frame in its own camera (rotation_only; diverges from solver.py:141, decision 027)
            scale = None

            if n_overlap > 0:
                prev_idx = list(range(n_prev - n_overlap, n_prev))
                curr_idx = list(range(n_overlap))
                scale = calculate_pairwise_frame_scale(
                    submap, curr_idx, prev_submap, prev_idx, self.min_conf_points
                )

            if scale is None:
                logger.warning(
                    "Submap %d → %d: no usable overlap points (missing, misaligned or degenerate); scale 1.0",
                    prev_submap.submap_id,
                    submap.submap_id,
                )
                scale = 1.0

            # H_w = H_overlap @ T @ H_scale, VGGT-SLAM solver.py:153-154 (its proj_mats hold K, so T matches)
            H_scale = np.diag([scale, scale, scale, 1.0])
            prev_submap_last_nid = self._submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = self.get_homography(prev_submap_last_nid)
            H_w = H_overlap @ T @ H_scale
            self.add_homography(node_ids_this[0], H_w)

            # Inter-submap edge from the previous submap's last node
            H_rel_inter = np.linalg.inv(H_overlap) @ H_w
            self.add_between_factor(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

        # Chain consecutive frames: H_inner = P[i-1] @ inv(P[i]), each node seeded from its predecessor
        poses = submap.poses.astype(np.float64)

        for local_i in range(1, k):
            H_inner = poses[local_i - 1] @ np.linalg.inv(poses[local_i])
            prev_H = self.get_homography(node_ids_this[local_i - 1])
            self.add_homography(node_ids_this[local_i], prev_H @ H_inner)
            self.add_between_factor(
                node_ids_this[local_i - 1], node_ids_this[local_i], H_inner
            )

        # Book the submap for later overlap lookups and extraction
        self._submap_node_ids[submap.submap_id] = node_ids_this
        self._global_node_id += k
        self._submaps_seen.append(submap)

    def add_loop_edge(self, lc: Submap) -> None:
        """
        Add one verified loop closure as a 3-edge SL(4) chain through two LC nodes.

        - chain: query -> LC frame 0 -> LC frame 1 -> detected; LC nodes map to no output frame
        - both LC image paths must belong to submaps already in the graph
        - anchor scale falls back to 1.0, with a warning, when LC points are unusable

        Args:
            lc: 2-frame loop-closure Submap (query, detected).
        """
        # Image path -> (node id, submap, local index); the earliest submap holding the path wins
        located: dict[Path, tuple[int, Submap, int]] = {}

        for sub in self._submaps_seen:
            for local_i, p in enumerate(sub.image_paths):
                located.setdefault(
                    p, (self._submap_node_ids[sub.submap_id][local_i], sub, local_i)
                )

        # Graph node, submap and local frame of each carrier image
        nid_q, sub_q, qi = located[lc.image_paths[0]]
        nid_d, sub_d, di = located[lc.image_paths[1]]

        # Anchor scales on identical images, argument order as VGGT-SLAM's add_edge (solver.py:286-287)
        s_a = calculate_pairwise_frame_scale(lc, 0, sub_q, qi, self.min_conf_points)
        s_b = calculate_pairwise_frame_scale(sub_d, di, lc, 1, self.min_conf_points)

        if s_a is None or s_b is None:
            logger.warning(
                "Loop submap %d → %d: LC points unavailable or grid-misaligned; using anchor scale 1.0",
                sub_q.submap_id,
                sub_d.submap_id,
            )
            s_a = 1.0 if s_a is None else s_a
            s_b = 1.0 if s_b is None else s_b

        # Per-frame intrinsics as 4×4 for the K change across anchors (I for a shared camera)
        K_q = intrinsics_4x4(sub_q.intrinsics[qi].astype(np.float64))
        K_lc0 = intrinsics_4x4(lc.intrinsics[0].astype(np.float64))
        K_lc1 = intrinsics_4x4(lc.intrinsics[1].astype(np.float64))
        K_d = intrinsics_4x4(sub_d.intrinsics[di].astype(np.float64))

        # Chain relatives H_j = H_i @ M (solver.py:286-287): anchors fold K and scale, inner edge is LC pose
        H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])
        P_lc0 = lc.poses[0].astype(np.float64)
        P_lc1 = lc.poses[1].astype(np.float64)
        H_inner_lc = P_lc0 @ np.linalg.inv(P_lc1)
        H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])

        # LC nodes chained from the query node's state (solver.py:153-158); no output frame (vggt_slam/map.py:128-129)
        nid_lc0, nid_lc1 = self._global_node_id, self._global_node_id + 1
        self._global_node_id += 2
        H_q_state = self.get_homography(nid_q)
        self.add_homography(nid_lc0, H_q_state @ H_rel_a)
        self.add_homography(nid_lc1, H_q_state @ H_rel_a @ H_inner_lc)

        # Three-edge chain: query -> LC0 -> LC1 -> detected
        self.add_between_factor(nid_q, nid_lc0, H_rel_a)
        self.add_between_factor(nid_lc0, nid_lc1, H_inner_lc)
        self.add_between_factor(nid_lc1, nid_d, H_rel_b)

    def submap_node_ids(self, submap_id: int) -> list[int]:
        """
        Graph node ids of one submap's frames, in frame order.

        - node ids are allocated per submap frame, overlap frames included, then 2 per loop carrier
        - so a frame's node id is not its global frame index

        Args:
            submap_id: id of a submap already added with add_submap.

        Returns:
            One node id per submap frame.
        """
        return list(self._submap_node_ids[submap_id])
