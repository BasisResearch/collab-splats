"""
SL(4) factor graph over per-frame camera projection matrices, for loop closure.

- optimizes poses on the SL(4) manifold to correct drift and close loops
- one node per frame, keyed by global frame index
- ported from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/graph.py (SL4 backend),
  vggt_slam/solver.py (add_edge), vggt_slam/scale_solver.py (estimate_scale_pairwise)
- decompose_camera (vggt_slam/slam_utils.py) lives in geometry.transforms
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X as _X

from collab_splats.geometry.transforms import (
    decompose_camera,
    intrinsics_4x4,
    transform_points,
)

from .submap import Submap

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

########################################################################
# Scale and overlap
########################################################################


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
    - VGGT-SLAM writes the overlap frame twice (MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/map.py:142-162)
    - evo (MichaelGrupp/evo) keeps the first
    - keeping the first makes both pipelines agree on the boundary pose

    Args:
        submap_ids: submap ids, in trajectory order.
        submap_starts: global frame index of each submap's first frame.
        corrected: per-submap (K_i, 4, 4) corrected poses, keyed by submap id.
        total_frames: output length.

    Returns:
        (total_frames, 4, 4) float32 poses.
    """
    # Identity everywhere, then fill each frame from the first submap that covers it
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


def estimate_scale_pairwise(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Scale between two paired point clouds: median(||Y[i]|| / ||X[i]||).

    - initializes the inter-submap SL(4) edge with the right scale
    - port of MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/scale_solver.py:estimate_scale_pairwise
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

    # Median norm ratio over points with a non-degenerate source norm
    x_norms = np.linalg.norm(X, axis=1)
    y_norms = np.linalg.norm(Y, axis=1)
    valid = x_norms > 1e-8
    if not np.any(valid):
        return 1.0
    return float(np.median(y_norms[valid] / x_norms[valid]))


def calculate_pairwise_frame_scale(
    curr_submap: Submap,
    curr_idx: int | list[int],
    prior_submap: Submap,
    prior_idx: int | list[int],
    min_conf_points: int,
) -> float | None:
    """
    Scale taking one frame's points into another's units, for two frames of the same image.

    - each frame's points are moved into its own camera first (divergence: docs/parity.md)
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
    curr_idx, prior_idx = list(np.atleast_1d(curr_idx)), list(np.atleast_1d(prior_idx))

    # Each frame's points in its own camera, rotated by inv(K_prior) @ K_curr (rotation_only)
    curr_pts, prior_pts = [], []
    for ci, pi in zip(curr_idx, prior_idx):
        c = transform_points(curr_submap.points[ci].reshape(-1, 3), curr_submap.poses[ci].astype(np.float64))
        p = transform_points(prior_submap.points[pi].reshape(-1, 3), prior_submap.poses[pi].astype(np.float64))
        K_prior = prior_submap.intrinsics[pi].astype(np.float64)
        K_curr = curr_submap.intrinsics[ci].astype(np.float64)
        curr_pts.append(c @ (np.linalg.inv(K_prior) @ K_curr).T)
        prior_pts.append(p)
    curr_pts, prior_pts = np.concatenate(curr_pts), np.concatenate(prior_pts)

    # Confidence tiers against the prior's threshold; a missing prior conf passes all
    mask = np.ones(curr_pts.shape[0], dtype=bool)
    if prior_submap.conf is not None:
        curr_conf = curr_submap.conf[curr_idx].reshape(-1) if curr_submap.conf is not None else None
        prior_conf = prior_submap.conf[prior_idx].reshape(-1)
        mask = _conf_fallback_mask(curr_conf, prior_conf, prior_submap.conf_threshold, min_conf_points)

    # NaN/inf depth pixels can poison the median; a scale <= 0 makes the SL(4) factor singular
    s = estimate_scale_pairwise(curr_pts[mask], prior_pts[mask])
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
        # Factor graph, initial values and the set of inserted node ids
        self.min_conf_points = min_conf_points
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._node_ids: set[int] = set()

        # Uniform sigma 0.05 on every edge, 1e-6 on the first-frame prior
        # - matches VGGT-SLAM exactly: sequential, inter-submap and loop-chain edges
        self._seq_noise = gtsam.noiseModel.Diagonal.Sigmas(0.05 * np.ones(15, dtype=np.float64))
        self._anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(15, 1e-6, dtype=np.float64))

        # Incremental submap-build bookkeeping (per-submap cadence state)
        self._global_node_id = 0
        self._frame_to_node: dict[tuple[int, int], int] = {}
        self._submap_node_ids: dict[int, list[int]] = {}
        self._submaps_seen: list = []  # prior submaps, for inter-submap overlap lookups

    ####################################################################
    # Node management
    ####################################################################

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
        key_i, key_j = _X(id_i), _X(id_j)
        H_rel = np.array(H_rel, dtype=np.float64)
        self._graph.add(gtsam.BetweenFactorSL4(key_i, key_j, gtsam.SL4(H_rel), self._seq_noise))

    ####################################################################
    # Optimization
    ####################################################################

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
            logger.warning("GTSAM optimization failed: %s — returning initial values", exc)

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
        node_ids_this: list[int] = []

        for local_i in range(k):
            nid = self._global_node_id + local_i
            self._frame_to_node[(submap.submap_id, local_i)] = nid
            node_ids_this.append(nid)

        # K_4x4 from 3x3 intrinsics, as VGGT-SLAM's proj_mats; see decompose_camera
        K_4x4 = intrinsics_4x4(submap.intrinsics.astype(np.float64))

        if not self._submaps_seen:
            # First node = I, prior = I, as VGGT-SLAM @ fd3fd218, vggt_slam/solver.py:169-170
            self.add_homography(node_ids_this[0], np.eye(4))
            self.add_prior_factor(node_ids_this[0], np.eye(4))
        else:
            # Previous submap's intrinsics and the usable overlap length
            prev_submap = self._submaps_seen[-1]
            prev_K_4x4 = intrinsics_4x4(prev_submap.intrinsics.astype(np.float64))
            O = min(overlap_frames, k, len(prev_submap.poses))

            # T = inv(K_prev[-1]) @ K_curr[0], as VGGT-SLAM's P_temp (vggt_slam/solver.py:140)
            # - intrinsics ratio; see decompose_camera
            T = np.linalg.inv(prev_K_4x4[-1]) @ K_4x4[0]

            # Inter-submap scale from the overlap frames, each in its own camera
            # - rotation_only, as VGGT-SLAM (vggt_slam/solver.py:141)
            # - diverges from solver.py:141-142, which reads points in frame 0; docs/parity.md
            scale = None
            if O > 0:
                prev_idx = list(range(len(prev_submap.poses) - O, len(prev_submap.poses)))
                curr_idx = list(range(O))
                scale = calculate_pairwise_frame_scale(submap, curr_idx, prev_submap, prev_idx, self.min_conf_points)
            if scale is None:
                logger.warning(
                    "Submap %d → %d: no usable overlap points (missing, misaligned or degenerate); scale 1.0",
                    prev_submap.submap_id,
                    submap.submap_id,
                )
                scale = 1.0

            # H_w = H_overlap @ T @ H_scale, as VGGT-SLAM @ fd3fd218, vggt_slam/solver.py:153-154
            # - SLAM: H_overlap_prev_node @ inv(K_prev) @ K_curr @ H_scale
            # - SLAM's proj_mats holds K (param named intrinsics_inv, submap.py:36-41; solver.py:247 passes K_4x4)
            # - so its inv(proj_mats[-1]) @ proj_mats[0] equals our T
            H_scale = np.diag([scale, scale, scale, 1.0])
            prev_submap_last_nid = self._submap_node_ids[prev_submap.submap_id][-1]
            H_overlap = self.get_homography(prev_submap_last_nid)
            H_w = H_overlap @ T @ H_scale
            self.add_homography(node_ids_this[0], H_w)

            # Inter-submap edge from the previous submap's last node
            H_rel_inter = np.linalg.inv(self.get_homography(prev_submap_last_nid)) @ H_w
            self.add_between_factor(prev_submap_last_nid, node_ids_this[0], H_rel_inter)

        # Chain consecutive frames inside the submap
        self._add_inner_chain(submap, node_ids_this)

        # Book the submap for later overlap lookups and extraction
        self._submap_node_ids[submap.submap_id] = node_ids_this
        self._global_node_id += k
        self._submaps_seen.append(submap)

    def _add_inner_chain(self, submap: Submap, node_ids: list[int]) -> None:
        """
        Chain edges between consecutive frames inside one submap.

        - H_inner = P[i-1] @ inv(P[i]); each node starts at its predecessor's value @ H_inner
        - node_ids: one graph node per frame, in frame order
        """
        # Seed each node from its predecessor and add the between factor
        for local_i in range(1, submap.poses.shape[0]):
            H_inner = submap.poses[local_i - 1].astype(np.float64) @ np.linalg.inv(
                submap.poses[local_i].astype(np.float64)
            )
            prev_H = self.get_homography(node_ids[local_i - 1])
            self.add_homography(node_ids[local_i], prev_H @ H_inner)
            self.add_between_factor(node_ids[local_i - 1], node_ids[local_i], H_inner)

    def add_loop_edge(self, lc: Submap, self_submaps: list[Submap]) -> None:
        """
        Add one verified loop closure as a 3-edge SL(4) chain through two LC nodes.

        - chain: query -> LC frame 0 -> LC frame 1 -> detected; LC nodes map to no output frame
        - skipped unless `lc` holds exactly 2 frames that both resolve to graph nodes
        - anchor scale falls back to 1.0, with a warning, when LC points are unusable

        Args:
            lc: 2-frame loop-closure Submap (query, detected).
            self_submaps: the regular submaps already in the graph.
        """
        # Only a 2-frame LC submap (query, detected) carries a loop edge; see wrapper._run_lc_loop
        if lc.poses.shape[0] != 2:
            return

        # Resolve both LC frames to graph nodes; skip when either is missing
        path_q, path_d = lc.image_paths[0], lc.image_paths[1]
        loc_q = _resolve_frame_node(self._frame_to_node, self_submaps, path_q)
        loc_d = _resolve_frame_node(self._frame_to_node, self_submaps, path_d)
        if loc_q is None or loc_d is None:
            return
        nid_q, sub_q, qi = loc_q
        nid_d, sub_d, di = loc_d

        # Anchor scales on pixel-aligned identical images
        # - 1.0 when the LC backend supplies poses only; the direction fix still applies
        # - argument order as VGGT-SLAM's add_edge pair (vggt_slam/solver.py:286-287)
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

        H_rel_a, H_inner_lc, H_rel_b = _loop_chain_relatives(lc.poses[0], lc.poses[1], s_a, s_b, K_q, K_lc0, K_lc1, K_d)

        # LC nodes chained from the query node's current graph state
        # - upstream vggt_slam/solver.py:153-158 via the add_edge pair at 286-287
        # - they map to no output frame, as vggt_slam/map.py:128-129 skips LC submaps
        nid_lc0, nid_lc1 = self._global_node_id, self._global_node_id + 1
        self._global_node_id += 2
        H_q_state = self.get_homography(nid_q)
        self.add_homography(nid_lc0, H_q_state @ H_rel_a)
        self.add_homography(nid_lc1, H_q_state @ H_rel_a @ H_inner_lc)

        # Three-edge chain: query -> LC0 -> LC1 -> detected
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

        # Decompose each node's K @ inv(H_opt) into a world-to-cam pose, per submap
        corrected_per_submap: dict[int, np.ndarray] = {}
        for submap in self._submaps_seen:
            node_ids = self._submap_node_ids[submap.submap_id]
            k = len(node_ids)
            poses_out = np.zeros((k, 4, 4), dtype=np.float32)

            # Extraction as VGGT-SLAM: proj_mats[i] @ inv(H_opt[i]) (vggt_slam/submap.py:118)
            # - K cancels; see decompose_camera
            s_K = intrinsics_4x4(submap.intrinsics.astype(np.float64))
            for local_i, nid in enumerate(node_ids):
                H_opt = self.get_homography(nid)
                local_proj = s_K[local_i]  # camera intrinsics as 4×4
                corrected = local_proj @ np.linalg.inv(H_opt)
                _, R, t, _ = decompose_camera(corrected)

                # Store R^T: decompose_camera's R is camera-to-world; see transforms.invert_poses
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = R.T.astype(np.float32)
                mat[:3, 3] = t.astype(np.float32)
                poses_out[local_i] = mat
            corrected_per_submap[submap.submap_id] = poses_out

        # One pose per frame; each overlap frame from the earlier submap
        return dedup_overlap(
            submap_ids=[s.submap_id for s in self._submaps_seen],
            submap_starts=[s.frame_start for s in self._submaps_seen],
            corrected=corrected_per_submap,
            total_frames=total_frames,
        )


########################################################################
# Helpers
########################################################################


def _resolve_frame_node(
    frame_to_node: dict[tuple[int, int], int],
    submaps: list[Submap],
    image_path: Path,
) -> tuple[int, Submap, int] | None:
    """
    Resolve an image path to (node_id, submap, local frame index).

    - frame_to_node keys: (submap_id, local index)
    - None when no node holds the path
    """
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                nid = frame_to_node.get((submap.submap_id, local_i))
                if nid is not None:
                    return nid, submap, local_i
    return None


def _conf_fallback_mask(
    curr_conf: np.ndarray | None, prior_conf: np.ndarray, conf_threshold: float, min_conf_points: int
) -> np.ndarray:
    """
    Points both frames trust, relaxed in tiers until min_conf_points survive.

    - tiers: both > threshold, then prior > threshold, then prior > 0 (vggt_slam/solver.py:129-138)
    - a missing curr_conf drops the first tier to prior > threshold
    """
    # First tier with enough survivors wins; else prior > 0
    joint = prior_conf > conf_threshold
    if curr_conf is not None:
        joint = joint & (curr_conf > conf_threshold)
    for mask in (joint, prior_conf > conf_threshold):
        if mask.sum() >= min_conf_points:
            return mask
    return prior_conf > 0


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

    - anchor A (query -> LC0, identical image): K change plus scale fold s_a (LC -> query-submap units)
    - inner edge: the LC relative P_lc0 @ inv(P_lc1), no K change
    - anchor B (LC1 -> detected, identical image): K change plus scale fold s_b (detected-submap -> LC units)
    - an LC run at k-times scale gets s_a = 1/k, s_b = k, which cancel to the metric relative
    - follows MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/solver.py:286-287 (add_edge, 118-195)
    - P_lc0, P_lc1 and all K_* are 4x4; returns (H_rel_a, H_inner, H_rel_b), each 4x4
    """
    # Anchor A: query -> LC frame 0 (identical image), K change plus scale fold s_a
    H_rel_a = np.linalg.inv(K_q) @ K_lc0 @ np.diag([s_a, s_a, s_a, 1.0])

    # Inner edge: the LC pair's own relative pose, no K change
    H_inner = P_lc0.astype(np.float64) @ np.linalg.inv(P_lc1.astype(np.float64))

    # Anchor B: LC frame 1 -> detected (identical image), K change plus scale fold s_b
    H_rel_b = np.linalg.inv(K_lc1) @ K_d @ np.diag([s_b, s_b, s_b, 1.0])
    return H_rel_a, H_inner, H_rel_b
