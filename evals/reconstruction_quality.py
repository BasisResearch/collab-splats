"""Submap alignment quality metrics for LC validation.

Three metrics, each computed before and after pose-graph optimization (PGO):
  loop_match_residual  — camera-position distance for accepted LC match pairs
  submap_boundary_gap  — camera-position gap at consecutive submap stitches
  pointcloud_chamfer   — symmetric Chamfer distance between matched-submap camera positions

All return JSON-serializable dicts. compute_alignment_metrics(lc_creator) aggregates
all three from a post-run LoopClosure instance.
"""
from __future__ import annotations

import numpy as np


########################################
########## Internal helpers ############
########################################

def _cam_positions(poses_w2c: np.ndarray) -> np.ndarray:
    """(N, 4, 4) world-to-cam → (N, 3) camera positions in world frame."""
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between (M, 3) and (N, 3) point sets."""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d_a = cKDTree(b).query(a)[0].mean()
    d_b = cKDTree(a).query(b)[0].mean()
    return float((d_a + d_b) / 2)


def _global_frame(submap, local_frame_idx: int) -> int:
    """Convert local frame index within a submap to global frame index."""
    return submap.frame_start + local_frame_idx


########################################
########## Public metrics ##############
########################################

def loop_match_residual(
    matches: list,
    submaps: list,
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
) -> dict:
    """Mean/max camera-position distance for accepted loop match pairs.

    Args:
        matches: list of LoopMatch-like objects with accepted, query_submap_id,
                 query_frame_idx, detected_submap_id, detected_frame_idx.
        submaps: list of Submap-like objects with submap_id, frame_start, poses.
        pre_ext: (N, 4, 4) pre-PGO world-to-cam extrinsics.
        post_ext: (N, 4, 4) post-PGO world-to-cam extrinsics.

    Returns:
        dict with mean_before, max_before, mean_after, max_after, n_matches.
    """
    accepted = [m for m in matches if m.accepted]
    if not accepted:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_matches": 0}

    sid_to_submap = {s.submap_id: s for s in submaps}
    pre_pos = _cam_positions(pre_ext)
    post_pos = _cam_positions(post_ext)

    dists_before, dists_after = [], []
    for m in accepted:
        q_submap = sid_to_submap.get(m.query_submap_id)
        d_submap = sid_to_submap.get(m.detected_submap_id)
        if q_submap is None or d_submap is None:
            continue
        q_g = _global_frame(q_submap, m.query_frame_idx)
        d_g = _global_frame(d_submap, m.detected_frame_idx)
        if q_g < len(pre_pos) and d_g < len(pre_pos):
            dists_before.append(float(np.linalg.norm(pre_pos[q_g] - pre_pos[d_g])))
            dists_after.append(float(np.linalg.norm(post_pos[q_g] - post_pos[d_g])))

    if not dists_before:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_matches": 0}

    return {
        "mean_before": float(np.mean(dists_before)),
        "max_before":  float(np.max(dists_before)),
        "mean_after":  float(np.mean(dists_after)),
        "max_after":   float(np.max(dists_after)),
        "n_matches":   len(dists_before),
    }


def submap_boundary_gap(
    submaps: list,
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
) -> dict:
    """Mean/max camera-position gap at consecutive submap stitches.

    For each consecutive pair (submap[i], submap[i+1]), measures distance between
    the last global frame of submap[i] and the first global frame of submap[i+1].
    """
    normal = [s for s in submaps if not getattr(s, "is_lc_submap", False)]
    if len(normal) < 2:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_boundaries": 0}

    pre_pos = _cam_positions(pre_ext)
    post_pos = _cam_positions(post_ext)
    gaps_before, gaps_after = [], []

    # Measure distance between last frame of submap[i] and first frame of submap[i+1]
    for i in range(len(normal) - 1):
        s_cur = normal[i]
        s_nxt = normal[i + 1]
        last_g = s_cur.frame_start + len(s_cur.poses) - 1
        first_g = s_nxt.frame_start
        if last_g < len(pre_pos) and first_g < len(pre_pos):
            gaps_before.append(float(np.linalg.norm(pre_pos[last_g] - pre_pos[first_g])))
            gaps_after.append(float(np.linalg.norm(post_pos[last_g] - post_pos[first_g])))

    if not gaps_before:
        return {"mean_before": None, "max_before": None,
                "mean_after": None, "max_after": None, "n_boundaries": 0}

    return {
        "mean_before":   float(np.mean(gaps_before)),
        "max_before":    float(np.max(gaps_before)),
        "mean_after":    float(np.mean(gaps_after)),
        "max_after":     float(np.max(gaps_after)),
        "n_boundaries":  len(gaps_before),
    }


def pointcloud_chamfer(
    matches: list,
    submaps: list,
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
    max_pairs: int = 20,
) -> dict:
    """Mean Chamfer distance (on camera positions) between matched-submap pairs.

    Uses camera positions of all frames within each matched submap. Caps at max_pairs.
    """
    accepted = [m for m in matches if m.accepted][:max_pairs]
    if not accepted:
        return {"mean_before": None, "mean_after": None, "n_pairs": 0}

    sid_to_submap = {s.submap_id: s for s in submaps}
    pre_pos = _cam_positions(pre_ext)
    post_pos = _cam_positions(post_ext)

    chamfers_before, chamfers_after = [], []
    for m in accepted:
        q_s = sid_to_submap.get(m.query_submap_id)
        d_s = sid_to_submap.get(m.detected_submap_id)
        if q_s is None or d_s is None:
            continue

        # Extract camera-position slices for each submap
        q_range = slice(q_s.frame_start, q_s.frame_start + len(q_s.poses))
        d_range = slice(d_s.frame_start, d_s.frame_start + len(d_s.poses))
        q_pre = pre_pos[q_range]
        d_pre = pre_pos[d_range]
        q_post = post_pos[q_range]
        d_post = post_pos[d_range]
        if len(q_pre) > 0 and len(d_pre) > 0:
            chamfers_before.append(_symmetric_chamfer(q_pre, d_pre))
            chamfers_after.append(_symmetric_chamfer(q_post, d_post))

    if not chamfers_before:
        return {"mean_before": None, "mean_after": None, "n_pairs": 0}

    return {
        "mean_before": float(np.nanmean(chamfers_before)),
        "mean_after":  float(np.nanmean(chamfers_after)),
        "n_pairs":     len(chamfers_before),
    }


########################################
########## Aggregator ##################
########################################

def compute_alignment_metrics(lc_creator) -> dict:
    """Aggregate all three alignment metrics from a post-run LoopClosure instance.

    Reads _lc_submaps, _lc_all_matches, _lc_precorrection_extrinsics,
    _lc_corrected_extrinsics from lc_creator.base. Returns empty dict if any
    required attribute is missing (e.g. no loop closures found).
    """
    base = lc_creator.base
    required = [
        "_lc_submaps", "_lc_all_matches",
        "_lc_precorrection_extrinsics", "_lc_corrected_extrinsics",
    ]
    # Return empty dict if any required attribute is absent
    for attr in required:
        if not hasattr(base, attr):
            return {}

    submaps = base._lc_submaps
    matches = base._lc_all_matches
    pre_ext = base._lc_precorrection_extrinsics
    post_ext = base._lc_corrected_extrinsics

    return {
        "loop_match_residual": loop_match_residual(matches, submaps, pre_ext, post_ext),
        "submap_boundary_gap": submap_boundary_gap(submaps, pre_ext, post_ext),
        "pointcloud_chamfer":  pointcloud_chamfer(matches, submaps, pre_ext, post_ext),
    }
