"""Shared test helpers for the loop-closure pose-graph tests."""

from __future__ import annotations

import numpy as np

from collab_splats.geometry.loop_closure.graph import PoseGraph


def drive_pose_graph(
    submaps,
    lc_submaps,
    total_frames,
    overlap_frames,
    conf_threshold=25.0,
    scale_method="se3",
    debug_out=None,
):
    """Test driver: batch per-submap PGO cadence via the incremental PoseGraph API."""
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
