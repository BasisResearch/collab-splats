"""Shared test helpers for the loop-closure pose-graph tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from collab_splats.geometry.loop_closure.graph import PoseGraph


def drive_pose_graph(
    submaps,
    lc_submaps,
    total_frames,
    overlap_frames,
):
    """Test driver: batch per-submap PGO cadence via the incremental PoseGraph API."""
    if not submaps:
        return np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
    pg = PoseGraph()
    for submap in submaps:
        pg.add_submap(submap, overlap_frames)
        pg.optimize()
    for lc in lc_submaps:
        pg.add_loop_edge(lc)
    pg.optimize()
    return graph_extrinsics(pg, total_frames)


def graph_extrinsics(pg: PoseGraph, total_frames: int) -> np.ndarray:
    """
    Per-frame world-to-cam poses from a solved graph, as LoopClosure._assemble_result fills them.

    - each submap's get_all_poses_world; the first submap to cover a frame wins
    - identity for frames no submap covers
    """
    out = np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
    assigned = np.zeros(total_frames, dtype=bool)

    for s in pg._submaps_seen:
        for local_i, pose in enumerate(s.get_all_poses_world(pg)):
            g = s.frame_start + local_i

            if g < total_frames and not assigned[g]:
                out[g] = pose
                assigned[g] = True

    return out


def masked_world_points(submap, graph: PoseGraph, skip: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    A dense submap's conf-masked world points and colors past `skip` leading frames.

    - the read LoopClosure._viz_push_submap and _assemble_result make
    """
    grid = submap.get_world_grid(graph)[skip:]
    mask = submap.conf[skip:] > submap.conf_threshold
    return grid[mask], submap.colors[skip:][mask]


@contextmanager
def record_driven_submaps() -> Iterator[dict]:
    """
    Record every submap LoopClosure hands the pose graph, in call order.

    - spies PoseGraph.add_submap and add_loop_edge; the real methods still run

    Yields:
        dict filled on exit: "submaps" (add_submap) and "lc_submaps" (add_loop_edge).
    """
    with (
        patch.object(PoseGraph, "add_submap", autospec=True, side_effect=PoseGraph.add_submap) as add,
        patch.object(PoseGraph, "add_loop_edge", autospec=True, side_effect=PoseGraph.add_loop_edge) as loop,
    ):
        driven: dict = {}
        yield driven
    driven["submaps"] = [c.args[1] for c in add.call_args_list]
    driven["lc_submaps"] = [c.args[1] for c in loop.call_args_list]
