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
        pg.add_loop_edge(lc, submaps)
    pg.optimize()
    return pg.extract_extrinsics(total_frames)


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
