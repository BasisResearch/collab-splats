"""
Submap collection for loop closure, with ordered and world-frame reads.

- ported from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/map.py (GraphMap), adapted
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from collab_splats.geometry.loop_closure.submap import Submap

if TYPE_CHECKING:
    from collab_splats.geometry.loop_closure.graph import PoseGraph


class GraphMap:
    """
    Submaps keyed by submap_id.
    """

    def __init__(self) -> None:
        """
        Start with no submaps.
        """
        self.submaps: dict[int, Submap] = {}

    def add_submap(self, submap: Submap) -> None:
        """
        Insert a submap under its submap_id.

        Args:
            submap: submap to store; replaces any submap with the same id.
        """
        self.submaps[submap.submap_id] = submap

    def __len__(self) -> int:
        """
        Number of stored submaps.

        Returns:
            Count of submaps, loop carriers included.
        """
        return len(self.submaps)

    def ordered_submaps_by_key(self) -> list[Submap]:
        """
        All submaps, loop carriers included.

        Returns:
            Submaps in ascending submap_id order.
        """
        return [self.submaps[k] for k in sorted(self.submaps.keys())]

    def get_world_pointcloud(self, graph: PoseGraph, overlap: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Confidence-masked world-frame cloud over every submap with dense points.

        - skips loop-carrier submaps and submaps without dense points
        - overlap: see LoopClosureConfig

        Args:
            graph: optimized PoseGraph holding one homography per frame.
            overlap: leading frames dropped from each non-first submap.

        Returns:
            (points, colors): (M, 3) float32 world points and (M, 3) uint8 RGB.
        """
        pts_chunks, col_chunks = [], []
        for s in self.ordered_submaps_by_key():
            # No cloud from loop carriers or submaps without dense points
            # - carriers: see wrapper._run_lc_loop
            # - no dense points: MapAnything degraded path
            if s.is_lc_submap or s.points is None:
                continue

            # Drop a non-first submap's overlap frames; see LoopClosureConfig
            skip = overlap if s.frame_start > 0 else 0
            pts_chunks.append(s.get_points_in_world_frame(graph, skip_first=skip))
            col_chunks.append(s.get_points_colors(skip_first=skip))

        # Stack the chunks; an empty map gives empty arrays
        points = np.vstack(pts_chunks) if pts_chunks else np.zeros((0, 3), dtype=np.float32)
        colors = np.vstack(col_chunks) if col_chunks else np.zeros((0, 3), dtype=np.uint8)
        return points, colors
