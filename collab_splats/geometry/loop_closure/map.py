"""
Submap collection for loop closure, with ordered and world-frame reads.
"""

# Ported from VGGT-SLAM (github.com/MIT-SPARK/VGGT-SLAM), adapted for collab-splats.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .submap import Submap

if TYPE_CHECKING:
    from .graph import PoseGraph


class GraphMap:
    """
    Submaps keyed by submap_id.
    """

    def __init__(self) -> None:
        self.submaps: dict[int, Submap] = {}

    def add_submap(self, submap: Submap) -> None:
        """
        Insert a submap under its submap_id.

        Args:
            submap: submap to store; replaces any submap with the same id.
        """
        self.submaps[submap.submap_id] = submap

    def __len__(self) -> int:
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
        - without overlap, every seam between submaps is projected twice

        Args:
            graph: optimized PoseGraph holding one homography per frame.
            overlap: leading frames dropped from each non-first submap, which the
                previous submap already owns.

        Returns:
            (points, colors): (M, 3) float32 world points and (M, 3) uint8 RGB.
        """
        pts_chunks, col_chunks = [], []
        for s in self.ordered_submaps_by_key():
            # Skip LC submaps (2-frame loop carriers) and degraded submaps without
            # dense points (MapAnything degraded path) — neither contributes cloud.
            if s.is_lc_submap or s.points is None:
                continue
            # Non-first submaps: drop the leading overlap frames the earlier submap owns.
            skip = overlap if s.frame_start > 0 else 0
            pts_chunks.append(s.get_points_in_world_frame(graph, skip_first=skip))
            col_chunks.append(s.get_points_colors(skip_first=skip))
        points = np.vstack(pts_chunks) if pts_chunks else np.zeros((0, 3), dtype=np.float32)
        colors = np.vstack(col_chunks) if col_chunks else np.zeros((0, 3), dtype=np.uint8)
        return points, colors
