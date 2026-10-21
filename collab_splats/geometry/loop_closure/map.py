"""
Submap collection for loop closure, with ordered reads.

- ported from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/map.py (GraphMap), adapted
"""

from __future__ import annotations

from collab_splats.geometry.loop_closure.submap import Submap


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
