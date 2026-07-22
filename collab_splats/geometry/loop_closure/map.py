"""GraphMap — owns the submap collection + provides scene access."""

# Ported from VGGT-SLAM (github.com/MIT-SPARK/VGGT-SLAM), adapted for collab-splats.
from __future__ import annotations

import numpy as np

from .submap import Submap


class GraphMap:
    """Collection of Submaps keyed by submap_id, with ordered/scene access."""

    def __init__(self) -> None:
        self.submaps: dict[int, Submap] = {}

    def add_submap(self, submap: Submap) -> None:
        """Insert a submap under its submap_id."""
        self.submaps[submap.submap_id] = submap

    def get_submap(self, submap_id: int) -> Submap:
        """Return the submap with the given id."""
        return self.submaps[submap_id]

    def __len__(self) -> int:
        return len(self.submaps)

    def ordered_submaps_by_key(self) -> list[Submap]:
        """Submaps sorted by ascending submap_id."""
        return [self.submaps[k] for k in sorted(self.submaps.keys())]

    def _keys(self, ignore_loop_closure_submaps: bool = False) -> list[int]:
        """Sorted submap ids, optionally excluding loop-closure submaps."""
        keys = [k for k, s in self.submaps.items() if not (ignore_loop_closure_submaps and s.is_lc_submap)]
        return sorted(keys)

    def get_largest_key(self, ignore_loop_closure_submaps: bool = False) -> int | None:
        """Highest submap id (None if empty), optionally ignoring LC submaps."""
        keys = self._keys(ignore_loop_closure_submaps)
        return keys[-1] if keys else None

    def get_latest_submap(self, ignore_loop_closure_submaps: bool = False) -> Submap:
        """Submap with the highest id (optionally ignoring LC submaps)."""
        key = self.get_largest_key(ignore_loop_closure_submaps)
        if key is None:
            raise ValueError("GraphMap.get_latest_submap called on an empty map")
        return self.submaps[key]

    def get_world_pointcloud(self, graph, overlap: int = 0):
        """Concatenated conf-masked world-frame (points, colors) over non-LC submaps.

        overlap drops each non-first submap's leading `overlap` frames — they duplicate
        the previous submap's trailing frames (dedup_overlap's first-writer rule for
        poses, applied here to the dense cloud). Without it every seam is double-projected.
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

    def get_corrected_extrinsics(self, graph, total_frames):
        """Corrected world-to-cam extrinsics (N,4,4) via the trusted graph path."""
        return graph.extract_extrinsics(total_frames)
