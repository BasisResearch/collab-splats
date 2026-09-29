"""
One loop-closure window of frames, poses and dense points.

- poses are world-to-cam in the window's local frame (frame 0 at identity)
- world-frame reads apply the optimized per-frame SL(4) homographies from the pose graph
- ported from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): vggt_slam/submap.py, adapted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from collab_splats.geometry.transforms import decompose_camera, intrinsics_4x4

if TYPE_CHECKING:
    from collab_splats.geometry.loop_closure.graph import PoseGraph


@dataclass
class Submap:
    """
    One window of frames from a single forward pass, in that window's local frame.

    - poses are world-to-cam, frame 0 ≈ identity; the LC wrapper checks it per window
    - dense fields (points/colors/conf) are per pixel, full resolution (fat Submap, as VGGT-SLAM)
    - points are unprojected depth in the window's frame-0 camera, as VGGT-SLAM's pointclouds
    - is_lc_submap: see wrapper._run_lc_loop

    Args:
        submap_id: key in the GraphMap; windows first, then loop carriers.
        poses: world-to-cam homogeneous poses, (K, 4, 4) float32.
        intrinsics: camera intrinsics, (K, 3, 3) float32.
        retrieval_vectors: DINO-SALAD global descriptors, (K, D) float32.
        image_paths: one source image per frame.
        frames: raw image tensors on CPU, (K, 3, H, W).
        is_lc_submap: True for a 2-frame loop carrier.
        frame_start: global index of the first frame in the full sequence.
        points: dense local points, (K, H, W, 3) float32.
        colors: per-pixel RGB, (K, H, W, 3) uint8.
        conf: per-pixel depth confidence, (K, H, W) float32.
        conf_percentile: percentile of conf that sets conf_threshold; VGGT-SLAM --conf_threshold.
        conf_threshold: confidence cutoff; the conf_percentile percentile of conf when not given.
    """

    submap_id: int
    poses: np.ndarray
    intrinsics: np.ndarray
    retrieval_vectors: torch.Tensor
    image_paths: list[Path]
    frames: torch.Tensor | None = None
    is_lc_submap: bool = False
    frame_start: int = 0
    points: np.ndarray | None = field(default=None, repr=False)
    colors: np.ndarray | None = field(default=None, repr=False)
    conf: np.ndarray | None = field(default=None, repr=False)
    conf_percentile: float = 25.0
    conf_threshold: float | None = None

    def __post_init__(self) -> None:
        """
        Derive conf_threshold from dense conf when none was given.

        - percentile over every frame's conf, + 1e-6, as MIT-SPARK/VGGT-SLAM @ fd3fd218, vggt_slam/submap.py:36-41
        """
        if self.conf is not None and self.conf.size > 0 and self.conf_threshold is None:
            self.conf_threshold = float(np.percentile(self.conf, self.conf_percentile)) + 1e-6

    def set_dense_points(self, points: np.ndarray, colors: np.ndarray, conf: np.ndarray) -> None:
        """
        Store dense per-pixel fields and derive conf_threshold from conf.

        Args:
            points: local-frame points, (K, H, W, 3) float32.
            colors: per-pixel RGB, (K, H, W, 3) uint8.
            conf: per-pixel confidence, (K, H, W) float32.
        """
        self.points = points
        self.colors = colors
        self.conf = conf
        if conf is not None and conf.size > 0:
            self.conf_threshold = float(np.percentile(conf, self.conf_percentile)) + 1e-6

    ####################################################################
    # Graph-corrected world reads
    ####################################################################

    def get_points_in_world_frame(self, graph: PoseGraph, skip_first: int = 0) -> np.ndarray:
        """
        Dense points in the world frame, corrected by the pose graph and masked by confidence.

        - frame i uses the optimized SL(4) homography of node `frame_start + i`
        - frame order and mask match `get_points_colors` for the same skip_first

        Args:
            graph: optimized PoseGraph holding one homography per frame.
            skip_first: leading frames to drop; see LoopClosureConfig for overlap.

        Returns:
            World-frame points, (M, 3) float32; empty when every frame is skipped.

        Raises:
            ValueError: if the submap has no dense points or no conf.
        """
        if self.points is None:
            raise ValueError(f"Submap {self.submap_id} has no dense points")
        if self.conf is None:
            raise ValueError(f"Submap {self.submap_id} has no conf; cannot compute world-frame points")
        out = []
        for i in range(skip_first, self.points.shape[0]):
            # Lift the frame's points to homogeneous and apply its homography
            H = graph.get_homography(self.frame_start + i).astype(np.float64)
            flat = self.points[i].reshape(-1, 3).astype(np.float64)  # (H*W, 3)
            hom = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float64)])
            out_hom = (H @ hom.T).T  # (H*W, 4)

            # Dehomogenize by w, guarding near-zero w (plane at infinity)
            w = out_hom[:, 3:4]
            w = np.where(np.abs(w) < 1e-10, 1e-10, w)
            world = out_hom[:, :3] / w

            # Confidence mask; same predicate and order as get_points_colors
            mask = self.conf[i].reshape(-1) > self.conf_threshold
            out.append(world[mask])

        # A pure-overlap tail submap skips every frame; return empty
        if not out:
            return np.empty((0, 3), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    def get_points_colors(self, skip_first: int = 0) -> np.ndarray:
        """
        Per-point RGB aligned with `get_points_in_world_frame`.

        Args:
            skip_first: leading frames to drop; pass the value given to
                `get_points_in_world_frame`.

        Returns:
            RGB, (M, 3) uint8.

        Raises:
            ValueError: if the submap has no conf.
        """
        if self.conf is None:
            raise ValueError(f"Submap {self.submap_id} has no conf; cannot compute world-frame points")
        return self.colors[skip_first:][self.conf[skip_first:] > self.conf_threshold].reshape(-1, 3)

    def get_all_poses_world(self, graph: PoseGraph) -> np.ndarray:
        """
        World-to-cam poses decomposed from K @ inv(H_opt), one per frame.

        - must equal `PoseGraph.extract_extrinsics` for these frames
        - stores R.T, not upstream's R; see transforms.decompose_camera

        Args:
            graph: optimized PoseGraph holding one homography per frame.

        Returns:
            World-to-cam poses, (N, 4, 4) float32.
        """
        poses = []
        for i in range(self.poses.shape[0]):
            # Projection K @ inv(H_opt), normalized so its last entry is 1
            K_4x4 = intrinsics_4x4(self.intrinsics[i].astype(np.float64))
            proj = K_4x4 @ np.linalg.inv(graph.get_homography(self.frame_start + i))
            proj = proj / proj[-1, -1]

            # Decompose, then store as world-to-cam; see transforms.decompose_camera
            _, rot, trans, _ = decompose_camera(proj[:3, :])
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot.T.astype(np.float32)
            pose[:3, 3] = trans.astype(np.float32)
            poses.append(pose)
        return np.stack(poses, axis=0)
