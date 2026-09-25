"""
One loop-closure window of frames, poses and dense points.

- poses are world-to-cam in the window's local frame (frame 0 at identity)
- world-frame reads apply the optimized per-frame SL(4) homographies from the pose graph
"""

# Ported from VGGT-SLAM (github.com/MIT-SPARK/VGGT-SLAM), adapted for collab-splats.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .graph import PoseGraph


@dataclass
class Submap:
    """
    One window of frames from a single forward pass, in that window's local frame.

    - poses are world-to-cam (frame 0 ≈ identity); `assert_world_to_cam` checks it
    - dense fields (points/colors/conf) are per pixel; conf_threshold is their 25th percentile
    - is_lc_submap marks a 2-frame loop carrier, which never contributes points
    """

    submap_id: int
    poses: np.ndarray  # (K, 4, 4) float32 — world-to-cam homogeneous
    intrinsics: np.ndarray  # (K, 3, 3) float32 — camera intrinsics
    retrieval_vectors: torch.Tensor  # (K, D) float32 — DINO-SALAD global descriptors
    image_paths: list[Path]
    frames: torch.Tensor | None = None  # (K, 3, H, W) — raw image tensors on CPU
    is_lc_submap: bool = False
    frame_start: int = 0  # global index of this submap's first frame in the full sequence
    world_points: np.ndarray | None = field(
        default=None, repr=False
    )  # (K, P, 3) float32 — per-frame 3D points in local frame
    world_points_conf: np.ndarray | None = field(default=None, repr=False)  # (K, P) float32 — per-point confidence
    # Dense per-frame data (fat Submap, mirrors VGGT-SLAM): full-resolution points/colors/conf.
    points: np.ndarray | None = field(default=None, repr=False)  # (K, H, W, 3) float32 — dense local points
    colors: np.ndarray | None = field(default=None, repr=False)  # (K, H, W, 3) uint8 — per-pixel RGB
    conf: np.ndarray | None = field(default=None, repr=False)  # (K, H, W) float32 — per-pixel confidence
    conf_masks: np.ndarray | None = field(default=None, repr=False)  # (K, H, W) bool — conf >= threshold
    conf_threshold: float | None = None  # percentile-derived confidence cutoff

    # VGGT-SLAM's default confidence percentile for thresholding dense points.
    _CONF_PCT = 25.0

    def __post_init__(self) -> None:
        """
        Derive conf_threshold from dense conf when none was given.
        """
        if self.conf is not None and self.conf.size > 0 and self.conf_threshold is None:
            self.conf_threshold = float(np.percentile(self.conf, self._CONF_PCT)) + 1e-6

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
            self.conf_threshold = float(np.percentile(conf, self._CONF_PCT)) + 1e-6

    ########################################
    ###### Graph-corrected world reads #####
    ########################################

    def get_points_in_world_frame(self, graph: PoseGraph, skip_first: int = 0) -> np.ndarray:
        """
        Dense points in the world frame, corrected by the pose graph and masked by confidence.

        - frame i uses the optimized SL(4) homography of node `frame_start + i`
        - frame order and mask match `get_points_colors` for the same skip_first

        Args:
            graph: optimized PoseGraph holding one homography per frame.
            skip_first: leading frames to drop (overlap owned by the previous submap).

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
            H = graph.get_homography(self.frame_start + i).astype(np.float64)
            flat = self.points[i].reshape(-1, 3).astype(np.float64)  # (H*W, 3)
            hom = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float64)])
            out_hom = (H @ hom.T).T  # (H*W, 4)
            # Dehomogenize by /w, guarding near-zero w (projective plane-at-infinity).
            w = out_hom[:, 3:4]
            w = np.where(np.abs(w) < 1e-10, 1e-10, w)
            world = out_hom[:, :3] / w
            # Confidence mask this frame's points — same predicate/order as get_points_colors.
            mask = self.conf[i].reshape(-1) > self.conf_threshold
            out.append(world[mask])
        # skip_first can consume every frame (a tail submap that is pure overlap) — return empty.
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
        - stores R.T, not upstream's R: `decompose_camera` returns camera-to-world R

        Args:
            graph: optimized PoseGraph holding one homography per frame.

        Returns:
            World-to-cam poses, (N, 4, 4) float32.
        """
        # Inline import breaks the graph↔submap circular import.
        from .graph import decompose_camera

        poses = []
        for i in range(self.poses.shape[0]):
            K_4x4 = np.eye(4, dtype=np.float64)
            K_4x4[:3, :3] = self.intrinsics[i].astype(np.float64)
            proj = K_4x4 @ np.linalg.inv(graph.get_homography(self.frame_start + i))
            proj = proj / proj[-1, -1]
            _, rot, trans, _ = decompose_camera(proj[:3, :])
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot.T.astype(np.float32)
            pose[:3, 3] = trans.astype(np.float32)
            poses.append(pose)
        return np.stack(poses, axis=0)


def assert_world_to_cam(poses: np.ndarray, atol: float = 0.1) -> None:
    """
    Check that a window's poses are world-to-cam with frame 0 at the origin.

    - call on raw["extrinsic"] to catch a backend that returns cam-to-world

    Args:
        poses: window poses, (K, 4, 4).
        atol: absolute tolerance on poses[0] versus identity.

    Raises:
        ValueError: if poses is not (K, 4, 4) or poses[0] is not near identity.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (K, 4, 4), got {poses.shape}")
    if not np.allclose(poses[0], np.eye(4, dtype=poses.dtype), atol=atol):
        raise ValueError(
            f"Pose convention violation: expected poses[0] ≈ eye(4) (world-to-cam, "
            f"first frame at origin), got:\n{poses[0]}\n"
            "If your backend outputs cam-to-world, invert before creating Submap."
        )
