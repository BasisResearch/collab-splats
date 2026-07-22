# Ported from VGGT-SLAM (github.com/MIT-SPARK/VGGT-SLAM), adapted for collab-splats.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch  # TODO(spec-2): torch dep in submap violates coupling rule — fix when Submap.frames type changes


@dataclass
class Submap:
    submap_id: int
    poses: np.ndarray  # (K, 4, 4) float32 — world-to-cam homogeneous
    intrinsics: np.ndarray  # (K, 3, 3) float32 — camera intrinsics
    retrieval_vectors: torch.Tensor  # (K, D) float32 — DINO-SALAD global descriptors
    image_paths: list[Path]
    frames: torch.Tensor | None = None  # (K, 3, H, W) — raw image tensors on CPU
    is_lc_submap: bool = False
    frame_start: int = 0  # global index of this submap's first frame in the full sequence
    raw_outputs: dict | None = field(default=None, repr=False)  # raw _forward() dict, for merging
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
        """Derive conf_threshold from dense conf when not explicitly provided."""
        if self.conf is not None and self.conf.size > 0 and self.conf_threshold is None:
            self.conf_threshold = float(np.percentile(self.conf, self._CONF_PCT)) + 1e-6

    def set_dense_points(self, points: np.ndarray, colors: np.ndarray, conf: np.ndarray) -> None:
        """Set dense per-frame points/colors/conf and derive conf_threshold (mirrors __post_init__)."""
        self.points = points
        self.colors = colors
        self.conf = conf
        if conf is not None and conf.size > 0:
            self.conf_threshold = float(np.percentile(conf, self._CONF_PCT)) + 1e-6

    def get_world_points(self, H: np.ndarray | None = None) -> np.ndarray:
        """Return world_points in global frame.

        H=None: return local frame as-is (per-submap visualization).
        H=(4,4): apply SL(4) projective transform, dehomogenize by /w.
        Requires world_points in per-camera local frame (PR #39 constraint).
        """
        if self.world_points is None:
            raise ValueError(f"Submap {self.submap_id} has no world_points")
        pts = self.world_points  # (K, P, 3)
        if H is None:
            return pts.copy()
        H = np.array(H, dtype=np.float64)
        k, p, _ = pts.shape
        flat = pts.reshape(-1, 3).astype(np.float64)  # (K*P, 3)
        ones = np.ones((flat.shape[0], 1), dtype=np.float64)
        hom = np.hstack([flat, ones])  # (K*P, 4)
        out_hom = (H @ hom.T).T  # (K*P, 4)
        # Dehomogenize by /w, guarding against near-zero w (projective transform
        # can push points toward the plane at infinity).
        w = out_hom[:, 3:4]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        return (out_hom[:, :3] / w).reshape(k, p, 3).astype(np.float32)

    def get_poses_world(self, H: np.ndarray | None = None) -> np.ndarray:
        """Return (K, 4, 4) poses in global frame.

        H=None: return self.poses unchanged (local, first pose ≈ identity).
        H=(4,4): apply corrected anchor H to each frame's local pose.
        """
        if H is None:
            return self.poses.copy()
        H = np.array(H, dtype=np.float64)
        return np.stack([(H @ p.astype(np.float64)).astype(np.float32) for p in self.poses])

    ########################################
    ###### Graph-corrected world reads #####
    ########################################

    def filter_data_by_confidence(self, data: np.ndarray) -> np.ndarray:
        """Boolean-index a per-frame (K, H, W, ...) array by conf > conf_threshold."""
        return data[self.conf > self.conf_threshold]

    def get_points_in_world_frame(self, graph) -> np.ndarray:
        """Return (M, 3) graph-corrected, confidence-masked dense points in world frame.

        Per frame i: apply the optimized SL(4) homography for node ``frame_start + i``
        to this frame's dense points, dehomogenize by /w, then keep only points with
        ``conf > conf_threshold``. Frame order + conf mask match get_points_colors.
        """
        if self.points is None:
            raise ValueError(f"Submap {self.submap_id} has no dense points")
        if self.conf is None:
            raise ValueError(f"Submap {self.submap_id} has no conf; cannot compute world-frame points")
        out = []
        for i in range(self.points.shape[0]):
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
        return np.vstack(out).astype(np.float32)

    def get_points_colors(self) -> np.ndarray:
        """Return (M, 3) per-point RGB, conf-masked to align with get_points_in_world_frame."""
        if self.conf is None:
            raise ValueError(f"Submap {self.submap_id} has no conf; cannot compute world-frame points")
        return self.filter_data_by_confidence(self.colors).reshape(-1, 3)

    def get_all_poses_world(self, graph) -> np.ndarray:
        """Return (S, 4, 4) world-to-cam poses via K @ inv(H_opt) → decompose_camera.

        Must produce the SAME poses as PoseGraph.extract_extrinsics for these frames.
        decompose_camera implements only VGGT-SLAM's no_inverse=True branch (R is
        camera-to-world, t = inv(K) @ P[:,3]), so we store R.T (world-to-cam) — NOT R
        as upstream get_all_poses_world does — mirroring extract_extrinsics' convention.
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
    """Assert poses[0] is approximately identity (world-to-cam convention, frame-0 at origin).

    VGGT and MapAnything return extrinsics where the first frame of each inference window
    is at identity. This is the world-to-cam convention we use internally.
    Call this after extracting raw["extrinsic"] to guard against backend mismatch.

    Raises:
        ValueError: if poses[0] is not approximately identity.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (K, 4, 4), got {poses.shape}")
    if not np.allclose(poses[0], np.eye(4, dtype=poses.dtype), atol=atol):
        raise ValueError(
            f"Pose convention violation: expected poses[0] ≈ eye(4) (world-to-cam, "
            f"first frame at origin), got:\n{poses[0]}\n"
            "If your backend outputs cam-to-world, invert before creating Submap."
        )
