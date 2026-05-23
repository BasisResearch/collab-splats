from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch  # TODO(spec-2): torch dep in submap violates coupling rule — fix when Submap.frames type changes


@dataclass
class Submap:
    submap_id: int
    frames: torch.Tensor            # (K, 3, H, W) — raw image tensors on CPU
    poses: np.ndarray               # (K, 4, 4) float32 — world-to-cam homogeneous
    intrinsics: np.ndarray          # (K, 3, 3) float32 — camera intrinsics
    retrieval_vectors: torch.Tensor  # (K, D) float32 — DINO-SALAD global descriptors
    image_paths: list[Path]
    is_lc_submap: bool = False
    frame_start: int = 0            # global index of this submap's first frame in the full sequence
    raw_outputs: dict | None = field(default=None, repr=False)  # raw _forward() dict, for merging
    world_points: np.ndarray | None = field(default=None, repr=False)       # (K, P, 3) float32 — per-frame 3D points in local frame
    world_points_conf: np.ndarray | None = field(default=None, repr=False)  # (K, P) float32 — per-point confidence

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
        flat = pts.reshape(-1, 3).astype(np.float64)           # (K*P, 3)
        ones = np.ones((flat.shape[0], 1), dtype=np.float64)
        hom = np.hstack([flat, ones])                          # (K*P, 4)
        out_hom = (H @ hom.T).T                                # (K*P, 4)
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
        return np.stack([
            (H @ p.astype(np.float64)).astype(np.float32)
            for p in self.poses
        ])


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
