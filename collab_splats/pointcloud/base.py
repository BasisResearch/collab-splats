# collab_splats/pointcloud/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


class CoordinateFrame(str, Enum):
    COLMAP = "colmap"         # w2c, OpenCV axes, world -Y up
    NERFSTUDIO = "nerfstudio" # c2w, OpenGL axes, world +Z up


@dataclass
class PointcloudResult:
    points: np.ndarray                    # (N, 3) float32, world XYZ
    colors: np.ndarray                    # (N, 3) uint8, RGB
    confidence: np.ndarray | None = None  # (N,) float32 — feedforward only
    camera_poses: np.ndarray | None = None       # (M, 4, 4) float32
    camera_intrinsics: np.ndarray | None = None  # (M, 3, 3) float32, K per image
    colmap_reconstruction: Any | None = None     # pycolmap.Reconstruction — BA + pycolmap API
    frame: CoordinateFrame = CoordinateFrame.NERFSTUDIO
    world_transform: np.ndarray | None = None
    # (3, 4) applied_transform: COLMAP world → nerfstudio world.
    # Matches transforms.json["applied_transform"].
    # None when keep_original_world_coordinate=True.


class BasePointcloudCreator(ABC):
    @abstractmethod
    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        """images in image_dir → sparse pointcloud + camera poses written to output_dir.

        Produces:
            {output_dir}/colmap/sparse/0/{cameras,images,points3D}.bin
            {output_dir}/transforms.json
            {output_dir}/sparse_pc.ply

        Raises:
            RuntimeError: if reconstruction fails
            FileNotFoundError: if image_dir does not exist
        """
        ...

    def _write_transforms(self, sparse_dir: Path, output_dir: Path) -> None:
        from nerfstudio.process_data.colmap_utils import colmap_to_json
        colmap_to_json(recon_dir=sparse_dir, output_dir=output_dir)


# Standard COLMAP world (-Y up) → nerfstudio world (+Z up) rotation, stored as (3, 4).
# Matches transforms.json["applied_transform"] for standard COLMAP data.
_WORLD_TRANSFORM = np.array([
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0, -1, 0, 0],
], dtype=np.float32)


def _colmap_recon_to_result(
    recon: pycolmap.Reconstruction,
    confidence: np.ndarray | None = None,
) -> PointcloudResult:
    """Convert pycolmap.Reconstruction to PointcloudResult in nerfstudio world frame.

    Applies two transforms to each camera pose:
      A — OpenCV → OpenGL camera axes: c2w[:3, 1:3] *= -1
      B — COLMAP world (-Y) → nerfstudio world (+Z): row-swap + negate
    """
    pts3d = recon.points3D
    if pts3d:
        points = np.array([p.xyz for p in pts3d.values()], dtype=np.float32)
        colors = np.array([p.color for p in pts3d.values()], dtype=np.uint8)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)

    poses = []
    intrinsics = []
    for image in recon.images.values():
        if not image.registered:
            continue
        w2c_34 = image.cam_from_world.matrix()            # (3, 4)
        w2c = np.vstack([w2c_34, [0.0, 0.0, 0.0, 1.0]])  # (4, 4)
        c2w = np.linalg.inv(w2c)

        # Transform A: OpenCV → OpenGL camera axes
        c2w[:3, 1:3] *= -1

        # Transform B: COLMAP world (-Y up) → nerfstudio world (+Z up)
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        poses.append(c2w.astype(np.float32))

        cam = recon.cameras[image.camera_id]
        fx = getattr(cam, "focal_length_x", None) or cam.focal_length
        fy = getattr(cam, "focal_length_y", None) or cam.focal_length
        cx, cy = cam.principal_point_x, cam.principal_point_y
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrinsics.append(K)

    return PointcloudResult(
        points=points,
        colors=colors,
        confidence=confidence,
        camera_poses=np.stack(poses) if poses else None,
        camera_intrinsics=np.stack(intrinsics) if intrinsics else None,
        colmap_reconstruction=recon,
        frame=CoordinateFrame.NERFSTUDIO,
        world_transform=_WORLD_TRANSFORM.copy(),
    )
