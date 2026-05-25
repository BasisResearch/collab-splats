# collab_splats/pointcloud/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
import pycolmap

from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses


class CoordinateFrame(str, Enum):
    COLMAP = "colmap"         # w2c, OpenCV axes, world -Y up
    NERFSTUDIO = "nerfstudio" # c2w, OpenGL axes, world +Z up


@dataclass
class PointcloudResult:
    """Sparse reconstruction output: pycolmap.Reconstruction + scene metadata.

    reconstruction is the primary store for cameras, images, and 3D points.
    frame declares the coordinate system of the world origin in reconstruction.
    image_paths defines the canonical frame ordering for extrinsics/intrinsics.
    """

    reconstruction: pycolmap.Reconstruction          # primary — always set
    frame: CoordinateFrame                           # coord system of world origin
    image_paths: list[Path]                          # canonical frame ordering (N entries)
    confidence: np.ndarray | None = None             # (P,) float32 — feedforward per-point
    world_transform: np.ndarray | None = None        # (3, 4) applied COLMAP→nerfstudio axis swap

    @property
    def points(self) -> np.ndarray:
        """(P, 3) float32 world XYZ of the tracked sparse point set, ordered by point3D_id.

        P is the filtered sparse set — smaller than FeedforwardResult.points which
        contains all feedforward model output including untracked points.
        Recomputes on each access from reconstruction.points3D — always reflects
        current reconstruction state.
        """
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([p.xyz for p in pts3d.values()], dtype=np.float32)

    @property
    def colors(self) -> np.ndarray:
        """(P, 3) uint8 RGB, same order as points."""
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array([p.color for p in pts3d.values()], dtype=np.uint8)

    @property
    def extrinsics(self) -> np.ndarray:
        """(N, 4, 4) float32 w2c transforms, ordered by image_paths.

        Convention: x_cam = E @ x_world (homogeneous). OpenCV camera axes
        (X right, Y down, Z into scene). Frame is declared by self.frame.
        """
        name_to_image = {img.name: img for img in self.reconstruction.images.values()}
        result = []
        for path in self.image_paths:
            img = name_to_image[path.name]
            R = img.cam_from_world().rotation.matrix()
            t = img.cam_from_world().translation
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3] = t
            result.append(E)
        return np.stack(result) if result else np.zeros((0, 4, 4), dtype=np.float32)

    @property
    def intrinsics(self) -> np.ndarray:
        """(N, 3, 3) float32 K matrices (PINHOLE/linear part only), ordered by image_paths.

        K[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        Distortion params are NOT captured here. Access reconstruction.cameras[id]
        directly for the full pycolmap.Camera when distortion-correct projection is needed.
        """
        name_to_image = {img.name: img for img in self.reconstruction.images.values()}
        cameras = self.reconstruction.cameras
        result = []
        for path in self.image_paths:
            img = name_to_image[path.name]
            cam = cameras[img.camera_id]
            # Use pycolmap canonical accessors — work for all camera models
            # (SIMPLE_PINHOLE has one focal length; PINHOLE has fx/fy separately)
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float32)
            result.append(K)
        return np.stack(result) if result else np.zeros((0, 3, 3), dtype=np.float32)


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
