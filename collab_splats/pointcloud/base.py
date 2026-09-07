"""
The sparse reconstruction contract every pointcloud backend returns.

- PointcloudResult wraps a pycolmap.Reconstruction plus the caller's frame order
- points / colors / extrinsics / intrinsics derive from it on each access
- BasePointcloudCreator is the one abstract method every backend implements
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


@dataclass
class PointcloudResult:
    """
    Sparse reconstruction output: a pycolmap.Reconstruction plus its canonical frame order.

    - ``reconstruction`` holds the cameras, images, and 3D points; ``points``/``colors``/
      ``extrinsics``/``intrinsics`` are derived from it on each access.
    - ``image_paths`` defines the ordering of ``extrinsics``/``intrinsics``.
    """

    reconstruction: pycolmap.Reconstruction
    image_paths: list[Path]

    @classmethod
    def from_colmap(cls, colmap_dir: Path, image_paths: list[Path]) -> PointcloudResult:
        """
        Load a written COLMAP model from `<colmap_dir>/sparse/0` in the caller's frame order.

        - names are matched on `Path.name`, the key `extrinsics`/`intrinsics` also use
        - a model may register more images than requested; the extras stay in `reconstruction`
          but are absent from `extrinsics`/`intrinsics`, which follow `image_paths`

        Args:
            colmap_dir: directory holding `sparse/0`.
            image_paths: the canonical ordering; every entry must be registered in the model.

        Returns:
            A PointcloudResult over the loaded reconstruction.

        Raises:
            ValueError: when any requested frame is not registered in the model.
        """
        # Every backend writes its model to <colmap_dir>/sparse/0
        sparse_dir = colmap_dir / "sparse" / "0"
        recon = pycolmap.Reconstruction(sparse_dir)

        # Requested names are a contract nothing enforces at write time
        # - unchecked, a mismatch surfaces as a bare KeyError from .extrinsics several
        #   frames into a downstream stage
        # - that error names neither of the two artifacts that disagree
        registered = {img.name for img in recon.images.values()}
        missing = [p.name for p in image_paths if p.name not in registered]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(image_paths)} requested frames are not registered in "
                f"{sparse_dir} (first: {missing[0]}); the model registers {len(registered)} images."
            )

        return cls(reconstruction=recon, image_paths=list(image_paths))

    def write_ply(self, path: Path) -> None:
        """
        Write this reconstruction's point set as a binary little-endian PLY.

        Args:
            path: destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.reconstruction.export_PLY(path)

        # Three call sites write into two directories — the log is the only record of which
        # file got how many points.
        logger.info("wrote %s (%d points)", path, self.reconstruction.num_points3D())

    @property
    def points(self) -> np.ndarray:
        """
        (P, 3) float32 world XYZ of the tracked sparse point set.

        - order is `reconstruction.points3D` iteration order, NOT sorted by point3D_id
        - to map a row back to an id, pair it with `list(reconstruction.points3D.keys())`,
          which iterates in the same order as the `.values()` this builds from — the keep-mask
          zip in Reconstructor's pointcloud stage does exactly that
        - `sorted(reconstruction.points3D)` misaligns the rows and silently touches the wrong points
        - `depth_align._tracked_point3d_ids` DOES return sorted ids: a different convention on a
          different result set, so never carry one over to the other
        - P is the filtered sparse set, smaller than FeedforwardResult.points, which holds all
          model output including untracked points
        - recomputed on each access, so it always reflects current reconstruction state

        Returns:
            (P, 3) float32 array; (0, 3) when the reconstruction has no points.
        """
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([p.xyz for p in pts3d.values()], dtype=np.float32)

    @property
    def colors(self) -> np.ndarray:
        """
        (P, 3) uint8 RGB, in the same order as points.

        Returns:
            (P, 3) uint8 array; (0, 3) when the reconstruction has no points.
        """
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array([p.color for p in pts3d.values()], dtype=np.uint8)

    @property
    def extrinsics(self) -> np.ndarray:
        """
        (N, 4, 4) float32 w2c transforms, ordered by image_paths.

        - convention: x_cam = E @ x_world, homogeneous
        - OpenCV camera axes (X right, Y down, Z into the scene); world axes follow COLMAP (Y-down)

        Returns:
            (N, 4, 4) float32 array; (0, 4, 4) when image_paths is empty.
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
        """
        (N, 3, 3) float32 K matrices, PINHOLE/linear part only, ordered by image_paths.

        - K[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        - distortion params are NOT captured here
        - for distortion-correct projection read `reconstruction.cameras[id]` directly

        Returns:
            (N, 3, 3) float32 array; (0, 3, 3) when image_paths is empty.
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
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            result.append(K)
        return np.stack(result) if result else np.zeros((0, 3, 3), dtype=np.float32)


class BasePointcloudCreator(ABC):
    """
    Abstract base every pointcloud backend implements.

    - one method: reconstruct(image_dir, output_dir) -> PointcloudResult
    - concrete backends resolve through the registry in this package's __init__
    """

    @abstractmethod
    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        """
        Images in image_dir -> sparse pointcloud + camera poses written to output_dir.

        Produces:

        - `{output_dir}/colmap/sparse/0/{cameras,images,points3D}.bin`
        - `{output_dir}/sparse_pc.ply` — binary little-endian, via PointcloudResult.write_ply

        Args:
            image_dir: directory of input frames.
            output_dir: directory the model and PLY are written into.

        Returns:
            The reconstruction as a PointcloudResult.

        Raises:
            RuntimeError: when reconstruction fails.
            FileNotFoundError: when image_dir does not exist.
        """
        ...
