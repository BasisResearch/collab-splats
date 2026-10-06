"""
The reconstruction contract every pointcloud backend returns.

- PointcloudResult: points, cameras and dense per-frame arrays, saved to and loaded from zarr
- two Ks per frame: `intrinsics` on the full-res frame, `model_intrinsics` on the model grid
- COLMAP is export-only (to_colmap), never loaded
- BasePointcloudCreator: create_pointcloud template; subclasses implement _reconstruct
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import torch
import zarr

from collab_splats.geometry.projection import unproject_frames
from collab_splats.geometry.transforms import (
    extract_intrinsics,
    rescale_intrinsics,
    shift_intrinsics,
)
from collab_splats.pointcloud.utils import clean_pointcloud
from collab_splats.preproc import frames
from collab_splats.utils.colmap import write_colmap_reconstruction
from collab_splats.utils.io import LZ4
from collab_splats.utils.torch_utils import to_numpy

logger = logging.getLogger(__name__)


########################################################################
# Result type
########################################################################


@dataclass
class PointcloudResult:
    """
    Typed output of every pointcloud backend, grouped by what each field describes.

    - P points, N images; each image has one pose and two Ks
    - model grid: the crop + resize from each original frame to the H x W model grid
    - dense: optional per-pixel maps on the model grid
    - arrays are float32 unless an entry says otherwise
    - `intrinsics=None` derives the full-res K from `model_intrinsics`
    - a `replace` that changes `model_intrinsics` must also pass `intrinsics=None`

    Args:
        points: (P, 3) world-space XYZ.
        colors: (P, 3) uint8 RGB in [0, 255].
        extrinsics: (N, 4, 4) w2c poses, OpenCV axes.
        intrinsics: (N, 3, 3) K on the full-res original frame.
        model_intrinsics: (N, 3, 3) K on the model grid; matches depth, world_points, pixel_indices.
        image_paths: N source image paths, in extrinsics order.
        original_coords: (N, 6) [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], crop box in original pixels.
        model_width: model grid width W in pixels.
        model_height: model grid height H in pixels.
        images: (N, 3, H, W) RGB in [0, 1].
        confidence: (N, H, W) per-pixel confidence.
        world_points: (N, H, W, 3) world point per pixel.
        depth: (N, H, W) z-depth.
        pixel_indices: (P, 3) int32 [frame, row, col] source pixel of each point.
    """

    # The point cloud
    points: np.ndarray
    colors: np.ndarray

    # Camera poses, intrinsics and image paths
    extrinsics: np.ndarray
    intrinsics: np.ndarray | None
    model_intrinsics: np.ndarray
    image_paths: list[Path]

    # How the model's input grid maps back to the full-res frames
    original_coords: np.ndarray
    model_width: int
    model_height: int

    # Optional per-pixel outputs
    images: "torch.Tensor | None" = None
    confidence: "torch.Tensor | None" = None
    world_points: "np.ndarray | None" = None
    depth: "np.ndarray | None" = None
    pixel_indices: "np.ndarray | None" = None

    # Path of the zarr store this result was loaded from
    _zarr_path: "Path | None" = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Full-res K from the model grid, when the caller did not supply it.
        """
        if self.intrinsics is not None:
            return

        # Get each frame's crop size on the full-res frame
        box = np.asarray(self.original_coords)
        crop_hw = np.stack([box[:, 3] - box[:, 1], box[:, 2] - box[:, 0]], axis=-1)

        # Scale K up to the crop size, then shift it by the crop's top-left corner
        K = rescale_intrinsics(self.model_intrinsics, (self.model_height, self.model_width), crop_hw)
        K = shift_intrinsics(K, box[:, :2])
        self.intrinsics = K.astype(np.float32)

    def save_zarr(self, path: Path, extra_attrs: dict | None = None) -> None:
        """
        Save to a zarr v3 store with lz4 compression.

        - dense arrays are chunked by frame
        - images are written, but load_zarr skips them by default
        - world_points are not written: load_zarr unprojects them from depth

        Args:
            path: directory for the zarr store, created if absent.
            extra_attrs: provenance, e.g. method and backend, merged into store.attrs.
        """
        store = zarr.open(str(path), mode="w")

        # Store sizes, image paths and extra metadata as attributes
        store.attrs["image_paths"] = [str(p) for p in self.image_paths]
        store.attrs["model_width"] = self.model_width
        store.attrs["model_height"] = self.model_height

        # Record which method and backend made this store
        if extra_attrs:
            for key, value in extra_attrs.items():
                store.attrs[key] = value

        # Write the required arrays, including both intrinsics
        for name, arr in (
            ("points", self.points),
            ("colors", self.colors),
            ("extrinsics", self.extrinsics),
            ("intrinsics", self.intrinsics),
            ("model_intrinsics", self.model_intrinsics),
            ("original_coords", self.original_coords),
        ):
            store.create_array(name, data=arr, chunks=arr.shape, compressors=LZ4)

        # Write the optional point-to-pixel map as one chunk
        if self.pixel_indices is not None:
            store.create_array(
                "pixel_indices", data=self.pixel_indices, chunks=self.pixel_indices.shape, compressors=LZ4
            )

        # Write the optional per-pixel arrays as numpy, one chunk per frame
        for name, arr in (
            ("depth", self.depth),
            ("confidence", self.confidence),
            ("images", self.images),
        ):
            if arr is None:
                continue

            arr = to_numpy(arr)
            store.create_array(name, data=arr, chunks=(1, *arr.shape[1:]), compressors=LZ4)

    @classmethod
    def load_zarr(
        cls,
        path: Path,
        load_images: bool = False,
        load_depth: bool = True,
        load_world_points: bool = True,
        load_confidence: bool = True,
        load_pixel_indices: bool = True,
    ) -> PointcloudResult:
        """
        Load from a zarr v3 store written by save_zarr().

        - dense arrays load by default; skip the ones you do not read, they are large
        - world_points are unprojected from depth under the stored extrinsics and model-grid K
        - confidence and images come back as torch tensors

        Args:
            path: directory of the zarr store.
            load_images: restore the (N, 3, H, W) images, needed to lift features after a load.
            load_depth: decode the depth maps.
            load_world_points: unproject the per-pixel world points; needs depth in the store.
            load_confidence: decode the confidence maps.
            load_pixel_indices: decode the per-point source pixels.

        Returns:
            PointcloudResult; skipped or absent fields are None.
        """
        store = zarr.open(str(path), mode="r")
        attrs = dict(store.attrs)

        # Read the required arrays
        pts3d = store["points"][:]
        colors = store["colors"][:]
        extrinsics = store["extrinsics"][:]
        intrinsics = store["intrinsics"][:]
        model_intrinsics = store["model_intrinsics"][:]
        original_coords = store["original_coords"][:]
        image_paths = [Path(p) for p in attrs["image_paths"]]
        model_width = int(attrs["model_width"])
        model_height = int(attrs["model_height"])

        # Read the optional arrays, or None when missing or not requested
        pixel_indices = store["pixel_indices"][:] if (load_pixel_indices and "pixel_indices" in store) else None
        depth = store["depth"][:] if ((load_depth or load_world_points) and "depth" in store) else None
        confidence = torch.from_numpy(store["confidence"][:]) if (load_confidence and "confidence" in store) else None

        # Rebuild world points from depth, then drop depth if it was read only for them
        world_points = None
        if load_world_points and depth is not None:
            world_points = unproject_frames(depth, extrinsics, model_intrinsics)

        if not load_depth:
            depth = None

        # Load images only when asked, since they are large
        images = torch.from_numpy(store["images"][:]) if load_images and "images" in store else None

        result = cls(
            points=pts3d,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            model_intrinsics=model_intrinsics,
            original_coords=original_coords,
            image_paths=image_paths,
            model_width=model_width,
            model_height=model_height,
            pixel_indices=pixel_indices,
            world_points=world_points,
            depth=depth,
            images=images,
            confidence=confidence,
        )
        result._zarr_path = Path(path)

        return result

    def reproject(self) -> PointcloudResult:
        """
        Points and world_points recomputed from depth and source pixels under the current extrinsics.

        - uses model_intrinsics: depth and pixel_indices are on the model grid too
        - points stay row-aligned with colors

        Returns:
            A new PointcloudResult with points and world_points replaced.

        Raises:
            ValueError: depth or pixel_indices is absent, or depth is not (N, H, W).
        """
        if self.depth is None or self.pixel_indices is None:
            raise ValueError(
                "reproject() requires depth and pixel_indices; load via load_zarr() "
                "or ensure the creator's _postprocess populated both fields."
            )

        if self.depth.ndim != 3:
            raise ValueError(f"depth must be (N, H, W), got {self.depth.shape}")

        # Unproject every frame under the current poses, then read each point's source pixel
        world_points = unproject_frames(self.depth, self.extrinsics, self.model_intrinsics)
        frame, row, col = self.pixel_indices.T
        points = world_points[frame, row, col]

        return replace(self, points=points, world_points=world_points)

    def select_points(self, mask: np.ndarray) -> PointcloudResult:
        """
        Subset of the points; per-frame fields are untouched.

        - one mask selects points, colors and pixel_indices, so they stay row-aligned

        Args:
            mask: (P,) bool, True for the points kept.

        Returns:
            A new PointcloudResult holding the kept points.
        """
        pixel_indices = None if self.pixel_indices is None else self.pixel_indices[mask]

        return replace(self, points=self.points[mask], colors=self.colors[mask], pixel_indices=pixel_indices)

    def to_colmap(self) -> pycolmap.Reconstruction:
        """
        In-memory pycolmap model of this result on the full-res frames.

        - one PINHOLE camera and one image per frame, K from `intrinsics`
        - points have empty tracks: no 2D-3D matches, so not usable for COLMAP BA

        Returns:
            pycolmap.Reconstruction with cameras, images and 3D points.

        Raises:
            TypeError: colors is not uint8.
        """
        if self.colors.dtype != np.uint8:
            raise TypeError(f"colors must be uint8, got {self.colors.dtype}")

        recon = pycolmap.Reconstruction()

        # Take the world-to-camera rotation and translation rows from each pose
        exts = self.extrinsics[:, :3, :]

        # Add every point without any observations
        for xyz, rgb in zip(self.points, self.colors):
            recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb)

        # Add one camera and one image per frame at full resolution
        for i, path in enumerate(self.image_paths):
            camera_id = i + 1
            image_id = i + 1

            camera = pycolmap.Camera(
                model="PINHOLE",
                width=int(self.original_coords[i][4]),
                height=int(self.original_coords[i][5]),
                params=extract_intrinsics(self.intrinsics[i]),
                camera_id=camera_id,
            )
            # Register the camera with its own rig, which pycolmap 4 requires before adding an image
            recon.add_camera_with_trivial_rig(camera)

            # Build the world-to-camera pose
            cam_from_world = pycolmap.Rigid3d(exts[i].astype(np.float64))

            # Add the image together with its pose
            image = pycolmap.Image(name=path.name, camera_id=camera_id, image_id=image_id)
            recon.add_image_with_trivial_frame(image, cam_from_world)

        return recon

    def write_ply(self, path: Path) -> None:
        """
        Points and colors as a binary PLY file.

        Args:
            path: destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write points and 0-255 colors through Open3D, which stores colors in [0, 1]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(self.points, dtype=np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(self.colors, dtype=np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd)

        # Log the output file and point count
        logger.info("wrote %s (%d points)", path, len(self.points))


########################################################################
# Creator contract
########################################################################


@dataclass
class BasePointcloudCreator(ABC):
    """
    Images directory to a cleaned PointcloudResult and an optional COLMAP model.

    - subclasses implement _reconstruct; one with its own COLMAP model overrides _colmap_model
    - every result gets outlier removal when clean, then the max_points cap

    Attributes:
        max_points: point cap, drawn at random after outlier removal.
        clean: run statistical outlier removal before the cap.
    """

    max_points: int = 500_000
    clean: bool = True

    def create_pointcloud(self, images_dir: Path, out_dir: Path, model_dir: Path | None = None) -> PointcloudResult:
        """
        Reconstruct the frames in images_dir, clean and cap, and optionally export a COLMAP model.

        - a previous model at model_dir is deleted first, so a failed run leaves none behind

        Args:
            images_dir: directory of keyframe images.
            out_dir: run directory, created if absent.
            model_dir: COLMAP model directory; None skips the export.

        Returns:
            The cleaned and capped PointcloudResult.

        Raises:
            FileNotFoundError: no frames to reconstruct, handed off or in images_dir.
        """
        paths = self._list_frames(images_dir)

        if not paths:
            raise FileNotFoundError(f"no images in {images_dir}")

        # Delete the old COLMAP model
        if model_dir is not None:
            shutil.rmtree(model_dir, ignore_errors=True)

        # Run the reconstruction into the output directory
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self._reconstruct(paths, out_dir)

        # Remove outliers and cap the number of points
        result = clean_pointcloud(result, remove_outliers=self.clean, max_points=self.max_points)
        logger.info("pointcloud: %d pts after clean + cap", len(result.points))

        # Export a COLMAP model if a directory was given
        if model_dir is not None:
            write_colmap_reconstruction(self._colmap_model(result), Path(model_dir))

        return result

    def _list_frames(self, images_dir: Path) -> list[Path]:
        """
        Frame image paths to reconstruct, in filename order, listed from images_dir.
        """
        return frames.frame_paths(images_dir)

    def _colmap_model(self, result: PointcloudResult) -> pycolmap.Reconstruction:
        """
        COLMAP model exported for the cleaned result: the result itself, trackless.
        """
        return result.to_colmap()

    @abstractmethod
    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        Frames to a PointcloudResult before the clean and cap.
        """
        ...
