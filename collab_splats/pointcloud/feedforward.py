# collab_splats/pointcloud/feedforward.py
from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class BaseFeedforwardCreator(BasePointcloudCreator):
    """Template for feedforward (depth-estimation) pointcloud creators.

    Subclasses implement _run_inference() which writes binary COLMAP files to
    output_dir/colmap/sparse/0/ and returns the pycolmap.Reconstruction.
    Base class calls _write_transforms() once after _run_inference() completes.
    """

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        recon = self._run_inference(image_dir, output_dir)
        self._write_transforms(sparse_dir, output_dir)
        return _colmap_recon_to_result(recon)

    @abstractmethod
    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        """Run model inference, write binary to output_dir/colmap/sparse/0/, return Reconstruction."""
        ...


def build_pycolmap_reconstruction(
    pts3d: np.ndarray,
    colors: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: list[str],
    camera_model: str = "PINHOLE",
) -> pycolmap.Reconstruction:
    """Build pycolmap Reconstruction from pointcloud + camera data. No Point2D tracks."""
    recon = pycolmap.Reconstruction()
    exts = extrinsics[:, :3, :] if extrinsics.shape[1] == 4 else extrinsics
    colors_u8 = (
        colors if colors.dtype == np.uint8
        else (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    )

    for xyz, rgb in zip(pts3d, colors_u8):
        recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb)

    for i, name in enumerate(image_names):
        camera_id = i + 1
        image_id = i + 1
        frame_id = i + 1

        K = intrinsics[i]
        if camera_model == "PINHOLE":
            params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        else:  # SIMPLE_PINHOLE
            params = [(K[0, 0] + K[1, 1]) / 2.0, K[0, 2], K[1, 2]]
        camera = pycolmap.Camera(
            model=camera_model,
            width=image_width,
            height=image_height,
            params=params,
            camera_id=camera_id,
        )
        recon.add_camera(camera)

        # pycolmap 4.x: images require a Rig + Frame hierarchy to carry pose
        rig = pycolmap.Rig(rig_id=camera_id)
        sensor_id = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=camera_id)
        rig.add_ref_sensor(sensor_id)
        recon.add_rig(rig)

        image = pycolmap.Image(name=name, camera_id=camera_id, image_id=image_id)
        data_id = image.data_id

        R = exts[i, :3, :3].astype(np.float64)
        t = exts[i, :3, 3].astype(np.float64)
        rigid = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)

        frame = pycolmap.Frame(rig_id=camera_id, frame_id=frame_id, rig_from_world=rigid)
        frame.add_data_id(data_id)
        recon.add_frame(frame)

        image.frame_id = frame_id
        recon.add_image(image)

    return recon


def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: List[Path],
    original_image_sizes: np.ndarray,
    image_size: Tuple[int, int],
    shared_camera: bool = False,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> Any:
    """Rescale reconstruction from model resolution to original dimensions.

    This function is adapted from nerfstudio's vggt_utils module.
    It rescales camera intrinsics and image dimensions from the model's
    fixed resolution (e.g., 336x518 for MapAnything) to the original image sizes.

    Args:
        reconstruction: pycolmap Reconstruction object
        image_paths: List of Path objects for the images
        original_image_sizes: Array of shape (N, 6) with format:
            [top_left_x, top_left_y, crop_right, crop_bottom, original_width, original_height]
            For MapAnything (which resizes without cropping), use:
            [0, 0, model_width, model_height, original_width, original_height]
        image_size: Model image size as (width, height)
        shared_camera: Whether using a single shared camera for all images
        shift_point2d_to_original_res: Whether to shift point2D observations to original resolution
        verbose: Whether to print progress information

    Returns:
        Updated pycolmap Reconstruction object with rescaled cameras
    """
    if verbose:
        sample_image = original_image_sizes[0, -2:]
        original_width, original_height = sample_image
        print(
            f"Rescaling reconstruction from {image_size[0]}x{image_size[1]} "
            f"to original dimensions"
        )
        print(f"  Original image sizes (WxH): {int(original_width)}x{int(original_height)}")

    rescale_camera = True

    # Shared-camera state (computed once)
    shared_intrinsics = None
    shared_width = None
    shared_height = None

    for pyimageid in reconstruction.images:
        # Get image and camera objects
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]

        # Rename image to original name
        pyimage.name = image_paths[pyimageid - 1].name

        # Copy camera parameters
        pred_params = copy.deepcopy(pycamera.params)

        # Get original width/height and compute scale factors
        real_image_size = original_image_sizes[pyimageid - 1, -2:]
        scale_x = real_image_size[0] / image_size[0]
        scale_y = real_image_size[1] / image_size[1]

        # --------------------------------
        # Rescale camera intrinsics
        # --------------------------------
        # Non-shared: rescale every time
        # Shared: rescale exactly once
        if rescale_camera and (not shared_camera or shared_intrinsics is None):

            # Rescale focal length parameters
            if pycamera.model.name == "SIMPLE_PINHOLE":
                pred_params[0] *= max(scale_x, scale_y)
            elif pycamera.model.name in ("PINHOLE", "OPENCV", "RADIAL", "OPENCV_FISHEYE"):
                pred_params[0] *= scale_x  # fx
                pred_params[1] *= scale_y  # fy

            # Rescale principal point (cx, cy)
            pred_params[-2] *= scale_x
            pred_params[-1] *= scale_y

            # Apply back to camera object
            if shared_camera:
                # First image defines the shared camera
                shared_intrinsics = pred_params
                shared_width = int(real_image_size[0])
                shared_height = int(real_image_size[1])

                pycamera.params = shared_intrinsics
                pycamera.width = shared_width
                pycamera.height = shared_height
            else:
                pycamera.params = pred_params
                pycamera.width = int(real_image_size[0])
                pycamera.height = int(real_image_size[1])

        # --------------------------------
        # Ensure shared camera is consistent
        # --------------------------------
        if shared_camera and shared_intrinsics is not None:
            pycamera.params = shared_intrinsics
            pycamera.width = shared_width
            pycamera.height = shared_height

        # --------------------------------
        # Shift point2D if requested
        # --------------------------------
        if shift_point2d_to_original_res:
            top_left = original_image_sizes[pyimageid - 1, :2]

            scale_x = real_image_size[0] / image_size[0]
            scale_y = real_image_size[1] / image_size[1]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * np.array([scale_x, scale_y])

    if verbose:
        print("Rescaled reconstruction to original dimensions")

    return reconstruction


@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    """Pointcloud via MapAnything feedforward depth + pose estimation.

    No feature matching — depth and pose estimated directly by the network.
    Populates PointcloudResult.confidence via inference filtering.
    """

    model_name: str = "facebook/map-anything"
    confidence_percentile: float = 35.0   # keep points above this percentile (top 65%)
    use_multiview_confidence: bool = True  # multi-view depth consistency filter
    minibatch_size: int = 1               # frames processed at once (1 = most memory-efficient)

    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        from ._mapanything import run_mapanything

        pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = run_mapanything(
            image_dir, self.model_name,
            confidence_percentile=self.confidence_percentile,
            use_multiview_confidence=self.use_multiview_confidence,
            minibatch_size=self.minibatch_size,
        )
        recon = build_pycolmap_reconstruction(
            pts3d, colors, extrinsics, intrinsics, model_w, model_h,
            [p.name for p in image_paths],
        )
        recon = _rescale_reconstruction_to_original_dimensions(
            recon, image_paths, original_coords, (model_w, model_h)
        )
        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        recon.write_binary(str(sparse_dir))
        return pycolmap.Reconstruction(str(sparse_dir))


@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-X feedforward pose + depth estimation.

    use_global_alignment=False (default): per-frame depth only.
    use_global_alignment=True: cross-camera alignment via _run_global_alignment().
    Validate alignment before enabling — see _vggt.py notes.
    """

    use_global_alignment: bool = False
    model_name: str = "facebook/vggt"

    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        from ._vggt import run_vggt

        colmap_dir = output_dir / "colmap"
        pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = run_vggt(
            image_dir, colmap_dir=colmap_dir, model_name=self.model_name,
            use_global_alignment=self.use_global_alignment,
        )
        recon = build_pycolmap_reconstruction(
            pts3d, colors, extrinsics, intrinsics, model_w, model_h,
            [p.name for p in image_paths],
            camera_model="SIMPLE_PINHOLE",
        )
        recon = _rescale_reconstruction_to_original_dimensions(
            recon, image_paths, original_coords, (model_w, model_h)
        )
        sparse_dir = colmap_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        recon.write_binary(str(sparse_dir))
        return pycolmap.Reconstruction(str(sparse_dir))
