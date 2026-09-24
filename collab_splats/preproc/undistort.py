"""
Camera self-calibration and undistortion for the images/ directory.

- pycolmap self-calibrates one shared OPENCV camera; cv2 remaps the pixels
- framing is COLMAP's (pycolmap.undistort_camera): focal kept, canvas resized to the corners
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from collab_splats.preproc.frames import frame_paths

logger = logging.getLogger(__name__)


########################################
# Calibration (pycolmap self-calibration)
########################################


def calibrate_camera(
    images_dir: Path,
    *,
    max_frames: int = 60,
    min_images: int = 8,
    min_registered_frac: float = 0.6,
    num_threads: int = 8,
) -> pycolmap.Camera:
    """
    Self-calibrate one shared OPENCV camera from a scene's images.

    Args:
        images_dir: the scene's images/ directory, read in place.
        max_frames: how many evenly spaced images to calibrate from.
        min_images: fewer images cannot constrain k1 k2 p1 p2; raise below it.
        min_registered_frac: share of the subset that must register; raise below it.
        num_threads: SIFT threads; the pycolmap default (one per host core) can OOM.

    Returns:
        The pycolmap.Camera (model OPENCV) of the largest reconstruction, carrying
        fx fy cx cy and k1 k2 p1 p2 at the images' own resolution.

    Raises:
        ValueError: fewer than min_images images.
        RuntimeError: no model, or too few images registered.
    """
    paths = frame_paths(images_dir)
    if len(paths) < min_images:
        raise ValueError(f"calibrate_camera needs at least {min_images} images, found {len(paths)} in {images_dir}")

    # Evenly spaced subset: calibration wants baseline, not every frame
    idxs = np.unique(np.linspace(0, len(paths) - 1, min(max_frames, len(paths))).round().astype(int))
    names = [paths[int(i)].name for i in idxs]

    # The database and sparse output are scratch; the images are read from images_dir
    # in place, so nothing stages a second copy of the pixels
    with tempfile.TemporaryDirectory(prefix="calibrate_camera_") as tmp:
        database = Path(tmp) / "database.db"
        sparse = Path(tmp) / "sparse"
        sparse.mkdir()

        # One shared OPENCV camera across the subset; the mapper refines k1 k2 p1 p2.
        # camera_model lives on reader_options, not as a top-level kwarg (pycolmap 4.0.4 API).
        pycolmap.extract_features(
            database,
            images_dir,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(camera_model="OPENCV"),
            extraction_options=pycolmap.FeatureExtractionOptions(num_threads=num_threads),
        )
        pycolmap.match_exhaustive(
            database,
            matching_options=pycolmap.FeatureMatchingOptions(num_threads=num_threads),
        )
        reconstructions = pycolmap.incremental_mapping(database, images_dir, sparse)

        if not reconstructions:
            raise RuntimeError(
                f"calibrate_camera: no model registered from {len(names)} images in {images_dir} — "
                "the footage may be featureless or the motion degenerate"
            )

        # dict[int, Reconstruction] keyed by model id: take the largest, not key 0
        recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        registered = recon.num_reg_images()
        if registered < min_registered_frac * len(names):
            raise RuntimeError(
                f"calibrate_camera: only {registered} of {len(names)} images registered "
                f"(need {min_registered_frac:.0%}); the distortion params are not trustworthy"
            )

        camera = next(iter(recon.cameras.values()))

    logger.info(
        "calibrated %s from %d/%d images: %s",
        images_dir,
        registered,
        len(names),
        camera,
    )
    return camera


########################################
# Undistortion (COLMAP framing, cv2 remap)
########################################


def undistort_frames(frames_in: np.ndarray, camera: pycolmap.Camera) -> tuple[np.ndarray, pycolmap.Camera]:
    """
    Undistort a stack of frames onto COLMAP's undistorted framing.

    Args:
        frames_in: (N, H, W, 3) uint8; H, W must match the camera.
        camera: a distorted pycolmap.Camera from calibrate_camera.

    Returns:
        - (N, H', W', 3) uint8, same channel order in as out.
        - The PINHOLE camera for that canvas: focal preserved, canvas resized to the
          undistorted corners (grows for barrel distortion), so the center stays 1:1.

    Raises:
        ValueError: frames_in is not 4-D, or its H, W differ from the camera's.
    """
    frames_in = np.asarray(frames_in)
    if frames_in.ndim != 4 or frames_in.shape[1:3] != (camera.height, camera.width):
        raise ValueError(
            f"undistort_frames: frames are {frames_in.shape}, camera is (N, {camera.height}, {camera.width}, 3)"
        )

    # COLMAP picks the framing: focal fixed, canvas sized to hold the corners
    new_camera = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), camera)

    # cv2 moves the pixels: one dst->src map, built once, reused for every frame
    map1, map2 = cv2.initUndistortRectifyMap(
        camera.calibration_matrix(),
        np.asarray(camera.params[4:], dtype=np.float64),
        None,
        new_camera.calibration_matrix(),
        (new_camera.width, new_camera.height),
        cv2.CV_32FC1,
    )
    out = np.stack([cv2.remap(frame, map1, map2, cv2.INTER_LINEAR) for frame in frames_in])

    logger.info(
        "undistorted %d frames: %dx%d -> %dx%d, f=%.1f preserved",
        len(out),
        camera.width,
        camera.height,
        new_camera.width,
        new_camera.height,
        new_camera.focal_length_x,
    )
    return out, new_camera
