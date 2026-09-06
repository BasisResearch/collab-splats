"""
Camera calibration and undistortion at the images/ boundary.

pycolmap self-calibrates one shared OPENCV camera from the scene's images and cv2
moves the pixels. The camera IS a pycolmap.Camera — nothing round-trips through a
local dataclass mirroring the same numbers.

Every downstream consumer (feedforward backbones, InstantSfM SIFT, splat trainer,
localization DB export) assumes pinhole; undistorting once here fixes all of them.
COLMAP picks the undistorted framing (pycolmap.undistort_camera) and cv2 moves the
pixels; there is no crop, so the principal point cannot drift out of step with it.
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

# CPU SIFT threads: pycolmap 4.0.4 here is a CPU-only wheel (has_cuda False), so the
# default num_threads (-1) spawns one thread per HOST core — 96 on this machine — and
# per-thread RAM on 1920x1080 frames blows past the 46.6 GB container cgroup cap
# (measured: SIGKILL during calibration on a 300-frame GoPro scene). Mirrors
# pointcloud/sfm.py::_SIFT_NUM_THREADS.
_SIFT_NUM_THREADS = 8

# Below this share of the calibration subset the solve has not seen the lens
_MIN_REGISTERED_FRACTION = 0.6

# Two-view initialisation plus a margin; fewer images cannot constrain k1 k2 p1 p2
_MIN_CALIBRATION_IMAGES = 8


def calibrate_camera(images_dir: Path, *, max_frames: int = 60) -> pycolmap.Camera:
    """
    Self-calibrate one shared OPENCV camera from a scene's images.

    Args:
        images_dir: the scene's images/ directory, read in place.
        max_frames: how many evenly spaced images to calibrate from.

    Returns:
        The pycolmap.Camera (model OPENCV) of the largest reconstruction, carrying
        fx fy cx cy and k1 k2 p1 p2 at the images' own resolution.
    """
    paths = frame_paths(images_dir)
    if len(paths) < _MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"calibrate_camera needs at least {_MIN_CALIBRATION_IMAGES} images, found {len(paths)} in {images_dir}"
        )

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
            extraction_options=pycolmap.FeatureExtractionOptions(num_threads=_SIFT_NUM_THREADS),
        )
        pycolmap.match_exhaustive(
            database,
            matching_options=pycolmap.FeatureMatchingOptions(num_threads=_SIFT_NUM_THREADS),
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
        if registered < _MIN_REGISTERED_FRACTION * len(names):
            raise RuntimeError(
                f"calibrate_camera: only {registered} of {len(names)} images registered "
                f"(need {_MIN_REGISTERED_FRACTION:.0%}); the distortion params are not trustworthy"
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
        - The PINHOLE camera for that canvas: focal preserved, canvas grown to hold
          the corners, so the centre stays 1:1 and nothing is resampled down to fit.
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
