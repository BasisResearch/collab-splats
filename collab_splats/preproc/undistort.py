"""
Camera undistortion at the frames.zarr boundary: OPENCV-model self-calibration
(pycolmap) + splatfacto-exact cv2 undistort/crop.

Every downstream consumer (feedforward backbones, InstantSfM SIFT, splat
trainer, localization DB export) assumes pinhole; undistorting once here fixes
all of them. Port of nerfstudio full_images_datamanager._undistort_image
(nerfstudio @ 50e0e3c): getOptimalNewCameraMatrix(alpha=0) -> cv2.undistort ->
ROI crop -> K rewritten by the crop offset.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


########################################
# Profile
########################################


@dataclass(frozen=True)
class DistortionProfile:
    """
    OPENCV camera model (k1 k2 p1 p2) + calibrated pinhole K + input dims.
    """

    k1: float
    k2: float
    p1: float
    p2: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_dict(self) -> dict:
        """
        JSON-serialisable dict (frames.zarr provenance payload).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DistortionProfile:
        """
        Inverse of to_dict.
        """
        return cls(**d)

    @property
    def K(self) -> np.ndarray:
        """
        3x3 pinhole intrinsics of the calibration.
        """
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def dist_coeffs(self) -> np.ndarray:
        """
        cv2-ordered distortion vector (k1, k2, p1, p2).
        """
        return np.array([self.k1, self.k2, self.p1, self.p2])


########################################
# Estimation (pycolmap self-calibration)
########################################


def estimate_camera_distortion(frames: list[np.ndarray], max_frames: int = 60) -> DistortionProfile:
    """
    Self-calibrate one shared OPENCV camera from selected frames via pycolmap.

    - Runs SIFT + exhaustive matching + incremental mapping on <= max_frames
      evenly spaced frames (shared camera, ba_refine_extra_params on by default).
    - Raises ValueError when mapping fails or registers < 60% of the subset
      (too weak a solve to trust the distortion params).
    """
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError(
            "pycolmap is required for preproc.undistort.estimate_camera_distortion; "
            "install it in the reconstruction env"
        ) from e

    if not frames:
        raise ValueError("estimate_camera_distortion: no frames given")

    n = len(frames)
    idxs = np.unique(np.linspace(0, n - 1, min(max_frames, n)).round().astype(int))

    with tempfile.TemporaryDirectory(prefix="undistort_calib_") as tmp:
        tmp_path = Path(tmp)
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        out_dir = tmp_path / "sparse"
        out_dir.mkdir()
        database = tmp_path / "database.db"

        # Stage the calibration subset; store holds RGB, cv2 writes BGR
        for i in idxs:
            cv2.imwrite(
                str(image_dir / f"calib_{int(i):06d}.jpg"),
                cv2.cvtColor(frames[int(i)], cv2.COLOR_RGB2BGR),
            )

        # One shared OPENCV camera across the subset; mapper refines k1 k2 p1 p2
        pycolmap.extract_features(
            database,
            image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="OPENCV",
        )
        pycolmap.match_exhaustive(database)
        reconstructions = pycolmap.incremental_mapping(database, image_dir, out_dir)

        if not reconstructions:
            raise ValueError(
                f"undistort: self-calibration failed — pycolmap registered no model from {len(idxs)} frames"
            )
        recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        if recon.num_reg_images() < 0.6 * len(idxs):
            raise ValueError(
                f"undistort: self-calibration too weak — {recon.num_reg_images()}/{len(idxs)} frames "
                "registered; distortion params not trustworthy"
            )

        # Shared camera: exactly the largest model's camera params, OPENCV order
        camera = next(iter(recon.cameras.values()))
        fx, fy, cx, cy, k1, k2, p1, p2 = (float(v) for v in camera.params)

    height, width = frames[0].shape[:2]
    profile = DistortionProfile(
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )
    logger.info(
        "undistort: calibrated k1=%.5f k2=%.5f p1=%.5f p2=%.5f over %d/%d frames",
        k1,
        k2,
        p1,
        p2,
        recon.num_reg_images(),
        len(idxs),
    )
    return profile


########################################
# Undistortion (splatfacto-exact cv2 path)
########################################


def undistort_frames(
    frames: list[np.ndarray], profile: DistortionProfile
) -> tuple[list[np.ndarray], np.ndarray, tuple[int, int, int, int]]:
    """
    Undistort frames with alpha=0 crop; returns (frames, K_new, roi).

    - roi is (x, y, w, h) with w/h forced even (codec/model friendliness);
      K_new is the optimal new camera matrix shifted by the crop offset.
    - Raises ValueError when frame dims disagree with the profile.
    """
    if not frames:
        raise ValueError("undistort_frames: no frames given")

    # Every frame must match the profile's calibrated dims; name the first offender
    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]
        if (width, height) != (profile.width, profile.height):
            raise ValueError(
                f"undistort: frame {i} dims {width}x{height} != profile dims {profile.width}x{profile.height}"
            )
    height, width = frames[0].shape[:2]

    # alpha=0: zoom so the valid (distortion-free) region fills the ROI
    K_new, roi = cv2.getOptimalNewCameraMatrix(profile.K, profile.dist_coeffs, (width, height), 0)
    x, y, w, h = roi
    w -= w % 2
    h -= h % 2

    # Per-frame: remap to the new camera, crop to the even ROI
    out = [cv2.undistort(frame, profile.K, profile.dist_coeffs, None, K_new)[y : y + h, x : x + w] for frame in frames]

    # Principal point moves with the crop
    K_out = K_new.copy()
    K_out[0, 2] -= x
    K_out[1, 2] -= y
    return out, K_out, (x, y, w, h)
