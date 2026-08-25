"""
Tests for preproc undistortion: profile round-trip, cv2 undistort/crop path.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from collab_splats.preproc.undistort import (
    DistortionProfile,
    estimate_camera_distortion,
    undistort_frames,
)
from collab_splats.preproc.video import iter_frames


def _profile(width=640, height=480, k1=0.006, k2=-0.003):
    # GoPro-Linear-magnitude distortion, centred principal point
    return DistortionProfile(
        k1=k1,
        k2=k2,
        p1=0.0,
        p2=0.0,
        fx=500.0,
        fy=500.0,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
    )


def _distort_image(image, profile):
    # cv2.undistortPoints maps distorted->normalized-undistorted; projecting those
    # back through K gives, for each DISTORTED pixel, its undistorted location.
    # Remapping the clean image at those locations SAMPLES the clean image where
    # the undistorted content of each distorted pixel lives — i.e. it distorts it.
    h, w = image.shape[:2]
    K = np.array([[profile.fx, 0, profile.cx], [0, profile.fy, profile.cy], [0, 0, 1]])
    dist = np.array([profile.k1, profile.k2, profile.p1, profile.p2])
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)[:, None, :]
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(image, und[..., 0], und[..., 1], cv2.INTER_LINEAR)


def test_profile_dict_roundtrip():
    profile = _profile()
    assert DistortionProfile.from_dict(profile.to_dict()) == profile


def test_undistort_recovers_synthetic_distortion():
    # Checkerboard so residuals are visible; distort with the known profile,
    # undistort with the same profile, compare against the clean original.
    profile = _profile()
    tile = np.kron(np.indices((12, 16)).sum(0) % 2, np.ones((40, 40))) * 255
    clean = np.repeat(tile.astype(np.uint8)[:, :, None], 3, axis=2)
    distorted = _distort_image(clean, profile)

    restored, K_new, roi = undistort_frames([distorted], profile)
    x, y, w, h = roi
    reference = clean[y : y + h, x : x + w]

    # Interior compare (border interpolation is lossy either way)
    diff = np.abs(restored[0][20:-20, 20:-20].astype(int) - reference[20:-20, 20:-20].astype(int))
    assert diff.mean() < 10.0


def test_crop_dims_even_and_k_consistent():
    profile = _profile(width=641, height=481)  # odd input dims force the even-crop path
    frames = [np.zeros((481, 641, 3), dtype=np.uint8)]
    out, K_new, roi = undistort_frames(frames, profile)
    x, y, w, h = roi

    assert w % 2 == 0 and h % 2 == 0
    assert out[0].shape == (h, w, 3)
    # Principal point shifted by the crop offset, still inside the crop
    assert 0 < K_new[0, 2] < w and 0 < K_new[1, 2] < h


def test_mismatched_frame_shape_raises():
    # Second frame disagrees with the profile dims even though frames[0] matches
    profile = _profile()
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 320, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="frame 1"):
        undistort_frames(frames, profile)


def test_wrong_frame_dims_raise():
    profile = _profile(width=640, height=480)
    with pytest.raises(ValueError, match="dims"):
        undistort_frames([np.zeros((100, 100, 3), dtype=np.uint8)], profile)


def test_estimate_camera_distortion_tutorial_smoke():
    # Real-footage smoke: SIFT + exhaustive + mapper on 20 tutorial frames.
    # Slow (~1-3 min CPU); asserts a sane shared-camera OPENCV solve, not
    # specific distortion values.
    pytest.importorskip("pycolmap")

    video = Path("data/tutorial/tutorial_example-video.mp4")
    if not video.exists():
        pytest.skip("tutorial video not present")

    # 20 frames over a tighter window keeps enough overlap for exhaustive matching
    frames = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for _, bgr in iter_frames(video, indices=list(range(0, 400, 20)))]
    assert len(frames) == 20
    profile = estimate_camera_distortion(frames, max_frames=20)

    height, width = frames[0].shape[:2]
    assert (profile.width, profile.height) == (width, height)
    assert 0 < profile.fx < 4 * width and 0 < profile.fy < 4 * width
    assert abs(profile.k1) < 0.5 and abs(profile.k2) < 0.5
    assert np.isfinite([profile.k1, profile.k2, profile.p1, profile.p2]).all()
