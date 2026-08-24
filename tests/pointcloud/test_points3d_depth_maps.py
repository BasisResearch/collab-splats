"""
Tests for points3d_depth_maps: sparse per-view depth targets from a reconstruction's points3D.
"""

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.utils import points3d_depth_maps

W, H = 64, 48
K_PARAMS = [100.0, 100.0, 32.0, 24.0]  # fx, fy, cx, cy


def _recon(points_xyz, translations):
    """
    One PINHOLE camera, one image per translation (identity rotation), every image
    observing every point3D.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=K_PARAMS, camera_id=1)
    recon.add_camera_with_trivial_rig(cam)

    # Images carry one placeholder Point2D per point3D; add_point3D's track back-fills
    # each observation's point3D_id (helper ignores stored xy — it reprojects)
    for i, t in enumerate(translations):
        im = pycolmap.Image(name=f"frame_{i:06d}", camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.zeros(2)) for _ in points_xyz]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.asarray(t, dtype=np.float64))
        recon.add_image_with_trivial_frame(im, pose)

    for j, xyz in enumerate(points_xyz):
        track = pycolmap.Track()
        for i in range(len(translations)):
            track.add_element(i + 1, j)
        recon.add_point3D(np.asarray(xyz, dtype=np.float64), track, np.zeros(3, dtype=np.uint8))
    return recon


def test_projected_depths_land_on_expected_pixels():
    # (0,0,5) hits the principal point at depth 5 in view 0 and depth 6 in view 1 (t_z=1)
    recon = _recon([(0.0, 0.0, 5.0)], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    maps = points3d_depth_maps(recon, ["frame_000000", "frame_000001"], H, W)

    assert maps.shape == (2, H, W)
    assert maps.dtype == np.float32
    assert maps[0, 24, 32] == pytest.approx(5.0)
    assert maps[1, 24, 32] == pytest.approx(6.0)
    assert np.count_nonzero(maps[0]) == 1


def test_out_of_bounds_and_behind_camera_dropped():
    # (3,0,5) projects to u=92 (off the 64-wide grid); (0,0,-2) is behind the camera
    recon = _recon([(3.0, 0.0, 5.0), (0.0, 0.0, -2.0)], [(0.0, 0.0, 0.0)])
    maps = points3d_depth_maps(recon, ["frame_000000"], H, W)
    assert np.count_nonzero(maps) == 0


def test_pixel_collision_keeps_nearer_point():
    # Both points project to the principal point; depth 5 must win over depth 10
    recon = _recon([(0.0, 0.0, 10.0), (0.0, 0.0, 5.0)], [(0.0, 0.0, 0.0)])
    maps = points3d_depth_maps(recon, ["frame_000000"], H, W)
    assert maps[0, 24, 32] == pytest.approx(5.0)


def test_image_without_observations_gets_empty_map():
    # No points3D at all — the image has zero observations
    recon = _recon([], [(0.0, 0.0, 0.0)])
    maps = points3d_depth_maps(recon, ["frame_000000"], H, W)
    assert np.count_nonzero(maps) == 0


def test_resolution_mismatch_raises():
    recon = _recon([(0.0, 0.0, 5.0)], [(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="native frame resolution"):
        points3d_depth_maps(recon, ["frame_000000"], H * 2, W * 2)


def test_unknown_image_name_raises():
    recon = _recon([(0.0, 0.0, 5.0)], [(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="not in reconstruction"):
        points3d_depth_maps(recon, ["frame_999999"], H, W)
