"""
Depth alignment: track-observation correspondences, scale fit, affine-in-disparity fit.
"""

import numpy as np
import pytest

from collab_splats.pointcloud import sfm

GRID_H, GRID_W = 16, 32
CAM_W, CAM_H = 64, 32


def _fake_reconstruction(per_image_depths):
    """
    Duck-typed pycolmap stand-in: one image per entry, observations at fixed pixels.

    - per_image_depths maps "frame_NNNNNN.jpg" -> list of (u_grid, v_grid, d_colmap).
      Pixels are given in DEPTH-GRID coordinates and scaled up to camera resolution here,
      so a test states where in the depth map an observation lands.
    """

    class _Camera:
        def __init__(self):
            self.width, self.height = CAM_W, CAM_H

    class _Point2D:
        def __init__(self, xy, point3D_id):
            self.xy = np.asarray(xy, dtype=np.float64)
            self.point3D_id = point3D_id

        def has_point3D(self):
            return self.point3D_id is not None

    class _Point3D:
        def __init__(self, xyz):
            self.xyz = np.asarray(xyz, dtype=np.float64)

    class _Image:
        def __init__(self, name, points2D):
            self.name = name
            self.camera_id = 1
            self.points2D = points2D

        def cam_from_world(self):
            # Identity pose: a point's world xyz IS its camera-frame xyz, so xyz[2] = d_colmap
            class _Pose:
                def matrix(self_inner):
                    return np.eye(4)

            return _Pose()

    points3D, images = {}, {}
    next_id = 1
    for image_id, (name, observations) in enumerate(per_image_depths.items(), start=1):
        points2D = []
        for u_grid, v_grid, d_colmap in observations:
            points3D[next_id] = _Point3D([0.0, 0.0, d_colmap])
            points2D.append(_Point2D([u_grid * CAM_W / GRID_W, v_grid * CAM_H / GRID_H], next_id))
            next_id += 1
        images[image_id] = _Image(name, points2D)

    class _Recon:
        pass

    recon = _Recon()
    recon.points3D = points3D
    recon.images = images
    recon.cameras = {1: _Camera()}
    return recon


def test_correspondences_pair_track_depth_with_sampled_vda_depth():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0), (7, 8, 20.0)]})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0
    depth[0, 8, 7] = 8.0

    pairs = sfm._depth_correspondences(recon, ["frame_000000.jpg"], depth)
    assert len(pairs) == 1
    d_colmap, d_vda = pairs[0]
    np.testing.assert_allclose(sorted(d_colmap), [10.0, 20.0])
    np.testing.assert_allclose(sorted(d_vda), [5.0, 8.0])


def test_correspondences_drop_zero_and_out_of_bounds_samples():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0), (5, 5, 20.0)]})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0  # (5,5) is left at 0 -> dropped

    d_colmap, d_vda = sfm._depth_correspondences(recon, ["frame_000000.jpg"], depth)[0]
    assert len(d_colmap) == 1


def test_correspondences_raise_on_unregistered_name():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0)]})
    depth = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)
    with pytest.raises(ValueError, match="not in reconstruction"):
        sfm._depth_correspondences(recon, ["frame_000000.jpg", "frame_000009.jpg"], depth)


def test_scale_alignment_recovers_a_constant_ratio():
    # 30 observations at exactly 2x -> the frame's fitted scale is 2.0
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)

    scales, stats = sfm.align_depth_to_reconstruction(recon, ["frame_000000.jpg"], depth)
    assert scales[0] == pytest.approx(2.0, rel=1e-6)
    assert stats["n_fallback"] == 0
