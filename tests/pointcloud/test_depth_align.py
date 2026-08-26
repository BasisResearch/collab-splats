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


def _affine_observations(a, b, n_obs=200, d_min=2.0, d_max=40.0, seed=0):
    """
    Observations plus one depth row whose true mapping is 1/d_colmap = a*(1/d_vda) + b.
    """
    rng = np.random.default_rng(seed)
    d_vda_values = rng.uniform(d_min, d_max, n_obs)
    depth_row = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    observations = []
    for i, d_vda in enumerate(d_vda_values):
        u, v = i % GRID_W, (i // GRID_W) % GRID_H
        depth_row[v, u] = d_vda
        observations.append((u, v, float(1.0 / (a / d_vda + b))))
    return observations, depth_row


def _affine_scene(a, b, n_obs=200, d_min=2.0, d_max=40.0):
    """
    One frame whose true mapping is 1/d_colmap = a*(1/d_vda) + b, sampled on the depth grid.
    """
    observations, depth_row = _affine_observations(a, b, n_obs, d_min, d_max)
    return _fake_reconstruction({"frame_000000.jpg": observations}), depth_row[None]


def test_affine_recovers_the_generating_coefficients():
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    coeffs, _far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 0] == pytest.approx(1.5, rel=1e-4)
    assert coeffs[0, 1] == pytest.approx(-0.004, abs=1e-6)
    assert stats["n_fallback"] == 0


def test_affine_is_robust_to_gross_outliers():
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    # Corrupt 5% of the observations with a 100x depth error
    for point in list(recon.points3D.values())[:10]:
        point.xyz[2] *= 100.0

    coeffs, _far, _stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 0] == pytest.approx(1.5, rel=5e-3)
    assert coeffs[0, 1] == pytest.approx(-0.004, abs=5e-5)


def test_affine_falls_back_to_scale_below_the_obs_floor():
    # 30 observations: above MIN_ALIGN_OBS (20) but below MIN_AFFINE_OBS (50)
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)

    coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 1] == 0.0  # scale-only: b is exactly zero
    assert coeffs[0, 0] == pytest.approx(0.5)  # a = 1/s with s = 2.0
    assert stats["n_fallback"] == 1
    assert stats["n_below_obs_floor"] == 1
    assert (stats["n_unsolvable"], stats["n_nonpositive_a"], stats["n_saturating"]) == (0, 0, 0)


def test_affine_sub_floor_frame_gets_no_far_bound():
    # Frame 0 carries the fit; frame 1 has 3 observations, all landing on near surfaces.
    # Its model is the GLOBAL scale, so it has no per-frame range evidence to be bounded by
    # and must be supervised exactly as fully as the scale path supervises it.
    obs_fitted, row_fitted = _affine_observations(a=1.5, b=-0.004)
    obs_thin, row_thin = _affine_observations(a=1.5, b=-0.004, n_obs=3, d_min=4.0, d_max=5.5, seed=7)
    recon = _fake_reconstruction({"frame_000000.jpg": obs_fitted, "frame_000001.jpg": obs_thin})
    depth = np.stack([row_fitted, row_thin])

    _coeffs, far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg", "frame_000001.jpg"], depth)
    assert np.isfinite(far_limits[0])
    assert far_limits[1] == np.inf
    assert stats["n_below_obs_floor"] == 1


def test_affine_far_limit_is_the_furthest_fitted_observation():
    recon, depth = _affine_scene(a=1.0, b=0.0, d_min=2.0, d_max=40.0)
    _coeffs, far_limits, _stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    # Observations stop at 40 m, so nothing beyond ~40 m has evidence
    assert far_limits[0] == pytest.approx(40.0, rel=0.05)


def test_apply_affine_inverts_the_fitted_mapping():
    depth = np.array([[5.0, 10.0, 20.0]], dtype=np.float32)
    out = sfm._apply_affine_depth(depth, a=1.5, b=-0.004, far_limit=np.inf)
    expected = depth / (1.5 - 0.004 * depth)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_apply_affine_zeroes_saturated_and_far_pixels():
    depth = np.array([[5.0, 100.0, 400.0]], dtype=np.float32)
    # a + b*d goes non-positive at d = 250; far_limit cuts at 90
    out = sfm._apply_affine_depth(depth, a=1.0, b=-0.004, far_limit=90.0)
    assert out[0, 0] > 0.0
    assert out[0, 1] == 0.0  # beyond far_limit
    assert out[0, 2] == 0.0  # saturated AND beyond far_limit


def test_apply_affine_masks_past_the_saturation_horizon():
    # a=1.5, b=-0.03 saturates at d = -a/b = 50; far_limit is deliberately set past it
    out = sfm._apply_affine_depth(np.array([[10.0, 50.0, 55.0]]), a=1.5, b=-0.03, far_limit=60.0)
    assert out[0, 0] == pytest.approx(10.0 / 1.2)
    assert out[0, 1] == 0.0  # exactly at the horizon: the denominator is zero
    assert out[0, 2] == 0.0  # past the horizon, inside far_limit: masked, not a negative depth


def test_apply_affine_keeps_zeros_zero():
    depth = np.array([[0.0, 5.0]], dtype=np.float32)
    out = sfm._apply_affine_depth(depth, a=1.0, b=0.0, far_limit=np.inf)
    assert out[0, 0] == 0.0
