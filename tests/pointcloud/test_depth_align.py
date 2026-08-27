"""
Depth alignment: track-observation correspondences, scale fit, affine-in-disparity fit.
"""

import json
from pathlib import Path

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


def test_correspondences_drop_zero_out_of_bounds_and_behind_camera_samples():
    recon = _fake_reconstruction(
        {
            "frame_000000.jpg": [
                (3, 4, 10.0),  # kept
                (5, 5, 20.0),  # VDA depth left at 0 -> dropped
                (GRID_W + 8, 4, 10.0),  # pixel off the right edge of the depth grid -> dropped
                (7, 8, -10.0),  # point behind the camera, on a pixel that HAS depth -> dropped
            ]
        }
    )
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0
    depth[0, 8, 7] = 9.0

    d_colmap, d_vda = sfm._depth_correspondences(recon, ["frame_000000.jpg"], depth)[0]
    np.testing.assert_allclose(d_colmap, [10.0])
    np.testing.assert_allclose(d_vda, [5.0])


def test_correspondences_raise_on_unregistered_name():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0)]})
    depth = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)
    with pytest.raises(ValueError, match="not in reconstruction"):
        sfm._depth_correspondences(recon, ["frame_000000.jpg", "frame_000009.jpg"], depth)


def test_scale_alignment_recovers_a_constant_ratio():
    # 30 observations at exactly 2x, three of them 100x wrong: the median holds at 2.0 while
    # the mean of the same ratios is 21.8
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    observations[:3] = [(u, v, d * 100.0) for u, v, d in observations[:3]]
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


def _two_depth_scene(n_obs, a=1.5, b=-0.004, d_near=10.0, d_far=30.0, n_far=20):
    """
    One frame with only two distinct depths, a strict majority of them at the near one.

    - Two distinct depths make the least-squares fit exact and give every observation in a
      group the same residual, so the MAD median is exactly zero and the rejection rounds
      drop nothing. The observation count is then the only thing deciding whether the fit
      survives, which is what an MIN_AFFINE_OBS boundary test needs.
    """
    observations = []
    depth_row = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for i in range(n_obs):
        u, v = i % GRID_W, (i // GRID_W) % GRID_H
        d_vda = d_far if i < n_far else d_near
        depth_row[v, u] = d_vda
        observations.append((u, v, float(1.0 / (a / d_vda + b))))
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


def test_affine_rejection_is_tight_enough_to_drop_a_moderate_outlier():
    # 20 of 200 observations off by only 1.5x. At 3 MAD they are rejected and the fit is
    # exact; at a much looser threshold they survive and drag the fit to a=1.36, b=+0.0017
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    for point in list(recon.points3D.values())[:20]:
        point.xyz[2] *= 1.5

    coeffs, _far, _stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 0] == pytest.approx(1.5, rel=1e-4)
    assert coeffs[0, 1] == pytest.approx(-0.004, abs=1e-6)


def test_affine_survives_a_single_sky_pixel():
    # The far-end guard reads p99 of the depth map, not its max. One unobserved 5000-unit sky
    # pixel puts the fitted disparity at -0.0097 at the max and +0.0275 at p99, so max would
    # throw this frame away. Measured 2026-08-26: max rejected 78/300 frames, p99 14/300
    recon, depth = _affine_scene(a=1.5, b=-0.01)
    depth[0, GRID_H - 1, GRID_W - 1] = 5000.0

    _coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert stats["n_fitted"] == 1
    assert stats["n_pole_too_close"] == 0


def test_affine_stats_report_the_fit_residual():
    # The counters say how many frames fitted, not how well; a clean synthetic scene must
    # come back at ~0 residual, which is what makes a real scene's number readable
    recon, depth = _affine_scene(a=1.5, b=-0.004)

    _coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert stats["n_fitted"] == 1
    assert len(stats["disparity_residual_p10_p50_p90"]) == 3
    assert max(stats["disparity_residual_p10_p50_p90"]) < 1e-6


def test_alignment_rejects_a_names_to_depth_row_mismatch():
    # One name short of the depth stack used to walk off the end of image_names silently,
    # aligning row i of the depth with frame i of a different list
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    two_rows = np.concatenate([depth, depth])

    with pytest.raises(ValueError, match="rows would misalign"):
        sfm.align_depth_affine(recon, ["frame_000000.jpg"], two_rows)


def test_affine_rejects_a_fit_whose_pole_crowds_the_far_end():
    # a=1.5, b=-0.03 saturates at d = 50 while the frame only reaches 40, so the disparity
    # there is positive (0.0075) and a positivity-only floor waves it through — at 5x the
    # scale-only depth, inside every downstream mask. The relative floor rejects it
    recon, depth = _affine_scene(a=1.5, b=-0.03)

    coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert stats["n_fitted"] == 0
    assert stats["n_pole_too_close"] == 1
    assert coeffs[0, 1] == 0.0  # scale-only


def test_affine_rejects_a_fit_that_saturates_inside_the_frame():
    # a=1.5, b=-0.03 saturates at d = 50. Tracks only reach 40, but the depth map itself runs
    # to 60, so p99 (60) sits past the horizon and the fit must not be used at all. p50 (25.9)
    # would wave it through
    recon, depth = _affine_scene(a=1.5, b=-0.03)
    depth[0, GRID_H - 1, :] = 60.0

    coeffs, far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert stats["n_fitted"] == 0
    assert stats["n_pole_too_close"] == 1
    assert coeffs[0, 1] == 0.0  # scale-only
    assert far_limits[0] == np.inf  # a rejected fit leaves no range evidence behind


def test_affine_rejects_a_fit_that_inverts_depth():
    # d_colmap FALLS as d_vda rises, so the least-squares slope comes out negative
    recon, depth = _affine_scene(a=-1.0, b=0.6)

    coeffs, far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert stats["n_fitted"] == 0
    assert stats["n_nonpositive_a"] == 1
    assert coeffs[0, 1] == 0.0
    assert far_limits[0] == np.inf


def test_affine_obs_floor_is_exactly_min_affine_obs():
    # 49 falls back, 50 and 51 fit. Pins the constant and which side of it is strict; the
    # two-depth fixture rejects nothing, so the observation count is the only variable
    for n_obs, expected_fitted in ((49, 0), (50, 1), (51, 1)):
        recon, depth = _two_depth_scene(n_obs)
        _coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
        assert stats["n_fitted"] == expected_fitted, f"n_obs={n_obs}"


def test_affine_falls_back_to_scale_below_the_obs_floor():
    # 30 observations: above MIN_ALIGN_OBS (20) but below MIN_AFFINE_OBS (50). Three are 100x
    # wrong, so the fallback scale is a median (2.0), not a mean (21.8)
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    observations[:3] = [(u, v, d * 100.0) for u, v, d in observations[:3]]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)

    coeffs, _far, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert coeffs[0, 1] == 0.0  # scale-only: b is exactly zero
    assert coeffs[0, 0] == pytest.approx(0.5)  # a = 1/s with s = 2.0
    assert stats["n_fallback"] == 1
    assert stats["n_below_obs_floor"] == 1
    assert (stats["n_unsolvable"], stats["n_nonpositive_a"], stats["n_pole_too_close"]) == (0, 0, 0)


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


def test_affine_far_limit_is_the_furthest_surviving_vda_inlier():
    # a=2.0, b=-0.002 puts d_colmap on a visibly different scale from d_vda (max 20.8 vs
    # 39.9), so a bound sourced from the COLMAP side instead of the VDA side is detectable
    recon, depth = _affine_scene(a=2.0, b=-0.002, d_min=2.0, d_max=40.0)
    colmap_max = max(float(point.xyz[2]) for point in recon.points3D.values())

    _coeffs, far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth)
    assert stats["n_fitted"] == 1
    assert far_limits[0] == pytest.approx(float(depth[0].max()))
    assert far_limits[0] > 1.5 * colmap_max


def test_affine_far_limit_ignores_a_rejected_outlier():
    # One track lands on a 200-unit sky pixel while reporting a 3 m COLMAP depth. MAD throws
    # it out of the fit, so it must not stretch the supported range to where there is no fit
    observations, depth_row = _affine_observations(a=1.5, b=-0.004)
    depth_row[GRID_H - 1, GRID_W - 1] = 200.0
    observations.append((GRID_W - 1, GRID_H - 1, 3.0))
    recon = _fake_reconstruction({"frame_000000.jpg": observations})

    _coeffs, far_limits, stats = sfm.align_depth_affine(recon, ["frame_000000.jpg"], depth_row[None])
    assert stats["n_fitted"] == 1
    assert far_limits[0] == pytest.approx(float(np.sort(depth_row.ravel())[-2]))


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


def test_apply_affine_targets_only_positive_input_depths():
    # 0 = no VDA depth, and a negative depth is nonsense; neither may become a target. With
    # a=0.5, b=-0.01 an unguarded negative pixel would map to a finite negative depth
    depth = np.array([[0.0, -5.0, 5.0]], dtype=np.float32)
    out = sfm._apply_affine_depth(depth, a=0.5, b=-0.01, far_limit=np.inf)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.0
    assert out[0, 2] == pytest.approx(5.0 / 0.45)


########################################################################
# apply_depth_alignment: model selection, provenance attrs, world points
########################################################################


class _Result:
    """
    Minimal FeedforwardResult stand-in: the fields apply_depth_alignment touches.
    """

    def __init__(self, depth):
        n_frames = depth.shape[0]
        self.depth = depth
        self.image_paths = [Path(f"frame_{row:06d}.jpg") for row in range(n_frames)]
        self.extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], n_frames, axis=0)
        self.intrinsics = np.repeat(
            np.array([[[10.0, 0.0, GRID_W / 2], [0.0, 10.0, GRID_H / 2], [0.0, 0.0, 1.0]]], dtype=np.float32),
            n_frames,
            axis=0,
        )
        self.world_points = None


def _scale_scene(ratio=2.0):
    """
    One frame whose COLMAP depths are a constant multiple of its VDA depths.
    """
    n_obs = 30  # comfortably over MIN_ALIGN_OBS, under MIN_AFFINE_OBS: a scale-path fixture
    observations = [(i % GRID_W, i % GRID_H, ratio * (i + 1)) for i in range(n_obs)]
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(n_obs):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)
    return _fake_reconstruction({"frame_000000.jpg": observations}), depth


def test_apply_depth_alignment_scale_is_the_default_and_stamps_the_model():
    recon, depth = _scale_scene()
    result = _Result(depth.copy())

    attrs = sfm.apply_depth_alignment(result, recon)
    assert attrs["depth_scale"] == "colmap"
    assert attrs["depth_align_model"] == "scale"
    assert attrs["depth_scales"] == [pytest.approx(2.0)]
    assert result.depth[0, 4, 4] == pytest.approx(2.0 * depth[0, 4, 4])


def test_apply_depth_alignment_reunprojects_world_points_from_the_aligned_depth():
    # Identity extrinsics, so a world point's z IS its depth in that view: world_points must
    # be re-derived from the ALIGNED depth, never left at None or carried over unscaled
    recon, depth = _scale_scene()
    result = _Result(depth.copy())

    sfm.apply_depth_alignment(result, recon)
    assert result.world_points is not None
    assert result.world_points.shape == (1, GRID_H, GRID_W, 3)
    np.testing.assert_allclose(result.world_points[..., 2], result.depth, rtol=1e-5)
    assert result.world_points[0, 4, 4, 2] == pytest.approx(10.0)


def test_apply_depth_alignment_affine_stamps_coefficients():
    recon, depth = _affine_scene(a=1.5, b=-0.004)
    result = _Result(depth.copy())

    attrs = sfm.apply_depth_alignment(result, recon, model="affine")
    assert attrs["depth_scale"] == "colmap"
    assert attrs["depth_align_model"] == "affine"
    assert attrs["depth_affine_ab"][0][0] == pytest.approx(1.5, rel=1e-4)
    assert attrs["depth_affine_ab"][0][1] == pytest.approx(-0.004, abs=1e-6)
    assert result.world_points is not None
    np.testing.assert_allclose(result.world_points[..., 2], result.depth, rtol=1e-5)


def test_apply_depth_alignment_affine_applies_the_affine_mapping_not_a_scale():
    # a=1.5, b=-0.015 runs the true d_colmap/d_vda ratio from 0.68 to 1.11 across the frame,
    # so an affine-aligned pixel lands nowhere near its scale-aligned value
    recon, depth = _affine_scene(a=1.5, b=-0.015)
    affine_result, scale_result = _Result(depth.copy()), _Result(depth.copy())

    sfm.apply_depth_alignment(affine_result, recon, model="affine")
    sfm.apply_depth_alignment(scale_result, recon, model="scale")

    # Every surviving pixel must equal d / (a + b*d), the mapping the fit defines
    supported = affine_result.depth[0] > 0
    expected = depth[0] / (1.5 - 0.015 * depth[0])
    np.testing.assert_allclose(affine_result.depth[0][supported], expected[supported], rtol=1e-4)
    assert np.abs(affine_result.depth[0][supported] - scale_result.depth[0][supported]).max() > 5.0


def test_apply_depth_alignment_affine_masks_beyond_the_fitted_range():
    # An unobserved 100-unit pixel sits well inside the saturation horizon (-a/b = 375) but
    # far past the furthest fitted observation (~40), so only the far bound can mask it
    observations, depth_row = _affine_observations(a=1.5, b=-0.004)
    depth_row[GRID_H - 1, GRID_W - 1] = 100.0
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    result = _Result(depth_row[None].copy())

    attrs = sfm.apply_depth_alignment(result, recon, model="affine")
    assert result.depth[0, GRID_H - 1, GRID_W - 1] == 0.0
    assert 0.0 < attrs["depth_masked_fraction"] < 0.05
    assert attrs["depth_far_limits"][0] == pytest.approx(float(np.sort(depth_row.ravel())[-2]))


def test_apply_depth_alignment_affine_writes_a_scale_only_far_limit_as_null():
    # Frame 1 falls under the observation floor, so its bound is inf. Zarr attrs are JSON and
    # inf is not valid JSON — zarr writes it through as a bare `Infinity` token
    obs_fitted, row_fitted = _affine_observations(a=1.5, b=-0.004)
    obs_thin, row_thin = _affine_observations(a=1.5, b=-0.004, n_obs=3, d_min=4.0, d_max=5.5, seed=7)
    recon = _fake_reconstruction({"frame_000000.jpg": obs_fitted, "frame_000001.jpg": obs_thin})
    result = _Result(np.stack([row_fitted, row_thin]))

    attrs = sfm.apply_depth_alignment(result, recon, model="affine")
    assert attrs["depth_far_limits"][1] is None
    assert attrs["depth_far_limits"][0] is not None
    assert attrs["depth_scale_fallback_frames"] == ["frame_000001.jpg"]
    json.dumps(attrs, allow_nan=False)


def test_apply_depth_alignment_rejects_an_unknown_model():
    recon, depth = _affine_scene(a=1.0, b=0.0)
    with pytest.raises(ValueError, match="depth_align"):
        sfm.apply_depth_alignment(_Result(depth.copy()), recon, model="quadratic")
