import numpy as np
import pytest
from collab_splats.geometry.transforms import (
    OPENGL_TO_OPENCV,
    estimate_intrinsics_from_points,
    extrinsics_to_homogeneous,
    invert_poses,
    extract_intrinsics,
    intrinsics_4x4,
    project_to_so3,
    rescale_intrinsics,
    fit_dominant_plane,
    rotation_align_vectors,
    shift_intrinsics,
    transform_points,
    _compute_weighted_median,
    umeyama_se3,
    umeyama_sim3,
)


def _random_rigid(*shape):
    """Build a random valid rigid-body (SE3) transform via QR decomposition."""
    A = np.random.randn(*shape, 3, 3)
    Q, _ = np.linalg.qr(A)
    t = np.random.randn(*shape, 3, 1)
    poses = np.zeros(shape + (4, 4))
    poses[..., :3, :3] = Q
    poses[..., :3, 3:] = t
    poses[..., 3, 3] = 1.0
    return poses.astype(np.float64)


def test_transform_points_matches_homogeneous_matmul():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Q, [0.5, -2.0, 3.0]
    pts = rng.standard_normal((6, 5, 3))  # leading batch shape, like an (H, W, 3) grid
    hom = np.concatenate([pts, np.ones((6, 5, 1))], axis=-1)
    np.testing.assert_allclose(transform_points(pts, T), (hom @ T.T)[..., :3], atol=1e-12)


def test_transform_points_inverse_roundtrip():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Q, [1.0, 2.0, -0.5]
    pts = rng.standard_normal((10, 3))
    np.testing.assert_allclose(transform_points(transform_points(pts, T), invert_poses(T)), pts, atol=1e-12)


def test_extrinsics_to_homogeneous_batched():
    ext = np.random.rand(4, 3, 4).astype(np.float32)
    out = extrinsics_to_homogeneous(ext)
    assert out.shape == (4, 4, 4)
    np.testing.assert_array_equal(out[:, 3, :], [[0, 0, 0, 1]] * 4)
    np.testing.assert_array_equal(out[:, :3, :], ext)


def test_extrinsics_to_homogeneous_single():
    ext = np.random.rand(3, 4).astype(np.float32)
    out = extrinsics_to_homogeneous(ext)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out[3, :], [0, 0, 0, 1])
    np.testing.assert_array_equal(out[:3, :], ext)


def test_extrinsics_to_homogeneous_dtype_preserved():
    ext = np.random.rand(2, 3, 4).astype(np.float64)
    out = extrinsics_to_homogeneous(ext)
    assert out.dtype == np.float64


def test_invert_poses_single_roundtrip():
    T = _random_rigid()
    np.testing.assert_allclose(invert_poses(T) @ T, np.eye(4), atol=1e-10)


def test_invert_poses_batched_roundtrip():
    T = _random_rigid(5)
    result = invert_poses(T) @ T
    np.testing.assert_allclose(result, np.eye(4)[None].repeat(5, 0), atol=1e-10)


def test_invert_poses_arbitrary_batch_shape():
    T = _random_rigid(3, 7)
    result = invert_poses(T) @ T
    eye = np.eye(4)[None, None].repeat(3, 0).repeat(7, 1)
    np.testing.assert_allclose(result, eye, atol=1e-10)


def test_invert_poses_dtype_preserved():
    T = _random_rigid().astype(np.float32)
    assert invert_poses(T).dtype == np.float32


def test_extract_intrinsics_basic():
    K = np.array([[500.0, 0, 320.0], [0, 480.0, 240.0], [0, 0, 1.0]])
    fx, fy, cx, cy = extract_intrinsics(K)
    assert fx == 500.0
    assert fy == 480.0
    assert cx == 320.0
    assert cy == 240.0


def test_extract_intrinsics_returns_floats():
    K = np.eye(3, dtype=np.float32)
    fx, fy, cx, cy = extract_intrinsics(K)
    assert isinstance(fx, float)


@pytest.mark.parametrize("batch", [(), (5,)])
def test_intrinsics_4x4_embeds_k_top_left(batch):
    K = np.random.default_rng(0).uniform(1.0, 500.0, size=batch + (3, 3)).astype(np.float32)
    out = intrinsics_4x4(K)
    assert out.shape == batch + (4, 4)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[..., :3, :3], K)
    np.testing.assert_array_equal(out[..., 3, 3], 1.0)
    np.testing.assert_array_equal(out[..., 3, :3], 0.0)
    np.testing.assert_array_equal(out[..., :3, 3], 0.0)


def _model_to_original(K: np.ndarray, crops: np.ndarray, model_hw: tuple[int, int]) -> np.ndarray:
    """Undo crop-then-resize the way callers do: resize model -> crop size, then shift by +tl."""
    crops = np.asarray(crops, dtype=np.float64)
    crop_hw = np.stack([crops[..., 3] - crops[..., 1], crops[..., 2] - crops[..., 0]], axis=-1)
    K = rescale_intrinsics(K, model_hw, crop_hw)
    return shift_intrinsics(K, crops[..., :2])


def test_model_to_original_known_answer():
    """Crop (11,7)-(59,47) resized to a 12x8 model grid: sx=0.25, sy=0.2."""
    K = np.array([[10.0, 0, 6.0], [0, 10.0, 4.0], [0, 0, 1]])
    out = _model_to_original(K, (11.0, 7.0, 59.0, 47.0), (8, 12))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[40.0, 0, 35.0], [0, 50.0, 27.0], [0, 0, 1]])


def test_model_to_original_round_trips_a_projection_through_the_crop_box():
    """A model-K projection mapped through the crop box lands on the original-K pixel.

    - non-square crop, off-centre origin, distinct fx/fy, so swapping sx/sy or dropping tl fails
    - batched call equals the per-frame calls, so (N, 3, 3) with (N, 2) hw/offset broadcasts per row
    """
    rng = np.random.default_rng(0)
    crops = np.array([[16.0, 8.0, 48.0, 40.0], [5.0, 30.0, 105.0, 90.0]])
    model_hw = (24, 40)
    K_model = np.array([[[30.0, 0, 19.5], [0, 22.0, 11.5], [0, 0, 1]], [[35.0, 0, 20.0], [0, 18.0, 12.0], [0, 0, 1]]])
    K_orig = _model_to_original(K_model, crops, model_hw)

    for k in range(2):
        np.testing.assert_array_equal(K_orig[k], _model_to_original(K_model[k], crops[k], model_hw))

        # Project camera-frame points on both grids
        X = rng.uniform([-1, -1, 2], [1, 1, 6], size=(50, 3))
        uv_model = (X @ K_model[k].T)[:, :2] / X[:, 2:]
        uv_orig = (X @ K_orig[k].T)[:, :2] / X[:, 2:]

        # Map model pixels through the crop box by hand
        tl_x, tl_y, cr_x, cr_y = crops[k]
        scale = np.array([(cr_x - tl_x) / model_hw[1], (cr_y - tl_y) / model_hw[0]])
        np.testing.assert_allclose(uv_model * scale + [tl_x, tl_y], uv_orig, rtol=0, atol=1e-9)


def test_crop_then_resize_and_its_undo_round_trip_k():
    """original -> model (shift -tl, resize crop -> model) -> original returns K; ops undo in reverse order."""
    crops = np.array([[16.0, 8.0, 48.0, 40.0], [5.0, 30.0, 105.0, 90.0]])
    crop_hw = np.stack([crops[:, 3] - crops[:, 1], crops[:, 2] - crops[:, 0]], axis=-1)
    model_hw = (24, 40)
    K = np.array([[[300.0, 0, 60.0], [0, 280.0, 45.0], [0, 0, 1]], [[500.0, 0, 55.0], [0, 520.0, 70.0], [0, 0, 1]]])

    K_model = shift_intrinsics(K, -crops[:, :2])
    K_model = rescale_intrinsics(K_model, crop_hw, model_hw)
    np.testing.assert_allclose(_model_to_original(K_model, crops, model_hw), K, rtol=0, atol=1e-12)


def test_rescale_intrinsics_scales_whole_rows_including_skew():
    """x row (fx, skew, cx) by dst_w/src_w, y row (fy, cy) by dst_h/src_h; bottom row untouched."""
    K = np.array([[500.0, 7.0, 250.0], [0, 400.0, 200.0], [0, 0, 1]])
    out = rescale_intrinsics(K, (400, 500), (100, 250))
    np.testing.assert_allclose(out, [[250.0, 3.5, 125.0], [0, 100.0, 50.0], [0, 0, 1]])


def test_shift_intrinsics_moves_only_the_principal_point():
    K = np.array([[500.0, 7.0, 250.0], [0, 400.0, 200.0], [0, 0, 1]])
    out = shift_intrinsics(K, (-10.0, 30.0))
    np.testing.assert_array_equal(out, [[500.0, 7.0, 240.0], [0, 400.0, 230.0], [0, 0, 1]])


def test_project_to_so3_returns_nearest_rotation_with_det_one():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 3, 3))
    R = project_to_so3(M)
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(R), 1.0)


def test_project_to_so3_fixes_a_reflection_and_keeps_a_rotation_bit_exact():
    refl = np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(project_to_so3(refl)), 1.0)
    c, s = np.cos(0.3), np.sin(0.3)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    U, _, Vt = np.linalg.svd(Rz)
    assert np.array_equal(project_to_so3(Rz), U @ Vt)


def test_project_to_so3_batch_flips_only_the_reflected_member():
    refl = np.diag([1.0, 1.0, -1.0])
    batch = project_to_so3(np.stack([np.eye(3), refl]))
    assert np.array_equal(batch[0], project_to_so3(np.eye(3)))
    assert np.allclose(np.linalg.det(batch), 1.0)


def test_opengl_to_opencv_shape():
    assert OPENGL_TO_OPENCV.shape == (4, 4)


def test_opengl_to_opencv_flips_yz():
    expected = np.diag([1, -1, -1, 1]).astype(np.float64)
    np.testing.assert_array_equal(OPENGL_TO_OPENCV, expected)


def test_rotation_align_vectors_identity():
    """Aligning a vector to itself returns identity."""
    src = np.array([0.0, 0.0, 1.0])
    R = rotation_align_vectors(src, src)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_rotation_align_vectors_aligns_correctly():
    """R @ src ≈ dst."""
    src = np.array([0.0, 1.0, 0.0])
    dst = np.array([0.0, 0.0, 1.0])
    R = rotation_align_vectors(src, dst)
    result = R @ src
    np.testing.assert_allclose(result, dst, atol=1e-10)


def test_rotation_align_vectors_is_rotation():
    """det(R) == 1 and R @ R.T == I."""
    src = np.array([1.0, 0.0, 0.0])
    dst = np.array([0.0, 1.0, 0.0])
    R = rotation_align_vectors(src, dst)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_rotation_align_vectors_antiparallel():
    """180-degree case: src = -dst still returns a valid rotation."""
    src = np.array([0.0, 0.0, 1.0])
    dst = np.array([0.0, 0.0, -1.0])
    R = rotation_align_vectors(src, dst)
    result = R @ src
    np.testing.assert_allclose(result, dst, atol=1e-6)


def test_compute_weighted_median_equal_weights_matches_plain_median():
    # With uniform weights the weighted median is the ordinary median.
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    weights = np.ones_like(values)
    assert _compute_weighted_median(values, weights) == pytest.approx(3.0)


def test_compute_weighted_median_follows_the_weight_mass():
    # Weight concentrated on the low values pulls the median down, even though
    # the high values are the numerical majority by count.
    # Deliberately unsorted: this is what pins the argsort. Pre-sorted input would
    # pass even if the sort were deleted.
    values = np.array([9.0, 1.0, 9.0, 1.0, 9.0], dtype=np.float32)
    weights = np.array([1.0, 50.0, 1.0, 50.0, 1.0], dtype=np.float32)
    assert _compute_weighted_median(values, weights) == pytest.approx(1.0)


def test_compute_weighted_median_empty_returns_none():
    # Signals "no estimate" to the caller, which raises rather than falling back.
    assert _compute_weighted_median(np.array([]), np.array([])) is None


def test_compute_weighted_median_zero_weights_returns_none():
    # No confidence mass to bisect. Must not silently return the smallest value.
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _compute_weighted_median(values, np.zeros(3, dtype=np.float32)) is None


def test_compute_weighted_median_subsamples_deterministically():
    # Above max_n the seeded RNG must give the same answer every call — the
    # intrinsics fit is otherwise non-reproducible at production frame counts.
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 10.0, size=200_000).astype(np.float32)
    weights = np.ones_like(values)
    first = _compute_weighted_median(values, weights, max_n=1000)
    assert first == _compute_weighted_median(values, weights, max_n=1000)
    # ...and that subsampling did not move the estimate off the true median.
    assert first == pytest.approx(float(np.median(values)), abs=1.0)


def _synthetic_local_points(h: int, w: int, fx: float, fy: float, depth: float = 2.0) -> np.ndarray:
    """Exact camera-frame pointmap for a centre-principal pinhole camera, shape (1,H,W,3).

    Built about the same centre the estimator assumes, cx=(W-1)/2 and cy=(H-1)/2,
    so recovery is exact and the principal-point assertion below is meaningful.
    """
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )
    z = np.full((h, w), depth, dtype=np.float32)
    return np.stack([uu * z / fx, vv * z / fy, z], axis=-1)[None].astype(np.float32)


@pytest.mark.parametrize("fx,fy", [(320.0, 320.0), (352.0, 320.0)])
def test_fit_recovers_known_intrinsics(fx, fy):
    # The anisotropic row (fx/fy = 1.10) is what protects the aspect-ratio argument:
    # our resize rounds each axis to a multiple of 14 independently, so a real camera
    # genuinely produces fx != fy at model resolution. Any "simplification" that
    # averages them fails here.
    h, w = 224, 308
    pts = _synthetic_local_points(h, w, fx, fy)
    conf = np.ones((1, h, w), dtype=np.float32)

    k = estimate_intrinsics_from_points(pts, conf)

    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(fx, rel=1e-3)
    assert k[1, 1] == pytest.approx(fy, rel=1e-3)
    # cx/cy are decided by the estimator's own centred grid, not by the caller.
    assert k[0, 2] == pytest.approx((w - 1) / 2.0)
    assert k[1, 2] == pytest.approx((h - 1) / 2.0)
    assert k[2, 2] == pytest.approx(1.0)
    if fx != fy:
        assert k[0, 0] != pytest.approx(k[1, 1], rel=1e-3)


def test_fit_survives_confident_outliers():
    # Corrupt 30% of pixels AND give them full confidence. A weighted median is
    # unmoved; a least-squares fit would be dragged toward the corrupted focal.
    h, w = 112, 154
    fx = fy = 160.0
    pts = _synthetic_local_points(h, w, fx, fy)
    conf = np.ones((1, h, w), dtype=np.float32)

    rng = np.random.default_rng(7)
    bad = rng.random((1, h, w)) < 0.30
    pts[bad, 0] *= 0.5  # halving X doubles the implied fx for those pixels
    pts[bad, 1] *= 0.5

    k = estimate_intrinsics_from_points(pts, conf)

    assert k[0, 0] == pytest.approx(fx, rel=1e-2)
    assert k[1, 1] == pytest.approx(fy, rel=1e-2)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p, c: (p, np.zeros_like(c)), id="empty_conf_mask"),
        pytest.param(lambda p, c: (np.full_like(p, np.nan), c), id="nan_points"),
        pytest.param(lambda p, c: (p * np.array([1, 1, -1], np.float32), c), id="negative_depth"),
    ],
)
def test_degenerate_input_raises_instead_of_falling_back(mutate):
    # No 1.2*max(W,H) fallback focal, by design: a silently-wrong K is exactly the
    # regression class of be24be2, which produced a plausible mesh from a bad camera.
    h, w = 56, 70
    pts, conf = mutate(_synthetic_local_points(h, w, 80.0, 80.0), np.ones((1, h, w), np.float32))
    with pytest.raises(RuntimeError, match="intrinsics fit failed"):
        estimate_intrinsics_from_points(pts, conf)


def test_fit_accepts_trailing_axis_confidence():
    # LoGeR's conf head emits (N,H,W,1); the estimator must not require a squeeze
    # from its caller, since _forward and the tests reach it by different routes.
    h, w = 56, 70
    pts = _synthetic_local_points(h, w, 80.0, 80.0)
    k4 = estimate_intrinsics_from_points(pts, np.ones((1, h, w, 1), np.float32))
    k3 = estimate_intrinsics_from_points(pts, np.ones((1, h, w), np.float32))
    np.testing.assert_allclose(k4, k3)


def test_fit_weights_by_confidence_not_by_count():
    # A plain np.median would follow the 60% majority; the weighted median follows the
    # mass. Every other estimator test uses uniform confidence, so swapping np.median in
    # survives all of them.
    h, w = 112, 154
    good = _synthetic_local_points(h, w, 400.0, 400.0)
    bad = _synthetic_local_points(h, w, 800.0, 800.0)

    # 60% of pixels carry the wrong focal, but only marginal confidence.
    rng = np.random.default_rng(3)
    is_bad = rng.random((1, h, w)) < 0.60
    pts = np.where(is_bad[..., None], bad, good)
    conf = np.where(is_bad, np.float32(0.11), np.float32(1.0))

    k = estimate_intrinsics_from_points(pts, conf)

    assert k[0, 0] == pytest.approx(400.0, rel=1e-2)
    assert k[1, 1] == pytest.approx(400.0, rel=1e-2)


def test_conf_threshold_gates_below_the_floor():
    # The gate is a hard floor, separate from the weighting: sub-floor pixels must not
    # contribute at all. Deleting `conf > conf_threshold` from the validity mask leaves
    # every other test green, because the one test using conf=0.0 is already caught by
    # the weighted median's own zero-mass guard.
    h, w = 56, 70
    pts = _synthetic_local_points(h, w, 80.0, 80.0)

    # Uniformly just under the floor: nothing survives, so it must fail loudly.
    with pytest.raises(RuntimeError, match="intrinsics fit failed"):
        estimate_intrinsics_from_points(pts, np.full((1, h, w), 0.09, np.float32))

    # Just over it: the same pointmap fits cleanly.
    k = estimate_intrinsics_from_points(pts, np.full((1, h, w), 0.11, np.float32))
    assert k[0, 0] == pytest.approx(80.0, rel=1e-3)


def test_rotation_angle_deg():
    """Geodesic angle: identity -> 0, known z-rotation -> its angle, clip guards trace noise."""
    from collab_splats.geometry.transforms import rotation_angle_deg

    assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)
    a = np.radians(30.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert rotation_angle_deg(Rz) == pytest.approx(30.0, abs=1e-6)
    # trace marginally above 3 from float error must not NaN through arccos
    assert rotation_angle_deg(np.eye(3) * (1 + 1e-12)) == pytest.approx(0.0, abs=1e-3)


def test_focal_bounds_reject_both_infinities():
    # Non-finite focals pass the validity mask — `z > 1e-3` and `|x| > 1e-6` constrain
    # the point, not the quotient — so ONLY the FOV bounds stop them. Both bounds are
    # load-bearing, and this pins one each: fx is corrupted to -inf (caught by the lower
    # bound) and fy to +inf (caught by the upper).
    #
    # The corrupted fraction has to exceed half. Every clean pixel here carries exactly
    # fx=400, so a minority of leaked infinities would shift the half-mass index inside a
    # constant array and change nothing — the mutation is only observable when the
    # infinity is itself returned and trips the isfinite check.
    h, w = 112, 154
    pts = _synthetic_local_points(h, w, 400.0, 400.0)
    conf = np.ones((1, h, w), dtype=np.float32)

    # Give X the sign opposite to its centred column so u_c * Z / X is negative, then
    # overflow it: |uu| * 1e38 already exceeds float32 range before the tiny divisor.
    uu = np.broadcast_to(np.arange(w, dtype=np.float32) - (w - 1) / 2.0, (1, h, w))
    rng = np.random.default_rng(11)
    hit = rng.random((1, h, w)) < 0.60
    pts[..., 0] = np.where(hit, -np.sign(uu) * 2e-6, pts[..., 0])
    pts[..., 2] = np.where(hit, np.float32(1e38), pts[..., 2])

    # Y is untouched, so inflating Z drives fy = v_c * Z / Y to +inf on the same pixels.
    k = estimate_intrinsics_from_points(pts, conf)

    assert k[0, 0] == pytest.approx(400.0, rel=1e-2)
    assert k[1, 1] == pytest.approx(400.0, rel=1e-2)



########################################################################
########## Point-set alignment #########################################
########################################################################


def test_umeyama_sim3_recovers_known_similarity():
    """umeyama_sim3 recovers the (s, R, t) that generated the target points."""
    from collab_splats.geometry.transforms import umeyama_sim3

    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))
    # Known 90 deg rotation about z, scale 2.5, translation (1, -2, 3)
    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s_true, t_true = 2.5, np.array([1.0, -2.0, 3.0])
    dst = s_true * (R_true @ src.T).T + t_true

    s, R, t = umeyama_sim3(src, dst)
    assert np.isclose(s, s_true, atol=1e-5)
    assert np.allclose(R, R_true, atol=1e-5)
    assert np.allclose(t, t_true, atol=1e-4)


@pytest.mark.parametrize("fn", [umeyama_se3, umeyama_sim3])
def test_umeyama_raises_on_fewer_than_three_points(fn):
    with pytest.raises(ValueError, match="at least 3"):
        fn(np.zeros((2, 3)), np.ones((2, 3)))


@pytest.mark.parametrize("fn", [umeyama_se3, umeyama_sim3])
def test_umeyama_raises_on_zero_total_weight(fn):
    rng = np.random.default_rng(0)
    src = rng.normal(size=(5, 3))
    with pytest.raises(ValueError, match="zero total weight"):
        fn(src, src, weights=np.zeros(5))


def test_umeyama_se3_recovers_known_rigid_transform():
    """umeyama_se3 recovers a rigid transform as a (4,4) homogeneous matrix."""
    from collab_splats.geometry.transforms import umeyama_se3

    rng = np.random.default_rng(1)
    src = rng.normal(size=(10, 3))
    R_true = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t_true = np.array([0.5, 0.25, -1.0])
    dst = (R_true @ src.T).T + t_true

    T = umeyama_se3(src, dst)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], R_true, atol=1e-5)
    assert np.allclose(T[:3, 3], t_true, atol=1e-5)


def test_bundle_adjustment_does_not_import_loop_closure():
    """BA and loop closure are siblings: BA must not depend on LC.

    umeyama_sim3 used to live in loop_closure/graph.py. Importing it from there would
    invert the layering and tie BA to that module's gtsam dependency, so it lives in
    transforms.py instead.
    """
    import ast
    import pathlib

    import collab_splats.geometry.bundle_adjustment as ba_mod

    tree = ast.parse(pathlib.Path(ba_mod.__file__).read_text(encoding="utf-8"))
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ] + [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ]
    assert not any("loop_closure" in m for m in imported), f"BA imports loop closure: {imported}"


@pytest.mark.parametrize(
    "box",
    [
        [5, 0, 5, 10],
        [0, 8, 10, 3],
        [0, 0, float("nan"), 10],
    ],
)
def test_model_to_original_rejects_bad_boxes(box):
    """Zero width, negative height, NaN width: the crop size reaches rescale_intrinsics as a bad hw."""
    with pytest.raises(ValueError):
        _model_to_original(np.eye(3), np.array(box, dtype=float), (4, 4))


@pytest.mark.parametrize("src_hw, dst_hw", [((0, 4), (4, 4)), ((4, 4), (4, -2)), ((4, float("nan")), (4, 4))])
def test_rescale_intrinsics_rejects_bad_sizes(src_hw, dst_hw):
    with pytest.raises(ValueError):
        rescale_intrinsics(np.eye(3), src_hw, dst_hw)


def test_rescale_intrinsics_rejects_hw_stack_with_single_k():
    """(3, 3) K can't pair with an (N, 2) hw stack: N doesn't broadcast into K's () leading dims."""
    with pytest.raises(ValueError, match="broadcast"):
        rescale_intrinsics(np.eye(3), (4, 4), np.full((2, 2), 8.0))


def test_shift_intrinsics_rejects_offset_stack_with_single_k():
    with pytest.raises(ValueError, match="broadcast"):
        shift_intrinsics(np.eye(3), np.zeros((2, 2)))


########################################################################
########## fit_dominant_plane ##########################################
########################################################################


def test_fit_dominant_plane_flat_z_up():
    """Flat ground at z=-1 → R≈I, t brings floor to z=0."""
    rng = np.random.default_rng(42)
    # Ground plane at z = -1 with small noise
    xy = rng.uniform(-5, 5, (800, 2)).astype(np.float32)
    z = rng.normal(-1.0, 0.005, (800,)).astype(np.float32)
    ground = np.column_stack([xy, z])
    # Scatter above-ground points
    above_xy = rng.uniform(-5, 5, (100, 2)).astype(np.float32)
    above_z = rng.uniform(-0.5, 2.0, (100,)).astype(np.float32)
    above = np.column_stack([above_xy, above_z])
    points = np.vstack([ground, above])

    R, t = fit_dominant_plane(points)

    assert R.shape == (3, 3)
    assert t.shape == (3,)
    # After applying transform, floor z-mean should be ≈ 0
    pts_aligned = (R @ points[:800].T).T + t
    np.testing.assert_allclose(pts_aligned[:, 2].mean(), 0.0, atol=0.1)


def test_fit_dominant_plane_returns_valid_rotation():
    """R is a proper rotation matrix (det=1, orthogonal)."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    pts[:400, 2] = rng.normal(0, 0.01, 400)  # flat-ish ground at z=0
    R, t = fit_dominant_plane(pts)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
