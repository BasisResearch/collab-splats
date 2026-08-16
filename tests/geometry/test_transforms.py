import numpy as np
import pytest
from collab_splats.geometry.transforms import (
    OPENGL_TO_OPENCV,
    estimate_intrinsics_from_points,
    extrinsics_to_homogeneous,
    invert_poses,
    extract_intrinsics,
    rotation_align_vectors,
    _compute_weighted_median,
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

