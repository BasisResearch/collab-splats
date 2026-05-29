import numpy as np
import pytest
from collab_splats.utils.geometry import (
    OPENGL_TO_OPENCV,
    extrinsics_to_homogeneous,
    invert_poses,
    extract_intrinsics,
    rotation_align_vectors,
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


