import numpy as np
import pytest
from collab_splats.utils.geometry import (
    OPENGL_TO_OPENCV,
    OPENCV_TO_OPENGL,
    extrinsics_to_homogeneous,
    invert_poses,
    extract_intrinsics,
)


class TestExtrinsicsToHomogeneous:
    def test_batched(self):
        ext = np.random.rand(4, 3, 4).astype(np.float32)
        out = extrinsics_to_homogeneous(ext)
        assert out.shape == (4, 4, 4)
        np.testing.assert_array_equal(out[:, 3, :], [[0, 0, 0, 1]] * 4)
        np.testing.assert_array_equal(out[:, :3, :], ext)

    def test_single(self):
        ext = np.random.rand(3, 4).astype(np.float32)
        out = extrinsics_to_homogeneous(ext)
        assert out.shape == (4, 4)
        np.testing.assert_array_equal(out[3, :], [0, 0, 0, 1])
        np.testing.assert_array_equal(out[:3, :], ext)

    def test_dtype_preserved(self):
        ext = np.random.rand(2, 3, 4).astype(np.float64)
        out = extrinsics_to_homogeneous(ext)
        assert out.dtype == np.float64


class TestInvertPoses:
    def _random_rigid(self, *shape):
        # Build valid rotation via QR decomposition
        A = np.random.randn(*shape, 3, 3)
        Q, _ = np.linalg.qr(A)
        t = np.random.randn(*shape, 3, 1)
        poses = np.zeros(shape + (4, 4))
        poses[..., :3, :3] = Q
        poses[..., :3, 3:] = t
        poses[..., 3, 3] = 1.0
        return poses.astype(np.float64)

    def test_single_roundtrip(self):
        T = self._random_rigid()
        np.testing.assert_allclose(invert_poses(T) @ T, np.eye(4), atol=1e-10)

    def test_batched_roundtrip(self):
        T = self._random_rigid(5)
        result = invert_poses(T) @ T
        np.testing.assert_allclose(result, np.eye(4)[None].repeat(5, 0), atol=1e-10)

    def test_arbitrary_batch_shape(self):
        T = self._random_rigid(3, 7)
        result = invert_poses(T) @ T
        eye = np.eye(4)[None, None].repeat(3, 0).repeat(7, 1)
        np.testing.assert_allclose(result, eye, atol=1e-10)

    def test_dtype_preserved(self):
        T = self._random_rigid().astype(np.float32)
        assert invert_poses(T).dtype == np.float32


class TestExtractIntrinsics:
    def test_basic(self):
        K = np.array([[500.0, 0, 320.0], [0, 480.0, 240.0], [0, 0, 1.0]])
        fx, fy, cx, cy = extract_intrinsics(K)
        assert fx == 500.0
        assert fy == 480.0
        assert cx == 320.0
        assert cy == 240.0

    def test_returns_floats(self):
        K = np.eye(3, dtype=np.float32)
        fx, fy, cx, cy = extract_intrinsics(K)
        assert isinstance(fx, float)


class TestConstants:
    def test_opengl_to_opencv_shape(self):
        assert OPENGL_TO_OPENCV.shape == (4, 4)

    def test_self_inverse(self):
        np.testing.assert_array_equal(OPENGL_TO_OPENCV, OPENCV_TO_OPENGL)

    def test_flips_yz(self):
        expected = np.diag([1, -1, -1, 1]).astype(np.float64)
        np.testing.assert_array_equal(OPENGL_TO_OPENCV, expected)
