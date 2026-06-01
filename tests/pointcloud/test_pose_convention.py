"""Tests for F5 (default verifier) and F7 (pose convention assertion)."""
import numpy as np
import pytest



def test_assert_world_to_cam_passes_identity_first():
    from collab_splats.pointcloud.loop_closure.submap import assert_world_to_cam
    poses = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    poses[1:, :3, 3] = np.random.randn(4, 3).astype(np.float32)  # non-identity rest
    assert_world_to_cam(poses)  # must not raise


def test_assert_world_to_cam_raises_non_identity_first():
    from collab_splats.pointcloud.loop_closure.submap import assert_world_to_cam
    poses = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    poses[0, :3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # non-identity first
    with pytest.raises(ValueError, match="Pose convention violation"):
        assert_world_to_cam(poses)


def test_assert_world_to_cam_wrong_shape():
    from collab_splats.pointcloud.loop_closure.submap import assert_world_to_cam
    with pytest.raises(ValueError, match="poses must be"):
        assert_world_to_cam(np.eye(4))
