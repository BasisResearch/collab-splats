"""Tests for F5 (default verifier) and F7 (pose convention assertion)."""
import numpy as np
import pytest


def test_default_verifier_raises():
    """Base class _verify_loop_candidate must raise NotImplementedError."""
    from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator

    class _DummyCreator(BaseFeedforwardCreator):
        def _load_model(self, device): pass
        def _forward(self, model, views, **kw): return {}
        def _preprocess(self, image_dir): return None, [], np.zeros((0, 2))
        def _postprocess(self, raw, **kw):
            from collab_splats.pointcloud.feedforward import FeedforwardResult
            return FeedforwardResult(points=np.zeros((1, 3)), colors=np.zeros((1, 3)),
                                     extrinsics=np.eye(4)[None], intrinsics=np.eye(3)[None],
                                     image_paths=[], model_width=1, model_height=1)

    dummy = object.__new__(_DummyCreator)
    with pytest.raises(NotImplementedError, match="_verify_loop_candidate"):
        dummy._verify_loop_candidate(None, None)


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
