import numpy as np
import pytest
import pycolmap
import torch
from collab_splats.pointcloud.feedforward import build_pycolmap_reconstruction


def _make_inputs(n=3, p=50):
    pts3d = np.random.randn(p, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (p, 3), dtype=np.uint8)
    extrinsics = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)  # (n, 4, 4)
    intrinsics = np.tile(
        np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    names = [f"frame_{i:04d}.jpg" for i in range(n)]
    return pts3d, colors, extrinsics, intrinsics, names


def test_camera_image_point_counts():
    pts3d, colors, extrinsics, intrinsics, names = _make_inputs(n=3, p=50)
    recon = build_pycolmap_reconstruction(pts3d, colors, extrinsics, intrinsics, 512, 512, names)
    assert len(recon.cameras) == 3
    assert len(recon.images) == 3
    assert len(recon.points3D) == 50


def test_accepts_4x4_extrinsics():
    pts3d, colors, _, intrinsics, names = _make_inputs(n=2, p=10)
    extrinsics_4x4 = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)  # (2, 4, 4)
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics_4x4, intrinsics, 512, 512, names[:2]
    )
    assert len(recon.cameras) == 2


def test_loop_closure_stripped_from_base():
    """After refactor, BaseFeedforwardCreator must NOT have enable_loop_closure or loop_closure_config."""
    import dataclasses
    from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator
    field_names = {f.name for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "enable_loop_closure" not in field_names
    assert "loop_closure_config" not in field_names


def test_simple_pinhole_model():
    pts3d, colors, extrinsics, intrinsics, names = _make_inputs(n=2, p=5)
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, 518, 518, names[:2],
        camera_model="SIMPLE_PINHOLE",
    )
    assert len(recon.cameras) == 2


# ---------------------------------------------------------------------------
# _verify_loop_candidate on BaseFeedforwardCreator
# ---------------------------------------------------------------------------

from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator


class _StubCreator(BaseFeedforwardCreator):
    """Minimal concrete subclass of BaseFeedforwardCreator for base-method tests.

    Implements all abstract methods as no-ops.  Instantiate via object.__new__ to
    bypass the dataclass __init__ (no real model or paths needed for these tests).
    Set creator._stubbed_features before calling _verify_loop_candidate.
    """

    def _load_model(self, device): return None
    def _preprocess(self, image_paths, **kwargs): pass
    def _forward(self, model, views, **kwargs): pass
    def _postprocess(self, raw_outputs, **kwargs): pass
    def _reproject_ba(self, raw_outputs, extrinsics_3x4, intrinsics): pass

    def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
        # Return the pre-configured stub features for this test
        return self._stubbed_features

    def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
        return np.zeros((0, 3)), np.zeros((0, 3))


def _make_stub() -> _StubCreator:
    """Bypass the dataclass __init__; no real model or paths needed."""
    from unittest.mock import MagicMock
    creator = object.__new__(_StubCreator)
    # _verify_loop_candidate calls next(self.model.parameters()).device
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    creator.model = mock_model
    return creator


def _high_ratio_features():
    """Build q/k tensors whose cross-frame attention ratio will be > 0.8."""
    torch.manual_seed(0)
    B, heads, hd = 1, 2, 4
    half = torch.randn(B, heads, 10, hd) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    return k, q


def _low_ratio_features():
    """Build q/k tensors whose cross-frame attention ratio will be < 0.2."""
    B, heads, hd = 1, 1, 4
    N = 20
    k = torch.zeros(B, heads, N, hd)
    q = torch.zeros(B, heads, N, hd)
    # First frame in dim 0, second frame in orthogonal dim 1
    k[:, :, :10, 0] = 10.0
    q[:, :, :10, 0] = 10.0
    k[:, :, 10:, 1] = 10.0
    q[:, :, 10:, 1] = 10.0
    return k, q


def test_verify_loop_candidate_rejected():
    """Ratio below threshold → (False, None) regardless of poses presence."""
    creator = _make_stub()
    k, q = _low_ratio_features()
    creator._stubbed_features = {"q": q, "k": k}
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, poses = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.85, layer_index=-1
    )
    assert accepted is False
    assert poses is None


def test_verify_loop_candidate_accepted_no_poses():
    """Ratio above threshold but no 'poses' key → (True, None) — contract violation
    the wrapper call site guards against (rejects with 'no_joint_poses')."""
    creator = _make_stub()
    k, q = _high_ratio_features()
    # No "poses" key — backend does not produce decoded poses
    creator._stubbed_features = {"q": q, "k": k}
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, lc_data = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.5, layer_index=-1
    )
    assert accepted is True
    assert lc_data is None


def test_verify_loop_candidate_accepted_with_poses():
    """Ratio above threshold + 'poses' key present → (True, lc_data dict)."""
    creator = _make_stub()
    k, q = _high_ratio_features()
    fake_poses = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)  # (2, 4, 4)
    creator._stubbed_features = {"q": q, "k": k, "poses": fake_poses}
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, lc_data = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.5, layer_index=-1
    )
    assert accepted is True
    assert lc_data is not None
    assert lc_data["poses"].shape == (2, 4, 4)
    # Backend supplied no geometry — lc_data carries explicit None placeholders
    assert lc_data["world_points"] is None
    assert lc_data["conf"] is None


def test_verify_loop_candidate_accepted_with_geometry():
    """'world_points'/'conf' feature keys are folded into lc_data on accept."""
    creator = _make_stub()
    k, q = _high_ratio_features()
    fake_poses = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)   # (2, 4, 4)
    wp = np.zeros((2, 8, 8, 3), dtype=np.float32)                      # (2, H, W, 3)
    conf = np.ones((2, 8, 8), dtype=np.float32)                        # (2, H, W)
    creator._stubbed_features = {
        "q": q, "k": k, "poses": fake_poses, "world_points": wp, "conf": conf
    }
    accepted, lc_data = creator._verify_loop_candidate(
        torch.zeros(3, 16, 16), torch.zeros(3, 16, 16),
        verify_match_ratio=0.5, layer_index=-1,
    )
    assert accepted is True
    assert lc_data["world_points"].shape == (2, 8, 8, 3)
    assert lc_data["conf"].shape == (2, 8, 8)


def test_verify_loop_candidate_layer_index_forwarded():
    """layer_index is passed through to extract_intermediate_features."""
    creator = _make_stub()
    captured_index = []

    def _capture_extract(frames, layer_index=-1, **kwargs):
        captured_index.append(layer_index)
        k, q = _high_ratio_features()
        return {"q": q, "k": k}

    creator.extract_intermediate_features = _capture_extract
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    creator._verify_loop_candidate(frame1, frame2, verify_match_ratio=0.5, layer_index=2)
    assert captured_index == [2]


def test_extract_intermediate_features_is_abstract():
    """BaseFeedforwardCreator.extract_intermediate_features must be abstract."""
    import inspect
    abstracts = {
        name for name, _ in inspect.getmembers(BaseFeedforwardCreator)
        if getattr(getattr(BaseFeedforwardCreator, name, None), '__isabstractmethod__', False)
    }
    assert 'extract_intermediate_features' in abstracts
