import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.geometry import LoopClosure
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator


def _make_result(n=3, p=50, colors=None):
    """
    n PINHOLE frames on a 512x512 full frame, p random points.
    """
    rng = np.random.default_rng(0)
    intrinsics = np.tile(
        np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return PointcloudResult(
        points=rng.standard_normal((p, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (p, 3), dtype=np.uint8)
        if colors is None
        else colors,
        extrinsics=np.tile(np.eye(4), (n, 1, 1)).astype(np.float32),
        intrinsics=None,
        model_intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i:04d}.jpg") for i in range(n)],
        original_coords=np.array([[0, 0, 512, 512, 512, 512]] * n, dtype=np.float32),
        model_width=512,
        model_height=512,
    )


def test_camera_image_point_counts():
    recon = _make_result(n=3, p=50).to_colmap()
    assert len(recon.cameras) == 3
    assert len(recon.images) == 3
    assert len(recon.points3D) == 50


def test_loop_closure_stripped_from_base():
    """After refactor, BaseFeedforwardCreator must NOT have enable_loop_closure or loop_closure_config."""
    import dataclasses

    from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator

    field_names = {f.name for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "enable_loop_closure" not in field_names
    assert "loop_closure_config" not in field_names


def test_to_colmap_rejects_float_colors():
    result = _make_result(p=50, colors=np.full((50, 3), 0.5, dtype=np.float32))
    with pytest.raises(TypeError, match="uint8"):
        result.to_colmap()


# ---------------------------------------------------------------------------
# _verify_loop_candidate on BaseFeedforwardCreator
# ---------------------------------------------------------------------------


class _StubCreator(BaseFeedforwardCreator):
    """Minimal concrete subclass of BaseFeedforwardCreator for base-method tests.

    Implements all abstract methods as no-ops.  Instantiate via object.__new__ to
    bypass the dataclass __init__ (no real model or paths needed for these tests).
    Set creator._stubbed_features before calling _verify_loop_candidate.
    """

    def _load_model(self, device):
        return None

    def _preprocess(self, image_paths, **kwargs):
        pass

    def _forward(self, model, views):
        pass

    def _postprocess(self, raw_outputs):
        pass

    # LC calibration an LC-capable backend sets
    _lc_layer_index = 20
    _lc_token_offset = 5

    def extract_intermediate_features(self, frames, layer_index):
        # Return the pre-configured stub features for this test
        return self._stubbed_features


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
        frame1, frame2, verify_match_ratio=0.85
    )
    assert accepted is False
    assert poses is None


def test_verify_raises_when_backend_supplies_no_poses():
    creator = _make_stub()
    k, q = _high_ratio_features()
    creator._stubbed_features = {"q": q, "k": k}
    frame = torch.zeros(3, 16, 16)
    with pytest.raises(KeyError, match="poses"):
        creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.5)


def test_verify_loop_candidate_accepted_with_geometry():
    """'world_points'/'conf' feature keys are folded into lc_data on accept."""
    creator = _make_stub()
    k, q = _high_ratio_features()
    fake_poses = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)  # (2, 4, 4)
    wp = np.zeros((2, 8, 8, 3), dtype=np.float32)  # (2, H, W, 3)
    conf = np.ones((2, 8, 8), dtype=np.float32)  # (2, H, W)
    creator._stubbed_features = {
        "q": q,
        "k": k,
        "poses": fake_poses,
        "world_points": wp,
        "conf": conf,
    }
    accepted, lc_data = creator._verify_loop_candidate(
        torch.zeros(3, 16, 16),
        torch.zeros(3, 16, 16),
        verify_match_ratio=0.5,
    )
    assert accepted is True
    assert lc_data["world_points"].shape == (2, 8, 8, 3)
    assert lc_data["conf"].shape == (2, 8, 8)


def test_verify_taps_the_calibrated_layer():
    creator = _make_stub()
    seen = {}

    def _extract(frames, layer_index):
        seen["layer"] = layer_index
        k, q = _high_ratio_features()
        return {
            "q": q,
            "k": k,
            "poses": np.zeros((2, 4, 4), np.float32),
            "world_points": np.zeros((2, 1, 1, 3), np.float32),
            "conf": np.ones((2, 1, 1), np.float32),
        }

    creator._lc_layer_index = 7
    creator.extract_intermediate_features = _extract
    frame = torch.zeros(3, 16, 16)
    creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.0)
    assert seen["layer"] == 7


def test_loop_closure_raises_without_calibrated_ratio():
    base = _make_stub()
    with pytest.raises(NotImplementedError, match="default_verify_match_ratio"):
        LoopClosure(base)


def test_extract_intermediate_features_base_raises():
    creator = _make_stub()
    with pytest.raises(NotImplementedError):
        BaseFeedforwardCreator.extract_intermediate_features(
            creator, torch.zeros(2, 3, 4, 4), 0
        )


def test_base_creator_no_extractor_name():
    """extractor_name field removed; creators no longer own lifting."""
    fields = {f.name for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "extractor_name" not in fields


########################################################
########## _postprocess pixel bookkeeping ##############
########################################################


def _postprocess(
    depth_conf: np.ndarray, images: torch.Tensor, conf_percentile: float
) -> PointcloudResult:
    """
    Base _postprocess over flat depth-2 frames with these confidences and images.
    """
    n, _, h, w = images.shape
    intrinsics = np.array(
        [[w, 0.0, (w - 1) / 2], [0.0, w, (h - 1) / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    raw = {
        "images": images,
        "extrinsic": np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1)),
        "intrinsics": np.tile(intrinsics, (n, 1, 1)),
        "depth": np.full((n, h, w), 2.0, dtype=np.float32),
        "depth_conf": depth_conf,
    }
    creator = object.__new__(_StubCreator)
    creator.conf_percentile = conf_percentile
    creator.min_views = 0
    creator.image_paths = [Path(f"frame_{i:06d}") for i in range(n)]
    creator.original_coords = np.tile(
        np.array([0, 0, w, h, w, h], dtype=np.float32), (n, 1)
    )
    return BaseFeedforwardCreator._postprocess(creator, raw)


def test_postprocess_returns_pixel_indices():
    n, h, w = 3, 8, 8
    depth_conf = np.random.default_rng(0).random((n, h, w)).astype(np.float32)
    out = _postprocess(depth_conf, torch.zeros(n, 3, h, w), conf_percentile=0.5)

    assert out.pixel_indices.shape == (len(out.points), 3)
    assert out.pixel_indices.dtype == np.int32
    assert (out.pixel_indices.min(axis=0) >= 0).all()
    assert (out.pixel_indices.max(axis=0) < [n, h, w]).all()


def test_pixel_indices_align_with_colors():
    """colors[p] must come from the same pixel as pixel_indices[p]."""
    n, h, w = 2, 8, 8

    # Encode pixel identity into image: pixel (r, c) = r*10 + c across all channels
    images = torch.zeros(n, 3, h, w)
    for r in range(h):
        for c in range(w):
            images[:, :, r, c] = (r * 10 + c) / 255.0

    out = _postprocess(
        np.ones((n, h, w), dtype=np.float32), images, conf_percentile=0.0
    )

    images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    fi, ri, ci = out.pixel_indices.T
    np.testing.assert_array_equal(out.colors, images_np[fi, ri, ci])
