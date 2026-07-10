"""Tests for VGGTSPARKCreator._verify_loop_candidate native similarity override."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _full_outputs(ratio: float, h: int = 112, w: int = 112) -> dict:
    """Forward outputs with pose_enc/depth/depth_conf — the SPARK forward always returns these.

    pose_enc encodes identity rotation (quat xyzw = (0,0,0,1)), zero translation,
    and fov ≈ 1 rad so intrinsics decode to finite values.
    """
    pose_enc = torch.zeros(1, 2, 9)
    pose_enc[..., 6] = 1.0
    pose_enc[..., 7:9] = 1.0
    return {
        "image_match_ratio": torch.tensor(ratio),
        "pose_enc": pose_enc,
        "depth": torch.ones(1, 2, h, w, 1),
        "depth_conf": torch.ones(1, 2, h, w),
    }


def _make_creator():
    """Build a minimal VGGTSPARKCreator without loading a model."""
    from collab_splats.pointcloud.feedforward.vggt_spark_creator import VGGTSPARKCreator

    creator = object.__new__(VGGTSPARKCreator)
    creator.model = None  # replaced per test
    return creator


def _stub_model(return_value):
    """MagicMock model with a real parameter so next(model.parameters()).device/.dtype work.

    Production does next(self.model.parameters()) then .to(p.device, dtype=p.dtype);
    parameters() must yield a fresh iterator each call.
    """
    model = MagicMock(return_value=return_value)
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    return model


def test_native_verify_accepts_above_threshold():
    creator = _make_creator()
    creator.model = _stub_model(_full_outputs(0.97))
    frame = torch.zeros(3, 112, 112)
    accepted, lc_data = creator._verify_loop_candidate(frame, frame)
    assert accepted is True
    # Accepting verify returns the full lc_data contract from the same forward
    assert lc_data is not None
    assert lc_data["poses"].shape == (2, 4, 4)
    assert lc_data["poses"].dtype == np.float32
    assert np.allclose(lc_data["poses"][0], np.eye(4), atol=1e-5)  # frame-0 canonical
    assert lc_data["world_points"].shape == (2, 112, 112, 3)
    assert lc_data["world_points"].dtype == np.float32
    assert lc_data["conf"].shape == (2, 112, 112)
    # Model called with compute_similarity=True
    call_kwargs = creator.model.call_args
    assert call_kwargs.kwargs.get("compute_similarity") is True


def test_native_verify_rejects_below_threshold():
    # Reject path returns before pose extraction — a ratio-only stub must suffice.
    creator = _make_creator()
    creator.model = _stub_model({"image_match_ratio": torch.tensor(0.90)})
    frame = torch.zeros(3, 224, 224)
    accepted, lc_data = creator._verify_loop_candidate(frame, frame)
    assert accepted is False
    assert lc_data is None


def test_native_verify_respects_custom_threshold():
    creator = _make_creator()
    creator.model = _stub_model(_full_outputs(0.93))
    frame = torch.zeros(3, 112, 112)
    accepted, _ = creator._verify_loop_candidate(frame, frame)
    assert accepted is False  # default 0.95 → rejected
    accepted, _ = creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.90)
    assert accepted is True  # 0.90 threshold → accepted


def test_native_verify_stacks_two_frames_as_input():
    creator = _make_creator()
    captured = {}

    def fake_forward(images, compute_similarity=False):
        captured["shape"] = tuple(images.shape)
        return _full_outputs(0.97)

    # MagicMock wrapper keeps a real parameters() for next(...).to(device, dtype)
    # while side_effect runs fake_forward to capture the stacked input shape.
    creator.model = MagicMock(side_effect=fake_forward)
    creator.model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    frame = torch.zeros(3, 112, 112)
    creator._verify_loop_candidate(frame, frame)
    assert captured["shape"] == (2, 3, 112, 112)
