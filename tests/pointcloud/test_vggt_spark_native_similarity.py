"""Tests for VGGTSPARKCreator._verify_loop_candidate native similarity override."""
from unittest.mock import MagicMock

import pytest
import torch


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
    creator.model = _stub_model({"image_match_ratio": torch.tensor(0.97)})
    frame = torch.zeros(3, 224, 224)
    accepted, poses = creator._verify_loop_candidate(frame, frame)
    assert accepted is True
    assert poses is None
    # Model called with compute_similarity=True
    call_kwargs = creator.model.call_args
    assert call_kwargs.kwargs.get("compute_similarity") is True


def test_native_verify_rejects_below_threshold():
    creator = _make_creator()
    creator.model = _stub_model({"image_match_ratio": torch.tensor(0.90)})
    frame = torch.zeros(3, 224, 224)
    accepted, poses = creator._verify_loop_candidate(frame, frame)
    assert accepted is False
    assert poses is None


def test_native_verify_respects_custom_threshold():
    creator = _make_creator()
    creator.model = _stub_model({"image_match_ratio": torch.tensor(0.93)})
    frame = torch.zeros(3, 224, 224)
    accepted, _ = creator._verify_loop_candidate(frame, frame)
    assert accepted is False  # default 0.95 → rejected
    accepted, _ = creator._verify_loop_candidate(frame, frame, verify_match_ratio=0.90)
    assert accepted is True  # 0.90 threshold → accepted


def test_native_verify_stacks_two_frames_as_input():
    creator = _make_creator()
    captured = {}

    def fake_forward(images, compute_similarity=False):
        captured["shape"] = tuple(images.shape)
        return {"image_match_ratio": torch.tensor(0.97)}

    # MagicMock wrapper keeps a real parameters() for next(...).to(device, dtype)
    # while side_effect runs fake_forward to capture the stacked input shape.
    creator.model = MagicMock(side_effect=fake_forward)
    creator.model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    frame = torch.zeros(3, 112, 112)
    creator._verify_loop_candidate(frame, frame)
    assert captured["shape"] == (2, 3, 112, 112)
