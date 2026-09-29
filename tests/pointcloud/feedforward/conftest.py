"""Shared fakes for feedforward LC tests."""

import types
from unittest.mock import patch

import numpy as np
import torch

from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator
from tests.pointcloud.conftest import _sized_paths

# _FakeQKV / _FakeMapAnythingModel moved verbatim from test_verify_lc_data.py
# (the copy in test_mapanything_creator.py was byte-identical).
# Consumers import these classes directly rather than via fixtures — deliberate:
# the constructors take args (canned preds), which zero-arg fixtures can't supply.


class _FakeQKV(torch.nn.Module):
    """Real nn.Module so register_forward_hook works; never actually called."""

    def forward(self, x):
        return x


class _FakeMapAnythingModel(torch.nn.Module):
    """Minimal stand-in: info_sharing block tree + forward returning canned preds."""

    def __init__(self, preds):
        super().__init__()
        self._preds = preds
        self._param = torch.nn.Parameter(torch.zeros(1))
        attn = types.SimpleNamespace(num_heads=2, qkv=_FakeQKV())
        block = types.SimpleNamespace(attn=attn)
        self.info_sharing = types.SimpleNamespace(self_attention_blocks=[block])

    def forward(self, views, memory_efficient_inference=False, minibatch_size=1):
        return self._preds


def _mapanything_boxes(sizes: list[tuple[int, int]], model_w: int, model_h: int) -> np.ndarray:
    """(N, 6) boxes MapAnythingCreator._preprocess returns for (w, h) frames on a (model_w, model_h) grid."""
    paths, image = _sized_paths(sizes)
    views = [{"img": torch.empty(1, 3, model_h, model_w)}]
    with (
        patch("collab_splats.pointcloud.feedforward.mapanything.Image", image),
        patch("collab_splats.pointcloud.feedforward.mapanything.load_images", return_value=views),
        patch("collab_splats.pointcloud.feedforward.mapanything.validate_input_views_for_inference"),
        patch("collab_splats.pointcloud.feedforward.mapanything.preprocess_input_views_for_inference"),
    ):
        _, coords = MapAnythingCreator()._preprocess(paths)
    return coords
