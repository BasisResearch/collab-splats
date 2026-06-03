"""Tests for SplitViewer: off-screen, synthetic data, no GPU/models."""

import numpy as np
import pytest

from collab_splats.dashboard.viewer import SplitViewer


class _FakeResult:
    def __init__(self, p=20):
        self.points = np.random.rand(p, 3).astype(np.float32)
        self.colors = (np.random.rand(p, 3) * 255).astype(np.uint8)


def test_viewer_loads_pointcloud_offscreen():
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    assert v.left_actor is not None


def test_set_mode_pcd_to_mesh_toggles(tmp_path):
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v.set_mode("mesh")          # no mesh.ply -> status set, no crash
    assert v.mode == "mesh"
    v.set_mode("pointcloud")
    assert v.mode == "pointcloud"


def test_recolor_by_similarity_updates_right(monkeypatch):
    v = SplitViewer(off_screen=True)
    res = _FakeResult(p=20)
    v.load(res, mesh_path=None)
    # Inject lifted features directly (normalised) and a fake extractor
    v._lifted_normed = np.random.rand(20, 8).astype(np.float32)

    class _Ext:
        def encode_text(self, texts):
            import torch
            return torch.ones(1, 8)

    monkeypatch.setattr(v, "_get_extractor", lambda name: _Ext())
    colors = v.query("chair", extractor_name="talk2dino")
    assert colors.shape == (20, 3)
    assert colors.dtype == np.uint8
