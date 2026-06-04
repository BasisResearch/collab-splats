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


def test_load_computes_view_transform_by_default():
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    assert v._view_T is not None  # normalization on by default


def test_normalize_toggle_off_drops_transform_and_rerenders():
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v.set_normalize_view(False)
    assert v._view_T is None
    assert v.left_actor is not None  # re-rendered without crashing
    v.set_normalize_view(True)
    assert v._view_T is not None


def test_set_mode_pcd_to_mesh_toggles(tmp_path):
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v.set_mode("mesh")  # no mesh.ply -> status set, no crash
    assert v.mode == "mesh"
    assert "not found" in v._status
    v.set_mode("pointcloud")
    assert v.mode == "pointcloud"


def test_recolor_by_similarity_updates_right(monkeypatch):
    v = SplitViewer(off_screen=True)
    res = _FakeResult(p=20)
    v.load(res, mesh_path=None)
    # Inject lifted features directly (normalised) and a fake queryable extractor
    v._lifted_normed = np.random.rand(20, 8).astype(np.float32)

    seen = {}

    class _Ext:
        def score_queries(self, features, positive, negative=None):
            import torch

            seen["positive"] = positive
            seen["negative"] = negative
            return torch.rand(features.shape[0])

    monkeypatch.setattr(v, "_get_extractor", lambda name: _Ext())
    colors = v.score_query(positive=["chair", "stool"], negative=["floor"], extractor_name="talk2dino")
    assert colors.shape == (20, 3)
    assert colors.dtype == np.uint8
    assert seen["positive"] == ["chair", "stool"]
    assert seen["negative"] == ["floor"]


def test_query_empty_positive_resets_right(monkeypatch):
    v = SplitViewer(off_screen=True)
    res = _FakeResult(p=12)
    v.load(res, mesh_path=None)
    v._lifted_normed = np.random.rand(12, 8).astype(np.float32)
    # No positive terms -> reset to RGB, no extractor call
    colors = v.score_query(positive=[], negative=[], extractor_name="talk2dino")
    assert np.array_equal(colors, res.colors)


from collab_splats.dashboard.viewer import _decimate_indices


def test_decimate_indices_caps_to_budget():
    idx = _decimate_indices(n=1000, max_points=150)
    assert idx.shape[0] == 150
    assert idx.max() < 1000
    assert len(np.unique(idx)) == 150  # no duplicates


def test_decimate_indices_noop_when_under_budget():
    idx = _decimate_indices(n=100, max_points=150)
    assert idx.shape[0] == 100
    assert np.array_equal(idx, np.arange(100))


def test_decimate_indices_nonpositive_budget_is_noop():
    idx = _decimate_indices(n=100, max_points=0)
    assert np.array_equal(idx, np.arange(100))


from unittest.mock import MagicMock


def test_score_query_returns_colors_without_rendering():
    v = SplitViewer(off_screen=True)
    v._result = MagicMock()
    v._result.colors = np.zeros((4, 3), dtype=np.uint8)
    v._lifted_normed = np.eye(4, dtype=np.float32)
    fake = MagicMock()
    import torch

    fake.score_queries.return_value = torch.tensor([0.1, 0.9, 0.5, 0.2])
    v._extractor_cache["talk2dino"] = fake
    colors = v.score_query(positive=["chair"], negative=["floor"], extractor_name="talk2dino")
    assert colors.shape == (4, 3)
    fake.score_queries.assert_called_once()


def test_score_query_blank_positive_returns_rgb():
    v = SplitViewer(off_screen=True)
    v._result = MagicMock()
    v._result.colors = np.full((4, 3), 7, dtype=np.uint8)
    v._lifted_normed = None
    colors = v.score_query(positive=[], negative=[], extractor_name="talk2dino")
    assert np.array_equal(colors, v._result.colors)


def test_score_query_lazily_lifts_on_first_query(monkeypatch):
    import torch

    v = SplitViewer(off_screen=True)
    v._result = MagicMock()
    v._result.colors = np.zeros((4, 3), dtype=np.uint8)
    v._lifted_normed = None
    v._semantics_dir = "semdir"  # set by load(); triggers lazy lift

    called = {}

    def fake_lift(result, semantics_dir):
        called["dir"] = semantics_dir
        return np.eye(4, dtype=np.float32)

    monkeypatch.setattr("collab_splats.dashboard.viewer.load_lifted_normed", fake_lift)
    ext = MagicMock()
    ext.score_queries.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(v, "_get_extractor", lambda n: ext)

    colors = v.score_query(positive=["chair"], extractor_name="talk2dino")
    assert called["dir"] == "semdir"  # lifted lazily on first query
    assert colors.shape == (4, 3)


def test_score_query_blank_positive_does_not_lift(monkeypatch):
    v = SplitViewer(off_screen=True)
    v._result = MagicMock()
    v._result.colors = np.full((4, 3), 5, dtype=np.uint8)
    v._lifted_normed = None
    v._semantics_dir = "semdir"
    lifted_called = {"n": 0}

    def fake_lift(result, semantics_dir):
        lifted_called["n"] += 1
        return np.eye(4, dtype=np.float32)

    monkeypatch.setattr("collab_splats.dashboard.viewer.load_lifted_normed", fake_lift)
    v.score_query(positive=[], extractor_name="talk2dino")
    assert lifted_called["n"] == 0  # blank query must not pay the 6-min lift
