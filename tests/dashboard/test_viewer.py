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


def test_query_similarity_persists_across_mode_switch():
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)
    assert v.active_query() is None
    # Simulate a completed pointcloud query (cache colours for the current mode).
    colors = (np.random.rand(20, 3) * 255).astype(np.uint8)
    v._last_query = (["chair"], [], "talk2dino")
    v._query_colors["pointcloud"] = colors
    # Switch to mesh: query still active, mesh has no cached colours yet (app would re-score).
    v.set_mode("mesh")
    assert v.active_query() == (["chair"], [], "talk2dino")
    assert v.cached_query_colors("mesh") is None
    # Switch back: pointcloud similarity is cached -> reused, no re-score.
    v.set_mode("pointcloud")
    assert v.cached_query_colors("pointcloud") is colors


def test_load_resets_query_cache():
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v._last_query = (["x"], [], "talk2dino")
    v._query_colors["pointcloud"] = np.zeros((20, 3), np.uint8)
    v.load(_FakeResult(), mesh_path=None)  # new scene
    assert v.active_query() is None
    assert v.cached_query_colors("pointcloud") is None


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


def test_load_mesh_vertex_features_normalizes(tmp_path):
    from collab_splats.dashboard.viewer import load_mesh_vertex_features

    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    feats = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    np.save(mesh_dir / "vertex_features.npy", feats)

    out = load_mesh_vertex_features(mesh_dir)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)


def test_load_mesh_vertex_features_missing_returns_none(tmp_path):
    from collab_splats.dashboard.viewer import load_mesh_vertex_features

    assert load_mesh_vertex_features(tmp_path / "nope") is None


def test_score_query_uses_mesh_features_in_mesh_mode(monkeypatch):
    import torch

    from collab_splats.dashboard.viewer import SplitViewer

    captured = {}

    class _StubExtractor:
        def score_queries(self, features, positive, negative=None):
            captured["n"] = features.shape[0]
            return torch.ones(features.shape[0])

    v = SplitViewer.__new__(SplitViewer)  # bypass __init__ (no GUI in tests)
    v.mode = "mesh"
    v._result = type("R", (), {"colors": np.zeros((5, 3), dtype=np.uint8)})()
    v._lifted_normed = np.ones((5, 4), dtype=np.float32)  # 5 points
    v._mesh_vertex_features = np.ones((3, 4), dtype=np.float32)  # 3 vertices
    v._extractor_cache = {"talk2dino": _StubExtractor()}
    monkeypatch.setattr(v, "ensure_lifted", lambda op_log=None: None)

    colors = v.score_query(positive=["chair"], op_log=None)

    assert captured["n"] == 3
    assert colors.shape[0] == 3


def _write_tiny_mesh(path):
    import open3d as o3d

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((3, 3)) * 0.5)
    o3d.io.write_triangle_mesh(str(path), mesh)
    return verts


def test_render_right_colors_mesh_per_vertex(tmp_path):
    mesh_path = tmp_path / "mesh_tsdf.ply"
    verts = _write_tiny_mesh(mesh_path)

    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=mesh_path)
    v.mode = "mesh"

    # Per-vertex colors aligned with the 3 mesh vertices.
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    v._render_right(colors)
    assert v.right_actor is not None


def test_render_right_mesh_size_mismatch_falls_back(tmp_path):
    mesh_path = tmp_path / "mesh_tsdf.ply"
    _write_tiny_mesh(mesh_path)

    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=mesh_path)
    v.mode = "mesh"

    # 20 point-length colors != 3 mesh vertices -> plain RGB mesh, no crash.
    colors = (np.random.rand(20, 3) * 255).astype(np.uint8)
    v._render_right(colors)
    assert v.right_actor is not None


def test_mesh_read_from_disk_once_across_renders(tmp_path, monkeypatch):
    """Mesh is read once at load and reused; mode/normalize/query don't re-read from disk."""
    import collab_splats.dashboard.viewer as viewer_mod

    mesh_path = tmp_path / "mesh_tsdf.ply"
    _write_tiny_mesh(mesh_path)

    # Count pv.read calls from BEFORE load: the read-at-load is the single allowed read.
    reads = {"n": 0}
    real_read = viewer_mod.pv.read

    def counting_read(path):
        reads["n"] += 1
        return real_read(path)

    monkeypatch.setattr(viewer_mod.pv, "read", counting_read)

    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=mesh_path)
    cached_points = v._mesh_polydata.points.copy()  # raw world-space geometry at load
    v.set_mode("mesh")
    v.set_mode("pointcloud")
    v.set_mode("mesh")
    v.set_normalize_view(False)
    assert reads["n"] == 1  # load reads exactly once; renders reuse the cache
    # Renders (incl. normalized mesh renders) must never mutate the cached geometry.
    np.testing.assert_array_equal(v._mesh_polydata.points, cached_points)


def _write_mesh_6v(path):
    import open3d as o3d

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 4]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    o3d.io.write_triangle_mesh(str(path), mesh)
    return verts


def test_score_query_lazy_transfers_mesh_features_when_absent(tmp_path):
    import torch

    mesh_path = tmp_path / "mesh_tsdf.ply"
    verts = _write_mesh_6v(mesh_path)

    captured = {}

    class _Stub:
        def score_queries(self, features, positive, negative=None):
            captured["n"] = features.shape[0]
            return torch.ones(features.shape[0])

    v = SplitViewer(off_screen=True)
    res = _FakeResult(p=6)
    res.points = verts.astype(np.float32)  # points sit on the 6 mesh vertices
    v.load(res, mesh_path=mesh_path)
    assert v._mesh_vertex_features is None  # no persisted vertex_features.npy (old run)

    v.mode = "mesh"
    v._lifted_normed = np.eye(6, 4, dtype=np.float32)  # 6 point features
    v._extractor_cache = {"talk2dino": _Stub()}

    colors = v.score_query(positive=["x"], op_log=None)

    # Lazily transferred + cached, scored the 6 vertices (not the 6 points by coincidence:
    # assert the cache was populated to prove the transfer ran).
    assert v._mesh_vertex_features is not None
    assert v._mesh_vertex_features.shape[0] == 6
    assert captured["n"] == 6
    assert colors.shape[0] == 6
