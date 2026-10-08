"""Tests for SplitViewer: off-screen, synthetic data, no GPU/models."""

import numpy as np
import open3d as o3d
import pytest
import pyvista as pv
import torch

from collab_splats.dashboard import viewer as viewer_module
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.viewer import (
    SplitViewer,
    _decimate_indices,
    apply_viridis,
    compute_view_transform,
)
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.store import write_point_features

########
# Helpers
########


class _FakeResult:
    """Seeded stand-in for PointcloudResult: P random points and colors, no cameras."""

    def __init__(self, p=20):
        rng = np.random.default_rng(0)
        self.points = rng.random((p, 3)).astype(np.float32)
        self.colors = (rng.random((p, 3)) * 255).astype(np.uint8)
        self.extrinsics = None


class _CountingExtractor:
    """Stub queryable extractor that records the scored feature rows and query terms."""

    def __init__(self):
        self.features = None
        self.positive = None
        self.negative = None

    def score_queries(self, features, positive, negative=None):
        self.features = features
        self.positive = positive
        self.negative = negative
        return torch.arange(features.shape[0], dtype=torch.float32)


def _write_tiny_mesh(path):
    """Write a one-triangle grey mesh; 3 vertices."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((3, 3)) * 0.5)
    o3d.io.write_triangle_mesh(str(path), mesh)


def _write_mesh_6v(path):
    """Write a three-triangle mesh; 6 vertices."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 4]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    )
    o3d.io.write_triangle_mesh(str(path), mesh)


def _counting_read(monkeypatch):
    """Patch pv.read with an empty-PolyData stub; returns the list of paths it was called with."""
    reads = []
    monkeypatch.setattr(
        viewer_module.pv, "read", lambda p: reads.append(p) or pv.PolyData()
    )
    return reads


########
# Load, normalize and mode
########


def test_viewer_loads_pointcloud_offscreen():
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    assert v._left.actors


def test_normalize_toggle_drops_and_restores_the_view_transform():
    """Normalization is on after load; toggling it off re-renders without the transform."""
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    assert v._view_T is not None
    v.set_normalize_view(False)
    assert v._view_T is None
    assert v._left.actors
    v.set_normalize_view(True)
    assert v._view_T is not None


def test_normalize_toggle_keeps_query_heat():
    """Toggling normalization re-renders the right pane with the active query colors."""
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)
    colors = np.full((20, 3), 7, dtype=np.uint8)
    v._query_colors["pointcloud"] = colors
    v.set_normalize_view(False)
    assert np.array_equal(v._right_cloud["RGB"], colors)


def test_set_mode_without_a_mesh_logs_not_found_and_switches_back():
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v.set_mode("mesh")
    assert v.mode == "mesh"
    assert any("mesh not found" in line for line in v._op_log.log_lines)
    v.set_mode("pointcloud")
    assert v.mode == "pointcloud"


def test_view_geometry_is_built_once_and_queries_leave_it_untouched(monkeypatch):
    """Normalize toggles reuse the load-time fit; query colors never reach the shared view-space cloud."""
    calls = []
    real = viewer_module.compute_view_transform
    monkeypatch.setattr(
        viewer_module,
        "compute_view_transform",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)
    v.set_normalize_view(False)
    v.set_normalize_view(True)
    assert len(calls) == 1

    cloud = v._cloud_in_view()
    rgb_before = np.asarray(cloud["RGB"]).copy()
    v.render_query(np.full((20, 3), 7, dtype=np.uint8))
    v._render_right(np.full((20, 3), 9, dtype=np.uint8))
    assert v._cloud_in_view() is cloud
    np.testing.assert_array_equal(cloud["RGB"], rgb_before)


def test_decimate_indices_caps_to_budget():
    idx = _decimate_indices(n=1000, max_points=150)
    assert idx.shape[0] == 150
    assert idx.max() < 1000
    assert len(np.unique(idx)) == 150


@pytest.mark.parametrize(
    "max_points", [150, 0], ids=["under_budget", "nonpositive_budget"]
)
def test_decimate_indices_keeps_every_point(max_points):
    assert np.array_equal(
        _decimate_indices(n=100, max_points=max_points), np.arange(100)
    )


########
# Query cache across loads and modes
########


def test_query_similarity_persists_across_mode_switch():
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)
    assert v.active_query() is None

    # A completed pointcloud query: colors cached for the current mode
    colors = np.full((20, 3), 5, dtype=np.uint8)
    v._last_query = (["chair"], [], "talk2dino")
    v._query_colors["pointcloud"] = colors

    # Mesh mode: query still active, no cached mesh colors yet (the app re-scores)
    v.set_mode("mesh")
    assert v.active_query() == (["chair"], [], "talk2dino")
    assert v.cached_query_colors("mesh") is None

    # Back to pointcloud: its colors are reused, no re-score
    v.set_mode("pointcloud")
    assert v.cached_query_colors("pointcloud") is colors


def test_load_resets_query_cache():
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v._last_query = (["x"], [], "talk2dino")
    v._query_colors["pointcloud"] = np.zeros((20, 3), np.uint8)
    v.load(_FakeResult(), mesh_path=None)
    assert v.active_query() is None
    assert v.cached_query_colors("pointcloud") is None


########
# score_query
########


def test_score_query_forwards_terms_and_returns_uint8_colors(monkeypatch):
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)
    v._point_features = np.random.default_rng(0).random((20, 8)).astype(np.float32)
    stub = _CountingExtractor()
    monkeypatch.setattr(v, "_get_extractor", lambda name: stub)
    colors = v.score_query(
        positive=["chair", "stool"], negative=["floor"], extractor_name="talk2dino"
    )
    assert colors.shape == (20, 3)
    assert colors.dtype == np.uint8
    assert stub.positive == ["chair", "stool"]
    assert stub.negative == ["floor"]


def test_score_query_blank_positive_does_not_read_store(tmp_path, monkeypatch):
    """An empty positive query shows plain RGB without touching the lifted store."""
    reads = []
    monkeypatch.setattr(
        "collab_splats.dashboard.viewer.read_point_features", reads.append
    )
    v = SplitViewer(OperationLog(), off_screen=True)
    result = _FakeResult(20)
    v.load(result, mesh_path=None, lifted_store=tmp_path / "talk2dino_lifted.zarr")
    colors = v.score_query(positive=[], negative=[], extractor_name="talk2dino")
    assert np.array_equal(colors, result.colors)
    assert reads == []


def test_score_query_without_lifted_store_reports_and_returns_rgb():
    """No lifted store: score_query returns plain RGB and tells the console to run semantics."""
    log = OperationLog()
    v = SplitViewer(log, off_screen=True)
    result = _FakeResult(20)
    v.load(result, mesh_path=None, lifted_store=None)
    colors = v.score_query(positive=["chair"], negative=[], extractor_name="talk2dino")
    assert np.array_equal(colors, result.colors)
    assert any("run the semantics stage" in line for line in log.log_lines)


def test_score_query_targets_requested_mode(tmp_path):
    """Mode switches score in the target feature space, not the outgoing current one."""
    mesh_path = tmp_path / "mesh.ply"
    _write_mesh_6v(mesh_path)
    store = tmp_path / "talk2dino_lifted.zarr"
    vertex_features = np.eye(6, 4, dtype=np.float16)
    write_point_features(
        store,
        np.ones((20, 4), dtype=np.float32),
        None,
        vertex_arrays={"vertex_features": vertex_features},
    )
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(20), mesh_path=mesh_path, lifted_store=store)
    v.mode = "mesh"
    stub = _CountingExtractor()
    v._extractor_cache = {"talk2dino": stub}

    # Outgoing mode is mesh, but a switch to pointcloud scores the 20 points
    colors = v.score_query(
        positive=["x"], negative=[], extractor_name="talk2dino", mode="pointcloud"
    )
    assert stub.features.shape[0] == 20
    assert len(colors) == 20

    # Without a target the current (mesh) mode scores its 6 vertices
    colors = v.score_query(positive=["x"], negative=[], extractor_name="talk2dino")
    assert len(colors) == 6


def test_score_query_reads_stored_vertex_features_in_mesh_mode(tmp_path):
    """Mesh queries score the stored vertex_features once per load; all-zero rows draw grey."""
    mesh_path = tmp_path / "mesh.ply"
    _write_mesh_6v(mesh_path)
    store = tmp_path / "talk2dino_lifted.zarr"
    vertex_features = np.eye(6, 4, dtype=np.float16)
    write_point_features(
        store,
        np.ones((20, 4), dtype=np.float32),
        None,
        vertex_arrays={"vertex_features": vertex_features},
    )

    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(20), mesh_path=mesh_path, lifted_store=store)
    v.mode = "mesh"
    stub = _CountingExtractor()
    v._extractor_cache = {"talk2dino": stub}
    colors = v.score_query(positive=["x"], negative=[], extractor_name="talk2dino")

    # Six vertex rows scored, L2-normalized; rows 4-5 are all-zero -> unobserved grey
    assert stub.features.shape == (6, 4)
    assert np.allclose(stub.features[:4].norm(dim=1).numpy(), 1.0, atol=1e-3)
    assert v._point_features is None
    assert colors.shape == (6, 3)
    assert (colors[4:] == 128).all()
    assert not (colors[:4] == 128).all(axis=1).any()

    # Cached for the rest of the load
    cached = v._mesh_vertex_features
    v.score_query(positive=["y"], negative=[], extractor_name="talk2dino")
    assert v._mesh_vertex_features is cached


def test_mesh_query_without_vertex_features_reports_and_returns_rgb(tmp_path):
    """A store without vertex_features leaves the mesh uncolored and tells the console how to fix it."""
    mesh_path = tmp_path / "mesh.ply"
    _write_mesh_6v(mesh_path)
    store = tmp_path / "talk2dino_lifted.zarr"
    write_point_features(store, np.ones((20, 4), dtype=np.float32), None)

    log = OperationLog()
    v = SplitViewer(log, off_screen=True)
    result = _FakeResult(20)
    v.load(result, mesh_path=mesh_path, lifted_store=store)
    v.mode = "mesh"
    v._extractor_cache = {"talk2dino": _CountingExtractor()}
    colors = v.score_query(positive=["x"], negative=[], extractor_name="talk2dino")

    assert np.array_equal(colors, result.colors)
    assert any("mesh query needs vertex_features" in line for line in log.log_lines)
    assert v.cached_query_colors("mesh") is None


########
# Lifted store
########


def test_ensure_lifted_reads_the_lifted_store(tmp_path):
    """First query decodes <extractor>_lifted.zarr through the autoencoder stored inside it."""
    store = tmp_path / "talk2dino_lifted.zarr"
    codes = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    write_point_features(store, codes, FeatureAutoencoder(32, 8))
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(20), mesh_path=None, lifted_store=store)
    v.ensure_lifted()
    assert v._point_features.shape == (20, 32)
    assert np.allclose(np.linalg.norm(v._point_features, axis=1), 1.0, atol=1e-5)


def test_ensure_lifted_raises_when_latent_store_lacks_autoencoder(tmp_path):
    """A latent-code store without autoencoder.pt is unreadable and the error surfaces."""
    store = tmp_path / "talk2dino_lifted.zarr"
    write_point_features(
        store, np.zeros((20, 8), dtype=np.float32), FeatureAutoencoder(32, 8)
    )
    (store / "autoencoder.pt").unlink()
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(20), mesh_path=None, lifted_store=store)

    with pytest.raises(FileNotFoundError):
        v.ensure_lifted()


########
# Rendering the right pane
########


def test_recolor_updates_scalars_without_rebuilding(monkeypatch):
    """A second query recolor updates point RGB in place; it does not clear + rebuild the pane."""
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)

    clears = {"n": 0}
    real_clear = v._right.clear

    def counting_clear(*a, **k):
        clears["n"] += 1
        return real_clear(*a, **k)

    monkeypatch.setattr(v._right, "clear", counting_clear)

    # First recolor after load may rebuild; the second must not clear again
    n = len(v._result.points)
    v.render_query(np.zeros((n, 3), dtype=np.uint8))
    baseline = clears["n"]
    colors2 = np.full((n, 3), 7, dtype=np.uint8)
    v.render_query(colors2)
    assert clears["n"] == baseline

    # The new colors landed on the displayed cloud
    idx = v._display_idx
    np.testing.assert_array_equal(
        np.asarray(v._right_cloud["RGB"])[:5], colors2[idx][:5]
    )


def test_recolor_fast_path_does_not_reapply_view(monkeypatch):
    """In-place recolor does not re-run camera/light setup, which would yank the user's view."""
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=None)

    calls = {"n": 0}
    real_apply = v._apply_view

    def counting_apply(plotter):
        calls["n"] += 1
        return real_apply(plotter)

    monkeypatch.setattr(v, "_apply_view", counting_apply)

    # Both recolors hit the cached-cloud fast path -> zero camera/light reapplies
    n = len(v._result.points)
    v.render_query(np.zeros((n, 3), dtype=np.uint8))
    v.render_query(np.full((n, 3), 7, dtype=np.uint8))
    assert calls["n"] == 0


def test_render_query_length_mismatch_falls_back():
    """Stale wrong-length colors (e.g. mesh-vertex colors) do not crash the fast-path recolor."""
    op_log = OperationLog()
    v = SplitViewer(op_log, off_screen=True)
    v.load(_FakeResult(20), mesh_path=None)
    v.render_query(np.zeros((7, 3), dtype=np.uint8))
    assert any("don't match" in line for line in op_log.log_lines)


@pytest.mark.parametrize(
    "n_colors", [3, 20], ids=["per_vertex", "size_mismatch_falls_back"]
)
def test_render_right_colors_the_mesh(tmp_path, n_colors):
    """Per-vertex colors paint the 3-vertex mesh; point-length colors fall back to plain RGB."""
    mesh_path = tmp_path / "mesh.ply"
    _write_tiny_mesh(mesh_path)

    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(p=20), mesh_path=mesh_path)
    v.ensure_mesh_polydata()
    v.mode = "mesh"
    v._render_right(np.full((n_colors, 3), 200, dtype=np.uint8))
    assert v._right.actors


def test_apply_viridis_draws_nan_grey():
    rgb = apply_viridis(np.array([0.0, np.nan, 1.0], dtype=np.float32))
    assert (rgb[1] == 128).all()
    assert not np.array_equal(rgb[0], rgb[2])


def test_apply_viridis_shape_and_dtype():
    sims = np.linspace(-1.0, 1.0, 12).astype(np.float32)
    rgb = apply_viridis(sims)
    assert rgb.shape == (12, 3)
    assert rgb.dtype == np.uint8


########
# Mesh read
########


def test_mesh_read_from_disk_once_across_renders(tmp_path, monkeypatch):
    """load() defers the read; ensure reads once; mode/normalize renders reuse and never mutate it."""
    mesh_path = tmp_path / "mesh.ply"
    _write_tiny_mesh(mesh_path)

    # Count pv.read calls from before load: the lazy ensure is the single allowed read
    reads = {"n": 0}
    real_read = viewer_module.pv.read

    def counting_read(path):
        reads["n"] += 1
        return real_read(path)

    monkeypatch.setattr(viewer_module.pv, "read", counting_read)

    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=mesh_path)
    assert reads["n"] == 0
    assert v._mesh_polydata is None

    v.ensure_mesh_polydata()
    cached_points = v._mesh_polydata.points.copy()
    v.set_mode("mesh")
    v.set_mode("pointcloud")
    v.set_mode("mesh")
    v.set_normalize_view(False)
    assert reads["n"] == 1
    np.testing.assert_array_equal(v._mesh_polydata.points, cached_points)


def test_ensure_mesh_polydata_uses_preloaded_without_reading(monkeypatch, tmp_path):
    """A shared-cache PolyData handed in as preloaded skips the disk read entirely."""
    reads = _counting_read(monkeypatch)
    mesh_path = tmp_path / "mesh.ply"
    mesh_path.touch()
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=mesh_path)
    cached = pv.PolyData()
    assert v.ensure_mesh_polydata(preloaded=cached) is True
    assert v.mesh_polydata() is cached
    assert reads == []


def test_ensure_mesh_polydata_missing_mesh_returns_false():
    v = SplitViewer(OperationLog(), off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    assert v.ensure_mesh_polydata() is False


########
# compute_view_transform
########


def _apply(T, pts):
    """Apply a 4x4 homogeneous transform to (P, 3) points."""
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (h @ T.T)[:, :3]


def _bbox_center(pts):
    """Geometric midpoint of a point set's bounding box."""
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


def test_view_transform_centers_on_origin():
    """The inlier bbox center (flyers clipped by radius) maps to the origin."""
    pts = np.random.default_rng(0).random((200, 3)).astype(np.float32) + np.array(
        [10.0, 5.0, -3.0]
    )
    T = compute_view_transform(pts, extrinsics=None, percentile=95.0)
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    inliers = pts[d <= np.percentile(d, 95.0)]
    assert np.allclose(_apply(T, _bbox_center(inliers)[None]), 0.0, atol=1e-4)


def test_view_transform_scales_to_target_radius():
    """A 100x-inflated cloud's 95th-percentile radius maps to target_radius."""
    pts = (np.random.default_rng(0).random((500, 3)).astype(np.float32) - 0.5) * 100.0
    T = compute_view_transform(pts, extrinsics=None, target_radius=0.7, percentile=95.0)
    out = _apply(T, pts)

    # Radii about the origin, where T maps the inlier center; the bbox of `out` includes flyers
    r = np.percentile(np.linalg.norm(out, axis=1), 95.0)
    assert abs(r - 0.7) < 1e-6


def test_view_transform_aligns_mean_camera_up_to_plus_z():
    """Identity-rotation w2c cameras have world up -Y; T maps -Y onto +Z."""
    extr = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    pts = np.random.default_rng(0).random((50, 3)).astype(np.float32)
    T = compute_view_transform(pts, extrinsics=extr)
    d = T[:3, :3] @ np.array([0.0, -1.0, 0.0])
    d /= np.linalg.norm(d)
    assert np.allclose(d, [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    "cancelling_cameras", [True, False], ids=["opposite_ups", "no_extrinsics"]
)
def test_view_transform_without_a_mean_up_skips_rotation(cancelling_cameras):
    """Opposite camera ups (mean ~0) or no cameras: the 3x3 block is a pure scaling, no mixing."""
    extr = None

    if cancelling_cameras:
        flipped = np.eye(4)
        flipped[1, :3] = [0.0, -1.0, 0.0]
        extr = np.stack([np.eye(4), flipped]).astype(np.float32)

    pts = (np.random.default_rng(0).random((50, 3)).astype(np.float32) - 0.5) * 4.0
    T = compute_view_transform(pts, extrinsics=extr)
    off_diag = T[:3, :3] - np.diag(np.diag(T[:3, :3]))
    assert np.allclose(off_diag, 0.0, atol=1e-6)
