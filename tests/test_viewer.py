"""Smoke tests for the generic viser scene Viewer (no browser needed)."""

import hashlib
import logging
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh
from PIL import Image

import collab_splats.viewer as viewer_module
from collab_splats.semantics.features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
)
from collab_splats.semantics.store import read_point_features, write_point_features
from collab_splats.viewer import Viewer, _build, _find_stores


@pytest.fixture(scope="module")
def viewer():
    # Ephemeral free port so parallel test runs don't collide
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    v = Viewer(port=port)
    yield v
    v.server.stop()


def test_add_points_registers_node(viewer):
    pts = np.zeros((10, 3), dtype=np.float32)
    cols = np.full((10, 3), 128, dtype=np.uint8)
    viewer.add_points("cloud_a", pts, cols)
    assert "cloud_a" in viewer.points


def test_add_points_upserts_by_name(viewer):
    pts = np.ones((5, 3), dtype=np.float32)
    cols = np.zeros((5, 3), dtype=np.uint8)
    viewer.add_points("cloud_a", pts, cols)
    # Same name replaces: registry holds the new arrays under one entry
    assert viewer.points["cloud_a"][1].shape == (5, 3)


def test_add_frustum(viewer):
    pose = np.eye(4, dtype=np.float32)  # world-to-cam identity
    intrinsic = np.array(
        [[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32
    )
    viewer.add_frustum("cams/frame_0", pose, intrinsic)
    assert "cams/frame_0" in viewer.frustums
    assert viewer.frustums["cams/frame_0"].visible


def test_add_frustum_with_image(viewer):
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array(
        [[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    viewer.add_frustum("cams/frame_1", pose, intrinsic, image=image)
    assert "cams/frame_1" in viewer.frustums


def test_add_lines(viewer):
    segments = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)  # (1, 2, 3)
    viewer.add_lines("loop_edges", segments)  # smoke: no exception


def test_serve_forever_returns_when_stop_already_set():
    """serve_forever exits promptly when its stop event is pre-set (no port bind needed)."""
    # __new__ avoids binding a viser port; serve_forever only touches _stop.
    v = Viewer.__new__(Viewer)
    v._stop = threading.Event()
    v._stop.set()
    v.serve_forever(poll=0.01)  # returns immediately since stop is already set


def _quad():
    """Unit square in z=0 as two triangles; vertex 3 is the (1, 1) corner."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    colors = np.full((4, 3), 200, np.uint8)
    return vertices, faces, colors


def test_add_mesh_registers_node(viewer):
    viewer.add_mesh("quad", *_quad())
    assert "quad" in viewer.meshes


def test_add_mesh_shows_the_texture_and_picks_on_the_given_vertices(
    viewer, monkeypatch
):
    vertices, faces, colors = _quad()
    corners = vertices[faces].reshape(-1, 3)
    uv = corners[:, :2]
    image = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    textured = trimesh.Trimesh(
        corners, np.arange(6).reshape(2, 3), visual=visual, process=False
    )
    shown = []
    monkeypatch.setattr(
        viewer.server.scene, "add_mesh_trimesh", lambda name, mesh: shown.append(mesh)
    )

    viewer.add_mesh("textured_quad", vertices, faces, colors, textured=textured)

    assert shown == [textured]
    assert (
        viewer._pick_vertex("textured_quad", (0.9, 0.9, 1.0), (0.0, 0.0, -1.0))[1] == 3
    )


def test_pick_vertex_returns_the_hit_triangles_nearest_vertex(viewer):
    viewer.add_mesh("quad", *_quad())
    t, vertex = viewer._pick_vertex("quad", (0.9, 0.9, 1.0), (0.0, 0.0, -1.0))
    assert t == pytest.approx(1.0)
    assert vertex == 3
    assert viewer._pick_vertex("quad", (5.0, 5.0, 1.0), (0.0, 0.0, -1.0)) is None


def test_on_click_hands_the_picked_vertex_to_the_callback(viewer):
    viewer.add_mesh("quad", *_quad())
    picked = []
    viewer.on_click("quad", picked.append)
    event = SimpleNamespace(ray_origin=(0.1, 0.1, 1.0), ray_direction=(0.0, 0.0, -1.0))
    viewer._dispatch_click(event)
    assert picked == [0]


def _capture_heat(viewer, monkeypatch):
    """Record (name, mesh) for every add_mesh_trimesh; handles get a no-op remove."""
    sent = []

    def add(name, mesh):
        sent.append((name, mesh))
        return SimpleNamespace(remove=lambda: None)

    monkeypatch.setattr(viewer.server.scene, "add_mesh_trimesh", add)
    return sent


def test_show_heat_keeps_faces_reaching_the_floor_on_their_used_vertices(
    viewer, monkeypatch
):
    vertices, faces, colors = _quad()
    viewer.add_mesh("quad", vertices, faces, colors)
    sent = _capture_heat(viewer, monkeypatch)

    # Only vertex 3 reaches the floor: face [1, 3, 2] stays, face [0, 1, 2] goes
    viewer.show_heat("quad", np.array([0.0, 0.1, 0.2, 0.9]), floor=0.5, offset=0.0)

    assert [name for name, _ in sent] == ["quad/heat"]
    heat = sent[0][1]
    assert heat.vertices.tolist() == vertices[[1, 2, 3]].tolist()
    assert heat.faces.tolist() == [[0, 2, 1]]


def test_show_heat_lifts_along_whole_mesh_normals_cached_until_the_mesh_changes(
    viewer, monkeypatch
):
    sent = _capture_heat(viewer, monkeypatch)
    vertices, faces, colors = _quad()
    scores = np.array([0.0, 0.0, 0.0, 1.0])

    # Median edge 1, normal +z: lifted by the offset
    viewer.add_mesh("quad", vertices, faces, colors)
    viewer.show_heat("quad", scores, floor=0.5, offset=0.25)
    np.testing.assert_allclose(sent[-1][1].vertices[:, 2], 0.25)

    # Re-adding at twice the size drops the cached lift: median edge 2
    viewer.add_mesh("quad", vertices * 2, faces, colors)
    viewer.show_heat("quad", scores, floor=0.5, offset=0.25)
    np.testing.assert_allclose(sent[-1][1].vertices[:, 2], 0.5)


def test_show_heat_never_draws_a_face_touching_a_nan_vertex(viewer, monkeypatch):
    viewer.add_mesh("quad", *_quad())
    sent = _capture_heat(viewer, monkeypatch)

    # Vertex 0 has no data; face [0, 1, 2] has hot vertices but touches it
    viewer.show_heat("quad", np.array([np.nan, 1.0, 0.9, 0.8]), floor=0.5, offset=0.0)

    heat = sent[0][1]
    assert len(heat.faces) == 1
    assert len(heat.vertices) == 3


def test_show_heat_colors_follow_the_score(viewer, monkeypatch):
    viewer.add_mesh("quad", *_quad())
    sent = _capture_heat(viewer, monkeypatch)

    viewer.show_heat("quad", np.array([0.6, 0.7, 0.8, 0.9]), floor=0.5)

    rgba = sent[0][1].visual.vertex_colors
    # Viridis runs dark purple to yellow: brightness rises with the score
    assert np.all(np.diff(rgba[:, :3].astype(int).sum(axis=1)) > 0)


def test_show_heat_offset_lifts_along_normals(viewer, monkeypatch):
    vertices, faces, colors = _quad()
    viewer.add_mesh("quad", vertices, faces, colors)
    sent = _capture_heat(viewer, monkeypatch)

    viewer.show_heat("quad", np.ones(4), floor=0.5, offset=0.5)

    # Flat quad, normals along +z, median edge 1: every vertex moves 0.5 in z
    assert np.allclose(np.abs(sent[0][1].vertices[:, 2]), 0.5)
    assert np.allclose(sent[0][1].vertices[:, :2], vertices[:, :2])


def test_show_heat_replaces_clears_and_sends_nothing_below_the_floor(
    viewer, monkeypatch
):
    viewer.add_mesh("quad", *_quad())
    viewer.heats.pop("quad", None)
    removed = []

    def add(name, mesh):
        return SimpleNamespace(remove=lambda: removed.append(name))

    monkeypatch.setattr(viewer.server.scene, "add_mesh_trimesh", add)

    # A second call drops the first overlay
    viewer.show_heat("quad", np.ones(4), floor=0.5)
    viewer.show_heat("quad", np.ones(4), floor=0.5)
    assert removed == ["quad/heat"]

    # All below the floor: previous overlay gone, nothing new kept
    viewer.show_heat("quad", np.zeros(4), floor=0.5)
    assert removed == ["quad/heat"] * 2
    assert "quad" not in viewer.heats

    # None clears
    viewer.show_heat("quad", np.ones(4), floor=0.5)
    viewer.show_heat("quad", None, floor=0.5)
    assert removed == ["quad/heat"] * 3
    assert "quad" not in viewer.heats


def test_add_label_list_ranks_by_weight_and_hands_words_to_on_select(viewer):
    viewer.add_mesh("quad", *_quad())
    selected = []
    words = ["rock", "tree", "sky"]
    weights = np.array([12.0, 28_718.4, 950.0])

    viewer.add_label_list("quad", words, weights, selected.append, top_n=5)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == [
        "Clear",
        "tree (~28.7k)",
        "sky (~950)",
        "rock (~12)",
    ]

    # Word buttons hand their word over; Clear hands None
    buttons[1]._impl.update_cb[0](None)
    buttons[0]._impl.update_cb[0](None)
    assert selected == ["tree", None]

    # Re-adding replaces the list rather than stacking a second one
    viewer.add_label_list("quad", words, weights, selected.append, top_n=1)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == ["Clear", "tree (~28.7k)"]


def test_on_click_registers_one_scene_handler_and_the_nearest_mesh_wins(
    viewer, monkeypatch
):
    registered = []
    monkeypatch.setattr(viewer.server.scene, "on_click", lambda: registered.append)
    monkeypatch.setattr(viewer, "mesh_clicks", {})
    monkeypatch.setattr(viewer, "_click_registered", False)
    vertices, faces, colors = _quad()
    below = vertices - np.array([0, 0, 1], np.float32)
    picked = []

    # Two stacked quads; a downward ray hits "top" first
    viewer.add_mesh("top", vertices, faces, colors)
    viewer.add_mesh("below", below, faces, colors)
    viewer.on_click("below", lambda vertex: picked.append(("below", vertex)))
    viewer.on_click("top", lambda vertex: picked.append(("top", vertex)))
    assert registered == [viewer._dispatch_click]

    assert viewer.pick((0.9, 0.9, 1.0), (0.0, 0.0, -1.0)) == ("top", 3)
    event = SimpleNamespace(ray_origin=(0.9, 0.9, 1.0), ray_direction=(0.0, 0.0, -1.0))
    viewer._dispatch_click(event)
    assert picked == [("top", 3)]

    # A ray that misses both meshes calls nothing
    assert viewer.pick((5.0, 5.0, 1.0), (0.0, 0.0, -1.0)) is None
    miss = SimpleNamespace(ray_origin=(5.0, 5.0, 1.0), ray_direction=(0.0, 0.0, -1.0))
    viewer._dispatch_click(miss)
    assert picked == [("top", 3)]


def test_clicks_still_pick_after_show_heat(viewer, monkeypatch):
    monkeypatch.setattr(viewer, "mesh_clicks", {})
    vertices, faces, colors = _quad()
    picked = []

    viewer.add_mesh("quad", vertices, faces, colors)
    viewer.on_click("quad", picked.append)
    _capture_heat(viewer, monkeypatch)
    viewer.show_heat("quad", np.ones(4), floor=0.5)

    event = SimpleNamespace(ray_origin=(0.9, 0.9, 1.0), ray_direction=(0.0, 0.0, -1.0))
    viewer._dispatch_click(event)
    assert picked == [3]


def test_reset_view_frames_a_mesh_only_scene(viewer, monkeypatch):
    camera = SimpleNamespace(look_at=None, position=None)
    monkeypatch.setattr(viewer, "points", {})
    monkeypatch.setattr(viewer, "meshes", {})
    monkeypatch.setattr(
        viewer.server, "get_clients", lambda: {0: SimpleNamespace(camera=camera)}
    )

    viewer.add_mesh("quad", *_quad())
    viewer._reset_view()
    assert camera.look_at.tolist() == [0.5, 0.5, 0.0]


class _StubText(BaseQueryableExtractor):
    """Two-axis text space: "x" is axis 0, anything else axis 1; counts constructions, keeps the last."""

    built = 0
    last = None

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__("max_size", 16)
        self.scale = scale
        self._device = torch.device("cpu")
        type(self).built += 1
        type(self).last = self

    def encode_text(self, texts):
        rows = [[1.0, 0.0] if text == "x" else [0.0, 1.0] for text in texts]
        return torch.tensor(rows)


@pytest.fixture
def stub_text():
    """_StubText registered as "stub_text" for one test."""
    BaseQueryableExtractor.register("stub_text")(_StubText)
    _StubText.built = 0
    yield _StubText
    BaseFeatureExtractor._registry.pop("stub_text")


def _backend(tmp_path):
    """Backend dir holding the quad as mesh.ply."""
    vertices, faces, _ = _quad()
    trimesh.Trimesh(vertices, faces, process=False).export(tmp_path / "mesh.ply")
    return tmp_path


def _sha(backend):
    return hashlib.sha256((backend / "mesh.ply").read_bytes()).hexdigest()


def _vertex_store(backend, extractor, codes, kwargs=None, mesh_sha256=None):
    """Lifted store with full-width vertex codes, lifted onto the backend's mesh.ply unless told otherwise."""
    path = backend / "semantics" / f"{extractor}_lifted.zarr"
    attrs = {
        "extractor": extractor,
        "extractor_kwargs": kwargs or {},
        "mesh_sha256": mesh_sha256 or _sha(backend),
    }
    vertex_arrays = {"vertex_features": np.asarray(codes, np.float16)}
    write_point_features(
        path,
        np.ones((1, 2), np.float32),
        None,
        vertex_arrays=vertex_arrays,
        attrs=attrs,
    )
    return path


def _word_store(backend, word_ids, word_probs, words):
    """ocr_lens lifted store with per-vertex word ids + probs."""
    path = backend / "semantics" / "ocr_lens_lifted.zarr"
    attrs = {
        "extractor": "ocr_lens",
        "extractor_kwargs": {},
        "mesh_sha256": _sha(backend),
        "words": words,
    }
    vertex_arrays = {
        "vertex_word_ids": np.asarray(word_ids, np.int16),
        "vertex_word_probs": np.asarray(word_probs, np.float16),
    }
    write_point_features(
        path,
        np.ones((1, 2), np.float32),
        None,
        vertex_arrays=vertex_arrays,
        attrs=attrs,
    )
    return path


def test_find_stores_without_semantics_is_empty(tmp_path):
    assert _find_stores(_backend(tmp_path)) == {}


def test_find_stores_lists_vertex_stores_only(tmp_path):
    backend = _backend(tmp_path)
    text = _vertex_store(backend, "stub_text", np.eye(4, 2))
    words = _word_store(backend, [[0], [0], [0], [0]], [[1.0]] * 4, ["a"])
    write_point_features(
        backend / "semantics" / "dinov2_lifted.zarr", np.ones((1, 2), np.float32), None
    )

    assert _find_stores(backend) == {"ocr_lens": words, "stub_text": text}


def test_find_stores_skips_a_store_lifted_onto_another_mesh(tmp_path, caplog):
    backend = _backend(tmp_path)
    _vertex_store(backend, "stub_text", np.eye(4, 2), mesh_sha256="old")

    with caplog.at_level(logging.WARNING):
        assert _find_stores(backend) == {}

    assert "re-run semantics" in caplog.text


@pytest.fixture
def fresh_viewer():
    """Viewer of its own: _build adds GUI that must not leak across tests."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    v = Viewer(port=port)
    yield v
    v.server.stop()


def _inputs(viewer):
    """Live GUI inputs by label; removed handles drop out."""
    return {h.label: h for h in viewer.server.gui._gui_input_handle_from_uuid.values()}


def _record_heat(viewer, monkeypatch):
    """Record every show_heat call as (name, scores, floor)."""
    calls = []
    monkeypatch.setattr(
        viewer,
        "show_heat",
        lambda name, scores, floor: calls.append((name, scores, floor)),
    )
    return calls


def test_build_greys_a_mesh_without_vertex_colors(tmp_path, fresh_viewer, monkeypatch):
    added = []
    monkeypatch.setattr(
        fresh_viewer,
        "add_mesh",
        lambda name, v, f, c, textured=None: added.append((v, f, c, textured)),
    )
    _build(fresh_viewer, _backend(tmp_path), textured=False, texture_size=64)

    vertices, faces, colors, textured = added[0]
    assert vertices.shape == (4, 3) and faces.shape == (2, 3)
    assert (colors == 200).all() and textured is None


def test_build_without_stores_shows_the_mesh_and_no_dropdown(tmp_path, fresh_viewer):
    dropdown = _build(fresh_viewer, _backend(tmp_path), textured=False, texture_size=64)

    assert dropdown is None
    assert "mesh" in fresh_viewer.meshes
    assert "Semantics" not in _inputs(fresh_viewer)


def test_text_mode_heat_is_score_queries_with_unobserved_nan(
    tmp_path, fresh_viewer, stub_text, monkeypatch
):
    # Vertex 2 has all-zero codes: unobserved (NaN), though its contrast score would be 0.5
    codes = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], np.float32)
    path = _vertex_store(_backend(tmp_path), "stub_text", codes, {"scale": 2.0})
    expected = (
        stub_text()
        .score_queries(
            torch.from_numpy(read_point_features(path, name="vertex_features")),
            ["x"],
            ["object"],
        )
        .numpy()
    )
    expected[2] = np.nan
    stub_text.built = 0

    calls = _record_heat(fresh_viewer, monkeypatch)
    _build(fresh_viewer, tmp_path, textured=False, texture_size=64)
    inputs = _inputs(fresh_viewer)
    inputs["Query"].value = "x"

    # Two searches, one extractor build
    inputs["Search"]._impl.update_cb[0](None)
    inputs["Search"]._impl.update_cb[0](None)

    name, scores, floor = calls[-1]
    assert name == "mesh" and floor == inputs["Query min score"].value
    np.testing.assert_allclose(scores, expected, atol=1e-6, equal_nan=True)
    assert stub_text.built == 1
    assert stub_text.last.scale == 2.0


def test_text_mode_empty_query_clears_the_heat(
    tmp_path, fresh_viewer, stub_text, monkeypatch
):
    _vertex_store(_backend(tmp_path), "stub_text", np.eye(4, 2))
    calls = _record_heat(fresh_viewer, monkeypatch)
    _build(fresh_viewer, tmp_path, textured=False, texture_size=64)

    _inputs(fresh_viewer)["Search"]._impl.update_cb[0](None)

    assert calls[-1][1] is None
    assert stub_text.built == 0


def test_switching_to_none_clears_heat_and_removes_the_mode_gui(
    tmp_path, fresh_viewer, stub_text, monkeypatch
):
    _vertex_store(_backend(tmp_path), "stub_text", np.eye(4, 2))
    calls = _record_heat(fresh_viewer, monkeypatch)
    dropdown = _build(fresh_viewer, tmp_path, textured=False, texture_size=64)
    assert set(dropdown.options) == {"none", "stub_text"}

    assert dropdown.value == "stub_text"
    assert {"Query", "Negatives", "Query min score", "Search"} <= set(
        _inputs(fresh_viewer)
    )

    dropdown.value = "none"
    assert calls[-1] == ("mesh", None, 0.0)
    assert not {"Query", "Negatives", "Query min score", "Search"} & set(
        _inputs(fresh_viewer)
    )
    assert "Semantics" in _inputs(fresh_viewer)


WORDS = ["apple", "tree", "sky"]


def _word_backend(tmp_path):
    """Quad backend with an ocr_lens store; vertex 3 unobserved."""
    backend = _backend(tmp_path)
    ids = [[0, 1], [1, 2], [2, 0], [0, 1]]
    probs = [[0.75, 0.25], [0.5, 0.5], [1.0, 0.0], [0.0, 0.0]]
    _word_store(backend, ids, probs, WORDS)
    return backend


def test_build_selects_ocr_lens_by_default(tmp_path, fresh_viewer):
    backend = _word_backend(tmp_path)
    _vertex_store(backend, "stub_text", np.eye(4, 2))
    dropdown = _build(fresh_viewer, backend, textured=False, texture_size=64)

    assert dropdown.value == "ocr_lens"
    assert set(dropdown.options) == {"none", "ocr_lens", "stub_text"}
    assert "Query min p" in _inputs(fresh_viewer)


def test_word_mode_heat_is_the_summed_probability_of_the_query_words(
    tmp_path, fresh_viewer, monkeypatch
):
    calls = _record_heat(fresh_viewer, monkeypatch)

    # Markdown handles are not tracked by viser; keep the unknown-word note
    notes = []
    add_markdown = fresh_viewer.server.gui.add_markdown
    monkeypatch.setattr(
        fresh_viewer.server.gui,
        "add_markdown",
        lambda *a, **k: notes.append(add_markdown(*a, **k)) or notes[-1],
    )
    _build(fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64)
    inputs = _inputs(fresh_viewer)
    inputs["Query"].value = "apple, Sky, nope"
    inputs["Search"]._impl.update_cb[0](None)

    name, scores, floor = calls[-1]
    assert name == "mesh" and floor == inputs["Query min p"].value
    np.testing.assert_allclose(scores, [0.75, 0.5, 1.0, np.nan], equal_nan=True)
    assert notes[-1].content == "not in vocabulary: nope"


def test_word_mode_label_list_ranks_words_by_probability_mass(tmp_path, fresh_viewer):
    _build(fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64)

    _, buttons = fresh_viewer.label_lists["mesh"]
    assert [b.label for b in buttons] == [
        "Clear",
        "sky (~2)",
        "apple (~1)",
        "tree (~1)",
    ]


def test_word_mode_probe_charts_scene_terms_and_flags_unobserved(
    tmp_path, fresh_viewer, monkeypatch
):
    # 11 stored words per vertex; w10 sits only past an observed top-10, w12 only on unobserved vertex 3
    backend = _backend(tmp_path)
    words = [f"w{i}" for i in range(13)]
    ids = [[*range(10), 11], [*range(11)], [*range(10), 11], [12, *range(10)]]
    probs = [
        [0.5, 0.25] + [0.0] * 9,
        [0.3] + [0.05] * 10,
        [1.0] + [0.0] * 10,
        [0.0] * 11,
    ]
    _word_store(backend, ids, probs, words)

    charts = []
    monkeypatch.setattr(
        viewer_module,
        "_chart",
        lambda title, words, probs: charts.append((title, list(words), probs)) or "",
    )
    _build(fresh_viewer, backend, textured=False, texture_size=64)

    for vertex in (0, 1, 3):
        fresh_viewer.mesh_clicks["mesh"](vertex)

    # Vertex 0: renormalized over its scene terms
    assert charts[0][:2] == ("vertex 0", ["w0", "w1"])
    np.testing.assert_allclose(charts[0][2], [2 / 3, 1 / 3], rtol=1e-3)

    # Vertex 1: w10 is no scene term, so the top 10 renormalize over 0.75
    assert charts[1][:2] == ("vertex 1", [f"w{i}" for i in range(10)])
    np.testing.assert_allclose(
        charts[1][2], np.array([0.3] + [0.05] * 9) / 0.75, rtol=1e-3
    )
    assert charts[2][0] == "vertex 3: unobserved"


def test_switching_away_from_word_mode_removes_labels_clicks_and_probe(
    tmp_path, fresh_viewer
):
    dropdown = _build(
        fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64
    )
    fresh_viewer.mesh_clicks["mesh"](0)
    dropdown.value = "none"

    assert (
        "mesh" not in fresh_viewer.label_lists
        and "mesh" not in fresh_viewer.mesh_clicks
    )
    assert "/probe" not in fresh_viewer.server.scene._handle_from_node_name
    assert not {"Query", "Query min p", "Search"} & set(_inputs(fresh_viewer))


def test_reentering_word_mode_keeps_one_click_dispatcher(tmp_path, fresh_viewer):
    dropdown = _build(
        fresh_viewer, _word_backend(tmp_path), textured=False, texture_size=64
    )
    dropdown.value = "none"
    dropdown.value = "ocr_lens"
    dropdown.value = "none"
    dropdown.value = "ocr_lens"

    assert len(fresh_viewer.server.scene._scene_pointer_cb) == 1


def test_viewer_module_never_imports_the_reconstructor():
    # reconstructor imports Viewer; the reverse would be a cycle
    code = "import sys, collab_splats.viewer; print('collab_splats.reconstructor' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    # Last line only: importing geometry lets warp print its init banner to stdout
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_main_runs_as_a_module():
    out = subprocess.run(
        [sys.executable, "-m", "collab_splats.viewer", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "backend_dir" in out.stdout and "--texture_size" in out.stdout
