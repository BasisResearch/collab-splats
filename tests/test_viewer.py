"""Smoke tests for the generic viser scene Viewer (no browser needed)."""

import hashlib
import io
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
from collab_splats.viewer import (
    Viewer,
    _browse,
    _build,
    _find_backends,
    _find_stores,
    _progress,
    _pull_stats,
    _remote,
    _status,
    _texture_glb,
)


@pytest.fixture(scope="module")
def viewer():
    # Ephemeral free port so parallel test runs don't collide
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    v = Viewer(port=port)
    yield v
    v.server.stop()


@pytest.fixture(autouse=True)
def _clear_glbs():
    """Empty the session GLB cache after each test."""
    yield
    viewer_module._GLBS.clear()


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


def _textured_backend(tmp_path, atlas=128):
    """Backend dir holding the quad as mesh.ply plus texture/mesh.obj with an atlas-px atlas."""
    backend = _backend(tmp_path)
    vertices, faces, _ = _quad()
    (backend / "texture").mkdir()
    visual = trimesh.visual.TextureVisuals(
        uv=vertices[:, :2], image=Image.new("RGB", (atlas, atlas), (255, 0, 0))
    )
    trimesh.Trimesh(vertices, faces, visual=visual, process=False).export(
        backend / "texture" / "mesh.obj"
    )
    return backend


def _join_texture():
    """Wait for every running texture thread."""
    for thread in threading.enumerate():
        if thread.name == "viewer-texture":
            thread.join(timeout=30)
            assert not thread.is_alive()


def _record_glbs(viewer, monkeypatch):
    """Record every add_glb send as (name, bytes); add_mesh_trimesh sends through it too."""
    sent = []
    add_glb = viewer.server.scene.add_glb

    def record(name, glb_data, **kwargs):
        sent.append((name, glb_data))
        return add_glb(name, glb_data, **kwargs)

    monkeypatch.setattr(viewer.server.scene, "add_glb", record)
    return sent


def _jpeg(data):
    """True when a GLB embeds a JPEG texture."""
    return b"\xff\xd8\xff" in data


def _gated_texture(monkeypatch):
    """Hold _texture_glb until the returned event is set; count its calls."""
    gate, calls = threading.Event(), []
    real = viewer_module._texture_glb

    def held(backend_dir, texture_size):
        calls.append(backend_dir)
        gate.wait(30)
        return real(backend_dir, texture_size)

    monkeypatch.setattr(viewer_module, "_texture_glb", held)
    return gate, calls


def _record_status(monkeypatch):
    """Record every _status text, still driving the real bar and note."""
    texts = []
    real = viewer_module._status

    def record(progress, text, value=None, *, busy=True):
        texts.append(text)
        real(progress, text, value, busy=busy)

    monkeypatch.setattr(viewer_module, "_status", record)
    return texts


def _capture_progress(monkeypatch):
    """Keep every (bar, note) pair _progress makes."""
    made = []
    real = viewer_module._progress

    def capture(viewer):
        made.append(real(viewer))
        return made[-1]

    monkeypatch.setattr(viewer_module, "_progress", capture)
    return made


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
    """Live GUI inputs by label; removed handles and unlabeled progress bars drop out."""
    handles = viewer.server.gui._gui_input_handle_from_uuid.values()
    return {h.label: h for h in handles if hasattr(h, "label")}


def _record_heat(viewer, monkeypatch):
    """Record every show_heat call as (name, scores, floor)."""
    calls = []
    monkeypatch.setattr(
        viewer,
        "show_heat",
        lambda name, scores, floor: calls.append((name, scores, floor)),
    )
    return calls


def test_status_shows_a_striped_step_a_solid_percent_and_hides_on_none(fresh_viewer):
    progress = _progress(fresh_viewer)
    bar, note = progress
    assert not bar.visible and not note.visible

    # A step with no percent: striped bar at 100
    _status(progress, "loading mesh")
    assert bar.visible and note.visible and bar.animated and bar.value == 100.0
    assert note.content == "loading mesh"

    # A percent: solid bar at that value
    _status(progress, "pulling mesh: 25%", 25.0)
    assert bar.value == 25.0 and not bar.animated

    # A failure keeps the status, hides the bar; None hides both
    _status(progress, "pull failed: exit 7", busy=False)
    assert not bar.visible and note.visible
    _status(progress, None)
    assert not bar.visible and not note.visible


def test_pull_stats_moves_the_bar_to_the_rclone_percent_and_skips_other_lines(
    fresh_viewer,
):
    progress = _progress(fresh_viewer)
    show = _pull_stats(progress, "pulling mesh")

    show(
        "2026/10/22 05:01:30 NOTICE:    34.035 MiB / 89.907 MiB, 38%, 1.023 MiB/s, ETA 54s"
    )
    assert progress[0].value == 38.0
    assert (
        progress[1].content
        == "pulling mesh: 34.035 MiB / 89.907 MiB, 38%, 1.023 MiB/s, ETA 54s"
    )

    # rclone's first line has no percent; error lines neither
    show("2026/10/22 05:01:28 NOTICE:    0 B / 0 B, -, 0 B/s, ETA -")
    show("2026/10/22 05:01:31 ERROR : a/mesh.ply: failed to copy")
    assert progress[0].value == 38.0


def test_build_greys_a_mesh_without_vertex_colors(tmp_path, fresh_viewer, monkeypatch):
    added = []
    monkeypatch.setattr(
        fresh_viewer,
        "add_mesh",
        lambda name, v, f, c: added.append((v, f, c)),
    )
    _build(
        fresh_viewer,
        _backend(tmp_path),
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )

    vertices, faces, colors = added[0]
    assert vertices.shape == (4, 3) and faces.shape == (2, 3)
    assert (colors == 200).all()


def test_build_without_stores_shows_the_mesh_and_no_dropdown(tmp_path, fresh_viewer):
    dropdown = _build(
        fresh_viewer,
        _backend(tmp_path),
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )

    assert dropdown is None
    assert "mesh" in fresh_viewer.meshes
    assert "Semantics" not in _inputs(fresh_viewer)


@pytest.mark.parametrize("atlas, size", [(128, 64), (100, 64), (32, 32)])
def test_texture_glb_embeds_a_jpeg_atlas_at_texture_size(tmp_path, atlas, size):
    # 128 = 2 x 64 takes Image.reduce; 100 takes LANCZOS; 32 is never upscaled
    glb = _texture_glb(_textured_backend(tmp_path, atlas), 64)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    material = next(iter(scene.geometry.values())).visual.material

    assert _jpeg(glb) and b"\x89PNG" not in glb
    assert material.baseColorTexture.size == (size, size)


def test_build_shows_vertex_colors_first_then_the_textured_glb(
    tmp_path, fresh_viewer, monkeypatch
):
    sent = _record_glbs(fresh_viewer, monkeypatch)
    texts = _record_status(monkeypatch)
    gate, _ = _gated_texture(monkeypatch)
    progress = _progress(fresh_viewer)
    _build(fresh_viewer, _textured_backend(tmp_path), True, 64, progress, 0)

    # Vertex colors are up while the texture is held
    assert [name for name, _ in sent] == ["mesh"] and not _jpeg(sent[0][1])
    assert "mesh" in fresh_viewer.meshes

    # Released: the JPEG-textured GLB replaces the node; bar and status hidden
    gate.set()
    _join_texture()
    assert sent[-1][0] == "mesh" and _jpeg(sent[-1][1])
    assert texts[0] == "loading mesh" and texts[-1] is None
    assert any(t and t.startswith("texture: uploading") for t in texts)
    assert not progress[0].visible and not progress[1].visible


def test_build_keeps_the_heat_child_when_the_texture_lands(
    tmp_path, fresh_viewer, monkeypatch
):
    gate, _ = _gated_texture(monkeypatch)
    _build(
        fresh_viewer, _textured_backend(tmp_path), True, 64, _progress(fresh_viewer), 0
    )
    fresh_viewer.show_heat("mesh", np.ones(4), 0.0)

    gate.set()
    _join_texture()
    assert "/mesh/heat" in fresh_viewer.server.scene._handle_from_node_name


def test_build_drops_a_texture_that_lands_after_the_scene_changed(
    tmp_path, fresh_viewer, monkeypatch
):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    sent = _record_glbs(fresh_viewer, monkeypatch)
    gate, _ = _gated_texture(monkeypatch)
    progress = _progress(fresh_viewer)

    # a's texture is held while a new pick builds b
    _build(fresh_viewer, _textured_backend(a), True, 64, progress, 0)
    fresh_viewer.picks += 1
    _build(fresh_viewer, _backend(b), False, 64, progress, 1)
    gate.set()
    _join_texture()

    assert not any(_jpeg(data) for _, data in sent)
    assert not progress[0].visible and not progress[1].visible


def test_browse_stale_texture_leaves_the_next_picks_status_alone(
    tmp_path, fresh_viewer, monkeypatch
):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    sent = _record_glbs(fresh_viewer, monkeypatch)
    gate_a, _ = _gated_texture(monkeypatch)
    made = _capture_progress(monkeypatch)

    # Hold b's mesh load, signalling once it is reached
    entered, gate_b = threading.Event(), threading.Event()
    load_mesh = viewer_module._load_mesh

    def held(backend_dir):
        if backend_dir == b:
            entered.set()
            gate_b.wait(30)

        return load_mesh(backend_dir)

    monkeypatch.setattr(viewer_module, "_load_mesh", held)

    # a's texture is held; picking b stalls in its mesh load
    backends = {"a": _textured_backend(a), "b": _backend(b)}
    scene = _browse(fresh_viewer, backends, textured=True, texture_size=64)
    bar, note = made[0]
    pick_b = threading.Thread(target=setattr, args=(scene, "value", "b"))
    pick_b.start()
    assert entered.wait(30)

    # a's texture finishes during b's load: dropped, b's status untouched
    gate_a.set()
    _join_texture()
    assert note.content == "loading mesh" and bar.visible and note.visible
    assert not any(_jpeg(data) for _, data in sent)

    gate_b.set()
    pick_b.join(timeout=30)
    assert not pick_b.is_alive() and note.content == "no texture"


def test_browse_disables_the_scene_dropdown_while_a_pick_loads(
    tmp_path, fresh_viewer, monkeypatch
):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    # Hold b's mesh load, signalling once it is reached
    entered, gate = threading.Event(), threading.Event()
    load_mesh = viewer_module._load_mesh

    def held(backend_dir):
        if backend_dir == b:
            entered.set()
            gate.wait(30)

        return load_mesh(backend_dir)

    monkeypatch.setattr(viewer_module, "_load_mesh", held)
    backends = {"a": _backend(a), "b": _backend(b)}
    scene = _browse(fresh_viewer, backends, textured=False, texture_size=64)
    assert scene.disabled is False

    # Disabled while b loads; enabled once it is shown
    pick_b = threading.Thread(target=setattr, args=(scene, "value", "b"))
    pick_b.start()
    assert entered.wait(30)
    assert scene.disabled is True

    gate.set()
    pick_b.join(timeout=30)
    assert not pick_b.is_alive() and scene.disabled is False


def test_browse_failed_load_reenables_the_scene_dropdown(tmp_path, fresh_viewer):
    (tmp_path / "mesh.ply").write_bytes(b"garbage")
    scene = _browse(fresh_viewer, {"a": tmp_path}, textured=False, texture_size=64)

    assert scene.disabled is False


def test_browse_failed_load_reports_it_and_hides_the_bar(
    tmp_path, fresh_viewer, monkeypatch, caplog
):
    made = _capture_progress(monkeypatch)
    (tmp_path / "mesh.ply").write_bytes(b"garbage")
    _browse(fresh_viewer, {"a": tmp_path}, textured=False, texture_size=64)

    bar, note = made[0]
    assert note.content.startswith("load failed:") and note.visible
    assert not bar.visible and "cannot load" in caplog.text


def test_build_reopens_a_backend_from_the_glb_cache(
    tmp_path, fresh_viewer, monkeypatch
):
    sent = _record_glbs(fresh_viewer, monkeypatch)
    gate, calls = _gated_texture(monkeypatch)
    gate.set()
    backend = _textured_backend(tmp_path)

    # Two opens of one backend: one OBJ load, two textured sends
    for _ in range(2):
        _build(fresh_viewer, backend, True, 64, _progress(fresh_viewer), 0)
        _join_texture()

    assert calls == [backend]
    assert sum(_jpeg(data) for _, data in sent) == 2


def test_build_without_a_texture_says_so_and_keeps_vertex_colors(
    tmp_path, fresh_viewer
):
    progress = _progress(fresh_viewer)
    _build(fresh_viewer, _backend(tmp_path), True, 64, progress, 0)
    _join_texture()

    assert progress[1].visible and progress[1].content == "no texture"
    assert not progress[0].visible and "mesh" in fresh_viewer.meshes


def test_build_reports_a_texture_that_fails_to_load(
    tmp_path, fresh_viewer, monkeypatch
):
    def _broken(backend_dir, texture_size):
        raise ValueError("bad atlas")

    monkeypatch.setattr(viewer_module, "_texture_glb", _broken)
    progress = _progress(fresh_viewer)
    _build(fresh_viewer, _textured_backend(tmp_path), True, 64, progress, 0)
    _join_texture()

    assert progress[1].content == "texture failed: bad atlas"
    assert not progress[0].visible


def test_build_reports_a_textured_send_that_fails(tmp_path, fresh_viewer, monkeypatch):
    add_glb = fresh_viewer.server.scene.add_glb

    # Vertex-color sends pass; the JPEG-textured send raises
    def send(name, glb_data, **kwargs):
        if _jpeg(glb_data):
            raise ConnectionError("client gone")

        return add_glb(name, glb_data, **kwargs)

    monkeypatch.setattr(fresh_viewer.server.scene, "add_glb", send)
    progress = _progress(fresh_viewer)
    _build(fresh_viewer, _textured_backend(tmp_path), True, 64, progress, 0)
    _join_texture()

    assert progress[1].content == "texture failed: client gone"
    assert progress[1].visible and not progress[0].visible


def test_browse_texture_keeps_the_token_of_the_pick_that_started_it(
    tmp_path, fresh_viewer, monkeypatch
):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    sent = _record_glbs(fresh_viewer, monkeypatch)
    gate, _ = _gated_texture(monkeypatch)
    clear = viewer_module._clear

    # A later pick bumps the counter after b's bump, before b's build starts
    def bumped(semantics):
        if fresh_viewer.picks == 2:
            fresh_viewer.picks += 1

        clear(semantics)

    monkeypatch.setattr(viewer_module, "_clear", bumped)
    backends = {"a": _backend(a), "b": _textured_backend(b)}
    scene = _browse(fresh_viewer, backends, textured=True, texture_size=64)
    scene.value = "b"
    gate.set()
    _join_texture()

    assert fresh_viewer.picks == 3
    assert not any(_jpeg(data) for _, data in sent)


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
    _build(
        fresh_viewer,
        tmp_path,
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )
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
    _build(
        fresh_viewer,
        tmp_path,
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )

    _inputs(fresh_viewer)["Search"]._impl.update_cb[0](None)

    assert calls[-1][1] is None
    assert stub_text.built == 0


def test_switching_to_none_clears_heat_and_removes_the_mode_gui(
    tmp_path, fresh_viewer, stub_text, monkeypatch
):
    _vertex_store(_backend(tmp_path), "stub_text", np.eye(4, 2))
    calls = _record_heat(fresh_viewer, monkeypatch)
    dropdown = _build(
        fresh_viewer,
        tmp_path,
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )
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
    dropdown = _build(
        fresh_viewer,
        backend,
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )

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
    _build(
        fresh_viewer,
        _word_backend(tmp_path),
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )
    inputs = _inputs(fresh_viewer)
    inputs["Query"].value = "apple, Sky, nope"
    inputs["Search"]._impl.update_cb[0](None)

    name, scores, floor = calls[-1]
    assert name == "mesh" and floor == inputs["Query min p"].value
    np.testing.assert_allclose(scores, [0.75, 0.5, 1.0, np.nan], equal_nan=True)
    assert notes[-1].content == "not in vocabulary: nope"


def test_word_mode_label_list_ranks_words_by_probability_mass(tmp_path, fresh_viewer):
    _build(
        fresh_viewer,
        _word_backend(tmp_path),
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )

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
    _build(
        fresh_viewer,
        backend,
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
    )

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
        fresh_viewer,
        _word_backend(tmp_path),
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
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
        fresh_viewer,
        _word_backend(tmp_path),
        textured=False,
        texture_size=64,
        progress=_progress(fresh_viewer),
        pick=fresh_viewer.picks,
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
    ).stdout

    assert "local" in out and "remote" in out


def test_parser_local_and_remote_help_name_their_options(capsys):
    parser = viewer_module._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["local", "--help"])

    local = capsys.readouterr().out
    assert "outputs root" in local and "--texture-size" in local
    assert "--no-texture" in local and "--textured" not in local

    with pytest.raises(SystemExit):
        parser.parse_args(["remote", "--help"])

    remote = capsys.readouterr().out
    assert "--cache" in remote and "--port" in remote


def test_main_remote_exits_when_the_bucket_cannot_be_listed(monkeypatch, capsys):
    class _Unreachable:
        def list_processed_scenes(self):
            raise RuntimeError("rclone is not available")

    monkeypatch.setattr(viewer_module, "SceneSource", _Unreachable)

    with pytest.raises(SystemExit) as exit_info:
        viewer_module.main(["remote"])

    assert exit_info.value.code == 2
    assert "rclone is not available" in capsys.readouterr().err


def test_main_remote_deletes_pulled_scenes_on_exit(tmp_path, monkeypatch):
    # Listed scenes s1 (pulled) and s2 (never picked); an unrelated dir shares the cache
    class _Listing:
        def list_processed_scenes(self):
            return ["s1", "s2"]

    def _pull(viewer, source, scenes, cache_dir, textured, texture_size):
        (cache_dir / "s1/omega").mkdir(parents=True)
        _backend(cache_dir / "s1/omega")

    def _exit_at_once(viewer):
        viewer.server.stop()

    handlers = {}
    (tmp_path / "keep").mkdir()
    monkeypatch.setattr(viewer_module, "SceneSource", _Listing)
    monkeypatch.setattr(viewer_module, "_remote", _pull)
    monkeypatch.setattr(viewer_module.Viewer, "serve_forever", _exit_at_once)
    monkeypatch.setattr(viewer_module.signal, "signal", handlers.__setitem__)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    viewer_module.main(["remote", "--cache", str(tmp_path), "--port", str(port)])

    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep"]
    assert set(handlers) == {viewer_module.signal.SIGTERM, viewer_module.signal.SIGHUP}


def test_main_remote_clears_listed_scenes_left_by_a_killed_run(tmp_path, monkeypatch):
    # s2 survived a SIGKILL; keep is not a listed scene
    class _Listing:
        def list_processed_scenes(self):
            return ["s1", "s2"]

    seen = []

    def _pull(viewer, source, scenes, cache_dir, textured, texture_size):
        seen.append((sorted(p.name for p in cache_dir.iterdir()), textured))

    (tmp_path / "s2/omega").mkdir(parents=True)
    (tmp_path / "keep").mkdir()
    monkeypatch.setattr(viewer_module, "SceneSource", _Listing)
    monkeypatch.setattr(viewer_module, "_remote", _pull)
    monkeypatch.setattr(
        viewer_module.Viewer, "serve_forever", lambda v: v.server.stop()
    )
    monkeypatch.setattr(viewer_module.signal, "signal", lambda *a: None)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    viewer_module.main(["remote", "--cache", str(tmp_path), "--port", str(port)])
    viewer_module.main(
        ["remote", "--cache", str(tmp_path), "--port", str(port), "--no-texture"]
    )

    assert seen == [(["keep"], True), (["keep"], False)]


def test_main_local_exits_on_a_root_without_meshes(tmp_path, capsys):
    with pytest.raises(SystemExit):
        viewer_module.main(["local", str(tmp_path)])

    assert "no mesh.ply" in capsys.readouterr().err


def test_find_backends_lists_meshed_scene_backends_by_label(tmp_path):
    # Two meshed backends, one backend without a mesh, one mesh too shallow to be a backend
    for backend in ("b_scene/vggt_omega", "a_scene/vggt_omega", "a_scene/vggt_x"):
        (tmp_path / backend).mkdir(parents=True)

    _backend(tmp_path / "b_scene/vggt_omega")
    _backend(tmp_path / "a_scene/vggt_omega")
    _backend(tmp_path / "a_scene")

    assert _find_backends(tmp_path) == {
        "a_scene/vggt_omega": tmp_path / "a_scene/vggt_omega",
        "b_scene/vggt_omega": tmp_path / "b_scene/vggt_omega",
    }


def test_browse_switching_scenes_replaces_the_mesh_and_rebuilds_semantics(
    tmp_path, fresh_viewer, stub_text
):
    # Scene a has a stub_text store; scene b is a larger quad with none
    a, b = tmp_path / "a/omega", tmp_path / "b/omega"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _vertex_store(_backend(a), "stub_text", np.eye(4, 2))
    vertices, faces, _ = _quad()
    trimesh.Trimesh(vertices * 2, faces, process=False).export(b / "mesh.ply")

    scene = _browse(
        fresh_viewer, _find_backends(tmp_path), textured=False, texture_size=64
    )
    assert scene.options == ("a/omega", "b/omega") and scene.value == "a/omega"
    assert {"Semantics", "Query"} <= set(_inputs(fresh_viewer))

    scene.value = "b/omega"
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], vertices * 2)
    assert not {"Semantics", "Query"} & set(_inputs(fresh_viewer))

    # Back to a: exactly one Semantics dropdown, on a's store
    scene.value = "a/omega"
    handles = fresh_viewer.server.gui._gui_input_handle_from_uuid.values()
    labels = [h.label for h in handles if hasattr(h, "label")]
    assert labels.count("Semantics") == 1 and labels.count("Query") == 1
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], vertices)


_STATS = "2026/10/22 05:01:30 NOTICE:    1 MiB / 4 MiB, 23%, 1 MiB/s, ETA 3s"
_SEMANTICS = (
    "/*/semantics/*_lifted.zarr/zarr.json",
    "/*/semantics/*_lifted.zarr/autoencoder.pt",
    "/*/semantics/*_lifted.zarr/vertex_*/**",
)
_FAST = ("--transfers", "16", "--checkers", "32")


class _FakeSource:
    """
    SceneSource stand-in: records each pull, reports one stats line, writes mesh.ply when asked.
    """

    def __init__(self, scale=1.0, fail=False):
        self.pulls, self.scale, self.fail = [], scale, fail

    def pull_processed(self, scene, dest_dir, includes=(), extra=(), on_line=None):
        self.pulls.append((scene, dest_dir, includes, extra))

        if self.fail:
            raise RuntimeError("rclone copy failed (exit 7)")

        on_line(_STATS)

        if "/*/mesh.ply" in includes:
            vertices, faces, _ = _quad()
            (dest_dir / "omega").mkdir(parents=True, exist_ok=True)
            trimesh.Trimesh(vertices * self.scale, faces, process=False).export(
                dest_dir / "omega" / "mesh.ply"
            )

        return dest_dir


def test_remote_pulls_the_mesh_then_texture_and_semantics_and_shows_it(
    tmp_path, fresh_viewer, monkeypatch
):
    texts = _record_status(monkeypatch)
    source = _FakeSource()
    scene = _remote(
        fresh_viewer, source, ["s1", "s2"], tmp_path, textured=True, texture_size=64
    )

    # Two pulls: mesh.ply alone, then texture + vertex semantics; never point features
    assert scene.options == ("s1", "s2") and scene.label == "Scene"
    assert source.pulls == [
        ("s1", tmp_path / "s1", ("/*/mesh.ply",), _FAST),
        ("s1", tmp_path / "s1", ("/*/texture/**", *_SEMANTICS), _FAST),
    ]
    assert "pulling mesh: 1 MiB / 4 MiB, 23%, 1 MiB/s, ETA 3s" in texts
    assert "pulling texture + semantics: 1 MiB / 4 MiB, 23%, 1 MiB/s, ETA 3s" in texts
    assert texts[-1] == "no texture"
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], _quad()[0])

    # Picking s2 pulls it and swaps the mesh
    source.scale = 2.0
    scene.value = "s2"
    assert source.pulls[-1][:2] == ("s2", tmp_path / "s2")
    assert scene.label == "Scene" and scene.disabled is False
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], _quad()[0] * 2)


def test_remote_without_texture_pulls_semantics_only(tmp_path, fresh_viewer):
    source = _FakeSource()
    _remote(fresh_viewer, source, ["s1"], tmp_path, textured=False, texture_size=64)

    assert source.pulls[1][2] == _SEMANTICS


def test_remote_failed_pull_reports_it_and_keeps_the_shown_mesh(
    tmp_path, fresh_viewer, monkeypatch, caplog
):
    made = _capture_progress(monkeypatch)
    source = _FakeSource()
    scene = _remote(
        fresh_viewer, source, ["s1", "s2"], tmp_path, textured=False, texture_size=64
    )

    source.fail = True
    scene.value = "s2"

    bar, note = made[0]
    assert (
        note.content == "pull failed: rclone copy failed (exit 7)" and not bar.visible
    )
    assert (
        scene.label == "Scene" and scene.disabled is False and "exit 7" in caplog.text
    )
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], _quad()[0])


def test_remote_failed_second_pull_keeps_the_new_vertex_color_mesh(
    tmp_path, fresh_viewer, monkeypatch
):
    made = _capture_progress(monkeypatch)
    source = _FakeSource()
    scene = _remote(
        fresh_viewer, source, ["s1", "s2"], tmp_path, textured=False, texture_size=64
    )

    # s2's mesh pulls; the texture + semantics pull fails
    pull = source.pull_processed

    def _mesh_only(scene, dest_dir, includes=(), extra=(), on_line=None):
        if "/*/mesh.ply" not in includes:
            raise RuntimeError("rclone copy failed (exit 5)")

        return pull(scene, dest_dir, includes, extra, on_line)

    source.pull_processed = _mesh_only
    source.scale = 2.0
    scene.value = "s2"

    assert made[0][1].content == "pull failed: rclone copy failed (exit 5)"
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], _quad()[0] * 2)


def test_remote_failed_build_reports_it_and_keeps_the_vertex_color_mesh(
    tmp_path, fresh_viewer, monkeypatch, caplog
):
    made = _capture_progress(monkeypatch)
    source = _FakeSource()
    scene = _remote(
        fresh_viewer, source, ["s1", "s2"], tmp_path, textured=False, texture_size=64
    )

    # s2 pulls, but the full build fails
    def _broken_build(*args, **kwargs):
        raise ValueError("bad mesh")

    monkeypatch.setattr(viewer_module, "_build", _broken_build)
    source.scale = 2.0
    scene.value = "s2"

    assert made[0][1].content == "load failed: bad mesh" and "bad mesh" in caplog.text
    assert scene.disabled is False
    np.testing.assert_allclose(fresh_viewer.meshes["mesh"][0], _quad()[0] * 2)


def test_remote_pull_without_a_mesh_counts_as_failed(
    tmp_path, fresh_viewer, monkeypatch
):
    made = _capture_progress(monkeypatch)
    source = _FakeSource()
    source.pull_processed = lambda scene, dest_dir, **kwargs: dest_dir

    _remote(fresh_viewer, source, ["s1"], tmp_path, textured=False, texture_size=64)

    assert made[0][1].content.startswith("pull failed: no <backend>/mesh.ply")
    assert "mesh" not in fresh_viewer.meshes


def test_remote_os_error_in_either_pull_reports_it_and_frees_the_bar(
    tmp_path, fresh_viewer, monkeypatch, caplog
):
    made = _capture_progress(monkeypatch)
    source = _FakeSource()
    scene = _remote(
        fresh_viewer, source, ["s1", "s2", "s3"], tmp_path, False, texture_size=64
    )
    bar, note = made[0]
    pull = source.pull_processed

    # s2: the mesh pull's mkdir hits the quota
    def _quota_mesh(scene, dest_dir, includes=(), extra=(), on_line=None):
        raise OSError(122, "Disk quota exceeded")

    source.pull_processed = _quota_mesh
    scene.value = "s2"
    assert note.content == "pull failed: [Errno 122] Disk quota exceeded"
    assert not bar.visible and scene.disabled is False
    assert "Disk quota exceeded" in caplog.text

    # s3: the mesh pulls, the texture + semantics pull hits the quota
    def _quota_rest(scene, dest_dir, includes=(), extra=(), on_line=None):
        if "/*/mesh.ply" not in includes:
            raise OSError(122, "Disk quota exceeded")

        return pull(scene, dest_dir, includes, extra, on_line)

    source.pull_processed = _quota_rest
    scene.value = "s3"
    assert note.content == "pull failed: [Errno 122] Disk quota exceeded"
    assert not bar.visible and scene.disabled is False


def test_remote_resets_the_view_once_per_pick(tmp_path, fresh_viewer, monkeypatch):
    resets = []
    monkeypatch.setattr(fresh_viewer, "_reset_view", lambda: resets.append(1))
    scene = _remote(
        fresh_viewer, _FakeSource(), ["s1", "s2"], tmp_path, False, texture_size=64
    )
    assert len(resets) == 1

    scene.value = "s2"
    assert len(resets) == 2
