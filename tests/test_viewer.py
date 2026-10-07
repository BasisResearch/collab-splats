"""Smoke tests for the generic viser scene Viewer (no browser needed)."""

import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from collab_splats.viewer import Viewer


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
    intrinsic = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32)
    viewer.add_frustum("cams/frame_0", pose, intrinsic)
    assert "cams/frame_0" in viewer.frustums
    assert viewer.frustums["cams/frame_0"].visible


def test_add_frustum_with_image(viewer):
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float32)
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


def test_add_mesh_shows_the_texture_and_picks_on_the_given_vertices(viewer, monkeypatch):
    vertices, faces, colors = _quad()
    corners = vertices[faces].reshape(-1, 3)
    uv = corners[:, :2]
    image = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    textured = trimesh.Trimesh(corners, np.arange(6).reshape(2, 3), visual=visual, process=False)
    shown = []
    monkeypatch.setattr(viewer.server.scene, "add_mesh_trimesh", lambda name, mesh: shown.append(mesh))

    viewer.add_mesh("textured_quad", vertices, faces, colors, textured=textured)

    assert shown == [textured]
    assert viewer._pick_vertex("textured_quad", (0.9, 0.9, 1.0), (0.0, 0.0, -1.0))[1] == 3


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


def test_show_heat_keeps_faces_reaching_the_floor_on_their_used_vertices(viewer, monkeypatch):
    vertices, faces, colors = _quad()
    viewer.add_mesh("quad", vertices, faces, colors)
    sent = _capture_heat(viewer, monkeypatch)

    # Only vertex 3 reaches the floor: face [1, 3, 2] stays, face [0, 1, 2] goes
    viewer.show_heat("quad", np.array([0.0, 0.1, 0.2, 0.9]), floor=0.5, offset=0.0)

    assert [name for name, _ in sent] == ["quad/heat"]
    heat = sent[0][1]
    assert heat.vertices.tolist() == vertices[[1, 2, 3]].tolist()
    assert heat.faces.tolist() == [[0, 2, 1]]


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


def test_show_heat_replaces_clears_and_sends_nothing_below_the_floor(viewer, monkeypatch):
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
    assert [b.label for b in buttons] == ["Clear", "tree (~28.7k)", "sky (~950)", "rock (~12)"]

    # Word buttons hand their word over; Clear hands None
    buttons[1]._impl.update_cb[0](None)
    buttons[0]._impl.update_cb[0](None)
    assert selected == ["tree", None]

    # Re-adding replaces the list rather than stacking a second one
    viewer.add_label_list("quad", words, weights, selected.append, top_n=1)
    _, buttons = viewer.label_lists["quad"]
    assert [b.label for b in buttons] == ["Clear", "tree (~28.7k)"]


def test_on_click_registers_one_scene_handler_and_the_nearest_mesh_wins(viewer, monkeypatch):
    registered = []
    monkeypatch.setattr(viewer.server.scene, "on_click", lambda: registered.append)
    monkeypatch.setattr(viewer, "mesh_clicks", {})
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
    monkeypatch.setattr(viewer.server, "get_clients", lambda: {0: SimpleNamespace(camera=camera)})

    viewer.add_mesh("quad", *_quad())
    viewer._reset_view()
    assert camera.look_at.tolist() == [0.5, 0.5, 0.0]
