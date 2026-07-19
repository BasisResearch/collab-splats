"""Smoke tests for the generic viser scene Viewer (no browser needed)."""

import socket

import numpy as np
import pytest

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
