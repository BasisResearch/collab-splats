"""Binary PLY export: header shape, byte layout, open3d round-trip, density cap."""

import numpy as np
import open3d as o3d
import pytest

from collab_splats.pointcloud.export import write_pointcloud_ply


def _cloud(n=100):
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
    return points, colors


def test_writes_binary_little_endian_header(tmp_path):
    points, colors = _cloud(10)
    out = write_pointcloud_ply(points, colors, tmp_path / "sparse_pc.ply")
    head = out.read_bytes().split(b"end_header\n")[0]
    assert b"format binary_little_endian 1.0" in head
    assert b"element vertex 10" in head
    assert b"property float x" in head
    assert b"property uchar red" in head


def test_payload_is_15_bytes_per_vertex(tmp_path):
    points, colors = _cloud(50)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply")
    raw = out.read_bytes()
    payload = raw.split(b"end_header\n", 1)[1]
    assert len(payload) == 50 * 15


def test_open3d_reads_back_exact_values(tmp_path):
    points, colors = _cloud(200)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply")
    pcd = o3d.io.read_point_cloud(str(out))
    got_pts = np.asarray(pcd.points, dtype=np.float32)
    got_col = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    assert got_pts.shape == points.shape
    np.testing.assert_array_equal(got_pts, points)
    np.testing.assert_array_equal(got_col, colors)


def test_colors_default_to_mid_grey_when_absent(tmp_path):
    points, _ = _cloud(20)
    out = write_pointcloud_ply(points, None, tmp_path / "p.ply")
    pcd = o3d.io.read_point_cloud(str(out))
    col = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    assert col.shape == (20, 3)
    assert np.all(col == 128)


def test_max_points_caps_density(tmp_path):
    n = 1000
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    # Encode each point's own index into its color (little-endian in R, G) so a
    # surviving point's color can be decoded back to the original index — this
    # catches points and colors being reindexed out of step, which a bare count
    # assertion cannot.
    idx = np.arange(n)
    colors = np.stack([idx % 256, idx // 256, np.zeros(n, dtype=np.int64)], axis=1).astype(np.uint8)

    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply", max_points=100)
    pcd = o3d.io.read_point_cloud(str(out))
    got_pts = np.asarray(pcd.points, dtype=np.float32)
    got_col = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    assert len(got_pts) == 100

    decoded_idx = got_col[:, 0].astype(np.int64) + got_col[:, 1].astype(np.int64) * 256
    np.testing.assert_array_equal(got_pts, points[decoded_idx])


def test_max_points_none_keeps_every_point(tmp_path):
    points, colors = _cloud(1000)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply", max_points=None)
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) == 1000


def test_rejects_mismatched_colors(tmp_path):
    points, colors = _cloud(30)
    with pytest.raises(ValueError, match="colors length"):
        write_pointcloud_ply(points, colors[:5], tmp_path / "p.ply")


def test_rejects_rgba_colors(tmp_path):
    points, colors = _cloud(10)
    rgba = np.concatenate([colors, np.full((10, 1), 255, dtype=np.uint8)], axis=1)
    with pytest.raises(ValueError, match="colors must be"):
        write_pointcloud_ply(points, rgba, tmp_path / "p.ply")


def test_rejects_1d_colors(tmp_path):
    points, _ = _cloud(10)
    grey = np.full(10, 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="colors must be"):
        write_pointcloud_ply(points, grey, tmp_path / "p.ply")


def test_rejects_float_colors(tmp_path):
    points, colors = _cloud(10)
    float_colors = colors.astype(np.float32) / 255.0
    with pytest.raises(ValueError, match="uint8"):
        write_pointcloud_ply(points, float_colors, tmp_path / "p.ply")


def test_creates_parent_directories(tmp_path):
    points, colors = _cloud(5)
    out = write_pointcloud_ply(points, colors, tmp_path / "nested" / "dir" / "p.ply")
    assert out.exists()
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) == 5


def test_empty_cloud_writes_zero_vertex_header(tmp_path):
    points = np.zeros((0, 3), dtype=np.float32)
    colors = np.zeros((0, 3), dtype=np.uint8)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply")
    raw = out.read_bytes()
    head, payload = raw.split(b"end_header\n", 1)
    assert b"element vertex 0" in head
    assert len(payload) == 0
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) == 0
