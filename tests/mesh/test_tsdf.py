import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.tsdf import create_tsdf_mesh


def _views(n=3, h=32, w=32):
    """n cameras looking down +z at a plane 1 unit away, shifted 0.05 along x per view."""
    depths = np.ones((n, h, w), dtype=np.float32)
    rgbs = np.full((n, h, w, 3), 128, dtype=np.uint8)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, 0, 3] = 0.05 * np.arange(n)
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64), (n, 1, 1))
    return depths, rgbs, c2w, K


def test_create_tsdf_mesh_writes_mesh_ply(tmp_path):
    depths, rgbs, c2w, K = _views()
    out = create_tsdf_mesh(depths, rgbs, c2w, K, tmp_path / "a" / "b", voxel_size=0.02, depth_trunc=2.0)
    assert out == tmp_path / "a" / "b" / "mesh.ply" and out.exists()
    mesh = o3d.io.read_triangle_mesh(str(out))
    assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()
    assert np.allclose(np.asarray(mesh.vertex_colors), 128 / 255, atol=0.05)


def test_create_tsdf_mesh_falls_back_to_cpu_without_cuda(tmp_path, monkeypatch):
    depths, rgbs, c2w, K = _views()
    gpu = create_tsdf_mesh(depths, rgbs, c2w, K, tmp_path / "gpu", voxel_size=0.02, depth_trunc=2.0)

    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    cpu = create_tsdf_mesh(depths, rgbs, c2w, K, tmp_path / "cpu", voxel_size=0.02, depth_trunc=2.0)
    gpu = o3d.io.read_triangle_mesh(str(gpu))
    cpu = o3d.io.read_triangle_mesh(str(cpu))

    assert len(cpu.triangles) > 0 and cpu.has_vertex_colors()
    assert np.allclose(cpu.get_center(), gpu.get_center(), atol=0.02)  # same plane, within a voxel


def test_create_tsdf_mesh_rejects_float_rgb(tmp_path):
    depths, rgbs, c2w, K = _views()

    with pytest.raises(ValueError, match="uint8"):
        create_tsdf_mesh(depths, rgbs.astype(np.float32) / 255, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_create_tsdf_mesh_rejects_principal_point_outside_grid(tmp_path):
    depths, rgbs, c2w, K = _views()
    K[:, 0, 2] = 64  # cx beyond the 32-wide grid: K is at a different resolution than depth

    with pytest.raises(ValueError, match="Principal point"):
        create_tsdf_mesh(depths, rgbs, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_create_tsdf_mesh_rejects_frame_count_mismatch(tmp_path):
    depths, rgbs, c2w, K = _views()

    with pytest.raises(ValueError, match="views"):
        create_tsdf_mesh(depths, rgbs[:2], c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_create_tsdf_mesh_sdf_trunc_defaults_to_four_voxels(tmp_path, monkeypatch):
    seen = []

    class _Recorder:
        def __init__(self, *, voxel_size, **kwargs):
            seen.append(voxel_size)

        def compute_unique_block_coordinates(self, *args):
            seen.append(args[-1])

        def integrate(self, *args):
            seen.append(args[-1])

        def extract_triangle_mesh(self, weight_threshold):
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.1)
            sphere.paint_uniform_color((0.5, 0.5, 0.5))
            return o3d.t.geometry.TriangleMesh.from_legacy(sphere)

    monkeypatch.setattr(o3d.t.geometry, "VoxelBlockGrid", _Recorder)
    depths, rgbs, c2w, K = _views(n=1)
    create_tsdf_mesh(depths, rgbs, c2w, K, tmp_path, voxel_size=0.01, depth_trunc=2.0)
    assert seen == [0.01, pytest.approx(4.0), pytest.approx(4.0)]


def test_create_tsdf_mesh_refuses_a_band_narrower_than_a_voxel(tmp_path):
    depths = np.ones((1, 4, 4), np.float32)
    rgbs = np.zeros((1, 4, 4, 3), np.uint8)

    with pytest.raises(ValueError, match="sdf_trunc"):
        create_tsdf_mesh(
            depths, rgbs, np.eye(4)[None], np.eye(3)[None], tmp_path, voxel_size=0.1, depth_trunc=5.0, sdf_trunc=0.05
        )
