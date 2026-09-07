import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.tsdf import fuse_tsdf


def _views(n=3, h=32, w=32):
    """n cameras looking down +z at a plane 1 unit away, shifted 0.05 along x per view."""
    depths = np.ones((n, h, w), dtype=np.float32)
    rgbs = np.full((n, h, w, 3), 128, dtype=np.uint8)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, 0, 3] = 0.05 * np.arange(n)
    K = np.tile(np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64), (n, 1, 1))
    return depths, rgbs, c2w, K


def test_fuse_tsdf_writes_mesh_ply(tmp_path):
    depths, rgbs, c2w, K = _views()
    out = fuse_tsdf(depths, rgbs, c2w, K, tmp_path / "a" / "b", voxel_size=0.02, depth_trunc=2.0)
    assert out == tmp_path / "a" / "b" / "mesh.ply" and out.exists()
    mesh = o3d.io.read_triangle_mesh(str(out))
    assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()
    assert np.allclose(np.asarray(mesh.vertex_colors), 128 / 255, atol=0.05)


def test_fuse_tsdf_rejects_float_rgb(tmp_path):
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="uint8"):
        fuse_tsdf(depths, rgbs.astype(np.float32) / 255, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_rejects_principal_point_outside_grid(tmp_path):
    depths, rgbs, c2w, K = _views()
    K[:, 0, 2] = 64  # cx beyond the 32-wide grid: K is at a different resolution than depth
    with pytest.raises(ValueError, match="Principal point"):
        fuse_tsdf(depths, rgbs, c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_rejects_frame_count_mismatch(tmp_path):
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="views"):
        fuse_tsdf(depths, rgbs[:2], c2w, K, tmp_path, voxel_size=0.02, depth_trunc=2.0)


def test_fuse_tsdf_sdf_trunc_defaults_to_four_voxels(tmp_path, monkeypatch):
    seen = []

    class _Recorder:
        def __init__(self, voxel_length, sdf_trunc, color_type):
            seen.append((voxel_length, sdf_trunc, color_type))

        def integrate(self, *args, **kwargs):
            pass

        def extract_triangle_mesh(self):
            return o3d.geometry.TriangleMesh.create_sphere(0.1)

    monkeypatch.setattr(o3d.pipelines.integration, "ScalableTSDFVolume", _Recorder)
    depths, rgbs, c2w, K = _views(n=1)
    fuse_tsdf(depths, rgbs, c2w, K, tmp_path, voxel_size=0.01, depth_trunc=2.0)
    assert len(seen) == 1
    assert seen[0][0] == 0.01 and seen[0][1] == pytest.approx(0.04)
    assert seen[0][2] == o3d.pipelines.integration.TSDFVolumeColorType.RGB8
