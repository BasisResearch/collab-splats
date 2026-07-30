"""pointcloud.max_points / export_max_points reach the creator and the PLY."""

import numpy as np
import open3d as o3d

from collab_splats.wrapper.reconstructor import Reconstructor


def _config(tmp_path, **pc):
    return {
        "input_path": str(tmp_path / "in.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega", **pc},
    }


def test_defaults_expose_max_points(tmp_path):
    r = Reconstructor(_config(tmp_path))
    assert r.config["pointcloud"]["max_points"] == 500_000
    assert r.config["pointcloud"]["export_max_points"] is None


def test_max_points_override_survives_merge(tmp_path):
    r = Reconstructor(_config(tmp_path, max_points=120_000, export_max_points=50_000))
    assert r.config["pointcloud"]["max_points"] == 120_000
    assert r.config["pointcloud"]["export_max_points"] == 50_000


def test_export_pointcloud_ply_applies_export_cap(tmp_path):
    """_export_pointcloud_ply writes backend_dir/sparse_pc.ply, thinned to export_max_points."""

    class _Result:
        points = np.zeros((300, 3), dtype=np.float32)
        colors = None

    r = Reconstructor(_config(tmp_path, export_max_points=100))
    r.backend_dir.mkdir(parents=True, exist_ok=True)
    out = r._export_pointcloud_ply(_Result())
    assert out == r.backend_dir / "sparse_pc.ply"
    assert len(o3d.io.read_point_cloud(str(out)).points) == 100
