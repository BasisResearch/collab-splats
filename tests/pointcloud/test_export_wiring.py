"""BasePointcloudCreator._write_ply turns a result into a binary sparse_pc.ply."""

from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

from collab_splats.pointcloud.base import (
    BasePointcloudCreator,
    CoordinateFrame,
    PointcloudResult,
)


class _StubResult:
    """Minimal stand-in for PointcloudResult — only points/colors are read."""

    def __init__(self, points, colors):
        self.points = points
        self.colors = colors


class _Creator(BasePointcloudCreator):
    def reconstruct(self, image_dir, output_dir):  # pragma: no cover - not exercised
        raise NotImplementedError


def test_write_ply_emits_binary_sparse_pc(tmp_path):
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, size=(40, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(40, 3)).astype(np.uint8)

    out = _Creator()._write_ply(_StubResult(points, colors), tmp_path)

    assert out == tmp_path / "sparse_pc.ply"
    assert b"format binary_little_endian 1.0" in out.read_bytes()[:80]
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) == 40
    # open3d normalises colours to 0-1 floats on read — round-trip back to uint8 to compare
    np.testing.assert_allclose(np.asarray(pcd.points), points, atol=1e-6)
    np.testing.assert_array_equal((np.asarray(pcd.colors) * 255).round().astype(np.uint8), colors)


def test_write_ply_honours_max_points(tmp_path):
    points = np.zeros((500, 3), dtype=np.float32)
    out = _Creator()._write_ply(_StubResult(points, None), tmp_path, max_points=50)
    assert len(o3d.io.read_point_cloud(str(out)).points) == 50


def test_write_ply_accepts_real_pointcloud_result(tmp_path):
    """_write_ply reads points/colors off an actual PointcloudResult, not just a duck-type."""
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=4, height=4, params=[2.0, 2.0, 2.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    img = pycolmap.Image(name="frame_0000.jpg", camera_id=1, image_id=1)
    recon.add_image_with_trivial_frame(img, pycolmap.Rigid3d())
    for i in range(3):
        recon.add_point3D(
            xyz=np.array([float(i), 0.0, 1.0]),
            track=pycolmap.Track(),
            color=np.array([10 * i, 20 * i, 30 * i], dtype=np.uint8),
        )

    result = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=[Path("frame_0000.jpg")],
    )
    out = _Creator()._write_ply(result, tmp_path)
    assert len(o3d.io.read_point_cloud(str(out)).points) == 3
