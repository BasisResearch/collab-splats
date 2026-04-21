import numpy as np
import pytest
import pycolmap
from collab_splats.pointcloud.feedforward import build_pycolmap_reconstruction


def _make_inputs(n=3, p=50):
    pts3d = np.random.randn(p, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (p, 3), dtype=np.uint8)
    extrinsics = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)[:, :3, :]  # (n, 3, 4)
    intrinsics = np.tile(
        np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    names = [f"frame_{i:04d}.jpg" for i in range(n)]
    return pts3d, colors, extrinsics, intrinsics, names


def test_camera_image_point_counts():
    pts3d, colors, extrinsics, intrinsics, names = _make_inputs(n=3, p=50)
    recon = build_pycolmap_reconstruction(pts3d, colors, extrinsics, intrinsics, 512, 512, names)
    assert len(recon.cameras) == 3
    assert len(recon.images) == 3
    assert len(recon.points3D) == 50


def test_accepts_4x4_extrinsics():
    pts3d, colors, _, intrinsics, names = _make_inputs(n=2, p=10)
    extrinsics_4x4 = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)  # (2, 4, 4)
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics_4x4, intrinsics, 512, 512, names[:2]
    )
    assert len(recon.cameras) == 2


def test_simple_pinhole_model():
    pts3d, colors, extrinsics, intrinsics, names = _make_inputs(n=2, p=5)
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, 518, 518, names[:2],
        camera_model="SIMPLE_PINHOLE",
    )
    assert len(recon.cameras) == 2
