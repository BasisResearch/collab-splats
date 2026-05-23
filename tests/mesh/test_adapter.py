from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from collab_splats.mesh import pointcloud_to_mesh
from collab_splats.mesh.base import MeshResult
from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(N=2, H=4, W=6, world_points_z=1.5, with_world_points=True):
    """Build a minimal FeedforwardResult with real image files in a temp dir."""
    rng = np.random.default_rng(42)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # identity w2c

    world_points = None
    if with_world_points:
        world_points = rng.random((N, H, W, 3)).astype(np.float32)
        world_points[..., 2] = world_points_z

    tmpdir = tempfile.mkdtemp()
    image_paths = []
    for i in range(N):
        img_arr = (rng.random((H * 4, W * 4, 3)) * 255).astype(np.uint8)
        p = Path(tmpdir) / f"frame_{i:04d}.png"
        PILImage.fromarray(img_arr).save(p)
        image_paths.append(p)

    original_coords = np.tile([0, 0, W * 4, H * 4, W * 4, H * 4], (N, 1)).astype(np.float32)
    return FeedforwardResult(
        pts3d=rng.random((10, 3)).astype(np.float32),
        colors=(rng.random((10, 3)) * 255).astype(np.uint8),
        extrinsics=extrinsics,
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(N, axis=0),
        image_paths=image_paths,
        original_coords=original_coords,
        model_width=W * 4,
        model_height=H * 4,
        world_points=world_points,
    )


def test_pointcloud_to_mesh_returns_mesh_result(tmp_path):
    result = _make_result(N=2, H=32, W=32)
    mesh_result = pointcloud_to_mesh(
        result, tmp_path / "mesh",
        method="open3d_tsdf",
        voxel_size=0.05,
        sdf_trunc=0.2,
        clean_repair=False,
    )
    assert isinstance(mesh_result, MeshResult)
    assert mesh_result.mesh_path.exists()


def test_pointcloud_to_mesh_raises_on_none_world_points(tmp_path):
    result = _make_result(with_world_points=False)
    with pytest.raises(ValueError, match="world_points"):
        pointcloud_to_mesh(result, tmp_path / "mesh")


def test_pointcloud_to_mesh_invalid_method(tmp_path):
    result = _make_result()
    with pytest.raises(ValueError, match="Unknown mesh method"):
        pointcloud_to_mesh(result, tmp_path / "mesh", method="nonexistent")
