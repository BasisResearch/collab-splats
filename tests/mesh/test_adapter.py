from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.mesh import pointcloud_to_mesh
from collab_splats.mesh.base import MeshResult
from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(N=2, H=32, W=32, depth_z=1.5, with_depth=True, with_images=True):
    """Minimal FeedforwardResult carrying the model-res depth + RGB the adapter consumes."""
    rng = np.random.default_rng(42)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # identity w2c

    # depth/images are what the model actually produced — no disk IO, no re-derivation
    depth = np.full((N, H, W), depth_z, dtype=np.float32) if with_depth else None
    images = torch.from_numpy(rng.random((N, 3, H, W)).astype(np.float32)) if with_images else None

    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = float(W)
    intrinsics[:, 0, 2] = W / 2
    intrinsics[:, 1, 2] = H / 2

    return FeedforwardResult(
        points=rng.random((10, 3)).astype(np.float32),
        colors=(rng.random((10, 3)) * 255).astype(np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i:04d}.png") for i in range(N)],
        original_coords=np.tile([0, 0, W, H, W, H], (N, 1)).astype(np.float32),
        model_width=W,
        model_height=H,
        images=images,
        depth=depth,
    )


def test_adapter_returns_depth_and_intrinsics_unchanged():
    """Depth and K pass through untouched — both are model-resolution and already aligned."""
    result = _make_result(N=2, H=8, W=6, depth_z=2.25)
    depths, _, _, intrinsics = _feedforward_to_tsdf_inputs(result)

    np.testing.assert_array_equal(depths, result.depth)
    np.testing.assert_array_equal(intrinsics, result.intrinsics)


def test_adapter_returns_rgb_in_unit_range_hwc():
    """images is (N, 3, H, W) in [0, 1]; the adapter transposes it, it does not rescale it."""
    result = _make_result(N=2, H=8, W=6)
    _, rgbs, _, _ = _feedforward_to_tsdf_inputs(result)

    assert rgbs.shape == (2, 8, 6, 3)
    assert rgbs.dtype == np.float32
    assert rgbs.max() <= 1.0
    np.testing.assert_allclose(rgbs, result.images.numpy().transpose(0, 2, 3, 1), rtol=0, atol=0)


def test_adapter_inverts_extrinsics_to_c2w():
    """c2w is the inverse of the stored w2c extrinsics."""
    result = _make_result(N=2, H=8, W=6)
    result.extrinsics[:, :3, 3] = [0.5, -1.0, 2.0]
    _, _, c2w, _ = _feedforward_to_tsdf_inputs(result)

    identity = np.eye(4, dtype=np.float32)[None].repeat(len(c2w), axis=0)
    np.testing.assert_allclose(c2w @ result.extrinsics, identity, atol=1e-5)


def test_adapter_raises_on_missing_depth():
    result = _make_result(with_depth=False)
    with pytest.raises(ValueError, match="depth"):
        _feedforward_to_tsdf_inputs(result)


def test_adapter_raises_on_missing_images():
    result = _make_result(with_images=False)
    with pytest.raises(ValueError, match="images"):
        _feedforward_to_tsdf_inputs(result)


def test_pointcloud_to_mesh_returns_mesh_result(tmp_path):
    result = _make_result(N=2, H=32, W=32)
    mesh_result = pointcloud_to_mesh(
        result,
        tmp_path / "mesh",
        method="open3d_tsdf",
        voxel_size=0.05,
        sdf_trunc=0.2,
        clean_repair=False,
    )
    assert isinstance(mesh_result, MeshResult)
    assert mesh_result.mesh_path.exists()


def test_pointcloud_to_mesh_invalid_method(tmp_path):
    result = _make_result()
    with pytest.raises(ValueError, match="Unknown mesh method"):
        pointcloud_to_mesh(result, tmp_path / "mesh", method="nonexistent")
