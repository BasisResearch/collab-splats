"""Tests for PointcloudResult zarr backend (save_zarr / load_zarr).

Covers:
- Roundtrip of all core required fields
- world_points never written; load_zarr unprojects them from depth
- Missing optional fields load as None
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from collab_splats.geometry.projection import project, unproject
from collab_splats.pointcloud.base import PointcloudResult


def _make_result(
    n_frames: int = 3,
    n_pts: int = 100,
    h: int = 8,
    w: int = 8,
    *,
    with_world_points: bool = False,
    with_images: bool = False,
    with_pixel_indices: bool = False,
) -> PointcloudResult:
    """Build a minimal PointcloudResult for testing."""
    rng = np.random.default_rng(42)
    return PointcloudResult(
        points=rng.random((n_pts, 3), dtype=np.float32),
        colors=rng.integers(0, 256, (n_pts, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n_frames, axis=0),
        intrinsics=None,
        model_intrinsics=np.eye(3, dtype=np.float32)[None].repeat(n_frames, axis=0),
        image_paths=[Path(f"/tmp/frame_{i:04d}.png") for i in range(n_frames)],
        original_coords=np.tile(
            np.array([0, 60, 640, 420, 640, 480], dtype=np.float32), (n_frames, 1)
        ),
        model_width=w,
        model_height=h,
        world_points=rng.random((n_frames, h, w, 3), dtype=np.float32)
        if with_world_points
        else None,
        pixel_indices=rng.integers(0, n_frames, (n_pts, 3), dtype=np.int32)
        if with_pixel_indices
        else None,
    )


def test_zarr_roundtrip_core_fields(tmp_path):
    """Core required arrays and metadata survive a save/load roundtrip."""
    result = _make_result(n_frames=3, n_pts=50)
    store_path = tmp_path / "result.zarr"

    result.save_zarr(store_path)
    loaded = PointcloudResult.load_zarr(store_path)

    np.testing.assert_array_equal(loaded.points, result.points)
    np.testing.assert_array_equal(loaded.colors, result.colors)
    np.testing.assert_array_equal(loaded.extrinsics, result.extrinsics)
    np.testing.assert_array_equal(loaded.intrinsics, result.intrinsics)
    np.testing.assert_array_equal(loaded.original_coords, result.original_coords)

    assert loaded.model_width == result.model_width
    assert loaded.model_height == result.model_height
    assert loaded.image_paths == result.image_paths


def _posed_result_with_depth(
    n_frames: int = 3, h: int = 12, w: int = 16
) -> PointcloudResult:
    """Result with random depth, distinct w2c poses and a pinhole model-grid K."""
    rng = np.random.default_rng(0)
    result = _make_result(n_frames=n_frames, h=h, w=w, with_world_points=True)
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    extrinsics[:, :3, 3] = rng.normal(size=(n_frames, 3))
    angle = np.linspace(0.0, 0.5, n_frames)
    extrinsics[:, 0, 0] = extrinsics[:, 2, 2] = np.cos(angle)
    extrinsics[:, 0, 2] = np.sin(angle)
    extrinsics[:, 2, 0] = -np.sin(angle)
    K = np.array([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]], dtype=np.float32)
    result.extrinsics = extrinsics
    result.model_intrinsics = np.tile(K, (n_frames, 1, 1))
    result.depth = rng.uniform(1.0, 5.0, (n_frames, h, w)).astype(np.float32)
    return result


def test_zarr_does_not_write_world_points(tmp_path):
    """save_zarr leaves world_points out of the store even when the result holds them."""
    result = _posed_result_with_depth()
    store_path = tmp_path / "result.zarr"

    result.save_zarr(store_path)

    assert "world_points" not in zarr.open(str(store_path), mode="r")


def test_zarr_world_points_unprojected_from_depth(tmp_path):
    """Loaded world_points equal depth unprojected under the stored extrinsics and model-grid K."""
    result = _posed_result_with_depth()
    store_path = tmp_path / "result.zarr"
    result.save_zarr(store_path)

    loaded = PointcloudResult.load_zarr(store_path)
    expected = unproject(
        torch.from_numpy(result.depth),
        torch.from_numpy(result.extrinsics),
        torch.from_numpy(result.model_intrinsics),
    )

    np.testing.assert_allclose(loaded.world_points, expected.numpy(), atol=1e-4)


def test_zarr_world_points_reproject_to_their_pixels(tmp_path):
    """Each loaded world point projects back onto its own pixel in its own camera."""
    result = _posed_result_with_depth(n_frames=2, h=6, w=8)
    store_path = tmp_path / "result.zarr"
    result.save_zarr(store_path)

    loaded = PointcloudResult.load_zarr(store_path)
    grid_v, grid_u = np.meshgrid(np.arange(6), np.arange(8), indexing="ij")

    for i in range(2):
        pixels, _ = project(
            torch.from_numpy(loaded.world_points[i]),
            torch.from_numpy(loaded.extrinsics[i]),
            torch.from_numpy(loaded.model_intrinsics[i]),
        )
        np.testing.assert_allclose(pixels[..., 0].numpy(), grid_u, atol=1e-3)
        np.testing.assert_allclose(pixels[..., 1].numpy(), grid_v, atol=1e-3)


def test_zarr_world_points_without_depth(tmp_path):
    """load_depth=False still unprojects world_points but returns no depth."""
    result = _posed_result_with_depth()
    store_path = tmp_path / "result.zarr"
    result.save_zarr(store_path)

    loaded = PointcloudResult.load_zarr(store_path, load_depth=False)

    assert loaded.depth is None
    assert loaded.world_points.shape == (*result.depth.shape, 3)


def test_zarr_missing_optional_fields_load_as_none(tmp_path):
    """Optional fields absent from the store load as None."""
    result = _make_result(n_frames=2, with_world_points=False, with_pixel_indices=False)
    store_path = tmp_path / "result_minimal.zarr"

    result.save_zarr(store_path)
    loaded = PointcloudResult.load_zarr(store_path)

    assert loaded.world_points is None
    assert loaded.pixel_indices is None
    # images always None on load
    assert loaded.images is None
    assert loaded.confidence is None


def test_zarr_optional_fields_roundtrip(tmp_path):
    """pixel_indices survives a roundtrip when present."""
    result = _make_result(n_pts=80, with_pixel_indices=True)
    store_path = tmp_path / "result.zarr"

    result.save_zarr(store_path)
    loaded = PointcloudResult.load_zarr(store_path)

    assert loaded.pixel_indices is not None
    np.testing.assert_array_equal(loaded.pixel_indices, result.pixel_indices)
