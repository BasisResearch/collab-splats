"""Tests for FeedforwardResult zarr backend (save_zarr / load_zarr).

Covers:
- Roundtrip of all core required fields
- world_points optional field included and chunked by frame
- Missing optional fields load as None
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(
    n_frames: int = 3,
    n_pts: int = 100,
    h: int = 8,
    w: int = 8,
    *,
    with_world_points: bool = False,
    with_images: bool = False,
    with_features: bool = False,
    with_pixel_indices: bool = False,
) -> FeedforwardResult:
    """Build a minimal FeedforwardResult for testing."""
    rng = np.random.default_rng(42)
    return FeedforwardResult(
        points=rng.random((n_pts, 3), dtype=np.float32),
        colors=rng.integers(0, 256, (n_pts, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n_frames, axis=0),
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(n_frames, axis=0),
        image_paths=[Path(f"/tmp/frame_{i:04d}.png") for i in range(n_frames)],
        original_coords=rng.random((n_frames, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
        world_points=rng.random((n_frames, h, w, 3), dtype=np.float32) if with_world_points else None,
        features=rng.random((n_pts, 64), dtype=np.float32) if with_features else None,
        pixel_indices=rng.integers(0, n_frames, (n_pts, 3), dtype=np.int32) if with_pixel_indices else None,
    )


def test_zarr_roundtrip_core_fields(tmp_path):
    """Core required arrays and metadata survive a save/load roundtrip."""
    result = _make_result(n_frames=3, n_pts=50)
    store_path = tmp_path / "result.zarr"

    result.save_zarr(store_path)
    loaded = FeedforwardResult.load_zarr(store_path)

    np.testing.assert_array_equal(loaded.points, result.points)
    np.testing.assert_array_equal(loaded.colors, result.colors)
    np.testing.assert_array_equal(loaded.extrinsics, result.extrinsics)
    np.testing.assert_array_equal(loaded.intrinsics, result.intrinsics)
    np.testing.assert_array_equal(loaded.original_coords, result.original_coords)

    assert loaded.model_width == result.model_width
    assert loaded.model_height == result.model_height
    assert loaded.image_paths == result.image_paths


def test_zarr_includes_world_points(tmp_path):
    """world_points is saved and loaded correctly when present."""
    result = _make_result(n_frames=4, h=16, w=16, with_world_points=True)
    store_path = tmp_path / "result_wp.zarr"

    result.save_zarr(store_path)
    loaded = FeedforwardResult.load_zarr(store_path)

    assert loaded.world_points is not None
    np.testing.assert_array_equal(loaded.world_points, result.world_points)


def test_zarr_world_points_chunked_by_frame(tmp_path):
    """world_points array is chunked with chunk size 1 along the frame axis."""
    import zarr

    n_frames = 5
    h, w = 12, 10
    result = _make_result(n_frames=n_frames, h=h, w=w, with_world_points=True)
    store_path = tmp_path / "result_chunks.zarr"

    result.save_zarr(store_path)

    store = zarr.open(str(store_path), mode="r")
    arr = store["world_points"]
    # chunk dim 0 should be 1 (one chunk per frame)
    assert arr.chunks[0] == 1
    assert arr.chunks[1] == h
    assert arr.chunks[2] == w


def test_zarr_missing_optional_fields_load_as_none(tmp_path):
    """Optional fields absent from the store load as None."""
    result = _make_result(n_frames=2, with_world_points=False, with_features=False, with_pixel_indices=False)
    store_path = tmp_path / "result_minimal.zarr"

    result.save_zarr(store_path)
    loaded = FeedforwardResult.load_zarr(store_path)

    assert loaded.world_points is None
    assert loaded.features is None
    assert loaded.pixel_indices is None
    # images always None on load
    assert loaded.images is None
    assert loaded.confidence is None


def test_zarr_optional_fields_roundtrip(tmp_path):
    """features and pixel_indices survive a roundtrip when present."""
    result = _make_result(n_pts=80, with_features=True, with_pixel_indices=True)
    store_path = tmp_path / "result_feats.zarr"

    result.save_zarr(store_path)
    loaded = FeedforwardResult.load_zarr(store_path)

    assert loaded.features is not None
    np.testing.assert_array_equal(loaded.features, result.features)
    assert loaded.pixel_indices is not None
    np.testing.assert_array_equal(loaded.pixel_indices, result.pixel_indices)
