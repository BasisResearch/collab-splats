"""mv arrays are written only when computed — absent keys, never a zeros array."""

import numpy as np
import zarr

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _minimal_result(**extra) -> FeedforwardResult:
    """Smallest FeedforwardResult save_zarr accepts, plus whatever mv fields the test sets."""
    N, H, W = 2, 4, 4
    return FeedforwardResult(
        points=np.zeros((5, 3), np.float32),
        colors=np.zeros((5, 3), np.uint8),
        extrinsics=np.stack([np.eye(4, dtype=np.float32)] * N),
        intrinsics=np.stack([np.eye(3, dtype=np.float32)] * N),
        image_paths=[],
        original_coords=np.zeros((N, 6), np.float32),
        model_width=W,
        model_height=H,
        **extra,
    )


def test_mv_arrays_absent_when_not_computed(tmp_path):
    """A consumer cannot tell a zeros array from 'every pixel disagreed' — so write nothing."""
    _minimal_result().save_zarr(tmp_path / "ff.zarr")
    store = zarr.open(str(tmp_path / "ff.zarr"), mode="r")
    assert "mv_ratio" not in store
    assert "mv_inlier_count" not in store
    assert "mv_valid_count" not in store


def test_zarr_roundtrip_mv_arrays(tmp_path):
    """Ratio and both counts survive the round trip with their dtypes and per-frame chunking."""
    N, H, W = 2, 4, 4
    ratio = np.random.default_rng(0).random((N, H, W)).astype(np.float32)
    inlier = np.full((N, H, W), 3, np.int32)
    valid = np.full((N, H, W), 5, np.int32)
    _minimal_result(mv_ratio=ratio, mv_inlier_count=inlier, mv_valid_count=valid).save_zarr(tmp_path / "ff.zarr")

    store = zarr.open(str(tmp_path / "ff.zarr"), mode="r")
    np.testing.assert_array_equal(store["mv_ratio"][:], ratio)
    np.testing.assert_array_equal(store["mv_inlier_count"][:], inlier)
    np.testing.assert_array_equal(store["mv_valid_count"][:], valid)
    assert store["mv_inlier_count"].dtype == np.int32
    # chunked by frame, like its neighbours
    assert store["mv_ratio"].chunks == (1, H, W)
