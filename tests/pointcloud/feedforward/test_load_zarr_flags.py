"""load_zarr must be able to skip decoding dense optional arrays for the display path."""

import numpy as np

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _tiny_result_with_dense(tmp_path):
    """Minimal FeedforwardResult with required members + a dense depth array; save_zarr it."""
    n, p, h, w = 2, 5, 4, 4
    result = FeedforwardResult(
        points=np.zeros((p, 3), dtype=np.float32),
        colors=np.zeros((p, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        image_paths=[tmp_path / f"{i:05d}.jpg" for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
        depth=np.ones((n, h, w), dtype=np.float32),
    )
    store = tmp_path / "feedforward.zarr"
    result.save_zarr(store)
    return store


def test_load_zarr_can_skip_depth(tmp_path):
    store = _tiny_result_with_dense(tmp_path)
    lean = FeedforwardResult.load_zarr(store, load_depth=False)
    assert lean.depth is None  # skipped, not decoded
    full = FeedforwardResult.load_zarr(store)
    assert full.depth is not None  # default unchanged (back-compat)
    assert full.points.shape == (5, 3)


def test_load_zarr_lean_flags_skip_all_dense(tmp_path):
    store = _tiny_result_with_dense(tmp_path)
    lean = FeedforwardResult.load_zarr(
        store,
        load_depth=False,
        load_world_points=False,
        load_confidence=False,
        load_features=False,
        load_pixel_indices=False,
    )
    # Required members always decoded; every dense optional skipped.
    assert lean.points.shape == (5, 3)
    assert lean.extrinsics.shape == (2, 4, 4)
    for field in ("depth", "world_points", "confidence", "features", "pixel_indices"):
        assert getattr(lean, field) is None
    # The source path is kept so consumers can reload dense members on demand.
    assert lean._zarr_path == store
