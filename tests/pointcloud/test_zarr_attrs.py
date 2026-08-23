import numpy as np
import zarr

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _tiny_result():
    n, h, w = 2, 4, 6
    return FeedforwardResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        image_paths=[f"frame_{i:06d}.jpg" for i in range(n)],
        original_coords=np.zeros((n, 6), dtype=np.float32),
        model_width=w,
        model_height=h,
    )


def test_save_zarr_writes_extra_attrs(tmp_path):
    out = tmp_path / "pointcloud.zarr"
    _tiny_result().save_zarr(
        out, extra_attrs={"method": "sfm", "backend": "instantsfm", "instantsfm_version": "0.3.0"}
    )
    store = zarr.open(str(out), mode="r")
    assert store.attrs["method"] == "sfm"
    assert store.attrs["backend"] == "instantsfm"
    assert store.attrs["instantsfm_version"] == "0.3.0"


def test_save_zarr_no_extra_attrs_unchanged(tmp_path):
    out = tmp_path / "pointcloud.zarr"
    _tiny_result().save_zarr(out)
    store = zarr.open(str(out), mode="r")
    assert "method" not in store.attrs
    assert store.attrs["model_width"] == 6
