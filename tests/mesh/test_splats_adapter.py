"""
_splats_to_tsdf_inputs: splats.zarr -> (depths, rgbs, c2w, K) with alpha as the confidence gate.
"""

import numpy as np
import pytest
import zarr

from collab_splats.mesh.utils import _splats_to_tsdf_inputs


def _write_splats_zarr(path, n_views=3, height=4, width=5):
    store = zarr.open_group(path, mode="w")
    rgb = np.full((n_views, height, width, 3), 7, np.uint8)
    depth = np.ones((n_views, height, width), np.float32)
    alpha = np.linspace(0.0, 1.0, n_views * height * width, dtype=np.float32).reshape(n_views, height, width)
    c2w = np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1))
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1))
    for name, array in (("rgb", rgb), ("depth", depth), ("alpha", alpha), ("c2w", c2w), ("K", intrinsics)):
        store.create_array(name, data=array)
    store.attrs["primitive"] = "3dgs"
    return alpha


def test_adapter_shapes_and_types(tmp_path):
    _write_splats_zarr(tmp_path / "splats.zarr")
    depths, rgbs, c2w, intrinsics = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=None)
    assert depths.shape == (3, 4, 5) and depths.dtype == np.float32
    assert rgbs.shape == (3, 4, 5, 3) and rgbs.dtype == np.uint8
    assert c2w.shape == (3, 4, 4) and intrinsics.shape == (3, 3, 3)
    assert depths.max() == 1.0  # no percentile -> nothing dropped except alpha == 0


def test_adapter_alpha_percentile_zeroes_depth(tmp_path):
    alpha = _write_splats_zarr(tmp_path / "splats.zarr")
    depths, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=50)
    dropped = depths == 0.0
    assert 0.45 < dropped.mean() < 0.55  # global percentile over all views
    assert np.all(alpha[dropped] <= np.percentile(alpha, 50))


def test_adapter_always_drops_zero_alpha(tmp_path):
    _write_splats_zarr(tmp_path / "splats.zarr")
    depths, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=None)
    assert depths.flat[0] == 0.0  # alpha == 0 at the first pixel -> no observation
    assert depths.flat[1] == 1.0  # every alpha > 0 pixel survives


def test_adapter_missing_zarr_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="splats.zarr"):
        _splats_to_tsdf_inputs(tmp_path / "splats.zarr", conf_percentile=None)


def _write_median_splats_zarr(path, with_median=True, n_views=2, height=4, width=4):
    """
    Write a minimal splats.zarr whose expected and median depths are distinguishable.
    """
    store = zarr.open_group(str(path), mode="w")
    store.create_array("depth", data=np.full((n_views, height, width), 1.0, dtype=np.float32))
    if with_median:
        store.create_array("median_depth", data=np.full((n_views, height, width), 3.0, dtype=np.float32))
    store.create_array("alpha", data=np.ones((n_views, height, width), dtype=np.float32))
    store.create_array("rgb", data=np.zeros((n_views, height, width, 3), dtype=np.uint8))
    store.create_array("c2w", data=np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1)))
    store.create_array("K", data=np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1)))


def test_splat_depth_median_reads_the_median_array(tmp_path):
    path = tmp_path / "splats.zarr"
    _write_median_splats_zarr(path)

    expected, _rgbs, _c2w, _K = _splats_to_tsdf_inputs(path, splat_depth="expected")
    median, _rgbs, _c2w, _K = _splats_to_tsdf_inputs(path, splat_depth="median")
    assert expected[0, 0, 0] == 1.0
    assert median[0, 0, 0] == 3.0


def test_splat_depth_median_missing_raises_actionably(tmp_path):
    path = tmp_path / "splats.zarr"
    _write_median_splats_zarr(path, with_median=False)

    with pytest.raises(ValueError, match="median_depth"):
        _splats_to_tsdf_inputs(path, splat_depth="median")


def test_splat_depth_rejects_an_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="splat_depth"):
        _splats_to_tsdf_inputs(tmp_path / "splats.zarr", splat_depth="surf")
