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

    # Pin the default against a store that HAS median_depth — otherwise a flipped default
    # only trips the missing-array guard, and that catch dies the day a fixture gains the array
    default, _rgbs, _c2w, _K = _splats_to_tsdf_inputs(path)
    assert default[0, 0, 0] == 1.0  # omitted splat_depth means expected, not median


def test_splat_depth_median_missing_raises_actionably(tmp_path):
    path = tmp_path / "splats.zarr"
    _write_median_splats_zarr(path, with_median=False)

    with pytest.raises(ValueError, match="median_depth"):
        _splats_to_tsdf_inputs(path, splat_depth="median")


def test_a_store_without_depth_raises_instead_of_keyerror(tmp_path):
    # The median path already said what was wrong; the default path used to fall through to a
    # bare KeyError('depth') from zarr, which reads as a library bug rather than a bad store
    path = tmp_path / "splats.zarr"
    _write_splats_zarr(path)
    del zarr.open_group(path, mode="a")["depth"]

    with pytest.raises(ValueError, match="truncated"):
        _splats_to_tsdf_inputs(path)


def test_splat_depth_rejects_an_unknown_value(tmp_path):
    with pytest.raises(ValueError, match=r"mesh\.splat_depth"):
        _splats_to_tsdf_inputs(tmp_path / "splats.zarr", splat_depth="surf")


########
# Pre-fusion depth cuts: far-field and discontinuity
########


def _write_moving_camera_zarr(path, depths, n_views=None):
    """Splats store whose cameras travel 10 units along x, with caller-supplied depth."""
    depths = np.asarray(depths, dtype=np.float32)
    n_views, height, width = depths.shape
    store = zarr.open_group(str(path), mode="w")
    c2w = np.tile(np.eye(4, dtype=np.float32), (n_views, 1, 1))
    c2w[:, 0, 3] = np.linspace(0.0, 10.0, n_views)
    store.create_array("depth", data=depths)
    store.create_array("alpha", data=np.ones_like(depths))
    store.create_array("rgb", data=np.zeros((n_views, height, width, 3), dtype=np.uint8))
    store.create_array("c2w", data=c2w)
    store.create_array("K", data=np.tile(np.eye(3, dtype=np.float32), (n_views, 1, 1)))
    return path


def test_far_depth_cut_drops_depth_past_the_camera_trajectory(tmp_path):
    """Rendered depth well beyond where the cameras went is unconstrained by any view."""
    depths = np.full((4, 2, 2), 1.0, dtype=np.float32)
    depths[:, 0, 0] = 500.0  # blown-out background
    _write_moving_camera_zarr(tmp_path / "splats.zarr", depths)

    out, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", max_depth_frac=0.75)
    assert np.all(out[:, 0, 0] == 0.0)  # the far pixel went
    assert np.all(out[:, 0, 1] == 1.0)  # everything inside the trajectory stayed


def test_far_depth_cut_skipped_when_the_cameras_never_move(tmp_path):
    """A zero-extent rig has no trajectory to measure against — cut it and nothing survives."""
    _write_splats_zarr(tmp_path / "splats.zarr")  # every c2w is the identity

    out, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", max_depth_frac=0.75)
    assert out.max() == 1.0


def test_depth_gradient_cut_drops_both_sides_of_a_jump(tmp_path):
    """A silhouette is a jump between two good surfaces; TSDF welds a tendril across it."""
    depths = np.full((2, 3, 4), 1.0, dtype=np.float32)
    depths[:, :, 2:] = 2.0  # step edge between columns 1 and 2
    _write_moving_camera_zarr(tmp_path / "splats.zarr", depths)

    out, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", max_depth_grad=0.3)
    assert np.all(out[:, :, 1] == 0.0) and np.all(out[:, :, 2] == 0.0)  # both sides go
    assert np.all(out[:, :, 0] == 1.0) and np.all(out[:, :, 3] == 2.0)  # the surfaces stay


def test_depth_gradient_cut_ignores_already_dropped_pixels(tmp_path):
    """A zero neighbour is a hole, not a surface — it must not take live pixels with it."""
    depths = np.full((2, 3, 4), 1.0, dtype=np.float32)
    depths[:, :, 0] = 0.0  # no observation in the first column
    _write_moving_camera_zarr(tmp_path / "splats.zarr", depths)

    out, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr", max_depth_grad=0.3)
    assert np.all(out[:, :, 1:] == 1.0)


def test_depth_cuts_ship_off(tmp_path):
    """Neither cut runs unless a config asks for it — both delete real surface if mistuned."""
    depths = np.full((2, 3, 4), 1.0, dtype=np.float32)
    depths[:, :, 2:] = 500.0
    _write_moving_camera_zarr(tmp_path / "splats.zarr", depths)

    # No cut arguments at all: the 1 -> 500 step would trip either filter if one defaulted on
    out, _, _, _ = _splats_to_tsdf_inputs(tmp_path / "splats.zarr")
    assert out.min() == 1.0 and out.max() == 500.0
