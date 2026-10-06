"""Tests for collab_splats.utils.torch_utils — device, batching, GC, to_numpy and vendored_path helpers."""

import logging
import sys
import threading

import numpy as np
import pytest
import torch
import zarr

from collab_splats.utils.torch_utils import (
    batch_iterator,
    full_fp32_matmul,
    hold_matmul_precision,
    infer_batch_size,
    load_features,
    pytorch_gc,
    to_numpy,
    vendored_path,
)


def test_pytorch_gc_no_error():
    pytorch_gc()


def test_infer_batch_size_cpu():
    if not torch.cuda.is_available():
        assert infer_batch_size(3.0) == 1


def test_infer_batch_size_negative_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(-1.0)


def test_infer_batch_size_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(0.0)


def test_batch_iterator_basic():
    items = list(range(10))
    batches = list(batch_iterator(3, items))
    assert len(batches) == 4
    assert batches[0] == [[0, 1, 2]]
    assert batches[-1] == [[9]]


def test_batch_iterator_multiple_args():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    batches = list(batch_iterator(2, a, b))
    assert len(batches) == 2
    assert batches[0] == [[1, 2], [5, 6]]


def test_batch_iterator_mismatched_raises():
    with pytest.raises(AssertionError):
        list(batch_iterator(2, [1, 2], [3]))


def test_to_numpy_casts_bf16_tensor_to_float32():
    out = to_numpy(torch.ones(2, 3, dtype=torch.bfloat16))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)


def test_to_numpy_detaches_a_grad_tensor():
    out = to_numpy(torch.ones(2, requires_grad=True))
    np.testing.assert_array_equal(out, np.ones(2, dtype=np.float32))


def test_to_numpy_passes_ndarray_through():
    arr = np.arange(4, dtype=np.int16)
    assert to_numpy(arr) is arr


def test_vendored_path_inserts_then_removes(tmp_path):
    with vendored_path(tmp_path, "hint"):
        assert sys.path[0] == str(tmp_path)
    assert str(tmp_path) not in sys.path


def test_vendored_path_keeps_a_preexisting_entry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    with vendored_path(tmp_path, "hint"):
        pass
    assert str(tmp_path) in sys.path


def test_vendored_path_missing_root_names_the_fix(tmp_path):
    with pytest.raises(ImportError, match="run setup.sh"):
        with vendored_path(tmp_path / "absent", "run setup.sh"):
            pass


########################################################
########## load_features ###############################
########################################################


def _as_zarr(tmp_path, array: np.ndarray) -> zarr.Array:
    store = zarr.create_array(
        str(tmp_path / "f.zarr"), shape=array.shape, chunks=(1, *array.shape[1:]), dtype=array.dtype
    )
    store[:] = array
    return store


@pytest.mark.parametrize("shape", [(5, 3), (5, 3, 4, 2), (5, 3, 7)])
def test_load_features_maps_item_and_trailing_index_to_rows(shape):
    array = np.random.default_rng(0).random(shape, dtype=np.float32)
    out = load_features(array, "cpu", read_gb=1e-9)

    # Row k * S + s is item k at flattened trailing index s
    per_item = int(np.prod(shape[2:]))
    expected = array.reshape(shape[0], shape[1], per_item).transpose(0, 2, 1).reshape(-1, shape[1])
    assert out.shape == (shape[0] * per_item, shape[1])
    np.testing.assert_array_equal(out.numpy(), expected)


def test_load_features_torch_numpy_and_zarr_agree(tmp_path):
    array = np.random.default_rng(1).random((6, 4, 3, 2), dtype=np.float32)

    from_numpy = load_features(array, "cpu")
    from_torch = load_features(torch.from_numpy(array), "cpu")
    from_zarr = load_features(_as_zarr(tmp_path, array), "cpu", read_gb=1e-9)

    assert torch.equal(from_numpy, from_torch)
    assert torch.equal(from_numpy, from_zarr)


def test_load_features_casts_fp16_to_fp32(tmp_path):
    array = np.random.default_rng(2).random((3, 4, 2, 2)).astype(np.float16)
    out = load_features(_as_zarr(tmp_path, array), "cpu")

    assert out.dtype == torch.float32
    np.testing.assert_array_equal(out[:4].numpy(), array[0].reshape(4, 4).T.astype(np.float32))


def test_load_features_over_budget_keeps_evenly_spaced_items(caplog):
    # Item k is constant k, so each output row names the item it came from
    array = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 4, 2)).copy()
    item_gb = 2 * 4 * 4 / 2**30

    with caplog.at_level(logging.INFO, logger="collab_splats.utils.torch_utils"):
        out = load_features(array, "cpu", max_gb=4 * item_gb, read_gb=1e-9)

    # Budget of 4 items over 10 -> every 3rd item: 0, 3, 6, 9
    assert out.shape == (4 * 2, 4)
    assert out.shape[0] * out.shape[1] * 4 / 2**30 <= 4 * item_gb
    assert out[::2, 0].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert "kept 4 of 10 items" in caplog.text


def test_load_features_over_budget_zarr_strided_reads_match_numpy(tmp_path):
    # Item k is constant k, so each output row names the item it came from
    array = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 4, 2)).copy()
    item_gb = 2 * 4 * 4 / 2**30

    # Budget of 4 items over 10 -> stride 3; 2 items per read -> reads [0, 3] then [6, 9]
    from_numpy = load_features(array, "cpu", max_gb=4 * item_gb, read_gb=2 * item_gb)
    from_zarr = load_features(_as_zarr(tmp_path, array), "cpu", max_gb=4 * item_gb, read_gb=2 * item_gb)

    assert from_zarr[::2, 0].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert torch.equal(from_zarr, from_numpy)


def test_load_features_within_budget_does_not_log(caplog):
    with caplog.at_level(logging.INFO, logger="collab_splats.utils.torch_utils"):
        load_features(np.ones((3, 2), dtype=np.float32), "cpu")

    assert "kept" not in caplog.text


def test_load_features_empty_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        load_features(np.zeros((0, 4, 2, 2), dtype=np.float16), "cpu")


def test_load_features_item_over_budget_raises():
    with pytest.raises(ValueError, match="over max_gb"):
        load_features(np.ones((2, 4, 8), dtype=np.float32), "cpu", max_gb=1e-12)


@pytest.mark.usefixtures("matmul_precision")
def test_full_fp32_matmul_sets_highest_and_restores_on_error():
    torch.set_float32_matmul_precision("high")

    # Inside the block TF32 is off; an exception still restores "high"
    with pytest.raises(RuntimeError):
        with full_fp32_matmul():
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("boom")

    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.usefixtures("matmul_precision")
def test_full_fp32_matmul_waits_for_a_held_precision():
    torch.set_float32_matmul_precision("high")
    held, release, seen = threading.Event(), threading.Event(), []

    def forward():
        with hold_matmul_precision():
            held.set()
            release.wait(5)
            seen.append(torch.get_float32_matmul_precision())

    def guarded():
        with full_fp32_matmul():
            seen.append("guarded")

    try:
        # A guarded call started while a forward holds the precision blocks until it exits
        t1 = threading.Thread(target=forward)
        t1.start()
        assert held.wait(5)
        t2 = threading.Thread(target=guarded)
        t2.start()
        t2.join(0.2)
        assert t2.is_alive()

        release.set()
        t1.join(5)
        t2.join(5)
        assert seen == ["high", "guarded"]
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        release.set()
