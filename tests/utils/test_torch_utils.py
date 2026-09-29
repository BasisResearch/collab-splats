"""Tests for collab_splats.utils.torch_utils — device, batching, GC, to_numpy and vendored_path helpers."""

import sys

import numpy as np
import pytest
import torch

from collab_splats.utils.torch_utils import (
    batch_iterator,
    infer_batch_size,
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
