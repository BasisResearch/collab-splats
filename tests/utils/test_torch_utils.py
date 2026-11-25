"""Tests for collab_splats.utils.torch_utils — device, batching, GC, to_numpy and vendored_path helpers."""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import collab_splats.utils.torch_utils as torch_utils
from collab_splats.utils.torch_utils import (
    batch_iterator,
    check_hf_access,
    full_fp32_matmul,
    hold_matmul_precision,
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
    with pytest.raises(ValueError, match="same length"):
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


def _hub(monkeypatch, cached=None, error=None):
    """Stub the Hub cache lookup and auth_check; returns the list of repos auth_check saw."""
    seen = []

    def auth(repo):
        seen.append(repo)

        if error is not None:
            raise error

    monkeypatch.setattr(torch_utils, "try_to_load_from_cache", lambda r, f: cached)
    monkeypatch.setattr(torch_utils, "auth_check", auth)
    return seen


def test_check_hf_access_cached_weights_skip_the_hub(monkeypatch):
    seen = _hub(monkeypatch, cached="/models/w.pt")
    check_hf_access("org/repo", "w.pt")
    assert seen == []


def test_check_hf_access_passes_with_access(monkeypatch):
    seen = _hub(monkeypatch)
    check_hf_access("org/repo", "w.pt")
    assert seen == ["org/repo"]


@pytest.mark.parametrize("error_cls", [GatedRepoError, RepositoryNotFoundError])
def test_check_hf_access_denied_names_repo_and_both_logins(monkeypatch, error_cls):
    _hub(monkeypatch, error=error_cls("denied", response=MagicMock()))

    with pytest.raises(RuntimeError) as info:
        check_hf_access("org/repo", "w.pt")

    message = str(info.value)
    assert "https://huggingface.co/org/repo" in message
    assert "hf auth login" in message
    assert "HF_TOKEN" in message


def test_check_hf_access_network_error_only_warns(monkeypatch, caplog):
    _hub(monkeypatch, error=ConnectionError("offline"))
    check_hf_access("org/repo", "w.pt")
    assert "could not check access to org/repo" in caplog.text


def test_dockerignore_keeps_env_files_out_of_the_image():
    lines = (Path(__file__).parents[2] / ".dockerignore").read_text().splitlines()
    assert ".env" in lines
    assert ".env.*" in lines
