"""Frame handoff shared by the array-capable feedforward backends."""

import logging

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTOmegaCreator,
    VGGTXCreator,
)
from tests.pointcloud.conftest import _frame_files
from tests.pointcloud.feedforward.conftest import _assert_views_equal


def _assert_tensors_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    """Two image batches are bit-equal."""
    assert torch.equal(a, b)


_backends = pytest.mark.parametrize(
    "creator_cls, assert_views_equal",
    [
        (VGGTXCreator, _assert_tensors_equal),
        (VGGTOmegaCreator, _assert_tensors_equal),
        (MapAnythingCreator, _assert_views_equal),
    ],
    ids=["vggtx", "vggt_omega", "mapanything"],
)


@_backends
def test_preprocess_falls_back_to_files_when_a_frame_is_missing(
    tmp_path, caplog, creator_cls, assert_views_equal
):
    """
    A partial handoff is ignored with one warning: every frame is read from disk.
    """
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (48, 96, 3), dtype=np.uint8) for _ in range(3)]
    paths = _frame_files(frames, tmp_path)

    creator = creator_cls()
    creator.frames = {paths[0].name: np.zeros_like(frames[0])}

    with caplog.at_level(
        logging.WARNING, logger="collab_splats.pointcloud.feedforward.base"
    ):
        views, _ = creator._preprocess(paths)

    assert [r.getMessage() for r in caplog.records] == [
        "frames handed off but 2 of 3 paths missing; reading files"
    ]
    assert_views_equal(views, creator_cls()._preprocess(paths)[0])


@_backends
def test_preprocess_takes_the_array_path(tmp_path, creator_cls, assert_views_equal):
    """
    Handed-off arrays win over differing PNGs on disk: no silent fallback to the files.
    """
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 256, (48, 96, 3), dtype=np.uint8) for _ in range(3)]
    real_paths = _frame_files(frames, tmp_path / "real")
    zero_paths = _frame_files([np.zeros_like(f) for f in frames], tmp_path / "zeros")

    creator = creator_cls()
    creator.frames = {p.name: f for p, f in zip(zero_paths, frames, strict=True)}
    views, _ = creator._preprocess(zero_paths)

    assert_views_equal(views, creator_cls()._preprocess(real_paths)[0])

    with pytest.raises(AssertionError):
        assert_views_equal(views, creator_cls()._preprocess(zero_paths)[0])
