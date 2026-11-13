"""
Log lines of the feedforward base and MapAnything: model load, preprocess, inference, point count.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    MapAnythingCreator,
)
from tests.pointcloud.conftest import _frame_files, _frames


@dataclass
class _MockCreator(BaseFeedforwardCreator):
    def _load_model(self, device: str) -> Any:
        return object()

    def _preprocess(self, paths: list[Path]):
        return [], np.zeros((2, 6), dtype=np.float32)

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        return {}

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> PointcloudResult:
        return PointcloudResult(
            points=np.zeros((100, 3), dtype=np.float32),
            colors=np.zeros((100, 3), dtype=np.uint8),
            extrinsics=np.tile(np.eye(3, 4), (2, 1, 1)),
            intrinsics=None,
            model_intrinsics=np.tile(np.eye(3), (2, 1, 1)),
            image_paths=[Path("a.jpg"), Path("b.jpg")],
            original_coords=np.tile(
                np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (2, 1)
            ),  # full-frame box
            model_width=100,
            model_height=100,
        )


BASE_LOGGER = "collab_splats.pointcloud.feedforward.base"
CREATOR_LOGGER = "collab_splats.pointcloud.base"
PATHS = [Path("a.png"), Path("b.png")]
MAPANYTHING_LOGGER = "collab_splats.pointcloud.feedforward.mapanything"


def test_load_model_logs_device(caplog):
    creator = _MockCreator()
    with caplog.at_level(logging.INFO, logger=BASE_LOGGER):
        creator.load_model()
    assert "Loading model (" in caplog.text
    assert "done in" in caplog.text


def test_setup_inference_logs_image_count(caplog):
    creator = _MockCreator()
    creator.load_model()
    with caplog.at_level(logging.INFO, logger=BASE_LOGGER):
        creator.setup_inference(PATHS)
    assert "Preprocessed 2 images in" in caplog.text


def test_run_inference_logs_timing(caplog):
    creator = _MockCreator()
    creator.load_model()
    creator.setup_inference(PATHS)
    with caplog.at_level(logging.INFO, logger=BASE_LOGGER):
        creator.run_inference()
    assert "Running inference" in caplog.text
    assert "done in" in caplog.text


def test_create_pointcloud_logs_point_count(caplog, tmp_path):
    creator = _MockCreator(clean=False)
    _frame_files(_frames([(8, 8), (8, 8)]), tmp_path / "images")
    with caplog.at_level(logging.INFO, logger=CREATOR_LOGGER):
        creator.create_pointcloud(tmp_path / "images", tmp_path / "out")
    assert "pointcloud: 100 pts" in caplog.text


def test_mapanything_forward_logs_minibatch_info(caplog):
    mock_model = MagicMock()
    mock_model.infer.return_value = []
    # Make next(model.parameters()).device return a real CPU device
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_model.forward.return_value = []

    creator = MapAnythingCreator(minibatch_size=3)
    creator.model = mock_model
    creator.original_coords = np.tile(
        np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (6, 1)
    )  # full-frame box
    creator.image_paths = [Path(f"{i}.jpg") for i in range(6)]

    # _forward takes the full-sequence branch only when `views is self.views`
    # - set both so the call skips the LC-window preprocess
    # - hits the "N images, minibatch_size=..." log
    views = [{"img": torch.zeros(1, 3, 224, 224)} for _ in range(6)]
    creator._processed_views = views
    creator.views = views
    with (
        caplog.at_level(logging.DEBUG, logger=MAPANYTHING_LOGGER),
        patch.object(MapAnythingCreator, "_stack_predictions"),
    ):
        creator._forward(mock_model, views)

    assert "6 images" in caplog.text
    assert "minibatch_size=3" in caplog.text
