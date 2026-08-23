# tests/test_feedforward_logging.py
import numpy as np
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from collab_splats.pointcloud.feedforward import BaseFeedforwardCreator, FeedforwardResult


@dataclass
class _MockCreator(BaseFeedforwardCreator):
    def _load_model(self, device: str) -> Any:
        return object()

    def _decode_source(self, source: Any):
        # Bypass real decode (setup_inference now decodes the source first); the
        # logging tests only need a 2-frame batch, not on-disk images.
        return ([None, None], [0, 1])

    def _preprocess(self, frames: Any, frame_idxs: list[int]):
        return (
            [],
            [Path("a.jpg"), Path("b.jpg")],
            np.zeros((2, 6), dtype=np.float32),
        )

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        return {}

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        return FeedforwardResult(
            points=np.zeros((100, 3), dtype=np.float32),
            colors=np.zeros((100, 3), dtype=np.uint8),
            extrinsics=np.tile(np.eye(3, 4), (2, 1, 1)),
            intrinsics=np.tile(np.eye(3), (2, 1, 1)),
            image_paths=[Path("a.jpg"), Path("b.jpg")],
            original_coords=np.zeros((2, 6), dtype=np.float32),
            model_width=100,
            model_height=100,
        )

    def extract_intermediate_features(self, frames, layer_index=-1, **kwargs):
        return {}

    def _reproject(self, raw_outputs, extrinsics_3x4, intrinsics):
        import numpy as np
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)


def test_load_model_logs_device(capsys):
    creator = _MockCreator()
    creator.load_model(device="cpu")
    out = capsys.readouterr().out
    assert "Loading model (cpu)" in out
    assert "done in" in out


def test_setup_inference_logs_image_count(capsys, tmp_path):
    creator = _MockCreator()
    creator.load_model(device="cpu")
    capsys.readouterr()
    creator.setup_inference(tmp_path)
    out = capsys.readouterr().out
    assert "2 images" in out
    assert "done in" in out


def test_run_inference_logs_timing(capsys, tmp_path):
    creator = _MockCreator()
    creator.load_model(device="cpu")
    creator.setup_inference(tmp_path)
    capsys.readouterr()
    creator.run_inference()
    out = capsys.readouterr().out
    assert "Running inference" in out
    assert "done in" in out


def test_postprocess_logs_point_count(capsys, tmp_path):
    creator = _MockCreator()
    creator.load_model(device="cpu")
    creator.setup_inference(tmp_path)
    creator.run_inference()
    capsys.readouterr()
    creator.postprocess()
    out = capsys.readouterr().out
    assert "100" in out
    assert "pts" in out
    assert "done in" in out


def test_build_colmap_logs_timing(capsys, tmp_path):
    creator = _MockCreator()
    creator.load_model(device="cpu")
    creator.setup_inference(tmp_path)
    creator.run_inference()
    creator.postprocess()
    capsys.readouterr()
    mock_recon = MagicMock()
    # points3D has to be a real mapping: a bare MagicMock is truthy but iterates empty, so
    # PointcloudResult.points skips its empty-guard and yields a (0,) array that the
    # sparse_pc.ply writer build_colmap now calls rejects as mis-shaped.
    mock_recon.points3D = {}
    with patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction", return_value=mock_recon), \
         patch("collab_splats.pointcloud.feedforward.base._rescale_reconstruction_to_original_dimensions", return_value=mock_recon):
        creator.build_colmap(tmp_path)
    out = capsys.readouterr().out
    assert "COLMAP" in out
    assert "done in" in out


def test_mapanything_forward_logs_minibatch_info(capsys):
    pytest.importorskip("mapanything")
    import torch
    from collab_splats.pointcloud.feedforward import MapAnythingCreator

    mock_model = MagicMock()
    mock_model.infer.return_value = []
    # Make next(model.parameters()).device return a real CPU device
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_model.forward.return_value = []

    creator = MapAnythingCreator(minibatch_size=3)
    creator.model = mock_model
    creator.original_coords = np.zeros((6, 6), dtype=np.float32)
    creator.image_paths = [Path(f"{i}.jpg") for i in range(6)]

    # _forward picks the full-sequence branch only when `views is self.views`
    # (discriminator in mapanything.py:241-245). Set both so the call skips the
    # LC-window key validation and hits the "N images, minibatch_size=..." log.
    views = [{"img": torch.zeros(1, 3, 224, 224)} for _ in range(6)]
    creator._processed_views = views
    creator.views = views
    creator._forward(mock_model, views)

    out = capsys.readouterr().out
    assert "6 images" in out
    assert "minibatch_size=3" in out
