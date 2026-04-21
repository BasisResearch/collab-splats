import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from collab_splats.pointcloud.feedforward import MapAnythingCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


def test_mapanything_defaults():
    c = MapAnythingCreator()
    assert c.model_name == "facebook/map-anything"
    assert c.confidence_percentile == 35.0
    assert c.use_multiview_confidence is True
    assert c.minibatch_size == 1


def test_mapanything_is_feedforward_creator():
    assert issubclass(MapAnythingCreator, BaseFeedforwardCreator)


def test_mapanything_missing_image_dir_raises(tmp_path):
    c = MapAnythingCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_mapanything_run_inference_passes_inference_params(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    n, p = 2, 10

    mock_8tuple = (
        np.zeros((p, 3), dtype=np.float32),
        np.zeros((p, 3), dtype=np.uint8),
        np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32),
        np.eye(3)[None].repeat(n, axis=0).astype(np.float32),
        [image_dir / f"frame_{i:04d}.jpg" for i in range(n)],
        np.zeros((n, 6), dtype=np.float32),
        518, 336,
    )
    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("collab_splats.pointcloud._mapanything.run_mapanything",
               return_value=mock_8tuple) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=mock_recon), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=mock_recon), \
         patch("pycolmap.Reconstruction", return_value=mock_recon), \
         patch.object(MapAnythingCreator, "_write_transforms"):
        creator = MapAnythingCreator(confidence_percentile=50.0, minibatch_size=2)
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run.call_args
        assert kwargs["confidence_percentile"] == 50.0
        assert kwargs["minibatch_size"] == 2
        assert kwargs["use_multiview_confidence"] is True


@pytest.mark.gpu
def test_mapanything_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = MapAnythingCreator()
    result = c.reconstruct(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
    assert result.points.shape[1] == 3
    assert (tmp_path / "out" / "transforms.json").exists()
    assert (tmp_path / "out" / "colmap" / "sparse" / "0" / "cameras.bin").exists()
