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
    """Test that _run_inference passes parameters to run_mapanything_inference."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    sparse_path = output_dir / "colmap" / "sparse" / "0"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    # Use a mock for sys.modules to avoid stage import issues
    import sys
    mock_stage = MagicMock()
    mock_utils = MagicMock()
    mock_utils.load_mapanything_model.return_value = MagicMock()
    mock_utils.load_and_preprocess_images.return_value = (
        [{"img": np.zeros((3, 518, 336))}],
        [image_dir / "img.jpg"]
    )
    mock_utils.run_mapanything_inference.return_value = [{}]
    mock_utils.export_to_colmap.return_value = sparse_path
    mock_utils.rescale_to_original_dimensions.return_value = sparse_path

    sys.modules['stage'] = mock_stage
    sys.modules['stage.mapanything_utils'] = mock_utils

    try:
        with patch("pycolmap.Reconstruction", return_value=mock_recon), \
             patch.object(MapAnythingCreator, "_write_transforms"):
            creator = MapAnythingCreator(confidence_percentile=50.0, minibatch_size=2)
            creator._run_inference(image_dir, output_dir)

            # Verify inference was called with correct parameters
            call_kwargs = mock_utils.run_mapanything_inference.call_args[1]
            assert call_kwargs["confidence_percentile"] == 50.0
            assert call_kwargs["minibatch_size"] == 2
            assert call_kwargs["use_multiview_confidence"] is True
            assert call_kwargs["apply_mask"] is True
            assert call_kwargs["mask_edges"] is True
            assert call_kwargs["apply_confidence_mask"] is True
    finally:
        sys.modules.pop('stage', None)
        sys.modules.pop('stage.mapanything_utils', None)


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
