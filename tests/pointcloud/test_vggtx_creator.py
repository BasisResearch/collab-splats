import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from collab_splats.pointcloud.feedforward import VGGTXCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


def test_vggtx_defaults():
    c = VGGTXCreator()
    assert c.use_global_alignment is False


def test_vggtx_is_feedforward_creator():
    assert issubclass(VGGTXCreator, BaseFeedforwardCreator)


def test_vggtx_missing_image_dir_raises(tmp_path):
    c = VGGTXCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_vggtx_calls_run_vggt_with_correct_colmap_dir(tmp_path):
    """VGGTXCreator must call run_vggt with colmap_dir=output_dir/colmap."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    # Use sys.modules to avoid import issues with stage.vggt_utils
    import sys
    mock_stage = MagicMock()
    mock_utils = MagicMock()
    mock_utils.run_vggt = MagicMock()

    sys.modules['stage'] = mock_stage
    sys.modules['stage.vggt_utils'] = mock_utils

    try:
        with patch("pycolmap.Reconstruction", return_value=mock_recon), \
             patch.object(VGGTXCreator, "_write_transforms"):
            creator = VGGTXCreator()
            creator._run_inference(image_dir, output_dir)

            mock_utils.run_vggt.assert_called_once()
            call_args, call_kwargs = mock_utils.run_vggt.call_args
            expected_colmap_dir = str(output_dir / "colmap")
            assert call_kwargs.get("colmap_dir") == expected_colmap_dir
            assert call_kwargs.get("image_dir") == str(image_dir)
            assert call_kwargs.get("use_global_alignment") is False
    finally:
        sys.modules.pop('stage', None)
        sys.modules.pop('stage.vggt_utils', None)


def test_vggtx_global_alignment_passed_through(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    import sys
    mock_stage = MagicMock()
    mock_utils = MagicMock()
    mock_utils.run_vggt = MagicMock()

    sys.modules['stage'] = mock_stage
    sys.modules['stage.vggt_utils'] = mock_utils

    try:
        with patch("pycolmap.Reconstruction", return_value=mock_recon), \
             patch.object(VGGTXCreator, "_write_transforms"):
            creator = VGGTXCreator(use_global_alignment=True)
            creator._run_inference(image_dir, output_dir)

            _, call_kwargs = mock_utils.run_vggt.call_args
            assert call_kwargs.get("use_global_alignment") is True
    finally:
        sys.modules.pop('stage', None)
        sys.modules.pop('stage.vggt_utils', None)


@pytest.mark.gpu
def test_vggtx_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = VGGTXCreator()
    result = c.reconstruct(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
    assert (tmp_path / "out" / "transforms.json").exists()
    assert (tmp_path / "out" / "colmap" / "sparse" / "0" / "cameras.bin").exists()
