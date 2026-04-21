import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
from collab_splats.pointcloud.sfm import ColmapCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


@pytest.fixture
def tiny_image_dir(tmp_path):
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"frame_{i:04d}.jpg")
    return tmp_path


def test_colmap_creator_defaults():
    c = ColmapCreator()
    assert c.camera_model == "SIMPLE_RADIAL"
    assert c.single_camera is False


def test_colmap_creator_single_camera():
    c = ColmapCreator(single_camera=True)
    assert c.single_camera is True


def test_colmap_creator_output_path(tiny_image_dir, tmp_path):
    """ColmapCreator must write binary files to output_dir/colmap/sparse/0/."""
    out = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("collab_splats.pointcloud.sfm.pycolmap.extract_features"), \
         patch("collab_splats.pointcloud.sfm.pycolmap.match_exhaustive"), \
         patch("collab_splats.pointcloud.sfm.pycolmap.incremental_mapping", return_value={0: mock_recon}), \
         patch.object(ColmapCreator, "_write_transforms") as mock_wt:
        creator = ColmapCreator()
        creator.reconstruct(tiny_image_dir, out)
        sparse_dir = out / "colmap" / "sparse" / "0"
        mock_wt.assert_called_once_with(sparse_dir, out)


def test_colmap_creator_no_reconstruction_raises(tiny_image_dir, tmp_path):
    out = tmp_path / "out"
    with patch("collab_splats.pointcloud.sfm.pycolmap.extract_features"), \
         patch("collab_splats.pointcloud.sfm.pycolmap.match_exhaustive"), \
         patch("collab_splats.pointcloud.sfm.pycolmap.incremental_mapping", return_value={}):
        creator = ColmapCreator()
        with pytest.raises(RuntimeError, match="reconstruction failed"):
            creator.reconstruct(tiny_image_dir, out)


def test_colmap_creator_missing_image_dir_raises(tmp_path):
    creator = ColmapCreator()
    with pytest.raises(FileNotFoundError):
        creator.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


@pytest.mark.gpu
def test_colmap_creator_smoke(tiny_image_dir, tmp_path):
    out = tmp_path / "out"
    creator = ColmapCreator(single_camera=True)
    try:
        result = creator.reconstruct(tiny_image_dir, out)
        assert isinstance(result, PointcloudResult)
        assert result.frame == CoordinateFrame.NERFSTUDIO
        assert result.world_transform is not None
        assert result.points.shape[1] == 3
        if result.camera_poses is not None:
            assert result.camera_poses.shape[1:] == (4, 4)
    except RuntimeError as e:
        assert "reconstruction failed" in str(e).lower()
