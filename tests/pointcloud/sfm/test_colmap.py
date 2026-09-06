import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from collab_splats.pointcloud.sfm import ColmapCreator
from collab_splats.pointcloud.base import PointcloudResult


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

    with patch("collab_splats.pointcloud.sfm.colmap.pycolmap.extract_features"), \
         patch("collab_splats.pointcloud.sfm.colmap.pycolmap.match_exhaustive"), \
         patch("collab_splats.pointcloud.sfm.colmap.pycolmap.incremental_mapping", return_value={0: mock_recon}):
        creator = ColmapCreator()
        creator.reconstruct(tiny_image_dir, out)
        sparse_dir = out / "colmap" / "sparse" / "0"
        mock_recon.write_binary.assert_called_once_with(str(sparse_dir))


def test_colmap_creator_no_reconstruction_raises(tiny_image_dir, tmp_path):
    out = tmp_path / "out"
    with patch("collab_splats.pointcloud.sfm.colmap.pycolmap.extract_features"), \
         patch("collab_splats.pointcloud.sfm.colmap.pycolmap.match_exhaustive"), \
         patch("collab_splats.pointcloud.sfm.colmap.pycolmap.incremental_mapping", return_value={}):
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
        assert result.points.shape[1] == 3
        assert result.extrinsics.shape[1:] == (4, 4)
    except RuntimeError as e:
        assert "reconstruction failed" in str(e).lower()
