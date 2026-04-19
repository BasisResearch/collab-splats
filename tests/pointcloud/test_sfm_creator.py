import numpy as np
import pytest
from PIL import Image
from collab_splats.pointcloud.sfm import NerfstudioSfmCreator
from collab_splats.pointcloud.base import PointcloudResult


@pytest.fixture
def tiny_image_dir(tmp_path):
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"frame_{i:04d}.jpg")
    return tmp_path


def test_sfm_defaults():
    c = NerfstudioSfmCreator()
    assert c.use_hloc is True
    assert c.feature_type == "superpoint_aachen"
    assert c.matcher_type == "superglue"
    assert c.num_matched == 50
    assert c.camera_model == "SIMPLE_RADIAL"
    assert c.single_camera is False


def test_sfm_pycolmap_mode():
    c = NerfstudioSfmCreator(use_hloc=False, single_camera=True)
    assert c.use_hloc is False
    assert c.single_camera is True


def test_sfm_create_result_or_graceful_error(tiny_image_dir, tmp_path):
    """Synthetic images likely fail SfM — accept either valid result or RuntimeError."""
    creator = NerfstudioSfmCreator(use_hloc=False, single_camera=True)
    try:
        result = creator.create(tiny_image_dir, tmp_path / "out")
        assert isinstance(result, PointcloudResult)
        assert result.points.shape[1] == 3
        assert result.colors.dtype == np.uint8
        if result.camera_poses is not None:
            assert result.camera_poses.shape[1:] == (4, 4)
    except RuntimeError as e:
        assert "reconstruction" in str(e).lower() or "colmap" in str(e).lower()
