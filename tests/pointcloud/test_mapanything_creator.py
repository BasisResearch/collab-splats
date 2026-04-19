import numpy as np
import pytest
from unittest.mock import MagicMock
from collab_splats.pointcloud.feedforward import MapAnythingCreator
from collab_splats.pointcloud.base import PointcloudResult


def test_mapanything_defaults():
    c = MapAnythingCreator()
    assert c.conf_threshold == 1.5
    assert c.subsample_factor == 1
    assert c.extractor is None
    assert c.model_name == "mapanything"


def test_mapanything_accepts_extractor():
    semantics = pytest.importorskip("collab_splats.semantics.extractors")
    mock = MagicMock(spec=semantics.BaseExtractor)
    c = MapAnythingCreator(extractor=mock)
    assert c.extractor is mock


def test_mapanything_create_missing_dir_raises(tmp_path):
    c = MapAnythingCreator()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        c.create(tmp_path / "nonexistent", tmp_path / "out")


# Run only on GPU: pytest -m gpu
@pytest.mark.gpu
def test_mapanything_create_smoke(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = MapAnythingCreator(subsample_factor=4)
    result = c.create(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.points.shape[1] == 3
    assert result.confidence is not None
    assert result.camera_poses is not None
    assert result.camera_poses.shape[1:] == (4, 4)
    assert result.camera_intrinsics is not None
    assert result.camera_intrinsics.shape[1:] == (3, 3)
