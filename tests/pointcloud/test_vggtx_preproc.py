import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import torch


def _make_fake_images(n=2, size=518):
    return torch.zeros(n, 3, size, size), torch.zeros(n, 6)


def _make_image_dir(tmp_path, n=2):
    for i in range(n):
        (tmp_path / f"frame_{i:04d}.jpg").touch()
    return tmp_path


def test_vggtx_creator_default_preproc_is_ratio():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator()
    assert c.image_preproc == "ratio"


def test_vggtx_creator_accepts_square_preproc():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator(image_preproc="square")
    assert c.image_preproc == "square"


def test_vggtx_creator_rejects_invalid_preproc():
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    with pytest.raises(ValueError, match="image_preproc"):
        VGGTXCreator(image_preproc="invalid")


def test_preprocess_ratio_calls_ratio_fn(tmp_path):
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path)
    c = VGGTXCreator(image_preproc="ratio")
    fake = _make_fake_images()
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_ratio", return_value=fake) as m, \
         patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_square") as ms:
        c._preprocess(img_dir)
        assert m.called
        assert not ms.called


def test_preprocess_square_calls_square_fn(tmp_path):
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path)
    c = VGGTXCreator(image_preproc="square")
    fake = _make_fake_images()
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_square", return_value=fake) as m, \
         patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images_ratio") as mr:
        c._preprocess(img_dir)
        assert m.called
        assert not mr.called
