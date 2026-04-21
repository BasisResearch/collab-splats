import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from collab_splats.pointcloud.feedforward import VGGTXCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


def test_vggtx_defaults():
    c = VGGTXCreator()
    assert c.use_global_alignment is False
    assert c.model_name == "facebook/vggt"


def test_vggtx_is_feedforward_creator():
    assert issubclass(VGGTXCreator, BaseFeedforwardCreator)


def test_vggtx_missing_image_dir_raises(tmp_path):
    c = VGGTXCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def _run_vggt_mock_8tuple(image_dir, n=2, p=10):
    return (
        np.zeros((p, 3), dtype=np.float32),
        np.zeros((p, 3), dtype=np.uint8),
        np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32),
        np.eye(3)[None].repeat(n, axis=0).astype(np.float32),
        [image_dir / f"frame_{i:04d}.jpg" for i in range(n)],
        np.zeros((n, 6), dtype=np.float32),
        518, 518,
    )


def _mock_recon():
    m = MagicMock()
    m.images = {}
    m.cameras = {}
    m.points3D = {}
    return m


def test_vggtx_calls_run_vggt_with_correct_colmap_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("collab_splats.pointcloud._vggt.run_vggt",
               return_value=_run_vggt_mock_8tuple(image_dir)) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=_mock_recon()), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=_mock_recon()), \
         patch("pycolmap.Reconstruction", return_value=_mock_recon()), \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator()
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run.call_args
        assert kwargs["colmap_dir"] == output_dir / "colmap"
        assert kwargs["use_global_alignment"] is False


def test_vggtx_global_alignment_passed_through(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("collab_splats.pointcloud._vggt.run_vggt",
               return_value=_run_vggt_mock_8tuple(image_dir)) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=_mock_recon()), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=_mock_recon()), \
         patch("pycolmap.Reconstruction", return_value=_mock_recon()), \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator(use_global_alignment=True)
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run.call_args
        assert kwargs["use_global_alignment"] is True


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
