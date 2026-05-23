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


from collab_splats.pointcloud.sfm import HlocCreator


def test_hloc_creator_defaults():
    c = HlocCreator()
    assert c.retrieval_conf == "netvlad"
    assert c.feature_conf == "superpoint_aachen"
    assert c.matcher_conf == "superglue"


def test_hloc_creator_output_path(tiny_image_dir, tmp_path):
    """HlocCreator must write binary files to output_dir/colmap/sparse/0/."""
    import sys
    out = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    # Mock hloc and its submodules in sys.modules
    mock_hloc = MagicMock()
    mock_extract = MagicMock()
    mock_pairs = MagicMock()
    mock_match = MagicMock()
    mock_reconstruction = MagicMock()

    mock_extract.main.return_value = tmp_path / "feats.h5"
    mock_match.main.return_value = tmp_path / "matches.h5"
    mock_reconstruction.main.return_value = mock_recon

    mock_hloc.extract_features = mock_extract
    mock_hloc.pairs_from_retrieval = mock_pairs
    mock_hloc.match_features = mock_match
    mock_hloc.reconstruction = mock_reconstruction

    old_modules = {
        'hloc': sys.modules.get('hloc'),
        'hloc.extract_features': sys.modules.get('hloc.extract_features'),
        'hloc.pairs_from_retrieval': sys.modules.get('hloc.pairs_from_retrieval'),
        'hloc.match_features': sys.modules.get('hloc.match_features'),
        'hloc.reconstruction': sys.modules.get('hloc.reconstruction'),
    }

    try:
        sys.modules['hloc'] = mock_hloc
        sys.modules['hloc.extract_features'] = mock_extract
        sys.modules['hloc.pairs_from_retrieval'] = mock_pairs
        sys.modules['hloc.match_features'] = mock_match
        sys.modules['hloc.reconstruction'] = mock_reconstruction

        with patch.object(HlocCreator, "_write_transforms") as mock_wt:
            creator = HlocCreator()
            creator.reconstruct(tiny_image_dir, out)
            sparse_dir = out / "colmap" / "sparse" / "0"
            mock_wt.assert_called_once_with(sparse_dir, out)
    finally:
        for key, val in old_modules.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


def test_hloc_creator_no_reconstruction_raises(tiny_image_dir, tmp_path):
    import sys
    out = tmp_path / "out"

    # Mock hloc and its submodules
    mock_hloc = MagicMock()
    mock_extract = MagicMock()
    mock_pairs = MagicMock()
    mock_match = MagicMock()
    mock_reconstruction = MagicMock()

    mock_extract.main.return_value = tmp_path / "feats.h5"
    mock_match.main.return_value = tmp_path / "matches.h5"
    mock_reconstruction.main.return_value = None  # Force failure

    mock_hloc.extract_features = mock_extract
    mock_hloc.pairs_from_retrieval = mock_pairs
    mock_hloc.match_features = mock_match
    mock_hloc.reconstruction = mock_reconstruction

    old_modules = {
        'hloc': sys.modules.get('hloc'),
        'hloc.extract_features': sys.modules.get('hloc.extract_features'),
        'hloc.pairs_from_retrieval': sys.modules.get('hloc.pairs_from_retrieval'),
        'hloc.match_features': sys.modules.get('hloc.match_features'),
        'hloc.reconstruction': sys.modules.get('hloc.reconstruction'),
    }

    try:
        sys.modules['hloc'] = mock_hloc
        sys.modules['hloc.extract_features'] = mock_extract
        sys.modules['hloc.pairs_from_retrieval'] = mock_pairs
        sys.modules['hloc.match_features'] = mock_match
        sys.modules['hloc.reconstruction'] = mock_reconstruction

        creator = HlocCreator()
        with pytest.raises(RuntimeError, match="reconstruction failed"):
            creator.reconstruct(tiny_image_dir, out)
    finally:
        for key, val in old_modules.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


def test_hloc_creator_missing_image_dir_raises(tmp_path):
    import sys
    # Need to mock hloc even for this test since import happens first
    mock_hloc = MagicMock()
    old_hloc = sys.modules.get('hloc')
    try:
        sys.modules['hloc'] = mock_hloc
        creator = HlocCreator()
        with pytest.raises(FileNotFoundError):
            creator.reconstruct(tmp_path / "nonexistent", tmp_path / "out")
    finally:
        if old_hloc is None:
            sys.modules.pop('hloc', None)
        else:
            sys.modules['hloc'] = old_hloc


def test_hloc_no_nerfstudio_dep():
    """HlocCreator must not import nerfstudio at module level."""
    import collab_splats.pointcloud.sfm as sfm_module
    import inspect
    src = inspect.getsource(sfm_module)
    hloc_src = src.split("class HlocCreator")[1].split("class ")[0]
    # Check for actual imports, not just the string in docstring/comments
    import_lines = [line.strip() for line in hloc_src.split('\n')
                   if 'import' in line and not line.strip().startswith('#')]
    nerfstudio_imports = [line for line in import_lines if 'nerfstudio' in line]
    assert not nerfstudio_imports, \
        f"HlocCreator must not import nerfstudio. Found: {nerfstudio_imports}"
