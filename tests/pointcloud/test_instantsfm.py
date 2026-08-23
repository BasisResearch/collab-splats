import numpy as np
import pytest

from collab_splats.pointcloud import sfm


def _recon_with_keypoint(xy):
    """
    Minimal stand-in for pycolmap objects: one point3D (id 7) observed once in
    image 1 ("frame_000000.jpg") at original-res keypoint `xy`.
    """

    class _Elem:
        image_id, point2D_idx = 1, 0

    class _Track:
        elements = [_Elem()]

    class _P3D:
        track = _Track()

    class _Pt2D:
        pass

    _Pt2D.xy = np.asarray(xy, dtype=np.float64)

    class _Img:
        name = "frame_000000.jpg"
        points2D = [_Pt2D()]

    class _Recon:
        points3D = {7: _P3D()}
        images = {1: _Img()}

    return _Recon()


def test_vda_missing_clone_raises_actionable_import_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match="setup.sh"):
        sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path)


def test_vda_skips_when_depths_exist(tmp_path):
    out = tmp_path / "depth_vda"
    out.mkdir()
    np.savez_compressed(out / "depths.npz", depths=np.ones((2, 4, 4), dtype=np.float32))
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    # Existing depths.npz -> returned untouched, no VDA import attempted
    path = sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path)
    assert path == out / "depths.npz"


def test_pixel_indices_from_reconstruction_scales_to_depth_res():
    # Original 800x600 -> depth 80x60 = scale 0.1; keypoint (200, 100) -> (row 10, col 20)
    idx = sfm._pixel_indices_from_reconstruction(
        _recon_with_keypoint((200.0, 100.0)),
        point3d_ids=[7],
        name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1,
        scale_y=0.1,
        depth_hw=(60, 80),
    )
    assert idx.shape == (1, 3)
    assert idx.dtype == np.int32
    assert idx.tolist() == [[0, 10, 20]]  # [frame_row, row=y*0.1, col=x*0.1]


def test_pixel_indices_clamped_to_grid():
    # Edge keypoint -> scaled index must stay in-grid
    idx = sfm._pixel_indices_from_reconstruction(
        _recon_with_keypoint((799.9, 599.9)),
        point3d_ids=[7],
        name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1,
        scale_y=0.1,
        depth_hw=(60, 80),
    )
    assert 0 <= idx[0, 1] < 60 and 0 <= idx[0, 2] < 80


def test_creator_config_copy_prevents_module_dict_leak():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.controllers.config import RUNTIME_OPTIONS

    before = dict(RUNTIME_OPTIONS)
    cfg = sfm.InstantSfMCreator(features="colmap")._build_config()
    cfg.RUNTIME_OPTIONS["use_depths"] = True

    # Module-level dict must be untouched — Config aliases it; creator must copy
    assert RUNTIME_OPTIONS == before
