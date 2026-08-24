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


_NAMES = ["frame_000000.jpg", "frame_000001.jpg"]


def test_vda_missing_clone_raises_actionable_import_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match="setup.sh"):
        sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path, names=_NAMES)


def test_vda_skips_when_depths_exist(tmp_path):
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for name in _NAMES:
        np.save(npy_dir / f"{name[:-4]}.npy", np.ones((4, 4), dtype=np.float32))
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    # Complete per-frame npy set -> returned untouched, no VDA import attempted
    path = sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path, names=_NAMES)
    assert path == tmp_path / "depth_vda"


def test_vda_incomplete_npy_set_does_not_skip(tmp_path, monkeypatch):
    # One map for two names -> treated as missing -> inference path -> missing clone raises
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match="setup.sh"):
        sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path, names=_NAMES)


def test_vda_depth_complete_is_an_exact_stem_set_check(tmp_path):
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert not sfm.vda_depth_complete(tmp_path, _NAMES)  # no dir
    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))
    assert not sfm.vda_depth_complete(tmp_path, _NAMES)  # partial
    np.save(npy_dir / "frame_000001.npy", np.ones((4, 4), dtype=np.float32))
    assert sfm.vda_depth_complete(tmp_path, _NAMES)  # exact
    np.save(npy_dir / "frame_000009.npy", np.ones((4, 4), dtype=np.float32))
    assert not sfm.vda_depth_complete(tmp_path, _NAMES)  # leftover extra file


def test_vda_wrong_named_npy_set_does_not_skip(tmp_path, monkeypatch):
    # Right COUNT, wrong stems (a stale selection) -> the skip gate compares stem sets, not counts
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for stem in ("frame_000000", "frame_000007"):
        np.save(npy_dir / f"{stem}.npy", np.ones((4, 4), dtype=np.float32))
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match="setup.sh"):
        sfm.generate_vda_depth(frames, fps=30.0, out_dir=tmp_path, names=_NAMES)


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


def test_track_id_patch_renumbers_packed_64bit_ids():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.processors.track_establishment import TrackEngine

    sfm._patch_instantsfm_track_ids()

    # Idempotent — a second call must not wrap the wrapper
    patched = TrackEngine.FindTracksForProblem
    sfm._patch_instantsfm_track_ids()
    assert TrackEngine.FindTracksForProblem is patched

    # Packed global id (image 4, feature 12273) overflows int32 unless renumbered —
    # exactly the id that OverflowError'd the smoke run under numpy 2
    class _Images:
        is_registered = np.ones(3, dtype=bool)

        def __len__(self):
            return 3

    engine = TrackEngine(view_graph=None, images=_Images())
    packed_id = (4 << 32) | 12273
    obs = np.array([[0, 10], [1, 20], [2, 30]])
    tracks = engine.FindTracksForProblem(
        {packed_id: obs},
        {"min_num_view_per_track": 2, "max_num_view_per_track": 100},
    )

    # One surviving track, id stored without overflow, observations preserved
    assert len(tracks) == 1
    assert tracks.ids[0] == 0
    np.testing.assert_array_equal(tracks.observations[0], obs)


def test_pypose_robustmodel_target_patch_defaults_none():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    import inspect

    from pypose.optim.optimizer import RobustModel

    sfm._patch_pypose_robustmodel_target()

    # Idempotent — a second call must not wrap the wrapper
    patched = RobustModel.forward
    sfm._patch_pypose_robustmodel_target()
    assert RobustModel.forward is patched

    # bae's LM.step calls self.model(input) with no target — the patched signature
    # must default it to None instead of raising TypeError
    assert inspect.signature(RobustModel.forward).parameters["target"].default is None


def test_bae_pcg_patch_keeps_column_shape():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    import torch

    from bae.utils.pysolvers import PCG

    sfm._patch_bae_pcg_column_shape()

    # Idempotent — a second call must not wrap the wrapper
    patched = PCG.forward
    sfm._patch_bae_pcg_column_shape()
    assert PCG.forward is patched

    # bae LM passes a column rhs (-J_T @ R.view(-1, 1)); pypose-0.7.5 CG squeezes it
    # to 1-D and the unpatched wrapper returned 1-D, crashing TrustRegion's (J @ D).mT.
    # bae's preconditioner build (spdiags_) is a Triton kernel — CUDA tensors only.
    if not torch.cuda.is_available():
        pytest.skip("bae PCG preconditioner needs CUDA")
    solver = PCG(tol=1e-6)
    A = torch.eye(4, dtype=torch.float64, device="cuda").to_sparse_csr()
    b = torch.arange(1.0, 5.0, dtype=torch.float64, device="cuda")
    column = solver(A, b[:, None])
    vector = solver(A, b)
    assert column.shape == (4, 1)
    assert vector.shape == (4,)
    torch.testing.assert_close(column[:, 0], b, rtol=1e-4, atol=1e-6)
