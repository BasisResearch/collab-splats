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


def test_tracked_point3d_ids_drops_observationless_points():
    # InstantSfM exports sub-min-track-length points with empty tracks — the result
    # tail must exclude them (pixel_indices reads track.elements[0])
    class _EmptyTrack:
        elements = []

    class _EmptyP3D:
        track = _EmptyTrack()

    recon = _recon_with_keypoint((1.0, 2.0))
    recon.points3D[3] = _EmptyP3D()
    assert sfm._tracked_point3d_ids(recon) == [7]


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


def test_creator_retriangulation_flag_flips_skip_retriangulation():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.controllers.config import GENERAL_OPTIONS

    # Default off matches upstream; enabling must flip only the copied dict
    assert sfm.InstantSfMCreator()._build_config().OPTIONS["skip_retriangulation"] is True
    assert sfm.InstantSfMCreator(retriangulation=True)._build_config().OPTIONS["skip_retriangulation"] is False
    assert GENERAL_OPTIONS["skip_retriangulation"] is True


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


def test_colmap_write_patch_produces_pycolmap_readable_model(tmp_path):
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    import pycolmap
    from instantsfm.scene.reconstruction import Reconstruction as InsfmReconstruction

    sfm._patch_instantsfm_colmap_write()

    # Idempotent — a second call must not wrap the wrapper
    patched = InsfmReconstruction._write_images_binary
    sfm._patch_instantsfm_colmap_write()
    assert InsfmReconstruction._write_images_binary is patched

    # Minimal scene reproducing the smoke-run crash: upstream compressed each
    # image's points2D to the valid-track subset while points3D observations kept
    # ORIGINAL feature indices (vector::_M_range_check on read-back). Image 0 is
    # unregistered; track 1 is below min_track_length — both contribute
    # observations that must be dropped, not written.
    class _ModelId:
        value = 1  # PINHOLE

    class _Cam:
        model_id = _ModelId()
        width, height = 64, 48
        params = [50.0, 50.0, 32.0, 24.0]

    class _Images:
        world2cams = [np.eye(4)] * 3
        cam_ids = [0, 0, 0]
        filenames = ["frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg"]
        features = [np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])] * 3

        def __len__(self):
            return 3

    class _Tracks:
        xyzs = np.zeros((2, 3))
        colors = np.zeros((2, 3), dtype=np.uint8)
        observations = [
            np.array([[1, 2], [2, 1], [0, 1]]),  # feat 2 of image 1 crashed the old writer
            np.array([[1, 0]]),  # sub-min-length: exported point, no valid obs
        ]

        def __len__(self):
            return 2

    recon = InsfmReconstruction([_Cam()], _Images(), _Tracks())
    recon._selected_indices = np.array([1, 2])
    recon.build_correspondences(min_track_length=2)
    recon.write_binary(str(tmp_path))

    # pycolmap must accept the model; full keypoint lists, filtered observations
    read_back = pycolmap.Reconstruction(str(tmp_path))
    assert set(read_back.images) == {1, 2}
    assert all(len(img.points2D) == 3 for img in read_back.images.values())
    assert len(read_back.points3D[0].track.elements) == 2
    assert len(read_back.points3D[1].track.elements) == 0
    assert read_back.images[1].points2D[2].point3D_id == 0


def test_nudge_edge_keypoints_pulls_exact_edge_inward_only():
    feats = np.array([[10.0, 20.0], [1918.0, 5.0], [7.0, 1078.0], [1930.0, 3.0]], dtype=np.float32)
    out = sfm._nudge_edge_keypoints(feats, 1918, 1078)

    # Exact-edge coords move just inside; interior and beyond-edge coords are untouched
    assert out[1, 0] < 1918 and out[1, 0] > 1917.9
    assert out[2, 1] < 1078 and out[2, 1] > 1077.9
    np.testing.assert_array_equal(out[[0, 3]], feats[[0, 3]])
    assert feats[1, 0] == 1918.0  # input not mutated
    assert sfm._nudge_edge_keypoints(np.empty((0, 2)), 10, 10).size == 0


def _stub_vda_model(monkeypatch):
    """
    Patch _load_vda_model with a stub whose depth map for row i is filled with i + 1.
    """

    def _fake_model(**_kwargs):
        class _M:
            def load_state_dict(self, *_a, **_k):
                return None

            def to(self, *_a, **_k):
                return self

            def eval(self):
                return self

            def infer_video_depth(self, frames, fps, **_k):
                maps = np.stack([np.full((8, 8), float(i + 1), dtype=np.float32) for i in range(len(frames))])
                return maps, fps

        return _M()

    monkeypatch.setattr(sfm, "_load_vda_model", _fake_model)


def test_keep_rows_length_must_match_names(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"keep_rows \(3\) must align one-to-one"):
        sfm.generate_vda_depth(frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[0, 2, 4])


def test_keep_rows_must_be_in_range(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of range"):
        sfm.generate_vda_depth(frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[0, 9])


def test_keep_rows_must_not_repeat_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)

    # A collision would hand both keyframes the same depth map instead of raising
    with pytest.raises(ValueError, match="duplicate rows"):
        sfm.generate_vda_depth(frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[2, 2])


def test_keep_rows_writes_only_the_requested_rows(tmp_path, monkeypatch):
    # Stub inference: 5 context frames in, one distinguishable depth map per frame
    _stub_vda_model(monkeypatch)
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    sfm.generate_vda_depth(frames, fps=8.0, out_dir=tmp_path, names=_NAMES, keep_rows=[1, 3], depth_width=8)

    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert sorted(p.name for p in npy_dir.iterdir()) == ["frame_000000.npy", "frame_000001.npy"]
    assert np.all(np.load(npy_dir / "frame_000000.npy") == 2.0)
    assert np.all(np.load(npy_dir / "frame_000001.npy") == 4.0)


def test_keep_rows_none_writes_every_frame_in_order(tmp_path, monkeypatch):
    # Default path: no row selection, so every context row is a keyframe
    _stub_vda_model(monkeypatch)
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    sfm.generate_vda_depth(frames, fps=2.0, out_dir=tmp_path, names=_NAMES, depth_width=8)

    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert sorted(p.name for p in npy_dir.iterdir()) == ["frame_000000.npy", "frame_000001.npy"]
    assert np.all(np.load(npy_dir / "frame_000000.npy") == 1.0)
    assert np.all(np.load(npy_dir / "frame_000001.npy") == 2.0)
