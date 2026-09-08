import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm import instantsfm


def test_creator_config_copy_prevents_module_dict_leak():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.controllers.config import RUNTIME_OPTIONS

    before = dict(RUNTIME_OPTIONS)
    cfg = instantsfm.InstantSfMCreator()._build_config()
    cfg.RUNTIME_OPTIONS["use_depths"] = True

    # Module-level dict must be untouched — Config aliases it; creator must copy
    assert RUNTIME_OPTIONS == before


def test_creator_retriangulation_flag_flips_skip_retriangulation():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.controllers.config import GENERAL_OPTIONS

    # Default off matches upstream; enabling must flip only the copied dict
    assert instantsfm.InstantSfMCreator()._build_config().OPTIONS["skip_retriangulation"] is True
    assert instantsfm.InstantSfMCreator(retriangulation=True)._build_config().OPTIONS["skip_retriangulation"] is False
    assert GENERAL_OPTIONS["skip_retriangulation"] is True


def test_track_id_patch_renumbers_packed_64bit_ids():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.processors.track_establishment import TrackEngine

    instantsfm._patch_instantsfm_track_ids()

    # Idempotent — a second call must not wrap the wrapper
    patched = TrackEngine.FindTracksForProblem
    instantsfm._patch_instantsfm_track_ids()
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

    instantsfm._patch_pypose_robustmodel_target()

    # Idempotent — a second call must not wrap the wrapper
    patched = RobustModel.forward
    instantsfm._patch_pypose_robustmodel_target()
    assert RobustModel.forward is patched

    # bae's LM.step calls self.model(input) with no target — the patched signature
    # must default it to None instead of raising TypeError
    assert inspect.signature(RobustModel.forward).parameters["target"].default is None


def test_bae_pcg_patch_keeps_column_shape():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    import torch
    from bae.utils.pysolvers import PCG

    instantsfm._patch_bae_pcg_column_shape()

    # Idempotent — a second call must not wrap the wrapper
    patched = PCG.forward
    instantsfm._patch_bae_pcg_column_shape()
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
    from instantsfm.scene.reconstruction import Reconstruction as InsfmReconstruction

    instantsfm._patch_instantsfm_colmap_write()

    # Idempotent — a second call must not wrap the wrapper
    patched = InsfmReconstruction._write_images_binary
    instantsfm._patch_instantsfm_colmap_write()
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
    out = instantsfm._nudge_edge_keypoints(feats, 1918, 1078)

    # Exact-edge coords move just inside; interior and beyond-edge coords are untouched
    assert out[1, 0] < 1918 and out[1, 0] > 1917.9
    assert out[2, 1] < 1078 and out[2, 1] > 1077.9
    np.testing.assert_array_equal(out[[0, 3]], feats[[0, 3]])
    assert feats[1, 0] == 1918.0  # input not mutated
    assert instantsfm._nudge_edge_keypoints(np.empty((0, 2)), 10, 10).size == 0


########################################################
########## Creator config surface #####################
########################################################


def test_instantsfm_random_seed_defaults_to_none():
    assert instantsfm.InstantSfMCreator().random_seed is None


def test_instantsfm_random_seed_reaches_runtime_options():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.controllers.config import RUNTIME_OPTIONS

    # Upstream SolveGlobalMapper reads RUNTIME_OPTIONS['random_seed'] with .get(key, None),
    # so an unset seed must leave the key ABSENT rather than write a default in
    assert "random_seed" not in instantsfm.InstantSfMCreator()._build_config().RUNTIME_OPTIONS

    # A set seed must reach the option verbatim — a hardcoded or coerced value silently
    # makes every scene reproduce to the same wrong reconstruction
    seeded = instantsfm.InstantSfMCreator(random_seed=1234)._build_config()
    assert seeded.RUNTIME_OPTIONS["random_seed"] == 1234

    # Config aliases the module-level dict; the seed must not leak out of this instance
    assert "random_seed" not in RUNTIME_OPTIONS


def test_instantsfm_min_num_view_per_track_defaults_to_none():
    assert instantsfm.InstantSfMCreator().min_num_view_per_track is None


def test_instantsfm_min_num_view_per_track_reaches_track_establishment_options():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.config.colmap import CONFIG

    # Unset must leave the upstream cut of 3 in place — the knob is opt-in
    assert instantsfm.InstantSfMCreator()._build_config().TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] == 3

    # A set value must reach the option verbatim; FindTracksForProblem compares against it
    cut = instantsfm.InstantSfMCreator(min_num_view_per_track=6)._build_config()
    assert cut.TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] == 6

    # Config aliases the module-level dict; the cut must not leak out of this instance
    assert CONFIG["TRACK_ESTABLISHMENT_OPTIONS"]["min_num_view_per_track"] == 3
    later = instantsfm.InstantSfMCreator()._build_config()
    assert later.TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] == 3


########################################################
########## SIFT database validity ######################
########################################################


# The image set a cached DB is checked against; reuse is gated on it matching exactly
_DB_NAMES = ["000001.png", "000002.png"]


def test_sift_database_valid_is_false_for_a_missing_file(tmp_path):
    """
    No DB at all is the first-run case, not a corruption.
    """
    assert instantsfm._sift_database_valid(tmp_path / "nope.db", _DB_NAMES) is False


def test_sift_database_valid_is_false_for_a_non_database_file(tmp_path):
    """
    A crashed colmap can leave a truncated/garbage file; pycolmap raises rather than returning.
    """
    db = tmp_path / "garbage.db"
    db.write_bytes(b"not a sqlite file")

    assert instantsfm._sift_database_valid(db, _DB_NAMES) is False


def test_sift_database_valid_is_false_for_an_empty_database(tmp_path):
    """
    An OOM-killed extractor leaves a well-formed but empty DB — an existence check would
    cache-hit on it and feed ReadColmapDatabase zero tracks.
    """
    db = tmp_path / "empty.db"
    pycolmap.Database.open(str(db)).close()

    assert instantsfm._sift_database_valid(db, _DB_NAMES) is False


def _sift_db(path, names=_DB_NAMES, *, verified_pair):
    """
    Minimal colmap SIFT database: one camera, one image per name with keypoints, optional verified pair.
    """
    db = pycolmap.Database.open(str(path))
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    db.write_camera(cam, True)
    keypoints = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    for image_id, name in enumerate(names, start=1):
        db.write_image(pycolmap.Image(name=name, camera_id=1, image_id=image_id), True)
        db.write_keypoints(image_id, keypoints)

    # Verified pairs land in two_view_geometries — what exhaustive_matcher writes, and the
    # only evidence that matching ran to completion rather than crashing after extraction
    if verified_pair:
        tvg = pycolmap.TwoViewGeometry()
        tvg.config = 2
        tvg.inlier_matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
        db.write_two_view_geometry(1, 2, tvg)
    db.close()


def test_sift_database_valid_is_true_for_a_complete_database(tmp_path):
    """
    Keypoints plus a verified pair is the cache-hit case — a full re-extraction is skipped.
    """
    db = tmp_path / "complete.db"
    _sift_db(db, verified_pair=True)

    assert instantsfm._sift_database_valid(db, _DB_NAMES) is True


def test_sift_database_valid_is_false_when_matching_never_finished(tmp_path):
    """
    Extraction finished, matching crashed: keypoints but zero verified pairs, which
    ReadColmapDatabase turns into an empty-tracks failure much later.
    """
    db = tmp_path / "unmatched.db"
    _sift_db(db, verified_pair=False)

    assert instantsfm._sift_database_valid(db, _DB_NAMES) is False


def test_sift_database_is_reused_for_the_same_image_set(tmp_path):
    """
    Same images in a different order is the same set — reordering must not force a re-extract.
    """
    db = tmp_path / "reused.db"
    _sift_db(db, verified_pair=True)

    assert instantsfm._sift_database_valid(db, list(reversed(_DB_NAMES))) is True


def test_sift_database_is_rebuilt_when_the_image_set_changed(tmp_path):
    """
    Nothing stages a per-run image copy any more, so the DB's own images table is the only
    record of which selection its features came from.
    """
    db = tmp_path / "stale.db"
    _sift_db(db, verified_pair=True)

    assert instantsfm._sift_database_valid(db, _DB_NAMES + ["000003.png"]) is False


########################################################
########## Stem rename #################################
########################################################


def _recon(names):
    """
    One PINHOLE camera, one image per name, one point3D observed in every image.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


def test_rename_images_to_stems_round_trips_through_write_binary(tmp_path):
    # InstantSfM names (frame_000000.jpg) -> contract stems, persisted in the rewritten model
    recon = _recon(["frame_000000.jpg", "frame_000003.jpg"])
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    instantsfm._rename_images_to_stems(recon, sparse_dir)
    assert sorted(im.name for im in recon.images.values()) == ["frame_000000", "frame_000003"]
    reread = pycolmap.Reconstruction(str(sparse_dir))
    assert sorted(im.name for im in reread.images.values()) == ["frame_000000", "frame_000003"]
    assert reread.num_points3D() == 1


########################################################
########## Image directory #############################
########################################################


def test_sfm_points_at_the_scene_images_dir_and_stages_nothing(tmp_path):
    """
    InstantSfM reads <scene>/images directly; no instantsfm/images/ copy is written.
    """
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)

    assert instantsfm._sfm_image_dir(scene / "images") == scene / "images"
    assert not (scene / "instantsfm" / "images").exists()


def test_sfm_image_dir_refuses_a_missing_directory(tmp_path):
    """
    A scene that was never preprocessed fails at the image dir, not deep inside InstantSfM.
    """
    with pytest.raises(FileNotFoundError, match="image directory"):
        instantsfm._sfm_image_dir(tmp_path / "scene" / "images")
