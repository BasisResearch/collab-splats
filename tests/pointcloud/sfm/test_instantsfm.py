import cv2
import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

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
    assert (
        instantsfm.InstantSfMCreator()._build_config().OPTIONS["skip_retriangulation"]
        is True
    )
    assert (
        instantsfm.InstantSfMCreator(retriangulation=True)
        ._build_config()
        .OPTIONS["skip_retriangulation"]
        is False
    )
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


def test_nudge_edge_keypoints_pulls_exact_edge_inward_only():
    feats = np.array(
        [[10.0, 20.0], [1918.0, 5.0], [7.0, 1078.0], [1930.0, 3.0]], dtype=np.float32
    )
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


def test_instantsfm_random_seed_reaches_runtime_options():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.controllers.config import RUNTIME_OPTIONS

    # Upstream SolveGlobalMapper reads RUNTIME_OPTIONS['random_seed'] with .get(key, None),
    # so an unset seed must leave the key ABSENT rather than write a default in
    assert (
        "random_seed"
        not in instantsfm.InstantSfMCreator()._build_config().RUNTIME_OPTIONS
    )

    # A set seed must reach the option verbatim — a hardcoded or coerced value silently
    # makes every scene reproduce to the same wrong reconstruction
    seeded = instantsfm.InstantSfMCreator(random_seed=1234)._build_config()
    assert seeded.RUNTIME_OPTIONS["random_seed"] == 1234

    # Config aliases the module-level dict; the seed must not leak out of this instance
    assert "random_seed" not in RUNTIME_OPTIONS


def test_instantsfm_min_num_view_per_track_reaches_track_establishment_options():
    pytest.importorskip("instantsfm")
    # Optional heavy dep, may be absent — imported inside the importorskip'd test body
    from instantsfm.config.colmap import CONFIG

    # Unset must leave the upstream cut of 3 in place — the knob is opt-in
    assert (
        instantsfm.InstantSfMCreator()
        ._build_config()
        .TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"]
        == 3
    )

    # A set value must reach the option verbatim; FindTracksForProblem compares against it
    cut = instantsfm.InstantSfMCreator(min_num_view_per_track=6)._build_config()
    assert cut.TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] == 6

    # Config aliases the module-level dict; the cut must not leak out of this instance
    assert CONFIG["TRACK_ESTABLISHMENT_OPTIONS"]["min_num_view_per_track"] == 3
    later = instantsfm.InstantSfMCreator()._build_config()
    assert later.TRACK_ESTABLISHMENT_OPTIONS["min_num_view_per_track"] == 3


def _insfm_scene(image_dir):
    """
    Two-cluster InstantSfM solve over five flat-color PNGs; image 1 unregistered, image 4 in cluster 1.

    - track 1 below min length 3; tracks 2, 3 share image 0 feature 2 (3 wins); track 4 outside cluster 0
    """
    # Optional heavy dep, may be absent — imported inside the importorskip'd helper
    from instantsfm.scene.defs import CameraModelId, Cameras, Images, Tracks

    # Two cameras of different models
    cameras = Cameras(2)
    cameras.widths[:], cameras.heights[:] = 64, 48
    cameras.set_params(0, [50.0, 32.0, 24.0, 0.01], CameraModelId.SIMPLE_RADIAL)
    cameras.set_params(1, [60.0, 55.0, 31.5, 24.5], CameraModelId.PINHOLE)

    # Five images; each file one flat color so bilinear samples are exact
    rng = np.random.default_rng(0)
    images = Images(5)
    images.cam_ids[:] = [0, 0, 1, 0, 0]
    images.is_registered[:] = [True, False, True, True, True]
    images.cluster_ids[:] = [0, 0, 0, 0, 1]
    for i in range(5):
        images.filenames[i] = f"frame_{i:06d}.png"
        images.world2cams[i, :3, :3] = Rotation.random(random_state=i).as_matrix()
        images.world2cams[i, :3, 3] = rng.normal(size=3)
        images.features[i] = rng.uniform(1, 40, size=(5, 2)).astype(np.float32)
        cv2.imwrite(
            str(image_dir / images.filenames[i]),
            np.full((48, 64, 3), (30, 20 + i, 10 * i), np.uint8),
        )

    # Five tracks exercising the filters
    tracks = Tracks(5)
    tracks.xyzs[:] = rng.normal(size=(5, 3))
    tracks.colors[:] = 7
    tracks.observations = [
        np.array(obs)
        for obs in (
            [[0, 1], [2, 0], [3, 2], [1, 4]],
            [[2, 1], [3, 1]],
            [[0, 2], [3, 3], [4, 0]],
            [[0, 2], [2, 3], [3, 4]],
            [[4, 1], [4, 2], [1, 0]],
        )
    ]
    return cameras, images, tracks


def test_to_pycolmap_matches_upstream_writer_selection(tmp_path):
    pytest.importorskip("instantsfm")
    cameras, images, tracks = _insfm_scene(tmp_path)
    recon = instantsfm._to_pycolmap(cameras, images, tracks, tmp_path)

    # Cameras verbatim, one trivial rig each
    assert {i: (c.model.name, c.params.tolist()) for i, c in recon.cameras.items()} == {
        0: ("SIMPLE_RADIAL", [50.0, 32.0, 24.0, 0.01]),
        1: ("PINHOLE", [60.0, 55.0, 31.5, 24.5]),
    }
    assert sorted(recon.rigs) == [0, 1]

    # Registered cluster-0 images keep full keypoint lists; ids only where a track round-trips
    none = pycolmap.INVALID_POINT3D_ID
    point3d_ids = {
        0: [none, 0, 3, none, none],
        2: [0, none, none, 3, none],
        3: [none, none, 0, 2, 3],
    }
    assert sorted(recon.reg_image_ids()) == [0, 2, 3]
    for i, image in recon.images.items():
        assert (image.name, image.camera_id) == (images.filenames[i], images.cam_ids[i])
        assert [p.point3D_id for p in image.points2D] == point3d_ids[i]
        np.testing.assert_array_equal(
            [p.xy for p in image.points2D], images.features[i]
        )
        np.testing.assert_allclose(
            image.cam_from_world().matrix(), images.world2cams[i, :3], atol=1e-12
        )

    # Every track is a point; colors are truncated means of the flat RGB (10i, 20+i, 30) frames
    points = {
        i: (p.color.tolist(), [(e.image_id, e.point2D_idx) for e in p.track.elements])
        for i, p in recon.points3D.items()
    }
    assert points == {
        0: ([16, 21, 30], [(0, 1), (2, 0), (3, 2)]),
        1: ([7, 7, 7], []),
        2: ([30, 23, 30], [(3, 3)]),
        3: ([16, 21, 30], [(0, 2), (2, 3), (3, 4)]),
        4: ([7, 7, 7], []),
    }
    np.testing.assert_array_equal(
        [recon.points3D[i].xyz for i in range(5)], tracks.xyzs
    )
    assert all(p.error == 0.0 for p in recon.points3D.values())

    # One cluster is kept whole, whatever its id; several need a cluster 0; none registered raises
    images.cluster_ids[:] = 5
    recon = instantsfm._to_pycolmap(cameras, images, tracks, tmp_path)
    assert sorted(recon.reg_image_ids()) == [0, 2, 3, 4]
    images.cluster_ids[:] = [1, 1, 1, 1, 2]
    with pytest.raises(RuntimeError, match="no cluster 0"):
        instantsfm._to_pycolmap(cameras, images, tracks, tmp_path)
    images.is_registered[:] = False
    with pytest.raises(RuntimeError, match="registered no images"):
        instantsfm._to_pycolmap(cameras, images, tracks, tmp_path)
