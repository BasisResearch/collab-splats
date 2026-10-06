"""Reference frames stay index-aligned once localized frames join the reference set.

LocalizationResult.ref_frame_indices indexes the merged reconstruction+localized frame list,
so image_paths / frame_sources / extrinsics must all agree in length. _extrinsics itself stays
reconstruction-only (world_points indexing depends on that) — the extrinsics property joins them.
"""

import numpy as np

from collab_splats.localization.localizer import CameraLocalizer

########
# Fakes
########


def _eight_keypoints(image: np.ndarray) -> np.ndarray:
    """Eight fixed keypoints, whatever the image."""
    return np.arange(16, dtype=np.float32).reshape(8, 2)


def _make_localizer(tmp_path, stub_matcher, n_frames=2):
    """Build a localizer from synthetic RGB arrays (no GPU, no image IO)."""
    images = [np.full((48, 64, 3), 128, dtype=np.uint8) for _ in range(n_frames)]
    ids = [f"{i:05d}.jpg" for i in range(n_frames)]
    wp = np.random.default_rng(0).normal(size=(n_frames, 48, 64, 3)).astype(np.float32)
    extr = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    return CameraLocalizer(wp, extr, images=images, ids=ids, extractor=stub_matcher(_eight_keypoints)), wp, extr


def _distinct_pose(tx: float) -> np.ndarray:
    """A pose distinguishable from the identity reconstruction extrinsics."""
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = tx
    return pose


########
# Tests
########


def test_extrinsics_matches_paths_after_add_localized_frame(tmp_path, stub_matcher):
    # 2 reconstruction frames + 1 localized → every reference view list must be length 3
    loc, _, _ = _make_localizer(tmp_path, stub_matcher, n_frames=2)
    pose = _distinct_pose(1.5)
    loc.add_localized_frame(tmp_path / "query.jpg", pose)

    assert len(loc.image_paths) == len(loc.frame_sources) == loc.extrinsics.shape[0] == 3
    assert loc.frame_sources == ["reconstruction", "reconstruction", "localized"]
    np.testing.assert_allclose(loc.extrinsics[2], pose)


def test_load_index_round_trip_preserves_localized_pose(tmp_path, stub_matcher):
    # save → append (persisted) → reload: the localized pose must survive and stay aligned
    loc, wp, extr = _make_localizer(tmp_path, stub_matcher, n_frames=2)
    zp = tmp_path / "pointcloud.zarr"
    loc.save_index(zp, "disk")
    pose = _distinct_pose(2.5)
    loc.add_localized_frame(tmp_path / "query.jpg", pose, zarr_path=zp, extractor_name="disk")

    reloaded = CameraLocalizer.load_index(zp, "disk", wp, extr, extractor=stub_matcher(_eight_keypoints))

    assert len(reloaded.image_paths) == len(reloaded.frame_sources) == reloaded.extrinsics.shape[0] == 3
    assert reloaded.frame_sources == ["reconstruction", "reconstruction", "localized"]
    np.testing.assert_allclose(reloaded.extrinsics[2], pose, atol=1e-6)
    assert reloaded._extrinsics.shape == (2, 4, 4)
