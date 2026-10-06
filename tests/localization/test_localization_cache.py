"""Tests for localization feature + track cache (zarr-backed)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import zarr

from collab_splats.localization import CameraLocalizer, LocalFeatures
from collab_splats.localization import localizer as localizer_mod
from collab_splats.localization.localizer import read_localization_db
from collab_splats.utils.image import open_image

########################################################################
# Helpers
########################################################################


def _make_features(n_kpts: int = 10, desc_dim: int = 128) -> LocalFeatures:
    """Synthetic LocalFeatures for mocking — no GPU needed."""
    return LocalFeatures(
        keypoints=torch.rand(n_kpts, 2) * 60,
        descriptors=torch.rand(n_kpts, desc_dim),
        image_size=(64, 64),
    )


def _make_scene(n_frames: int = 3, n_pts: int = 20):
    """Minimal reconstruction scene: sparse pts, dense per-frame world maps, cameras."""
    rng = np.random.default_rng(42)
    extrinsics = np.zeros((n_frames, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((n_frames, 3, 3), dtype=np.float32)
    for i in range(n_frames):
        extrinsics[i] = np.eye(4)
        extrinsics[i, 0, 3] = i * 0.5
        intrinsics[i] = np.array([[32, 0, 32], [0, 32, 32], [0, 0, 1]], dtype=np.float32)
    pts3d = rng.random((n_pts, 3)).astype(np.float32)
    pts3d[:, 2] += 2.0
    world_points = rng.random((n_frames, 64, 64, 3)).astype(np.float32)
    return pts3d, world_points, extrinsics, intrinsics


def _make_image_files(tmp_path: Path, n: int = 3, size: int = 64) -> list[Path]:
    """Write tiny solid-color JPEG images to tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = np.full((size, size, 3), fill_value=80 + i * 40, dtype=np.uint8)
        p = tmp_path / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _random_keypoints(image: np.ndarray) -> np.ndarray:
    """
    Ten keypoints inside the image.
    """
    return np.random.default_rng(0).random((10, 2)) * 60


def _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher):
    """Build CameraLocalizer backed by a stub matcher (no GPU)."""
    mock_ext = stub_matcher(_random_keypoints)
    images = [np.asarray(open_image(p).convert("RGB")) for p in image_paths]
    ids = [Path(p).name for p in image_paths]
    localizer = CameraLocalizer(
        world_points=world_points,
        extrinsics=extrinsics,
        images=images,
        ids=ids,
        extractor=mock_ext,
    )
    return localizer, mock_ext


def _empty_zarr(tmp_path: Path) -> Path:
    """Create an empty pointcloud.zarr store and return its path."""
    zarr_path = tmp_path / "pointcloud.zarr"
    zarr.open(str(zarr_path), mode="w")
    return zarr_path


########################################################################
# Feature DB save and load
########################################################################


def test_camera_localizer_stores_image_paths_and_sources(tmp_path, stub_matcher):
    """CameraLocalizer should maintain _image_paths and _frame_sources after build."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    assert len(localizer.image_paths) == 3
    assert localizer.frame_sources == ["reconstruction", "reconstruction", "reconstruction"]


def test_save_index_creates_reconstruction_group(tmp_path, stub_matcher):
    """save_index writes local_features/{name}/reconstruction/ with expected arrays."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/reconstruction" in store
    grp = store["local_features/disk/reconstruction"]
    assert "frame_offsets" in grp
    assert "keypoints" in grp
    assert "descriptors" in grp
    assert len(grp.attrs["image_paths"]) == 3
    assert grp["frame_offsets"].shape == (4,)  # N+1 = 3+1
    assert grp["keypoints"].shape[1] == 2
    assert grp["descriptors"].shape[0] == grp["keypoints"].shape[0]


def test_load_index_missing_extractor_raises(tmp_path):
    """load_index raises KeyError when extractor cache not found."""
    zarr_path = _empty_zarr(tmp_path)
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    with pytest.raises(KeyError, match="disk"):
        CameraLocalizer.load_index(
            zarr_path=zarr_path,
            extractor_name="disk",
            world_points=world_points,
            extrinsics=extrinsics,
        )


########################################################################
# Index updates and localized frames
########################################################################


def test_update_index_appends_new_frames(tmp_path, stub_matcher):
    """update_index extracts + appends new reconstruction frames to zarr; memory is unchanged."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    # Create 2 new frames as in-memory RGB arrays + string ids (no image IO)
    new_images = [
        np.full((64, 64, 3), 200, dtype=np.uint8),
        np.full((64, 64, 3), 220, dtype=np.uint8),
    ]
    new_ids = ["frame_098.jpg", "frame_099.jpg"]
    new_ext = stub_matcher(_random_keypoints)
    localizer._extractor = new_ext

    localizer.update_index(
        new_images=new_images,
        new_ids=new_ids,
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    assert new_ext.n_extract == 2
    assert len(localizer._frame_features) == 3
    assert localizer.image_paths[-1] == "frame_002.jpg"

    # Verify zarr updated
    store = zarr.open(str(zarr_path), mode="r")
    grp = store["local_features/disk/reconstruction"]
    assert grp["frame_offsets"].shape == (6,)  # 5+1
    assert len(grp.attrs["image_paths"]) == 5


def test_add_localized_frame_extends_index_in_memory(tmp_path, stub_matcher):
    """add_localized_frame records id and pose in memory; reference features are unchanged."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    assert len(localizer._frame_features) == 3

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0, 3] = 2.0
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(
        image_path=new_path,
        pose=new_pose,
    )

    assert len(localizer._frame_features) == 3
    assert localizer.frame_sources[-1] == "localized"
    assert localizer.image_paths[-1] == str(new_path)


def test_add_localized_frame_persists_to_zarr(tmp_path, stub_matcher):
    """add_localized_frame with zarr_path writes to localized/ group."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    new_pose = np.eye(4, dtype=np.float32)
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(
        image_path=new_path,
        pose=new_pose,
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" in store
    loc_grp = store["local_features/disk/localized"]
    assert loc_grp["extrinsics"].shape == (1, 4, 4)
    assert len(loc_grp.attrs["image_paths"]) == 1


def test_add_localized_frame_duplicate_skipped(tmp_path, stub_matcher):
    """Duplicate image_path is silently skipped — no double-add."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=2)
    image_paths = _make_image_files(tmp_path, n=2)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    new_pose = np.eye(4, dtype=np.float32)
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(new_path, new_pose)
    localizer.add_localized_frame(new_path, new_pose)  # duplicate

    assert len(localizer.image_paths) == 3  # 2 rec + 1 loc, not 4


def test_load_index_includes_localized_frames(tmp_path, stub_matcher):
    """After add + reload, localized frame is in the loaded index."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0, 3] = 1.5
    new_path = tmp_path / "query.jpg"
    localizer.add_localized_frame(new_path, new_pose, zarr_path=zarr_path, extractor_name="disk")

    # Reload from zarr
    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path,
        extractor_name="disk",
        world_points=world_points,
        extrinsics=extrinsics,
        extractor=stub_matcher(_random_keypoints),
    )

    assert len(loaded._frame_features) == 3
    assert loaded.frame_sources == ["reconstruction"] * 3 + ["localized"]
    assert loaded.image_paths[-1] == str(new_path)
    np.testing.assert_allclose(loaded.extrinsics[-1], new_pose)


########################################################################
# Clearing localized frames
########################################################################


def test_clear_localized_frames_removes_zarr_group(tmp_path, stub_matcher):
    """clear_localized_frames deletes localized/ group; reconstruction/ untouched."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths, stub_matcher)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    # Add a localized frame
    localizer.add_localized_frame(
        tmp_path / "q.jpg",
        np.eye(4, dtype=np.float32),
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" in store

    # Clear
    CameraLocalizer.clear_localized_frames(zarr_path, "disk")

    store2 = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" not in store2
    assert "local_features/disk/reconstruction" in store2  # untouched


########################################################################
# Cache reader and from_pointcloud
########################################################################


def test_save_load_index_round_trip(tmp_path, replay_matcher):
    """save_index then load_index returns the ids, grid, features, sources and global descriptors written."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=2)
    image_paths = _make_image_files(tmp_path / "imgs", n=2)
    feats = [_make_features(n_kpts=5), _make_features(n_kpts=8)]
    images = [np.asarray(open_image(p).convert("RGB")) for p in image_paths]
    ids = [Path(p).name for p in image_paths]
    localizer = CameraLocalizer(world_points, extrinsics, images, ids, extractor=replay_matcher(feats))
    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "xfeat")

    loaded = CameraLocalizer.load_index(zarr_path, "xfeat", world_points, extrinsics, extractor=replay_matcher([]))

    assert loaded.image_paths == ids and loaded._image_hw == (64, 64)
    assert loaded.frame_sources == ["reconstruction"] * 2
    assert [len(f.keypoints) for f in loaded._frame_features] == [5, 8]

    for want, have in zip(feats, loaded._frame_features):
        np.testing.assert_allclose(have.keypoints.numpy(), want.keypoints.numpy())
        np.testing.assert_allclose(have.descriptors.numpy(), want.descriptors.numpy())

    np.testing.assert_allclose(loaded._global_desc, localizer._global_desc)


def test_read_localization_db_missing_raises(tmp_path):
    """Missing cache raises KeyError naming the extractor."""
    zarr_path = _empty_zarr(tmp_path)
    with pytest.raises(KeyError, match="disk"):
        read_localization_db(zarr_path, "disk")


def _mock_result(world_points: np.ndarray) -> SimpleNamespace:
    """
    PointcloudResult stand-in carrying what from_pointcloud reads.
    """
    n, H, W = world_points.shape[:3]
    return SimpleNamespace(
        world_points=world_points,
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        image_paths=[Path(f"frame_{i:06d}") for i in range(n)],
        original_coords=np.tile(np.array([0, 0, W, H, W, H], np.float32), (n, 1)),
    )


def _no_keypoints(image: np.ndarray) -> np.ndarray:
    """
    Zero keypoints.
    """
    return np.zeros((0, 2))


def _three_keypoints(image: np.ndarray) -> np.ndarray:
    """
    Three keypoints at the origin.
    """
    return np.zeros((3, 2))


def test_from_pointcloud_cache_miss_builds_and_saves(tmp_path, stub_matcher):
    wp = np.zeros((3, 8, 8, 3), np.float32)
    matcher = stub_matcher(_three_keypoints)
    ids = [f"frame_{i:06d}.png" for i in range(3)]

    CameraLocalizer.from_pointcloud(
        _mock_result(wp),
        zarr_path=tmp_path / "pc.zarr",
        images=[np.zeros((8, 8, 3), np.uint8)] * 3,
        ids=ids,
        extractor=matcher,
    )

    assert matcher.n_extract == 3
    store = zarr.open(str(tmp_path / "pc.zarr"), mode="r")
    assert "local_features/stub/reconstruction" in store


def test_from_pointcloud_miss_without_images_raises(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)

    with pytest.raises(ValueError, match="needs images"):
        CameraLocalizer.from_pointcloud(
            _mock_result(wp), zarr_path=tmp_path / "pc.zarr", ids=["a"], extractor=stub_matcher(_three_keypoints)
        )


def test_load_index_without_global_desc_raises_keyerror(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [np.zeros((8, 8, 3), np.uint8)],
        ["a"],
        extractor=stub_matcher(_no_keypoints),
    )
    loc.save_index(tmp_path / "pc.zarr", "stub")
    store = zarr.open(str(tmp_path / "pc.zarr"), mode="a")
    del store["local_features/stub/reconstruction/global_desc"]

    with pytest.raises(KeyError, match="global_desc"):
        CameraLocalizer.load_index(
            tmp_path / "pc.zarr", "stub", wp, loc.extrinsics, extractor=stub_matcher(_no_keypoints)
        )


def _lazy_frames(reads: list[int], n: int):
    """
    Yield n blank frames, recording each draw in reads.
    """
    for _ in range(n):
        reads.append(1)
        yield np.zeros((8, 8, 3), np.uint8)


def test_from_pointcloud_cache_hit_reads_zero_frames(tmp_path, stub_matcher):
    wp = np.zeros((2, 8, 8, 3), np.float32)
    result = _mock_result(wp)
    ids = ["frame_000000.png", "frame_000001.png"]
    built = CameraLocalizer.from_pointcloud(
        result,
        zarr_path=tmp_path / "pc.zarr",
        images=[np.zeros((8, 8, 3), np.uint8)] * 2,
        ids=ids,
        extractor=stub_matcher(_three_keypoints),
    )
    reads = []

    hit = CameraLocalizer.from_pointcloud(
        result,
        zarr_path=tmp_path / "pc.zarr",
        images=_lazy_frames(reads, 2),
        ids=ids,
        extractor=stub_matcher(_three_keypoints),
    )

    assert reads == [] and hit.image_paths == built.image_paths


def test_from_pointcloud_stale_ids_rebuild(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    result = _mock_result(wp)
    CameraLocalizer.from_pointcloud(
        result,
        zarr_path=tmp_path / "pc.zarr",
        images=[np.zeros((8, 8, 3), np.uint8)],
        ids=["frame_000000.jpg"],
        extractor=stub_matcher(_three_keypoints),
    )
    matcher = stub_matcher(_three_keypoints)

    CameraLocalizer.from_pointcloud(
        result,
        zarr_path=tmp_path / "pc.zarr",
        images=[np.zeros((8, 8, 3), np.uint8)],
        ids=["frame_000000.png"],
        extractor=matcher,
    )

    assert matcher.n_extract == 1


def test_update_index_one_write_per_array(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [np.zeros((8, 8, 3), np.uint8)],
        ["a"],
        extractor=stub_matcher(_three_keypoints),
    )
    loc.save_index(tmp_path / "pc.zarr", "stub")

    loc.update_index([np.zeros((8, 8, 3), np.uint8)] * 5, [f"n{i}" for i in range(5)], tmp_path / "pc.zarr", "stub")

    feats, paths, _ = read_localization_db(tmp_path / "pc.zarr", "stub")
    assert paths == ["a", "n0", "n1", "n2", "n3", "n4"] and len(feats) == 6
    store = zarr.open(str(tmp_path / "pc.zarr"), mode="r")
    assert store["local_features/stub/reconstruction/global_desc"].shape[0] == 6


def test_add_localized_frame_exact_id_guard(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [np.zeros((8, 8, 3), np.uint8)],
        ["q.png"],
        extractor=stub_matcher(_three_keypoints),
    )
    loc.add_localized_frame("q.png", np.eye(4))
    loc.add_localized_frame("q.jpg", np.eye(4))

    assert loc.image_paths == ["q.png", "q.jpg"]


def _blank_frames(n: int) -> list[np.ndarray]:
    """
    n blank 8x8 RGB frames.
    """
    return [np.zeros((8, 8, 3), np.uint8)] * n


def test_rebuild_drops_stale_localized_frames(tmp_path, stub_matcher):
    wp = np.zeros((2, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    loc = CameraLocalizer.from_pointcloud(
        _mock_result(wp),
        zarr_path=zp,
        ids=["a", "b"],
        images=_blank_frames(2),
        extractor=stub_matcher(_three_keypoints),
    )
    loc.add_localized_frame("q", np.eye(4), zarr_path=zp, extractor_name="stub")

    CameraLocalizer.from_pointcloud(
        _mock_result(wp),
        zarr_path=zp,
        ids=["c", "d"],
        images=_blank_frames(2),
        extractor=stub_matcher(_three_keypoints),
    )
    reloaded = CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["c", "d"], extractor=stub_matcher(_three_keypoints)
    )

    assert reloaded.image_paths == ["c", "d"]
    assert reloaded.frame_sources == ["reconstruction"] * 2


def test_localized_then_update_index_round_trip(tmp_path, stub_matcher):
    wp = np.zeros((2, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    loc = CameraLocalizer.from_pointcloud(
        _mock_result(wp),
        zarr_path=zp,
        ids=["a", "b"],
        images=_blank_frames(2),
        extractor=stub_matcher(_three_keypoints),
    )
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 7.0
    loc.add_localized_frame("q", pose, zarr_path=zp, extractor_name="stub")

    loc.update_index(_blank_frames(1), ["e"], zp, "stub")

    assert loc.image_paths == ["a", "b", "q"]
    assert len(loc.image_paths) == loc.extrinsics.shape[0] == len(loc.frame_sources)
    rec_feats, rec_ids, _ = read_localization_db(zp, "stub")
    assert rec_ids == ["a", "b", "e"]
    assert [len(f.keypoints) for f in rec_feats] == [3, 3, 3]
    wp3 = np.zeros((3, 8, 8, 3), np.float32)
    reloaded = CameraLocalizer.from_pointcloud(
        _mock_result(wp3), zarr_path=zp, ids=["a", "b", "e"], extractor=stub_matcher(_three_keypoints)
    )
    assert reloaded.image_paths == ["a", "b", "e", "q"]
    assert reloaded.frame_sources == ["reconstruction"] * 3 + ["localized"]
    assert reloaded._global_desc.shape[0] == len(reloaded._frame_features) == 3
    np.testing.assert_allclose(reloaded.extrinsics[3], pose)


def test_update_index_without_db_writes_then_appends(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(
        wp, np.eye(4, dtype=np.float32)[None], _blank_frames(1), ["a"], extractor=stub_matcher(_three_keypoints)
    )

    loc.update_index(_blank_frames(1), ["e"], tmp_path / "pc.zarr", "stub")

    _, rec_ids, _ = read_localization_db(tmp_path / "pc.zarr", "stub")
    assert rec_ids == ["a", "e"]
    assert loc.image_paths == ["a"]


def test_from_pointcloud_rebuilds_when_commit_marker_missing(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=stub_matcher(_three_keypoints)
    )
    group = zarr.open(str(zp), mode="a")["local_features/stub/reconstruction"]
    del group.attrs["image_paths"]
    matcher = stub_matcher(_three_keypoints)

    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=matcher
    )

    assert matcher.n_extract == 1


def test_from_pointcloud_rebuilds_inconsistent_db(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=stub_matcher(_three_keypoints)
    )
    group = zarr.open(str(zp), mode="a")["local_features/stub/reconstruction"]
    localizer_mod._append_rows(group, "global_desc", group["global_desc"][:])
    matcher = stub_matcher(_three_keypoints)

    with pytest.raises(KeyError, match="inconsistent"):
        read_localization_db(zp, "stub")

    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=matcher
    )

    assert matcher.n_extract == 1


def test_from_pointcloud_rebuilds_on_keypoint_cap_change(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=stub_matcher(_three_keypoints)
    )
    matcher = stub_matcher(_three_keypoints)
    matcher.max_num_keypoints = 512

    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=matcher
    )

    assert matcher.n_extract == 1


def test_from_pointcloud_rebuilds_on_resolution_change(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    CameraLocalizer.from_pointcloud(
        _mock_result(wp), zarr_path=zp, ids=["a"], images=_blank_frames(1), extractor=stub_matcher(_three_keypoints)
    )
    result = _mock_result(wp)
    result.original_coords = np.array([[0, 0, 16, 16, 16, 16]], np.float32)
    matcher = stub_matcher(_three_keypoints)

    CameraLocalizer.from_pointcloud(
        result, zarr_path=zp, ids=["a"], images=[np.zeros((16, 16, 3), np.uint8)], extractor=matcher
    )

    assert matcher.n_extract == 1


def test_from_pointcloud_refuses_misaligned_result(tmp_path, stub_matcher):
    result = _mock_result(np.zeros((2, 8, 8, 3), np.float32))
    result.extrinsics = result.extrinsics[:1]

    with pytest.raises(ValueError, match="align"):
        CameraLocalizer.from_pointcloud(
            result, zarr_path=tmp_path / "pc.zarr", ids=["a", "b"], extractor=stub_matcher(_three_keypoints)
        )


def test_load_index_refuses_geometry_short_of_db(tmp_path, stub_matcher):
    wp = np.zeros((2, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    loc = CameraLocalizer(
        wp,
        np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        _blank_frames(2),
        ["a", "b"],
        extractor=stub_matcher(_three_keypoints),
    )
    loc.save_index(zp, "stub")
    loc.update_index(_blank_frames(1), ["c"], zp, "stub")

    with pytest.raises(ValueError, match="3 frames"):
        CameraLocalizer.load_index(zp, "stub", wp, loc.extrinsics, extractor=stub_matcher(_three_keypoints))


def test_from_pointcloud_refuses_ids_length_mismatch(tmp_path, stub_matcher):
    result = _mock_result(np.zeros((2, 8, 8, 3), np.float32))

    with pytest.raises(ValueError, match="align"):
        CameraLocalizer.from_pointcloud(
            result, zarr_path=tmp_path / "pc.zarr", ids=["a"], extractor=stub_matcher(_three_keypoints)
        )


def test_localized_append_after_crash_stays_aligned(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    zp = tmp_path / "pc.zarr"
    loc = CameraLocalizer(
        wp, np.eye(4, dtype=np.float32)[None], _blank_frames(1), ["a"], extractor=stub_matcher(_three_keypoints)
    )
    loc.save_index(zp, "stub")
    loc.add_localized_frame("q1", np.eye(4), zarr_path=zp, extractor_name="stub")

    # A crashed append: extrinsics grew, image_paths never did
    group = zarr.open(str(zp), mode="a")["local_features/stub/localized"]
    localizer_mod._append_rows(group, "extrinsics", np.full((1, 4, 4), 9.0))
    pose = np.eye(4)
    pose[0, 3] = 2.0
    loc.add_localized_frame("q2", pose, zarr_path=zp, extractor_name="stub")

    group = zarr.open(str(zp), mode="r")["local_features/stub/localized"]
    assert group.attrs["image_paths"] == ["q1", "q2"]
    assert group["extrinsics"].shape == (2, 4, 4)
    np.testing.assert_allclose(group["extrinsics"][1], pose)


def test_write_csr_row_chunks_round_trip(tmp_path):
    feats = [
        LocalFeatures(
            keypoints=torch.rand(n, 2),
            descriptors=torch.rand(n, 4),
            keypoints_normalized=torch.rand(n, 2),
            image_size=(8, 8),
        )
        for n in (5, 7, 3)
    ]
    group = zarr.open_group(str(tmp_path / "db.zarr"), mode="w")

    localizer_mod._write_csr(group, feats, chunk_rows=4)

    # Every per-keypoint array splits into chunk_rows-row chunks
    for name in ("keypoints", "descriptors", "keypoints_normalized"):
        assert group[name].chunks[0] == 4

    # Round trip returns the written features exactly
    got = localizer_mod._features_from_csr(group, [(8, 8)] * 3)

    for want, have in zip(feats, got):
        np.testing.assert_array_equal(have.keypoints.numpy(), want.keypoints.numpy())
        np.testing.assert_array_equal(have.descriptors.numpy(), want.descriptors.numpy())
        np.testing.assert_array_equal(have.keypoints_normalized.numpy(), want.keypoints_normalized.numpy())


def test_write_csr_small_db_is_one_chunk(tmp_path):
    feats = [LocalFeatures(keypoints=torch.rand(5, 2), descriptors=torch.rand(5, 4), image_size=(8, 8))]
    group = zarr.open_group(str(tmp_path / "db.zarr"), mode="w")

    localizer_mod._write_csr(group, feats)

    assert group["descriptors"].chunks == (5, 4)
