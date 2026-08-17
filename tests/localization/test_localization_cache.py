"""Tests for localization feature + track cache (zarr-backed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from collab_splats.localization import (
    CameraLocalizer,
    LocalFeatures,
    LocalizationResult,
)
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.utils.image import open_image

# ── Shared fixtures ──────────────────────────────────────────────────────────


def _make_features(n_kpts: int = 10, desc_dim: int = 128) -> LocalFeatures:
    """Synthetic LocalFeatures for mocking — no GPU needed."""
    return LocalFeatures(
        keypoints=torch.rand(n_kpts, 2) * 60,
        descriptors=torch.rand(n_kpts, desc_dim),
        scores=None,
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
    """Write tiny solid-colour JPEG images to tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = np.full((size, size, 3), fill_value=80 + i * 40, dtype=np.uint8)
        p = tmp_path / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _build_localizer_with_mock(world_points, extrinsics, image_paths, desc_dim=128):
    """Build CameraLocalizer backed by a mock extractor (no GPU)."""
    mock_ext = MagicMock()
    mock_ext.extract.return_value = _make_features(desc_dim=desc_dim)
    mock_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)
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
    """Create an empty feedforward.zarr store and return its path."""
    import zarr

    zarr_path = tmp_path / "feedforward.zarr"
    zarr.open(str(zarr_path), mode="w")
    return zarr_path


# ── Task 1 test ──────────────────────────────────────────────────────────────


def test_load_zarr_sets_zarr_path(tmp_path):
    """FeedforwardResult.load_zarr should set _zarr_path on the result."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    result = FeedforwardResult(
        points=pts3d,
        colors=np.zeros((len(pts3d), 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=image_paths,
        original_coords=np.zeros((3, 6), dtype=np.float32),
        model_width=64,
        model_height=64,
    )
    zarr_path = tmp_path / "test.zarr"
    result.save_zarr(zarr_path)

    loaded = FeedforwardResult.load_zarr(zarr_path)
    assert hasattr(loaded, "_zarr_path")
    assert loaded._zarr_path == zarr_path


# ── Task 2 tests ─────────────────────────────────────────────────────────────


def test_localize_populates_query_features(tmp_path):
    """localize() should always set query_features on the result."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    query_img = np.zeros((64, 64, 3), dtype=np.uint8)
    query_K = intrinsics[0]
    result = localizer.localize(query_img, query_K)

    assert result.query_features is not None
    assert hasattr(result.query_features, "keypoints")
    assert hasattr(result.query_features, "descriptors")


def test_camera_localizer_stores_image_paths_and_sources(tmp_path):
    """CameraLocalizer should maintain _image_paths and _frame_sources after build."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    assert len(localizer._image_paths) == 3
    assert len(localizer._frame_sources) == 3
    assert all(s == "reconstruction" for s in localizer._frame_sources)
    assert localizer.frame_sources == ["reconstruction", "reconstruction", "reconstruction"]


def test_save_index_creates_reconstruction_group(tmp_path):
    """save_index writes local_features/{name}/reconstruction/ with expected arrays."""
    import zarr

    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

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


def test_load_index_round_trip(tmp_path):
    """save_index + load_index: loaded localizer has same frame count and sources."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, mock_ext = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path,
        extractor_name="disk",
        world_points=world_points,
        extrinsics=extrinsics,
        extractor=MagicMock(),  # duck-typed descriptor extractor — the LocalMatcher default is pairwise and needs ref images
    )

    assert len(loaded._frame_features) == 3
    assert len(loaded._frame_sources) == 3
    assert all(s == "reconstruction" for s in loaded._frame_sources)
    assert loaded.frame_sources == ["reconstruction", "reconstruction", "reconstruction"]
    # Keypoint count preserved
    assert loaded._frame_features[0].keypoints.shape[1] == 2
    # _image_paths holds the stable string ids the localizer was built with, round-tripped
    ids = [Path(p).name for p in image_paths]
    assert loaded.image_paths == ids


def test_load_index_round_trips_scales(tmp_path):
    """Optional per-keypoint scales (dense XFeat*) survive save_index → load_index."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    # Stamp synthetic scales onto the cached frame features (dense XFeat* path)
    for f in localizer._frame_features:
        f.scales = torch.rand(len(f.keypoints))

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "xfeat-star")

    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path,
        extractor_name="xfeat-star",
        world_points=world_points,
        extrinsics=extrinsics,
        extractor=MagicMock(),  # duck-typed descriptor extractor — the LocalMatcher default is pairwise and needs ref images
    )

    for orig, rt in zip(localizer._frame_features, loaded._frame_features):
        assert rt.scales is not None
        np.testing.assert_allclose(rt.scales.numpy(), orig.scales.numpy(), rtol=1e-6)


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


# ── Task 5 tests ─────────────────────────────────────────────────────────────


def _make_ff_result(world_points, extrinsics, image_paths):
    """Minimal FeedforwardResult mock for testing from_feedforward."""
    result = MagicMock()
    result.world_points = world_points
    result.extrinsics = extrinsics
    result.image_paths = image_paths
    result._zarr_path = None
    return result


def test_from_feedforward_cache_miss_builds_and_saves(tmp_path):
    """Cache miss: from_feedforward runs GPU inference and saves to zarr."""
    import zarr

    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    zarr_path = _empty_zarr(tmp_path)

    mock_ext = MagicMock()
    mock_ext.extract.return_value = _make_features()
    mock_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)

    result = _make_ff_result(world_points, extrinsics, image_paths)
    result._zarr_path = zarr_path

    # Caller builds (images, ids) aligned to result; consumed only on cache miss
    images = [np.asarray(open_image(p).convert("RGB")) for p in image_paths]
    ids = [Path(p).name for p in image_paths]
    localizer = CameraLocalizer.from_feedforward(
        result, images=images, ids=ids, extractor=mock_ext, extractor_name="disk"
    )

    assert mock_ext.extract.call_count == 3  # GPU ran for each frame
    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/reconstruction" in store


def test_from_feedforward_cache_hit_skips_extraction(tmp_path):
    """Cache hit: from_feedforward loads from zarr, does not call extractor.extract."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    zarr_path = _empty_zarr(tmp_path)

    # Populate cache via first build
    build_ext = MagicMock()
    build_ext.extract.return_value = _make_features()
    build_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)
    result = _make_ff_result(world_points, extrinsics, image_paths)
    images = [np.asarray(open_image(p).convert("RGB")) for p in image_paths]
    ids = [Path(p).name for p in image_paths]
    CameraLocalizer.from_feedforward(
        result, images=images, ids=ids, extractor=build_ext, extractor_name="disk", zarr_path=zarr_path
    )

    # Second build — should hit cache; images/ids are ignored on a hit
    load_ext = MagicMock()
    load_ext.extract.return_value = _make_features()
    loaded = CameraLocalizer.from_feedforward(
        result, images=None, ids=None, extractor=load_ext, extractor_name="disk", zarr_path=zarr_path
    )

    assert load_ext.extract.call_count == 0  # no GPU inference on cache hit
    assert len(loaded._frame_features) == 3


# ── Task 6 tests ─────────────────────────────────────────────────────────────


def test_update_index_appends_new_frames(tmp_path):
    """update_index extracts + appends new reconstruction frames to zarr."""
    import zarr

    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    # Create 2 new frames as in-memory RGB arrays + string ids (no image IO)
    new_images = [
        np.full((64, 64, 3), 200, dtype=np.uint8),
        np.full((64, 64, 3), 220, dtype=np.uint8),
    ]
    new_ids = ["frame_098.jpg", "frame_099.jpg"]
    new_ext = MagicMock()
    new_ext.extract.return_value = _make_features()
    new_ext.match.return_value = torch.zeros((0, 2), dtype=torch.long)
    localizer._extractor = new_ext

    localizer.update_index(
        new_images=new_images,
        new_ids=new_ids,
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    assert new_ext.extract.call_count == 2
    assert len(localizer._frame_features) == 5  # 3 + 2
    assert localizer._frame_sources.count("reconstruction") == 5
    assert localizer.image_paths[-1] == "frame_099.jpg"

    # Verify zarr updated
    store = zarr.open(str(zarr_path), mode="r")
    grp = store["local_features/disk/reconstruction"]
    assert grp["frame_offsets"].shape == (6,)  # 5+1
    assert len(grp.attrs["image_paths"]) == 5


def test_add_localized_frame_extends_index_in_memory(tmp_path):
    """add_localized_frame appends a localized frame to in-memory reference set."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path, n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    assert len(localizer._frame_features) == 3

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0, 3] = 2.0
    new_intr = intrinsics[0].copy()
    new_feats = _make_features()
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(
        image_path=new_path,
        pose=new_pose,
        intrinsics=new_intr,
        features=new_feats,
    )

    assert len(localizer._frame_features) == 4
    assert localizer._frame_sources[-1] == "localized"
    assert localizer._image_paths[-1] == str(new_path)


def test_add_localized_frame_persists_to_zarr(tmp_path):
    """add_localized_frame with zarr_path writes to localized/ group."""
    import zarr

    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    new_pose = np.eye(4, dtype=np.float32)
    new_intr = intrinsics[0].copy()
    new_feats = _make_features()
    new_path = tmp_path / "query.jpg"

    localizer.add_localized_frame(
        image_path=new_path,
        pose=new_pose,
        intrinsics=new_intr,
        features=new_feats,
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    store = zarr.open(str(zarr_path), mode="r")
    assert "local_features/disk/localized" in store
    loc_grp = store["local_features/disk/localized"]
    assert loc_grp["extrinsics"].shape == (1, 4, 4)
    assert loc_grp["intrinsics"].shape == (1, 3, 3)
    assert len(loc_grp.attrs["image_paths"]) == 1


def test_add_localized_frame_duplicate_skipped(tmp_path):
    """Duplicate image_path is silently skipped — no double-add."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=2)
    image_paths = _make_image_files(tmp_path, n=2)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    new_pose = np.eye(4, dtype=np.float32)
    new_path = tmp_path / "query.jpg"
    new_feats = _make_features()

    localizer.add_localized_frame(new_path, new_pose, intrinsics[0], new_feats)
    localizer.add_localized_frame(new_path, new_pose, intrinsics[0], new_feats)  # duplicate

    assert len(localizer._frame_features) == 3  # 2 rec + 1 loc, not 4


def test_load_index_includes_localized_frames(tmp_path):
    """After add + reload, localized frame is in the loaded index."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    new_pose = np.eye(4, dtype=np.float32)
    new_pose[0, 3] = 1.5
    new_path = tmp_path / "query.jpg"
    localizer.add_localized_frame(
        new_path, new_pose, intrinsics[0], _make_features(), zarr_path=zarr_path, extractor_name="disk"
    )

    # Reload from zarr
    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path,
        extractor_name="disk",
        world_points=world_points,
        extrinsics=extrinsics,
        extractor=MagicMock(),  # duck-typed descriptor extractor — the LocalMatcher default is pairwise and needs ref images
    )

    assert len(loaded._frame_features) == 4
    assert loaded._frame_sources == ["reconstruction"] * 3 + ["localized"]
    assert loaded._image_paths[-1] == str(new_path)


# ── Task 8 tests ─────────────────────────────────────────────────────────────


def test_clear_localized_frames_removes_zarr_group(tmp_path):
    """clear_localized_frames deletes localized/ group; reconstruction/ untouched."""
    import zarr

    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")

    # Add a localized frame
    localizer.add_localized_frame(
        tmp_path / "q.jpg",
        np.eye(4, dtype=np.float32),
        intrinsics[0],
        _make_features(),
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


def test_load_index_after_clear_has_only_reconstruction(tmp_path):
    """After clear_localized_frames, load_index returns only reconstruction frames."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=3)
    image_paths = _make_image_files(tmp_path / "imgs", n=3)
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")
    localizer.add_localized_frame(
        tmp_path / "q.jpg",
        np.eye(4, dtype=np.float32),
        intrinsics[0],
        _make_features(),
        zarr_path=zarr_path,
        extractor_name="disk",
    )

    CameraLocalizer.clear_localized_frames(zarr_path, "disk")

    loaded = CameraLocalizer.load_index(
        zarr_path=zarr_path,
        extractor_name="disk",
        world_points=world_points,
        extrinsics=extrinsics,
        extractor=MagicMock(),  # duck-typed descriptor extractor — the LocalMatcher default is pairwise and needs ref images
    )
    assert len(loaded._frame_features) == 3
    assert all(s == "reconstruction" for s in loaded._frame_sources)


# ── Task 2 (localization module) tests: shared cache reader ──────────────────


def test_load_reconstruction_features_roundtrip(tmp_path):
    """Module-level cache reader returns exactly what save_index wrote."""
    from collab_splats.localization.localizer import load_reconstruction_features

    pts3d, world_points, extrinsics, intrinsics = _make_scene(n_frames=2)
    image_paths = _make_image_files(tmp_path / "imgs", n=2)
    feats = [_make_features(n_kpts=5), _make_features(n_kpts=8)]

    mock_ext = MagicMock()
    mock_ext.extract.side_effect = feats
    images = [np.asarray(open_image(p).convert("RGB")) for p in image_paths]
    ids = [Path(p).name for p in image_paths]
    localizer = CameraLocalizer(
        world_points=world_points,
        extrinsics=extrinsics,
        images=images,
        ids=ids,
        extractor=mock_ext,
    )

    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "xfeat")

    out_feats, out_ids, hw = load_reconstruction_features(zarr_path, "xfeat")
    assert out_ids == ids and hw == (64, 64)
    assert [len(f.keypoints) for f in out_feats] == [5, 8]
    np.testing.assert_allclose(out_feats[1].keypoints.numpy(), feats[1].keypoints.numpy())
    np.testing.assert_allclose(out_feats[0].descriptors.numpy(), feats[0].descriptors.numpy())


def test_load_reconstruction_features_missing_raises(tmp_path):
    """Missing cache raises KeyError naming the extractor."""
    from collab_splats.localization.localizer import load_reconstruction_features

    zarr_path = _empty_zarr(tmp_path)
    with pytest.raises(KeyError, match="disk"):
        load_reconstruction_features(zarr_path, "disk")
