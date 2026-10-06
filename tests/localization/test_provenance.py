"""Provenance attrs round-trip through the zarr feature cache."""

import numpy as np
import zarr

from collab_splats.localization.localizer import CameraLocalizer


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


def test_save_index_writes_build_attrs(tmp_path, stub_matcher):
    loc, *_ = _make_localizer(tmp_path, stub_matcher)
    zp = tmp_path / "pointcloud.zarr"
    loc.save_index(zp, "disk", attrs={"backbone": "vggtx", "ba": True, "lc": False, "built_at": "2026-07-14T00:00:00"})
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert group.attrs["backbone"] == "vggtx"
    assert group.attrs["ba"] is True
    assert group.attrs["extractor"] == "disk"


def test_save_index_without_attrs_still_stamps_extractor(tmp_path, stub_matcher):
    loc, *_ = _make_localizer(tmp_path, stub_matcher)
    zp = tmp_path / "pointcloud.zarr"
    loc.save_index(zp, "disk")
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert group.attrs["extractor"] == "disk"


def test_save_index_rebuild_replaces_stale_attrs(tmp_path, stub_matcher):
    # Rebuild without attrs must not inherit provenance from a previous build
    loc, *_ = _make_localizer(tmp_path, stub_matcher)
    zp = tmp_path / "pointcloud.zarr"
    loc.save_index(zp, "disk", attrs={"backbone": "vggtx", "ba": True})
    loc.save_index(zp, "disk")
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert "ba" not in group.attrs
    assert "backbone" not in group.attrs
    assert group.attrs["extractor"] == "disk"


def test_add_localized_frame_records_provenance(tmp_path, stub_matcher):
    loc, wp, extr = _make_localizer(tmp_path, stub_matcher)
    zp = tmp_path / "pointcloud.zarr"
    loc.save_index(zp, "disk")

    pose = np.eye(4, dtype=np.float32)
    prov = {
        "video_ref": "2024_02_06-office-cam_01",
        "scene": "2024_02_06-office-cam_01",
        "frame_idx": 42,
    }
    loc.add_localized_frame(tmp_path / "query.jpg", pose, zarr_path=zp, extractor_name="disk", provenance=prov)

    lg = zarr.open(str(zp), mode="r")["local_features/disk/localized"]
    assert lg.attrs["provenance"][0]["frame_idx"] == 42
    assert lg.attrs["provenance"][0]["scene"] == "2024_02_06-office-cam_01"


def test_provenance_list_grows_per_frame(tmp_path, stub_matcher):
    loc, wp, extr = _make_localizer(tmp_path, stub_matcher)
    zp = tmp_path / "pointcloud.zarr"
    loc.save_index(zp, "disk")
    pose = np.eye(4, dtype=np.float32)
    for i in range(2):
        loc.add_localized_frame(
            tmp_path / f"q{i}.jpg",
            pose,
            zarr_path=zp,
            extractor_name="disk",
            provenance={"frame_idx": i},
        )
    lg = zarr.open(str(zp), mode="r")["local_features/disk/localized"]
    assert len(lg.attrs["provenance"]) == 2
