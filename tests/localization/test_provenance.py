"""Provenance attrs round-trip through the zarr feature cache."""
import numpy as np
import pytest
import torch
import zarr

from collab_splats.localization.extractors import LocalFeatures
from collab_splats.localization.localizer import CameraLocalizer


class _FakeExtractor:
    """Deterministic extractor: 8 fixed keypoints, 4-dim descriptors."""

    def extract(self, rgb):
        k = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        d = torch.ones(8, 4)
        return LocalFeatures(keypoints=k, descriptors=d, scores=None)

    def match(self, a, b, hw):
        return torch.zeros((0, 2), dtype=torch.int64)


def _make_localizer(tmp_path, n_frames=2):
    """Build a localizer from synthetic images on disk (no GPU)."""
    import cv2

    paths = []
    for i in range(n_frames):
        p = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 128, dtype=np.uint8))
        paths.append(p)
    pts3d = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    extr = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    intr = np.tile(np.array([[60, 0, 32], [0, 60, 24], [0, 0, 1]], np.float32), (n_frames, 1, 1))
    return CameraLocalizer(pts3d, extr, intr, paths, extractor=_FakeExtractor()), pts3d, extr, intr


def test_save_index_writes_build_attrs(tmp_path):
    loc, *_ = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk", attrs={"backbone": "vggtx", "ba": True, "lc": False,
                                      "built_at": "2026-07-14T00:00:00"})
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert group.attrs["backbone"] == "vggtx"
    assert group.attrs["ba"] is True
    assert group.attrs["extractor"] == "disk"


def test_save_index_without_attrs_still_stamps_extractor(tmp_path):
    loc, *_ = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert group.attrs["extractor"] == "disk"


def test_add_localized_frame_records_provenance(tmp_path):
    loc, pts3d, extr, intr = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")

    feats = _FakeExtractor().extract(None)
    pose = np.eye(4, dtype=np.float32)
    prov = {"video_ref": "2024_02_06-session_0001/rgb_1/cam.mp4",
            "session": "2024_02_06-session_0001", "camera": "rgb_1", "frame_idx": 42}
    loc.add_localized_frame(tmp_path / "query.jpg", pose, intr[0], feats,
                            zarr_path=zp, extractor_name="disk", provenance=prov)

    lg = zarr.open(str(zp), mode="r")["local_features/disk/localized"]
    assert lg.attrs["provenance"][0]["frame_idx"] == 42
    assert lg.attrs["provenance"][0]["camera"] == "rgb_1"


def test_provenance_list_grows_per_frame(tmp_path):
    loc, pts3d, extr, intr = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")
    feats = _FakeExtractor().extract(None)
    pose = np.eye(4, dtype=np.float32)
    for i in range(2):
        loc.add_localized_frame(tmp_path / f"q{i}.jpg", pose, intr[0], feats,
                                zarr_path=zp, extractor_name="disk",
                                provenance={"frame_idx": i})
    lg = zarr.open(str(zp), mode="r")["local_features/disk/localized"]
    assert len(lg.attrs["provenance"]) == 2
