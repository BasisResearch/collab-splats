"""Reference frames stay index-aligned once localized frames join the reference set.

LocalizationResult.ref_frame_indices indexes the merged reconstruction+localized frame list,
so image_paths / frame_sources / extrinsics must all agree in length. _extrinsics itself stays
reconstruction-only (assignment building depends on that) — the extrinsics property joins them.
"""

import cv2
import numpy as np
import torch

from collab_splats.localization.extractors import LocalFeatures
from collab_splats.localization.localizer import CameraLocalizer

########
# Fakes
########


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
    paths = []
    for i in range(n_frames):
        p = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 128, dtype=np.uint8))
        paths.append(p)
    pts3d = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    extr = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    intr = np.tile(np.array([[60, 0, 32], [0, 60, 24], [0, 0, 1]], np.float32), (n_frames, 1, 1))
    return CameraLocalizer(pts3d, extr, intr, paths, extractor=_FakeExtractor()), pts3d, extr, intr


def _distinct_pose(tx: float) -> np.ndarray:
    """A pose distinguishable from the identity reconstruction extrinsics."""
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = tx
    return pose


########
# Tests
########


def test_extrinsics_matches_paths_after_add_localized_frame(tmp_path):
    # 2 reconstruction frames + 1 localized → every reference view list must be length 3
    loc, _, _, intr = _make_localizer(tmp_path, n_frames=2)
    pose = _distinct_pose(1.5)
    loc.add_localized_frame(tmp_path / "query.jpg", pose, intr[0], _FakeExtractor().extract(None))

    assert len(loc.image_paths) == len(loc.frame_sources) == loc.extrinsics.shape[0] == 3
    assert loc.frame_sources == ["reconstruction", "reconstruction", "localized"]
    np.testing.assert_allclose(loc.extrinsics[2], pose)


def test_extrinsics_property_is_reconstruction_only_before_any_append(tmp_path):
    loc, _, extr, _ = _make_localizer(tmp_path, n_frames=2)
    assert loc.extrinsics.shape == (2, 4, 4)
    np.testing.assert_allclose(loc.extrinsics, extr)


def test_private_extrinsics_stays_reconstruction_only(tmp_path):
    # Assignment-building code indexes _extrinsics as reconstruction frames; keep it unpolluted
    loc, _, _, intr = _make_localizer(tmp_path, n_frames=2)
    loc.add_localized_frame(tmp_path / "query.jpg", _distinct_pose(1.5), intr[0], _FakeExtractor().extract(None))
    assert loc._extrinsics.shape == (2, 4, 4)


def test_load_index_round_trip_preserves_localized_pose(tmp_path):
    # save → append (persisted) → reload: the localized pose must survive and stay aligned
    loc, pts3d, extr, intr = _make_localizer(tmp_path, n_frames=2)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")
    pose = _distinct_pose(2.5)
    loc.add_localized_frame(
        tmp_path / "query.jpg", pose, intr[0], _FakeExtractor().extract(None), zarr_path=zp, extractor_name="disk"
    )

    reloaded = CameraLocalizer.load_index(zp, "disk", pts3d, extr, intr, extractor=_FakeExtractor())

    assert len(reloaded.image_paths) == len(reloaded.frame_sources) == reloaded.extrinsics.shape[0] == 3
    assert reloaded.frame_sources == ["reconstruction", "reconstruction", "localized"]
    np.testing.assert_allclose(reloaded.extrinsics[2], pose, atol=1e-6)
    assert reloaded._extrinsics.shape == (2, 4, 4)
