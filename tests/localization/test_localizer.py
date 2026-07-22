import logging

import numpy as np
import pytest
import torch

from collab_splats.localization import (
    BaseLocalExtractor,
    BaseRetrievalExtractor,
    DinoSaladExtractor,
    DiskExtractor,
    LocalFeatures,
    LocalizationResult,
    XFeatExtractor,
)
from collab_splats.localization.localizer import sample_world_points


def test_submodules_have_logger():
    from collab_splats.localization import extractors, localizer, retrieval, viz

    for mod in (extractors, localizer, retrieval, viz):
        assert hasattr(mod, "logger")
        assert isinstance(mod.logger, logging.Logger)


def test_registry_get_dino_salad():
    cls = BaseRetrievalExtractor.get("dino-salad")
    assert cls is DinoSaladExtractor


def test_registry_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        BaseRetrievalExtractor.get("nonexistent-model")


def test_base_local_extractor_registry():
    assert BaseLocalExtractor.get("disk") is DiskExtractor
    assert BaseLocalExtractor.get("xfeat") is XFeatExtractor


@pytest.mark.slow
def test_disk_extractor_returns_keypoints_and_descriptors():
    """Requires network access to download DISK weights (~4 MB)."""
    from collab_splats.localization import DiskExtractor

    extractor = DiskExtractor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = extractor.extract(image)
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.ndim == 2 and feats.keypoints.shape[1] == 2
    assert feats.descriptors.ndim == 2 and feats.descriptors.shape[1] == 128
    assert len(feats.keypoints) == len(feats.descriptors)
    assert len(feats.keypoints) > 0
    assert feats.scores is None


@pytest.mark.slow
def test_disk_extractor_match_returns_index_pairs():
    from collab_splats.localization import DiskExtractor

    extractor = DiskExtractor()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = extractor.extract(img)
    matches = extractor.match(feats, feats, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2
    diag = (matches[:, 0] == matches[:, 1]).sum()
    assert diag > 0


@pytest.mark.slow
def test_xfeat_extractor_returns_keypoints_and_descriptors():
    from collab_splats.localization import XFeatExtractor

    extractor = XFeatExtractor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = extractor.extract(image)
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.ndim == 2 and feats.keypoints.shape[1] == 2
    assert feats.descriptors.ndim == 2 and feats.descriptors.shape[1] == 64
    assert len(feats.keypoints) == len(feats.descriptors)
    assert len(feats.keypoints) > 0
    assert feats.scores is not None
    assert feats.scores.ndim == 1 and len(feats.scores) == len(feats.keypoints)


@pytest.mark.slow
def test_xfeat_extractor_match_returns_index_pairs():
    from collab_splats.localization import XFeatExtractor

    extractor = XFeatExtractor()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = extractor.extract(img)
    matches = extractor.match(feats, feats, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2


def test_build_frame_assignments_assigns_visible_points():
    """Identity camera: projected 3D points → keypoints at exact positions."""
    from collab_splats.localization.localizer import _build_frame_assignments

    # Two points in front of identity camera at z=5
    pts3d = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float32)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None]  # (1, 4, 4) identity
    intrinsics = K[None]  # (1, 3, 3)

    # Expected projections: [0,0,5] → [320,240], [1,0,5] → [420,240]
    kpts = torch.tensor([[320.0, 240.0], [420.0, 240.0]])
    assignments = _build_frame_assignments(pts3d, extrinsics, intrinsics, [kpts], image_hw=(480, 640), radius=2.0)

    assert len(assignments) == 1
    assert assignments[0][0] == 0  # kpt 0 → pt3d 0
    assert assignments[0][1] == 1  # kpt 1 → pt3d 1


def test_build_frame_assignments_ignores_points_behind_camera():
    from collab_splats.localization.localizer import _build_frame_assignments

    # One point in front, one behind
    pts3d = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None]
    intrinsics = K[None]
    kpts = torch.tensor([[320.0, 240.0], [320.0, 241.0]])

    assignments = _build_frame_assignments(pts3d, extrinsics, intrinsics, [kpts], image_hw=(480, 640), radius=2.0)

    assert 0 in assignments[0]  # front point assigned
    assert 1 not in assignments[0]  # behind-camera point not assigned


def test_build_frame_assignments_respects_radius():
    from collab_splats.localization.localizer import _build_frame_assignments

    pts3d = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)  # projects to [320, 240]
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None]
    intrinsics = K[None]

    # Keypoint 50px away from projection — outside radius=10
    kpts = torch.tensor([[370.0, 240.0]])
    assignments = _build_frame_assignments(pts3d, extrinsics, intrinsics, [kpts], image_hw=(480, 640), radius=10.0)
    assert 0 not in assignments[0]


def _make_synthetic_scene(n_pts: int = 30, n_frames: int = 4):
    """Synthetic scene: n_pts points in front of n_frames cameras."""
    rng = np.random.default_rng(0)
    pts3d = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
    pts3d[:, 2] += 5.0  # push in front of cameras

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (n_frames, 1, 1))  # (N, 3, 3)

    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    for i in range(n_frames):
        extrinsics[i, 0, 3] = i * 0.1  # slight lateral shift

    return pts3d, extrinsics, intrinsics


class _MockExtractor:
    """Extractor that returns exact projected 3D positions as keypoints.

    descriptor[j] = one-hot-ish encoding of pt3d index j — matching is exact.
    """

    def __init__(self, pts3d, extrinsics, intrinsics):
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._intrinsics = intrinsics
        self._call = 0
        n = len(pts3d)
        # Descriptors: identity matrix padded/truncated to 128 dims
        self._descs = torch.zeros(n, 128)
        for j in range(min(n, 128)):
            self._descs[j, j] = 1.0

    def extract(self, image):
        i = self._call % len(self._extrinsics)
        self._call += 1
        R = self._extrinsics[i, :3, :3]
        t = self._extrinsics[i, :3, 3]
        K = self._intrinsics[i]
        pts_cam = self._pts3d @ R.T + t
        visible = pts_cam[:, 2] > 0
        pts_proj = pts_cam[visible] @ K.T
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
        kpts = torch.from_numpy(pts_proj).float()
        descs = self._descs[visible]
        return LocalFeatures(keypoints=kpts, descriptors=descs)

    def match(self, query: LocalFeatures, db: LocalFeatures, image_hw=None):
        sim = query.descriptors @ db.descriptors.T  # (N, M)
        best = sim.argmax(dim=1)
        valid = sim.max(dim=1).values > 0.5
        idx_q = torch.where(valid)[0]
        idx_db = best[valid]
        if len(idx_q) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.stack([idx_q, idx_db], dim=1)


def test_camera_localizer_from_feedforward_classmethod():
    """from_feedforward classmethod constructs CameraLocalizer correctly."""
    from unittest.mock import MagicMock

    from collab_splats.localization import CameraLocalizer

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    result = MagicMock()
    result.points = pts3d
    result.extrinsics = extrinsics
    result.intrinsics = intrinsics
    result._zarr_path = None

    # Caller builds (images, ids) aligned to result; consumed on cache miss
    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]
    result.image_paths = ids

    extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
    loc = CameraLocalizer.from_feedforward(result, images=images, ids=ids, extractor=extractor)
    assert loc is not None


def test_camera_localizer_recovers_known_pose():
    """CameraLocalizer should recover the identity pose for a camera at extrinsics[0]."""
    from collab_splats.localization import CameraLocalizer

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    K = intrinsics[0]

    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
    loc = CameraLocalizer(pts3d, extrinsics, intrinsics, images=images, ids=ids, extractor=extractor)

    # Query = camera 0 (identity extrinsic)
    query_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = loc.localize(query_image, K)

    assert isinstance(result, LocalizationResult)
    assert result.pose is not None, "localize() returned None pose — pycolmap failed"
    assert result.pose.shape == (4, 4)
    np.testing.assert_allclose(result.pose[:3, :3], extrinsics[0, :3, :3], atol=0.05)
    np.testing.assert_allclose(result.pose[:3, 3], extrinsics[0, :3, 3], atol=0.05)
    assert result.n_inliers > 0
    assert result.inlier_mask is not None
    assert result.inlier_mask.dtype == bool
    assert len(result.inlier_mask) == result.n_correspondences


def test_localization_result_fields_on_success():
    from collab_splats.localization import CameraLocalizer, LocalizationResult

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    K = intrinsics[0]

    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
    loc = CameraLocalizer(pts3d, extrinsics, intrinsics, images=images, ids=ids, extractor=extractor)
    result = loc.localize(np.zeros((480, 640, 3), dtype=np.uint8), K)

    assert isinstance(result, LocalizationResult)
    assert result.pose is not None
    assert result.inlier_mask is not None
    assert result.inlier_mask.dtype == bool
    assert len(result.inlier_mask) == result.n_correspondences
    assert result.n_inliers > 0
    assert result.pts2d_ref is not None
    assert result.pts2d_ref.shape == result.pts2d.shape
    assert result.ref_frame_indices is not None
    assert len(result.ref_frame_indices) == result.n_correspondences


def test_camera_localizer_calls_progress_callback():
    """progress_callback(i, total) called once per reference frame, 0-indexed."""
    from collab_splats.localization import CameraLocalizer

    pts3d, extrinsics, intrinsics = _make_synthetic_scene()
    extractor = _MockExtractor(pts3d, extrinsics, intrinsics)
    calls = []

    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    CameraLocalizer(
        pts3d,
        extrinsics,
        intrinsics,
        images=images,
        ids=ids,
        extractor=extractor,
        progress_callback=lambda i, total: calls.append((i, total)),
    )

    total = len(extrinsics)
    assert len(calls) == total
    assert calls[0] == (0, total)
    assert calls[-1] == (total - 1, total)


def test_sample_world_points_bilinear_and_invalid():
    """Exact-pixel and bilinear samples return grid values; NaN cells are invalid."""
    # 4x4 grid whose world point at (row r, col c) is (c, r, 1)
    H = W = 4
    wp = np.stack(list(np.meshgrid(np.arange(W), np.arange(H))) + [np.ones((H, W))], axis=-1).astype(np.float32)
    wp[0, 0] = np.nan  # unmapped pixel

    px = np.array([[2.0, 1.0], [1.5, 2.5], [0.0, 0.0]], dtype=np.float32)  # xy
    pts, valid = sample_world_points(wp, px)
    assert pts.shape == (3, 3) and valid.dtype == bool
    np.testing.assert_allclose(pts[0], [2.0, 1.0, 1.0], atol=1e-5)   # exact pixel
    np.testing.assert_allclose(pts[1], [1.5, 2.5, 1.0], atol=1e-5)   # bilinear midpoint
    assert not valid[2] and valid[0] and valid[1]                     # NaN cell dropped


def test_sample_world_points_out_of_bounds():
    """Pixels outside the image bounds are marked invalid."""
    H = W = 4
    wp = np.stack(list(np.meshgrid(np.arange(W), np.arange(H))) + [np.ones((H, W))], axis=-1).astype(np.float32)

    px = np.array([[10.0, 1.0]], dtype=np.float32)  # x beyond W-1
    _, valid = sample_world_points(wp, px)
    assert not valid[0]
