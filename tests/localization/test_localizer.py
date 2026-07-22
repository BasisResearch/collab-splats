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
from collab_splats.localization.extractors import MatchResult
from collab_splats.localization.localizer import CameraLocalizer, sample_world_points


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


def _make_synthetic_scene(n_pts: int = 30, n_frames: int = 4, H: int = 480, W: int = 640):
    """Planar synthetic scene: n_pts points at z=5 plus dense per-frame world maps.

    A fronto-parallel plane keeps the dense world map linear in pixel coords, so
    bilinear sampling in sample_world_points is exact at any subpixel location.
    """
    rng = np.random.default_rng(0)
    pts3d = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
    pts3d[:, 2] = 5.0  # planar scene at z=5

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    for i in range(n_frames):
        extrinsics[i, 0, 3] = i * 0.1  # slight lateral shift

    # Dense per-frame world maps: backproject each pixel at plane depth z=5, undo pose
    # (R=I → p_world = p_cam - t)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * 5.0, (ys - K[1, 2]) / K[1, 1] * 5.0, np.full(xs.shape, 5.0)], -1)
    world_points = np.stack([cam - extrinsics[i, :3, 3] for i in range(n_frames)]).astype(np.float32)

    return pts3d, world_points, extrinsics, K


class _MockExtractor:
    """Extractor that returns exact projected 3D positions as keypoints.

    descriptor[j] = one-hot-ish encoding of pt3d index j — matching is exact,
    and match() returns pixel-pair MatchResults per the new contract.
    """

    def __init__(self, pts3d, extrinsics, K):
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._K = K
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
        pts_cam = self._pts3d @ R.T + t
        visible = pts_cam[:, 2] > 0
        pts_proj = pts_cam[visible] @ self._K.T
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
        return MatchResult(
            query_px=query.keypoints[idx_q].numpy().astype(np.float32),
            ref_px=db.keypoints[idx_db].numpy().astype(np.float32),
        )


def test_camera_localizer_from_feedforward_classmethod():
    """from_feedforward classmethod constructs CameraLocalizer correctly."""
    from unittest.mock import MagicMock

    pts3d, world_points, extrinsics, K = _make_synthetic_scene()
    result = MagicMock()
    result.world_points = world_points
    result.extrinsics = extrinsics
    result._zarr_path = None

    # Caller builds (images, ids) aligned to result; consumed on cache miss
    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]
    result.image_paths = ids

    extractor = _MockExtractor(pts3d, extrinsics, K)
    loc = CameraLocalizer.from_feedforward(result, images=images, ids=ids, extractor=extractor)
    assert loc is not None


def test_camera_localizer_recovers_known_pose():
    """CameraLocalizer should recover the identity pose for a camera at extrinsics[0]."""
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()

    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    extractor = _MockExtractor(pts3d, extrinsics, K)
    loc = CameraLocalizer(world_points, extrinsics, images=images, ids=ids, extractor=extractor)

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
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()

    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    extractor = _MockExtractor(pts3d, extrinsics, K)
    loc = CameraLocalizer(world_points, extrinsics, images=images, ids=ids, extractor=extractor)
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
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()
    extractor = _MockExtractor(pts3d, extrinsics, K)
    calls = []

    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(len(extrinsics))]
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    CameraLocalizer(
        world_points,
        extrinsics,
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
    np.testing.assert_allclose(pts[0], [2.0, 1.0, 1.0], atol=1e-5)  # exact pixel
    np.testing.assert_allclose(pts[1], [1.5, 2.5, 1.0], atol=1e-5)  # bilinear midpoint
    assert not valid[2] and valid[0] and valid[1]  # NaN cell dropped


def test_sample_world_points_out_of_bounds():
    """Pixels outside the image bounds are marked invalid."""
    H = W = 4
    wp = np.stack(list(np.meshgrid(np.arange(W), np.arange(H))) + [np.ones((H, W))], axis=-1).astype(np.float32)

    px = np.array([[10.0, 1.0]], dtype=np.float32)  # x beyond W-1
    _, valid = sample_world_points(wp, px)
    assert not valid[0]


def test_localize_via_depth_lookup():
    """Query identical to ref frame 0 localizes at ref 0's pose via world_points sampling."""
    H = W = 64
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    # Planar scene at z=2 under pinhole K
    f = 50.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    z = 2.0
    wp = np.stack([(xs - W / 2) / f * z, (ys - H / 2) / f * z, np.full(xs.shape, z, dtype=np.float64)], -1).astype(
        np.float32
    )

    class StubExtractor(BaseLocalExtractor):
        def extract(self, image):
            # 4x3 grid — collinear keypoints would make planar PnP degenerate
            k = torch.tensor([[8.0 + 14 * (i % 4), 8.0 + 18 * (i // 4)] for i in range(12)])
            return LocalFeatures(keypoints=k, descriptors=torch.zeros(12, 4))

        def match(self, query, db, image_hw):
            px = query.keypoints.numpy().astype(np.float32)
            return MatchResult(query_px=px, ref_px=px.copy())  # identity matches

    loc = CameraLocalizer(
        world_points=wp[None],  # (1, H, W, 3)
        extrinsics=np.eye(4, dtype=np.float32)[None],
        images=[img],
        ids=["frame_0"],
        extractor=StubExtractor(),
    )
    res = loc.localize(img, query_intrinsics=K)
    assert res.pose is not None
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)
    assert res.n_correspondences >= 10
