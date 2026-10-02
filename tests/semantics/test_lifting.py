import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.semantics.lifting import lift_features, transfer_features

########################################################
########## lift_features (multi-view) ##################
########################################################


def _make_lift_result(
    pts3d: np.ndarray,
    pixel_indices: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    n: int,
    h: int,
    w: int,
) -> PointcloudResult:
    """Build a minimal PointcloudResult sufficient for lift_features.

    Uses identity world-to-cam extrinsics and a pinhole intrinsics centered at (w/2, h/2).
    """
    extrinsics_3x4 = np.tile(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1))
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1, 1))
    extrinsics_4x4 = np.concatenate([extrinsics_3x4, bottom], axis=1)
    intrinsics = np.tile(np.array([[100, 0, w / 2], [0, 100, h / 2], [0, 0, 1]], dtype=np.float32), (n, 1, 1))
    return PointcloudResult(
        points=pts3d,
        colors=np.zeros((pts3d.shape[0], 3), dtype=np.uint8),
        extrinsics=extrinsics_4x4,
        intrinsics=None,
        model_intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i}.png") for i in range(n)],
        original_coords=np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n, 1)),  # full-frame box
        model_width=w,
        model_height=h,
        pixel_indices=pixel_indices,
        depth=depth,
        confidence=torch.from_numpy(conf),
    )


def test_lift_features_aggregates_across_frames():
    """Point visible in 2 frames with conf (1, 2) → weighted mean of feats (1, 2)."""
    n, h, w, D = 2, 8, 8, 4
    # Point at world (0, 0, 1); identity extrinsics → both frames project to (cx, cy) = (4, 4)
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    # Depth at the projected pixel matches z=1 in both frames → depth-consistent
    depth = np.ones((n, h, w), dtype=np.float32)
    # Distinct conf per frame: 1 in frame 0, 2 in frame 1
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[0, 4, 4] = 1.0
    conf[1, 4, 4] = 2.0
    # Feature maps: frame 0 all ones, frame 1 all twos
    feature_maps = [
        torch.full((D, h, w), 1.0, dtype=torch.float32),
        torch.full((D, h, w), 2.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps.__getitem__, result)

    # Weighted mean: (1*1 + 2*2) / (1 + 2) = 5/3
    expected = 5.0 / 3.0
    np.testing.assert_allclose(feats[0].numpy(), [expected] * D, atol=1e-4)


def test_lift_features_source_fallback_for_unseen():
    """Point with zero accumulated weight everywhere → source-frame sample fallback."""
    n, h, w, D = 2, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    # Depth far from z=1 in both frames → depth-consistency fails everywhere → zero weight
    depth = np.full((n, h, w), 100.0, dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    # Distinct features per frame; fallback should sample from frame 0 (source)
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps.__getitem__, result)

    # Source frame is 0 (pixel_indices says so) → expect feature 7.0
    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_without_pixel_indices_leaves_unseen_zero():
    """No pixel_indices (e.g. mesh vertices): an unseen point stays zero, a seen one is lifted."""
    n, h, w, D = 2, 8, 8, 4
    # Point 0 at z=1 matches the depth maps; point 1 at z=5 fails the depth test everywhere
    pts3d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]], dtype=np.float32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    feature_maps = [torch.full((D, h, w), 3.0, dtype=torch.float32)] * n

    result = _make_lift_result(pts3d, None, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps.__getitem__, result)

    np.testing.assert_allclose(feats[0].numpy(), [3.0] * D, atol=1e-4)
    np.testing.assert_allclose(feats[1].numpy(), [0.0] * D, atol=1e-6)


def test_lift_features_reads_depth_at_rounded_pixel():
    """Visibility reads the depth map at the ROUNDED projected pixel (depth_residual's nearest)."""
    n, h, w, D = 2, 8, 8, 4
    # Point projects to u = 4 + 100 * 0.006 = 4.6, v = 4: floor reads col 4, round reads col 5
    # - col 4 agrees with z=1, col 5 does not, so only a floor read would see the point
    # - seen nowhere → source-frame fallback (frame 0 → 7.0); a floor read gives mean(7, 9) = 8.0
    pts3d = np.array([[0.006, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.full((n, h, w), 100.0, dtype=np.float32)
    depth[:, :, 4] = 1.0
    conf = np.ones((n, h, w), dtype=np.float32)
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps.__getitem__, result)

    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_weights_by_confidence_at_rounded_pixel():
    """The confidence weight is read at the same ROUNDED pixel as the depth test."""
    n, h, w, D = 2, 8, 8, 4
    # Point projects to u = 4.6, v = 4 and is depth-consistent everywhere
    # - frame 0 has confidence only at col 5 (round), frame 1 only at col 4 (floor)
    # - a round read weights frame 0 alone → 7.0; a floor read weights frame 1 alone → 9.0
    pts3d = np.array([[0.006, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[0, :, 5] = 1.0
    conf[1, :, 4] = 1.0
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps.__getitem__, result)

    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_raises_missing_depth():
    """Missing required field (depth) raises ValueError naming the field."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    # Drop depth to trigger the error
    result = dataclasses.replace(result, depth=None)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(ValueError, match="depth"):
        lift_features(feature_maps.__getitem__, result)


def test_lift_features_rejects_4d_depth():
    """depth must be (N, H, W); a trailing channel dim raises ValueError."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    result = dataclasses.replace(result, depth=depth[..., None])  # (N, H, W, 1)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(ValueError, match="depth"):
        lift_features(feature_maps.__getitem__, result)


def test_lift_features_rejects_3x4_extrinsics():
    """extrinsics must be (N, 4, 4); a 3x4 (no homogeneous row) matrix raises ValueError."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    result = dataclasses.replace(result, extrinsics=result.extrinsics[:, :3, :])  # (N, 3, 4)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(ValueError, match="4, 4"):
        lift_features(feature_maps.__getitem__, result)


def test_lift_features_calls_loader_once_per_frame():
    """Visible point, no fallback: the loader sees frames 0..N-1 exactly once, in order."""
    n, h, w, D = 3, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    maps = [torch.full((D, h, w), float(i)) for i in range(n)]
    calls = []

    def loader(i: int) -> torch.Tensor:
        calls.append(i)
        return maps[i]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(loader, result)

    assert calls == [0, 1, 2]
    np.testing.assert_allclose(feats[0].numpy(), [1.0] * D, atol=1e-4)


def test_lift_features_fallback_loads_only_frames_with_unseen_points():
    """Zero-weight point in frame 1: after the main pass, only frame 1 is reloaded."""
    n, h, w, D = 3, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[1, 4, 4]], dtype=np.int32)
    depth = np.full((n, h, w), 100.0, dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    maps = [torch.full((D, h, w), float(i)) for i in range(n)]
    calls = []

    def loader(i: int) -> torch.Tensor:
        calls.append(i)
        return maps[i]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(loader, result)

    assert calls == [0, 1, 2, 1]
    np.testing.assert_allclose(feats[0].numpy(), [1.0] * D, atol=1e-4)


def test_lift_features_casts_fp16_maps():
    """fp16 maps (the cache dtype) lift to float32 features."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    maps = [torch.full((D, h, w), 0.5, dtype=torch.float16)]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(maps.__getitem__, result)

    assert feats.dtype == torch.float32
    np.testing.assert_allclose(feats[0].numpy(), [0.5] * D, atol=1e-4)


########################################################
########## transfer_features ###########################
########################################################


def _transfer_features_numpy_reference(targets, points, features, k=5, max_dist=0.03):
    """CPU reference: same kernel as the torch path, written with np.add.at."""
    vertices = np.asarray(targets)
    M, D = len(vertices), features.shape[1]
    distances, indices = cKDTree(vertices).query(points, k=k)

    if k == 1:
        distances, indices = distances[:, None], indices[:, None]

    valid_mask = distances[:, 0] <= max_dist

    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)

    distances, indices, feats = distances[valid_mask], indices[valid_mask], features[valid_mask]
    sigma = np.mean(distances)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((M, D), dtype=np.float64)
    wsum = np.zeros((M, 1), dtype=np.float64)

    for j in range(k):
        np.add.at(out, indices[:, j], feats * weights[:, j : j + 1])
        np.add.at(wsum, indices[:, j], weights[:, j : j + 1])

    nz = wsum[:, 0] > 0
    out[nz] /= wsum[nz]
    return out.astype(features.dtype)


def test_transfer_features_output_shape():
    rng = np.random.default_rng(0)
    verts = rng.random((50, 3))
    pts = rng.random((200, 3))
    feats = rng.random((200, 16)).astype(np.float32)
    assert transfer_features(verts, pts, feats, k=5).shape == (50, 16)


def test_transfer_features_matches_numpy_reference():
    rng = np.random.default_rng(0)
    verts = rng.random((200, 3))
    pts = rng.random((1000, 3))
    feats = rng.random((1000, 8)).astype(np.float32)
    out = transfer_features(verts, pts, feats, k=5, max_dist=0.1)
    ref = _transfer_features_numpy_reference(verts, pts, feats, k=5, max_dist=0.1)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_transfer_features_all_far_returns_zeros():
    verts = np.zeros((10, 3))
    pts = np.full((20, 3), 100.0)
    feats = np.ones((20, 4), dtype=np.float32)
    out = transfer_features(verts, pts, feats, k=3)
    assert out.shape == (10, 4) and not out.any()


def test_transfer_features_dtype_preserved():
    rng = np.random.default_rng(1)
    verts = rng.random((50, 3)).astype(np.float32)
    pts = rng.random((100, 3)).astype(np.float32)
    feats = rng.random((100, 6)).astype(np.float32)
    out = transfer_features(verts, pts, feats, k=4, max_dist=0.2)
    assert out.dtype == np.float32 and out.shape == (50, 6)


def test_transfer_features_onto_the_points_themselves_with_k1_is_identity():
    rng = np.random.default_rng(2)
    pts = rng.random((30, 3))
    feats = rng.random((30, 4)).astype(np.float32)
    out = transfer_features(pts, pts, feats, k=1)
    np.testing.assert_allclose(out, feats, rtol=1e-6)
