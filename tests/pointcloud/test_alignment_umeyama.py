import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR
from collab_splats.pointcloud.loop_closure.alignment import umeyama_se3, umeyama_sim3


def test_umeyama_identity():
    """Identity transform: source == target."""
    np.random.seed(0)
    src = np.random.randn(20, 3).astype(np.float32)
    T = umeyama_se3(src, src)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-5)


def test_umeyama_known_translation():
    """Pure translation: recovers shift within 1e-5."""
    np.random.seed(1)
    src = np.random.randn(50, 3).astype(np.float64)
    shift = np.array([1.0, -2.0, 3.0])
    tgt = src + shift
    T = umeyama_se3(src, tgt)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(T[:3, 3], shift, atol=1e-5)


def test_umeyama_known_rotation_translation():
    """Known SE(3): recover rotation + translation within 1e-4."""
    np.random.seed(2)
    src = np.random.randn(100, 3).astype(np.float64)
    R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t_true = np.array([1.0, 2.0, 3.0])
    tgt = (R_true @ src.T).T + t_true
    T = umeyama_se3(src, tgt)
    np.testing.assert_allclose(T[:3, :3], R_true, atol=1e-4)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-4)


def test_umeyama_weighted_ignores_outliers():
    """Weights near zero suppress outlier influence."""
    np.random.seed(3)
    M = 100
    src = np.random.randn(M, 3).astype(np.float64)
    shift = np.array([1.0, 0.0, 0.0])
    tgt = src + shift
    tgt[-20:] += np.random.randn(20, 3) * 100.0
    weights = np.ones(M, dtype=np.float64)
    weights[-20:] = 1e-6
    T = umeyama_se3(src, tgt, weights=weights)
    np.testing.assert_allclose(T[:3, 3], shift, atol=0.05)


def test_umeyama_no_reflection():
    """Output rotation must have det == +1 (not -1)."""
    np.random.seed(4)
    src = np.random.randn(30, 3).astype(np.float64)
    T = umeyama_se3(src, src * np.array([-1, 1, 1]))
    assert np.linalg.det(T[:3, :3]) > 0


def test_umeyama_sim3_known_scale():
    rng = np.random.default_rng(0)
    source = rng.standard_normal((50, 3)).astype(np.float32)
    s_gt, R_gt = 2.5, ScipyR.from_euler("z", 30, degrees=True).as_matrix().astype(np.float32)
    t_gt = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    target = (s_gt * (R_gt @ source.T).T + t_gt)

    s, R, t = umeyama_sim3(source, target)

    assert abs(s - s_gt) < 0.01, f"scale off: {s} vs {s_gt}"
    assert np.allclose(R, R_gt, atol=0.01), f"R off: {R}"
    assert np.allclose(t, t_gt, atol=0.05), f"t off: {t}"


def test_umeyama_sim3_unit_scale():
    rng = np.random.default_rng(1)
    source = rng.standard_normal((30, 3)).astype(np.float32)
    target = source + np.array([0.5, 0.0, -1.0])

    s, R, t = umeyama_sim3(source, target)

    assert abs(s - 1.0) < 0.05
    assert np.allclose(R, np.eye(3), atol=0.05)


def test_umeyama_sim3_weighted():
    rng = np.random.default_rng(2)
    source = rng.standard_normal((100, 3)).astype(np.float32)
    s_gt, t_gt = 1.8, np.array([2.0, 0.0, 0.0], dtype=np.float32)
    target = s_gt * source + t_gt
    weights = np.ones(100, dtype=np.float32)

    s, R, t = umeyama_sim3(source, target, weights=weights)
    assert abs(s - s_gt) < 0.05


def test_umeyama_sim3_insufficient_points_returns_unit():
    source = np.zeros((0, 3), dtype=np.float32)
    target = np.zeros((0, 3), dtype=np.float32)
    s, R, t = umeyama_sim3(source, target)
    assert s == 1.0
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))


import torch
from pathlib import Path
from collab_splats.pointcloud.loop_closure.alignment import overlap_region_align_sim3


def _make_submap(world_points, world_points_conf=None):
    """Minimal Submap stub for alignment tests."""
    from collab_splats.pointcloud.loop_closure.submap import Submap
    K = world_points.shape[0] if world_points is not None else 2
    return Submap(
        submap_id=0,
        frames=torch.zeros(K, 3, 4, 4),
        poses=np.eye(4, dtype=np.float32)[None].repeat(K, axis=0),
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(K, axis=0),
        retrieval_vectors=torch.zeros(K, 128),
        image_paths=[Path(f"f{i}.jpg") for i in range(K)],
        world_points=world_points,
        world_points_conf=world_points_conf,
    )


def test_overlap_region_align_sim3_no_world_points_returns_unit():
    a = _make_submap(None)
    b = _make_submap(None)
    s, R, t = overlap_region_align_sim3(a, b, overlap_frames=2)
    assert s == 1.0
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))


def test_overlap_region_align_sim3_known_scale():
    rng = np.random.default_rng(42)
    # submap_a has 4 frames, submap_b has 4 frames, overlap=2
    # last 2 frames of a and first 2 of b share same physical points at 2x scale
    pts_shared = rng.standard_normal((2, 10, 3)).astype(np.float32)
    pts_b_scaled = pts_shared * 2.0  # s=2.0

    wp_a = np.concatenate([rng.standard_normal((2, 10, 3)).astype(np.float32), pts_shared], axis=0)
    wp_b = np.concatenate([pts_b_scaled, rng.standard_normal((2, 10, 3)).astype(np.float32)], axis=0)

    a = _make_submap(wp_a)
    b = _make_submap(wp_b)

    s, R, t = overlap_region_align_sim3(a, b, overlap_frames=2)
    # s should be ~0.5 (maps b's 2x points back to a's 1x)
    assert 0.3 < s < 0.7, f"Expected s≈0.5, got {s}"
