"""
Pinhole projection helpers: unprojection, projection, multiview depth confidence.
"""

import numpy as np
import pytest
import torch

from vggt.utils.geometry import unproject_depth_map_to_point_map

from collab_splats.geometry.projection import (
    depth_agreement,
    depth_residual,
    multiview_depth_confidence,
    project,
    sample_world_points,
    unproject,
    unproject_frames,
)


def _pose(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A non-identity float64 w2c (4, 4) and K (3, 3).

    - an identity fixture cannot tell R from R.T
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = q, rng.normal(size=3)
    K = np.array([[300.0, 0, 31.0], [0, 280.0, 22.0], [0, 0, 1]])
    return torch.from_numpy(w2c), torch.from_numpy(K)


def test_unproject_then_project_round_trips_integer_pixels():
    w2c, K = _pose()
    depth = torch.full((1, 6, 8), 2.5, dtype=torch.float64)
    pts = unproject(depth, w2c[None], K[None])
    assert pts.shape == (1, 6, 8, 3)

    px, cam = project(pts.reshape(-1, 3), w2c, K)
    v, u = torch.meshgrid(torch.arange(6.0), torch.arange(8.0), indexing="ij")
    expected = torch.stack([u, v], -1).reshape(-1, 2).double()
    assert torch.allclose(px, expected, atol=1e-9)
    assert torch.allclose(cam[:, 2], torch.full((48,), 2.5, dtype=torch.float64))


def test_unproject_keeps_input_dtype_and_batch():
    w2c, K = _pose()
    depth = torch.ones((3, 4, 5), dtype=torch.float32)
    w2c_batch = w2c.float()[None].expand(3, 4, 4)
    K_batch = K.float()[None].expand(3, 3, 3)
    out = unproject(depth, w2c_batch, K_batch)
    assert out.shape == (3, 4, 5, 3) and out.dtype == torch.float32


def test_unproject_accepts_3x4_pose():
    w2c, K = _pose(1)
    depth = torch.rand((1, 4, 5), dtype=torch.float64) + 1.0
    full = unproject(depth, w2c[None], K[None])
    top = unproject(depth, w2c[None, :3], K[None])
    assert torch.equal(full, top)


def _rot_z(theta: float) -> np.ndarray:
    """
    3x3 float32 rotation about z by theta radians.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _frames_scene(k: int, H: int, W: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Varied float32 depth (k, H, W), non-identity (k, 3, 4) w2c and (k, 3, 3) K.
    """
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.5, 50.0, (k, H, W)).astype(np.float32)
    ext = np.zeros((k, 3, 4), dtype=np.float32)

    for i in range(k):
        ext[i, :3, :3] = _rot_z(0.3 * i)
        ext[i, :3, 3] = 10 * rng.standard_normal(3)

    K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]], dtype=np.float32)
    K = np.tile(K, (k, 1, 1))

    return depth, ext, K


def test_unproject_frames_matches_vggt_numpy():
    """
    The float32 unproject equals vggt's numpy unproject to float32 rounding.
    """
    rng = np.random.default_rng(0)
    k, H, W = 70, 12, 16
    depth = rng.uniform(0.5, 5.0, (k, H, W)).astype(np.float32)
    ext = np.zeros((k, 3, 4), dtype=np.float32)

    for i in range(k):
        ext[i, :3, :3] = _rot_z(0.01 * i)
        ext[i, :3, 3] = rng.standard_normal(3)

    K = np.array([[50.0, 0, 8], [0, 50.0, 6], [0, 0, 1]], dtype=np.float32)
    K = np.tile(K, (k, 1, 1))

    expected = unproject_depth_map_to_point_map(depth[..., None], ext, K).astype(np.float32)
    got = unproject_frames(depth, ext, K)

    assert got.dtype == np.float32 and got.shape == (k, H, W, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_unproject_frames_batches_match_one_pass():
    """
    A batch_size smaller than N, with a ragged last batch, fills every frame as one pass does.
    """
    depth, ext, K = _frames_scene(7, 8, 10, seed=2)

    one_pass = unproject_frames(depth, ext, K, batch_size=7)
    batched = unproject_frames(depth, ext, K, batch_size=3)

    np.testing.assert_array_equal(batched, one_pass)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 only exists on CUDA")
def test_unproject_frames_ignores_global_tf32():
    """
    A process-wide TF32 setting (mapanything enables it) neither rounds the unproject nor leaks.
    """
    depth, ext, K = _frames_scene(4, 64, 96, seed=1)

    before = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    try:
        exact = unproject_frames(depth, ext, K)
        torch.set_float32_matmul_precision("high")
        under_tf32 = unproject_frames(depth, ext, K)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(before)

    np.testing.assert_array_equal(under_tf32, exact)


def test_project_clamps_depth_behind_camera():
    _, K = _pose()
    behind = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64)
    px, cam = project(behind, torch.eye(4, dtype=torch.float64), K)
    assert torch.isfinite(px).all() and cam[0, 2] < 0


def test_project_divides_by_the_camera_depth_and_floors_it_at_min_depth():
    _, K = _pose()
    points = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)

    pixels, points_cam = project(points, identity, K)
    floored, _ = project(points, identity, K, min_depth=1e3)

    # Identity pose: the camera-frame point is the world point and the divide is by z = 4
    assert torch.equal(points_cam, points)
    assert torch.allclose(pixels, torch.tensor([[300.0 / 4 + 31.0, 2 * 280.0 / 4 + 22.0]], dtype=torch.float64))

    # A floor above z takes over the divide, collapsing the point towards the principal point
    assert torch.allclose(floored, torch.tensor([[300.0 / 1e3 + 31.0, 2 * 280.0 / 1e3 + 22.0]], dtype=torch.float64))


def test_project_one_camera_keeps_point_shape_and_points_dtype():
    _, K = _pose()
    identity = torch.eye(4, dtype=torch.float32)
    points = torch.tensor([[1.0, 2.0, 4.0], [0.5, -1.0, 3.0]], dtype=torch.float32)

    single, _ = project(points[0], identity, K)
    pixels, _ = project(points, identity, K)

    # A (3,) point gives (2,) pixels; float64 K does not promote float32 (P, 3) points
    assert K.dtype == torch.float64
    assert single.shape == (2,)
    assert pixels.shape == (2, 2) and pixels.dtype == torch.float32


def test_depth_residual_reads_the_other_views_depth_at_the_projected_pixel():
    # World frame = view i; view j sits at a non-identity pose from it
    w2c_j, _ = _pose(3)

    # View j: non-square K and grid, depth ramp that tells u from v
    K = torch.tensor([[40.0, 0, 3.5], [0, 55.0, 2.25], [0, 0, 1]], dtype=torch.float64)
    v, u = torch.meshgrid(torch.arange(5.0), torch.arange(7.0), indexing="ij")
    depth_j = (2.0 + 0.1 * u + 0.01 * v).double()

    # Camera-j points at known pixels and z, then taken to world through view i's pose
    # - rows 0-2: interior integer pixels, z differs from the stored depth
    # - row 3: projects right of the grid; row 4: behind camera j
    pixels = torch.tensor([[1.0, 4.0], [5.0, 1.0], [3.0, 2.0], [9.0, 2.0], [3.0, 2.0]], dtype=torch.float64)
    z = torch.tensor([2.5, 1.0, 4.0, 3.0, -3.0], dtype=torch.float64)
    rays = torch.stack(
        [(pixels[:, 0] - K[0, 2]) / K[0, 0], (pixels[:, 1] - K[1, 2]) / K[1, 1], torch.ones(5, dtype=torch.float64)], -1
    )
    points_cam_j = rays * z[:, None]
    points_world = (points_cam_j - w2c_j[:3, 3]) @ w2c_j[:3, :3]

    residual, expected, sampled, valid, _ = depth_residual(points_world, w2c_j, K, depth_j)

    # Stored depth at (u, v) is 2 + 0.1 u + 0.01 v; the residual is expected minus sampled
    stored = torch.tensor([2.14, 2.51, 2.32], dtype=torch.float64)
    assert torch.equal(valid, torch.tensor([True, True, True, False, False]))
    assert torch.allclose(expected, z, atol=1e-9)
    assert torch.allclose(sampled[:3], stored, atol=1e-12)
    assert torch.allclose(residual[:3], z[:3] - stored, atol=1e-9)
    assert torch.equal(residual, expected - sampled)


def test_depth_residual_returns_project_pixels():
    points = torch.tensor([[0.1, -0.2, 2.0], [0.0, 0.0, 3.0]])
    world_to_cam = torch.eye(4)
    intrinsics = torch.tensor([[10.0, 0, 4.5], [0, 10.0, 3.5], [0, 0, 1]])

    *_, pixels = depth_residual(points, world_to_cam, intrinsics, torch.ones(8, 10))
    expected, _ = project(points, world_to_cam, intrinsics)

    assert torch.equal(pixels, expected)


def test_project_unproject_round_trip_non_identity():
    # Three non-identity poses, one per frame; a cropped K: non-square grid, off-center principal point
    poses = torch.stack([_pose(seed)[0] for seed in (4, 5, 6)])
    K = torch.tensor([[210.0, 0, 13.25], [0, 190.0, 5.5], [0, 0, 1]], dtype=torch.float64)
    intrinsics = K[None].expand(3, 3, 3)
    depth = torch.rand((3, 9, 16), dtype=torch.float64, generator=torch.Generator().manual_seed(0)) + 0.5
    points = unproject(depth, poses, intrinsics)

    # Each frame's points land back on its own integer grid at its own depth
    v, u = torch.meshgrid(torch.arange(9.0), torch.arange(16.0), indexing="ij")
    grid = torch.stack([u, v], -1).double()
    for i in range(3):
        pixels, points_cam = project(points[i], poses[i], K)
        assert torch.allclose(pixels, grid, atol=1e-9)
        assert torch.allclose(points_cam[..., 2], depth[i], atol=1e-12)


def _two_view_scene(depth_b: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two identical cameras at the origin looking down +Z at a fronto-parallel plane.

    - view 0 at depth 2 everywhere; view 1 at depth_b everywhere
    """
    h, w = 8, 10
    depth = torch.stack([torch.full((h, w), 2.0), torch.full((h, w), depth_b)])
    intrinsics = torch.tensor([[10.0, 0, (w - 1) / 2], [0, 10.0, (h - 1) / 2], [0, 0, 1]]).expand(2, 3, 3)
    extrinsics = torch.eye(4).expand(2, 4, 4)
    return depth, intrinsics, extrinsics


def test_multiview_depth_confidence_agree():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert agree.shape == seen.shape == (2, 8, 10)
    assert (agree == 1).all() and (seen == 1).all()


def test_multiview_depth_confidence_occluded_is_unseen():
    # View 1 sees a surface in front of view 0's points: occluded, left out of seen
    depth, intrinsics, extrinsics = _two_view_scene(1.0)

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 0).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_free_space_violation_is_seen_not_agree():
    # View 1 sees behind view 0's points: counted as seen, not as agreeing
    depth, intrinsics, extrinsics = _two_view_scene(3.0)

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 1).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_out_of_bounds_is_unseen():
    # View 1 shifted far sideways: nothing projects in bounds
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    extrinsics = extrinsics.clone()
    extrinsics[1, 0, 3] = 100.0

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 0).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_zero_source_depth_counts_nothing():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    depth[0, :2] = 0.0

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0, :2] == 0).all() and (agree[0, :2] == 0).all()
    assert (seen[0, 2:] == 1).all()


def test_multiview_depth_confidence_zero_target_depth_is_seen_not_agree():
    # A hole in view 1: the pixel is seen (not occluded) but has nothing to agree with
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    depth[1] = 0.0

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 1).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_rejects_original_res_intrinsics():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 2] = 50.0

    with pytest.raises(ValueError, match="principal point"):
        multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)


def test_multiview_depth_confidence_rejects_length_mismatch():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)

    with pytest.raises(ValueError, match="length mismatch"):
        multiview_depth_confidence(depth.numpy(), intrinsics[:1].numpy(), extrinsics.numpy(), rel_thresh=0.01)


def _pinhole(h: int, w: int) -> np.ndarray:
    """
    Pinhole K with focal = W and the principal point at the image center.
    """
    return np.array([[float(w), 0.0, w / 2.0], [0.0, float(h), h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def test_multiview_depth_confidence_nearest_sampling_fabricates_no_depth():
    # Depth step at mid-image, small baseline: bilinear would sample a depth on no surface
    # - every in-bounds pixel sits on a rigid surface both cameras see, so seen ones all agree
    n, h, w = 2, 16, 16
    depth = np.empty((n, h, w), dtype=np.float32)
    depth[:, :, : w // 2] = 2.0
    depth[:, :, w // 2 :] = 8.0
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * n)
    extrinsics[1, 0, 3] = 0.05

    agree, seen = multiview_depth_confidence(depth, np.stack([_pinhole(h, w)] * n), extrinsics, rel_thresh=0.05)

    assert (seen[0] > 0).sum() > 0
    np.testing.assert_array_equal(agree[0], seen[0])


def test_multiview_depth_confidence_is_scale_invariant():
    # Depth and translations times a power of two: exact in float32, so counts match bit for bit
    n, h, w = 3, 8, 8
    rng = np.random.default_rng(11)
    depth = (rng.random((n, h, w)).astype(np.float32) + 0.5) * 2.0
    intrinsics = np.stack([_pinhole(h, w)] * n)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * n)
    extrinsics[:, 0, 3] = 0.25 * np.arange(n)

    base = multiview_depth_confidence(depth, intrinsics, extrinsics, rel_thresh=0.05)

    s = 8.0
    extrinsics_s = extrinsics.copy()
    extrinsics_s[:, :3, 3] *= s
    scaled = multiview_depth_confidence(depth * s, intrinsics, extrinsics_s, rel_thresh=0.05)

    np.testing.assert_array_equal(base[0], scaled[0])
    np.testing.assert_array_equal(base[1], scaled[1])


def test_depth_agreement_returns_the_camera_depth_project_computes():
    """expected is project's camera z, so callers need no second transform."""
    rng = np.random.default_rng(3)
    points = torch.as_tensor(rng.uniform(-1, 1, (500, 3)) + [0, 0, 4], dtype=torch.float32)
    w2c = torch.eye(4)
    w2c[0, 3] = 0.3
    K = torch.tensor([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]])
    depth = torch.full((16, 16), 4.0)

    *_, expected = depth_agreement(points, w2c, K, depth, 0.05)
    _, points_cam = project(points, w2c, K)
    assert torch.equal(expected, points_cam[:, 2])


def _views(hw=(12, 16), seed=0):
    """Four distinct cameras around a noisy depth field with holes, and points incl. some behind."""
    rng = np.random.default_rng(seed)
    b, (h, w) = 4, hw
    K = torch.tensor([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1.0]]).repeat(b, 1, 1)
    K[:, 0, 0] = torch.tensor([18.0, 20, 22, 24])
    w2c = torch.eye(4).repeat(b, 1, 1)
    w2c[1, :3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c, sn = np.cos(0.1), np.sin(0.1)
    w2c[2, :3, :3] = torch.tensor([[1.0, 0, 0], [0, c, -sn], [0, sn, c]], dtype=torch.float32)
    w2c[:, :3, 3] = torch.as_tensor(rng.uniform(-0.3, 0.3, (b, 3)), dtype=torch.float32)
    depth = torch.as_tensor(rng.uniform(3, 5, (b, h, w)), dtype=torch.float32)
    depth[:, 0, :3] = 0.0
    depth[:, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2] = 0.0
    points = torch.as_tensor(rng.uniform(-1, 1, (400, 3)) + [0, 0, 4], dtype=torch.float32)
    points = torch.cat([points, torch.tensor([[0.0, 0, -2], [0.5, 0, -3]])])
    return points, w2c, K, depth


def test_project_over_a_pose_batch_matches_one_camera_at_a_time():
    points, w2c, K, _ = _views()
    pixels, cam = project(points, w2c, K)
    assert pixels.shape == (4, 402, 2) and cam.shape == (4, 402, 3)
    for b in range(4):
        p1, c1 = project(points, w2c[b], K[b])
        torch.testing.assert_close(pixels[b], p1, rtol=0, atol=1e-5)
        torch.testing.assert_close(cam[b], c1, rtol=0, atol=1e-6)


def test_depth_residual_over_a_pose_batch_matches_one_view_at_a_time():
    points, w2c, K, depth = _views()
    residual, expected, sampled, valid, pixels = depth_residual(points, w2c, K, depth)
    assert residual.shape == (4, 402) and pixels.shape == (4, 402, 2)
    assert (valid & (sampled == 0)).any()
    assert (expected <= 0).any()
    for b in range(4):
        r1, e1, s1, v1, p1 = depth_residual(points, w2c[b], K[b], depth[b])
        torch.testing.assert_close(residual[b], r1, rtol=0, atol=1e-5)
        torch.testing.assert_close(expected[b], e1, rtol=0, atol=1e-6)
        torch.testing.assert_close(sampled[b], s1, rtol=0, atol=0)
        torch.testing.assert_close(pixels[b], p1, rtol=0, atol=1e-5)
        assert torch.equal(valid[b], v1)


def test_depth_agreement_over_a_pose_batch_matches_one_view_at_a_time():
    points, w2c, K, depth = _views()
    agree, seen, rel, expected = depth_agreement(points, w2c, K, depth, 0.05)
    assert agree.shape == seen.shape == rel.shape == expected.shape == (4, 402)
    assert agree.any() and (seen & ~agree).any()
    for b in range(4):
        a1, s1, r1, e1 = depth_agreement(points, w2c[b], K[b], depth[b], 0.05)
        assert torch.equal(agree[b], a1) and torch.equal(seen[b], s1)
        torch.testing.assert_close(rel[b], r1, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(expected[b], e1, rtol=0, atol=1e-6)


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
