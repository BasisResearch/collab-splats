import logging
import re

import numpy as np
import open3d as o3d
import pytest
import torch

from collab_splats.mesh import tsdf
from collab_splats.mesh.clean import VolumeGround
from collab_splats.mesh.tsdf import (
    calibrate_confidence,
    compute_tsdf_voxel_size,
    create_tsdf_mesh,
    lift_weights,
    pixel_weights,
)


def _views(n=3, h=32, w=32):
    """n cameras looking down +z at a plane 1 unit away, shifted 0.05 along x per view."""
    depths = np.ones((n, h, w), dtype=np.float32)
    rgbs = np.full((n, h, w, 3), 128, dtype=np.uint8)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, 0, 3] = 0.05 * np.arange(n)
    K = np.tile(
        np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64), (n, 1, 1)
    )
    return depths, rgbs, c2w, K


def test_create_tsdf_mesh_returns_a_colored_mesh():
    depths, rgbs, c2w, K = _views()
    mesh = create_tsdf_mesh(
        depths,
        rgbs,
        c2w,
        K,
        weights=np.ones_like(depths),
        voxel_size=0.02,
        depth_trunc=2.0,
    )
    assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0
    assert mesh.has_vertex_colors()
    assert np.allclose(np.asarray(mesh.vertex_colors), 128 / 255, atol=0.05)


def test_create_tsdf_mesh_skips_a_view_with_no_valid_depth():
    """An all-zero depth view (e.g. all sky) touches no block; Open3D would raise on it."""
    depths, rgbs, c2w, K = _views()
    depths[1] = 0.0
    mesh = create_tsdf_mesh(
        depths,
        rgbs,
        c2w,
        K,
        weights=np.ones_like(depths),
        voxel_size=0.02,
        depth_trunc=2.0,
    )
    assert len(mesh.triangles) > 0


def test_create_tsdf_mesh_falls_back_to_cpu_without_cuda(monkeypatch):
    depths, rgbs, c2w, K = _views()
    weights = np.ones_like(depths)
    gpu = create_tsdf_mesh(
        depths, rgbs, c2w, K, weights=weights, voxel_size=0.02, depth_trunc=2.0
    )

    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    cpu = create_tsdf_mesh(
        depths, rgbs, c2w, K, weights=weights, voxel_size=0.02, depth_trunc=2.0
    )

    # Same torch math on both devices: same vertices up to float error, same counts
    assert len(cpu.triangles) > 0 and cpu.has_vertex_colors()
    assert len(cpu.vertices) == len(gpu.vertices)
    assert len(cpu.triangles) == len(gpu.triangles)
    np.testing.assert_allclose(
        np.unique(np.asarray(cpu.vertices), axis=0),
        np.unique(np.asarray(gpu.vertices), axis=0),
        atol=1e-5,
    )


def test_create_tsdf_mesh_rejects_float_rgb():
    depths, rgbs, c2w, K = _views()

    with pytest.raises(ValueError, match="uint8"):
        create_tsdf_mesh(
            depths,
            rgbs.astype(np.float32) / 255,
            c2w,
            K,
            weights=np.ones_like(depths),
            voxel_size=0.02,
            depth_trunc=2.0,
        )


def test_create_tsdf_mesh_rejects_principal_point_outside_grid():
    depths, rgbs, c2w, K = _views()
    K[:, 0, 2] = (
        64  # cx beyond the 32-wide grid: K is at a different resolution than depth
    )

    with pytest.raises(ValueError, match="Principal point"):
        create_tsdf_mesh(
            depths,
            rgbs,
            c2w,
            K,
            weights=np.ones_like(depths),
            voxel_size=0.02,
            depth_trunc=2.0,
        )


def test_create_tsdf_mesh_rejects_frame_count_mismatch():
    depths, rgbs, c2w, K = _views()

    with pytest.raises(ValueError, match="views"):
        create_tsdf_mesh(
            depths,
            rgbs[:2],
            c2w,
            K,
            weights=np.ones_like(depths),
            voxel_size=0.02,
            depth_trunc=2.0,
        )


def test_create_tsdf_mesh_sdf_trunc_defaults_to_four_voxels(monkeypatch):
    # Record the band each view integrates with; omitted sdf_trunc means 4 voxels
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    seen = []
    integrate = tsdf._integrate_weighted

    def _record(*args, **kwargs):
        seen.append(kwargs["sdf_trunc"])
        return integrate(*args, **kwargs)

    monkeypatch.setattr(tsdf, "_integrate_weighted", _record)
    depths, rgbs, c2w, K = _views()
    create_tsdf_mesh(
        depths,
        rgbs,
        c2w,
        K,
        weights=np.ones_like(depths),
        voxel_size=0.02,
        depth_trunc=2.0,
    )

    assert len(seen) == len(depths)
    assert seen == [pytest.approx(4 * 0.02)] * len(depths)


def _canonical(mesh):
    """
    Vertex rows deduplicated by position, and triangles as sorted position tuples.

    - Open3D emits duplicate vertices at block borders; tiling welds them, so compare by position
    """
    v = np.asarray(mesh.vertices)
    used = np.unique(np.asarray(mesh.triangles))
    tris = np.sort(v[np.asarray(mesh.triangles)].reshape(-1, 3, 3), axis=1).reshape(
        -1, 9
    )
    attrs = np.hstack(
        [v, np.asarray(mesh.vertex_normals), np.asarray(mesh.vertex_colors)]
    )
    _, first = np.unique(v[used], axis=0, return_index=True)
    return attrs[used][first], tris[np.lexsort(tris.T[::-1])]


def test_create_tsdf_mesh_tiles_to_the_single_grid_mesh(monkeypatch):
    # Bumpy plane seen from 4 shifted views; 40 blocks per grid forces several tiles
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=4, h=48, w=48)
    yy, xx = np.mgrid[:48, :48]
    depths = depths * (1 + 0.1 * np.sin(xx / 5.0) * np.cos(yy / 7.0)).astype(np.float32)
    rgbs[..., 0] = (xx * 5).astype(np.uint8)
    args = (depths, rgbs, c2w, K)

    weights = np.ones_like(depths)
    single = create_tsdf_mesh(*args, weights=weights, voxel_size=0.01, depth_trunc=2.0)
    tiled = create_tsdf_mesh(
        *args, weights=weights, voxel_size=0.01, depth_trunc=2.0, max_blocks=40
    )

    single_v, single_t = _canonical(single)
    tiled_v, tiled_t = _canonical(tiled)
    assert len(single.triangles) == len(tiled.triangles) > 0
    assert np.array_equal(single_t, tiled_t)
    assert np.array_equal(single_v[:, :3], tiled_v[:, :3])
    assert np.allclose(single_v[:, 3:], tiled_v[:, 3:], atol=1e-6)


def test_create_tsdf_mesh_refuses_a_band_narrower_than_a_voxel():
    depths = np.ones((1, 4, 4), np.float32)
    rgbs = np.zeros((1, 4, 4, 3), np.uint8)

    with pytest.raises(ValueError, match="sdf_trunc"):
        create_tsdf_mesh(
            depths,
            rgbs,
            np.eye(4)[None],
            np.eye(3)[None],
            weights=np.ones_like(depths),
            voxel_size=0.1,
            depth_trunc=5.0,
            sdf_trunc=0.05,
        )


def test_compute_tsdf_voxel_size_spans_depth_px_footprints_at_the_ref_depth():
    # Plane at depth 1, fx 32: one pixel spans 1/32, so 4 pixels make a 0.125 voxel
    depths, _, c2w, K = _views()
    assert compute_tsdf_voxel_size(depths, c2w, K, depth_fx=32.0) == pytest.approx(
        0.125
    )
    assert compute_tsdf_voxel_size(
        2 * depths, c2w, K, depth_fx=32.0, depth_px=2.0
    ) == pytest.approx(0.125)


def test_compute_tsdf_voxel_size_coarsens_to_the_memory_budget():
    depths, _, c2w, K = _views(h=256, w=256)
    free = compute_tsdf_voxel_size(
        depths, c2w, K, depth_fx=256.0, depth_px=0.01, stride=1
    )
    capped = compute_tsdf_voxel_size(
        depths, c2w, K, depth_fx=256.0, depth_px=0.01, stride=1, max_gb=1e-3
    )
    assert capped > free


def test_compute_tsdf_voxel_size_refuses_empty_depth():
    depths, _, c2w, K = _views()
    with pytest.raises(ValueError, match="no positive depth"):
        compute_tsdf_voxel_size(np.zeros_like(depths), c2w, K, depth_fx=32.0)


def test_create_tsdf_mesh_without_depth_trunc_keeps_every_depth():
    depths, rgbs, c2w, K = _views()
    mesh = create_tsdf_mesh(
        5 * depths, rgbs, c2w, K, weights=np.ones_like(depths), voxel_size=0.1
    )
    assert len(mesh.triangles) > 0


def test_create_tsdf_mesh_reads_k_as_pixel_center(monkeypatch):
    # Pixel 0's footprint starts at center-convention u = -0.5: x = y = -1.05 for c 10, f 10
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths = np.ones((1, 20, 20), np.float32)
    rgbs = np.zeros((1, 20, 20, 3), np.uint8)
    K = np.array([[[10.0, 0, 10.0], [0, 10.0, 10.0], [0, 0, 1]]])

    mesh = create_tsdf_mesh(
        depths,
        rgbs,
        np.eye(4)[None],
        K,
        weights=np.full_like(depths, 2.0),
        voxel_size=0.01,
        depth_trunc=2.0,
    )

    corner = np.asarray(mesh.vertices)[:, :2].min(axis=0)
    assert corner == pytest.approx([-1.05, -1.05], abs=0.01)


def test_create_tsdf_mesh_zero_weight_view_adds_nothing(monkeypatch):
    # A view at a wrong depth with weight 0 leaves the mesh as the other views make it
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=4)
    weights = np.ones_like(depths)
    reference = create_tsdf_mesh(
        depths[:3],
        rgbs[:3],
        c2w[:3],
        K[:3],
        weights=weights[:3],
        voxel_size=0.02,
        depth_trunc=2.0,
    )

    depths[3] = 1.5
    weights[3] = 0.0
    mesh = create_tsdf_mesh(
        depths, rgbs, c2w, K, weights=weights, voxel_size=0.02, depth_trunc=2.0
    )

    assert len(mesh.triangles) == len(reference.triangles)
    np.testing.assert_allclose(np.asarray(mesh.vertices)[:, 2], 1.0, atol=0.02)


def test_create_tsdf_mesh_surface_needs_weight_above_w_min(monkeypatch):
    # One view at weight 1: nothing passes w_min 1.0, everything passes w_min 0.5
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=1)
    weights = np.ones_like(depths)
    kwargs = dict(weights=weights, voxel_size=0.02, depth_trunc=2.0)

    assert len(create_tsdf_mesh(depths, rgbs, c2w, K, **kwargs).triangles) == 0
    assert (
        len(create_tsdf_mesh(depths, rgbs, c2w, K, w_min=0.5, **kwargs).triangles) > 0
    )


def test_create_tsdf_mesh_weights_pull_the_surface(monkeypatch):
    # Two depths, 1.00 and 1.04; weight 3:1 puts the surface at the weighted mean 1.01
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=2)
    c2w[1] = c2w[0]
    depths[1] = 1.04
    weights = np.stack([np.full(depths.shape[1:], 3.0), np.full(depths.shape[1:], 1.0)])
    weights = weights.astype(np.float32)

    mesh = create_tsdf_mesh(
        depths,
        rgbs,
        c2w,
        K,
        weights=weights,
        voxel_size=0.01,
        depth_trunc=2.0,
        sdf_trunc=0.08,
    )

    z = np.asarray(mesh.vertices)[:, 2]
    assert np.median(z) == pytest.approx(1.01, abs=0.003)


def test_create_tsdf_mesh_rejects_weights_shape_mismatch():
    depths, rgbs, c2w, K = _views()
    with pytest.raises(ValueError, match="weights"):
        create_tsdf_mesh(
            depths,
            rgbs,
            c2w,
            K,
            weights=np.ones((3, 4, 4), np.float32),
            voxel_size=0.02,
        )


def test_pixel_weights_gate_and_power():
    # Flat depth (no edges); conf 1, 2, 4, 8 against c* 2, gamma 1 -> power 2
    depth = np.ones((1, 4, 4), np.float32)
    conf = np.tile(np.array([1.0, 2.0, 4.0, 8.0], np.float32), (1, 4, 1))

    weights = pixel_weights(depth, conf, c_star=2.0, gamma=1.0)

    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights[0, 0], [0.0, 1.0, 4.0, 16.0])


def test_pixel_weights_zero_on_depth_edges_and_missing_depth():
    # Step from depth 1 to 2 between columns 3 and 4; column 0 has no depth
    depth = np.ones((1, 6, 8), np.float32)
    depth[:, :, 4:] = 2.0
    depth[:, :, 0] = 0.0

    weights = pixel_weights(depth)

    assert (weights[0, :, [3, 4]] == 0).all()  # both sides of the step
    assert (weights[0, :, [0, 1]] == 0).all()  # missing depth and its neighbor
    assert (weights[0, :, [2, 5, 6, 7]] == 1).all()


def test_pixel_weights_without_confidence_is_unit():
    depth = np.ones((2, 5, 5), np.float32)
    weights = pixel_weights(depth, None)
    np.testing.assert_array_equal(weights, np.ones_like(depth))


def test_lift_weights_nearest_through_crop_box():
    # 2x2 model grid into a 4x6 frame through the box x 1..5, y 0..4
    weights = np.array([[[1.0, 2.0], [3.0, 4.0]]], np.float32)
    coords = np.array([[1, 0, 5, 4, 6, 4]], np.float32)

    lifted = lift_weights(weights, coords, (4, 6))

    assert lifted.shape == (1, 4, 6)
    assert (lifted[0, :, [0, 5]] == 0).all()  # outside the box
    np.testing.assert_array_equal(lifted[0, 0, 1:5], [1, 1, 2, 2])
    np.testing.assert_array_equal(lifted[0, 3, 1:5], [3, 3, 4, 4])


def _noisy_planes(a, gamma, n=24, h=64, w=64, seed=0):
    """
    Plane z=1 seen by n x-shifted cameras; depth noise std a * conf^-gamma, conf ramps by row.

    - x-shift keeps a point on its row, so neighbor views sample the same confidence
    """
    rng = np.random.default_rng(seed)
    conf = np.tile(np.geomspace(1.1, 70, h)[:, None], (n, 1, w)).astype(np.float32)
    noise = a * conf**-gamma * rng.standard_normal((n, h, w))
    depth = (1 + noise).astype(np.float32)
    w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    w2c[:, 0, 3] = -0.01 * np.arange(n)
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float32)
    return depth, conf, w2c, np.tile(K, (n, 1, 1))


def test_calibrate_confidence_recovers_gamma():
    for gamma in (0.9, 1.2):
        depth, conf, w2c, K = _noisy_planes(0.02, gamma)
        _, fitted = calibrate_confidence(depth, conf, w2c, K, sdf_trunc=0.005)
        assert fitted == pytest.approx(gamma, rel=0.05)


def test_calibrate_confidence_ignores_global_tf32():
    # A global TF32 (mapanything import) must not move the fit
    planes = _noisy_planes(0.02, 1.2)
    precision = torch.get_float32_matmul_precision()

    try:
        torch.set_float32_matmul_precision("highest")
        reference = calibrate_confidence(*planes, sdf_trunc=0.005)
        torch.set_float32_matmul_precision("high")
        under_tf32 = calibrate_confidence(*planes, sdf_trunc=0.005)
    finally:
        torch.set_float32_matmul_precision(precision)

    assert under_tf32 == pytest.approx(reference, rel=1e-6)


def test_calibrate_confidence_matches_reference_values():
    # Values recorded when the recipe was verified on this synthetic scene
    c_star, gamma = calibrate_confidence(*_noisy_planes(0.02, 0.9), sdf_trunc=0.005)
    assert c_star == pytest.approx(3.865, rel=0.01)
    assert gamma == pytest.approx(0.906, rel=0.01)
    c_star, _ = calibrate_confidence(*_noisy_planes(0.04, 0.9), sdf_trunc=0.005)
    assert c_star == pytest.approx(8.30, rel=0.01)


def test_calibrate_confidence_needs_neighbor_views():
    depth, conf, w2c, K = _noisy_planes(0.02, 0.9, n=3)
    with pytest.raises(ValueError, match="neighbor"):
        calibrate_confidence(depth, conf, w2c, K, sdf_trunc=0.005)


def test_calibrate_confidence_gate_scales_with_noise():
    # Doubling the noise moves the break-even confidence by 2^(1/gamma)
    c1, g1 = calibrate_confidence(*_noisy_planes(0.02, 0.9), sdf_trunc=0.005)
    c2, _ = calibrate_confidence(*_noisy_planes(0.04, 0.9), sdf_trunc=0.005)
    assert c2 / c1 == pytest.approx(2 ** (1 / g1), rel=0.05)


def test_calibrate_confidence_needs_three_bins():
    depth, conf, w2c, K = _noisy_planes(0.02, 0.9)
    with pytest.raises(ValueError, match="bins"):
        calibrate_confidence(depth, np.full_like(conf, 2.0), w2c, K, sdf_trunc=0.005)


def _flat_ground(voxel_size=0.02, sdf_trunc=0.08, height=0.0, rgb=200.0):
    """VolumeGround for the _views plane at z=1: up toward the cameras, every column inside."""
    res = 4 * voxel_size
    shape = (int(round(2.0 / res)), int(round(2.0 / res)))
    xs = np.arange(-1.0, 1.0, voxel_size)
    zs = np.arange(
        1.0 - height - 2 * sdf_trunc, 1.0 - height + 2 * sdf_trunc, voxel_size
    )
    band = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), -1).reshape(-1, 3)
    return VolumeGround(
        frame=np.diag([1.0, -1.0, -1.0]),
        center=np.array([0.0, 0.0, 1.0]),
        lo=np.array([-1.0, -1.0]),
        res=res,
        voxel_size=voxel_size,
        height=np.full(shape, height, np.float32),
        inside=np.ones((4 * shape[0], 4 * shape[1]), bool),
        color=np.full(shape + (3,), rgb, np.float32),
        blocks=np.unique(np.floor(band / (16 * voxel_size)).astype(np.int64), axis=0),
        sdf_trunc=sdf_trunc,
    )


def _near(mesh, xy, radius):
    """Vertices within radius of xy, top-down."""
    v = np.asarray(mesh.vertices)
    return np.linalg.norm(v[:, :2] - xy, axis=1) < radius


def test_create_tsdf_mesh_ground_closes_a_depth_hole(monkeypatch):
    # Every view misses the plane around x 0.05; the ground writes it at the plane's height, in its color
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=3)
    depths[:, 12:20, 12:20] = 0.0
    kwargs = dict(
        weights=np.ones_like(depths), voxel_size=0.02, depth_trunc=2.0, sdf_trunc=0.08
    )

    holed = create_tsdf_mesh(depths, rgbs, c2w, K, **kwargs)
    closed = create_tsdf_mesh(depths, rgbs, c2w, K, ground=_flat_ground(), **kwargs)

    assert not _near(holed, [0.05, 0.0], 0.05).any()
    middle = _near(closed, [0.05, 0.0], 0.05)
    assert middle.any()
    np.testing.assert_allclose(np.asarray(closed.vertices)[:, 2], 1.0, atol=0.02)
    np.testing.assert_allclose(np.asarray(closed.vertices)[middle, 2], 1.0, atol=0.005)
    np.testing.assert_allclose(
        np.asarray(closed.vertex_colors)[middle], 200 / 255, atol=0.05
    )


def test_create_tsdf_mesh_ground_leaves_observed_voxels_alone(monkeypatch):
    # A ground 0.04 off the plane: voxels three views saw keep their surface at z 1
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=3)
    kwargs = dict(
        weights=np.ones_like(depths), voxel_size=0.02, depth_trunc=2.0, sdf_trunc=0.08
    )

    mesh = create_tsdf_mesh(
        depths, rgbs, c2w, K, ground=_flat_ground(height=0.04), **kwargs
    )

    seen = _near(mesh, [-0.35, 0.0], 0.05)
    assert seen.any()
    np.testing.assert_allclose(np.asarray(mesh.vertices)[seen, 2], 1.0, atol=0.005)


def test_create_tsdf_mesh_ground_stays_out_of_the_hull(monkeypatch):
    # Columns past x 0.06 are outside the hull: that part of the hole stays open
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=3)
    depths[:, 12:20, 12:20] = 0.0
    ground = _flat_ground()
    ground.inside[:, 53:] = False

    mesh = create_tsdf_mesh(
        depths,
        rgbs,
        c2w,
        K,
        weights=np.ones_like(depths),
        voxel_size=0.02,
        depth_trunc=2.0,
        sdf_trunc=0.08,
        ground=ground,
    )

    assert _near(mesh, [0.01, 0.0], 0.02).any()
    assert not _near(mesh, [0.095, 0.0], 0.02).any()


def test_create_tsdf_mesh_ground_tiles_to_the_single_grid_mesh(monkeypatch, caplog):
    # The hole scene with its ground, fused in several tiles, welds to the single grid's mesh
    monkeypatch.setattr(o3d.core.cuda, "is_available", lambda: False)
    depths, rgbs, c2w, K = _views(n=3)
    depths[:, 12:20, 12:20] = 0.0
    kwargs = dict(
        weights=np.ones_like(depths),
        voxel_size=0.02,
        depth_trunc=2.0,
        sdf_trunc=0.08,
        ground=_flat_ground(),
    )

    single = create_tsdf_mesh(depths, rgbs, c2w, K, **kwargs)
    with caplog.at_level(logging.INFO, logger="collab_splats.mesh.tsdf"):
        tiled = create_tsdf_mesh(depths, rgbs, c2w, K, max_blocks=60, **kwargs)

    # The log reports the tile count; more than one
    tiles = re.search(r"in (\d+) tiles", caplog.text)
    assert tiles is not None and int(tiles.group(1)) > 1

    assert len(single.vertices) == len(tiled.vertices) > 0
    assert len(single.triangles) == len(tiled.triangles)
    single_v = np.unique(np.round(np.asarray(single.vertices), 6), axis=0)
    tiled_v = np.unique(np.round(np.asarray(tiled.vertices), 6), axis=0)
    assert np.array_equal(single_v, tiled_v)
