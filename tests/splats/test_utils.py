"""
splats.utils: scene scale, Sim3 normalization, the coarse-to-fine schedule and per-view targets.
"""

from itertools import islice

import numpy as np
import pytest
import torch

from collab_splats.splats.utils import (
    cached_target,
    compute_scene_scale,
    denormalize_cameras,
    downscale_factor,
    downscale_image,
    knn_spacing,
    prepare_target,
    scene_normalization,
    view_order,
)


def _ring_cameras(n_views=4, radius=2.0):
    """
    n_views camera-to-world poses on a ring of the given radius in the xz plane.
    """
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(n_views)])

    for view in range(n_views):
        angle = 2 * np.pi * view / n_views
        poses[view, :3, 3] = [radius * np.cos(angle), 0.0, radius * np.sin(angle)]

    return poses


def test_compute_scene_scale_is_margin_times_max_radius():
    poses = torch.from_numpy(_ring_cameras(n_views=4, radius=2.0))
    # Centroid is the origin, every camera sits at radius 2, default margin 1.1
    assert compute_scene_scale(poses) == pytest.approx(2.2, rel=1e-6)


def test_compute_scene_scale_margin_is_keyword_only():
    poses = torch.from_numpy(_ring_cameras(n_views=4, radius=2.0))
    assert compute_scene_scale(poses, margin=1.0) == pytest.approx(2.0, rel=1e-6)

    # Signature is (cam_to_world, *, margin), so a positional second argument is a TypeError
    # - the count string is pinned, so no unrelated TypeError can pass this
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        compute_scene_scale(poses, 1.0)


def test_scene_normalization_centers_and_scales_to_unit_linf():
    poses = _ring_cameras(n_views=4, radius=2.0)
    poses[:, :3, 3] += np.array([10.0, 0.0, -5.0], dtype=np.float32)
    center, scale = scene_normalization(poses)

    assert center == pytest.approx([10.0, 0.0, -5.0], abs=1e-5)
    # L-inf spread of the ring is the radius, so the scale is its reciprocal
    assert scale == pytest.approx(0.5, rel=1e-6)


def test_scene_normalization_centers_an_asymmetric_camera_set():
    cam_to_world = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    cam_to_world[:, :3, 3] = [[0, 0, 0], [4, 0, 0], [2, 6, -2]]
    center, scale = scene_normalization(cam_to_world)

    # Center = mean position; scale = 1 / max |coord - center| (L-inf, splatfacto auto_scale_poses)
    np.testing.assert_allclose(center, [2, 2, -2 / 3], rtol=1e-6)
    normalized = (cam_to_world[:, :3, 3] - center) * scale
    assert np.isclose(np.abs(normalized).max(), 1.0)


def test_scene_normalization_rejects_coincident_cameras():
    poses = np.stack([np.eye(4, dtype=np.float32)] * 3)

    with pytest.raises(ValueError, match="cameras coincide"):
        scene_normalization(poses)


def test_denormalize_cameras_inverts_scene_normalization():
    poses = _ring_cameras(n_views=4, radius=2.0)
    poses[:, :3, 3] += np.array([10.0, 0.0, -5.0], dtype=np.float32)
    center, scale = scene_normalization(poses)

    normalized = torch.from_numpy(poses.copy())
    normalized[:, :3, 3] = (normalized[:, :3, 3] - torch.from_numpy(center)) * scale
    denormalize_cameras(normalized, center, scale)

    assert torch.allclose(normalized, torch.from_numpy(poses), atol=1e-5)


def test_downscale_factor_walks_the_coarse_to_fine_schedule():
    # num_downscales=2, resolution_schedule=3000: 4x until 3000, 2x until 6000, 1x after
    assert downscale_factor(0, 2, 3000) == 4
    assert downscale_factor(2999, 2, 3000) == 4
    assert downscale_factor(3000, 2, 3000) == 2
    assert downscale_factor(5999, 2, 3000) == 2
    assert downscale_factor(6000, 2, 3000) == 1
    assert downscale_factor(29999, 2, 3000) == 1
    assert downscale_factor(99999, 2, 3000) == 1


def test_downscale_factor_is_one_when_disabled():
    assert downscale_factor(0, 0, 3000) == 1


def test_downscale_image_halves_the_image():
    image = np.zeros((64, 32, 3), np.uint8)
    assert downscale_image(image, 2).shape == (32, 16, 3)


def test_downscale_image_floors_an_odd_size():
    image = np.zeros((45, 63, 3), np.uint8)
    assert downscale_image(image, 4).shape == (11, 15, 3)


def test_downscale_image_passes_through_at_factor_one():
    image = np.zeros((8, 8, 3), np.uint8)
    assert downscale_image(image, 1) is image


def test_prepare_target_scales_rgb_to_unit_range():
    image = np.full((4, 6, 3), 255, np.uint8)
    target = prepare_target(image, None, "cpu")

    assert target["rgb"].shape == (1, 4, 6, 3)
    assert torch.allclose(target["rgb"], torch.ones(1, 4, 6, 3))
    assert target["depth"] is None


def test_prepare_target_resizes_depth_to_the_image_grid():
    image = np.zeros((8, 8, 3), np.uint8)
    depth = np.full((4, 4), 2.5, np.float32)
    target = prepare_target(image, depth, "cpu")

    assert target["depth"].shape == (1, 8, 8, 1)
    assert torch.allclose(target["depth"], torch.full((1, 8, 8, 1), 2.5))


def test_prepare_target_keeps_missing_depth_exactly_zero():
    image = np.full((8, 8, 3), 255, np.uint8)
    depth = np.array([[1.0, 0.0], [2.0, 3.0]], np.float32)
    target = prepare_target(image, depth, "cpu")

    # Nearest resize: the 0 quadrant ("no target") must stay 0 rather than blend with its neighbors
    assert target["depth"].shape == (1, 8, 8, 1)
    assert target["depth"][0, 0, 0, 0] == 1.0
    assert target["depth"][0, 0, 7, 0] == 0.0
    assert target["depth"][0, 7, 7, 0] == 3.0
    # The corners alone are bit-identical under bilinear; an interior pixel and the count are not
    # - measured: bilinear puts 1.15625 at (3, 5) and leaves 4 exact zeros, not 16
    assert target["depth"][0, 3, 5, 0] == 0.0
    assert int((target["depth"] == 0).sum()) == 16


def _views(n_views=3, height=37, width=22):
    """
    Random uint8 frames (odd sizes, so each factor floors) and coarser depth with holes.
    """
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (n_views, height, width, 3), dtype=np.uint8)
    depths = rng.uniform(0.5, 4.0, (n_views, 9, 6)).astype(np.float32)
    depths[:, ::3, ::2] = 0.0
    return images, depths


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_cached_target_matches_the_per_step_build(factor):
    images, depths = _views()
    cache = {}

    for view in range(len(images)):
        cached = cached_target(cache, images, depths, view, factor, "cpu")
        expected = prepare_target(
            downscale_image(images[view], factor), depths[view], "cpu"
        )

        assert torch.equal(cached["rgb"], expected["rgb"])
        assert torch.equal(cached["depth"], expected["depth"])


def test_cached_target_builds_each_view_once_per_factor():
    images, _ = _views()
    cache = {}
    first = cached_target(cache, images, None, 1, 2, "cpu")

    assert cached_target(cache, images, None, 1, 2, "cpu") is first
    assert first["depth"] is None


def test_cached_target_drops_the_previous_factor():
    images, depths = _views()
    cache = {}

    for view in range(len(images)):
        cached_target(cache, images, depths, view, 4, "cpu")

    target = cached_target(cache, images, depths, 0, 2, "cpu")

    assert list(cache) == [2]
    assert list(cache[2]) == [0]
    assert target["rgb"].shape == (1, 18, 11, 3)


def test_view_order_reproduces_the_measured_shuffle_and_pop_sequence():
    # Measured against the ViewSampler this replaces: random.Random(42), shuffle, pop from the end
    assert list(islice(view_order(4), 12)) == [0, 3, 1, 2, 1, 0, 2, 3, 0, 2, 3, 1]


def test_view_order_reproduces_the_sequence_for_an_odd_view_count():
    assert list(islice(view_order(3), 9)) == [2, 0, 1, 0, 1, 2, 0, 2, 1]


def test_view_order_visits_every_view_once_per_epoch():
    drawn = list(islice(view_order(5), 15))

    for start in (0, 5, 10):
        assert sorted(drawn[start : start + 5]) == [0, 1, 2, 3, 4]


def test_view_order_seed_is_a_keyword_argument():
    assert list(islice(view_order(4, seed=7), 4)) != list(
        islice(view_order(4, seed=42), 4)
    )

    # Keyword-ONLY: a positional seed must not bind, or the name is not pinned
    # - the count string is pinned, so no unrelated TypeError can pass this
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        view_order(4, 7)


def test_knn_spacing_is_the_rms_distance_to_the_k_nearest_neighbors():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (50, 3))
    spacing = knn_spacing(points, 3)

    # Brute force: sorted pairwise distances, self (0) excluded
    dists = np.sort(np.linalg.norm(points[:, None] - points[None], axis=-1), axis=1)[
        :, 1:4
    ]
    assert spacing.shape == (50,)
    assert np.allclose(spacing, np.sqrt((dists**2).mean(-1)))
