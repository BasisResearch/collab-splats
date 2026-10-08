"""
render_points and plot_reprojection: the localized-query reprojection plot.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from collab_splats.utils.visualization import plot_reprojection, render_points

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
HW = (48, 64)
IDENTITY = np.eye(4)


def test_single_point_lands_at_expected_pixel():
    # x=0.2, y=-0.1 at z=2 projects to u = 32 + 100*0.1 = 42, v = 24 - 100*0.05 = 19
    points = np.array([[0.2, -0.1, 2.0]])
    colors = np.array([[1.0, 0.0, 0.0]])

    rgb, depth = render_points(points, colors, IDENTITY, K, HW, radius=0)

    assert rgb.shape == (48, 64, 3)
    assert depth.shape == (48, 64)
    np.testing.assert_allclose(rgb[19, 42], [1.0, 0.0, 0.0])
    assert depth[19, 42] == 2.0
    assert np.isfinite(depth).sum() == 1
    assert np.all(rgb[np.isinf(depth)] == 1.0)


def test_radius_writes_square_footprint():
    points = np.array([[0.0, 0.0, 1.0]])
    colors = np.array([[0.0, 0.0, 1.0]])

    _, depth = render_points(points, colors, IDENTITY, K, HW, radius=1)

    assert np.isfinite(depth).sum() == 9
    assert np.isfinite(depth[23:26, 31:34]).all()


def test_nearer_point_wins_in_either_order():
    near = [0.0, 0.0, 1.0]
    far = [0.0, 0.0, 3.0]
    red = [1.0, 0.0, 0.0]
    green = [0.0, 1.0, 0.0]

    for points, colors in [([near, far], [red, green]), ([far, near], [green, red])]:
        rgb, depth = render_points(
            np.array(points), np.array(colors), IDENTITY, K, HW, radius=1
        )

        np.testing.assert_allclose(rgb[24, 32], red)
        assert depth[24, 32] == 1.0


def test_points_behind_camera_are_dropped():
    points = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    rgb, depth = render_points(points, colors, IDENTITY, K, HW)

    assert np.isinf(depth).all()
    assert np.all(rgb == 1.0)


def test_uint8_colors_match_float_colors():
    rng = np.random.default_rng(1)
    points = rng.uniform([-1, -1, 2], [1, 1, 4], size=(200, 3))
    colors_u8 = rng.integers(0, 256, size=(200, 3), dtype=np.uint8)
    colors_f = colors_u8 / 255.0

    rgb_u8, depth_u8 = render_points(points, colors_u8, IDENTITY, K, HW)
    rgb_f, depth_f = render_points(points, colors_f, IDENTITY, K, HW)

    np.testing.assert_allclose(rgb_u8, rgb_f)
    np.testing.assert_array_equal(depth_u8, depth_f)


def test_plot_reprojection_three_panels_downscaled():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    K_full = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points = rng.uniform([-1, -1, 2], [1, 1, 4], size=(500, 3))
    colors = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)

    fig = plot_reprojection(
        image, points, colors, IDENTITY, K_full, max_width=320, title="check"
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert ax.images[0].get_array().shape[:2] == (240, 320)
    assert "coverage" in fig.axes[1].get_title()
    assert fig.get_suptitle() == "check"
