"""
Visualization helpers: reprojection plot, PCA/mask overlays, polydata, camera frustum, heatmap.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.figure import Figure

from collab_splats.utils.visualization import (
    PCD_KWARGS,
    _resolve_mesh_kwargs,
    camera_view,
    compute_heatmap,
    compute_masked_image,
    create_camera_frustum_pyvista,
    overlay_masks,
    pca_to_rgb,
    plot_reprojection,
    pointcloud_to_polydata,
    render_points,
    visualize_splat,
)

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


def test_plot_reprojection_rescales_pixel_center_k():
    # Center-K point at full px (15, 11); half scale puts it at (15.5 * 0.5 - 0.5, 11.5 * 0.5 - 0.5) = (7.25, 5.25)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    u, v, z = 15.0, 11.0, 2.0
    points = np.array([[(u - K[0, 2]) / 100 * z, (v - K[1, 2]) / 100 * z, z]])
    colors = np.array([[1.0, 0.0, 0.0]])

    fig = plot_reprojection(image, points, colors, IDENTITY, K, max_width=32, radius=0)

    render = np.asarray(fig.axes[1].images[0].get_array())
    hit = np.argwhere(render[..., 1] < 0.5)
    np.testing.assert_array_equal(hit, [[5, 7]])


########################################################################
########## PCA, masks and polydata #####################################
########################################################################


def make_features(C=32, pH=14, pW=14):
    return torch.randn(C, pH, pW)


def make_image(H=224, W=224):
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)


def test_pca_to_rgb_output_shape():
    features = make_features()
    image = make_image()
    result = pca_to_rgb(features, image)
    assert result.shape == image.shape, f"Expected {image.shape}, got {result.shape}"
    assert result.dtype == np.uint8


def test_pca_to_rgb_values_in_range():
    result = pca_to_rgb(make_features(), make_image())
    assert result.min() >= 0 and result.max() <= 255


def test_compute_masked_image_shape():
    image = make_image()
    sim_map = np.random.rand(image.shape[0], image.shape[1]).astype(np.float32)
    result = compute_masked_image(image, sim_map)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_compute_masked_image_blacks_low_sim():
    image = (np.ones((4, 4, 3)) * 200).astype(np.uint8)
    # all zeros sim_map → all pixels below threshold → all black
    sim_map = np.zeros((4, 4), dtype=np.float32)
    result = compute_masked_image(image, sim_map, threshold=0.5)
    assert result.sum() == 0, "All pixels should be black"


def test_compute_masked_image_keeps_high_sim():
    image = (np.ones((4, 4, 3)) * 200).astype(np.uint8)
    # all ones sim_map → all pixels above threshold → image unchanged
    sim_map = np.ones((4, 4), dtype=np.float32)
    result = compute_masked_image(image, sim_map, threshold=0.5)
    np.testing.assert_array_equal(result, image)


def test_overlay_masks_output_shape():
    image = make_image(H=64, W=64)
    masks = torch.zeros(3, 64, 64)
    masks[0, :32, :32] = 1.0
    masks[1, :32, 32:] = 1.0
    masks[2, 32:, :] = 1.0
    result = overlay_masks(image, masks)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_overlay_masks_values_in_range():
    image = make_image(H=32, W=32)
    masks = torch.ones(1, 32, 32)
    result = overlay_masks(image, masks)
    assert result.min() >= 0 and result.max() <= 255


def test_pointcloud_to_polydata_attaches_rgb():
    pts3d = np.random.rand(100, 3).astype(np.float32)
    colors = (np.random.rand(100, 3) * 255).astype(np.uint8)
    cloud = pointcloud_to_polydata(pts3d, RGB=colors)
    assert cloud.n_points == 100
    assert "RGB" in cloud.array_names


def test_pointcloud_to_polydata_multiple_scalars():
    pts3d = np.random.rand(50, 3).astype(np.float32)
    colors = (np.random.rand(50, 3) * 255).astype(np.uint8)
    scores = np.random.rand(50).astype(np.float32)
    cloud = pointcloud_to_polydata(pts3d, RGB=colors, similarity=scores)
    assert "RGB" in cloud.array_names
    assert "similarity" in cloud.array_names


def test_pointcloud_to_polydata_no_scalars():
    pts3d = np.zeros((10, 3), dtype=np.float32)
    cloud = pointcloud_to_polydata(pts3d)
    assert cloud.n_points == 10
    assert cloud.array_names == []


def make_rgb_cloud(N=100):
    pts = np.random.rand(N, 3).astype(np.float32)
    colors = (np.random.rand(N, 3) * 255).astype(np.uint8)
    cloud = pv.PolyData(pts)
    cloud["RGB"] = colors
    return cloud


def make_bare_cloud(N=100):
    pts = np.random.rand(N, 3).astype(np.float32)
    return pv.PolyData(pts)


def test_resolve_mesh_kwargs_rgb_returns_pcd_kwargs():
    cloud = make_rgb_cloud()
    result = _resolve_mesh_kwargs(cloud, {})
    assert result == PCD_KWARGS


def test_resolve_mesh_kwargs_explicit_passthrough():
    cloud = make_rgb_cloud()
    explicit = {"scalars": "RGB", "rgb": True, "point_size": 3.0}
    result = _resolve_mesh_kwargs(cloud, explicit)
    assert result is explicit  # exact same dict object, no copy


def test_resolve_mesh_kwargs_bare_cloud_returns_empty():
    cloud = make_bare_cloud()
    result = _resolve_mesh_kwargs(cloud, {})
    assert result == {}


def test_resolve_mesh_kwargs_no_rgb_key_returns_empty():
    # pv.Sphere() is a PolyData with faces but no "RGB" array — same "no RGB key" path as bare_cloud
    mesh = pv.Sphere()
    result = _resolve_mesh_kwargs(mesh, {})
    assert result == {}


########################################
####### Camera frustum #################
########################################


def test_frustum_apex_at_origin_for_identity_w2c():
    """Identity w2c → camera at world origin → apex at [0, 0, 0]."""
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    assert np.allclose(frustum.points[0], [0.0, 0.0, 0.0], atol=1e-5)


def test_frustum_apex_at_camera_position():
    """Camera translated to [1, 2, 3] → apex at [1, 2, 3] in world space."""
    # w2c with camera at world [1, 2, 3]: R=I, t = -R @ p = [-1, -2, -3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 3] = [-1.0, -2.0, -3.0]
    frustum = create_camera_frustum_pyvista(w2c)
    assert np.allclose(frustum.points[0], [1.0, 2.0, 3.0], atol=1e-5)


def test_frustum_near_plane_at_positive_z_for_identity():
    """OpenCV convention: identity w2c → camera looks +Z → near/far at +Z."""
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    # Vertices 1-4: near plane; 5-8: far plane
    assert np.all(frustum.points[1:5, 2] > 0), "near plane z must be positive"
    assert np.all(frustum.points[5:9, 2] > 0), "far plane z must be positive"


def test_frustum_rectangles_closed():
    """Near and far plane rectangles must be closed (48 total line array entries)."""
    # Closed rects: near=[5,1,2,3,4,1] + far=[5,5,6,7,8,5] = 12 entries total
    # Unclosed rects: near=[4,1,2,3,4] + far=[4,5,6,7,8] = 10 entries total
    # Other lines (apex→near×4, apex→far×4, near→far×4) = 36 entries
    # Total closed = 48, unclosed = 46
    w2c = np.eye(4, dtype=np.float32)
    frustum = create_camera_frustum_pyvista(w2c)
    assert len(frustum.lines) == 48, (
        f"Expected 48 line array entries (closed rects), got {len(frustum.lines)}"
    )


########################################
####### Camera view ####################
########################################


def _walk_poses(n: int = 5) -> np.ndarray:
    """
    w2c poses of a camera stepping along world +x, looking along +z with OpenCV y down.
    """
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = -np.arange(n, dtype=float)
    return poses


def test_camera_view_up_is_opposite_opencv_y():
    view = camera_view(_walk_poses())
    np.testing.assert_allclose(view["view_up"], [0, -1, 0], atol=1e-6)


def test_camera_view_keys_reset_apply_view_defaults():
    view = camera_view(_walk_poses())
    assert {"position", "focal_point", "view_up"} <= set(view)
    assert view["azimuth"] == 0 and view["elevation"] == 0


def test_camera_view_looks_from_the_side():
    # Walking along +x and looking along +z: the side eye sits along x from the focal point
    view = camera_view(_walk_poses())
    offset = np.asarray(view["position"]) - np.asarray(view["focal_point"])
    assert abs(offset[0]) > abs(offset[2])


def test_camera_view_points_set_focal_point():
    points = np.random.default_rng(0).normal(loc=[3, 0, 10], size=(1000, 3))
    view = camera_view(_walk_poses(), points)
    np.testing.assert_allclose(view["focal_point"], [3, 0, 10], atol=0.3)


########################################
####### visualize_splat ################
########################################


def test_visualize_splat_time_colored_frustums_get_scalar_bar():
    cloud = pv.PolyData(np.random.default_rng(0).normal(size=(50, 3)))
    plotter = visualize_splat(cloud, _walk_poses(), camera_kwargs={"n_poses": 1})
    assert "frame order" in plotter.scalar_bars
    plotter.close()


def test_visualize_splat_fixed_color_frustums_have_no_scalar_bar():
    cloud = pv.PolyData(np.random.default_rng(0).normal(size=(50, 3)))
    plotter = visualize_splat(cloud, _walk_poses(), camera_kwargs={"color": "red"})
    assert "frame order" not in plotter.scalar_bars
    plotter.close()


########################################################################
########## compute_heatmap #############################################
########################################################################


def _make_image(h=100, w=120):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _make_sim_map(h=100, w=120):
    rng = np.random.default_rng(1)
    return rng.random((h, w)).astype(np.float32)


def test_output_shape_same_size():
    image = _make_image(100, 120)
    sim_map = _make_sim_map(100, 120)
    result = compute_heatmap(image, sim_map)
    assert result.shape == (100, 120, 3)


def test_output_dtype_uint8():
    image = _make_image()
    sim_map = _make_sim_map()
    result = compute_heatmap(image, sim_map)
    assert result.dtype == np.uint8


def test_output_values_in_range():
    image = _make_image()
    sim_map = _make_sim_map()
    result = compute_heatmap(image, sim_map)
    assert result.min() >= 0
    assert result.max() <= 255


def test_resize_sim_map_to_image_size():
    image = _make_image(100, 120)
    sim_map = _make_sim_map(20, 24)  # patch grid, smaller
    result = compute_heatmap(image, sim_map)
    assert result.shape == (100, 120, 3)


def test_squeeze_hw1_input():
    image = _make_image(100, 120)
    sim_map = _make_sim_map(100, 120).reshape(100, 120, 1)
    result = compute_heatmap(image, sim_map)
    assert result.shape == (100, 120, 3)


def test_torch_tensor_input():
    image = _make_image()
    sim_map = torch.from_numpy(_make_sim_map())
    result = compute_heatmap(image, sim_map)
    assert result.shape == (100, 120, 3)
    assert result.dtype == np.uint8


def test_alpha_zero_returns_image():
    image = _make_image()
    sim_map = _make_sim_map()
    result = compute_heatmap(image, sim_map, alpha=0.0)
    np.testing.assert_array_equal(result, image)


def test_alpha_one_returns_heatmap_only():
    image = _make_image()
    sim_map = _make_sim_map()
    result = compute_heatmap(image, sim_map, alpha=1.0)
    # Independently compute expected colormap output
    s = sim_map.astype(float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    cmap = plt.get_cmap("viridis")
    expected = (cmap(s)[:, :, :3] * 255).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_constant_sim_map_uniform_output():
    image = _make_image()
    sim_map = np.ones((100, 120), dtype=np.float32)
    result = compute_heatmap(image, sim_map, alpha=1.0)
    assert result.shape == (100, 120, 3)
    # Constant input → uniform colormap → all pixels identical
    assert (result == result[0, 0]).all()


def test_custom_colormap():
    image = _make_image()
    sim_map = _make_sim_map()
    result_viridis = compute_heatmap(image, sim_map, colormap="viridis")
    result_plasma = compute_heatmap(image, sim_map, colormap="plasma")
    assert result_viridis.shape == (100, 120, 3)
    assert result_plasma.shape == (100, 120, 3)
    assert not np.array_equal(result_viridis, result_plasma)
