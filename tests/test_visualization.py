import numpy as np
import torch
import pyvista as pv

from collab_splats.utils.visualization import create_camera_frustum_pyvista


def make_features(C=32, pH=14, pW=14):
    return torch.randn(C, pH, pW)


def make_image(H=224, W=224):
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)


def test_pca_to_rgb_output_shape():
    from collab_splats.utils.visualization import pca_to_rgb
    features = make_features()
    image = make_image()
    result = pca_to_rgb(features, image)
    assert result.shape == image.shape, f"Expected {image.shape}, got {result.shape}"
    assert result.dtype == np.uint8


def test_pca_to_rgb_values_in_range():
    from collab_splats.utils.visualization import pca_to_rgb
    result = pca_to_rgb(make_features(), make_image())
    assert result.min() >= 0 and result.max() <= 255


def test_compute_masked_image_shape():
    from collab_splats.utils.visualization import compute_masked_image
    image = make_image()
    sim_map = np.random.rand(image.shape[0], image.shape[1]).astype(np.float32)
    result = compute_masked_image(image, sim_map)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_compute_masked_image_blacks_low_sim():
    from collab_splats.utils.visualization import compute_masked_image
    image = (np.ones((4, 4, 3)) * 200).astype(np.uint8)
    # all zeros sim_map → all pixels below threshold → all black
    sim_map = np.zeros((4, 4), dtype=np.float32)
    result = compute_masked_image(image, sim_map, threshold=0.5)
    assert result.sum() == 0, "All pixels should be black"


def test_compute_masked_image_keeps_high_sim():
    from collab_splats.utils.visualization import compute_masked_image
    image = (np.ones((4, 4, 3)) * 200).astype(np.uint8)
    # all ones sim_map → all pixels above threshold → image unchanged
    sim_map = np.ones((4, 4), dtype=np.float32)
    result = compute_masked_image(image, sim_map, threshold=0.5)
    np.testing.assert_array_equal(result, image)


def test_overlay_masks_output_shape():
    from collab_splats.utils.visualization import overlay_masks
    image = make_image(H=64, W=64)
    masks = torch.zeros(3, 64, 64)
    masks[0, :32, :32] = 1.0
    masks[1, :32, 32:] = 1.0
    masks[2, 32:, :] = 1.0
    result = overlay_masks(image, masks)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_overlay_masks_values_in_range():
    from collab_splats.utils.visualization import overlay_masks
    image = make_image(H=32, W=32)
    masks = torch.ones(1, 32, 32)
    result = overlay_masks(image, masks)
    assert result.min() >= 0 and result.max() <= 255


def test_pointcloud_to_polydata_attaches_rgb():
    from collab_splats.utils.visualization import pointcloud_to_polydata
    pts3d = np.random.rand(100, 3).astype(np.float32)
    colors = (np.random.rand(100, 3) * 255).astype(np.uint8)
    cloud = pointcloud_to_polydata(pts3d, RGB=colors)
    assert cloud.n_points == 100
    assert "RGB" in cloud.array_names


def test_pointcloud_to_polydata_multiple_scalars():
    from collab_splats.utils.visualization import pointcloud_to_polydata
    pts3d = np.random.rand(50, 3).astype(np.float32)
    colors = (np.random.rand(50, 3) * 255).astype(np.uint8)
    scores = np.random.rand(50).astype(np.float32)
    cloud = pointcloud_to_polydata(pts3d, RGB=colors, similarity=scores)
    assert "RGB" in cloud.array_names
    assert "similarity" in cloud.array_names


def test_pointcloud_to_polydata_no_scalars():
    from collab_splats.utils.visualization import pointcloud_to_polydata
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
    from collab_splats.utils.visualization import _resolve_mesh_kwargs, PCD_KWARGS
    cloud = make_rgb_cloud()
    result = _resolve_mesh_kwargs(cloud, {})
    assert result == PCD_KWARGS


def test_resolve_mesh_kwargs_explicit_passthrough():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs
    cloud = make_rgb_cloud()
    explicit = {"scalars": "RGB", "rgb": True, "point_size": 3.0}
    result = _resolve_mesh_kwargs(cloud, explicit)
    assert result is explicit  # exact same dict object, no copy


def test_resolve_mesh_kwargs_bare_cloud_returns_empty():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs
    cloud = make_bare_cloud()
    result = _resolve_mesh_kwargs(cloud, {})
    assert result == {}


def test_resolve_mesh_kwargs_no_rgb_key_returns_empty():
    from collab_splats.utils.visualization import _resolve_mesh_kwargs
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
