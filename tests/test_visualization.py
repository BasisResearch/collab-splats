import numpy as np
import torch


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
