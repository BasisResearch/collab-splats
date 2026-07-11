import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from collab_splats.utils.visualization import compute_heatmap


def _make_image(h=100, w=120):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _make_sim_map(h=100, w=120):
    rng = np.random.default_rng(1)
    return rng.random((h, w)).astype(np.float32)


class TestComputeHeatmap:
    def test_output_shape_same_size(self):
        image = _make_image(100, 120)
        sim_map = _make_sim_map(100, 120)
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_output_dtype_uint8(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map)
        assert result.dtype == np.uint8

    def test_output_values_in_range(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_resize_sim_map_to_image_size(self):
        image = _make_image(100, 120)
        sim_map = _make_sim_map(20, 24)  # patch grid, smaller
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_squeeze_hw1_input(self):
        image = _make_image(100, 120)
        sim_map = _make_sim_map(100, 120).reshape(100, 120, 1)
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)

    def test_torch_tensor_input(self):
        torch = pytest.importorskip("torch")
        image = _make_image()
        sim_map = torch.from_numpy(_make_sim_map())
        result = compute_heatmap(image, sim_map)
        assert result.shape == (100, 120, 3)
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_image(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map, alpha=0.0)
        np.testing.assert_array_equal(result, image)

    def test_alpha_one_returns_heatmap_only(self):
        import matplotlib.pyplot as plt

        image = _make_image()
        sim_map = _make_sim_map()
        result = compute_heatmap(image, sim_map, alpha=1.0)
        # Independently compute expected colormap output
        s = sim_map.astype(float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        cmap = plt.get_cmap("viridis")
        expected = (cmap(s)[:, :, :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_constant_sim_map_uniform_output(self):
        image = _make_image()
        sim_map = np.ones((100, 120), dtype=np.float32)
        result = compute_heatmap(image, sim_map, alpha=1.0)
        assert result.shape == (100, 120, 3)
        # Constant input → uniform colormap → all pixels identical
        assert (result == result[0, 0]).all()

    def test_custom_colormap(self):
        image = _make_image()
        sim_map = _make_sim_map()
        result_viridis = compute_heatmap(image, sim_map, colormap="viridis")
        result_plasma = compute_heatmap(image, sim_map, colormap="plasma")
        assert result_viridis.shape == (100, 120, 3)
        assert result_plasma.shape == (100, 120, 3)
        assert not np.array_equal(result_viridis, result_plasma)
