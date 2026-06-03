import numpy as np

from collab_splats.dashboard.viz_utils import apply_viridis, pointcloud_to_polydata


def test_polydata_has_points():
    pts = np.random.rand(10, 3).astype(np.float32)
    rgb = (np.random.rand(10, 3) * 255).astype(np.uint8)
    poly = pointcloud_to_polydata(pts, RGB=rgb)
    assert poly.n_points == 10


def test_apply_viridis_shape_and_dtype():
    sims = np.linspace(-1.0, 1.0, 12).astype(np.float32)
    rgb = apply_viridis(sims)
    assert rgb.shape == (12, 3)
    assert rgb.dtype == np.uint8
