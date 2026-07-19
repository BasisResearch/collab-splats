"""Tests for subsample_points (confidence filter + random count cap)."""

import numpy as np

from collab_splats.pointcloud.utils import subsample_points


def _cloud(n):
    rng = np.random.default_rng(0)
    return rng.random((n, 3)).astype(np.float32), rng.integers(0, 256, (n, 3), dtype=np.uint8)


def test_caps_count():
    pts, cols = _cloud(1000)
    out_pts, out_cols = subsample_points(pts, cols, max_points=100)
    assert out_pts.shape == (100, 3)
    assert out_cols.shape == (100, 3)


def test_noop_under_budget():
    pts, cols = _cloud(50)
    out_pts, out_cols = subsample_points(pts, cols, max_points=100)
    np.testing.assert_array_equal(out_pts, pts)
    np.testing.assert_array_equal(out_cols, cols)


def test_conf_filter_drops_low_conf():
    pts, cols = _cloud(100)
    conf = np.zeros(100, dtype=np.float32)
    conf[:20] = 1.0  # only first 20 points are confident
    out_pts, _ = subsample_points(pts, cols, conf=conf, max_points=1000, conf_percentile=50.0)
    # Everything at/below the 50th-percentile cutoff (the zeros) is dropped
    assert out_pts.shape[0] == 20
    assert set(map(tuple, out_pts)).issubset(set(map(tuple, pts[:20])))


def test_uniform_conf_keeps_all():
    pts, cols = _cloud(30)
    conf = np.ones(30, dtype=np.float32)
    out_pts, _ = subsample_points(pts, cols, conf=conf, max_points=1000)
    assert out_pts.shape[0] == 30


def test_colors_stay_aligned():
    pts, _ = _cloud(500)
    cols = np.repeat((np.arange(500) % 256).astype(np.uint8)[:, None], 3, axis=1)
    lookup = {tuple(p): c[0] for p, c in zip(pts, cols)}
    out_pts, out_cols = subsample_points(pts, cols, max_points=50)
    for p, c in zip(out_pts, out_cols):
        assert lookup[tuple(p)] == c[0]


def test_colors_none():
    pts, _ = _cloud(200)
    out_pts, out_cols = subsample_points(pts, None, max_points=50)
    assert out_pts.shape == (50, 3)
    assert out_cols is None
