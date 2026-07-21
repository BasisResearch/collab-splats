"""seed_intrinsics: model-free COLMAP-style proportions seed for query K."""

import numpy as np

from collab_splats.localization.localizer import LocalizationResult, seed_intrinsics


def test_seed_landscape_focal_and_center():
    K = seed_intrinsics(480, 640)  # H, W
    f = 1.2 * 640  # 1.2 * max(W, H)
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 0], f)  # fx
    assert np.isclose(K[1, 1], f)  # fy (square pixels)
    assert np.isclose(K[0, 2], 320.0)  # cx = W/2
    assert np.isclose(K[1, 2], 240.0)  # cy = H/2
    assert K.dtype == np.float32


def test_seed_portrait_uses_max_dimension():
    K = seed_intrinsics(800, 600)  # H > W
    f = 1.2 * 800
    assert np.isclose(K[0, 0], f)
    assert np.isclose(K[1, 1], f)
    assert np.isclose(K[0, 2], 300.0)
    assert np.isclose(K[1, 2], 400.0)


def test_localizationresult_carries_intrinsics_field():
    K = seed_intrinsics(480, 640)
    r = LocalizationResult(
        pose=None,
        n_correspondences=0,
        n_inliers=0,
        pts2d=None,
        pts3d_matched=None,
        inlier_mask=None,
        query_intrinsics=K,
    )
    assert np.allclose(r.query_intrinsics, K)
