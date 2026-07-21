"""plot_correspondences: resolution-mismatch handling between query and reference."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from collab_splats.localization.localizer import LocalizationResult
from collab_splats.localization.viz import plot_correspondences


def _fake_result(n: int = 6) -> LocalizationResult:
    """Minimal localization result: n correspondences, all on reference frame 0."""
    rng = np.random.default_rng(0)
    return LocalizationResult(
        pose=np.eye(4, dtype=np.float32),
        n_correspondences=n,
        n_inliers=n,
        pts2d=rng.uniform(0, 50, (n, 2)).astype(np.float32),
        pts3d_matched=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        inlier_mask=np.ones(n, dtype=bool),
        pts2d_ref=rng.uniform(0, 30, (n, 2)).astype(np.float32),
        ref_frame_indices=np.zeros(n, dtype=np.int32),
    )


def test_plot_correspondences_handles_resolution_mismatch():
    """A 2988p-style query vs 1080p-style reference must plot, not raise on concat."""
    # Query taller than the reference (the GoPro-vs-reconstruction case, scaled down)
    query = np.zeros((120, 160, 3), dtype=np.uint8)
    ref_image = np.zeros((40, 60, 3), dtype=np.uint8)
    fig = plot_correspondences(
        _fake_result(), query, ref_image, ref_idx=0, max_pairs=10, show=False, warp_corners=False
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_correspondences_same_resolution_still_works():
    query = np.zeros((40, 60, 3), dtype=np.uint8)
    ref_image = np.zeros((40, 60, 3), dtype=np.uint8)
    fig = plot_correspondences(
        _fake_result(), query, ref_image, ref_idx=0, max_pairs=10, show=False, warp_corners=False
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
