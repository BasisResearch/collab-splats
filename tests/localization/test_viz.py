"""
Localization figures: correspondence plots across resolutions, inlier distribution.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from collab_splats.localization.localizer import LocalizationResult
from collab_splats.localization.viz import (
    correspondences_for_ref,
    plot_correspondences,
    plot_inlier_distribution,
)


def _single_ref_result(n: int = 6) -> LocalizationResult:
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
        query,
        ref_image,
        *correspondences_for_ref(_single_ref_result(), 0),
        max_pairs=10,
        show=False,
        warp_corners=False,
    )
    assert fig is not None
    plt.close(fig)


def test_plot_correspondences_plain_arrays():
    """Array-based signature — no LocalizationResult required."""
    rng = np.random.default_rng(2)
    q = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    r = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    q_px = rng.uniform(0, 79, (10, 2)).astype(np.float32)
    r_px = rng.uniform(0, 79, (10, 2)).astype(np.float32)
    fig = plot_correspondences(
        q, r, q_px, r_px, inlier_mask=np.arange(10) % 2 == 0, show=False
    )
    assert fig is not None


def test_plot_correspondences_warp_boxes_stay_in_their_panels():
    """Warped boxes are drawn as clipped lines; the view leaves a margin around both images."""
    rng = np.random.default_rng(3)
    query = np.zeros((60, 100, 3), dtype=np.uint8)
    ref_image = np.zeros((100, 60, 3), dtype=np.uint8)
    q_px = rng.uniform(0, 59, (20, 2)).astype(np.float32)

    # Reference is the query shifted by 5 px: a clean homography
    fig = plot_correspondences(
        query, ref_image, q_px, q_px + 5.0, warp_corners=True, show=False
    )
    ax = fig.axes[0]
    boxes = [line for line in ax.get_lines() if line.get_color() in ("cyan", "yellow")]
    assert len(boxes) == 2
    assert all(line.get_clip_box() is not ax.bbox for line in boxes)

    # Query spans x 0..100 with a 25 px margin; reference is scaled to height 60
    x_min, x_max = ax.get_xlim()
    assert x_min == -25.0
    assert x_max == pytest.approx(100 + 25 + 9 + 36 + 9)
    plt.close(fig)


def test_correspondences_for_ref_slices_one_frame():
    loc = LocalizationResult(
        pose=None,
        n_correspondences=4,
        n_inliers=3,
        pts2d=np.zeros((4, 2), np.float32),
        pts3d_matched=np.zeros((4, 3), np.float32),
        inlier_mask=np.array([True, False, True, True]),
        pts2d_ref=np.ones((4, 2), np.float32),
        ref_frame_indices=np.array([0, 1, 1, 0]),
    )
    q_px, r_px, mask = correspondences_for_ref(loc, 1)
    assert len(q_px) == 2 and mask.tolist() == [False, True]


def test_correspondences_for_ref_rescales_to_display_hw():
    """ref_px scales from loc.ref_hw space to the display image's resolution."""
    loc = LocalizationResult(
        pose=None,
        n_correspondences=2,
        n_inliers=0,
        pts2d=np.zeros((2, 2), np.float32),
        pts3d_matched=np.zeros((2, 3), np.float32),
        inlier_mask=None,
        pts2d_ref=np.array([[10.0, 20.0], [30.0, 40.0]], np.float32),
        ref_frame_indices=np.array([0, 0]),
        ref_hw=(100, 200),
    )
    # Display image is 2x the indexed resolution in both axes
    _, r_px, _ = correspondences_for_ref(loc, 0, ref_image_hw=(200, 400))
    np.testing.assert_allclose(r_px, [[20.0, 40.0], [60.0, 80.0]])
    # Same resolution (or absent ref_hw) → untouched
    _, r_same, _ = correspondences_for_ref(loc, 0, ref_image_hw=(100, 200))
    np.testing.assert_allclose(r_same, loc.pts2d_ref)


def test_plot_correspondences_same_resolution_still_works():
    query = np.zeros((40, 60, 3), dtype=np.uint8)
    ref_image = np.zeros((40, 60, 3), dtype=np.uint8)
    fig = plot_correspondences(
        query,
        ref_image,
        *correspondences_for_ref(_single_ref_result(), 0),
        max_pairs=10,
        show=False,
        warp_corners=False,
    )
    assert fig is not None
    plt.close(fig)


########################################################################
########## Inlier distribution and ranked-ref figures ##################
########################################################################


def _fake_result(n_frames=4, per_frame=10):
    """Synthetic result: frame i contributes per_frame correspondences, i+1 inliers."""
    m = n_frames * per_frame
    ref_idx = np.repeat(np.arange(n_frames, dtype=np.int32), per_frame)
    inlier = np.zeros(m, dtype=bool)
    for i in range(n_frames):
        inlier[i * per_frame : i * per_frame + i + 1] = True
    rng = np.random.default_rng(0)
    return LocalizationResult(
        pose=np.eye(4, dtype=np.float32),
        n_correspondences=m,
        n_inliers=int(inlier.sum()),
        pts2d=rng.uniform(0, 64, (m, 2)).astype(np.float32),
        pts3d_matched=rng.normal(size=(m, 3)).astype(np.float32),
        inlier_mask=inlier,
        pts2d_ref=rng.uniform(0, 64, (m, 2)).astype(np.float32),
        ref_frame_indices=ref_idx,
    )


def test_distribution_returns_figure_uniform_totals():
    loc = _fake_result()
    fig = plot_inlier_distribution(loc.ref_frame_indices, loc.inlier_mask, n_frames=4)
    assert fig is not None
    # Uniform totals → one dashed hline, no per-bar ticks
    ax = fig.axes[0]
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    plt.close(fig)


def test_distribution_per_bar_ticks_when_totals_vary():
    loc = _fake_result()
    # Drop 3 correspondences from frame 0 → totals no longer uniform
    keep = np.ones(len(loc.ref_frame_indices), dtype=bool)
    keep[:3] = False
    fig = plot_inlier_distribution(
        loc.ref_frame_indices[keep], loc.inlier_mask[keep], n_frames=4
    )
    ax = fig.axes[0]
    # No global dashed line; per-bar ticks drawn as solid short hlines
    assert not any(line.get_linestyle() == "--" for line in ax.get_lines())
    assert len(ax.get_lines()) > 0
    plt.close(fig)


def test_distribution_clamps_short_n_frames():
    loc = _fake_result(n_frames=4)
    # Stale/short count must not crash
    fig = plot_inlier_distribution(loc.ref_frame_indices, loc.inlier_mask, n_frames=2)
    assert fig is not None
    assert len(fig.axes[0].patches) >= 4  # bars cover every referenced frame
    plt.close(fig)


def test_correspondences_returns_figure_for_selected_ref():
    loc = _fake_result()  # frame 2 has 3 inliers of 10
    query = np.full((48, 64, 3), 100, dtype=np.uint8)
    ref_image = np.full((48, 64, 3), 60, dtype=np.uint8)
    fig = plot_correspondences(
        query, ref_image, *correspondences_for_ref(loc, 2), show=False
    )
    assert fig is not None
    assert "3/10 inliers" in fig.axes[0].get_title()
    plt.close(fig)


def test_correspondences_uses_ranked_ref_frame():
    loc = _fake_result()  # frame 3 has most inliers (4)
    query = np.full((48, 64, 3), 100, dtype=np.uint8)
    ref_image = np.full((48, 64, 3), 60, dtype=np.uint8)
    ri = loc.ranked_ref_frames[0]
    fig = plot_correspondences(
        query, ref_image, *correspondences_for_ref(loc, ri), show=False
    )
    assert "4/10 inliers" in fig.axes[0].get_title()
    plt.close(fig)
