"""Figure smoke tests for localization visualisations."""

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from collab_splats.localization.localizer import LocalizationResult
from collab_splats.localization.viz import (
    plot_correspondences,
    plot_inlier_distribution,
)


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


def _write_ref_images(tmp_path, n=4):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 60, dtype=np.uint8))
        paths.append(p)
    return paths


def test_distribution_returns_figure_uniform_totals():
    loc = _fake_result()
    fig = plot_inlier_distribution(loc, n_frames=4)
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
    loc = LocalizationResult(
        pose=loc.pose,
        n_correspondences=int(keep.sum()),
        n_inliers=loc.n_inliers,
        pts2d=loc.pts2d[keep],
        pts3d_matched=loc.pts3d_matched[keep],
        inlier_mask=loc.inlier_mask[keep],
        pts2d_ref=loc.pts2d_ref[keep],
        ref_frame_indices=loc.ref_frame_indices[keep],
    )
    fig = plot_inlier_distribution(loc, n_frames=4)
    ax = fig.axes[0]
    # No global dashed line; per-bar ticks drawn as solid short hlines
    assert not any(line.get_linestyle() == "--" for line in ax.get_lines())
    assert len(ax.get_lines()) > 0
    plt.close(fig)


def test_distribution_clamps_short_n_frames():
    loc = _fake_result(n_frames=4)
    fig = plot_inlier_distribution(loc, n_frames=2)  # stale/short count must not crash
    assert fig is not None
    assert len(fig.axes[0].patches) >= 4  # bars cover every referenced frame
    plt.close(fig)


def test_distribution_marks_localized_frames():
    loc = _fake_result()
    fig = plot_inlier_distribution(loc, n_frames=4, frame_sources=["reconstruction"] * 3 + ["localized"])
    assert fig is not None
    plt.close(fig)


def test_correspondences_returns_figure_and_accepts_ref_idx(tmp_path):
    loc = _fake_result()
    query = np.full((48, 64, 3), 100, dtype=np.uint8)
    paths = _write_ref_images(tmp_path)
    fig = plot_correspondences(loc, query, paths, ref_idx=2, show=False)
    assert fig is not None
    assert "frame 2" in fig.axes[0].get_title()
    plt.close(fig)


def test_correspondences_default_picks_best_frame(tmp_path):
    loc = _fake_result()  # frame 3 has most inliers (4)
    query = np.full((48, 64, 3), 100, dtype=np.uint8)
    paths = _write_ref_images(tmp_path)
    fig = plot_correspondences(loc, query, paths, show=False)
    assert "frame 3" in fig.axes[0].get_title()
    plt.close(fig)
