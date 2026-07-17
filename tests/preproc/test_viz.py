import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from collab_splats.preproc.viz import (
    plot_disparity_sensitivity,
    plot_frame_grid,
    plot_frame_scores,
    plot_quality_examples,
    plot_selection,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fake_records(n=30):
    """Score records shaped like score_frames output."""
    rng = np.random.default_rng(0)
    reasons = [None, "blur", "exposure"]
    return [
        {
            "frame_idx": i,
            "blur_score": 200.0,
            "exposure_mean": 120.0,
            "exposure_std": 40.0,
            "reject_reason": reasons[i % 3],
            "disparity": float(rng.random() * 80),
            "rotation": float(rng.random() * 3),
            "histogram_similarity": float(rng.random()),
            "score": float(rng.random()),
            "selected": i % 5 == 0,
        }
        for i in range(n)
    ]


def test_plot_frame_grid_smoke():
    frames = [np.zeros((24, 32, 3), dtype=np.uint8)] * 4
    plot_frame_grid(frames, "grid")
    assert plt.gcf() is not None


def test_plot_selection_both_sets():
    plot_selection(100, fps_indices=[0, 10, 20], of_indices=[0, 5, 30])
    assert len(plt.gcf().axes) == 2


def test_plot_frame_scores_smoke():
    plot_frame_scores(_fake_records())
    assert len(plt.gcf().axes) == 3


def test_plot_frame_scores_empty_input():
    plot_frame_scores([])  # must not raise


def test_plot_disparity_sensitivity_monotonic():
    # Higher disparity threshold → same or fewer frames selected
    records = _fake_records(60)
    plot_disparity_sensitivity(records, [10.0, 50.0, 200.0])
    ax = plt.gcf().axes[0]
    counts = ax.lines[0].get_ydata()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


def test_plot_quality_examples_three_rows(monkeypatch):
    # Stub decode: viz must only ask for the frames it displays
    fake = lambda _path, idxs: [np.zeros((24, 32, 3), dtype=np.uint8) for _ in idxs]
    monkeypatch.setattr("collab_splats.preproc.viz.load_frames", fake)
    plot_quality_examples("unused.mp4", _fake_records(), n_examples=3)
    # One row per non-empty category (accepted / blur / exposure), n_examples cols
    assert len(plt.gcf().axes) == 9


def test_plot_quality_examples_skips_empty_categories(monkeypatch):
    fake = lambda _path, idxs: [np.zeros((24, 32, 3), dtype=np.uint8) for _ in idxs]
    monkeypatch.setattr("collab_splats.preproc.viz.load_frames", fake)
    records = [d for d in _fake_records() if d["reject_reason"] != "exposure"]
    plot_quality_examples("unused.mp4", records, n_examples=3)
    assert len(plt.gcf().axes) == 6  # accepted + blur rows only


def test_plot_quality_examples_empty_input():
    plot_quality_examples("unused.mp4", [])  # must not raise
