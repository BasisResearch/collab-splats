import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from collab_splats.preproc import viz as viz_module

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.viz import (
    plot_correlation,
    plot_disparity_sensitivity,
    plot_frame_extremes,
    plot_frame_grid,
    plot_frame_scores,
    plot_motion,
    plot_photometric,
    plot_quality_examples,
    plot_selection,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fake_records(n=30):
    """
    Score records shaped like sample_optical_flow output.
    """
    rng = np.random.default_rng(0)
    return [
        {
            "frame_idx": i,
            "blur_score": 200.0,
            "disparity": float(rng.random() * 80),
            "rotation": float(rng.random() * 3),
            "histogram_similarity": float(rng.random()),
            "score": float(rng.random()),
            "selected": i % 5 == 0,
        }
        for i in range(n)
    ]


def _fake_report(n=30, soft=(), badly_exposed=()):
    """
    A compute_video_quality-shaped report; `soft` frames fail sharpness, `badly_exposed` exposure.
    """
    frames = {
        "frame_idx": list(range(n)),
        "laplacian": [200.0] * n,
        "exposure_mean": [120.0] * n,
        "exposure_std": [40.0] * n,
        "blur": [0.3] * n,
    }
    for i in soft:
        frames["laplacian"][i] = 1.0
    for i in badly_exposed:
        frames["exposure_mean"][i] = 250.0

    return {"available": True, "frames": frames}


def _fake_store(tmp_path, n=30, h=24, w=32):
    """
    FrameStore covering source indices 0..n-1; each frame filled with its own index value.
    """
    frames = [np.full((h, w, 3), i % 256, dtype=np.uint8) for i in range(n)]
    records = [{"frame_idx": i, "blur_score": float(i)} for i in range(n)]
    FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance={"video_path": "v.mp4"})
    return FrameStore.open(tmp_path / "frames.zarr")


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


def test_plot_quality_examples_three_rows(tmp_path):
    store = _fake_store(tmp_path)
    report = _fake_report(soft=(1, 4, 7, 10), badly_exposed=(2, 5, 8, 11))
    plot_quality_examples(store, report, n_examples=3)
    # One row per non-empty category (usable / soft / exposure), n_examples cols
    axes = plt.gcf().axes
    assert len(axes) == 9
    # Displayed pixels for each axis must match the store's frame at that source index
    for ax in axes:
        frame_idx = int(ax.get_title().split()[0].lstrip("#"))
        displayed = np.asarray(ax.images[0].get_array())
        np.testing.assert_array_equal(displayed, store.image_by_frame_idx(frame_idx))


def test_plot_quality_examples_skips_empty_categories(tmp_path):
    store = _fake_store(tmp_path)
    report = _fake_report(soft=(1, 4, 7, 10))  # nothing badly exposed
    plot_quality_examples(store, report, n_examples=3)
    assert len(plt.gcf().axes) == 6  # usable + soft rows only
    row_labels = {t.get_text() for ax in plt.gcf().axes for t in ax.texts}
    assert "Usable" in row_labels
    assert "Rejected: soft" in row_labels
    assert "Rejected: exposure" not in row_labels


def test_plot_quality_examples_reads_only_frames_the_store_holds(tmp_path):
    # The report covers the whole video; frames.zarr holds only the keyframes, so
    # a plot that indexed the report directly would raise on the first gap.
    store = _fake_store(tmp_path, n=4)
    plot_quality_examples(store, _fake_report(n=30), n_examples=3)
    titles = [int(ax.get_title().split()[0].lstrip("#")) for ax in plt.gcf().axes]
    assert titles and all(i < 4 for i in titles)


def test_plot_quality_examples_empty_input(tmp_path):
    store = _fake_store(tmp_path, n=1)
    plot_quality_examples(store, _fake_report(n=0))  # must not raise


########################################################################
# Video quality report plots
########################################################################


def _fake_video_quality_report(n_frames=20, fps=10.0, stride=2, n_pairs=10):
    """
    Report shaped like qa.compute_video_quality output; pair 3 failed to match.
    """
    rng = np.random.default_rng(0)
    frames = {
        "frame_idx": list(range(n_frames)),
        "blur": rng.uniform(0.1, 0.9, n_frames).tolist(),
        "laplacian": rng.uniform(50, 500, n_frames).tolist(),
        "exposure_mean": rng.uniform(80, 160, n_frames).tolist(),
        "exposure_median": rng.uniform(80, 160, n_frames).tolist(),
        "exposure_std": rng.uniform(20, 60, n_frames).tolist(),
        "clipped_low_frac": rng.uniform(0, 0.05, n_frames).tolist(),
        "clipped_high_frac": rng.uniform(0, 0.05, n_frames).tolist(),
    }
    a = list(range(0, n_pairs * stride, stride))
    translation = rng.uniform(0, 30, n_pairs).tolist()
    parallax = rng.uniform(0, 1, n_pairs).tolist()
    # A failed pair is None in the dict (nan -> null in the JSON)
    if n_pairs > 3:
        translation[3] = None
        parallax[3] = None
    pairs = {
        "frame_idx_a": a,
        "frame_idx_b": [i + stride for i in a],
        "n_matches": rng.integers(0, 500, n_pairs).tolist(),
        "translation_px": translation,
        "parallax": parallax,
    }
    return {
        "available": True,
        "video": {"path": "/data/clip.mp4", "fps": fps, "total_frames": n_frames, "width": 64, "height": 48},
        "params": {"motion_stride": stride},
        "frames": frames,
        "pairs": pairs,
    }


def _assert_png(path, expected_name):
    assert path.name == expected_name
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_photometric_writes_png(tmp_path):
    _assert_png(plot_photometric(_fake_video_quality_report(), tmp_path), "photometric.png")
    # Overlay path, into a directory that does not exist yet
    _assert_png(plot_photometric(_fake_video_quality_report(), tmp_path / "new", selected=[3, 7]), "photometric.png")


def test_plot_motion_writes_png(tmp_path):
    _assert_png(plot_motion(_fake_video_quality_report(), tmp_path), "motion.png")
    _assert_png(plot_motion(_fake_video_quality_report(), tmp_path / "new", selected=[3, 7]), "motion.png")


def test_plot_motion_skips_empty_pairs(tmp_path):
    """
    A video shorter than the stride has zero pairs; no plot is written.
    """
    assert plot_motion(_fake_video_quality_report(n_pairs=0), tmp_path / "none") is None
    assert not (tmp_path / "none").exists()


def test_plot_correlation_writes_png(tmp_path):
    report = _fake_video_quality_report()
    # Per-frame vs per-pair: blur read at each pair's first frame
    _assert_png(
        plot_correlation(report, "translation_px", "blur", tmp_path / "new"), "correlation-translation_px-blur.png"
    )
    # Both per-pair, both per-frame
    _assert_png(plot_correlation(report, "n_matches", "parallax", tmp_path), "correlation-n_matches-parallax.png")
    _assert_png(plot_correlation(report, "blur", "exposure_mean", tmp_path), "correlation-blur-exposure_mean.png")
    # No usable pairs: nothing written
    assert plot_correlation(_fake_video_quality_report(n_pairs=0), "translation_px", "blur", tmp_path / "none") is None
    assert not (tmp_path / "none").exists()


def test_plot_frame_extremes_writes_png(tmp_path, monkeypatch):
    # Decode stubbed: the plot only needs an (H, W, 3) uint8 per requested frame
    monkeypatch.setattr(viz_module, "get_video_info", lambda path: {"fps": 10.0})
    seen = []
    monkeypatch.setattr(
        viz_module, "extract_frame", lambda path, i, info=None: seen.append(i) or np.zeros((8, 6, 3), np.uint8)
    )
    report = _fake_video_quality_report()
    _assert_png(plot_frame_extremes(report, "fake.mp4", tmp_path, column="blur", n=3), "extremes-blur.png")
    blur = np.asarray(report["frames"]["blur"])
    assert set(seen) == set(np.argsort(blur)[-3:]) | set(np.argsort(blur)[:3])
    _assert_png(plot_frame_extremes(report, "fake.mp4", tmp_path, column="exposure_mean"), "extremes-exposure_mean.png")
