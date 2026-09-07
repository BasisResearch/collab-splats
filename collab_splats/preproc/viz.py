"""
Matplotlib plots for frame sampling analysis — notebook use only.

- deliberately NOT re-exported from collab_splats.preproc.__init__, so pipeline consumers
  never import matplotlib
- import explicitly: `from collab_splats.preproc.viz import plot_frame_scores`
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from collab_splats.preproc.video import extract_frame, get_video_info


def plot_frame_grid(frames: list, title: str, n_cols: int = 6) -> None:
    """
    Display a grid of RGB frames.

    Args:
        frames: (H, W, 3) uint8 RGB frames.
        title: figure suptitle.
        n_cols: columns in the grid; rows follow from the frame count.
    """
    n = len(frames)
    n_cols = min(n_cols, n)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False)
    axes = np.array(axes).flatten()
    # Fill the grid; blank any unused trailing cells
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(frames[i])
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_selection(
    total_frames: int,
    fps_indices: list | None = None,
    of_indices: list | None = None,
) -> None:
    """
    Vertical-line timeline of selected frame indices; two panels when both sets given.

    Args:
        total_frames: source frame count, setting the x extent.
        fps_indices: indices chosen by fixed-rate sampling; None omits the panel.
        of_indices: indices chosen by optical-flow sampling; None omits the panel.
    """
    sets = [
        (fps_indices, "Uniform", "steelblue"),
        (of_indices, "Optical Flow", "darkorange"),
    ]
    active = [(idx, label, color) for idx, label, color in sets if idx is not None]
    fig, axes = plt.subplots(len(active), 1, figsize=(12, 2 * len(active)), squeeze=False)
    for ax, (indices, label, color) in zip(axes[:, 0], active):
        if indices:
            ax.vlines(indices, 0, 1, colors=color, linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xlabel("Frame index")
        ax.set_title(f"{label}  (n={len(indices) if indices else 0})", fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_frame_scores(frame_scores: list) -> None:
    """
    3-panel timeseries of per-frame signals: disparity / rotation / histogram similarity.

    - selected frames are marked with vertical grey lines

    Args:
        frame_scores: per-frame score records from sample_optical_flow.
    """
    if not frame_scores:
        plt.subplots(3, 1, figsize=(12, 6))
        plt.show()
        return
    idxs = [d["frame_idx"] for d in frame_scores]
    selected_idxs = [d["frame_idx"] for d in frame_scores if d["selected"]]
    panels = [
        ([d["disparity"] for d in frame_scores], "Disparity (px)", "steelblue"),
        ([d["rotation"] for d in frame_scores], "Rotation (deg)", "seagreen"),
        ([d["histogram_similarity"] for d in frame_scores], "Histogram similarity", "tomato"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.plot(idxs, values, color=color, linewidth=0.8)
        # Faint vertical line at each selected frame
        for x in selected_idxs:
            ax.axvline(x, color="gray", alpha=0.25, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Per-frame optical flow scores  (grey lines = selected frames)", fontsize=11)
    fig.tight_layout()
    plt.show()


########################################################################
# Video quality report plots
#
# - one PNG per measurement family of qa.compute_video_quality's report
# - every plotter draws EVERY frame / pair as shipped: raw columns, no
#   thresholds, no verdicts
# - `selected` only marks which frames made it into images/
# - headless: save and close, never plt.show()
# - title, overlay and save are inlined in each plotter on purpose — five
#   short duplicates beat a helper layer
########################################################################

_FIG_WIDTH_IN = 12
_PANEL_HEIGHT_IN = 2.8
_PNG_DPI = 90  # matches collab-data/track_reprojection/report.py
_THUMB_DPI = 150  # frame montages: blur has to be visible in the thumbnails
_HIST_BINS = 40


def _marginal_hist(ax_hist: Axes, values: np.ndarray) -> None:
    """
    Marginal distribution flush against a timeseries panel (shares its y axis), JointGrid style.

    Args:
        ax_hist: the marginal axis to draw into.
        values: the panel's y values; non-finite entries are dropped.
    """
    finite = values[np.isfinite(values)]
    sns.histplot(y=finite, bins=_HIST_BINS, ax=ax_hist, color="gray", edgecolor=None, alpha=0.6)
    ax_hist.set_xlim(left=0)
    ax_hist.tick_params(axis="y", labelleft=False, left=False)
    ax_hist.set_xlabel("")
    ax_hist.set_xticks([])
    sns.despine(ax=ax_hist, left=True, bottom=True)


def _timeseries_grid(n_panels: int) -> tuple[Figure, list[Axes], list[Axes]]:
    """
    n_panels rows of [timeseries | marginal histogram]; x shared down the column, y across each row.

    Args:
        n_panels: number of timeseries rows.

    Returns:
        (figure, timeseries axes, marginal axes) — one axis per panel in each array.
    """
    fig, grid = plt.subplots(
        n_panels,
        2,
        figsize=(_FIG_WIDTH_IN + 1.5, _PANEL_HEIGHT_IN * n_panels),
        sharex="col",
        sharey="row",
        width_ratios=[8, 1],
        squeeze=False,
    )
    axes, hists = list(grid[:, 0]), list(grid[:, 1])
    for ax in axes:
        ax.margins(x=0)
    return fig, axes, hists


def plot_photometric(report: dict, out_dir: str | Path, *, selected: Sequence[int] | None = None) -> Path:
    """
    Per-frame photometry vs seconds: blur, laplacian variance, exposure, clipped fractions.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        out_dir: directory the PNG is written into; created if absent.
        selected: source frame indices to mark; None marks none. Report-only — this marks
            which frames were kept, it does not judge them.

    Returns:
        Path to the written PNG.
    """
    # Unpack columns
    frames, video = report["frames"], report["video"]
    fps = video["fps"]
    blur = np.asarray(frames["blur"], dtype=float)
    laplacian = np.asarray(frames["laplacian"], dtype=float)
    mean = np.asarray(frames["exposure_mean"], dtype=float)
    median = np.asarray(frames["exposure_median"], dtype=float)
    std = np.asarray(frames["exposure_std"], dtype=float)
    clip_lo = np.asarray(frames["clipped_low_frac"], dtype=float)
    clip_hi = np.asarray(frames["clipped_high_frac"], dtype=float)

    # Time axis: wall-clock seconds
    t = np.asarray(frames["frame_idx"], dtype=float) / fps

    # Panels 1-2: opposite directions on purpose — a saturated blur reads against laplacian
    fig, axes, hists = _timeseries_grid(4)
    axes[0].plot(t, blur, linewidth=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("blur (↑ blurrier)")
    axes[1].plot(t, laplacian, linewidth=0.8)
    axes[1].set_ylabel("laplacian var (↑ sharper)")

    # Panels 3-4: brightness on the 8-bit scale, then the destroyed-pixel fractions
    axes[2].fill_between(t, mean - std, mean + std, alpha=0.2, label="mean ± std")
    axes[2].plot(t, mean, linewidth=0.8, label="mean")
    axes[2].plot(t, median, linewidth=0.8, linestyle="--", label="median")
    axes[2].set_ylim(0, 255)
    axes[2].set_ylabel("exposure (gray level)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].stackplot(t, clip_lo, clip_hi, labels=["clipped low", "clipped high"], alpha=0.7)
    axes[3].set_ylabel("clipped pixel fraction")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=8)

    # Marginal distributions: how each value is skewed over the whole video
    _marginal_hist(hists[0], blur)
    _marginal_hist(hists[1], laplacian)
    _marginal_hist(hists[2], mean)
    _marginal_hist(hists[3], clip_lo + clip_hi)

    # Selected-frame overlay: 1 px lines, never axvspan — a one-frame span is
    # under a pixel on a long video and vanishes
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        for ax in axes:
            ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    fig.suptitle(
        f"{Path(video['path']).name} — {len(report['frames']['frame_idx'])} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "photometric.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path


def plot_motion(
    report: dict,
    out_dir: str | Path,
    *,
    selected: Sequence[int] | None = None,
) -> Path | None:
    """
    Per-pair motion vs seconds: translation, parallax (failed pairs as red ticks at 0), match count.

    Args:
        report: a quality report carrying a "pairs" block.
        out_dir: directory the PNG is written into; created if absent.
        selected: source frame indices to mark; None marks none.

    Returns:
        Path to the written PNG, or None when the report has no pairs.
    """
    # Unpack columns: None (failed pair) -> nan in one place
    pairs, video = report["pairs"], report["video"]
    fps = video["fps"]
    translation = np.asarray(pairs["translation_px"], dtype=float)
    parallax = np.asarray(pairs["parallax"], dtype=float)
    n_matches = np.asarray(pairs["n_matches"], dtype=float)
    if translation.size == 0:
        return None

    # Time axis: a pair sits at its first member
    t = np.asarray(pairs["frame_idx_a"], dtype=float) / fps

    # Panels: raw per-pair values, nan (failed pair) leaves a gap; failed pairs are
    # the worst pairs — marked on the parallax panel, never dropped
    fig, axes, hists = _timeseries_grid(3)
    axes[0].plot(t, translation, linewidth=0.8)
    axes[0].set_ylabel("translation (px / pair)")
    axes[1].plot(t, parallax, linewidth=0.8)
    failed = np.isnan(parallax)
    if failed.any():
        axes[1].plot(t[failed], np.zeros(int(failed.sum())), "|", color="red", markersize=10, label="failed to match")
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("parallax")
    axes[2].plot(t, n_matches, linewidth=0.8)
    axes[2].set_ylabel("matches / pair")
    axes[2].set_xlabel("time (s)")

    # Marginal distributions
    _marginal_hist(hists[0], translation)
    _marginal_hist(hists[1], parallax)
    _marginal_hist(hists[2], n_matches)

    # Selected-frame overlay: 1 px lines, never axvspan — a one-frame span is
    # under a pixel on a long video and vanishes
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        for ax in axes:
            ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    fig.suptitle(
        f"{Path(video['path']).name} — {len(report['frames']['frame_idx'])} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "motion.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path


def plot_frame_extremes(
    report: dict, video_path: str | Path, out_dir: str | Path, *, column: str = "blur", n: int = 6
) -> Path:
    """
    The n highest and n lowest frames of one per-frame report column, decoded from the video.

    - top row = highest values, bottom row = lowest
    - each thumbnail is captioned with frame index, wall-clock time and the value, so a reader
      can judge what the number means on this footage
    - one ffmpeg seek per thumbnail (2n decodes) — for notebooks and spot checks

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        video_path: the source video the report was measured on.
        out_dir: directory the PNG is written into; created if absent.
        column: any key of report["frames"] except frame_idx — blur, laplacian, exposure_mean, ...
        n: thumbnails per row.

    Returns:
        Path to the written PNG.
    """
    # Unpack columns and rank
    frames, video = report["frames"], report["video"]
    fps = video["fps"]
    idx = np.asarray(frames["frame_idx"])
    values = np.asarray(frames[column], dtype=float)
    order = np.argsort(values)
    rows = [(f"highest {column}", order[::-1][:n]), (f"lowest {column}", order[:n])]

    # Panels: one thumbnail per extreme frame, decoded at source resolution
    info = get_video_info(video_path)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 14), squeeze=False)
    for (label, sel), axrow in zip(rows, axes):
        for ax, i in zip(axrow, sel):
            ax.imshow(extract_frame(video_path, int(idx[i]), info=info))
            ax.set_title(f"#{idx[i]}  t={idx[i] / fps:.1f}s\n{column}={values[i]:.3g}", fontsize=10)
            ax.axis("off")
        axrow[0].text(
            -0.04, 0.5, label, transform=axrow[0].transAxes, rotation=90, va="center", ha="right", fontsize=12
        )

    # Title, save and close
    fig.suptitle(f"{Path(video['path']).name} — {column} extremes ({len(idx)} frames @ {fps:.2f} fps)")
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"extremes-{column}.png"
    fig.savefig(path, dpi=_THUMB_DPI)
    plt.close(fig)
    return path


def plot_correlation(report: dict, x: str, y: str, out_dir: str | Path) -> Path | None:
    """
    seaborn jointplot of two report columns, with r and rho in the corner.

    - regression line plus marginal histograms
    - a per-frame column compared with a per-pair column is read at the pair's first frame
      (frame_idx_a), so blur vs translation_px compares sample for sample
    - null entries (failed pairs) are dropped from the scatter, the fit and the statistics

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        x: any column of report["frames"] or report["pairs"] — blur, translation_px, ...
        y: the column plotted against x.
        out_dir: directory the PNG is written into; created if absent.

    Returns:
        Path to the written PNG, or None when the two columns share no usable samples.
    """
    # Resolve each name to one sample per pair (or per frame when both are per-frame)
    frames, pairs, video = report["frames"], report["pairs"], report["video"]
    per_pair = x in pairs or y in pairs

    def column(name):
        if name in pairs:
            return np.asarray(pairs[name], dtype=float)
        values = np.asarray(frames[name], dtype=float)
        if not per_pair:
            return values
        lookup = dict(zip(frames["frame_idx"], values))
        return np.asarray([lookup.get(i, np.nan) for i in pairs["frame_idx_a"]], dtype=float)

    xs, ys = column(x), column(y)
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]

    if len(xs) < 2:
        return None

    # Statistics
    pearson = np.corrcoef(xs, ys)[0, 1]
    rho = spearmanr(xs, ys).statistic

    # Panel: seaborn JointGrid, statistics bottom right
    # - square regression joint with filled KDE marginals
    # - full seaborn theme (white, deep palette, notebook) scoped to this figure only,
    #   so the timeseries plotters keep matplotlib's look
    with sns.axes_style("white"), sns.plotting_context("notebook"), sns.color_palette("deep"):
        grid = sns.JointGrid(x=xs, y=ys, height=6.5)
        grid.plot_joint(
            sns.regplot,
            scatter_kws={"s": 12, "alpha": 0.5, "color": "lightblue", "edgecolor": "darkblue", "linewidths": 0.4},
            line_kws={"color": sns.color_palette("deep")[0], "linewidth": 1.5},
        )
        grid.plot_marginals(sns.kdeplot, fill=True, color=sns.color_palette("deep")[0], alpha=0.5, linewidth=1)
    fig, ax = grid.figure, grid.ax_joint

    # Axes clamped to the data range: the fit's confidence band must not extend past
    # the values (e.g. parallax below 0) and the marginals share these limits
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.margins(0.02)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.text(
        0.98,
        0.02,
        f"Pearson r = {pearson:.3f}\nSpearman ρ = {rho:.3f}\nn = {len(xs)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    # Title, save and close
    fig.suptitle(f"{Path(video['path']).name} — {y} vs {x}, stride {report['params']['motion_stride']}")
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"correlation-{x}-{y}.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path
