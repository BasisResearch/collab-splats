"""Matplotlib plots for frame sampling analysis — notebook use only.

Deliberately NOT re-exported from collab_splats.preproc.__init__ so pipeline
consumers never import matplotlib. Import explicitly:
`from collab_splats.preproc.viz import plot_frame_scores`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.sampling import _combine_scores


def plot_frame_grid(frames: list, title: str, n_cols: int = 6) -> None:
    """Display a grid of RGB frames."""
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
    """Vertical-line timeline of selected frame indices; two panels when both sets given."""
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
    """3-panel timeseries of per-frame signals: disparity / rotation / histogram similarity.

    Takes score_frames() records; selected frames marked with vertical grey lines.
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


def plot_disparity_sensitivity(frame_scores: list, disparity_values: list) -> None:
    """Approximate selected-frame count vs min_disparity threshold.

    Re-thresholds precomputed score_frames() records via the real scoring
    formula (_combine_scores) — no video re-decode, no formula drift.
    Approximate: ignores the stateful keyframe updates of a true re-run.
    """
    # Re-score each threshold from the recorded raw signals
    counts = []
    for threshold in disparity_values:
        n = sum(1 for d in frame_scores if _combine_scores(d["disparity"], d["histogram_similarity"], threshold) >= 0.5)
        counts.append(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(disparity_values, counts, marker="o", color="steelblue", linewidth=1.5)
    ax.set_xlabel("min_disparity threshold (px)")
    ax.set_ylabel("Frames selected (approx.)")
    ax.set_title("Frame count vs disparity threshold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_quality_examples(store: FrameStore, frame_scores: list, n_examples: int = 4) -> None:
    """Example frames per quality-gate outcome: accepted / blur- / exposure-rejected.

    Takes score_frames() records; reads only the displayed frames from the canonical
    frames.zarr store (no re-decode). Empty categories are dropped from the grid
    (counts still shown in the title).
    """
    categories = [
        ("Accepted", [d for d in frame_scores if d.get("reject_reason") is None]),
        ("Rejected: blur", [d for d in frame_scores if d.get("reject_reason") == "blur"]),
        ("Rejected: exposure", [d for d in frame_scores if d.get("reject_reason") == "exposure"]),
    ]
    counts = " · ".join(f"{key}: {len(recs)}" for key, (_, recs) in zip(("accepted", "blur", "exposure"), categories))
    # Spread picks evenly across each non-empty category rather than taking the first n
    rows = []
    for label, recs in categories:
        if recs:
            picks = [recs[i] for i in np.linspace(0, len(recs) - 1, min(n_examples, len(recs))).astype(int)]
            rows.append((label, picks))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No records ({counts})", ha="center", va="center")
        plt.show()
        return
    # Gather every picked frame_idx across all rows; read each once from the store
    all_idxs = sorted({d["frame_idx"] for _, recs in rows for d in recs})
    frame_by_idx = {i: store.image_by_frame_idx(i) for i in all_idxs}
    fig, axes = plt.subplots(len(rows), n_examples, figsize=(n_examples * 2.6, len(rows) * 2.4), squeeze=False)
    for row_axes, (label, recs) in zip(axes, rows):
        for ax, d in zip(row_axes, recs):
            ax.imshow(frame_by_idx[d["frame_idx"]])
            ax.set_title(f"#{d['frame_idx']}  blur {d['blur_score']:.0f} · mean {d['exposure_mean']:.0f}", fontsize=8)
        for ax in row_axes:
            ax.axis("off")
        # Row label survives axis("off") because it's a plain text artist
        row_axes[0].text(
            -0.06, 0.5, label, transform=row_axes[0].transAxes, rotation=90, va="center", ha="center", fontsize=10
        )
    fig.suptitle(f"Quality gate examples  ({counts})", fontsize=11)
    fig.tight_layout()
    plt.show()
