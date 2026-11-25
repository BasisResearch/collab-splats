"""
Matplotlib plots for the reconstruction quality report.

- saved beside the report, never shown; report-only, no thresholds
- imported by the Reconstructor report stage, like preproc.viz for the video report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_photometric_ncc(
    pairs: dict | None, path: str | Path, *, title: str = "", dpi: int = 120
) -> Path | None:
    """
    Cross-view NCC by frame gap: spread per gap, and each gap along the sequence.

    - left: median and p10 NCC against the index gap; a steep fall means poses drift apart
    - right: NCC per source frame, one line per gap; a dip at every gap marks a bad segment
    - p10 rather than the minimum: one occluded pair would set the minimum

    Args:
        pairs: the report's photometric_pairs table, {idx1, idx2, photometric_ncc, n_pixels}.
        path: PNG path to write; its directory is created if absent.
        title: figure suptitle.
        dpi: PNG resolution.

    Returns:
        Path to the written PNG; None when there are no pairs to plot.
    """
    if not pairs or not pairs["photometric_ncc"]:
        return None

    # Pair columns as arrays; the gap is the index difference
    idx1 = np.asarray(pairs["idx1"])
    gap = np.asarray(pairs["idx2"]) - idx1
    ncc = np.asarray(pairs["photometric_ncc"], dtype=np.float64)
    gaps = np.unique(gap)

    fig, (ax_gap, ax_seq) = plt.subplots(
        1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2.5]}
    )

    # Median and p10 per gap
    median = [np.median(ncc[gap == g]) for g in gaps]
    p10 = [np.percentile(ncc[gap == g], 10) for g in gaps]
    ax_gap.plot(gaps, median, marker="o", label="median")
    ax_gap.plot(gaps, p10, marker="o", linestyle="--", label="p10")
    ax_gap.set_xscale("log")
    ax_gap.set_xticks(gaps, [str(g) for g in gaps])
    ax_gap.set_xlabel("frame gap (j - i)")
    ax_gap.set_ylabel("photometric NCC")
    ax_gap.grid(alpha=0.3)
    ax_gap.legend()

    # One line per gap along the sequence, colored from near to far
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(gaps)))

    for g, color in zip(gaps, colors):
        keep = gap == g
        order = np.argsort(idx1[keep])
        ax_seq.plot(
            idx1[keep][order],
            ncc[keep][order],
            color=color,
            linewidth=0.8,
            label=f"gap {g}",
        )

    ax_seq.set_xlabel("source frame (array index)")
    ax_seq.set_ylabel("photometric NCC")
    ax_seq.grid(alpha=0.3)
    ax_seq.legend(ncol=len(gaps), fontsize=8, loc="lower left")

    # Title, write and close
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
