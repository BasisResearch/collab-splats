"""
Visualization helpers for localization results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from collab_splats.localization.localizer import LocalizationResult

logger = logging.getLogger(__name__)


########################################################################
# Adapters
########################################################################


def correspondences_for_ref(
    loc: LocalizationResult,
    ref_idx: int,
    ref_image_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Query px, ref px and inlier mask of the correspondences against one reference frame.

    - ref_px lives in the pixel space the localizer indexed (loc.ref_hw)

    Args:
        loc: localization result holding the correspondence arrays.
        ref_idx: reference frame index to slice out.
        ref_image_hw: (H, W) of the image to draw on; ref_px is rescaled into it when it differs from loc.ref_hw.

    Returns:
        (query_px, ref_px, inlier_mask) for that frame; the mask is None when loc has none.
    """
    assert loc.pts2d is not None and loc.pts2d_ref is not None
    sel = loc.ref_frame_indices == ref_idx
    mask = loc.inlier_mask[sel] if loc.inlier_mask is not None else None
    ref_px = loc.pts2d_ref[sel]
    ref_hw = loc.ref_hw

    # Rescale ref pixels from the result's native space to the display image's space
    if (
        ref_image_hw is not None
        and ref_hw is not None
        and tuple(ref_image_hw) != tuple(ref_hw)
    ):
        scale = np.array(
            [ref_image_hw[1] / ref_hw[1], ref_image_hw[0] / ref_hw[0]], dtype=np.float32
        )
        ref_px = ref_px * scale

    return loc.pts2d[sel], ref_px, mask


########################################################################
# Plots
########################################################################


def _fit_similarity(ref_px: np.ndarray, query_px: np.ndarray) -> np.ndarray | None:
    """
    Scale, rotation and shift taking reference px onto query px; None when the fit fails.

    - similarity, not homography: two cameras in a 3D scene share no plane-to-plane map
    - least-median fit: robust without a pixel threshold to tune
    - (2, 3) matrix, as cv2.warpAffine takes it
    """
    if len(ref_px) < 2:
        return None

    S, _ = cv2.estimateAffinePartial2D(ref_px, query_px, method=cv2.LMEDS)
    return S


def plot_correspondences(
    query_image: np.ndarray,
    ref_image: np.ndarray,
    query_px: np.ndarray,
    ref_px: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_pairs: int = 200,
    align: bool = False,
    show: bool = True,
) -> plt.Figure | None:
    """
    Side-by-side query and reference image with inlier/outlier connecting lines.

    - green lines for inliers, red for outliers
    - align: reference scaled, rotated and shifted into the query's frame, so a match sits at
      the same spot in both panels; a yellow box outlines the reference on both
    - align falls back to the plain layout when the inliers fix no similarity

    Args:
        query_image: HxWx3 uint8 RGB query image.
        ref_image: HxWx3 uint8 RGB reference image (caller-resolved).
        query_px: (K, 2) pixel coordinates in the query image.
        ref_px: (K, 2) pixel coordinates in the reference image; correspondences_for_ref slices them per frame.
        inlier_mask: (K,) bool inlier flags; None draws every pair as an inlier.
        max_pairs: cap on lines drawn; a random subsample is drawn beyond it.
        align: draw the reference in the query's frame via a similarity fit to the inliers.
        show: call plt.show() (notebook behavior); the dashboard passes False.

    Returns:
        The matplotlib Figure, or None when there is nothing to plot.
    """
    kpts0 = np.asarray(query_px, dtype=np.float32)
    kpts1 = np.asarray(ref_px, dtype=np.float32)

    # Nothing to draw without correspondences
    if len(kpts0) == 0:
        logger.warning("plot_correspondences: no correspondences to plot")
        return None

    inliers = (
        np.ones(len(kpts0), dtype=bool)
        if inlier_mask is None
        else np.asarray(inlier_mask, dtype=bool)
    )
    h, w = query_image.shape[:2]
    S = None

    # Similarity from all inliers before subsampling; a subsample degrades the fit
    if align:
        S = _fit_similarity(kpts1[inliers], kpts0[inliers])

        if S is None:
            logger.debug("plot_correspondences: no similarity fit — plain layout")

    # Random subsample beyond max_pairs
    if len(kpts0) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(kpts0), max_pairs, replace=False)
        kpts0, kpts1, inliers = kpts0[idx], kpts1[idx], inliers[idx]

    # Aligned: the reference resampled into the query's frame, grey where it has no pixels
    ref_shown = ref_image
    ref_pts = kpts1

    if S is not None:
        ref_shown = cv2.warpAffine(ref_image, S, (w, h), borderValue=(128, 128, 128))
        ref_pts = kpts1 @ S[:, :2].T + S[:, 2]

    # Reference panel beside the query at its height
    ref_scale = h / ref_shown.shape[0]
    ref_w = ref_shown.shape[1] * ref_scale
    ref_x0 = 1.04 * w
    ref_pts = ref_pts * ref_scale + [ref_x0, 0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(query_image, extent=(0, w, h, 0))
    ax.imshow(ref_shown, extent=(ref_x0, ref_x0 + ref_w, h, 0))

    # Reference outline on both panels, each clipped to its own panel
    if S is not None:
        rh, rw = ref_image.shape[:2]
        corners = np.array(
            [[0, 0], [rw, 0], [rw, rh], [0, rh], [0, 0]], dtype=np.float32
        )
        outline = corners @ S[:, :2].T + S[:, 2]

        for panel_x in (0.0, ref_x0):
            (line,) = ax.plot(
                outline[:, 0] + panel_x, outline[:, 1], color="yellow", linewidth=2
            )
            line.set_clip_path(Rectangle((panel_x, 0), w, h, transform=ax.transData))

    # Lines per pair, then keypoint dots on both sides
    for (x0, y0), (x1, y1), ok in zip(kpts0, ref_pts, inliers):
        color = "lime" if ok else "red"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.8, alpha=0.6)

    ax.scatter(kpts0[:, 0], kpts0[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.scatter(ref_pts[:, 0], ref_pts[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.set_xlim(0, ref_x0 + ref_w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"query ↔ reference — {inliers.sum()}/{len(inliers)} inliers shown")
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_inlier_distribution(
    ref_frame_indices: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    n_frames: int | None = None,
) -> plt.Figure | None:
    """
    Per-reference-image inlier bars with total-correspondence markers.

    - bars are colored viridis by frame index (frame order == time), matching the 3D camera view
    - each bar gets a black tick at that image's total correspondence count
    - uniform totals collapse the ticks to one dashed horizontal line

    Args:
        ref_frame_indices: (M,) reference frame index per correspondence.
        inlier_mask: (M,) bool inlier flags per correspondence.
        n_frames: total reference frames, so bars include zero-match frames; defaults to max(ref_frame_indices) + 1.

    Returns:
        The matplotlib Figure, or None when there is nothing to plot.
    """
    if ref_frame_indices is None or inlier_mask is None:
        logger.warning("plot_inlier_distribution: no correspondence data to plot")
        return None

    # Grow n to cover every referenced frame; bincount never truncates to a short n_frames
    idx = np.asarray(ref_frame_indices).astype(np.intp)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)
    n_used = int(idx.max()) + 1 if len(idx) else 0
    n = max(int(n_frames) if n_frames is not None else 0, n_used)
    totals = np.bincount(idx, minlength=n)
    inliers = np.bincount(idx[inlier_mask], minlength=n)

    # Viridis by frame index — matches the time coloring of the 3D camera plot
    cmap = plt.get_cmap("viridis")
    stops = np.linspace(0, 1, max(n, 2))
    colors = cmap(stops)[:n]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    x = np.arange(n)
    ax.bar(x, inliers, color=colors)

    # Totals: single dashed line when uniform, per-bar ticks otherwise
    nonzero = totals[totals > 0]

    if len(nonzero) and (nonzero == nonzero[0]).all():
        ax.axhline(int(nonzero[0]), linestyle="--", color="0.4", linewidth=1)
    else:
        for xi, t in zip(x, totals):
            if t > 0:
                ax.plot([xi - 0.4, xi + 0.4], [t, t], color="0.2", linewidth=1)

    # Counts derived from the arrays — no result object needed for the summary
    n_inliers = int(inlier_mask.sum())
    n_correspondences = len(idx)
    ax.set_xlabel("reference image (time →)")
    ax.set_ylabel("inliers")
    ax.set_title(
        f"{n_inliers}/{n_correspondences} inliers ({100 * n_inliers / max(n_correspondences, 1):.0f}%)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
