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


def _image_corners(image: np.ndarray) -> np.ndarray:
    """
    The four pixel-center corners of an image, as cv2.perspectiveTransform input (4, 1, 2).
    """
    h, w = image.shape[:2]
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    return corners.reshape(-1, 1, 2)


def plot_correspondences(
    query_image: np.ndarray,
    ref_image: np.ndarray,
    query_px: np.ndarray,
    ref_px: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_pairs: int = 200,
    warp_corners: bool = False,
    show: bool = True,
) -> plt.Figure | None:
    """
    Side-by-side query and reference image with inlier/outlier connecting lines.

    - green lines for inliers, red for outliers
    - warp_corners: cyan quad = query corners on the ref, yellow quad = ref corners on the query
    - each panel keeps a 25% margin; a quad is clipped to its own panel and margin

    Args:
        query_image: HxWx3 uint8 RGB query image.
        ref_image: HxWx3 uint8 RGB reference image (caller-resolved).
        query_px: (K, 2) pixel coordinates in the query image.
        ref_px: (K, 2) pixel coordinates in the reference image; correspondences_for_ref slices them per frame.
        inlier_mask: (K,) bool inlier flags; None draws every pair as an inlier.
        max_pairs: cap on lines drawn; a random subsample is drawn beyond it.
        warp_corners: draw homography-warped boundaries on both sides.
        show: call plt.show() (notebook behavior); the dashboard passes False.

    Returns:
        The matplotlib Figure, or None when there is nothing to plot.
    """
    kpts0 = np.asarray(query_px)
    kpts1 = np.asarray(ref_px)

    # Nothing to draw without correspondences
    if len(kpts0) == 0:
        logger.warning("plot_correspondences: no correspondences to plot")
        return None

    inliers = (
        np.ones(len(kpts0), dtype=bool)
        if inlier_mask is None
        else np.asarray(inlier_mask, dtype=bool)
    )
    H = None

    # Homography from all inliers before subsampling; a subsampled set degrades H
    if warp_corners:
        inlier_kpts0 = kpts0[inliers]
        inlier_kpts1 = kpts1[inliers]

        if len(inlier_kpts0) >= 4:
            H, _ = cv2.findHomography(
                inlier_kpts0,
                inlier_kpts1,
                cv2.USAC_MAGSAC,
                3.5,
                maxIters=1_000,
                confidence=0.999,
            )

            if H is None:
                logger.debug(
                    "plot_correspondences: homography degenerate — skipping corner warp"
                )
        else:
            logger.debug(
                "plot_correspondences: only %d inliers — need ≥4 for corner warp",
                len(inlier_kpts0),
            )

    # Random subsample beyond max_pairs
    if len(kpts0) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(kpts0), max_pairs, replace=False)
        kpts0, kpts1, inliers = kpts0[idx], kpts1[idx], inliers[idx]

    # Reference beside the query at its height; a margin around each holds boxes past the image
    h, w = query_image.shape[:2]
    ref_scale = h / ref_image.shape[0]
    ref_w = ref_image.shape[1] * ref_scale
    margin_q = 0.25 * w
    margin_r = 0.25 * ref_w
    margin_y = 0.25 * h
    ref_x0 = w + margin_q + margin_r

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(query_image, extent=(0, w, h, 0))
    ax.imshow(ref_image, extent=(ref_x0, ref_x0 + ref_w, h, 0))

    # Query corners onto the reference via H, reference corners onto the query via H⁻¹
    if warp_corners and H is not None:
        warped_q = cv2.perspectiveTransform(_image_corners(query_image), H)[:, 0]
        warped_r = cv2.perspectiveTransform(
            _image_corners(ref_image), np.linalg.inv(H)
        )[:, 0]
        boxes = [
            (
                warped_q * ref_scale + [ref_x0, 0],
                "cyan",
                ref_x0 - margin_r,
                ref_w + 2 * margin_r,
            ),
            (warped_r, "yellow", -margin_q, w + 2 * margin_q),
        ]

        # Each box clipped to its own panel plus margin, so it never spills into the other
        for quad, color, x_min, width in boxes:
            closed = np.vstack([quad, quad[:1]])
            (line,) = ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
            clip = Rectangle(
                (x_min, -margin_y), width, h + 2 * margin_y, transform=ax.transData
            )
            line.set_clip_path(clip)

    # Lines per pair, then keypoint dots on both sides
    for (x0, y0), (x1, y1), ok in zip(kpts0, kpts1, inliers):
        color = "lime" if ok else "red"
        ax.plot(
            [x0, x1 * ref_scale + ref_x0],
            [y0, y1 * ref_scale],
            color=color,
            linewidth=0.8,
            alpha=0.6,
        )

    ax.scatter(kpts0[:, 0], kpts0[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.scatter(
        kpts1[:, 0] * ref_scale + ref_x0,
        kpts1[:, 1] * ref_scale,
        s=8,
        c="white",
        zorder=3,
        linewidths=0,
    )
    ax.set_xlim(-margin_q, ref_x0 + ref_w + margin_r)
    ax.set_ylim(h + margin_y, -margin_y)
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
