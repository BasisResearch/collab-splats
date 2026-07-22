"""Visualization helpers for localization results."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


########################################################################
# Adapters
########################################################################


def correspondences_for_ref(loc, ref_idx: int, ref_image_hw: "tuple[int, int] | None" = None):
    """(query_px, ref_px, inlier_mask) for one reference frame, from any object
    exposing pts2d / pts2d_ref / ref_frame_indices / inlier_mask arrays.

    ref_px lives in the pixel space the localizer indexed (loc.ref_hw). Pass
    ref_image_hw=(H, W) of the image you will draw on to rescale ref_px into its
    space when the display resolution differs.
    """
    sel = loc.ref_frame_indices == ref_idx
    mask = loc.inlier_mask[sel] if loc.inlier_mask is not None else None
    ref_px = loc.pts2d_ref[sel]
    # Rescale ref pixels from the result's native space to the display image's space
    ref_hw = getattr(loc, "ref_hw", None)
    if ref_image_hw is not None and ref_hw is not None and tuple(ref_image_hw) != tuple(ref_hw):
        scale = np.array([ref_image_hw[1] / ref_hw[1], ref_image_hw[0] / ref_hw[0]], dtype=np.float32)
        ref_px = ref_px * scale
    return loc.pts2d[sel], ref_px, mask


########################################################################
# Plots
########################################################################


def plot_correspondences(
    query_image: np.ndarray,
    ref_image: np.ndarray,
    query_px: np.ndarray,
    ref_px: np.ndarray,
    inlier_mask: "np.ndarray | None" = None,
    max_pairs: int = 200,
    warp_corners: bool = False,
    show: bool = True,
) -> "plt.Figure | None":
    """Side-by-side query + reference image with inlier/outlier connecting lines.

    Lines are green for inliers, red for outliers. White dots mark each keypoint.
    When warp_corners=True, draws both warped boundaries under the inlier homography:
    cyan quad on the reference side (query corners → reference space) and yellow quad
    on the query side (reference corners → query space via H⁻¹). Both are always drawn
    so whichever fits inside its image is visible regardless of relative image sizes.

    Args:
        query_image:  HxWx3 uint8 RGB query image.
        ref_image:    HxWx3 uint8 RGB reference image (caller-resolved).
        query_px:     (K, 2) pixel coordinates in the query image.
        ref_px:       (K, 2) pixel coordinates in the reference image. Use
                      correspondences_for_ref() to slice a localizer result per frame.
        inlier_mask:  (K,) bool; None draws every pair as an inlier.
        max_pairs:    Cap on lines drawn — random subsample if exceeded.
        warp_corners: Draw homography-warped boundaries on both sides.
        show:         Call plt.show() (notebook behaviour). Dashboard passes False.

    Returns:
        The matplotlib Figure, or None when there is nothing to plot.
    """
    # Nothing to draw without correspondences
    kpts0 = np.asarray(query_px)
    kpts1 = np.asarray(ref_px)
    if len(kpts0) == 0:
        logger.warning("plot_correspondences: no correspondences to plot")
        return None
    inliers = np.ones(len(kpts0), dtype=bool) if inlier_mask is None else np.asarray(inlier_mask, dtype=bool)

    # Compute homography from all inliers before subsampling — subsampled set degrades H
    H = None
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
                logger.debug("plot_correspondences: homography degenerate — skipping corner warp")
        else:
            logger.debug(
                "plot_correspondences: only %d inliers — need ≥4 for corner warp",
                len(inlier_kpts0),
            )

    if len(kpts0) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(kpts0), max_pairs, replace=False)
        kpts0, kpts1, inliers = kpts0[idx], kpts1[idx], inliers[idx]

    W = query_image.shape[1]
    query_image_disp = query_image.copy()
    ref_image_disp = ref_image.copy()
    if warp_corners and H is not None:
        # Query corners → reference space: cyan quad on reference side
        h_q, w_q = query_image.shape[:2]
        corners_q = np.array(
            [[0, 0], [w_q - 1, 0], [w_q - 1, h_q - 1], [0, h_q - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped_q = cv2.perspectiveTransform(corners_q, H)
        for i in range(4):
            cv2.line(
                ref_image_disp,
                tuple(warped_q[i - 1][0].astype(int)),
                tuple(warped_q[i][0].astype(int)),
                (0, 255, 255),
                4,
            )  # cyan (RGB)

        # Reference corners → query space via H⁻¹: yellow quad on query side
        h_r, w_r = ref_image.shape[:2]
        corners_r = np.array(
            [[0, 0], [w_r - 1, 0], [w_r - 1, h_r - 1], [0, h_r - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        H_inv = np.linalg.inv(H)
        warped_r = cv2.perspectiveTransform(corners_r, H_inv)
        for i in range(4):
            cv2.line(
                query_image_disp,
                tuple(warped_r[i - 1][0].astype(int)),
                tuple(warped_r[i][0].astype(int)),
                (255, 255, 0),
                4,
            )  # yellow (RGB)
    # Query and reference may have different resolutions (e.g. 2988p GoPro query vs
    # 1080p reconstruction frames): rescale the reference side to the query height and
    # scale its keypoints identically, otherwise the side-by-side concat raises.
    ref_scale = query_image_disp.shape[0] / ref_image_disp.shape[0]
    if ref_scale != 1.0:
        new_w = max(1, int(round(ref_image_disp.shape[1] * ref_scale)))
        ref_image_disp = cv2.resize(ref_image_disp, (new_w, query_image_disp.shape[0]))
    combined = np.concatenate([query_image_disp, ref_image_disp], axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(combined)
    for (x0, y0), (x1, y1), ok in zip(kpts0, kpts1, inliers):
        color = "lime" if ok else "red"
        ax.plot([x0, x1 * ref_scale + W], [y0, y1 * ref_scale], color=color, linewidth=0.8, alpha=0.6)
    ax.scatter(kpts0[:, 0], kpts0[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.scatter(kpts1[:, 0] * ref_scale + W, kpts1[:, 1] * ref_scale, s=8, c="white", zorder=3, linewidths=0)
    ax.axvline(W, color="white", linewidth=1, alpha=0.5)
    ax.axis("off")
    ax.set_title(f"query ↔ reference — {inliers.sum()}/{len(inliers)} inliers shown")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_inlier_distribution(
    ref_frame_indices: "np.ndarray | None",
    inlier_mask: "np.ndarray | None",
    n_frames: "int | None" = None,
    frame_sources: "list[str] | None" = None,
) -> "plt.Figure | None":
    """Per-reference-image inlier bars with total-correspondence markers.

    Bars are coloured viridis by frame index (frame order == time) so this plot
    cross-reads with the 3D camera view. Each bar gets a black tick at that
    image's total correspondence count; when totals are uniform across images
    the ticks collapse to a single dashed horizontal line. Frames whose source
    is 'localized' get a red bar edge.

    Args:
        ref_frame_indices: (M,) reference frame index per correspondence.
        inlier_mask:       (M,) bool inlier flag per correspondence.
        n_frames:          Total reference frames (bars include zero-match frames);
                           defaults to max(ref_frame_indices) + 1.
        frame_sources:     Per-frame provenance list ('reconstruction' | 'localized').

    Returns:
        The matplotlib Figure, or None when there is nothing to plot.
    """
    if ref_frame_indices is None or inlier_mask is None:
        logger.warning("plot_inlier_distribution: no correspondence data to plot")
        return None

    # Clamp: bincount(minlength=n) never truncates, so a stale/short n_frames would
    # desync bar x-positions from counts — grow n to cover every referenced frame
    idx = np.asarray(ref_frame_indices).astype(np.intp)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)
    n_used = int(idx.max()) + 1 if len(idx) else 0
    n = max(int(n_frames) if n_frames is not None else 0, n_used)
    totals = np.bincount(idx, minlength=n)
    inliers = np.bincount(idx[inlier_mask], minlength=n)

    # Viridis by frame index — matches the time colouring of the 3D camera plot
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, max(n, 2)))[:n]
    edge = [
        "red" if frame_sources is not None and i < len(frame_sources) and frame_sources[i] == "localized" else "none"
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    x = np.arange(n)
    ax.bar(x, inliers, color=colors, edgecolor=edge, linewidth=1.5)

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
        f"{n_inliers}/{n_correspondences} inliers " f"({100 * n_inliers / max(n_correspondences, 1):.0f}%)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig
