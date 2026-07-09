"""Visualization helpers for localization results."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt

from .localizer import LocalizationResult

logger = logging.getLogger(__name__)


def plot_correspondences(
    loc: LocalizationResult,
    query_image: np.ndarray,
    image_paths: list,
    max_pairs: int = 200,
    warp_corners: bool = False,
) -> None:
    """Side-by-side query + best reference frame with inlier/outlier connecting lines.

    Best reference frame = one contributing the most inlier correspondences.
    Lines are green for inliers, red for outliers. White dots mark each keypoint.
    When warp_corners=True, draws both warped boundaries under the inlier homography:
    cyan quad on the reference side (query corners → reference space) and yellow quad
    on the query side (reference corners → query space via H⁻¹). Both are always drawn
    so whichever fits inside its image is visible regardless of relative image sizes.

    Args:
        loc:          LocalizationResult from CameraLocalizer.localize().
        query_image:  HxWx3 uint8 RGB query image.
        image_paths:  Reference image paths (same order as CameraLocalizer input).
        max_pairs:    Cap on lines drawn — random subsample if exceeded.
        warp_corners: Draw homography-warped boundaries on both sides.
    """

    if loc.pose is None or loc.inlier_mask is None or loc.pts2d_ref is None or loc.ref_frame_indices is None:
        logger.warning("plot_correspondences: no valid localization result to plot")
        return

    # Best reference frame = one with most inlier correspondences
    inlier_frames = loc.ref_frame_indices[loc.inlier_mask]
    if len(inlier_frames) == 0:
        logger.warning("plot_correspondences: zero inliers — nothing to plot")
        return
    best_ref_idx = int(np.bincount(inlier_frames.astype(np.intp)).argmax())
    frame_mask = loc.ref_frame_indices == best_ref_idx

    kpts0 = loc.pts2d[frame_mask]          # (K, 2) query
    kpts1 = loc.pts2d_ref[frame_mask]      # (K, 2) reference
    inliers = loc.inlier_mask[frame_mask]  # (K,) bool

    # Compute homography from all inliers before subsampling — subsampled set degrades H
    H = None
    if warp_corners:
        inlier_kpts0 = kpts0[inliers]
        inlier_kpts1 = kpts1[inliers]
        if len(inlier_kpts0) >= 4:
            H, _ = cv2.findHomography(
                inlier_kpts0, inlier_kpts1,
                cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999,
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

    ref_bgr = cv2.imread(str(image_paths[best_ref_idx]))
    if ref_bgr is None:
        raise FileNotFoundError(f"plot_correspondences: cannot read {image_paths[best_ref_idx]}")
    ref_image = ref_bgr[..., ::-1].copy()

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
            cv2.line(ref_image_disp,
                     tuple(warped_q[i - 1][0].astype(int)),
                     tuple(warped_q[i][0].astype(int)),
                     (0, 255, 255), 4)  # cyan (RGB)

        # Reference corners → query space via H⁻¹: yellow quad on query side
        h_r, w_r = ref_image.shape[:2]
        corners_r = np.array(
            [[0, 0], [w_r - 1, 0], [w_r - 1, h_r - 1], [0, h_r - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        H_inv = np.linalg.inv(H)
        warped_r = cv2.perspectiveTransform(corners_r, H_inv)
        for i in range(4):
            cv2.line(query_image_disp,
                     tuple(warped_r[i - 1][0].astype(int)),
                     tuple(warped_r[i][0].astype(int)),
                     (255, 255, 0), 4)  # yellow (RGB)
    combined = np.concatenate([query_image_disp, ref_image_disp], axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(combined)
    for (x0, y0), (x1, y1), ok in zip(kpts0, kpts1, inliers):
        color = "lime" if ok else "red"
        ax.plot([x0, x1 + W], [y0, y1], color=color, linewidth=0.8, alpha=0.6)
    ax.scatter(kpts0[:, 0], kpts0[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.scatter(kpts1[:, 0] + W, kpts1[:, 1], s=8, c="white", zorder=3, linewidths=0)
    ax.axvline(W, color="white", linewidth=1, alpha=0.5)
    ax.axis("off")
    ax.set_title(
        f"query ↔ reference frame {best_ref_idx} — "
        f"{inliers.sum()}/{len(inliers)} inliers shown"
    )
    plt.tight_layout()
    plt.show()
