"""
Reconstruction quality across frames: how well overlapping views agree, no ground truth needed.

- depth_pairs: depth error where two views overlap, n_pixels shared; plus a pooled residual histogram
- photometric_pairs: NCC of pixel colors after warping one view into another through its depth
- frames.median_abs_rel_depth_error: per frame, median |depth error| over the pairs touching it
- frames.covered_fraction: share of each original frame that survived the model's crop
- frames.confidence_median: median backbone confidence; not comparable across backbones
- report-only, scale-free; column meanings: docs/source/api/geometry.rst
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import intrinsics_to_original, transform_points
from collab_splats.preproc import frames

logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

# Keyframe stem contract (frame_{idx:06d}) guarding the source-frame join
# - frame_idx_from_path reads any numeric tail: IMG_1234 -> 1234, 00019 -> 19
# - a wrong index silently pairs real frames with the wrong rows, worse than none
# - six or more digits, so the contract holds past a million frames
_FRAME_STEM_RE = re.compile(r"frame_\d{6,}$")

########################################################################
# Results
########################################################################


@dataclass
class PairStats:
    """
    Depth agreement of one ordered frame pair, one depth_pairs row.

    - filled by the multiview depth pass (pointcloud/feedforward/base.py)
    - depth error quantity: (d_sampled - d_expected) / d_expected, frame j against frame i
    - frame separation is abs(idx1 - idx2), not a field

    Args:
        idx1: first frame index.
        idx2: second frame index.
        n_pixels: pixels the depth pass compared.
        median_rel_depth_error: signed scale offset s - 1; 0.02 means frame j is 2% deeper.
        iqr_rel_depth_error: spread with the bias removed, i.e. geometric noise.
        median_parallax_deg: how well this pair can see depth at all.
        median_depth: the "worse further away?" axis, as a column.
    """

    idx1: int
    idx2: int
    n_pixels: int | None = None
    median_rel_depth_error: float | None = None
    iqr_rel_depth_error: float | None = None
    median_parallax_deg: float | None = None
    median_depth: float | None = None


########################################################################
# Shared: residual axis
########################################################################


def residual_bin_edges(n_samples: int) -> np.ndarray:
    """
    Histogram edges for n_samples bounded residuals, fixed before any residual is seen.

    - per-pixel residuals number N^2*H*W: too many to hold, and a range pre-pass would double the cost
    - range (-1, 1) by construction: bounded_residual maps every residual into it, none clip
    - bin count from Rice's rule, k = 2 * n**(1/3): more pixels justify finer bins
    - below roughly 20 frames the pixel-level bins go coarse; per-pair medians are unaffected
    - the bin count is always even, so the histogram folds to |r| by adding the two halves

    Args:
        n_samples: residual count the histogram will hold.

    Returns:
        k + 1 evenly spaced edges over [-1, 1], k even.
    """
    k = 2 * max(1, int(round(max(int(n_samples), 1) ** (1.0 / 3.0))))
    return np.linspace(-1.0, 1.0, k + 1)


def bounded_residual(rel: np.ndarray | float) -> np.ndarray:
    """
    Map a relative depth residual onto (-1, 1) so a fixed histogram can never miss it.

    - r / (1 + |r|) is monotone, so quantiles survive the map exactly
    - invert with u / (1 - |u|)
    - no chosen range: clipping piles the tail into end bins, and np.histogram drops out-of-range values

    Args:
        rel: relative depth residuals, any shape.

    Returns:
        Float64 array of the same shape, in (-1, 1).
    """
    r = np.asarray(rel, dtype=np.float64)
    return r / (1.0 + np.abs(r))


########################################################################
# Per pair: depth
########################################################################


def compute_depth_error(collected: dict) -> tuple[dict, dict]:
    """
    Per-direction depth disagreement and the pixel residual histogram, both columnar.

    - model resolution: original resolution would sample guided-filtered depth instead
    - rows are ordered pair directions: (i, j) and (j, i) differ, occlusion is asymmetric

    Args:
        collected: the dict compute_multiview_depth_confidence(collect=...) filled.

    Returns:
        (depth_pairs, depth_residual_histogram)
        - depth_pairs: {idx1, idx2, n_pixels, median_rel_depth_error, iqr_rel_depth_error,
          median_parallax_deg, median_depth}, one entry per pair direction
        - depth_residual_histogram: {counts, bin_edges}
    """
    pairs = collected["pairs"]
    logger.info("Depth error: %d pair directions", len(pairs))

    # One entry per pair direction in every column, raw for the reader to bin
    depth_pairs = {
        "idx1": [p.idx1 for p in pairs],
        "idx2": [p.idx2 for p in pairs],
        "n_pixels": [p.n_pixels for p in pairs],
        "median_rel_depth_error": [p.median_rel_depth_error for p in pairs],
        "iqr_rel_depth_error": [p.iqr_rel_depth_error for p in pairs],
        "median_parallax_deg": [p.median_parallax_deg for p in pairs],
        "median_depth": [p.median_depth for p in pairs],
    }

    # The one pre-binned output, for the one per-pixel quantity
    histogram = {
        "counts": collected["rel_depth_error_counts"].tolist(),
        "bin_edges": collected["rel_depth_error_edges"].tolist(),
    }
    return depth_pairs, histogram


########################################################################
# Per pair: photometric
########################################################################


def compute_photometric_ncc(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray | None = None,
    max_separation: int = 2,
    min_samples: int = 32,
) -> dict:
    """
    Warp each frame into its neighbors through pose + depth and correlate the RGB.

    - zero-mean NCC via np.corrcoef: 1.0 is perfect agreement, 0.0 is none
    - normalizing cancels the [0, 255] vs [0, 1] image-scale split and exposure or gain change
    - the only appearance metric: disagreement seen only here points at image formation
    - runs at original resolution; model-grid depth is upsampled here together with its K
    - rows are unordered pairs (i < j)

    Args:
        images: (N, H, W, 3) RGB, original resolution.
        depth: (N, h, w) Z-depth on the model grid, or on the image grid already.
        intrinsics: (N, 3, 3) K matching `depth`'s grid; rescaled here if depth is.
        extrinsics: (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows, required only when depth needs upsampling.
        max_separation: pairs per frame; distant frames differ mostly by lighting and
            viewpoint, so the cost stays O(N * max_separation) rather than O(N^2).
        min_samples: floor on overlapping pixels (the `n_pixels` column); corrcoef on two
            values returns exactly +-1 whatever they are.

    Returns:
        {idx1, idx2, photometric_ncc, n_pixels}, columnar; empty lists when no pair correlates.

    Raises:
        ValueError: depth needs upsampling but original_coords is missing or describes
            another resolution.
    """
    t0 = time.perf_counter()
    N = len(depth)
    ih, iw = images.shape[1:3]
    logger.info("Photometric NCC: %d frames at %dx%d, max_separation=%d", N, iw, ih, max_separation)

    # Lift model-grid depth and its K to the image grid together
    # - see bundle_adjustment.check_model_resolution
    if depth.shape[1:] != (ih, iw):
        if original_coords is None:
            raise ValueError(
                f"depth is {depth.shape[1:]} but images are {(ih, iw)}; " "original_coords is required to upsample"
            )

        # Crop boxes are in original pixels, so `images` must be that canvas
        # - same check, same refusal as the native-resolution mesh path (mesh/io.py)
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))
        if (ih, iw) != expected_hw:
            raise ValueError(
                f"images are {(ih, iw)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )

        # Imported here to keep this metric import-light
        # - collab_splats.mesh reaches Warp and meshoptimizer through texture.py
        from collab_splats.mesh.io import upsample_depths

        model_h, model_w = depth.shape[1:]

        # Guide must be uint8 [0, 255]; the contract's `images` may be [0, 1] or [0, 255]
        # - scale decided once over the whole array, never per frame
        # - a per-frame decision would amplify a dark [0, 255] frame 255x
        rgb_scale = 255.0 if images.max() <= 1.0 else 1.0
        guides = np.clip(np.asarray(images) * rgb_scale, 0, 255).astype(np.uint8)
        lifted_d = upsample_depths(depth, guides, original_coords[:, :4])

        # Undo crop-then-resize on K: scale is model/crop, not model/canvas
        lifted_K = intrinsics_to_original(intrinsics, original_coords[:, :4], (model_h, model_w))
        depth, intrinsics = lifted_d, lifted_K

    # Pixel grid for unprojection
    H, W = depth.shape[1:]
    cam2world = np.linalg.inv(extrinsics)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)

    # Closed-form pair count, so the bar states the real unit of work
    # - counting frames would leave the reader to multiply
    n_pairs_expected = sum(min(N, i + max_separation + 1) - (i + 1) for i in range(N))
    cols: dict[str, list] = {"idx1": [], "idx2": [], "photometric_ncc": [], "n_pixels": []}
    for i in tqdm(range(N), desc=f"Photometric NCC ({n_pairs_expected} pairs)", unit="frame"):
        # Unproject frame i's pixels to world through its own K and pose
        # - names follow the multiview loop in pointcloud/feedforward/base.py
        pts_cam_i = (np.linalg.inv(intrinsics[i]) @ pix.T).T * depth[i].reshape(-1, 1)
        pts_world = transform_points(pts_cam_i, cam2world[i])

        for j in range(i + 1, min(N, i + max_separation + 1)):
            # Project into frame j and look up the color that landed there
            # - no occlusion test: unlike the depth loop, there is no frame-j depth to test
            # - occluded pixels stay in and read as disagreement
            pts_cam_j = transform_points(pts_world, extrinsics[j])
            proj_j = (intrinsics[j] @ pts_cam_j.T).T
            z = np.clip(proj_j[:, 2], 1e-6, None)

            # Nearest sampling, matching the depth pass
            # - bilinear across a depth discontinuity blends two surfaces into neither's color
            ui = np.round(proj_j[:, 0] / z).astype(np.int64)
            vi = np.round(proj_j[:, 1] / z).astype(np.int64)
            in_front = pts_cam_j[:, 2] > 0

            # Drop depth == 0: "no observation", not a surface at distance 0
            # - unprojected, it sits at frame i's camera center, possibly visible in frame j
            # - in_front does not cover it: the center is behind frame j only for some poses
            ok = in_front & (depth[i].ravel() > 0)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < min_samples:
                continue

            # Paired RGB samples: frame i's pixel against where it landed in frame j
            a = images[i].reshape(-1, 3)[ok].ravel().astype(np.float64)
            b = images[j][vi[ok], ui[ok]].ravel().astype(np.float64)

            # Skip flat patches: no variance to correlate
            # - corrcoef would return nan, but only after a divide-by-zero RuntimeWarning
            # - sky or a blank wall would warn once per pair
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            ncc = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(ncc):
                continue
            for col, v in zip(cols, (i, j, ncc, int(ok.sum()))):
                cols[col].append(v)

    logger.info(
        "Photometric NCC: %d pairs correlated in %.2fs", len(cols["photometric_ncc"]), time.perf_counter() - t0
    )
    return cols


########################################################################
# Per frame + assembly
########################################################################


def compute_reconstruction_quality(
    collected: dict,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray,
    image_names: list[str],
    confidence: np.ndarray | None,
    images: np.ndarray | None,
) -> dict:
    """
    Every table the report holds, from arrays; the Reconstructor stage owns all IO.

    - a missing optional input nulls exactly its table or columns; a failing measurement raises
    - column meanings: docs/source/api/geometry.rst

    Args:
        collected: the dict compute_multiview_depth_confidence(collect=...) filled.
        depth: (N, h, w) Z-depth on the model grid.
        intrinsics: (N, 3, 3) K on the model grid.
        extrinsics: (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows: x0, y0, x1, y1, original width, original height.
        image_names: per-frame image file names, reconstruction order.
        confidence: (N, h, w) per-pixel confidence, or None.
        images: (M, H, W, 3) original-resolution RGB, M <= N, or None.

    Returns:
        {"frames", "depth_pairs", "depth_residual_histogram", "photometric_pairs"}
        - photometric_pairs None without images
    """
    n = len(depth)
    coords = np.asarray(original_coords, dtype=np.float64)

    depth_pairs, histogram = compute_depth_error(collected)

    # Photometric over the frames that have images; read_frames order is reconstruction order
    photometric_pairs = None
    if images is not None:
        m = len(images)
        photometric_pairs = compute_photometric_ncc(
            images, depth[:m], intrinsics[:m], extrinsics[:m], original_coords=original_coords[:m]
        )

    # Per-frame median |residual| over the depth pairs touching each frame
    # - scanning every pair per frame is O(N * pairs) = O(N^3), so it carries a bar
    median_abs = []
    for k in tqdm(range(n), desc="Per-frame medians", unit="frame", leave=False):
        v = [abs(p.median_rel_depth_error) for p in collected["pairs"] if k in (p.idx1, p.idx2)]
        median_abs.append(float(np.median(v)) if v else None)

    # Source frame index only where the stem is on the frame_{idx:06d} contract
    frame_idx = [
        frames.frame_idx_from_path(name) if _FRAME_STEM_RE.fullmatch(Path(name).stem) else None
        for name in image_names
    ]

    # Fraction of each original frame the model crop reconstructed
    # - a center crop of a wide source loses a band no model-grid table can see
    covered = [float(max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / max(c[4] * c[5], 1e-9)) for c in coords]

    frames_table = {
        "frame_idx": frame_idx,
        "covered_fraction": covered,
        "median_abs_rel_depth_error": median_abs,
        "confidence_median": [None] * n if confidence is None else [float(np.median(c)) for c in confidence],
    }
    return {
        "frames": frames_table,
        "depth_pairs": depth_pairs,
        "depth_residual_histogram": histogram,
        "photometric_pairs": photometric_pairs,
    }
