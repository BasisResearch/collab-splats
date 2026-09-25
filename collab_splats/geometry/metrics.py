"""
Reference-free scene error metrics over a `pointcloud.zarr` result.

- depth cross-view residuals, photometric NCC, and verify's epipolar rows
- report-only: distributions, no verdicts, nothing fed back into a reconstruction
- reads the result of either pointcloud method (feedforward or sfm)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from collab_splats.geometry.verification import clean_for_json
from collab_splats.preproc import frames

if TYPE_CHECKING:
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

logger = logging.getLogger(__name__)

# Keyframe stem contract (frame_{idx:06d}) guarding the source-frame join
# - frame_idx_from_path reads any numeric tail: IMG_1234 -> 1234, 00019 -> 19
# - a wrong index silently pairs real frames with the wrong rows, worse than none
# - six or more digits, so the contract holds past a million frames
_FRAME_STEM_RE = re.compile(r"frame_\d{6,}$")

########################################
# The residual histogram's axis
########################################


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


########################################
# Putting depth error and pixel error on one axis
########################################


def depth_error_in_pixels(rel_residual: float, parallax_deg: float, focal_px: float) -> float | None:
    """
    Pixel equivalent of a pair's relative depth disagreement, at the pair's own parallax.

    - derived, not observed: states a depth residual in the units of pixel errors
    - |r| * d with disparity d = f * alpha (alpha = parallax in radians); baseline cancels
    - divide an observed pixel error by this: ~1 one error seen twice, >>1 pose or appearance
    - <<1 means the error lies along the ray, where this pair's baseline cannot see it
    - None under one pixel of disparity: the pair cannot see depth at all

    Args:
        rel_residual: signed relative depth residual, dZ / Z.
        parallax_deg: the pair's parallax angle, degrees.
        focal_px: focal length, pixels.

    Returns:
        Pixel-equivalent error, or None without parallax; a nan/inf rel_residual propagates
        as nan/inf ("no data"), so filter with ``result is not None and np.isfinite(result)``.
    """
    disparity_px = np.deg2rad(parallax_deg) * focal_px
    if disparity_px < 1.0:
        return None
    return abs(rel_residual) * disparity_px


########################################
# Depth cross-view error
########################################


def compute_depth_error(collected: dict, focal_px: float, resolution: str) -> dict:
    """
    How much the views disagree about depth: scale bias, geometric noise, parallax.

    - model resolution: original resolution would sample guided-filtered depth instead
    - rows are ordered pair directions: (i, j) and (j, i) differ, occlusion is asymmetric

    Args:
        collected: the dict compute_multiview_depth_confidence(collect=...) filled.
        focal_px: mean focal in pixels, used only to state the residual in pixel units.
        resolution: "WxH" of the grid, stamped into the output for the reader.

    Returns:
        Depth block with per-direction rows, the residual histogram and correlations;
        {"available": False, "reason": ...} when no pair produced residuals.
    """
    pairs = collected["pairs"]
    if not pairs:
        logger.info("Depth error: unavailable — no overlapping view pairs produced residuals")
        return {
            "available": False,
            "reason": "no overlapping view pairs produced depth residuals",
            "grid": "model",
            "resolution": resolution,
        }

    t0 = time.perf_counter()
    logger.info("Depth error: assembling %d pair directions at %s", len(pairs), resolution)

    # One row per pair. Every column is raw, so the reader bins, thresholds and plots.
    rows = [
        {
            "idx1": p.idx1,
            "idx2": p.idx2,
            "frame_separation": abs(p.idx1 - p.idx2),  # how far apart the two frames are
            "n_pixels": p.n_pixels,
            # Signed, so scale reads straight off it: s = 1 + median_rel_depth_error.
            "median_rel_depth_error": p.median_rel_depth_error,
            "iqr_rel_depth_error": p.iqr_rel_depth_error,  # bias removed: geometric noise
            "median_parallax_deg": p.median_parallax_deg,
            "median_depth": p.median_depth,
            # None, not 0.0 — under a pixel of disparity a zero would read as "no error"
            # when it means "cannot tell".
            "depth_error_px": depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, focal_px),
        }
        for p in pairs
    ]

    # Read correlation columns off the rows, never off `pairs` again
    # - a second copy of abs(p.idx1 - p.idx2) is out of reach of every row assertion
    abs_rel_depth_error = np.array([abs(r["median_rel_depth_error"]) for r in rows], dtype=np.float64)
    depths = np.array([r["median_depth"] for r in rows], dtype=np.float64)
    frame_seps = np.array([r["frame_separation"] for r in rows], dtype=np.float64)
    under_1px = sum(1 for r in rows if r["depth_error_px"] is None)

    # Quantiles off a histogram of the bounded residual, inverted back
    # - bounded_residual is monotone, so the inversion is exact
    counts, edges = collected["rel_depth_error_counts"], collected["rel_depth_error_edges"]
    grid = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)

    def read_quantiles(hist_counts: np.ndarray, hist_edges: np.ndarray) -> dict:
        """
        The quantile grid, read off one histogram and inverted through bounded_residual.
        """
        rv = stats.rv_histogram((hist_counts, hist_edges))
        out = {}
        for q in grid:
            u = float(rv.ppf(q))
            out[str(q)] = u / (1.0 - abs(u))
        return out

    # Signed quantiles, then the same histogram folded to |r|
    # - symmetric edges, even bin count: bins j and k-1-j share |u|, so the fold is exact
    # - signed answers "is there scale bias"; folded compares with other |rel| reports
    half = (len(edges) - 1) // 2
    quantiles = read_quantiles(counts, edges)
    abs_quantiles = read_quantiles(counts[half:] + counts[:half][::-1], edges[half:])

    # Two Spearman correlations, straight from scipy, unfiltered
    # - a tiny sample makes rho meaningless: n=2 gives ~1.0, n=1 gives nan
    # - publishable only because n_pair_directions ships beside it
    # - error_vs_depth: positive rho expected; near 0 or near 1 is the interesting outcome
    correlations = {
        "error_vs_depth": float(stats.spearmanr(depths, abs_rel_depth_error).statistic),
        "error_vs_frame_separation": float(stats.spearmanr(frame_seps, abs_rel_depth_error).statistic),
        "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual rises ~linearly in Z",
    }

    logger.info(
        "Depth error: %d pair directions, %d residual samples over %d bins, " "median |rel| %.4f, in %.2fs",
        len(pairs),
        int(counts.sum()),
        len(counts),
        abs_quantiles["0.5"],
        time.perf_counter() - t0,
    )

    return {
        "available": True,
        "grid": "model",
        "resolution": resolution,
        "units": "relative (dimensionless); parallax in degrees; pixel equivalent in px",
        # Directions, not pairs: the multiview loop is ordered
        # - (i, j) and (j, i) are separate rows, since occlusion is asymmetric
        # - photometric and epipolar count unordered "n_pairs"; this key avoids a phantom 2x
        "n_pair_directions": len(pairs),
        # The one pre-binned output, for the one per-pixel quantity
        # - rv_histogram(...).cdf(bounded_residual(x)) gives the pixel fraction below any x
        "residual_histogram": {
            "counts": counts.tolist(),
            "bin_edges": edges.tolist(),
            "total": int(counts.sum()),
            "quantiles": quantiles,
            "abs_quantiles": abs_quantiles,
            "axis": "bins are over r/(1+|r|); invert with u/(1-|u|)",
        },
        "pair_directions_under_one_pixel_disparity": under_1px,
        "correlations": correlations,
        "pair_directions": rows,
    }


########################################
# Photometric agreement
########################################


def _scale_intrinsics_to_original(
    intrinsics: np.ndarray,
    sx: float,
    sy: float,
    tl_x: float,
    tl_y: float,
) -> np.ndarray:
    """
    Map model-grid K back to original-image pixels.

    - (sx, sy) is model/crop, derived by the caller from the crop box
    - the crop origin is added after the scale is undone

    Args:
        intrinsics: (..., 3, 3) K on the model grid.
        sx: model/crop scale along x, divided out here.
        sy: model/crop scale along y, divided out here.
        tl_x: crop top-left x in original pixels.
        tl_y: crop top-left y in original pixels.

    Returns:
        (..., 3, 3) float64 K in original-image pixels.
    """
    intr = np.array(intrinsics, dtype=np.float64)
    intr[..., 0, 0] = intr[..., 0, 0] / sx
    intr[..., 1, 1] = intr[..., 1, 1] / sy
    intr[..., 0, 2] = intr[..., 0, 2] / sx + tl_x
    intr[..., 1, 2] = intr[..., 1, 2] / sy + tl_y
    return intr


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
    - rows are unordered pairs (i < j), like verify's epipolar block, hence "n_pairs"/"pairs"
    - "resolution" is derived from `images`, so it cannot disagree with the grid used

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
        Photometric block with per-pair rows and the NCC-vs-separation correlation;
        {"available": False, "reason": ...} when no pair produced a correlation.

    Raises:
        ValueError: depth needs upsampling but original_coords is missing or describes
            another resolution.
    """
    t0 = time.perf_counter()
    N = len(depth)
    ih, iw = images.shape[1:3]
    logger.info("Photometric NCC: %d frames at %dx%d, max_separation=%d", N, iw, ih, max_separation)

    # Lift model-grid depth and its K to the image grid together
    # - one grid's depth with the other grid's K collapses the geometry
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
        lifted_K = []
        for k in range(N):
            tlx, tly, crx, cry = (float(v) for v in original_coords[k][:4])
            sx, sy = model_w / (crx - tlx), model_h / (cry - tly)
            lifted_K.append(_scale_intrinsics_to_original(intrinsics[k], sx, sy, tlx, tly))
        depth, intrinsics = lifted_d, np.stack(lifted_K)

    H, W = depth.shape[1:]
    # Derived, never declared: the grid the numbers below are computed on
    resolution = f"{iw}x{ih}"
    cam2world = np.linalg.inv(extrinsics)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)
    ones = np.ones((H * W, 1))

    # Pair count is closed-form from N and max_separation, so the bar can state the real unit of
    # work up front rather than counting frames and leaving the reader to multiply.
    n_pairs_expected = sum(min(N, i + max_separation + 1) - (i + 1) for i in range(N))
    rows = []
    for i in tqdm(range(N), desc=f"Photometric NCC ({n_pairs_expected} pairs)", unit="frame"):
        # Unproject frame i's pixels to world through its own K and pose
        # - names follow the multiview loop in pointcloud/feedforward/base.py
        pts_cam_i = (np.linalg.inv(intrinsics[i]) @ pix.T).T * depth[i].reshape(-1, 1)
        pts_world = (cam2world[i] @ np.concatenate([pts_cam_i, ones], axis=-1).T).T[:, :3]
        # Homogeneous once per i: the j loop re-projects the SAME world points.
        pts_world_h = np.concatenate([pts_world, ones], axis=-1)

        for j in range(i + 1, min(N, i + max_separation + 1)):
            # Project into frame j and look up the color that landed there
            # - no occlusion test: unlike the depth loop, there is no frame-j depth to test
            # - occluded pixels stay in and read as disagreement
            pts_cam_j = (extrinsics[j] @ pts_world_h.T).T[:, :3]
            proj_j = (intrinsics[j] @ pts_cam_j.T).T
            z = np.clip(proj_j[:, 2], 1e-6, None)
            # Nearest sampling, matching the depth pass: bilinear across a depth discontinuity
            # blends two surfaces into a color present on neither.
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
            rows.append(
                {"idx1": i, "idx2": j, "frame_separation": j - i, "photometric_ncc": ncc, "n_pixels": int(ok.sum())}
            )

    if not rows:
        logger.info("Photometric NCC: unavailable — no view pair produced a correlation")
        return {
            "available": False,
            "reason": "no view pairs produced a photometric correlation",
            "grid": "original",
            "resolution": resolution,
        }

    # Read the correlation columns back OFF the rows, so nothing can drift from what ships.
    ncc = np.array([r["photometric_ncc"] for r in rows], dtype=np.float64)
    frame_seps = np.array([r["frame_separation"] for r in rows], dtype=np.float64)

    # Straight from scipy, unfiltered, as in the depth block
    # - two rows give +-1.0 by construction; "n_pairs" and the raw rows ship beside it
    # - suppressing it would be a verdict
    correlations = {"ncc_vs_frame_separation": float(stats.spearmanr(frame_seps, ncc).statistic)}

    logger.info(
        "Photometric NCC: %d pairs correlated, median NCC %.4f, in %.2fs",
        len(rows),
        float(np.median(ncc)),
        time.perf_counter() - t0,
    )

    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "zero-mean normalized cross-correlation; 1.0 = perfect agreement",
        # UNORDERED, like verify's epipolar block: one row per pair, and the rho's sample size.
        "n_pairs": len(rows),
        "correlations": correlations,
        "pairs": rows,
    }


def extract_photometric(result: "FeedforwardResult", images_dir: Path, n: int) -> dict:
    """
    Read a scene's RGB keyframes, then correlate them with compute_photometric_ncc.

    - a missing images dir disables only this metric; a failing measurement raises
    - compute_photometric_ncc stays array-only, so its tests need no scene on disk

    Args:
        result: the `pointcloud.zarr` result supplying depth, intrinsics, extrinsics, crop boxes.
        images_dir: the scene's images/ directory of original-resolution keyframes.
        n: reconstruction frame count; the read is capped at the first n frames.

    Returns:
        The photometric block, or {"available": False, "reason": ...} when the images are
        missing.
    """
    if not frames.frame_paths(images_dir):
        logger.info("Photometric NCC: unavailable — no frame images at %s", images_dir)
        return {"available": False, "grid": "original", "reason": f"no frame images at {images_dir}"}

    # read_frames returns frames in filename order, the reconstruction's order
    # - each name's frame_idx is a source-video position; slicing by it mispairs depth
    t0 = time.perf_counter()
    rgbs = frames.read_frames(images_dir)[:n].astype(np.float32)
    m = len(rgbs)
    logger.info(
        "Photometric NCC: read %d original-resolution frames from %s in %.2fs", m, images_dir, time.perf_counter() - t0
    )

    # No resolution argument: compute_photometric_ncc derives it from `rgbs` itself, so the
    # stamped grid cannot disagree with the grid the numbers were computed on.
    return compute_photometric_ncc(
        rgbs,
        result.depth[:m],
        result.intrinsics[:m],
        result.extrinsics[:m],
        original_coords=result.original_coords[:m],
    )


########################################
# Stage entry point
########################################


def _running_error(rows: list[dict], key: str) -> dict:
    """
    Cumulative |step| along the trajectory, one step per consecutive-frame pair.

    - separation-1 rows only: a longer pair is a revisit, not a step
    - rows grouped by unordered pair: the depth pass emits both (k, k+1) and (k+1, k)
    - mean of |value| per group, never signed: +0.10 and -0.09 would read as agreement

    Args:
        rows: per-pair rows carrying idx1, idx2, frame_separation and `key`.
        key: the column to accumulate.

    Returns:
        {"frame_index": each step's later frame, "cumulative": running sum of step errors}.
    """
    grouped: dict[tuple[int, int], list[float]] = {}
    for x in rows:
        if x["frame_separation"] != 1 or x[key] is None or not np.isfinite(x[key]):
            continue
        lo, hi = sorted((int(x["idx1"]), int(x["idx2"])))
        grouped.setdefault((lo, hi), []).append(abs(float(x[key])))

    steps = sorted(grouped.items())
    return {
        "frame_index": [hi for (_, hi), _ in steps],
        "cumulative": np.cumsum([float(np.mean(v)) for _, v in steps]).tolist(),
    }


def build_reconstruction_quality_report(
    zarr_path: Path,
    verification_json: Path,
    images_dir: Path,
    output_path: Path,
    backend: str,
    rel_thresh: float = 0.05,
) -> dict:
    """
    Run every metric that can run and write reconstruction_quality_report.json.

    - never raises on a dead metric: each missing input disables exactly one metric
    - missing inputs: confidence array, verification.json, or the images/ dir
    - a failing measurement, or a verification.json missing a key or row field, raises
    - no grades, causes or flagged frames

    Args:
        zarr_path: the `pointcloud.zarr` result to report on.
        verification_json: verify's verification.json; epipolar rows need it.
        images_dir: original-resolution keyframes for the photometric metric.
        output_path: where the report JSON is written.
        backend: pointcloud backend name, stamped into the report's scene block.
        rel_thresh: relative depth tolerance for the dense multiview pass.

    Returns:
        The report dict; the written JSON has every nan replaced by null.
    """
    # Deferred import: pointcloud.feedforward.base imports this module (cycle)
    from collab_splats.pointcloud.feedforward.base import (
        FeedforwardResult,
        compute_multiview_depth_confidence,
    )

    r = FeedforwardResult.load_zarr(zarr_path)
    n = len(r.depth)
    model_res = f"{r.model_width}x{r.model_height}"
    focal_px = float(r.intrinsics[:, 0, 0].mean() + r.intrinsics[:, 1, 1].mean()) / 2.0
    logger.info("Report on %s: %d frames, model resolution %s", zarr_path, n, model_res)

    # One dense pass: depth residuals, scale split, parallax, per-pair depth
    # - abs_thresh 0.0 keeps the pass scale-invariant across backbones
    collected: dict = {}
    compute_multiview_depth_confidence(
        r.depth, r.intrinsics, r.extrinsics, abs_thresh=0.0, rel_thresh=rel_thresh, collect=collected
    )
    depth_m = compute_depth_error(collected, focal_px, model_res)

    # Epipolar: verify's tables read off disk; the matcher is not re-run
    # - the only rows that never touch depth: moving here but not in depth rows is pose error
    # - already original resolution, like verify's keypoints
    image_width = int(r.original_coords[0][4])
    verification_json = Path(verification_json)
    if not verification_json.exists():
        logger.info(
            "Epipolar: unavailable — no verification.json at %s "
            "(set pointcloud.geometric_verification: true, or run --stages verify)",
            verification_json,
        )
        epipolar_m = {
            "available": False,
            "grid": "original",
            "reason": f"no verification.json at {verification_json} — set "
            "pointcloud.geometric_verification: true or run --stages verify",
        }
    else:
        t0 = time.perf_counter()
        logger.info("Epipolar: loading verify tables from %s", verification_json)
        data = json.loads(verification_json.read_text())
        # Per-row inlier_ratio, the same expression verify aggregates
        # - kept per row, not as a distribution, so it joins against the depth rows
        epi_pairs = []
        for s in data["pair_stats"]:
            n_m, n_i = s["num_matches"], s["num_inliers"]
            epi_pairs.append(
                {**s, "frame_separation": abs(s["idx1"] - s["idx2"]), "inlier_ratio": (n_i / n_m) if n_m else None}
            )
        # A bare pixel count is not comparable across backbones (a 518 crop against 448x592),
        # so the fraction of image width ships alongside it.
        epi_frames = []
        for name, fs in sorted(data["frame_stats"].items()):
            px = fs["mean_reproj_error_px"]
            epi_frames.append(
                {**fs, "name": name, "mean_reproj_error_frac_width": None if px is None else px / image_width}
            )
        logger.info(
            "Epipolar: %d verified pairs, %d frame reprojection rows, in %.2fs",
            len(epi_pairs),
            len(epi_frames),
            time.perf_counter() - t0,
        )
        epipolar_m = {
            "available": True,
            "grid": "original",
            "resolution": f"width={image_width}",
            "units": "degrees; reprojection in px and as a fraction of image width",
            "source": str(verification_json),
            "n_pairs": len(epi_pairs),
            "pairs": epi_pairs,
            "frames": epi_frames,
        }

    photometric_m = extract_photometric(r, images_dir, n)

    # Per-frame median |residual| — the column both the confidence check and the ranks read.
    # Scanning every pair per frame makes this O(N * pairs) = O(N^3), so it carries a bar.
    per_frame = {}
    for k in tqdm(range(n), desc="Per-frame medians", unit="frame", leave=False):
        v = [abs(p.median_rel_depth_error) for p in collected["pairs"] if k in (p.idx1, p.idx2)]
        if v:
            per_frame[k] = float(np.median(v))

    # Confidence vs error: does the model know when it is wrong?
    # - confidence is an input being validated: one correlation, not a metric of its own
    # - no small-sample floor; the sample count ships nested with the rho
    # - both None without confidence: never computed differs from a tiny sample
    conf_rho, conf_n = None, None
    if r.confidence is not None:
        conf = np.asarray(r.confidence)
        conf_n = len(per_frame)
        conf_rho = float(
            stats.spearmanr(
                np.array([float(np.median(conf[k])) for k in per_frame], dtype=np.float64),
                np.array(list(per_frame.values()), dtype=np.float64),
            ).statistic
        )

    # Each frame's 0..1 rank in this scene's own distribution, never a label
    # - within-scene ranks need no threshold comparable across backbones
    ks = list(per_frame)
    ranks = {}
    if len(ks) > 1:
        rk = (stats.rankdata([per_frame[k] for k in ks]) - 1) / (len(ks) - 1)
        ranks = {int(k): float(x) for k, x in zip(ks, rk)}

    # Does disagreement build along the trajectory?
    # - depth rows live under "pair_directions", epipolar under "pairs"; only the key differs
    # - a dead channel says so: an empty series would read as "error stayed at zero"
    running = {
        name: (_running_error(m[rows_key], key) if m["available"] else {"available": False, "reason": m["reason"]})
        for name, rows_key, key, m in (
            ("depth", "pair_directions", "median_rel_depth_error", depth_m),
            ("epipolar", "pairs", "rot_error_deg", epipolar_m),
        )
    }

    # Reconstruction index -> source video frame index
    # - per-frame blocks are keyed 0..N-1, not the source index once sampling skips frames
    # - from image_paths (always present), parsed only when the stem matches _FRAME_STEM_RE
    # - off-contract names yield None: the join degrades per frame, the report survives
    source_frame_indices: list[int | None] = []
    for p in r.image_paths:
        m = _FRAME_STEM_RE.fullmatch(Path(str(p)).stem)
        source_frame_indices.append(frames.frame_idx_from_path(p) if m else None)

    report = {
        "scene": {"backend": backend, "n_frames": n, "model_resolution": model_res, "zarr": str(zarr_path)},
        "measurements_available": sorted(
            k
            for k, m in (("epipolar", epipolar_m), ("depth", depth_m), ("photometric", photometric_m))
            if m["available"]
        ),
        "measurements": {"epipolar": epipolar_m, "depth": depth_m, "photometric": photometric_m},
        "confidence_vs_error": {"spearman": conf_rho, "n_frames": conf_n},
        # Read running_error against error_vs_frame_separation before calling it drift: frame index
        # is a confounded axis, since scene content, motion speed and exposure all track it.
        "running_error": running,
        "frame_percentile_ranks": ranks,
        "source_frame_indices": source_frame_indices,
        # Fraction of each original frame the model crop reconstructed
        # - a center crop of a wide source loses a band with no depth at all
        # - model-resolution metrics cannot see it: the model grid is the crop
        "crop_coverage": [
            {"index": k, "covered_fraction": float(max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / max(c[4] * c[5], 1e-9))}
            for k, c in enumerate(np.asarray(r.original_coords, dtype=np.float64))
        ],
        "notes": {
            "verdicts": "none by design — this describes distributions, it does not grade",
            "units": "scale-free or normalized throughout; 1 recon unit is NOT 1 meter",
            "attribution": "measurements differ in what they depend on; read them against each other",
            "source_frame_indices": "position = reconstruction index, value = source video frame index; "
            "null where the filename is not frame_{idx:06d}",
        },
    }
    # clean_for_json turns every nan into null. json.dumps otherwise writes a bare NaN, which no
    # strict JSON parser accepts; default= handles numpy scalars.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), indent=2, default=lambda o: o.item()))
    logger.info("Wrote %s (%d measurements available)", output_path, len(report["measurements_available"]))
    return report
