"""Reference-free scene error metrics: depth cross-view, photometric, and verify's epipolar rows.

Report-only. Nothing here feeds back into a reconstruction and nothing emits a verdict — the
output is distributions and how they vary, for a reader to interpret.

Every statistic comes from scipy or numpy. What lives here is the measurement those statistics
are computed over, not a reimplementation of them.
"""

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

from collab_splats.geometry.verification import clean_for_json
from collab_splats.preproc import frames

logger = logging.getLogger(__name__)

# The preprocess stage writes keyframes as frame_{idx:06d}. Matching that shape is the whole guard
# on the source-frame join: frames.frame_idx_from_path is int(stem.split("_")[-1]), which happily
# reads IMG_1234 as 1234 and 00019 as 19 — a confidently wrong index, which is worse here than a
# missing one, because a downstream join silently pairs real frames with the wrong rows. Six or
# more digits, so the contract does not break at a million frames.
_FRAME_STEM_RE = re.compile(r"frame_\d{6,}$")

########################################
# The residual histogram's axis
########################################


def residual_bin_edges(n_samples: int) -> np.ndarray:
    """Histogram edges for n_samples residuals. Nothing here is declared; both halves derive.

    Per-pixel depth residuals are N^2*H*W values (2.4e10 at 300 frames), far too many to hold,
    so they must accumulate into bins fixed BEFORE the loop starts. That rules out reading the
    bins off the data — a pre-pass to find the range would double the most expensive stage in
    the report. Bins need a range and a resolution, and both come from elsewhere:

      range       (-1, 1) by construction, because bounded_residual maps every possible
                  residual into it. No value can fall outside, so nothing is ever clipped.
      resolution  Rice's rule, k = 2 * n**(1/3) — the standard bin count for a sample of size
                  n. numpy implements it as np.histogram_bin_edges(a, bins="rice"), but that
                  wants the array in memory, which is the one thing there is too much of.

    Rice couples the two quantities the right way round: more pixels justify finer bins.
    Measured on a heavy-tailed population shaped like the real baseline, median recovery error
    is 5.1% at 512 bins, 1.7% at 1024, 0.66% at 1560 (a 60-frame scene) and 0.08% at 4584 (300
    frames). Below roughly 20 frames the bins do go coarse — 278 bins and 17% median error on
    a 5-frame scene — but only the PIXEL-level distribution loses resolution there. Per-pair
    medians ship as raw columns and are unaffected.

    The bin count is always even, so the histogram folds to |r| by adding the two halves.
    """
    k = 2 * max(1, int(round(max(int(n_samples), 1) ** (1.0 / 3.0))))
    return np.linspace(-1.0, 1.0, k + 1)


def bounded_residual(rel):
    """Map a relative depth residual onto (-1, 1) so a fixed histogram can never miss it.

    r / (1 + |r|) is monotone over all of R, so quantiles survive the map exactly: the qth
    quantile of the transformed values inverts back to the qth quantile of the originals.
    Invert with u / (1 - |u|).

    This exists so the histogram needs no chosen range and no clipping. A clipped range would
    silently pile the tail into the end bins, and np.histogram drops out-of-range values
    outright — either one makes a later "what fraction is above X" query quietly wrong.
    """
    r = np.asarray(rel, dtype=np.float64)
    return r / (1.0 + np.abs(r))


########################################
# Putting depth error and pixel error on one axis
########################################


def depth_error_in_pixels(rel_residual: float, parallax_deg: float, focal_px: float) -> float | None:
    """How many pixels this pair's depth disagreement is EQUIVALENT to, at its own parallax.

    A derived quantity, not a measurement: nothing is tracked or matched in the image here.
    It converts a depth residual into the pixel units a photometric or epipolar measurement
    already reports, so the two can be divided. A measured pixel error is a different number.

    For a pair with perpendicular baseline B, disparity is d = f*B/Z, and a depth error dZ at
    depth Z moves the point in the image by f*B*dZ/Z^2. Substituting r = dZ/Z:

        delta_d = r * d,   d = f * alpha

    where alpha is the parallax angle in radians. Baseline cancels out of the relation; the
    focal reappears only to state the answer in pixels.

    Converting first and dividing second is the only fair way to compare a pixel error against
    a depth error. The 1/Z hiding inside d is exactly why distant pixels disagree less in
    pixel terms while disagreeing more in depth terms.

    Divide a measured pixel error by this at the call site — no second function needed:
      ~1   one underlying error, seen twice.
      >>1  pixels moved more than any depth error explains, so the excess is pose (pose error
           moves pixels while leaving depths mutually consistent) or appearance.
      <<1  depth disagrees more than pixels do, so the error lies along the ray where this
           pair's baseline cannot see it. Low observability, not necessarily bad depth.

    Returns None when the pair carries under one pixel of disparity, because then it cannot
    see depth at all. That floor is derived from the focal, not chosen: it is the same
    quantity the return value is built from.

    Returns:
        A non-finite rel_residual propagates rather than becoming None: nan returns nan and
        inf returns inf. None means "this pair has no parallax to see depth with"; a
        non-finite value means "no data" — a different condition. Callers must filter the
        two separately, e.g. with ``result is not None and np.isfinite(result)``.
    """
    disparity_px = np.deg2rad(parallax_deg) * focal_px
    if disparity_px < 1.0:
        return None
    return abs(rel_residual) * disparity_px


########################################
# Depth cross-view error
########################################


def compute_depth_error(collected: dict, focal_px: float, resolution: str) -> dict:
    """How much the views disagree about depth: scale bias, geometric noise, parallax.

    Evaluated at MODEL resolution on purpose. Depth values are identical under nearest
    upsampling, so evaluating at original resolution returns the same number — but it would
    sample a guided-FILTERED depth map, reporting less disagreement than the model produced.
    That improvement belongs to the smoother, not the model.

    Args:
        collected:  the dict compute_multiview_depth_confidence(collect=...) filled.
        focal_px:   mean focal in pixels, used only to state the residual in pixel units.
        resolution: "WxH" of the grid, stamped into the output for the reader.
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

    # Read the correlation columns back OFF the rows, never off `pairs` a second time: a
    # second copy of `abs(p.idx1 - p.idx2)` is a copy that no row assertion can reach, and
    # sign-flipping it there silently reverses the reversed-row separations.
    abs_rel_depth_error = np.array([abs(r["median_rel_depth_error"]) for r in rows], dtype=np.float64)
    depths = np.array([r["median_depth"] for r in rows], dtype=np.float64)
    frame_seps = np.array([r["frame_separation"] for r in rows], dtype=np.float64)
    under_1px = sum(1 for r in rows if r["depth_error_px"] is None)

    # Quantiles off a histogram of the BOUNDED residual, inverted back to real residuals.
    # bounded_residual is monotone, so the qth quantile of the transformed values is the
    # transform of the qth quantile — the inversion is exact, not an approximation.
    counts, edges = collected["rel_depth_error_counts"], collected["rel_depth_error_edges"]
    grid = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)

    def read_quantiles(hist_counts, hist_edges) -> dict:
        """The grid, read off one histogram and inverted with the inverse of bounded_residual."""
        rv = stats.rv_histogram((hist_counts, hist_edges))
        out = {}
        for q in grid:
            u = float(rv.ppf(q))
            out[str(q)] = u / (1.0 - abs(u))
        return out

    # Signed, then the same histogram folded to |r|. The edges are symmetric about zero and
    # the bin count is always even, so bin j and bin k-1-j share |u| and the fold is exact
    # rather than a re-binning. Signed quantiles answer "is there scale bias"; folded ones are
    # the quantity every prior |rel| measurement in this repo reports, so they compare.
    half = (len(edges) - 1) // 2
    quantiles = read_quantiles(counts, edges)
    abs_quantiles = read_quantiles(counts[half:] + counts[:half][::-1], edges[half:])

    # Two questions, one number each, straight from scipy — whatever it returns, unfiltered.
    # A small sample makes rho meaningless (measured on scipy 1.17.1: n=2 gives
    # 0.9999999999999999, n=1 gives nan, neither raises), which is publishable only because
    # "n_pair_directions" ships right beside these numbers: a reader sees rho ~ 1.0 next to a
    # count of 2 and discounts it. error_vs_depth has null_hypothesis below to read against;
    # positive rho is expected, and near 0 or near 1 are the interesting outcomes.
    correlations = {
        "error_vs_depth": float(stats.spearmanr(depths, abs_rel_depth_error).statistic),
        "error_vs_frame_separation": float(stats.spearmanr(frame_seps, abs_rel_depth_error).statistic),
        "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual rises ~linearly in Z",
    }

    logger.info(
        "Depth error: %d pair directions, %d residual samples over %d bins, "
        "median |rel| %.4f, in %.2fs",
        len(pairs), int(counts.sum()), len(counts), abs_quantiles.get("0.5", float("nan")),
        time.perf_counter() - t0,
    )

    return {
        "available": True,
        "grid": "model",
        "resolution": resolution,
        "units": "relative (dimensionless); parallax in degrees; pixel equivalent in px",
        # DIRECTIONS, not pairs — which is why every key here says so. The mv loop is ordered:
        # (i,j) and (j,i) are separate rows with genuinely different values, because occlusion
        # is asymmetric — a pixel hidden looking one way is visible looking the other. The
        # photometric measurement and verify's epipolar block both count UNORDERED pairs under
        # the key "n_pairs", and a reader comparing the three would otherwise see a phantom 2x.
        "n_pair_directions": len(pairs),
        # The one pre-binned output, because it is the one per-pixel quantity. Counts plus
        # edges keeps threshold queries exact: rv_histogram(...).cdf(bounded_residual(x))
        # answers "what fraction of pixels fall below x" at any x.
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


def compute_photometric_ncc(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray | None = None,
    max_separation: int = 2,
    min_samples: int = 32,
) -> dict:
    """Warp each frame into its neighbours through pose+depth and correlate the RGB.

    The measure is zero-mean normalised cross-correlation: 1.0 is perfect agreement, 0.0 is
    none. np.corrcoef supplies it — NCC of two flattened patches IS their Pearson
    correlation, so there is nothing to write.

    Normalising buys two invariances a raw difference lacks: the [0, 255] (VGGT family) vs
    [0, 1] (MapAnything) image-scale split, so one number compares across backbones; and
    exposure or gain change, which would otherwise swamp the geometry being measured.

    This is the only measurement that reads appearance, so disagreement it sees that the depth
    and epipolar columns do not points at image formation rather than geometry.

    Runs at ORIGINAL resolution on purpose: RGB detail exists only there, and unlike depth
    this is a genuinely resolution-dependent quantity. When depth arrives on the smaller model
    grid it is upsampled here rather than in a separate wrapper.

    Rows are UNORDERED pairs, one per (i, j) with i < j, matching verification.py's epipolar
    block — hence "n_pairs"/"pairs" rather than the depth block's "pair_directions".

    The reported "resolution" is derived from `images` rather than passed in: a free-text
    argument can disagree with the grid the numbers were actually measured on.

    Args:
        images:          (N, H, W, 3) RGB, original resolution.
        depth:           (N, h, w) Z-depth on the model grid, or on the image grid already.
        intrinsics:      (N, 3, 3) K matching `depth`'s grid; rescaled here if depth is.
        extrinsics:      (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows, required only when depth needs upsampling.
        max_separation:  pairs per frame. Appearance agreement between distant frames is
                         dominated by lighting and viewpoint change, not by the error measured
                         here, so this stays O(N*max_separation) rather than O(N^2).
        min_samples:     floor on overlapping PIXELS, which is the quantity `n_pixels` ships.
                         RGB is ravelled before correlating, so np.corrcoef actually sees
                         3x this many values. A floor is needed either way — corrcoef on two
                         values returns exactly +-1 whatever they are.
    """
    t0 = time.perf_counter()
    N = len(depth)
    ih, iw = images.shape[1:3]
    logger.info("Photometric NCC: %d frames at %dx%d, max_separation=%d", N, iw, ih, max_separation)

    # Depth on the model grid, images on the original grid: lift depth and its K to match.
    # Pairing one grid's depth with the other grid's K is the 2026-08-11 mesh-collapse bug
    # class, so both move together or neither does.
    if depth.shape[1:] != (ih, iw):
        if original_coords is None:
            raise ValueError(
                f"depth is {depth.shape[1:]} but images are {(ih, iw)}; "
                "original_coords is required to upsample"
            )
        # The crop boxes are in ORIGINAL pixels, so `images` has to be the canvas they were
        # computed against; a same-count set at another resolution misplaces every crop.
        # Same check, same refusal, as the native-resolution mesh path (mesh/io.py).
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))
        if (ih, iw) != expected_hw:
            raise ValueError(
                f"images are {(ih, iw)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )
        # Imported here, not at module top: collab_splats.mesh reaches Warp and meshoptimizer
        # through texture.py, and collab_splats.geometry's own __init__ pulls bae/vggt/pypose.
        # A depth metric must not pay either import cost.
        from collab_splats.geometry.bundle_adjustment import _scale_intrinsics_to_original
        from collab_splats.mesh.io import upsample_depths

        model_h, model_w = depth.shape[1:]

        # The guide is documented uint8 and upsample_depths divides it by 255 internally.
        # FeedforwardResult.images is [0, 255] on VGGT-X but [0, 1] on MapAnything, so an
        # uncoerced guide is ~255x too flat on one backbone (measured: 0.398 max depth shift)
        # and float64 raises in OpenCV outright. That [0, 255] vs [0, 1] split is a property of
        # the BACKBONE, not of a frame, so the scale is decided ONCE off the whole array. Deciding
        # it per frame lets a nearly-black frame in a [0, 255] scene — a dark room, a tunnel, a
        # lens-capped shot, every pixel under 1.0 — read as [0, 1] and get amplified 255x:
        # measured 117.78/255 mean absolute guide error on such a frame, black turned near-white.
        rgb_scale = 255.0 if images.max() <= 1.0 else 1.0
        guides = np.clip(np.asarray(images) * rgb_scale, 0, 255).astype(np.uint8)
        lifted_d = upsample_depths(depth, guides, original_coords[:, :4])

        # The CROP was resized to the model grid, so the scale is model/crop, not model/canvas,
        # and the crop origin comes back onto the principal point. The K arithmetic that undoes
        # both is the forward's inverse, shared via _scale_intrinsics_to_original; the scale
        # itself is re-derived here because the forward's principal-point guard returns sx = 1.0
        # on the model-res K we pass.
        lifted_K = []
        for k in range(N):
            tlx, tly, crx, cry = (float(v) for v in original_coords[k][:4])
            sx, sy = model_w / (crx - tlx), model_h / (cry - tly)
            lifted_K.append(_scale_intrinsics_to_original(intrinsics[k], sx, sy, tlx, tly))
        depth, intrinsics = lifted_d, np.stack(lifted_K)

    H, W = depth.shape[1:]
    # Derived, never declared: the grid the numbers below are actually measured on.
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
        # Unproject frame i's pixels to world through its own K and pose. Local names follow
        # the multiview loop in pointcloud/feedforward/base.py (cam2world, pts_world,
        # pts_cam_j, proj_j, in_front) so the two warps read as the same operation.
        pts_cam_i = (np.linalg.inv(intrinsics[i]) @ pix.T).T * depth[i].reshape(-1, 1)
        pts_world = (cam2world[i] @ np.concatenate([pts_cam_i, ones], axis=-1).T).T[:, :3]
        # Homogeneous once per i: the j loop re-projects the SAME world points.
        pts_world_h = np.concatenate([pts_world, ones], axis=-1)

        for j in range(i + 1, min(N, i + max_separation + 1)):
            # Project them into frame j and look up the colour that landed there. No occlusion
            # test, unlike the depth loop in pointcloud/feedforward/base.py: that one has
            # frame j's own depth map to compare against, and here there is nothing to test a
            # hidden pixel against. Occluded pixels stay in and read as disagreement.
            pts_cam_j = (extrinsics[j] @ pts_world_h.T).T[:, :3]
            proj_j = (intrinsics[j] @ pts_cam_j.T).T
            z = np.clip(proj_j[:, 2], 1e-6, None)
            # Nearest sampling, matching the depth pass: bilinear across a depth discontinuity
            # blends two surfaces into a colour present on neither.
            ui = np.round(proj_j[:, 0] / z).astype(np.int64)
            vi = np.round(proj_j[:, 1] / z).astype(np.int64)
            in_front = pts_cam_j[:, 2] > 0
            # depth == 0 is "no observation", not a surface 0 away: unprojecting it puts the
            # pixel at frame i's own camera centre, which can land somewhere real in frame j
            # and contribute a colour that pixel never saw. in_front does not cover it —
            # the centre is only behind frame j for some poses.
            ok = in_front & (depth[i].ravel() > 0)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < min_samples:
                continue
            a = images[i].reshape(-1, 3)[ok].ravel().astype(np.float64)
            b = images[j][vi[ok], ui[ok]].ravel().astype(np.float64)
            # A flat patch has no variance to correlate; corrcoef returns nan, which is
            # dropped rather than counted as agreement. The isfinite check below catches the
            # same rows, but only AFTER corrcoef has divided by zero — on a real scene with
            # sky or a blank wall that is one RuntimeWarning per pair.
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            ncc = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(ncc):
                continue
            rows.append({"idx1": i, "idx2": j, "frame_separation": j - i,
                         "photometric_ncc": ncc, "n_pixels": int(ok.sum())})

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

    # Straight from scipy, unfiltered, exactly as the depth block does it. A short scene or a
    # heavily skipped one reaches two rows easily and scipy answers +-1.0 there by
    # construction — which is safe to publish only because "n_pairs" below sits next to the
    # number and the raw rows sit under it. Suppressing it would be a verdict, and this report
    # does not make verdicts.
    correlations = {"ncc_vs_frame_separation": float(stats.spearmanr(frame_seps, ncc).statistic)}

    logger.info(
        "Photometric NCC: %d pairs correlated, median NCC %.4f, in %.2fs",
        len(rows), float(np.median(ncc)), time.perf_counter() - t0,
    )

    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "zero-mean normalised cross-correlation; 1.0 = perfect agreement",
        # UNORDERED, like verify's epipolar block: one row per pair, and the rho's sample size.
        "n_pairs": len(rows),
        "correlations": correlations,
        "pairs": rows,
    }


def extract_photometric(result, images_dir: Path, n: int) -> dict:
    """
    Extract the photometric channel from a scene: read the RGB, then correlate it.

    The whole channel behind one call — unlike epipolar, whose numbers verify already computed,
    this is where the photometric measurement actually happens, and the 0.6s read is a rounding
    error against the correlation that follows it.

    Never fatal. A report must not fail a reconstruction, so a missing image directory or any
    exception below disables this one measurement and leaves the other two standing.

    The correlation itself stays in compute_photometric_ncc, which takes plain arrays and is
    pinned by 31 tests that must not need a scene on disk to run.

    Args:
        result: the FeedforwardResult supplying depth, intrinsics, extrinsics and crop boxes.
        images_dir: the scene's images/ directory of original-resolution keyframes.
        n: reconstruction frame count; the read is capped at the first n frames.

    Returns:
        The photometric measurement block, or {"available": False, "reason": ...} when the
        images are missing or the measurement raises.
    """
    if not frames.frame_paths(images_dir):
        logger.info("Photometric NCC: unavailable — no frame images at %s", images_dir)
        return {"available": False, "grid": "original",
                "reason": f"no frame images at {images_dir}"}
    try:
        # read_frames returns the selected frames in FILENAME order, which is the order the
        # reconstruction indexes by. The frame_idx encoded in each name is NOT that — it holds
        # source-video positions, so slicing by it would silently mispair depth with RGB.
        t0 = time.perf_counter()
        rgbs = frames.read_frames(images_dir)[:n].astype(np.float32)
        m = len(rgbs)
        logger.info("Photometric NCC: read %d original-resolution frames from %s in %.2fs",
                    m, images_dir, time.perf_counter() - t0)
        # No resolution argument: compute_photometric_ncc derives it from `rgbs` itself, so the
        # stamped grid cannot disagree with the grid the numbers were measured on.
        return compute_photometric_ncc(
            rgbs, result.depth[:m], result.intrinsics[:m], result.extrinsics[:m],
            original_coords=result.original_coords[:m],
        )
    except Exception as exc:  # noqa: BLE001 — a report must never fail a reconstruction
        logger.warning("photometric measurement failed: %s", exc, exc_info=True)
        return {"available": False, "grid": "original", "reason": f"{type(exc).__name__}: {exc}"}


########################################
# Stage entry point
########################################


def _running_error(rows: list[dict], key: str) -> dict:
    """Cumulative |step| along the trajectory, one step per consecutive-frame pair.

    Sequential pairs only: a separation-5 pair is a revisit, not a step, and summing it would
    count the same ground twice. Absolute values, because signed steps cancel and would hide
    the accumulation this exists to show.

    Rows are grouped by UNORDERED pair before summing. The depth pass is an ordered loop, so
    it emits both (k, k+1) and (k+1, k) — separation 1 in both directions, with genuinely
    different values, because occlusion is asymmetric. Summing the raw rows would add every
    trajectory step twice and repeat every frame_index. Epipolar rows are already one per
    unordered pair, so their groups hold a single member and the mean is the identity: one
    expression serves both channels with no per-channel branch.

    The mean is over |value|, never over the signed value. +0.10 and -0.09 average to +0.005,
    which reads as agreement when the two directions in fact disagree.
    """
    grouped: dict[tuple[int, int], list[float]] = {}
    for x in rows:
        if x.get("frame_separation") != 1 or x.get(key) is None or not np.isfinite(x[key]):
            continue
        lo, hi = sorted((int(x["idx1"]), int(x["idx2"])))
        grouped.setdefault((lo, hi), []).append(abs(float(x[key])))

    steps = sorted(grouped.items())
    return {
        "frame_index": [hi for (_, hi), _ in steps],
        "cumulative": np.cumsum([float(np.mean(v)) for _, v in steps]).tolist(),
    }


def build_reconstruction_quality_report(zarr_path: Path, verification_json: Path,
                                        images_dir: Path, output_path: Path,
                                        backend: str) -> dict:
    """Run every measurement that can run and write reconstruction_quality_report.json.

    Never raises on a dead measurement.

    Measurements are attempted independently: a missing confidence array, an absent
    verification.json or an unreadable images/ directory each disable exactly one of them.

    Nothing here grades the scene, names a cause or flags a frame. Absolute thresholds that
    would justify a verdict are exactly what this stage exists to inform, so inventing them
    now would be a guess dressed as a finding.

    A function, not a class: the report is built once and written once. Nothing mutates it,
    queries it in memory or subclasses it, so a class would add a constructor, attributes and
    a serialiser with no behaviour behind them.
    """
    from collab_splats.pointcloud.feedforward.base import (
        FeedforwardResult,
        compute_multiview_depth_confidence,
    )

    r = FeedforwardResult.load_zarr(zarr_path)
    n = len(r.depth)
    model_res = f"{r.model_width}x{r.model_height}"
    focal_px = float(r.intrinsics[:, 0, 0].mean() + r.intrinsics[:, 1, 1].mean()) / 2.0
    logger.info("Report on %s: %d frames, model resolution %s", zarr_path, n, model_res)

    # One dense pass yields the depth residual, the scale split, the parallax angles and the
    # per-pair depth. abs_thresh stays 0.0: scale invariance holds only there, and that is
    # what lets one function serve backbones whose depth scales differ completely.
    collected: dict = {}
    compute_multiview_depth_confidence(
        r.depth, r.intrinsics, r.extrinsics, abs_thresh=0.0, rel_thresh=0.05, collect=collected
    )
    depth_m = compute_depth_error(collected, focal_px, model_res)

    # Epipolar: verify's tables, read off disk. NOT a measurement — verify made these rows and
    # the matcher is never re-run here. They are the only rows that never touch depth, which is
    # why attribution works at all: something that moves here but not in the depth rows is a
    # pose error. Already original-resolution, since verify estimates from original-resolution
    # keypoints.
    image_width = int(r.original_coords[0][4])
    verification_json = Path(verification_json)
    if not verification_json.exists():
        logger.info(
            "Epipolar: unavailable — no verification.json at %s "
            "(set pointcloud.geometric_verification: true, or run --stages verify)",
            verification_json,
        )
        epipolar_m = {"available": False, "grid": "original",
                      "reason": f"no verification.json at {verification_json} — set "
                      "pointcloud.geometric_verification: true or run --stages verify"}
    else:
        t0 = time.perf_counter()
        logger.info("Epipolar: loading verify tables from %s", verification_json)
        data = json.loads(verification_json.read_text())
        # inlier_ratio is the same expression verify already aggregates over at
        # verification.py:363 (`p.num_inliers / p.num_matches ... if p.num_matches`) — per row
        # here rather than collapsed to a distribution, so it joins against the depth rows.
        epi_pairs = []
        for s in data.get("pair_stats", []):
            n_m, n_i = s.get("num_matches") or 0, s.get("num_inliers") or 0
            epi_pairs.append({**s, "frame_separation": abs(s["idx1"] - s["idx2"]),
                              "inlier_ratio": (n_i / n_m) if n_m else None})
        # A bare pixel count is not comparable across backbones (a 518 crop against 448x592),
        # so the fraction of image width ships alongside it.
        epi_frames = []
        for name, fs in sorted(data.get("frame_stats", {}).items()):
            px = fs.get("mean_reproj_error_px")
            epi_frames.append({**fs, "name": name, "mean_reproj_error_frac_width":
                               None if px is None else px / image_width})
        logger.info("Epipolar: %d verified pairs, %d frame reprojection rows, in %.2fs",
                    len(epi_pairs), len(epi_frames), time.perf_counter() - t0)
        epipolar_m = {"available": True, "grid": "original",
                      "resolution": f"width={image_width}",
                      "units": "degrees; reprojection in px and as a fraction of image width",
                      "source": str(verification_json), "n_pairs": len(epi_pairs),
                      "pairs": epi_pairs, "frames": epi_frames}

    photometric_m = extract_photometric(r, images_dir, n)

    # Per-frame median |residual| — the column both the confidence check and the ranks read.
    # Scanning every pair per frame makes this O(N * pairs) = O(N^3), so it carries a bar.
    per_frame = {}
    for k in tqdm(range(n), desc="Per-frame medians", unit="frame", leave=False):
        v = [abs(p.median_rel_depth_error) for p in collected["pairs"] if k in (p.idx1, p.idx2)]
        if v:
            per_frame[k] = float(np.median(v))

    # Does the model know when it is wrong? Confidence is an INPUT being validated, not an
    # error source, so it gets one correlation rather than a measurement of its own. Absent on
    # older zarr stores, which are never backfilled. scipy unguarded, like the two above: no
    # small-sample floor, because withholding a rho is a verdict and this report makes none.
    # That is only defensible with the count nested WITH the rho, unreadable apart from it —
    # and `frame_percentile_ranks` cannot supply it, being {} at len(ks) <= 1 while this
    # sample is len(per_frame). Both stay None with no confidence array: never computed is
    # not the same as computed over a tiny sample.
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

    # Where each frame sits in this scene's own distribution, 0..1. A NUMBER, never a label.
    # Within-scene ranks need no absolute threshold, which sidesteps the fact that pixel and
    # depth units are not comparable across backbones.
    ks = list(per_frame)
    ranks = {}
    if len(ks) > 1:
        rk = (stats.rankdata([per_frame[k] for k in ks]) - 1) / (len(ks) - 1)
        ranks = {int(k): float(x) for k, x in zip(ks, rk)}

    # Does disagreement build along the trajectory?
    # The row key differs by measurement: depth ships ORDERED directions under
    # "pair_directions", epipolar ships unordered pairs under "pairs". _running_error groups by
    # unordered key either way, so only the lookup name changes.
    # A channel that never ran must say so. Without the availability guard a dead channel takes
    # the .get default and ships {"frame_index": [], "cumulative": []} — indistinguishable from a
    # channel that ran and accumulated nothing, which reads as "error stayed at zero". That is a
    # verdict, and a false one.
    running = {
        name: (
            _running_error(m.get(rows_key, []), key)
            if m.get("available")
            else {"available": False, "reason": m.get("reason", "measurement unavailable")}
        )
        for name, rows_key, key, m in (
            ("depth", "pair_directions", "median_rel_depth_error", depth_m),
            ("epipolar", "pairs", "rot_error_deg", epipolar_m),
        )
    }

    # Reconstruction index -> SOURCE video frame index. Every per-frame block above is keyed
    # 0..N-1, which is not the source index once sampling skips frames, so without this map
    # nothing keyed on the source video can be joined to this report at all. Derived from
    # image_paths rather than the images/ filenames because image_paths is always on the
    # result while the image directory is optional here. The stem must match the documented
    # contract BEFORE it is parsed — frames.frame_idx_from_path guesses on any numeric tail, and
    # a guessed index is worse than no index. A name off-contract yields None rather than killing
    # the report: the join degrades per frame instead of disappearing.
    source_frame_indices: list[int | None] = []
    for p in r.image_paths:
        m = _FRAME_STEM_RE.fullmatch(Path(str(p)).stem)
        source_frame_indices.append(frames.frame_idx_from_path(p) if m else None)

    report = {
        "scene": {"backend": backend, "n_frames": n, "model_resolution": model_res,
                  "zarr": str(zarr_path)},
        "measurements_available": sorted(
            k for k, m in (("epipolar", epipolar_m), ("depth", depth_m), ("photometric", photometric_m))
            if m.get("available")
        ),
        "measurements": {"epipolar": epipolar_m, "depth": depth_m, "photometric": photometric_m},
        "confidence_vs_error": {"spearman": conf_rho, "n_frames": conf_n},
        # Read running_error against error_vs_frame_separation before calling it drift: frame index
        # is a confounded axis, since scene content, motion speed and exposure all track it.
        "running_error": running,
        "frame_percentile_ranks": ranks,
        "source_frame_indices": source_frame_indices,
        # Fraction of each ORIGINAL frame the model crop actually reconstructed. VGGTX resizes
        # width to 518 and centre-crops height to 518, so a 16:9 source loses a band with no
        # depth at all — and model-resolution evaluation is structurally blind to it, because
        # the model grid IS the crop.
        "crop_coverage": [
            {"index": k, "covered_fraction": float(
                max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / max(c[4] * c[5], 1e-9))}
            for k, c in enumerate(np.asarray(r.original_coords, dtype=np.float64))
        ],
        "notes": {
            "verdicts": "none by design — this describes distributions, it does not grade",
            "units": "scale-free or normalised throughout; 1 recon unit is NOT 1 metre",
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
