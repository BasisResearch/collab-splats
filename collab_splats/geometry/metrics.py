"""Reference-free scene error metrics: depth cross-view, photometric, and verify's epipolar rows.

Report-only. Nothing here feeds back into a reconstruction and nothing emits a verdict — the
output is distributions and how they vary, for a reader to interpret.

Every statistic comes from scipy or numpy. What lives here is the measurement those statistics
are computed over, not a reimplementation of them.
"""

import numpy as np
from scipy import stats

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
        return {
            "available": False,
            "reason": "no overlapping view pairs produced depth residuals",
            "grid": "model",
            "resolution": resolution,
        }

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
    # 0.9999999999999999, n=1 gives nan, neither raises), but this report makes no verdicts:
    # "n_pair_directions" ships right beside these numbers and the raw columns ship below
    # them, so a reader sees rho ~ 1.0 next to a count of 2 and discounts it. Nulling it here
    # would be this module deciding on the reader's behalf. scipy's own nan (a constant column,
    # or a non-finite row that slipped the producer) passes through the same way;
    # clean_for_json writes it as null. No nan_policy: the producers already drop non-finite
    # rows, and only nan_policy="omit" can raise.
    # error_vs_depth has null_hypothesis below to read against; positive rho is expected, and
    # near 0 or near 1 are the interesting outcomes.
    correlations = {
        "error_vs_depth": float(stats.spearmanr(depths, abs_rel_depth_error).statistic),
        "error_vs_frame_separation": float(stats.spearmanr(frame_seps, abs_rel_depth_error).statistic),
        "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual rises ~linearly in Z",
    }

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
        # Unconditional, and only publishable because "n_pair_directions" above and the raw
        # columns in "pair_directions" below ship with them: a reader can size the sample and
        # plot the shape behind each rho rather than taking it on trust.
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
    N = len(depth)
    ih, iw = images.shape[1:3]

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
        # Same check, same refusal, as the native-resolution mesh path (mesh/utils.py).
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))
        if (ih, iw) != expected_hw:
            raise ValueError(
                f"images are {(ih, iw)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )
        # Imported here, not at module top: mesh.utils imports pointcloud.feedforward.base,
        # which imports this module for bounded_residual/residual_bin_edges. A top-level
        # import would close that cycle and fail at load time. bundle_adjustment joins it
        # because it pulls in bae/vggt/pypose, which the depth path must not pay for.
        from collab_splats.geometry.bundle_adjustment import _scale_intrinsics_to_original
        from collab_splats.mesh.utils import guided_upsample_depth

        model_h, model_w = depth.shape[1:]
        # The guide is documented uint8 and guided_upsample_depth divides it by 255 internally.
        # FeedforwardResult.images is [0, 255] on VGGT-X but [0, 1] on MapAnything, so an
        # uncoerced guide is ~255x too flat on one backbone (measured: 0.398 max depth shift)
        # and float64 raises in OpenCV outright. That [0, 255] vs [0, 1] split is a property of
        # the BACKBONE, not of a frame, so the scale is decided ONCE off the whole array and
        # only applied per frame. Deciding it per frame lets a nearly-black frame in a [0, 255]
        # scene — a dark room, a tunnel, a lens-capped shot, every pixel under 1.0 — read as
        # [0, 1] and get amplified 255x: measured 117.78/255 mean absolute guide error on such
        # a frame, black turned near-white, with guided_upsample_depth guided by it.
        rgb_scale = 255.0 if images.max() <= 1.0 else 1.0
        lifted_d, lifted_K = [], []
        for k in range(N):
            tlx, tly, crx, cry = (float(v) for v in original_coords[k][:4])
            guide = np.clip(np.asarray(images[k]) * rgb_scale, 0, 255).astype(np.uint8)
            # rgb_full is the original-res canvas the crop came from — images[k] already is
            # that, so no re-read. crop_box is original_coords[:4], out_hw the canvas size.
            lifted_d.append(
                guided_upsample_depth(depth[k], guide,
                                      (int(tlx), int(tly), int(crx), int(cry)), (ih, iw))
            )
            # The CROP was resized to the model grid, so the scale is model/crop, not
            # model/canvas, and the crop origin comes back onto the principal point. Both are
            # _scale_intrinsics_to_model's own inverse rather than a second copy of it.
            sx, sy = model_w / (crx - tlx), model_h / (cry - tly)
            lifted_K.append(_scale_intrinsics_to_original(intrinsics[k], sx, sy, tlx, tly))
        depth, intrinsics = np.stack(lifted_d), np.stack(lifted_K)

    H, W = depth.shape[1:]
    # Derived, never declared: the grid the numbers below are actually measured on.
    resolution = f"{iw}x{ih}"
    cam2world = np.linalg.inv(extrinsics)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)
    ones = np.ones((H * W, 1))

    rows = []
    for i in range(N):
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

    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "zero-mean normalised cross-correlation; 1.0 = perfect agreement",
        # UNORDERED, like verify's epipolar block: one row per pair, not per direction. Also
        # the sample size for the rho below — it is not decoration, it is what makes the rho
        # readable.
        "n_pairs": len(rows),
        "correlations": correlations,
        "pairs": rows,
    }
