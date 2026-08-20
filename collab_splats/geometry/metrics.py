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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho over the finite rows of x and y; nan when fewer than 3 rows survive.

    The drop happens here rather than via scipy's nan_policy="omit" because that path RAISES
    ValueError("The input must have at least 3 entries!") once the drop leaves under three
    pairs — a crash on a short or partly-empty column. nan is the answer instead, which is
    also what scipy returns for a constant column, and what clean_for_json writes as null.
    """
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        return float("nan")
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


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
        # Two questions, one number each. nan means "cannot be computed" — a constant column,
        # or under three usable rows — and becomes null when the report is written. The columns
        # they read ship in "pair_directions", so a reader can plot the shape behind the rho.
        # error_vs_depth has the null_hypothesis below to read against; positive rho is
        # expected, and near 0 or near 1 are the interesting outcomes.
        "correlations": {
            "error_vs_depth": _spearman(depths, abs_rel_depth_error),
            "error_vs_frame_separation": _spearman(frame_seps, abs_rel_depth_error),
            "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual rises ~linearly in Z",
        },
        "pair_directions": rows,
    }
