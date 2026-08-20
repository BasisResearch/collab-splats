"""Unit tests for reference-free scene error metrics."""

import math

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import (
    bounded_residual,
    compute_depth_error,
    compute_photometric_ncc,
    depth_error_in_pixels,
    residual_bin_edges,
)
from collab_splats.geometry.verification import PairStats, _distribution


def test_pair_stats_is_keyed_on_frame_index():
    """Depth path has indices and no filenames; verify has filenames. Index is the join key."""
    p = PairStats(idx1=0, idx2=4)
    assert (p.idx1, p.idx2) == (0, 4)
    assert p.name1 is None and p.median_rel_depth_error is None


def test_separation_is_a_subtraction_not_a_field():
    """1->4 and 2->5 are both separation 3 — the distance-vs-error axis, derived not stored."""
    assert abs(PairStats(1, 4).idx1 - PairStats(1, 4).idx2) == 3
    assert abs(PairStats(2, 5).idx1 - PairStats(2, 5).idx2) == 3


def test_pair_stats_carries_epipolar_and_depth_together():
    """One row per pair. Two measurements fill different columns of it."""
    p = PairStats(0, 1, name1="f0.png", name2="f1.png", num_matches=500, num_inliers=450,
                  rot_error_deg=0.15, t_direction_error_deg=0.9, median_rel_depth_error=0.02)
    assert p.num_inliers == 450 and p.median_rel_depth_error == pytest.approx(0.02)


def test_distribution_skips_rows_that_did_not_fill_the_column():
    """The depth pass writes rows with the epipolar fields None; _distribution must survive them.

    _triangulate_and_summarize feeds p.rot_error_deg straight into np.asarray(..., float64),
    which coerces None to nan, and _distribution strips nan. That degradation is load-bearing
    now that a second pass emits rows leaving those columns empty — so it gets a test.
    """
    rows = [
        PairStats(0, 1, name1="f0.png", name2="f1.png", rot_error_deg=0.4),
        PairStats(1, 2, median_rel_depth_error=0.02, median_parallax_deg=3.0),  # depth-only
    ]
    d = _distribution([p.rot_error_deg for p in rows])
    assert d is not None and d["median"] == pytest.approx(0.4)
    assert d["p90"] == pytest.approx(0.4) and d["p99"] == pytest.approx(0.4)
    # A column no pass filled is absent, not a row of nan.
    assert _distribution([p.photometric_ncc for p in rows]) is None


def test_depth_error_in_pixels_is_r_times_disparity():
    """delta_d = r * d, and d = f * parallax(rad) for small angles."""
    assert depth_error_in_pixels(0.1, 2.0, 500.0) == pytest.approx(0.1 * np.deg2rad(2.0) * 500.0)


def test_depth_error_in_pixels_scales_linearly_in_r():
    assert depth_error_in_pixels(0.10, 3.0, 500.0) == pytest.approx(
        2 * depth_error_in_pixels(0.05, 3.0, 500.0)
    )


def test_depth_error_in_pixels_shrinks_with_parallax():
    """The whole far-pixel asymmetry: same depth error, less parallax, fewer pixels moved."""
    assert depth_error_in_pixels(0.1, 1.0, 500.0) < depth_error_in_pixels(0.1, 6.0, 500.0)


def test_depth_error_in_pixels_is_none_below_one_pixel_of_disparity():
    """The floor is derived from the focal, not chosen: 1 px of disparity is the limit."""
    f = 500.0
    just_under = np.rad2deg(0.9 / f)  # 0.9 px of disparity
    just_over = np.rad2deg(1.1 / f)
    assert depth_error_in_pixels(0.1, just_under, f) is None
    assert depth_error_in_pixels(0.1, just_over, f) is not None


def test_the_disparity_floor_moves_with_the_focal_length():
    """A longer lens resolves depth at a smaller angle — so the floor cannot be a constant."""
    angle = np.rad2deg(1.5 / 500.0)  # 1.5 px at f=500, but only 0.3 px at f=100
    assert depth_error_in_pixels(0.1, angle, 500.0) is not None
    assert depth_error_in_pixels(0.1, angle, 100.0) is None


def test_depth_error_in_pixels_uses_magnitude_not_sign():
    assert depth_error_in_pixels(-0.1, 2.0, 500.0) == pytest.approx(
        depth_error_in_pixels(0.1, 2.0, 500.0)
    )


def test_ratio_against_a_measured_pixel_error_needs_no_second_function():
    """rho is a division at the call site, not an API — measured / equivalent."""
    rel_residual, parallax_deg, focal_px = -0.1, 2.0, 500.0
    # Independent of depth_error_in_pixels: deg2rad(2.0) * 500 * |-0.1|, via math not numpy.
    expected_equiv = math.radians(parallax_deg) * focal_px * abs(rel_residual)
    measured_px = 8.0
    rho = measured_px / depth_error_in_pixels(rel_residual, parallax_deg, focal_px)
    assert rho == pytest.approx(measured_px / expected_equiv)


# Scene shapes as (frames, side), and the bin count Rice's rule gives each — measured, which
# is what a test asserts. The sample count here is the UNORDERED pair count, deliberately NOT
# production's expression: the mv loop is ordered and feeds the histogram N*(N-1)*H*W residuals.
# These numbers exist to put the bin count at a realistic magnitude, not to mirror production,
# so the difference is not a bug to "fix". Round-trip accuracy is a property of the BIN COUNT,
# not of array size, so the tests ask for a real scene's bin count and feed it a small array.
RICE_BINS = {(5, 518): 278, (60, 518): 1560, (300, 518): 4584}


def _n_samples(frames: int, side: int) -> int:
    return frames * (frames - 1) // 2 * side * side


def test_bounded_residual_is_monotone_and_never_leaves_the_bin_range():
    """No value can fall outside the histogram, so nothing is clipped and nothing is dropped."""
    edges = residual_bin_edges(_n_samples(60, 518))
    r = np.array([-1e6, -40.0, -0.3, 0.0, 0.3, 40.0, 1e6])
    u = bounded_residual(r)
    assert np.all(np.diff(u) > 0)
    assert u.min() > edges[0] and u.max() < edges[-1]


def test_bounded_residual_preserves_quantiles_through_the_histogram():
    """A monotone map commutes with quantiles — that is what makes the fixed range safe."""
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0, 0.03, 200_000), [12.0, -40.0]])
    edges = residual_bin_edges(_n_samples(60, 518))
    counts, _ = np.histogram(bounded_residual(r), bins=edges)
    assert counts.sum() == r.size  # nothing dropped, unlike a clipped fixed range
    u = stats.rv_histogram((counts, edges)).ppf(0.99)
    assert u / (1.0 - abs(u)) == pytest.approx(np.quantile(r, 0.99), abs=1e-4)


def test_bin_edges_are_derived_from_the_sample_count_not_declared():
    """Resolution is the only thing the bounded axis left undetermined, and n determines it."""
    assert len(residual_bin_edges(10**9)) > len(residual_bin_edges(10**6))
    # Rice's rule, k = 2 * n**(1/3), across the scene sizes the report runs at.
    for (frames, side), k in RICE_BINS.items():
        assert len(residual_bin_edges(_n_samples(frames, side))) - 1 == k


def test_bin_edges_always_span_the_whole_bounded_axis():
    """Whatever n is, the range is (-1, 1) by construction — only resolution moves."""
    for n in (1, 10**3, 10**11):
        e = residual_bin_edges(n)
        assert e[0] == -1.0 and e[-1] == 1.0 and len(e) % 2 == 1  # even bin count, so it folds


def test_bin_edges_clamp_zero_and_negative_sample_counts_to_the_two_bin_floor():
    """max(int(n_samples), 1) means an empty or malformed count still yields a valid axis."""
    for n in (0, -5, -10**6):
        e = residual_bin_edges(n)
        assert len(e) == 3  # k = 2 * max(1, round(1 ** (1/3))) = 2 -> 3 edges
        assert e[0] == -1.0 and e[-1] == 1.0


def test_folding_a_signed_histogram_recovers_absolute_quantiles():
    """The prior depth_disagreement.py numbers are |rel| — a signed histogram must fold first."""
    rng = np.random.default_rng(1)
    r = rng.standard_t(df=1.6, size=300_000) * 0.0055
    edges = residual_bin_edges(_n_samples(60, 518))
    counts, _ = np.histogram(bounded_residual(r), bins=edges)
    half = (len(edges) - 1) // 2
    folded, fedges = counts[half:] + counts[:half][::-1], edges[half:]
    u = float(stats.rv_histogram((folded, fedges)).ppf(0.9))
    assert u / (1.0 - abs(u)) == pytest.approx(np.quantile(np.abs(r), 0.9), rel=0.02)


def test_scipy_supplies_the_correlation_directly():
    """No wrapper: nan_policy drops pairs and verification.clean_for_json turns nan into null."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert stats.spearmanr(x, y, nan_policy="omit").statistic == pytest.approx(1.0)


########################################
# compute_depth_error
########################################


def _pair(i, j, rel, par, n=100, iqr=0.01, depth=4.0):
    return PairStats(i, j, n_pixels=n, median_rel_depth_error=rel, iqr_rel_depth_error=iqr,
                     median_parallax_deg=par, median_depth=depth)


def _collected(pairs):
    """The collect-dict shape compute_multiview_depth_confidence fills.

    Edges are sized for a real 60-frame scene, not for these few hundred fixture pixels:
    quantile recovery is a property of the bin count, so a fixture that derived its own
    coarse bins would be testing a resolution nothing ships at.
    """
    edges = residual_bin_edges(_n_samples(60, 518))
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for p in pairs:
        counts += np.histogram(
            bounded_residual(np.full(p.n_pixels, p.median_rel_depth_error)), bins=edges
        )[0]
    return {"pairs": pairs, "rel_depth_error_counts": counts, "rel_depth_error_edges": edges}


def test_depth_error_reports_grid_and_resolution():
    """Every block stamps its grid — model-res depth with original-res K is a known bug class."""
    m = compute_depth_error(_collected([_pair(0, 1, 0.0, 3.0)]), 500.0, "518x518")
    assert m["grid"] == "model" and m["resolution"] == "518x518"


def test_pair_rows_carry_separation():
    """4->1 and 2->5 both land at 3, so distance-vs-error is a column not a special case.

    One fixture is REVERSED on purpose. The producer's loop is ordered (base.py sets
    idx1=i, idx2=j for both directions), so rows with idx1 > idx2 genuinely ship. With two
    same-direction fixtures, idx2 - idx1 would also give [3, 3] — and that expression sends
    every reversed row negative, sign-flipping error_vs_frame_separation.
    """
    m = compute_depth_error(_collected([_pair(4, 1, 0.02, 3.0), _pair(2, 5, 0.03, 3.0)]), 500.0, "x")
    assert [r["frame_separation"] for r in m["pair_directions"]] == [3, 3]


def test_scale_bias_keeps_its_sign_on_the_row():
    """A pure scale error has a large median and a small spread; the sign must survive."""
    m = compute_depth_error(_collected([_pair(0, 1, -0.08, 3.0, iqr=0.005)]), 500.0, "x")
    assert m["pair_directions"][0]["median_rel_depth_error"] == pytest.approx(-0.08)
    assert m["pair_directions"][0]["iqr_rel_depth_error"] == pytest.approx(0.005)


def test_pixel_equivalent_lands_on_each_pair_row():
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, 2.0)]), 500.0, "x")
    assert m["pair_directions"][0]["depth_error_px"] == pytest.approx(
        depth_error_in_pixels(0.1, 2.0, 500.0)
    )


def test_pairs_under_one_pixel_of_disparity_report_null_not_zero():
    tiny = np.rad2deg(0.5 / 500.0)  # half a pixel of disparity
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, tiny)]), 500.0, "x")
    assert m["pair_directions"][0]["depth_error_px"] is None
    assert m["pair_directions_under_one_pixel_disparity"] == 1


def test_per_pair_columns_ship_raw():
    """Raw: an exact value round-trips onto the row, so no rounding or binning happened here.

    Key presence alone does not test "raw" — round(x, 2) on every column survives it. The
    fixture value has more digits than any plausible rounding would keep.
    """
    m = compute_depth_error(_collected([_pair(0, 1, 0.0123456789, 3.0, iqr=0.0098765432)]), 500.0, "x")
    row = m["pair_directions"][0]
    assert row["median_rel_depth_error"] == 0.0123456789  # exact, not approx
    assert row["iqr_rel_depth_error"] == 0.0098765432
    assert row["median_parallax_deg"] == 3.0 and row["median_depth"] == 4.0


def test_output_keys_are_the_contract_task_6_reads():
    """The report writer indexes these by name — a renamed or dropped key breaks it silently."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0) for k in range(4)]
    m = compute_depth_error(_collected(pairs), 500.0, "518x518")
    assert set(m) == {
        "available", "grid", "resolution", "units", "n_pair_directions",
        "residual_histogram", "pair_directions_under_one_pixel_disparity",
        "correlations", "pair_directions",
    }
    assert m["available"] is True
    # Ordered directions, so this is len(pairs) and NOT the unordered pair count.
    assert m["n_pair_directions"] == len(pairs) == len(m["pair_directions"])
    assert set(m["correlations"]) == {"error_vs_depth", "error_vs_frame_separation", "null_hypothesis"}
    h = m["residual_histogram"]
    assert set(h) == {"counts", "bin_edges", "total", "quantiles", "abs_quantiles", "axis"}
    # total is the histogram's own mass, not a row count: 4 pairs x 100 pixels each.
    assert h["total"] == int(np.asarray(h["counts"]).sum()) == sum(p.n_pixels for p in pairs)
    assert set(m["pair_directions"][0]) == {
        "idx1", "idx2", "frame_separation", "n_pixels", "median_rel_depth_error",
        "iqr_rel_depth_error", "median_parallax_deg", "median_depth", "depth_error_px",
    }


def test_per_pixel_residual_ships_as_counts_and_edges():
    """The one quantity too large to hold — so any threshold query stays exact.

    0.3, not 0.1: the bounded axis and the residual axis only diverge far from zero. At 0.1
    the un-inverted bin value is 0.0904, inside abs=0.01 of 0.1, so the assertion could not
    see whether the inversion ran at all. At 0.3 it reads 0.2301 un-inverted against 0.2989
    inverted — a 7x margin. Do not lower it back.
    """
    m = compute_depth_error(_collected([_pair(0, 1, 0.3, 3.0)]), 500.0, "x")
    h = m["residual_histogram"]
    assert len(h["bin_edges"]) == len(h["counts"]) + 1 and h["total"] > 0
    assert h["quantiles"]["0.5"] == pytest.approx(0.3, abs=0.01)  # inverted back to a residual


def test_signed_and_folded_quantiles_both_ship():
    """Every prior |rel| number in this repo is absolute, so the signed axis alone is not
    comparable — a negative bias reads as a negative quantile until the histogram is folded.

    -0.3 for the same reason as the test above: at -0.1 the un-inverted bin value passes.
    """
    h = compute_depth_error(_collected([_pair(0, 1, -0.3, 3.0)]), 500.0, "x")["residual_histogram"]
    assert h["quantiles"]["0.5"] == pytest.approx(-0.3, abs=0.01)  # sign kept: scale bias
    assert h["abs_quantiles"]["0.5"] == pytest.approx(0.3, abs=0.01)  # folded: magnitude


def test_rising_residual_with_depth_shows_as_a_positive_correlation():
    """One number replaces the depth-strata routine — the raw columns are in the JSON."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    assert m["correlations"]["error_vs_depth"] > 0.9


def test_the_correlation_reads_residual_MAGNITUDE_not_signed_residual():
    """A growing NEGATIVE bias is growing disagreement — dropping abs() would call it shrinking.

    The signed column is deliberately kept on the row (scale bias reads off its sign), so the
    correlation must take the magnitude itself. Every other correlation fixture uses positive
    rel, where signed and absolute agree and the abs() is invisible.
    """
    pairs = [_pair(k, k + 1, -0.01 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    assert m["correlations"]["error_vs_depth"] > 0.9  # magnitude rises with depth
    # The rows themselves stay signed, so the sign is recoverable and only the rho folds.
    assert all(r["median_rel_depth_error"] < 0 for r in m["pair_directions"])


def test_too_few_usable_rows_gives_nan_rather_than_raising():
    """scipy's nan_policy="omit" RAISES under 3 surviving pairs; the block must survive a stub scene.

    Both halves matter: a nan in one column with only 2 rows is the raising case, and 2 clean
    rows is the "cannot be computed" case that must also read nan rather than a spurious rho
    of 1.0 off two points.
    """
    with_nan = compute_depth_error(
        _collected([_pair(0, 1, 0.01, 3.0), _pair(1, 2, 0.02, 3.0, depth=float("nan"))]), 500.0, "x"
    )
    assert np.isnan(with_nan["correlations"]["error_vs_depth"])
    two_clean = compute_depth_error(
        _collected([_pair(0, 1, 0.01, 3.0, depth=1.0), _pair(1, 2, 0.02, 3.0, depth=2.0)]), 500.0, "x"
    )
    assert np.isnan(two_clean["correlations"]["error_vs_depth"])


def test_the_finite_mask_is_PAIRWISE_so_a_nan_drops_the_whole_row():
    """Masking each column by its own np.isfinite desyncs the rows and returns a wrong rho.

    Nothing else pins this. Every other correlation fixture is nan-free, so the two masks
    coincide; the one nan fixture above has 2 rows and short-circuits before the mask matters.
    Here the nans sit on DIFFERENT rows and in different columns, so independent masks still
    hand scipy two equal-length arrays — no crash, just a rho over rows that were never
    measured together (-0.1 instead of -0.2 on this fixture).
    """
    pairs = [
        _pair(0, 1, float("nan"), 3.0, depth=1.0),  # residual missing
        _pair(1, 3, -0.10, 3.0, depth=2.0),
        _pair(2, 5, 0.01, 3.0, depth=3.0),
        _pair(3, 7, 0.07, 3.0, depth=float("nan")),  # depth missing, a DIFFERENT row
        _pair(4, 9, 0.02, 3.0, depth=5.0),
        _pair(5, 11, 0.09, 3.0, depth=6.0),
    ]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    # The four rows where BOTH columns are finite, magnitudes as compute_depth_error takes them.
    co_finite = stats.spearmanr([2.0, 3.0, 5.0, 6.0], [0.10, 0.01, 0.02, 0.09]).statistic
    assert m["correlations"]["error_vs_depth"] == pytest.approx(co_finite)
    assert co_finite == pytest.approx(-0.2)  # anchor: the fixture is not accidentally symmetric
    # Same rule on the other correlation: frame_separation is never nan, so the residual's nan
    # alone decides which rows survive.
    assert m["correlations"]["error_vs_frame_separation"] == pytest.approx(
        stats.spearmanr([2.0, 3.0, 4.0, 5.0, 6.0], [0.10, 0.01, 0.07, 0.02, 0.09]).statistic
    )


def test_nothing_in_the_output_grades_the_scene():
    """Report-only: distributions and how they vary, never a verdict for the reader to inherit."""
    pairs = [_pair(k, k + 1, 0.5 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]  # awful scene
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    banned = {"verdict", "status", "grade", "quality", "pass", "passed", "failed", "ok", "healthy"}
    assert banned.isdisjoint(set(m) | set(m["correlations"]) | set(m["pair_directions"][0]))
    # The shipped strings describe units and hypotheses; none of them announce an outcome.
    strings = " ".join(v for v in m.values() if isinstance(v, str))
    strings += " " + m["correlations"]["null_hypothesis"] + " " + m["residual_histogram"]["axis"]
    assert not any(w in strings.lower() for w in ("good", "bad", "poor", "acceptable", "fail"))


def test_constant_depth_gives_nan_which_the_json_writer_turns_into_null():
    """scipy's answer, unwrapped — verification.clean_for_json does the nan -> null pass."""
    pairs = [_pair(k, k + 1, 0.01, 3.0, depth=4.0) for k in range(10)]
    assert np.isnan(compute_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_depth"])


def test_error_vs_frame_separation_is_reported():
    """Does disagreement grow with how far apart the two frames are?"""
    pairs = [_pair(0, k, 0.005 * k, 3.0) for k in range(1, 20)]
    assert compute_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_frame_separation"] > 0.9


def test_depth_error_is_unavailable_not_a_crash_when_empty():
    m = compute_depth_error(_collected([]), 500.0, "x")
    assert m["available"] is False and "reason" in m


########################################
# compute_photometric_ncc
########################################


def _plane(n=2, hw=32, seed=0):
    """n identical views of a fronto-parallel white-noise plane at depth 4, identity poses."""
    rng = np.random.default_rng(seed)
    tex = rng.uniform(0, 255, size=(hw, hw, 3)).astype(np.float32)
    K = np.array([[40.0, 0, hw / 2], [0, 40.0, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    return (
        np.stack([tex] * n),
        np.stack([np.full((hw, hw), 4.0, np.float32)] * n),
        np.stack([K] * n),
        np.stack([np.eye(4, dtype=np.float32)] * n),
    )


def _translated_pair(shift_px=4, hw=32, f=40.0, depth=4.0, seed=5):
    """Two views of a plane, the second placed so the warp is EXACTLY shift_px to the left.

    Both frames are cut from one wider noise field, so the overlap is an exact pixel
    correspondence: no wrap-around and no resampling. A correct warp therefore scores 1.0 and
    any other offset scores ~0 on white noise, which is the margin the fixture exists for.
    """
    rng = np.random.default_rng(seed)
    big = rng.uniform(0, 255, size=(hw, hw + shift_px, 3)).astype(np.float32)
    images = np.stack([big[:, :hw], big[:, shift_px:]])
    K = np.stack([np.array([[f, 0, hw / 2], [0, f, hw / 2], [0, 0, 1.0]], dtype=np.float32)] * 2)
    # Camera 1 sits at world x = shift_px * Z / f, so u' = u - shift_px for every plane pixel.
    e1 = np.eye(4, dtype=np.float32)
    e1[0, 3] = -shift_px * depth / f
    return (
        images,
        np.stack([np.full((hw, hw), depth, np.float32)] * 2),
        K,
        np.stack([np.eye(4, dtype=np.float32), e1]),
    )


def test_identical_poses_and_depth_warp_to_ncc_one():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, "32x32", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_the_warp_actually_moves_pixels_through_pose_and_depth():
    """Identity poses score 1.0 even with no warp at all, so correctness needs a moving camera.

    Frame 1 is frame 0 displaced by exactly the 4 px this pose and depth predict. On white
    noise a correct warp reads 1.0 and a warp through the wrong pose reads ~0.
    """
    img, d, K, e = _translated_pair()
    m = compute_photometric_ncc(img, d, K, e, "32x32", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.02)


def test_ncc_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number.

    Frame 1 is noised so the correlation sits near 0.5 rather than at 1.0. At 1.0 both an
    un-normalised covariance and a raw difference read the same under either scaling — the
    invariance would be asserted against a case that cannot distinguish them.
    """
    rng = np.random.default_rng(11)
    img, d, K, e = _plane()
    img[1] = img[1] + rng.normal(0, 120, img[1].shape)
    a = compute_photometric_ncc(img, d, K, e, "x", max_separation=1)["pairs"][0]
    b = compute_photometric_ncc(img / 255.0, d, K, e, "x", max_separation=1)["pairs"][0]
    assert a["photometric_ncc"] == pytest.approx(b["photometric_ncc"], abs=1e-4)
    assert 0.2 < a["photometric_ncc"] < 0.9  # anchor: not the trivial 1.0 case


def test_ncc_is_invariant_to_exposure_shift():
    """Otherwise a brightness change swamps the geometry this measurement exists for.

    The change is offset-DOMINATED (gain 0.15, offset 210) on purpose. A gain-only fixture is
    also invariant under plain cosine similarity, so it could not show whether the mean is
    being removed; this one reads ~0.88 without the zero-mean step.
    """
    img, d, K, e = _plane()
    shifted = img.copy()
    shifted[1] = shifted[1] * 0.15 + 210.0
    m = compute_photometric_ncc(shifted, d, K, e, "x", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_drops_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    img, d, K, e = _plane()
    noisy = img.copy()
    noisy[1] = noisy[1] + rng.normal(0, 90, noisy[1].shape)
    clean = compute_photometric_ncc(img, d, K, e, "x", max_separation=1)["pairs"][0]
    dirty = compute_photometric_ncc(noisy, d, K, e, "x", max_separation=1)["pairs"][0]
    assert dirty["photometric_ncc"] < clean["photometric_ncc"]


def test_flat_patch_is_skipped_not_a_divide_by_zero():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(np.full_like(img, 128.0), d, K, e, "x", max_separation=1)
    assert m["available"] is False


def test_two_overlapping_pixels_do_not_count_as_a_correlation():
    """np.corrcoef on 2 points returns exactly +-1 whatever the values — hence min_samples."""
    assert abs(np.corrcoef([0.0, 1.0], [5.0, -3.0])[0, 1]) == pytest.approx(1.0)
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, "x", max_separation=1, min_samples=10**9)
    assert m["available"] is False


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = compute_photometric_ncc(img, d, K, e, "16x16", max_separation=1)
    assert all(r["frame_separation"] <= 1 for r in m["pairs"])


def test_photometric_is_unavailable_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    m = compute_photometric_ncc(img, d, K, e, "16x16", max_separation=1)
    assert m["available"] is False and "reason" in m


def test_photometric_upsamples_model_res_depth_and_lifts_its_K_with_it():
    """One function, both grids — and the K must ride the SAME transform as the depth.

    The crop is a strict sub-region (32x32 taken from a 64x64 canvas at (16, 8)), so the model
    -> original scale is crop_w / model_w = 2 and NOT canvas_w / model_w = 4. Using the canvas
    width doubles the focal, the warp lands 8 px out instead of 4, and NCC collapses — the
    2026-08-11 mesh-collapse bug class, caught here rather than in a mesh.
    """
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    # Model-res depth and K describing the crop only: 16x16 grid over a 32x32 crop.
    model_d = np.stack([np.full((16, 16), 4.0, np.float32)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))
    m = compute_photometric_ncc(img, model_d, model_K, e, "64x64", original_coords=coords,
                                max_separation=1)
    assert m["available"] is True and m["grid"] == "original"
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.02)


def test_upsampling_without_crop_rows_is_a_refusal_not_a_guess():
    """Depth and K move together or neither does; guessing the crop is how they desync."""
    img, d, K, e = _plane(hw=64)
    with pytest.raises(ValueError, match="original_coords"):
        compute_photometric_ncc(img, d[:, ::2, ::2], K, e, "64x64", max_separation=1)


def test_photometric_grid_says_which_one_it_ran_on():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, "1920x1080", max_separation=1)
    assert m["grid"] == "original" and m["resolution"] == "1920x1080"


def test_the_correlation_is_nan_not_a_spurious_rho_off_two_pairs():
    """scipy hands back +-1.0 off two points; two pairs is not a trend, so it must read nan.

    Frame 1 is flat, which skips every pair touching it and leaves exactly two rows with
    DIFFERENT separations and DIFFERENT correlations — so neither column is constant and the
    nan can only come from the row count.
    """
    rng = np.random.default_rng(7)
    img, d, K, e = _plane(n=4)
    img[1] = 128.0
    img[3] = img[3] + rng.normal(0, 90, img[3].shape)
    m = compute_photometric_ncc(img, d, K, e, "x", max_separation=2)
    assert m["n_pairs"] == 2
    assert [r["frame_separation"] for r in m["pairs"]] == [2, 1]
    assert m["pairs"][0]["photometric_ncc"] != pytest.approx(m["pairs"][1]["photometric_ncc"])
    assert np.isnan(m["correlations"]["ncc_vs_frame_separation"])


def test_the_correlation_reads_the_shipped_pair_columns():
    """With enough rows it is computed, and off the same columns the reader can plot.

    Appearance drifts as a random WALK, not as independent per-frame noise: independent noise
    decorrelates every pair by the same amount whatever their separation (measured rho -0.12
    on that fixture), so it cannot anchor a falls-off-with-separation claim.
    """
    rng = np.random.default_rng(9)
    img, d, K, e = _plane(n=4)
    img = img + np.cumsum(rng.normal(0, 70, img.shape), axis=0)
    m = compute_photometric_ncc(img, d, K, e, "x", max_separation=3)
    rho = stats.spearmanr(
        [r["frame_separation"] for r in m["pairs"]], [r["photometric_ncc"] for r in m["pairs"]]
    ).statistic
    assert m["correlations"]["ncc_vs_frame_separation"] == pytest.approx(rho)
    assert rho < -0.5  # anchor: the fixture really does fall off with separation


def test_photometric_output_keys_are_the_contract_task_6_reads():
    """UNORDERED pairs, matching verification.py's epipolar block — n_pairs, not directions.

    The depth block ships "pair_directions" because its producer loop is ordered and emits
    both (i, j) and (j, i) with different values. This loop is `for j in range(i + 1, ...)`,
    one row per pair, so the two counts are not comparable and the key names must not suggest
    they are: a reader comparing them would otherwise see a phantom 2x.
    """
    img, d, K, e = _plane(n=4)
    m = compute_photometric_ncc(img, d, K, e, "64x48", max_separation=3)
    assert set(m) == {
        "available", "grid", "resolution", "units", "n_pairs", "correlations", "pairs",
    }
    assert m["available"] is True and m["grid"] == "original" and m["resolution"] == "64x48"
    assert set(m["correlations"]) == {"ncc_vs_frame_separation"}
    # C(4, 2) = 6 unordered pairs. An ordered loop over the same frames would ship 12.
    assert m["n_pairs"] == len(m["pairs"]) == 6
    assert [(r["idx1"], r["idx2"]) for r in m["pairs"]] == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    ]
    assert set(m["pairs"][0]) == {
        "idx1", "idx2", "frame_separation", "photometric_ncc", "n_pixels",
    }


def test_nothing_in_the_photometric_output_grades_the_scene():
    """Report-only, the same contract as the depth block: a number and its units, no verdict."""
    rng = np.random.default_rng(13)
    img, d, K, e = _plane(n=4)
    img[1:] = img[1:] + rng.normal(0, 200, img[1:].shape)  # an awful scene
    m = compute_photometric_ncc(img, d, K, e, "x", max_separation=3)
    banned = {"verdict", "status", "grade", "quality", "pass", "passed", "failed", "ok", "healthy"}
    assert banned.isdisjoint(set(m) | set(m["correlations"]) | set(m["pairs"][0]))
    strings = " ".join(v for v in m.values() if isinstance(v, str))
    assert not any(w in strings.lower() for w in ("good", "bad", "poor", "acceptable", "fail"))
