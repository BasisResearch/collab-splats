"""Unit tests for reference-free scene error metrics."""

import math

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import (
    bounded_residual,
    compute_depth_error,
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
