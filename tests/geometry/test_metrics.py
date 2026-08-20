"""Unit tests for reference-free scene error metrics."""

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import (
    _running_error,
    bounded_residual,
    build_report,
    compute_depth_error,
    compute_photometric_ncc,
    depth_error_in_pixels,
    residual_bin_edges,
)
from collab_splats.geometry.verification import PairStats, _distribution, clean_for_json
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import LEAF_STAGES, _STAGE_DEPS, _STAGE_ORDER


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


def test_scipy_supplies_the_correlation_directly():
    """The statistic IS scipy's, off the same columns that ship — no wrapper, no rewrite.

    The fixture is deliberately NOT monotone. A rho of exactly 1.0 is also what Pearson,
    Kendall and a hand-rolled rank difference all return, so a monotone fixture cannot tell
    which statistic actually ran; this one separates Spearman (0.83) from Pearson (0.89).
    """
    rels = [0.01, 0.05, 0.02, 0.08, 0.03, 0.09]
    depths = [2.0, 3.0, 1.0, 9.0, 4.0, 7.0]
    pairs = [_pair(k, k + 1, r, 3.0, depth=z) for k, (r, z) in enumerate(zip(rels, depths))]
    rho = compute_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_depth"]
    assert rho == stats.spearmanr(depths, [abs(r) for r in rels]).statistic
    # Anchors: a real intermediate rho, and one Pearson does NOT also produce.
    assert 0.0 < rho < 1.0
    assert rho != pytest.approx(float(np.corrcoef(depths, rels)[0, 1]))


def test_a_tiny_sample_ships_scipys_answer_NEXT_TO_the_count_that_qualifies_it():
    """Two rows cannot support a rho, and the block publishes scipy's answer anyway.

    Report-only means no verdicts, and "this sample is too small to correlate" is a verdict.
    scipy answers 0.9999999999999999 off two rows by construction and never raises (measured
    on 1.17.1); what makes that safe to publish is that the sample size ships in the same dict
    and the raw rows ship below it, so the reader discounts it rather than inheriting a
    judgement. The value and the count are ONE contract, so both are asserted here.
    """
    m = compute_depth_error(
        _collected([_pair(0, 1, 0.01, 3.0, depth=1.0), _pair(1, 2, 0.02, 3.0, depth=2.0)]), 500.0, "x"
    )
    assert m["correlations"]["error_vs_depth"] == stats.spearmanr([1.0, 2.0], [0.01, 0.02]).statistic
    assert m["correlations"]["error_vs_depth"] == pytest.approx(1.0)  # the spurious perfect fit
    assert m["n_pair_directions"] == 2 == len(m["pair_directions"])  # what makes it readable
    # One row is the same case at the other end: scipy returns nan, still without raising.
    one = compute_depth_error(_collected([_pair(0, 1, 0.01, 3.0, depth=1.0)]), 500.0, "x")
    assert np.isnan(one["correlations"]["error_vs_depth"]) and one["n_pair_directions"] == 1


def test_a_nan_row_makes_the_rho_nan_rather_than_correlating_a_SUBSET():
    """No silent row dropping: a missing value nans the whole rho, it does not shrink the sample.

    The producers guarantee finite columns, so this is unreachable today — it pins that if one
    ever regresses, the report says "cannot compute" instead of quietly reporting a rho over
    whichever rows happened to survive. Dropping rows here would answer -0.2 off the four
    co-finite rows; both nans sit on DIFFERENT rows, so a per-column drop would also desync
    the pairing and answer -0.1 off rows never measured together.
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
    assert np.isnan(m["correlations"]["error_vs_depth"])
    # Anchors: the two row-dropping answers this rejects are both finite and both wrong.
    assert stats.spearmanr([2.0, 3.0, 5.0, 6.0], [0.10, 0.01, 0.02, 0.09]).statistic == pytest.approx(-0.2)
    # frame_separation is never nan, so only the residual's nan reaches this column — still nan.
    assert np.isnan(m["correlations"]["error_vs_frame_separation"])


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
    """n identical views of a fronto-parallel white-noise plane at depth 4, identity poses.

    hw is a side length, or an (H, W) pair. A square frame cannot see an H/W swap in the
    bounds check — measured, `(ui < H) & (vi < W)` passed the whole square suite — so the
    non-square case is not decoration.
    """
    h, w = (hw, hw) if isinstance(hw, int) else hw
    rng = np.random.default_rng(seed)
    tex = rng.uniform(0, 255, size=(h, w, 3)).astype(np.float32)
    K = np.array([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1.0]], dtype=np.float32)
    return (
        np.stack([tex] * n),
        np.stack([np.full((h, w), 4.0, np.float32)] * n),
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
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_the_warp_actually_moves_pixels_through_pose_and_depth():
    """Identity poses score 1.0 even with no warp at all, so correctness needs a moving camera.

    Frame 1 is frame 0 displaced by exactly the 4 px this pose and depth predict. On white
    noise a correct warp reads 1.0 and a warp through the wrong pose reads ~0.
    """
    img, d, K, e = _translated_pair()
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.02)


def test_bounds_are_checked_against_the_right_axis_on_a_NON_SQUARE_frame():
    """W bounds u and H bounds v — on a square frame swapping them changes nothing.

    Measured: `(ui < H) & (vi < W)` passed all 51 tests, because every photometric fixture was
    square. On a 1920x1080 frame it IndexErrors at images[j][vi[ok], ui[ok]]. Here the frame is
    24 rows by 32 columns and the poses are identical, so EVERY pixel warps onto itself and
    lands in bounds — the swap silently drops the 8 rightmost columns instead of raising, so
    the pixel count is what discriminates, not availability.
    """
    img, d, K, e = _plane(hw=(24, 32))
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["n_pixels"] == 24 * 32
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_zero_depth_pixels_are_dropped_rather_than_warped_from_the_camera_centre():
    """depth == 0 means "no observation", not a surface at zero range.

    Unprojecting it puts the pixel at frame i's OWN camera centre, which projects to a real
    location in frame j and contributes a colour that pixel never saw. Identity poses hide
    this — the centre lands at z = 0 and `in_front` already drops it, which is why replacing
    the term with `in_front.copy()` passed the whole square-and-identity suite. Frame 1 is
    therefore pulled back along z, so frame 0's centre sits 2 units IN FRONT of it and the
    masked pixels would otherwise all pile onto its principal point.
    """
    img, d, K, e = _plane(n=2, hw=32)
    d = d.copy()
    d[0, :8, :] = 0.0  # 8 rows of frame 0 carry no observation
    e = e.copy()
    e[1, 2, 3] = 2.0  # camera 1 sits at world z = -2, looking the same way
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["n_pixels"] == 32 * 32 - 8 * 32


def test_ncc_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number.

    Frame 1 is noised so the correlation sits near 0.5 rather than at 1.0. At 1.0 both an
    un-normalised covariance and a raw difference read the same under either scaling — the
    invariance would be asserted against a case that cannot distinguish them.
    """
    rng = np.random.default_rng(11)
    img, d, K, e = _plane()
    img[1] = img[1] + rng.normal(0, 120, img[1].shape)
    a = compute_photometric_ncc(img, d, K, e, max_separation=1)["pairs"][0]
    b = compute_photometric_ncc(img / 255.0, d, K, e, max_separation=1)["pairs"][0]
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
    m = compute_photometric_ncc(shifted, d, K, e, max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_drops_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    img, d, K, e = _plane()
    noisy = img.copy()
    noisy[1] = noisy[1] + rng.normal(0, 90, noisy[1].shape)
    clean = compute_photometric_ncc(img, d, K, e, max_separation=1)["pairs"][0]
    dirty = compute_photometric_ncc(noisy, d, K, e, max_separation=1)["pairs"][0]
    assert dirty["photometric_ncc"] < clean["photometric_ncc"]


def test_flat_patch_is_skipped_not_a_divide_by_zero():
    """Every value equal means zero variance, and corrcoef would divide by it.

    `available is False` alone does NOT discriminate — measured, deleting the std guard leaves
    all 51 tests passing, because the isfinite check below it drops the same rows. What the
    guard buys is that corrcoef is never CALLED on a zero-variance patch: on a real scene with
    sky or a blank wall that is one RuntimeWarning per pair. simplefilter("error") is the
    assertion that pins it.
    """
    img, d, K, e = _plane()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_photometric_ncc(np.full_like(img, 128.0), d, K, e, max_separation=1)
    assert m["available"] is False


def test_two_overlapping_pixels_do_not_count_as_a_correlation():
    """np.corrcoef on 2 points returns exactly +-1 whatever the values — hence min_samples."""
    assert abs(np.corrcoef([0.0, 1.0], [5.0, -3.0])[0, 1]) == pytest.approx(1.0)
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, max_separation=1, min_samples=10**9)
    assert m["available"] is False


def test_min_samples_counts_pixels_not_the_ravelled_rgb_values():
    """The floor and the shipped `n_pixels` column are the same quantity, 3x smaller than the
    value count corrcoef sees — so a floor stated in values would gate at a third of the pixels.

    A 32x32 identity pair overlaps in exactly 1024 pixels and 3072 ravelled values. The floor
    admits it at 1024 and rejects it at 1025, which no value-count reading can produce.
    """
    img, d, K, e = _plane()
    assert compute_photometric_ncc(img, d, K, e, max_separation=1,
                                   min_samples=1024)["pairs"][0]["n_pixels"] == 1024
    assert compute_photometric_ncc(img, d, K, e, max_separation=1,
                                   min_samples=1025)["available"] is False


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert all(r["frame_separation"] <= 1 for r in m["pairs"])


def test_photometric_is_unavailable_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
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
    m = compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords,
                                max_separation=1)
    assert m["available"] is True and m["grid"] == "original"
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.02)


def test_the_K_lift_pins_the_Y_AXIS_TOO_on_a_NON_SQUARE_crop_with_Y_AND_Z_MOTION():
    """sy and tl_y only become load-bearing on a non-square crop with y AND z motion.

    Every other upsample fixture crops 32x32 onto a 16x16 grid, so sx == sy, and translates the
    camera along x alone. The y half of the lift is then unobservable IN PRINCIPLE rather than
    merely unobserved: with one shared K and no z motion the warp is
    v' = fy*(y_cam + t_y)/Z + cy = v + fy*t_y/Z, so t_y = 0 leaves v' = v for ANY fy and cy.
    Measured on the pre-existing suite: forcing sy onto the x axis, and dropping tl_y, each
    passed all 57 tests in this file.

    y motion alone still would not do it. cy has no term in v + fy*t_y/Z whatever the depth map
    holds, so it stays invisible until t_z != 0 turns the warp into a zoom about the principal
    point. Hence y AND z here.

    The crop is 32 wide by 16 tall onto a 16x16 grid, so sx = 0.5 and sy = 1.0 and the model K
    is anisotropic (fx 20, fy 40) exactly as a stretched crop must be; it lifts to fx = fy = 40,
    cx = 32, cy = 16. Frame 1 sits at t = (0, +0.2, -2), which halves the depth and so doubles
    the scale: u' = 2u - 32 and v' = 2v - 12, an exact integer map. Frame 0's crop is therefore
    a strided view of frame 1 and a correct warp scores exactly 1.0 on white noise. sy on the x
    axis lifts to fy = 80, cy = 24 and lands the warp 4 px out; tl_y = 0 lifts to cy = 8 and
    lands it 8 px out. Either collapses the NCC.

    The depth is a constant plane, unlike the two guide fixtures below. What is asserted here is
    the K arithmetic, and a constant depth is what keeps the warp an exact integer map and the
    1.0-vs-0 margin clean; a depth edge would zoom each half by a different factor and force
    resampling for no gain. The guided filter's use of its guide is pinned separately, by
    test_the_upsample_guide_is_normalised... and _scene_with_a_dark_frame, both of which do
    carry an edge because there the guide IS the subject.
    """
    rng = np.random.default_rng(7)
    img1 = rng.uniform(0, 255, size=(64, 64, 3)).astype(np.float32)
    # Only the crop carries depth, so only the crop carries content. Frame 0's crop is exactly
    # where every other pixel of frame 1 lands under the correct warp, so nothing resamples.
    img0 = np.zeros((64, 64, 3), np.float32)
    img0[8:24, 16:48] = img1[4:36:2, 0:64:2]
    # 16x16 model grid over a 32-wide, 16-tall crop at (16, 8) of a 64x64 canvas.
    model_d = np.stack([np.full((16, 16), 4.0, np.float32)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 40.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 24, 64, 64], dtype=np.float32), (2, 1))
    # z makes cy (hence tl_y) observable at all; y makes fy (hence sy) observable.
    e1 = np.eye(4, dtype=np.float32)
    e1[1, 3], e1[2, 3] = 0.2, -2.0
    e = np.stack([np.eye(4, dtype=np.float32), e1])

    m = compute_photometric_ncc(np.stack([img0, img1]), model_d, model_K, e,
                                original_coords=coords, max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.02)
    # Anchor: the whole crop warped in bounds, so that 1.0 is the full overlap rather than a
    # few surviving pixels that happened to agree.
    assert m["pairs"][0]["n_pixels"] == 16 * 32


def test_the_upsample_guide_is_normalised_whatever_the_backbones_image_scale(monkeypatch):
    """guided_upsample_depth documents a uint8 guide and divides it by 255 internally.

    FeedforwardResult.images is [0, 255] on VGGT-X and [0, 1] on MapAnything, so an uncoerced
    guide is ~255x too flat on one backbone — measured by the reviewer at 0.398 max / 0.013
    mean depth shift on depths of 1-5 — and a float64 guide raises in OpenCV outright. The two
    scales must therefore lift the SAME depth.

    The fixture depth carries an EDGE, not the constant plane every other upsample test uses:
    the guided filter only consults the guide where depth varies, so a constant map returns the
    same answer under any guide at all and could not see this.
    """
    from collab_splats.mesh import utils as mesh_utils

    real = mesh_utils.guided_upsample_depth
    lifted = []

    def spy(depth, rgb_full, *args, **kwargs):
        out = real(depth, rgb_full, *args, **kwargs)
        lifted.append(out)
        return out

    monkeypatch.setattr(mesh_utils, "guided_upsample_depth", spy)
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    model_d = np.stack([np.concatenate(
        [np.full((16, 8), 3.0, np.float32), np.full((16, 8), 5.0, np.float32)], axis=1)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))

    compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords, max_separation=1)
    compute_photometric_ncc(img / 255.0, model_d, model_K, e, original_coords=coords,
                            max_separation=1)
    assert len(lifted) == 4
    np.testing.assert_array_equal(lifted[0], lifted[2])
    np.testing.assert_array_equal(lifted[1], lifted[3])
    # Anchor: the guide really is load-bearing on this fixture, so the equality above is not
    # two runs of a filter that ignores its guide.
    flat_guide = real(model_d[0], np.zeros((64, 64, 3), np.uint8), (16, 8, 48, 40), (64, 64))
    assert not np.array_equal(lifted[0], flat_guide)


def _scene_with_a_dark_frame(dark_factor=0.0035, near=2.0, far=8.0):
    """Three views cut from one noise field, the middle one scaled to near-black.

    Frame 1's every pixel sits under 1.0 while the array as a whole peaks near 255 — a dark
    room, a tunnel, a lens-capped or blown frame in an otherwise [0, 255] scene. That is the
    only configuration in which a PER-FRAME scale decision disagrees with a per-ARRAY one.

    The model depth carries an EDGE (near/far half and half): the guided filter only consults
    its guide where depth varies, so the constant plane the other upsample fixtures use would
    return the same lift under any guide at all and could not see this. The cameras translate
    along x, so each frame's own lifted depth steers its warp and reaches the reported ncc.
    """
    rng = np.random.default_rng(3)
    big = rng.uniform(0, 255, size=(64, 64 + 8, 3)).astype(np.float32)
    images = np.stack([big[:, 0:64], big[:, 4:68] * dark_factor, big[:, 8:72]])
    model_d = np.stack([np.concatenate(
        [np.full((16, 8), near, np.float32), np.full((16, 8), far, np.float32)], axis=1)] * 3)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 3)
    ext = []
    for k in range(3):
        e = np.eye(4, dtype=np.float32)
        e[0, 3] = -k * 4 * 4.0 / 40.0
        ext.append(e)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (3, 1))
    return images, model_d, model_K, np.stack(ext), coords


def test_ncc_is_invariant_to_image_scale_convention_even_with_a_DARK_frame():
    """The report must not depend on which RGB convention the backbone happens to use.

    The scale split is [0, 255] (VGGT-X) vs [0, 1] (MapAnything) — a property of the BACKBONE,
    so one array must yield one interpretation. Deciding it per frame off `images[k].max()`
    breaks that: measured on this fixture, frame 1 (raw max 0.892) is read as [0, 1], amplified
    255x to a guide max of 228 in the [0, 255] run and clipped to a guide max of 0 in the
    [0, 1] run — black turned near-white in one run and left black in the other, with
    guided_upsample_depth steered by it. The two runs' ncc then diverge by 2.2e-2 on pair
    (1, 2), against 1.4e-9 when the scale is decided once for the whole array.

    Asserted on the shipped `ncc` rather than on the guide, because the contract is about the
    report, not about how the lift is spelled.
    """
    img, d, K, e, coords = _scene_with_a_dark_frame()
    # The fixture only has teeth if the dark frame really does trip a per-frame max() test
    # while the array does not. Without this the property below is vacuously true.
    assert img[1].max() < 1.0 < img.max()
    a = compute_photometric_ncc(img, d, K, e, original_coords=coords, max_separation=1)
    b = compute_photometric_ncc(img / 255.0, d, K, e, original_coords=coords, max_separation=1)
    assert [r["idx1"] for r in a["pairs"]] == [r["idx1"] for r in b["pairs"]] == [0, 1]
    for ra, rb in zip(a["pairs"], b["pairs"]):
        assert ra["photometric_ncc"] == pytest.approx(rb["photometric_ncc"], abs=1e-6)
    # Anchor: the depth edge really does reach the ncc, so the equality above is not two runs
    # of a warp that ignores the lifted depth. A perfect 1.0 would read the same either way.
    assert all(0.01 < r["photometric_ncc"] < 0.5 for r in a["pairs"])


def test_upsampling_without_crop_rows_is_a_refusal_not_a_guess():
    """Depth and K move together or neither does; guessing the crop is how they desync."""
    img, d, K, e = _plane(hw=64)
    with pytest.raises(ValueError, match="original_coords"):
        compute_photometric_ncc(img, d[:, ::2, ::2], K, e, max_separation=1)


def test_images_that_are_not_the_canvas_the_crops_were_cut_from_are_a_refusal():
    """The crop boxes are in ORIGINAL pixels, so a different-resolution image set misplaces
    every one of them. Same check and same refusal as the native-resolution mesh path.

    Without it the frames.zarr-vs-reconstruction mismatch is silent: the crop still indexes
    (it is in range on the smaller canvas) and simply cuts the wrong region of every frame.
    """
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    model_d = np.stack([np.full((16, 16), 4.0, np.float32)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    # original_coords claim a 128x128 canvas; the images are 64x64.
    coords = np.tile(np.array([16, 8, 48, 40, 128, 128], dtype=np.float32), (2, 1))
    with pytest.raises(ValueError, match="original_coords"):
        compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords,
                                max_separation=1)


def test_photometric_resolution_is_derived_from_the_images_not_declared():
    """It used to be a caller-supplied string, and a 32x32 fixture round-tripped "1920x1080".

    Non-square on purpose: "32x24" also pins the ORDER, which a square frame cannot see.
    """
    img, d, K, e = _plane(hw=(24, 32))
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["grid"] == "original" and m["resolution"] == "32x24"


def test_a_two_pair_rho_ships_NEXT_TO_the_n_pairs_that_qualifies_it():
    """scipy hands back +-1.0 off two points, and the block publishes it beside the count.

    Frame 1 is flat, which skips every pair touching it and leaves exactly two rows with
    DIFFERENT separations and DIFFERENT correlations — so neither column is constant and the
    +-1.0 is purely an artefact of the row count. That is precisely the case `n_pairs` exists
    for the reader to spot; withholding the value instead would be this module grading its own
    sample, which the report does not do.
    """
    rng = np.random.default_rng(7)
    img, d, K, e = _plane(n=4)
    img[1] = 128.0
    img[3] = img[3] + rng.normal(0, 90, img[3].shape)
    m = compute_photometric_ncc(img, d, K, e, max_separation=2)
    assert m["n_pairs"] == 2 == len(m["pairs"])  # the count that makes the rho readable
    assert [r["frame_separation"] for r in m["pairs"]] == [2, 1]
    assert m["pairs"][0]["photometric_ncc"] != pytest.approx(m["pairs"][1]["photometric_ncc"])
    assert m["correlations"]["ncc_vs_frame_separation"] == stats.spearmanr(
        [r["frame_separation"] for r in m["pairs"]], [r["photometric_ncc"] for r in m["pairs"]]
    ).statistic
    assert abs(m["correlations"]["ncc_vs_frame_separation"]) == pytest.approx(1.0)


def test_the_correlation_reads_the_shipped_pair_columns():
    """With enough rows it is computed, and off the same columns the reader can plot.

    Appearance drifts as a random WALK, not as independent per-frame noise: independent noise
    decorrelates every pair by the same amount whatever their separation (measured rho -0.12
    on that fixture), so it cannot anchor a falls-off-with-separation claim.
    """
    rng = np.random.default_rng(9)
    img, d, K, e = _plane(n=4)
    img = img + np.cumsum(rng.normal(0, 70, img.shape), axis=0)
    m = compute_photometric_ncc(img, d, K, e, max_separation=3)
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
    m = compute_photometric_ncc(img, d, K, e, max_separation=3)
    assert set(m) == {
        "available", "grid", "resolution", "units", "n_pairs", "correlations", "pairs",
    }
    assert m["available"] is True and m["grid"] == "original" and m["resolution"] == "32x32"
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
    m = compute_photometric_ncc(img, d, K, e, max_separation=3)
    banned = {"verdict", "status", "grade", "quality", "pass", "passed", "failed", "ok", "healthy"}
    assert banned.isdisjoint(set(m) | set(m["correlations"]) | set(m["pairs"][0]))
    strings = " ".join(v for v in m.values() if isinstance(v, str))
    assert not any(w in strings.lower() for w in ("good", "bad", "poor", "acceptable", "fail"))


########################################
# The stage and the running-error accumulator
########################################


def test_verify_writes_the_index_keys_so_no_merge_code_is_needed():
    """asdict() serialises whatever fields PairStats has — the shape lives at the source."""
    row = asdict(PairStats(3, 11, name1="a.png", name2="b.png", num_matches=500, num_inliers=450))
    assert row["idx1"] == 3 and row["idx2"] == 11
    assert abs(row["idx1"] - row["idx2"]) == 8
    assert row["num_inliers"] / row["num_matches"] == pytest.approx(0.9)


def test_report_is_a_leaf_stage_depending_only_on_pointcloud():
    assert "report" in LEAF_STAGES
    assert _STAGE_DEPS["report"] == ["pointcloud"]
    assert _STAGE_ORDER.index("report") > _STAGE_ORDER.index("pointcloud")


def test_report_does_not_demote_any_existing_leaf():
    """A new dependency edge would silently break another stage's disk re-run."""
    for s in ("refine", "semantics", "mesh", "localize", "verify"):
        assert s in LEAF_STAGES


def test_running_error_is_sequential_pairs_and_absolute_steps():
    """Signed steps cancel and hide accumulation; a separation-5 pair is a revisit not a step."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "median_rel_depth_error": 0.1},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "median_rel_depth_error": -0.1},
            {"idx1": 0, "idx2": 5, "frame_separation": 5, "median_rel_depth_error": 9.9}]
    out = _running_error(rows, "median_rel_depth_error")
    assert out["frame_index"] == [1, 2]  # separation-5 revisit excluded
    assert out["cumulative"] == pytest.approx([0.1, 0.2])  # |-0.1| added, not cancelled


def test_running_error_counts_an_ordered_pair_once():
    """The depth pass emits (i,j) AND (j,i); summing raw rows would double every step."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "median_rel_depth_error": 0.10},
            {"idx1": 1, "idx2": 0, "frame_separation": 1, "median_rel_depth_error": -0.20},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "median_rel_depth_error": 0.30},
            {"idx1": 2, "idx2": 1, "frame_separation": 1, "median_rel_depth_error": 0.30}]
    out = _running_error(rows, "median_rel_depth_error")
    # Two steps, not four, and each frame index appears once.
    assert out["frame_index"] == [1, 2]
    # Step 0->1 is mean(|0.10|, |-0.20|) = 0.15, NOT the signed mean (-0.05) and not the sum.
    assert out["cumulative"] == pytest.approx([0.15, 0.45])


def test_running_error_on_unordered_rows_is_the_identity():
    """Epipolar rows are one per unordered pair, so grouping must not alter them."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "rot_error_deg": 0.4},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "rot_error_deg": 0.6}]
    out = _running_error(rows, "rot_error_deg")
    assert out["frame_index"] == [1, 2]
    assert out["cumulative"] == pytest.approx([0.4, 1.0])


def test_running_error_drops_non_finite_and_missing_values():
    """A dead measurement leaves None or nan in the column; neither may enter the cumsum."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "rot_error_deg": 0.4},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "rot_error_deg": None},
            {"idx1": 2, "idx2": 3, "frame_separation": 1, "rot_error_deg": float("nan")},
            {"idx1": 3, "idx2": 4, "frame_separation": 1, "rot_error_deg": 0.6}]
    out = _running_error(rows, "rot_error_deg")
    assert out["frame_index"] == [1, 4]
    assert out["cumulative"] == pytest.approx([0.4, 1.0])


def test_report_json_is_valid_json_with_no_bare_nan():
    """json.dumps writes a bare NaN, which no strict parser accepts — clean_for_json prevents it."""
    text = json.dumps(clean_for_json({"rho": float("nan"), "nested": [float("nan"), 1.0]}))
    assert "NaN" not in text
    assert json.loads(text)["rho"] is None


########################################
# build_report end to end
########################################


def _write_tiny_scene(tmp_path, image_names, with_confidence=True):
    """A minimal feedforward.zarr that build_report can actually run on. Returns its path.

    Depth is a SLANTED plane, never a constant one: a constant-depth scene has a degenerate
    frustum AABB, compute_multiview_depth_confidence's pair gate then skips every pair, and
    the depth measurement would come back unavailable — a fixture that can observe nothing.
    Each frame carries a slightly different depth scale so the pairwise residuals are
    non-zero and the per-frame medians actually differ.

    The crop rows are OFF-CENTRE on a NON-SQUARE canvas so crop coverage is observable: an
    implementation that ignores the top-left origin reads 0.469 instead of 1/3, and a centred
    crop would hide exactly that.
    """
    # Heavy dep (pulls the vggt tree); imported here so the rest of this module stays light.
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    n, hw = len(image_names), 16
    rows = np.arange(hw, dtype=np.float32)[:, None]
    base = np.broadcast_to(3.0 + 0.1 * rows, (hw, hw)).astype(np.float32)
    depth = np.stack([base * (1.0 + 0.01 * k) for k in range(n)])
    K = np.array([[50.0, 0, hw / 2], [0, 50.0, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    extrinsics = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k in range(n):
        extrinsics[k][0, 3] = -0.15 * k  # camera centre slides along +x, so pairs have parallax
    rng = np.random.default_rng(4)
    coords = np.tile(np.array([4, 2, 20, 18, 32, 24], dtype=np.float32), (n, 1))
    result = FeedforwardResult(
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
        extrinsics=extrinsics,
        intrinsics=np.stack([K] * n),
        image_paths=[Path(p) for p in image_names],
        original_coords=coords,
        model_width=hw,
        model_height=hw,
        depth=depth,
        confidence=rng.uniform(0.5, 1.0, size=(n, hw, hw)).astype(np.float32) if with_confidence else None,
    )
    zarr_path = tmp_path / "feedforward.zarr"
    result.save_zarr(zarr_path)
    return zarr_path


def _build(tmp_path, image_names, with_confidence=True):
    """build_report over _write_tiny_scene with no verification.json and no frames.zarr."""
    return build_report(
        zarr_path=_write_tiny_scene(tmp_path, image_names, with_confidence),
        verification_json=tmp_path / "absent" / "verification.json",
        frames_zarr=tmp_path / "absent" / "frames.zarr",
        output_path=tmp_path / "report.json",
        backend="vggtx",
    )


def test_the_source_index_join_rests_on_the_frame_stem_naming_contract():
    """build_report derives its index map with this parser; a naming change must break loudly.

    frame_{idx:06d} is what FrameStore.export writes and what a reconstruction's image_paths
    carry. If that convention ever moves, every row of this report silently mispairs with
    every row of anything joined to it by source frame index — so the contract is pinned here
    rather than left to be discovered downstream.
    """
    assert FrameStore.frame_idx_from_path(Path("frame_000000.jpg")) == 0
    assert FrameStore.frame_idx_from_path(Path("/a/b/frame_002388.png")) == 2388
    # Zero padding is presentation only: the join key is the integer, not the string.
    assert FrameStore.frame_idx_from_path(Path("frame_000019.jpg")) == 19


def test_build_report_maps_recon_index_to_SOURCE_frame_index(tmp_path):
    """Every other per-frame block is keyed 0..N-1, which is NOT the source video index.

    The fixture's source indices are NON-CONTIGUOUS on purpose: sampling skips frames, so a
    map built as list(range(n)) or enumerate() would be wrong in production and completely
    invisible against 0, 1, 2. Without this key nothing keyed on the source video — a
    video-quality report, the frame store — can be joined to this one at all.
    """
    names = ["frame_000000.jpg", "frame_000007.jpg", "frame_000019.jpg"]
    report = _build(tmp_path, names)
    assert report["source_frame_indices"] == [0, 7, 19]
    # It is derived through the same parser, not re-implemented alongside it.
    assert report["source_frame_indices"] == [FrameStore.frame_idx_from_path(Path(p)) for p in names]
    # The join happens against the FILE, so the map has to survive serialisation.
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["source_frame_indices"] == [0, 7, 19]
    # One entry per reconstruction row, in reconstruction order.
    assert len(written["source_frame_indices"]) == written["scene"]["n_frames"] == 3


def test_an_off_contract_filename_yields_null_rather_than_a_guessed_index(tmp_path):
    """A guessed source index is worse than a missing one, so the contract is checked first.

    FrameStore.frame_idx_from_path is int(stem.split("_")[-1]): it raises only on a non-numeric
    tail, so IMG_1234 reads as 1234 and 00019 as 19 — plausible integers that are simply wrong.
    Those are the cases the map exists to prevent, because a downstream join then pairs real
    frames with the wrong rows and nothing looks broken. A null is visibly absent instead.

    The parser itself is deliberately NOT changed — other callers depend on its behaviour, and
    the two asserts below pin that it still guesses, so this guard is what stands between the
    guess and the report.
    """
    # The parser on its own would hand back a confident, wrong answer for all three of these.
    assert FrameStore.frame_idx_from_path(Path("IMG_1234.jpg")) == 1234
    assert FrameStore.frame_idx_from_path(Path("00019.jpg")) == 19
    assert FrameStore.frame_idx_from_path(Path("x_frame_000007.jpg")) == 7

    # Through build_report, all three come back null; the one on-contract name still resolves,
    # so the guard rejects by shape and is not just disabling the map wholesale. The prefixed
    # name is why the stem is matched WHOLE: a substring match would accept anything ending in
    # the right shape, which is the same guess this guard exists to refuse.
    # frame_1000000 is ON contract and must parse: 06d pads to six digits, it does not cap at
    # six, so a video past a million frames still names its keyframes by this rule. A guard
    # written as exactly-six would silently null every frame of such a scene.
    report = _build(tmp_path, ["IMG_1234.jpg", "00019.jpg", "x_frame_000007.jpg",
                               "frame_000007.jpg", "frame_1000000.jpg"])
    assert report["source_frame_indices"] == [None, None, None, 7, 1000000]
    # null, not the string "None" and not a dropped entry: one slot per reconstruction row.
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["source_frame_indices"] == [None, None, None, 7, 1000000]
    assert len(written["source_frame_indices"]) == written["scene"]["n_frames"] == 5


def test_a_measurement_that_cannot_run_disables_only_itself(tmp_path):
    """No verification.json and no frames.zarr: depth still ships, the other two say why not."""
    report = _build(tmp_path, ["frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg"])
    assert report["measurements_available"] == ["depth"]
    assert report["measurements"]["depth"]["available"] is True
    for dead in ("epipolar", "photometric"):
        assert report["measurements"][dead]["available"] is False
        assert report["measurements"][dead]["reason"]
    # A bare NaN is what json.dumps emits for a nan and no strict parser accepts it.
    assert "NaN" not in (tmp_path / "report.json").read_text()


def test_confidence_rho_ships_nested_with_the_n_frames_that_qualifies_it(tmp_path):
    """An unguarded rho is only publishable because its sample size cannot be read apart from it.

    n_frames is NOT recoverable from frame_percentile_ranks — ranks is {} at one frame while
    this rho's sample is len(per_frame) — so the count lives inside the same object.
    """
    report = _build(tmp_path, ["frame_000000.jpg", "frame_000007.jpg", "frame_000019.jpg"])
    block = report["confidence_vs_error"]
    assert set(block) == {"spearman", "n_frames"}
    assert block["n_frames"] == 3
    assert isinstance(block["spearman"], float)


def test_confidence_rho_and_its_count_are_both_none_without_a_confidence_array(tmp_path):
    """No array means the rho was never computed — distinct from a rho over a tiny sample."""
    report = _build(tmp_path, ["frame_000000.jpg", "frame_000007.jpg"], with_confidence=False)
    assert report["confidence_vs_error"] == {"spearman": None, "n_frames": None}


def test_crop_coverage_is_measured_against_the_ORIGINAL_canvas_not_the_model_grid(tmp_path):
    """The model grid IS the crop, so coverage is structurally invisible at model resolution.

    The fixture crops 16x16 out of a 32x24 canvas from an off-centre origin: 1/3 covered.
    Dropping the top-left origin gives 0.469 and using the model grid gives 1.0, so both
    mistakes are separated from the right answer.
    """
    report = _build(tmp_path, ["frame_000000.jpg", "frame_000007.jpg"])
    fractions = [c["covered_fraction"] for c in report["crop_coverage"]]
    assert fractions == pytest.approx([1.0 / 3.0, 1.0 / 3.0])
    assert [c["index"] for c in report["crop_coverage"]] == [0, 1]
