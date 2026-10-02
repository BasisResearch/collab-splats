"""Unit tests for reference-free scene error metrics."""

import ast
import inspect
import textwrap
import traceback
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

from collab_splats.geometry import metrics
from collab_splats.geometry.metrics import (
    PairStats,
    bounded_residual,
    compute_depth_error,
    compute_photometric_ncc,
    residual_bin_edges,
)
from collab_splats.geometry.projection import depth_agreement, unproject
from collab_splats.geometry.transforms import (
    invert_poses,
    rescale_intrinsics,
    shift_intrinsics,
)
from collab_splats.pointcloud.feedforward import base as ff_base
from collab_splats.preproc import frames as fr
from collab_splats.reconstructor import LEAF_STAGES, STAGES


def test_pair_stats_is_keyed_on_frame_index():
    """The depth pass keys a pair on frame indices; every measurement starts unset."""
    p = PairStats(idx1=0, idx2=4)
    assert (p.idx1, p.idx2) == (0, 4)
    assert p.median_rel_depth_error is None


def test_separation_is_a_subtraction_not_a_field():
    """1->4 and 2->5 are both separation 3 — the distance-vs-error axis, derived not stored."""
    assert abs(PairStats(1, 4).idx1 - PairStats(1, 4).idx2) == 3
    assert abs(PairStats(2, 5).idx1 - PairStats(2, 5).idx2) == 3


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


def test_bounded_residual_on_a_tensor_is_bit_identical_to_numpy():
    """The GPU histogram's map must match the numpy map bit for bit, so float64 in torch too."""
    rng = np.random.default_rng(0)
    rel = rng.standard_normal(10_000).astype(np.float32) * 3
    u_np = bounded_residual(rel)
    u_t = bounded_residual(torch.as_tensor(rel))
    assert u_t.dtype == torch.float64
    np.testing.assert_array_equal(u_t.numpy(), u_np)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]
)
def test_histogram_counts_match_np_histogram_including_the_edges(device):
    """Bins exactly as np.histogram, both closed ends included; NaN parks in the overflow bin."""
    edges = residual_bin_edges(_n_samples(300, 518))
    rng = np.random.default_rng(1)
    u = bounded_residual(rng.standard_normal(200_000) * 2)

    # Every edge value, plus both ends, which np.histogram closes into the outer bins
    u = np.concatenate([u, edges, [-1.0, 1.0, 1.0]])

    # An inf depth gives rel = +-inf, which maps to NaN; np.histogram drops NaN
    with np.errstate(invalid="ignore"):
        u = np.concatenate([u, bounded_residual(np.array([np.inf, -np.inf])), [np.nan]])

    got = metrics._histogram_counts(torch.as_tensor(u, device=device), torch.as_tensor(edges, device=device))
    got = got.cpu().numpy()
    assert len(got) == len(edges)
    np.testing.assert_array_equal(got[:-1], np.histogram(u, bins=edges)[0])
    assert got[-1] == 3


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
    """The collect-dict shape metrics._collect_pairs fills.

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


DEPTH_COLUMNS = {
    "idx1", "idx2", "n_pixels", "median_rel_depth_error", "iqr_rel_depth_error",
    "median_parallax_deg", "median_depth",
}


def test_depth_pairs_are_the_seven_columns_one_entry_per_direction():
    """Columnar, one entry per ordered direction, in the collected order; no derived column."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0) for k in range(4)] + [_pair(4, 1, 0.02, 3.0)]
    depth_pairs, histogram = compute_depth_error(_collected(pairs))
    assert set(depth_pairs) == DEPTH_COLUMNS
    assert all(len(v) == len(pairs) for v in depth_pairs.values())
    # A reversed direction ships as written: the producer's loop is ordered
    assert depth_pairs["idx1"] == [p.idx1 for p in pairs]
    assert depth_pairs["idx2"] == [p.idx2 for p in pairs]
    assert set(histogram) == {"counts", "bin_edges"}


def test_scale_bias_keeps_its_sign_on_the_row():
    """A pure scale error has a large median and a small spread; the sign must survive."""
    depth_pairs, _ = compute_depth_error(_collected([_pair(0, 1, -0.08, 3.0, iqr=0.005)]))
    assert depth_pairs["median_rel_depth_error"][0] == pytest.approx(-0.08)
    assert depth_pairs["iqr_rel_depth_error"][0] == pytest.approx(0.005)


def test_per_pair_columns_ship_raw():
    """Raw: an exact value round-trips onto the row, so no rounding or binning happened here.

    Key presence alone does not test "raw" — round(x, 2) on every column survives it. The
    fixture value has more digits than any plausible rounding would keep.
    """
    depth_pairs, _ = compute_depth_error(
        _collected([_pair(0, 1, 0.0123456789, 3.0, iqr=0.0098765432)])
    )
    assert depth_pairs["median_rel_depth_error"][0] == 0.0123456789  # exact, not approx
    assert depth_pairs["iqr_rel_depth_error"][0] == 0.0098765432
    assert depth_pairs["median_parallax_deg"][0] == 3.0 and depth_pairs["median_depth"][0] == 4.0


def test_per_pixel_residual_ships_as_the_collected_counts_and_edges():
    """The one quantity too large to hold ships as the pass binned it, untouched."""
    collected = _collected([_pair(0, 1, 0.3, 3.0)])
    _, histogram = compute_depth_error(collected)
    assert histogram["counts"] == collected["rel_depth_error_counts"].tolist()
    assert histogram["bin_edges"] == collected["rel_depth_error_edges"].tolist()
    assert len(histogram["bin_edges"]) == len(histogram["counts"]) + 1


def test_depth_error_is_empty_columns_not_a_crash_when_no_pair():
    depth_pairs, histogram = compute_depth_error(_collected([]))
    assert set(depth_pairs) == DEPTH_COLUMNS
    assert all(v == [] for v in depth_pairs.values())
    assert sum(histogram["counts"]) == 0


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
    assert m["photometric_ncc"][0] == pytest.approx(1.0, abs=0.05)


def test_the_warp_actually_moves_pixels_through_pose_and_depth():
    """Identity poses score 1.0 even with no warp at all, so correctness needs a moving camera.

    Frame 1 is frame 0 displaced by exactly the 4 px this pose and depth predict. On white
    noise a correct warp reads 1.0 and a warp through the wrong pose reads ~0.
    """
    img, d, K, e = _translated_pair()
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["photometric_ncc"][0] == pytest.approx(1.0, abs=0.02)


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
    assert m["n_pixels"][0] == 24 * 32
    assert m["photometric_ncc"][0] == pytest.approx(1.0, abs=0.05)


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
    assert m["n_pixels"][0] == 32 * 32 - 8 * 32


def test_ncc_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number.

    Frame 1 is noised so the correlation sits near 0.5 rather than at 1.0. At 1.0 both an
    un-normalized covariance and a raw difference read the same under either scaling — the
    invariance would be asserted against a case that cannot distinguish them.
    """
    rng = np.random.default_rng(11)
    img, d, K, e = _plane()
    img[1] = img[1] + rng.normal(0, 120, img[1].shape)
    a = compute_photometric_ncc(img, d, K, e, max_separation=1)["photometric_ncc"][0]
    b = compute_photometric_ncc(img / 255.0, d, K, e, max_separation=1)["photometric_ncc"][0]
    assert a == pytest.approx(b, abs=1e-4)
    assert 0.2 < a < 0.9  # anchor: not the trivial 1.0 case


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
    assert m["photometric_ncc"][0] == pytest.approx(1.0, abs=0.05)


def test_ncc_drops_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    img, d, K, e = _plane()
    noisy = img.copy()
    noisy[1] = noisy[1] + rng.normal(0, 90, noisy[1].shape)
    clean = compute_photometric_ncc(img, d, K, e, max_separation=1)["photometric_ncc"][0]
    dirty = compute_photometric_ncc(noisy, d, K, e, max_separation=1)["photometric_ncc"][0]
    assert dirty < clean


def test_flat_patch_is_skipped_not_a_divide_by_zero():
    """Every value equal means zero variance, and corrcoef would divide by it.

    An empty table alone does NOT discriminate — measured, deleting the std guard leaves
    all 51 tests passing, because the isfinite check below it drops the same rows. What the
    guard buys is that corrcoef is never CALLED on a zero-variance patch: on a real scene with
    sky or a blank wall that is one RuntimeWarning per pair. simplefilter("error") is the
    assertion that pins it.
    """
    img, d, K, e = _plane()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_photometric_ncc(np.full_like(img, 128.0), d, K, e, max_separation=1)
    assert m["photometric_ncc"] == []


def test_two_overlapping_pixels_do_not_count_as_a_correlation():
    """np.corrcoef on 2 points returns exactly +-1 whatever the values — hence min_samples."""
    assert abs(np.corrcoef([0.0, 1.0], [5.0, -3.0])[0, 1]) == pytest.approx(1.0)
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, max_separation=1, min_samples=10**9)
    assert m["photometric_ncc"] == []


def test_min_samples_counts_pixels_not_the_ravelled_rgb_values():
    """The floor and the shipped `n_pixels` column are the same quantity, 3x smaller than the
    value count corrcoef sees — so a floor stated in values would gate at a third of the pixels.

    A 32x32 identity pair overlaps in exactly 1024 pixels and 3072 ravelled values. The floor
    admits it at 1024 and rejects it at 1025, which no value-count reading can produce.
    """
    img, d, K, e = _plane()
    assert compute_photometric_ncc(img, d, K, e, max_separation=1,
                                   min_samples=1024)["n_pixels"] == [1024]
    assert compute_photometric_ncc(img, d, K, e, max_separation=1,
                                   min_samples=1025)["n_pixels"] == []


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["idx1"] and all(abs(i - j) <= 1 for i, j in zip(m["idx1"], m["idx2"]))


def test_photometric_is_empty_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m == {"idx1": [], "idx2": [], "photometric_ncc": [], "n_pixels": []}


def test_photometric_upsamples_model_res_depth_onto_the_images_grid_K():
    """Model-grid depth is upsampled; K arrives on the images' grid, lifted as PointcloudResult does.

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
    K = rescale_intrinsics(model_K, (16, 16), (32, 32))
    K = shift_intrinsics(K, coords[:, :2])
    m = compute_photometric_ncc(img, model_d, K, e, original_coords=coords,
                                max_separation=1)
    assert m["photometric_ncc"][0] == pytest.approx(1.0, abs=0.02)


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
    test_the_upsample_guide_is_normalized... and _scene_with_a_dark_frame, both of which do
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
    K = rescale_intrinsics(model_K, (16, 16), (16, 32))
    K = shift_intrinsics(K, coords[:, :2])
    # z makes cy (hence tl_y) observable at all; y makes fy (hence sy) observable.
    e1 = np.eye(4, dtype=np.float32)
    e1[1, 3], e1[2, 3] = 0.2, -2.0
    e = np.stack([np.eye(4, dtype=np.float32), e1])

    m = compute_photometric_ncc(np.stack([img0, img1]), model_d, K, e,
                                original_coords=coords, max_separation=1)
    assert m["photometric_ncc"][0] == pytest.approx(1.0, abs=0.02)
    # Anchor: the whole crop warped in bounds, so that 1.0 is the full overlap rather than a
    # few surviving pixels that happened to agree.
    assert m["n_pixels"][0] == 16 * 32


def test_the_upsample_guide_is_normalized_whatever_the_backbones_image_scale(monkeypatch):
    """upsample_depths documents a uint8 guide and divides it by 255 internally.

    PointcloudResult.images is [0, 255] on VGGT-X and [0, 1] on MapAnything, so an uncoerced
    guide is ~255x too flat on one backbone — measured by the reviewer at 0.398 max / 0.013
    mean depth shift on depths of 1-5 — and a float64 guide raises in OpenCV outright. The two
    scales must therefore lift the SAME depth.

    The fixture depth carries an EDGE, not the constant plane every other upsample test uses:
    the guided filter only consults the guide where depth varies, so a constant map returns the
    same answer under any guide at all and could not see this.
    """
    real = metrics.upsample_depths
    lifted = []

    def spy(depths, rgbs, crop_boxes):
        out = real(depths, rgbs, crop_boxes)
        lifted.append(out)
        return out

    monkeypatch.setattr(metrics, "upsample_depths", spy)
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    model_d = np.stack([np.concatenate(
        [np.full((16, 8), 3.0, np.float32), np.full((16, 8), 5.0, np.float32)], axis=1)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))
    K = rescale_intrinsics(model_K, (16, 16), (32, 32))
    K = shift_intrinsics(K, coords[:, :2])

    compute_photometric_ncc(img, model_d, K, e, original_coords=coords, max_separation=1)
    compute_photometric_ncc(img / 255.0, model_d, K, e, original_coords=coords,
                            max_separation=1)
    # One call per compute, lifting only frame 0: frame 1 has no partner at max_separation=1
    assert len(lifted) == 2
    assert lifted[0].shape == (1, 64, 64) and lifted[1].shape == (1, 64, 64)
    np.testing.assert_array_equal(lifted[0], lifted[1])
    # Anchor: the guide really is load-bearing on this fixture, so the equality above is not
    # two runs of a filter that ignores its guide.
    flat_guide = real(model_d[:1], np.zeros((1, 64, 64, 3), np.uint8), coords[:1, :4])[0]
    assert not np.array_equal(lifted[0][0], flat_guide)


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
    K = rescale_intrinsics(model_K, (16, 16), (32, 32))
    K = shift_intrinsics(K, coords[:, :2])
    return images, model_d, K, np.stack(ext), coords


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
    assert a["idx1"] == b["idx1"] == [0, 1]
    for na, nb in zip(a["photometric_ncc"], b["photometric_ncc"]):
        assert na == pytest.approx(nb, abs=1e-6)
    # Anchor: the depth edge really does reach the ncc, so the equality above is not two runs
    # of a warp that ignores the lifted depth. A perfect 1.0 would read the same either way.
    assert all(0.01 < v < 0.5 for v in a["photometric_ncc"])


def test_upsampling_without_crop_rows_is_a_refusal_not_a_guess():
    """Model-grid depth needs crop rows to upsample; guessing the crop misplaces it."""
    img, d, K, e = _plane(hw=64)
    with pytest.raises(ValueError, match="original_coords"):
        compute_photometric_ncc(img, d[:, ::2, ::2], K, e, max_separation=1)


def test_images_that_are_not_the_canvas_the_crops_were_cut_from_are_a_refusal():
    """The crop boxes are in ORIGINAL pixels, so a different-resolution image set misplaces
    every one of them. Same check and same refusal as the native-resolution mesh path.

    Without it the images/-vs-reconstruction mismatch is silent: the crop still indexes
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


def test_photometric_output_is_four_columns_over_unordered_pairs():
    """UNORDERED pairs (i < j); one entry per pair per column.

    This loop is `for j in range(i + 1, ...)`, one row per pair, unlike the depth pass's
    ordered directions — so the two row counts are not comparable.
    """
    img, d, K, e = _plane(n=4)
    m = compute_photometric_ncc(img, d, K, e, max_separation=3)
    assert set(m) == {"idx1", "idx2", "photometric_ncc", "n_pixels"}
    # C(4, 2) = 6 unordered pairs. An ordered loop over the same frames would ship 12.
    assert all(len(v) == 6 for v in m.values())
    assert list(zip(m["idx1"], m["idx2"])) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


########################################
# _collect_pairs: the report's cross-view depth pass
########################################


def _two_view(scale_j=1.0):
    """Two cameras with a 0.2-unit sideways baseline viewing a constant-depth plane.

    scale_j multiplies frame 1's depth, injecting a known relative residual.
    """
    H = W = 16
    K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], dtype=np.float32)
    depth = np.stack([np.full((H, W), 4.0, np.float32), np.full((H, W), 4.0 * scale_j, np.float32)])
    extr = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    extr[1, 0, 3] = -0.2  # world-to-cam translation => camera 1 sits at x=+0.2
    return depth, np.stack([K, K]), extr


def _collect(depth, K, extr, rel_thresh=0.05):
    return metrics._collect_pairs(depth, K, extr, rel_thresh)[0]


def _row_01(out):
    """The ordered (0, 1) row — the pair whose residual population is hand-derivable below."""
    return next(r for r in out["pairs"] if (r.idx1, r.idx2) == (0, 1))


def _bin_of(out, rel_value):
    """Index of the histogram bin that a residual of rel_value would land in."""
    return int(np.searchsorted(out["rel_depth_error_edges"], bounded_residual(rel_value), side="right") - 1)


def test_collect_fills_index_keyed_rows_and_the_one_histogram():
    depth, K, extr = _two_view()
    out = _collect(depth, K, extr)
    assert out["rel_depth_error_counts"].sum() > 0
    assert (out["pairs"][0].idx1, out["pairs"][0].idx2) == (0, 1)
    # ORDERED pair count: (i, j) and (j, i) both feed the one histogram
    n, h, w = depth.shape
    assert np.array_equal(out["rel_depth_error_edges"], residual_bin_edges(n * (n - 1) * h * w))


def test_signed_residual_recovers_an_injected_depth_scale():
    """Frame 1 depth x1.1 => median relative residual ~ +0.1 on the 0->1 pair."""
    depth, K, extr = _two_view(scale_j=1.1)
    assert _row_01(_collect(depth, K, extr, rel_thresh=0.5)).median_rel_depth_error == pytest.approx(0.1, abs=0.02)


def test_parallax_angle_and_median_depth_match_geometry():
    """0.2 baseline at depth 4 => atan(0.2/4) ~ 2.86 deg at the principal ray."""
    depth, K, extr = _two_view()
    row = _row_01(_collect(depth, K, extr))
    assert row.median_rel_depth_error == pytest.approx(0.0, abs=1e-3)
    assert row.median_parallax_deg == pytest.approx(np.degrees(np.arctan(0.2 / 4.0)), abs=0.5)
    assert row.median_depth == pytest.approx(4.0, abs=0.2)


def test_occluded_pixels_are_excluded_from_the_residual():
    """Occlusion is absent evidence, not disagreement — it must not pollute the scale bias."""
    depth, K, extr = _two_view()
    depth[1, :, :8] = 0.5  # a near occluder covering half of frame 1
    assert _row_01(_collect(depth, K, extr)).median_rel_depth_error == pytest.approx(0.0, abs=1e-3)


def test_residual_is_scale_invariant():
    """Multiplying depth and translation by s must leave the relative residual unchanged."""
    depth, K, extr = _two_view(scale_j=1.1)
    a = _row_01(_collect(depth, K, extr, rel_thresh=0.5))
    s = 7.0
    extr_s = extr.copy()
    extr_s[:, :3, 3] *= s
    b = _row_01(_collect(depth * s, K, extr_s, rel_thresh=0.5))
    assert a.median_rel_depth_error == pytest.approx(b.median_rel_depth_error, abs=1e-4)
    assert a.median_parallax_deg == pytest.approx(b.median_parallax_deg, abs=1e-3)


def test_iqr_reports_the_spread_not_the_lower_half():
    """Frame 1 half at residual 0.0, half at +0.2: q25 0.0, median 0.1, q75 0.2, IQR 0.2.

    The 1-pixel shift drops column 0 off frame 1's left edge: 15 x 16 = 240 residuals.
    """
    depth, K, extr = _two_view()
    depth[1, 8:, :] = 4.8
    row = _row_01(_collect(depth, K, extr))
    assert row.n_pixels == 240
    assert row.median_rel_depth_error == pytest.approx(0.1, abs=1e-3)
    assert row.iqr_rel_depth_error == pytest.approx(0.2, abs=1e-3)


def test_invalid_sampled_depth_is_excluded_not_counted_as_minus_one():
    """A zero sampled depth means "no measurement", not "100% too shallow"."""
    depth, K, extr = _two_view()
    control = _row_01(_collect(depth, K, extr))
    assert control.n_pixels == 240

    # Frame-1 columns 0..4 are what source columns 1..5 sample: 5 x 16 = 80 pixels
    depth[1, :, :5] = 0.0
    out = _collect(depth, K, extr)
    assert _row_01(out).n_pixels == control.n_pixels - 80
    assert out["rel_depth_error_counts"][_bin_of(out, -1.0)] == 0


def test_near_zero_expected_depth_is_excluded_not_divided_through():
    """Co-located cameras: a near-zero-depth pixel projects in bounds, but its quotient is noise."""
    depth, K, extr = _two_view()
    extr[1] = np.eye(4, dtype=np.float32)
    control = _row_01(_collect(depth, K, extr))
    assert control.n_pixels == 16 * 16

    depth[0, 0, 0] = 1e-7
    out = _collect(depth, K, extr)
    assert _row_01(out).n_pixels == control.n_pixels - 1
    assert out["rel_depth_error_counts"][_bin_of(out, 4.0 / 1e-6)] == 0


def test_report_multiview_agreement_is_one_on_consistent_scene():
    """Every seen pixel agrees on a consistent plane; the report carries it per frame."""
    depth, K, extr = _two_view()
    tables = metrics.compute_reconstruction_quality(
        depth,
        K,
        K,
        extr,
        np.tile([0, 0, 16, 16, 16, 16], (2, 1)),
        ["frame_000000.png", "frame_000001.png"],
        None,
        None,
    )
    assert tables["frames"]["multiview_agreement"] == pytest.approx([1.0, 1.0])


def test_multiview_agreement_is_null_when_no_other_view_sees_the_frame():
    depth, K, extr = _two_view()
    extr[1, 0, 3] = -100.0  # camera 1 far off to the side: nothing projects in bounds
    assert metrics._collect_pairs(depth, K, extr, 0.05)[1] == [None, None]


########################################
# The stage wiring
########################################


def test_report_is_a_leaf_stage_depending_only_on_pointcloud():
    assert "reconstruction_quality_report" in LEAF_STAGES
    assert STAGES["reconstruction_quality_report"] == ("pointcloud",)
    assert list(STAGES).index("reconstruction_quality_report") > list(STAGES).index("pointcloud")


def test_report_does_not_demote_any_existing_leaf():
    """A new dependency edge would silently break another stage's disk re-run."""
    for s in ("refine", "semantics", "mesh", "localize"):
        assert s in LEAF_STAGES


########################################
# compute_reconstruction_quality end to end
########################################


def _write_tiny_scene(tmp_path, image_names, with_confidence=True):
    """A minimal pointcloud.zarr that compute_reconstruction_quality can run on. Returns its path.

    Depth is a SLANTED plane and each frame carries a slightly different depth scale so the pairwise residuals are
    non-zero and the per-frame medians actually differ.

    The crop rows are OFF-CENTRE on a NON-SQUARE canvas so crop coverage is observable: an
    implementation that ignores the top-left origin reads 0.469 instead of 1/3, and a centred
    crop would hide exactly that.
    """
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
    result = ff_base.PointcloudResult(
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
        extrinsics=extrinsics,
        intrinsics=None,
        model_intrinsics=np.stack([K] * n),
        image_paths=[Path(p) for p in image_names],
        original_coords=coords,
        model_width=hw,
        model_height=hw,
        depth=depth,
        confidence=rng.uniform(0.5, 1.0, size=(n, hw, hw)).astype(np.float32) if with_confidence else None,
    )
    zarr_path = tmp_path / "pointcloud.zarr"
    result.save_zarr(zarr_path)
    return zarr_path


def _quality(tmp_path, image_names, with_confidence=True, images=None):
    """The stage's data path over _write_tiny_scene: load, compute. Returns (tables, collected)."""
    ff = ff_base.PointcloudResult.load_zarr(_write_tiny_scene(tmp_path, image_names, with_confidence))
    tables = metrics.compute_reconstruction_quality(
        ff.depth,
        ff.model_intrinsics,
        ff.intrinsics,
        ff.extrinsics,
        ff.original_coords,
        [Path(str(p)).name for p in ff.image_paths],
        ff.confidence,
        images,
    )
    collected, _ = metrics._collect_pairs(ff.depth, ff.model_intrinsics, ff.extrinsics, rel_thresh=0.05)
    return tables, collected


def _column_lengths(table):
    return {len(v) for v in table.values()}


def test_the_source_index_join_rests_on_the_frame_stem_naming_contract():
    """compute_reconstruction_quality derives frame_idx with this parser; a naming change must break loudly.

    frame_{idx:06d} is what the images/ store writes and what a reconstruction's image_paths
    carry. If that convention ever moves, every row of this report silently mispairs with
    every row of anything joined to it by source frame index — so the contract is pinned here
    rather than left to be discovered downstream.
    """
    assert fr.frame_idx_from_path(Path("frame_000000.jpg")) == 0
    assert fr.frame_idx_from_path(Path("/a/b/frame_002388.png")) == 2388
    # Zero padding is presentation only: the join key is the integer, not the string.
    assert fr.frame_idx_from_path(Path("frame_000019.jpg")) == 19


def test_frame_idx_is_the_SOURCE_frame_index_not_the_row(tmp_path):
    """Rows are 0..N-1; frame_idx is the source video index, NON-CONTIGUOUS because sampling skips."""
    names = ["frame_000000.jpg", "frame_000007.jpg", "frame_000019.jpg"]
    tables, _ = _quality(tmp_path, names)
    assert tables["frames"]["frame_idx"] == [0, 7, 19]
    # It is derived through the same parser, not re-implemented alongside it.
    assert tables["frames"]["frame_idx"] == [fr.frame_idx_from_path(Path(p)) for p in names]


def test_an_off_contract_filename_yields_null_rather_than_a_guessed_index(tmp_path):
    """A guessed source index is worse than a missing one, so the contract is checked first.

    fr.frame_idx_from_path is int(stem.split("_")[-1]): IMG_1234 reads as 1234 and 00019 as
    19 — plausible integers that are simply wrong. The parser is deliberately NOT changed;
    the asserts below pin that it still guesses, so the stem guard is what stands between
    the guess and the report. frame_1000000 is ON contract: 06d pads, it does not cap.
    """
    assert fr.frame_idx_from_path(Path("IMG_1234.jpg")) == 1234
    assert fr.frame_idx_from_path(Path("00019.jpg")) == 19
    assert fr.frame_idx_from_path(Path("x_frame_000007.jpg")) == 7

    tables, _ = _quality(tmp_path, ["IMG_1234.jpg", "00019.jpg", "x_frame_000007.jpg",
                                    "frame_000007.jpg", "frame_1000000.jpg"])
    assert tables["frames"]["frame_idx"] == [None, None, None, 7, 1000000]


def test_every_table_is_columnar_with_equal_column_lengths(tmp_path):
    """A table is {column: [values]}; a ragged column means rows no longer line up."""
    tables, _ = _quality(tmp_path, ["frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg"])
    for key in ("frames", "depth_pairs"):
        assert len(_column_lengths(tables[key])) == 1, key


def test_frames_table_has_one_row_per_reconstruction_frame(tmp_path):
    tables, _ = _quality(tmp_path, ["frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg"])
    assert _column_lengths(tables["frames"]) == {3}
    assert all(v is not None for v in tables["frames"]["median_abs_rel_depth_error"])


def test_depth_pairs_follow_the_collected_pair_order(tmp_path):
    """Rows are the collected PairStats, one per direction, in the order the depth pass emitted."""
    tables, collected = _quality(tmp_path, ["frame_000000.jpg", "frame_000004.jpg", "frame_000008.jpg"])
    assert collected["pairs"]
    assert tables["depth_pairs"]["idx1"] == [p.idx1 for p in collected["pairs"]]
    assert tables["depth_pairs"]["idx2"] == [p.idx2 for p in collected["pairs"]]


def test_photometric_is_null_without_images(tmp_path):
    """No images/: depth still ships, the photometric table is null."""
    tables, _ = _quality(tmp_path, ["frame_000000.jpg", "frame_000004.jpg"])
    assert tables["photometric_pairs"] is None
    assert tables["depth_pairs"]["idx1"]


def test_frame_columns_keep_their_order(tmp_path):
    """Index, then crop coverage, then the depth and confidence summaries."""
    tables, _ = _quality(tmp_path, ["frame_000000.jpg", "frame_000004.jpg"])
    assert list(tables["frames"]) == [
        "frame_idx",
        "covered_fraction",
        "median_abs_rel_depth_error",
        "multiview_agreement",
        "confidence_median",
    ]


def _pair_loop_lines() -> range:
    """Source lines of _collect_pairs' inner `for start` loop over target-view batches."""
    src = textwrap.dedent(inspect.getsource(metrics._collect_pairs))
    first = metrics._collect_pairs.__code__.co_firstlineno
    loop = next(
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.For) and getattr(node.target, "id", "") == "start"
    )
    return range(first + loop.lineno - 1, first + loop.end_lineno)


def _prune_block_lines() -> range:
    """
    Source lines of _collect_pairs' `if min_pair_overlap > 0` pruning block.
    """
    src = textwrap.dedent(inspect.getsource(metrics._collect_pairs))
    first = metrics._collect_pairs.__code__.co_firstlineno
    block = next(
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.If) and "min_pair_overlap" in ast.unparse(node.test)
    )
    return range(first + block.lineno - 1, first + block.end_lineno)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("min_pair_overlap", [0.0, 0.01])
@pytest.mark.parametrize("target_batch", [1, 2, None])
def test_collect_pairs_syncs_the_gpu_twice_per_target_batch(monkeypatch, target_batch, min_pair_overlap):
    """
    Each target batch syncs only at its counts and its nonzero; pruning adds three per source frame.
    """
    monkeypatch.setattr(metrics, "get_device", lambda: "cuda")
    n, h, w = 4, 20, 24
    rng = np.random.default_rng(0)
    depth = (3 + rng.uniform(-0.5, 0.5, (n, h, w))).astype(np.float32)
    K = np.tile(np.array([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]]), (n, 1, 1))
    E = np.tile(np.eye(4), (n, 1, 1))
    E[:, 0, 3] = np.linspace(0, 0.3, n)

    # Warm up lazy CUDA init, which syncs once on first use
    metrics._collect_pairs(depth, K, E, 0.05, target_batch=target_batch)

    # Attribute every sync warning to its _collect_pairs line
    lines = []

    def record(message, *args, **kwargs):
        if "synchroniz" in str(message):
            frames = [f for f in traceback.extract_stack() if f.name == "_collect_pairs"]
            lines.append(frames[-1].lineno if frames else None)

    torch.cuda.set_sync_debug_mode("warn")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = record
            collected, _ = metrics._collect_pairs(
                depth, K, E, 0.05, target_batch=target_batch, min_pair_overlap=min_pair_overlap
            )
    finally:
        torch.cuda.set_sync_debug_mode(0)

    # None sizes the batch from VRAM, capped at the n - 1 other frames
    per_batch = min(target_batch or n - 1, n - 1)
    n_batches = n * -(-(n - 1) // per_batch)
    loop = _pair_loop_lines()
    assert len(collected["pairs"]) == n * (n - 1)
    assert sum(line in loop for line in lines) == 2 * n_batches

    # Pruning syncs three times per source frame: the has_source mask, the overlap list, the index upload
    assert sum(line in _prune_block_lines() for line in lines) == (3 * n if min_pair_overlap else 0)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))],
)
def test_collect_pairs_is_identical_whatever_the_target_batch(monkeypatch, device):
    """Chunking target views changes the kernel shapes, never a pair row or a histogram count."""
    monkeypatch.setattr(metrics, "get_device", lambda: device)
    depth, K, extr = _two_view(scale_j=1.03)
    depth = np.concatenate([depth, depth[:1] * 1.01, depth[:1] * 0.99, depth[1:] * 1.02])
    K = np.concatenate([K, K[:1], K[:1], K[:1]])
    extr = np.concatenate([extr, extr[1:], extr[1:], extr[1:]])
    extr[2, 1, 3] = 0.1
    extr[4, 0, 3] = 0.1

    # Frame 2 faces away: an empty target mid-batch at 3, last in a batch at 2
    extr[2, :3, :3] = np.diag([-1.0, 1.0, -1.0])

    # TF32 on, as a mapanything import leaves it: a batched matmul would round differently
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")

    try:
        runs = [metrics._collect_pairs(depth, K, extr, 0.05, target_batch=b) for b in (1, 2, 3)]
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(precision)

    # Frame 2 forms no pair; source 0's batches hold it between seen targets 1 and 3
    one, agree_one = runs[0]
    keys = {(p.idx1, p.idx2) for p in one["pairs"]}
    assert not [k for k in keys if 2 in k]
    assert {(0, 1), (0, 3), (0, 4)} <= keys
    assert all(p.n_pixels > 0 and np.isfinite(p.median_rel_depth_error) for p in one["pairs"])

    for many, agree_many in runs[1:]:
        assert agree_one == agree_many
        np.testing.assert_array_equal(one["rel_depth_error_counts"], many["rel_depth_error_counts"])
        assert one["pairs"] == many["pairs"]


def test_frame_medians_equal_the_per_frame_scan():
    """One bucketing pass gives exactly what scanning every pair per frame gave."""
    rng = np.random.default_rng(2)
    n = 12
    pairs = [
        _pair(int(i), int(j), float(rng.normal(0, 0.05)), 5.0)
        for i, j in rng.integers(0, n, size=(80, 2))
        if i != j
    ]

    # The pre-change O(N * pairs) scan, frame 11 may well be untouched
    scan = []
    for k in range(n):
        v = [abs(p.median_rel_depth_error) for p in pairs if k in (p.idx1, p.idx2)]
        scan.append(float(np.median(v)) if v else None)

    assert metrics._frame_medians(pairs, n) == scan
    assert metrics._frame_medians([], 3) == [None, None, None]


def test_median_abs_rel_depth_error_is_the_median_magnitude_over_pairs_touching_the_frame(monkeypatch):
    """Both directions count for a frame, by magnitude: frame 1 sees |-0.02|, |0.04| and |0.10|."""
    collected = _collected([_pair(0, 1, -0.02, 2.0), _pair(1, 0, 0.04, 2.0), _pair(1, 2, 0.10, 2.0)])
    n, hw = 3, 8
    monkeypatch.setattr(metrics, "_collect_pairs", lambda *a, **k: (collected, [None] * n))
    K = np.array([[50.0, 0, hw / 2], [0, 50.0, hw / 2], [0, 0, 1.0]])
    tables = metrics.compute_reconstruction_quality(
        np.ones((n, hw, hw)),
        np.stack([K] * n),
        np.stack([K] * n),
        np.stack([np.eye(4)] * n),
        np.tile([0, 0, hw, hw, hw, hw], (n, 1)),
        [f"frame_{k:06d}.png" for k in range(n)],
        None,
        None,
    )
    assert tables["frames"]["median_abs_rel_depth_error"] == pytest.approx([0.03, 0.04, 0.10])


def test_confidence_median_is_null_without_a_confidence_array(tmp_path):
    """No array means the column was never computed — distinct from a low confidence."""
    tables, _ = _quality(tmp_path, ["frame_000000.jpg", "frame_000007.jpg"], with_confidence=False)
    assert tables["frames"]["confidence_median"] == [None, None]
    tables, _ = _quality(tmp_path / "c", ["frame_000000.jpg", "frame_000007.jpg"])
    assert all(0.5 <= v <= 1.0 for v in tables["frames"]["confidence_median"])


def test_crop_coverage_is_measured_against_the_ORIGINAL_canvas_not_the_model_grid(tmp_path):
    """The model grid IS the crop, so coverage is structurally invisible at model resolution.

    The fixture crops 16x16 out of a 32x24 canvas from an off-centre origin: 1/3 covered.
    Dropping the top-left origin gives 0.469 and using the model grid gives 1.0, so both
    mistakes are separated from the right answer.
    """
    tables, _ = _quality(tmp_path, ["frame_000000.jpg", "frame_000007.jpg"])
    assert tables["frames"]["covered_fraction"] == pytest.approx([1.0 / 3.0, 1.0 / 3.0])


def test_negative_sampled_depth_is_no_measurement():
    depth, K, extr = _two_view()
    depth[1] *= -1
    assert all((r.idx1, r.idx2) != (0, 1) for r in _collect(depth, K, extr)["pairs"])


def test_photometric_ncc_on_uint8_frames_is_the_float64_ncc_of_the_exact_warp():
    """The warp is exactly 4 px, so the NCC is np.corrcoef over known slices, to float64 precision."""
    img, depth, K, e = _translated_pair(shift_px=4, hw=32)
    img = img.copy()
    img[1] = 0.6 * img[1] + 0.4 * np.random.default_rng(3).uniform(0, 255, img[1].shape)
    img = img.round().astype(np.uint8)
    m = compute_photometric_ncc(img, depth, K, e, max_separation=1)

    # Frame 0's column u lands on frame 1's u - 4: columns 4.. against columns ..28
    a = img[0][:, 4:].astype(np.float64).ravel()
    b = img[1][:, :-4].astype(np.float64).ravel()
    assert m["n_pixels"] == [32 * 28]
    assert abs(m["photometric_ncc"][0] - np.corrcoef(a, b)[0, 1]) < 1e-12


def test_photometric_ncc_on_lifted_depth_matches_uint8_and_float_frames_exactly():
    """Model-grid depth lifted under a uint8 guide as-is or a float32 one cast down: same result."""
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    img = img.round()
    model_d = np.stack([np.concatenate(
        [np.full((16, 8), 3.0, np.float32), np.full((16, 8), 5.0, np.float32)], axis=1)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))
    K = rescale_intrinsics(model_K, (16, 16), (32, 32))
    K = shift_intrinsics(K, coords[:, :2])

    as_float = compute_photometric_ncc(img.astype(np.float32), model_d, K, e, original_coords=coords,
                                       max_separation=1)
    as_uint8 = compute_photometric_ncc(img.astype(np.uint8), model_d, K, e, original_coords=coords,
                                       max_separation=1)
    assert as_uint8["n_pixels"] == as_float["n_pixels"] and as_uint8["n_pixels"][0] > 0
    assert as_uint8["photometric_ncc"] == as_float["photometric_ncc"]


def test_photometric_ncc_is_invariant_to_a_far_world_origin():
    """Georeferenced poses sit ~10 km from the origin; float32 world points there lose the warp."""
    img, depth, K, e = _translated_pair(shift_px=4, hw=32)
    img = np.stack([img[0], 0.7 * img[1] + 0.3 * np.random.default_rng(3).uniform(0, 255, img[1].shape)])
    near = compute_photometric_ncc(img, depth, K, e, max_separation=1)

    # Same rig with the world origin moved 10 km away: t' = t - R @ offset for every camera
    offset = np.array([1e4, -1e4, 5e3])
    far_e = e.astype(np.float64)
    far_e[:, :3, 3] -= far_e[:, :3, :3] @ offset
    far = compute_photometric_ncc(img, depth, K, far_e, max_separation=1)

    assert far["n_pixels"] == near["n_pixels"]
    np.testing.assert_allclose(far["photometric_ncc"], near["photometric_ncc"], rtol=0, atol=1e-4)


def _rotated_pose(yaw, pitch, t):
    """World-to-cam pose: yaw about y, then pitch about x, then translation t."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    e = np.eye(4)
    e[:3, :3] = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    e[:3, 3] = t
    return e


def _reference_ncc(img, depth, K, e):
    """Frame 0 warped into frame 1 in float64 numpy through world coordinates: (n_pixels, NCC)."""
    h, w = depth.shape[1:]
    v, u = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    d = depth[0].astype(np.float64)
    pc0 = np.stack([(u - K[0, 0, 2]) / K[0, 0, 0] * d, (v - K[0, 1, 2]) / K[0, 1, 1] * d, d], -1).reshape(-1, 3)
    world = (pc0 - e[0, :3, 3]) @ e[0, :3, :3]
    pc1 = world @ e[1, :3, :3].T + e[1, :3, 3]
    z = np.maximum(pc1[:, 2], 1e-6)
    ui = np.round(pc1[:, 0] * K[1, 0, 0] / z + K[1, 0, 2]).astype(np.int64)
    vi = np.round(pc1[:, 1] * K[1, 1, 1] / z + K[1, 1, 2]).astype(np.int64)
    ok = (pc1[:, 2] > 0) & (d.ravel() > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    a = img[0].reshape(-1, 3)[ok].astype(np.float64).ravel()
    b = img[1][vi[ok], ui[ok]].astype(np.float64).ravel()
    return int(ok.sum()), float(np.corrcoef(a, b)[0, 1])


def _rotated_rig(h, w, seed=11):
    """Two rotated cameras with their own K (fx != fy), plus the same rig 100 km from the origin."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 255, size=(2, h, w, 3)).round().astype(np.uint8)
    depth = rng.uniform(3.0, 5.0, size=(2, h, w)).astype(np.float32)

    # Per-frame K on a 48x48 base, scaled to the image size
    sx, sy = w / 48, h / 48
    K = np.stack([
        np.array([[40.0 * sx, 0, 24.0 * sx], [0, 44.0 * sy, 22.0 * sy], [0, 0, 1.0]]),
        np.array([[38.0 * sx, 0, 25.0 * sx], [0, 41.0 * sy, 24.0 * sy], [0, 0, 1.0]]),
    ])
    e = np.stack([_rotated_pose(0.1, 0.02, [0.1, 0.0, 0.05]), _rotated_pose(-0.05, -0.04, [-0.2, 0.05, 0.0])])

    # Same rig with the world origin moved 100 km away: t' = t - R @ offset for every camera
    far_e = e.copy()
    far_e[:, :3, 3] -= far_e[:, :3, :3] @ np.array([1e5, -1e5, 5e4])
    return img, depth, K, e, far_e


def test_photometric_ncc_composes_rotated_poses_at_a_far_origin():
    """Both cameras rotated, own K each, origin 100 km away: the warp matches a float64 reference."""
    img, depth, K, e, far_e = _rotated_rig(48, 48)
    n_ref, ncc_ref = _reference_ncc(img, depth, K, e)

    for poses in (e, far_e):
        m = compute_photometric_ncc(img, depth, K, poses, max_separation=1)
        assert m["n_pixels"] == [n_ref] and n_ref > 1000
        assert abs(m["photometric_ncc"][0] - ncc_ref) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is a CUDA matmul mode")
def test_photometric_ncc_warp_is_exact_under_global_tf32(monkeypatch):
    """mapanything turns TF32 on at import; a float32 CUDA matmul in the warp would then flip pixels."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    img, depth, K, _, far_e = _rotated_rig(192, 256)
    n_ref, ncc_ref = _reference_ncc(img, depth, K, far_e)

    m = compute_photometric_ncc(img, depth, K, far_e, max_separation=1)
    assert m["n_pixels"] == [n_ref] and n_ref > 20000
    assert abs(m["photometric_ncc"][0] - ncc_ref) < 1e-4


def _three_views_one_facing_away():
    """Frames 0 and 1 share a view of a plane; frame 2 sits at the origin facing -z."""
    depth, K, extr = _two_view(scale_j=1.02)
    away = np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float32)
    return (np.concatenate([depth, depth[:1]]), np.concatenate([K, K[:1]]), np.concatenate([extr, away[None]]))


def test_frustum_overlap_is_one_for_the_same_camera_and_zero_facing_away():
    depth, K, extr = _three_views_one_facing_away()
    d, k, e = (torch.as_tensor(x) for x in (depth, K, extr))
    points = metrics.unproject(d[0], e[0], k[0]).reshape(-1, 3)
    overlap = metrics._frustum_overlap(points, e, k, *depth.shape[1:])
    assert overlap[0] == 1.0
    assert 0.0 < overlap[1] <= 1.0
    assert overlap[2] == 0.0


def test_frustum_overlap_of_no_points_is_zero():
    depth, K, extr = _three_views_one_facing_away()
    overlap = metrics._frustum_overlap(torch.zeros(0, 3), torch.as_tensor(extr), torch.as_tensor(K), 16, 16)
    assert overlap.tolist() == [0.0, 0.0, 0.0]


def _spy_pair_pass(monkeypatch):
    """
    Count target views handed to depth_agreement and calls to _frustum_overlap.
    """
    counts = {"views": 0, "overlap_calls": 0}
    agreement, overlap = metrics.depth_agreement, metrics._frustum_overlap

    def spy_agreement(points, extrinsics, *args, **kwargs):
        counts["views"] += len(extrinsics)
        return agreement(points, extrinsics, *args, **kwargs)

    def spy_overlap(*args, **kwargs):
        counts["overlap_calls"] += 1
        return overlap(*args, **kwargs)

    monkeypatch.setattr(metrics, "depth_agreement", spy_agreement)
    monkeypatch.setattr(metrics, "_frustum_overlap", spy_overlap)
    return counts


def test_min_pair_overlap_zero_is_the_unpruned_pass(monkeypatch):
    """
    At 0.0 the pre-pass never runs and every ordered pair reaches the depth pass.
    """
    depth, K, extr = _three_views_one_facing_away()
    counts = _spy_pair_pass(monkeypatch)
    metrics._collect_pairs(depth, K, extr, 0.05, min_pair_overlap=0.0)
    assert counts == {"views": 6, "overlap_calls": 0}


@pytest.mark.parametrize("target_batch", [1, 2])
def test_min_pair_overlap_drops_only_pairs_that_share_no_view(monkeypatch, target_batch):
    """
    Frame 2 sees nothing of 0 or 1, so pruning it changes no row, count or agreement.
    """
    depth, K, extr = _three_views_one_facing_away()
    full, agree_full = metrics._collect_pairs(depth, K, extr, 0.05, target_batch=target_batch)

    # Pruning skips the four pairs touching frame 2; one overlap pre-pass per source frame
    counts = _spy_pair_pass(monkeypatch)
    pruned, agree_pruned = metrics._collect_pairs(
        depth, K, extr, 0.05, min_pair_overlap=0.01, target_batch=target_batch
    )
    assert counts == {"views": 2, "overlap_calls": 3}

    assert pruned["pairs"] == full["pairs"] and agree_pruned == agree_full
    np.testing.assert_array_equal(pruned["rel_depth_error_counts"], full["rel_depth_error_counts"])
    assert {(p.idx1, p.idx2) for p in full["pairs"]} == {(0, 1), (1, 0)}


@pytest.mark.parametrize("target_batch", [1, 2])
def test_min_pair_overlap_prunes_a_middle_frame_without_shifting_the_rest(target_batch):
    """
    A pruned target ahead of a kept one must not shift which depth map the kept one reads.
    """
    depth, K, extr = _three_views_one_facing_away()
    order = [0, 2, 1]
    pruned, agree = metrics._collect_pairs(
        depth[order], K[order], extr[order], 0.05, min_pair_overlap=0.01, target_batch=target_batch
    )
    assert {(p.idx1, p.idx2) for p in pruned["pairs"]} == {(0, 2), (2, 0)}
    assert agree == [1.0, None, 1.0]


def test_min_pair_overlap_above_a_pairs_overlap_drops_that_pair():
    depth, K, extr = _three_views_one_facing_away()
    pruned, agree = metrics._collect_pairs(depth, K, extr, 0.05, min_pair_overlap=1.01)
    assert pruned["pairs"] == []
    assert agree == [None, None, None]
    assert pruned["rel_depth_error_counts"].sum() == 0


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))],
)
def test_pair_rows_are_the_per_pair_quantiles_and_medians(monkeypatch, device):
    """
    Batched pair rows equal torch.quantile and torch.median over each pair's own pixels, exactly.
    """
    monkeypatch.setattr(metrics, "get_device", lambda: device)
    n, h, w = 5, 20, 24
    rng = np.random.default_rng(1)

    # Depth rounded to 0.1 so residuals tie; holes in frame 1; frame 3 faces away, an empty pair mid-batch
    depth = np.round(3 + rng.uniform(-0.5, 0.5, (n, h, w)), 1).astype(np.float32)
    depth[1, :5] = 0
    K = np.tile(np.array([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]], np.float32), (n, 1, 1))
    E = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    E[:, 0, 3] = np.linspace(0, 0.3, n)
    E[3, :3, :3] = np.diag([-1.0, 1.0, -1.0])

    collected, _ = metrics._collect_pairs(depth, K, E, 0.05, target_batch=3)
    rows = {(p.idx1, p.idx2): p for p in collected["pairs"]}

    # Reference: one pair at a time, each reduced by torch.quantile and torch.median
    depth_t, K_t, E_t = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (depth, K, E))
    centers = torch.as_tensor(invert_poses(E)[:, :3, 3], dtype=torch.float32, device=device)
    quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)
    expected = {}

    for i in range(n):
        points = unproject(depth_t[i], E_t[i], K_t[i]).reshape(-1, 3)

        for j in range(n):
            if j == i:
                continue

            _, seen, rel, z = depth_agreement(points, E_t[j : j + 1], K_t[j : j + 1], depth_t[j : j + 1], 0.05)
            sel = seen[0] & (depth_t[i].reshape(-1) > 0) & (rel[0] > -1) & (z[0] > 1e-6)
            if not sel.any():
                continue

            ray_i = points[sel] - centers[i]
            ray_j = points[sel] - centers[j]
            cos_a = (ray_i * ray_j).sum(-1) / (ray_i.norm(dim=-1) * ray_j.norm(dim=-1)).clamp(min=1e-12)
            parallax = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))
            q = torch.quantile(rel[0][sel], quantiles)
            expected[(i, j)] = (int(sel.sum()), q[1].item(), (q[2] - q[0]).item(), parallax.median().item(), z[0][sel].median().item())

    assert not [k for k in expected if 3 in k]
    assert len({v[0] for v in expected.values()}) > 1
    assert set(rows) == set(expected)

    for key, want in expected.items():
        p = rows[key]
        assert (p.n_pixels, p.median_rel_depth_error, p.iqr_rel_depth_error, p.median_parallax_deg, p.median_depth) == want
