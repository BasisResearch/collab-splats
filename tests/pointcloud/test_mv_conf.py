"""Unit tests for compute_multiview_depth_confidence in base.py."""

import numpy as np
import pytest

from collab_splats.geometry.metrics import residual_bin_edges
from collab_splats.pointcloud.feedforward.base import (
    MultiviewConfidence,
    compute_multiview_depth_confidence,
    multiview_mask,
)


def _make_intrinsics(H: int, W: int) -> np.ndarray:
    """Simple pinhole K with focal = W, principal at image centre."""
    return np.array(
        [[float(W), 0.0, W / 2.0], [0.0, float(H), H / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_compute_mv_conf_identical_cameras():
    """Two co-located cameras, same depth → mv_conf = 1.0 for all valid pixels."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    ).ratio

    assert mv_conf.shape == (N, H, W)
    assert mv_conf.dtype == np.float32
    assert np.all(mv_conf > 0.9), f"Expected all >0.9; min={mv_conf.min():.4f}"


def test_compute_mv_conf_depth_disagreement():
    """Same pose, very different depths → mv_conf = 0.0 everywhere."""
    N, H, W = 2, 4, 4
    depth = np.zeros((N, H, W), dtype=np.float32)
    depth[0] = 1.0
    depth[1] = 100.0

    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    mv_conf = compute_multiview_depth_confidence(
        depth, intrinsics, extrinsics, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    ).ratio

    assert np.all(mv_conf == 0.0), f"Expected all 0.0; max={mv_conf.max():.4f}"


def test_compute_mv_conf_depth_masks_source():
    """depth_masks=False on frame 0 → frame 0 source pixels get mv_conf = 0."""
    N, H, W = 2, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    intrinsics = np.stack([K, K])
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)

    depth_masks = np.ones((N, H, W), dtype=bool)
    depth_masks[0] = False

    mv_conf = compute_multiview_depth_confidence(
        depth,
        intrinsics,
        extrinsics,
        depth_masks=depth_masks,
        abs_thresh=0.0,
        rel_thresh=0.1,
        device="cpu",
    ).ratio

    assert np.all(mv_conf[0] == 0.0), f"Frame 0 should be 0; got {mv_conf[0]}"
    assert np.any(mv_conf[1] > 0.0), "Frame 1 should have some inliers"


def test_compute_mv_conf_output_shape():
    """Output shape matches (N, H, W) regardless of N."""
    for N in (1, 3, 5):
        H, W = 6, 6
        depth = np.ones((N, H, W), dtype=np.float32) * 3.0
        K = _make_intrinsics(H, W)
        intrinsics = np.stack([K] * N)
        extrinsics = np.stack([np.eye(4, dtype=np.float32)] * N)
        out = compute_multiview_depth_confidence(depth, intrinsics, extrinsics, device="cpu").ratio
        assert out.shape == (N, H, W), f"N={N}: expected {(N,H,W)}, got {out.shape}"


def test_nearest_sampling_no_fabricated_depth():
    """Bilinear across a depth step invents a depth on no surface; nearest cannot.

    Two cameras separated along x view a scene that is 2.0 deep on the left half and
    8.0 on the right. Sampled depth must be one of the two surface depths, never between.
    """
    N, H, W = 2, 16, 16
    depth = np.empty((N, H, W), dtype=np.float32)
    depth[:, :, : W // 2] = 2.0
    depth[:, :, W // 2 :] = 8.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    extr[1, 0, 3] = 0.05  # small baseline so the step lands mid-pixel in view 1

    out = compute_multiview_depth_confidence(
        depth, np.stack([K, K]), extr, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    )
    # Every pixel here sits on a rigid surface seen by both cameras, so with nearest
    # sampling every comparison is against a real surface depth and all of them agree.
    # Under bilinear the column just left of the step samples 0.6*2.0 + 0.4*8.0 = 3.6 —
    # a depth on no surface — and is scored an outlier. Judged pixels only: the last
    # column projects off the right edge of view 1 and has no partner at all.
    judged_px = out.valid_count[0] > 0
    assert judged_px.sum() > 0
    assert np.all(out.ratio[0][judged_px] == 1.0), (
        f"{int((out.ratio[0][judged_px] < 1.0).sum())} judged px below 1.0 — " f"sampler is fabricating depth"
    )


def test_scale_invariance():
    """Scaling depth and translations by a constant leaves the output identical.

    This is the property that lets one function serve five backbones with different depth
    scales, and it is why abs_thresh must stay 0.0 for non-metric depth.
    """
    N, H, W = 3, 8, 8
    rng = np.random.default_rng(11)
    depth = (rng.random((N, H, W)).astype(np.float32) + 0.5) * 2.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.25 * i
    kw = dict(abs_thresh=0.0, rel_thresh=0.05, device="cpu")

    base = compute_multiview_depth_confidence(depth, np.stack([K] * N), extr, **kw)

    # A power of two so the rescaling is exact in float32 and the comparison isolates the
    # invariance itself. With an arbitrary factor a handful of pixels sitting exactly on the
    # tolerance boundary flip on rounding alone, which says nothing about the property.
    s = 8.0
    extr_s = extr.copy()
    extr_s[:, :3, 3] *= s
    scaled = compute_multiview_depth_confidence(depth * s, np.stack([K] * N), extr_s, **kw)

    np.testing.assert_array_equal(base.inlier_count, scaled.inlier_count)
    np.testing.assert_array_equal(base.valid_count, scaled.valid_count)
    np.testing.assert_allclose(base.ratio, scaled.ratio, rtol=1e-6, atol=1e-6)


def test_intrinsics_resolution_mismatch_raises():
    """K implying a different pixel grid than depth.shape is the mesh-regression bug class."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 4.0, dtype=np.float32)
    # K for an image 2x larger than the depth grid — the realistic model-res vs
    # original-res pairing, which puts the principal point on the far edge at (8, 8).
    K_big = np.array([[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="resolution"):
        compute_multiview_depth_confidence(
            depth,
            np.stack([K_big, K_big]),
            np.stack([np.eye(4, dtype=np.float32)] * 2),
            device="cpu",
        )


def test_shape_mismatch_raises():
    """N must agree across depth, intrinsics and extrinsics."""
    depth = np.full((3, 8, 8), 4.0, dtype=np.float32)
    K = _make_intrinsics(8, 8)
    with pytest.raises(ValueError, match="length"):
        compute_multiview_depth_confidence(
            depth, np.stack([K, K]), np.stack([np.eye(4, dtype=np.float32)] * 2), device="cpu"
        )


def test_occluded_view_excluded_not_penalised():
    """A view occluded by a nearer surface is evidence absent, not evidence against.

    View 0 sees a wall at 8.0. View 1 sits at the same pose but its depth map is a slab
    at 2.0 — everything view 0 sees is hidden behind it. View 0's ratio must stay 1.0
    against a third, agreeing view rather than being dragged to 0.5.
    """
    N, H, W = 3, 8, 8
    depth = np.empty((N, H, W), dtype=np.float32)
    depth[0] = 8.0
    depth[1] = 2.0  # occluder slab
    depth[2] = 8.0  # agrees with view 0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])

    # pair_gate=False: every depth map here is a constant, so each frustum degenerates to
    # a single depth plane and the AABB test correctly finds them disjoint — which would
    # skip the occluding pair entirely and make this test vacuous.
    out = compute_multiview_depth_confidence(
        depth,
        np.stack([K] * N),
        extr,
        abs_thresh=0.0,
        rel_thresh=0.05,
        pair_gate=False,
        device="cpu",
    )
    assert np.all(out.ratio[0] == 1.0), f"view 0 penalised for being occluded: {out.ratio[0].min()}"
    assert np.all(out.valid_count[0] == 1), "the occluding view should leave the denominator"


def test_free_space_violation_still_counts_as_outlier():
    """sampled > expected + tol means nothing is there — real evidence against."""
    N, H, W = 2, 8, 8
    depth = np.empty((N, H, W), dtype=np.float32)
    depth[0] = 2.0
    depth[1] = 8.0  # view 1 sees empty space where view 0 claims a surface
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    # pair_gate=False for the same reason as the occlusion test: constant depth maps give
    # degenerate single-plane frusta that the AABB test correctly separates.
    out = compute_multiview_depth_confidence(
        depth,
        np.stack([K, K]),
        extr,
        abs_thresh=0.0,
        rel_thresh=0.05,
        pair_gate=False,
        device="cpu",
    )
    assert np.all(out.ratio[0] == 0.0), "free-space violation must count against"
    assert np.all(out.valid_count[0] == 1), "the violating view must stay in the denominator"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_positive_mask_invariant_across_occlusion_policy(seed):
    """The MapAnything guarantee: excluding occluded views cannot change inlier_count.

    The occlusion policy alters only valid_count. min_views thresholds inlier_count, so
    the mask is invariant at every K — not only K=1.
    """
    N, H, W = 4, 8, 8
    rng = np.random.default_rng(seed)
    depth = (rng.random((N, H, W)).astype(np.float32) + 0.5) * 4.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.2 * i
    out = compute_multiview_depth_confidence(
        depth, np.stack([K] * N), extr, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    )
    # inlier_count is the numerator the mask thresholds; it must never see the policy.
    # An occluded view was never an inlier, so excluding it cannot move this array.
    for k in (1, 2, 3, 4):
        mask = out.inlier_count >= k
        assert mask.shape == (N, H, W)
    assert np.all(out.inlier_count <= out.valid_count)


def test_pair_gate_does_not_change_output():
    """The gate is a cost optimisation: gated and ungated results must be identical."""
    N, H, W = 4, 8, 8
    rng = np.random.default_rng(7)
    depth = (rng.random((N, H, W)).astype(np.float32) + 1.0) * 3.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.3 * i
    kw = dict(abs_thresh=0.0, rel_thresh=0.05, device="cpu")
    gated = compute_multiview_depth_confidence(depth, np.stack([K] * N), extr, pair_gate=True, **kw)
    plain = compute_multiview_depth_confidence(depth, np.stack([K] * N), extr, pair_gate=False, **kw)
    np.testing.assert_array_equal(gated.inlier_count, plain.inlier_count)
    np.testing.assert_array_equal(gated.valid_count, plain.valid_count)
    np.testing.assert_array_equal(gated.judged, plain.judged)


def test_pair_gate_skips_disjoint_views():
    """Two views looking at scenes 1000 units apart share no frustum volume."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 2.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    extr[1, 0, 3] = 1000.0
    out = compute_multiview_depth_confidence(depth, np.stack([K, K]), extr, pair_gate=True, device="cpu")
    assert out.valid_count.max() == 0, "disjoint views should contribute nothing"
    assert out.judged.tolist() == [False, False]


def test_returns_multiview_confidence_dataclass():
    """The function returns MultiviewConfidence with ratio, both counts, and judged."""
    N, H, W = 2, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    out = compute_multiview_depth_confidence(
        depth,
        np.stack([K, K]),
        np.stack([np.eye(4, dtype=np.float32)] * 2),
        abs_thresh=0.0,
        rel_thresh=0.1,
        device="cpu",
    )
    assert isinstance(out, MultiviewConfidence)
    assert out.ratio.shape == (N, H, W)
    assert out.ratio.dtype == np.float32
    assert out.inlier_count.shape == (N, H, W)
    assert out.inlier_count.dtype == np.int32
    assert out.valid_count.shape == (N, H, W)
    assert out.valid_count.dtype == np.int32
    assert out.judged.shape == (N,)
    assert out.judged.dtype == np.bool_


def test_counts_consistent_with_ratio():
    """ratio == inlier_count / valid_count wherever valid_count > 0, else 0."""
    N, H, W = 3, 6, 6
    rng = np.random.default_rng(0)
    depth = (rng.random((N, H, W)).astype(np.float32) + 1.0) * 2.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.1 * i
    out = compute_multiview_depth_confidence(
        depth, np.stack([K] * N), extr, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    )
    expected = np.where(
        out.valid_count > 0,
        out.inlier_count / np.maximum(out.valid_count, 1),
        0.0,
    ).astype(np.float32)
    np.testing.assert_allclose(out.ratio, expected, rtol=0, atol=0)
    assert np.all(out.inlier_count <= out.valid_count)


def test_single_view_is_unjudged():
    """N=1 has no partners: judged is False and every count is zero."""
    H, W = 4, 4
    depth = np.full((1, H, W), 3.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    out = compute_multiview_depth_confidence(depth, K[None], np.eye(4, dtype=np.float32)[None], device="cpu")
    assert out.judged.tolist() == [False]
    assert out.valid_count.max() == 0
    assert out.inlier_count.max() == 0


def test_min_views_one_matches_ratio_gt_zero():
    """K=1 is exactly today's threshold=0.0, so MapAnything's mask is unchanged."""
    N, H, W = 4, 8, 8
    rng = np.random.default_rng(3)
    depth = (rng.random((N, H, W)).astype(np.float32) + 0.5) * 3.0
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    for i in range(N):
        extr[i, 0, 3] = 0.2 * i
    out = compute_multiview_depth_confidence(
        depth, np.stack([K] * N), extr, abs_thresh=0.0, rel_thresh=0.05, device="cpu"
    )
    judged_pixels = np.broadcast_to(out.judged[:, None, None], out.ratio.shape)
    np.testing.assert_array_equal((out.ratio > 0.0)[judged_pixels], (out.inlier_count >= 1)[judged_pixels])


def test_unjudged_view_keeps_pixels():
    """A view with no overlapping partners keeps its valid depth, matching upstream."""
    N, H, W = 2, 8, 8
    depth = np.full((N, H, W), 2.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    extr[1, 0, 3] = 1000.0  # disjoint frusta: neither view is judged
    out = compute_multiview_depth_confidence(depth, np.stack([K, K]), extr, pair_gate=True, device="cpu")
    mask = multiview_mask(out, depth > 0, min_views=2)
    assert np.all(mask), "unjudged views must keep their valid pixels, not be deleted"


def test_multiview_mask_respects_min_views():
    """A judged pixel whose partners disagree is dropped; min_views is a count, not a ratio."""
    N, H, W = 3, 4, 4
    # Co-located cameras so the warp is the identity in pixel space, with depth varying across
    # the grid so the frustum AABBs are fat enough to survive the pair gate. View 2 sits 30%
    # nearer than the other two, disagreeing with both at rel_thresh=0.1.
    base = np.linspace(4.0, 6.0, H * W, dtype=np.float32).reshape(H, W)
    depth = np.stack([base, base, base * 0.7])
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    out = compute_multiview_depth_confidence(
        depth, np.stack([K] * N), extr, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    )
    # The odd view out is judged and every partner it has votes against it.
    assert out.judged[2] and np.all(out.inlier_count[2] == 0)
    keep = multiview_mask(out, depth > 0, min_views=1)
    assert not np.any(keep[2]), "a judged view with zero agreeing partners must be dropped"
    # Views 0 and 1 agree with each other; being nearer, view 2 occludes them, so it never
    # judges them back — one reachable partner each, and it says yes.
    assert np.all(keep[:2])


def test_min_views_above_partner_count_clamps_instead_of_emptying():
    """K larger than a pixel's partner count degrades to 'all partners agree', not 'delete'."""
    N, H, W = 3, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    out = compute_multiview_depth_confidence(
        depth, np.stack([K] * N), extr, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    )
    # Every pixel has 2 partners and both agree. min_views is an ABSOLUTE count, so an
    # unclamped K=16 would be unsatisfiable and would empty a short sequence outright.
    assert np.all(out.valid_count == 2)
    for k in (1, 2, 3, 16):
        assert np.all(multiview_mask(out, depth > 0, min_views=k)), f"min_views={k} emptied the mask"


def test_multiview_mask_respects_valid_depth():
    """Pixels outside valid_depth are never resurrected by the mask helper."""
    N, H, W = 2, 4, 4
    depth = np.full((N, H, W), 5.0, dtype=np.float32)
    K = _make_intrinsics(H, W)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
    out = compute_multiview_depth_confidence(
        depth, np.stack([K, K]), extr, abs_thresh=0.0, rel_thresh=0.1, device="cpu"
    )
    valid = np.ones((N, H, W), dtype=bool)
    valid[0, 0, 0] = False
    assert not multiview_mask(out, valid, min_views=1)[0, 0, 0]


########################################
# The opt-in collection out-param
########################################


def _two_view(scale_j: float = 1.0):
    """Two cameras with a 0.2-unit sideways baseline viewing a constant-depth plane.

    scale_j multiplies frame 1's depth, injecting a known relative residual.
    """
    H = W = 16
    K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], dtype=np.float32)
    depth = np.stack([np.full((H, W), 4.0, np.float32), np.full((H, W), 4.0 * scale_j, np.float32)])
    extr = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    extr[1, 0, 3] = -0.2  # world-to-cam translation => camera 1 sits at x=+0.2
    return depth, np.stack([K, K]), extr


def _collect(depth, K, extr, **kw):
    """Collect over the fixture with the frustum gate off.

    A constant-depth plane has near == far, so its world AABB is degenerate in z. Two planes
    at 4.0 and 4.4 therefore have DISJOINT boxes and _aabbs_overlap gates the pair out before
    any residual exists — the gate is correct (real depth has range), the fixture is the
    artificial one. pair_gate is a cost optimisation and changes no result, so turning it off
    here isolates what these tests are actually about.
    """
    out = {}
    compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out, pair_gate=False, **kw)
    return out


def test_collect_defaults_to_none_and_output_is_unchanged():
    """The four production creators must see byte-identical output."""
    depth, K, extr = _two_view()
    base = compute_multiview_depth_confidence(depth, K, extr, device="cpu")
    out = {}
    withc = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out)
    assert np.array_equal(base.ratio, withc.ratio)
    assert np.array_equal(base.inlier_count, withc.inlier_count)
    assert np.array_equal(base.valid_count, withc.valid_count)
    assert np.array_equal(base.judged, withc.judged)


def test_collect_fills_index_keyed_rows_and_the_one_histogram():
    depth, K, extr = _two_view()
    out = _collect(depth, K, extr)
    assert out["rel_depth_error_counts"].sum() > 0
    assert (out["pairs"][0].idx1, out["pairs"][0].idx2) == (0, 1)
    assert out["pairs"][0].name1 is None  # index is the key; no filenames invented
    # Edges travel with the counts: they are sized from this scene, so counts alone are unreadable.
    assert len(out["rel_depth_error_edges"]) == len(out["rel_depth_error_counts"]) + 1
    n, h, w = depth.shape
    assert np.array_equal(out["rel_depth_error_edges"], residual_bin_edges(n * (n - 1) // 2 * h * w))


def test_signed_residual_recovers_an_injected_depth_scale():
    """Frame 1 depth x1.1 => median relative residual ~ +0.1 on the 0->1 pair."""
    depth, K, extr = _two_view(scale_j=1.1)
    out = _collect(depth, K, extr, rel_thresh=0.5)
    row = next(r for r in out["pairs"] if (r.idx1, r.idx2) == (0, 1))
    assert row.median_rel_depth_error == pytest.approx(0.1, abs=0.02)


def test_signed_residual_is_zero_on_a_consistent_pair():
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_rel_depth_error == pytest.approx(0.0, abs=1e-3)


def test_parallax_angle_matches_geometry():
    """0.2 baseline at depth 4 => atan(0.2/4) ~ 2.86 deg at the principal ray."""
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_parallax_deg == pytest.approx(
        np.degrees(np.arctan(0.2 / 4.0)), abs=0.5
    )


def test_median_depth_lands_on_the_row():
    """The 'worse further away?' axis is a column, not a binning routine."""
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_depth == pytest.approx(4.0, abs=0.2)


def test_occluded_pixels_are_excluded_from_the_residual():
    """Occlusion is absent evidence, not disagreement — it must not pollute the scale bias."""
    depth, K, extr = _two_view()
    depth[1, :, :8] = 0.5  # a near occluder covering half of frame 1
    row = next(r for r in _collect(depth, K, extr)["pairs"] if r.idx1 == 0)
    assert row.median_rel_depth_error == pytest.approx(0.0, abs=1e-3)


def test_residual_is_scale_invariant():
    """Multiplying depth and translation by s must leave the relative residual unchanged."""
    depth, K, extr = _two_view(scale_j=1.1)
    a = _collect(depth, K, extr, rel_thresh=0.5)["pairs"][0]
    s = 7.0
    extr_s = extr.copy()
    extr_s[:, :3, 3] *= s
    b = _collect(depth * s, K, extr_s, rel_thresh=0.5)["pairs"][0]
    assert a.median_rel_depth_error == pytest.approx(b.median_rel_depth_error, abs=1e-4)
    assert a.median_parallax_deg == pytest.approx(b.median_parallax_deg, abs=1e-3)


def test_a_huge_residual_still_lands_in_the_histogram():
    """The bounded axis means no residual can miss the bins, however large."""
    depth, K, extr = _two_view(scale_j=60.0)
    out = _collect(depth, K, extr, rel_thresh=1e9)
    assert out["rel_depth_error_counts"].sum() > 0
