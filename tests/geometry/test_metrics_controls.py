"""Negative controls: each measurement must move under its own fault and stay still under others.

The report aggregates three measurements chosen because their DEPENDENCIES DIFFER — epipolar
reads poses, depth cross-view reads poses+depth, photometric reads poses+depth+appearance.
Attribution works only if that separation is real, so every claim below is a fault of known
magnitude and known location, asserted to land in exactly one channel.

Three things this file deliberately does NOT do:

  * It does not assert on an empty collection. Every test that loops asserts the pair count
    first, because two separate mechanisms delete pairs silently — the frustum gate (see
    ``test_a_constant_depth_fixture_is_silently_gated_out``) and the occlusion branch at the
    production tolerance (see ``test_the_controls_must_run_above_the_production_rel_thresh``).
    A loop over {} passes.
  * It does not divide an invented number by a measured one. The pose control perturbs real
    extrinsics and measures the pixel motion that perturbation actually produces.
  * It does not treat "the fault did not reach here" and "nothing could reach here" as the
    same evidence. Where an arm is exact-zero by construction it says so, and carries a
    second arm that has to move.
"""

import inspect

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import compute_photometric_ncc, depth_error_in_pixels
from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence

########################################
# Constants — every one of these is load-bearing, so none of them is a bare literal
########################################

# FOCAL, HW and BASELINE are jointly tuned against the one-pixel-of-disparity floor: at f=30 a
# 0.25 baseline at range ~4 gives the worst strafing pair ~1.75 px of disparity, clear of the
# floor, while FLOOR_STEP lands a forward pair under it. Moving any of them moves both sides of
# test_control_forward_motion_is_the_direction_that_falls_under_the_floor.
FOCAL = 30.0
HW = 24
BASELINE = 0.25

# The world surface is the plane Z = Z0 + TILT*X. TILT is the whole reason this fixture works;
# see _scene and test_a_constant_depth_fixture_is_silently_gated_out for why it is not zero.
Z0 = 4.0
TILT = 0.3

# The injected depth fault. x1.1 is deliberately larger than the production rel_thresh of 0.05,
# because a fault inside the tolerance is not a fault the measurement is supposed to report.
DEPTH_FAULT = 1.1

# Every control that injects a fault runs at this tolerance, NOT the production default of 0.05.
# At 0.05 the occlusion branch deletes the very pairs the fault produced: a from-frame-2 pair
# reads sampled < expected - tol, which the measurement classifies as OCCLUDED (absent evidence)
# and drops from the collection entirely. Measured: 12 ordered pairs here, 9 at the default,
# with all three from-frame-2 pairs gone. Pinned by
# test_the_controls_must_run_above_the_production_rel_thresh, so moving the controls "back to
# the default" fails loudly instead of quietly measuring six clean pairs.
CONTROL_REL_THRESH = 0.5

# The injected pose fault: camera 1's centre translated along world Y. That axis is chosen
# because the plane Z = Z0 + TILT*X contains it, so the SURFACE IS INVARIANT under the
# perturbation — the depth channel is structurally blind to this fault, which is precisely the
# claim under test — while every projection still shifts by f*dy/Z. An X or Z translation, or a
# rotation, would perturb depth on a slanted plane and confound the control.
POSE_FAULT_DY = 0.4

# Matched baseline for the forward-vs-strafe contrast. Both directions get the SAME magnitude,
# so the comparison is about direction and not about step size.
FLOOR_STEP = 0.1


########################################
# Fixture
########################################


def _scene(n=4, centers=None, tilt=TILT):
    """N cameras viewing the world plane Z = Z0 + tilt*X, depth rendered exactly per camera.

    A SLANTED plane, not a fronto-parallel one, and that is load-bearing rather than cosmetic.
    A constant depth map makes near == far, so ``_frustum_world_aabbs`` produces a zero-thickness
    slab and ``_aabbs_overlap`` — which needs overlap on every axis — rejects any pair whose
    slabs sit at different depths. Scaling one frame's depth is exactly such a displacement, so
    on a flat fixture the depth-scale control has no pairs to measure: measured, the flat version
    yields 6 ordered pairs instead of 12, with every pair touching the scaled frame gone. The
    pair-count guards in each test below turn that into a loud failure; without them the
    assertions would run on an empty set, which is how the originally specified version of this
    file passed. The tilt gives each frustum real depth extent, which is what a real scene has,
    and as a side effect spreads parallax over 3.3-10.1 deg instead of pinning every pixel at one
    angle.

    Rotations stay identity and the plane is independent of Y, which buys two exact properties
    the controls below rely on: depth is a function of the pixel's x ray-component alone, and
    the surface is invariant under camera translation along world Y.
    """
    K = np.array([[FOCAL, 0, HW / 2], [0, FOCAL, HW / 2], [0, 0, 1.0]], dtype=np.float32)
    if centers is None:
        centers = [(BASELINE * k, 0.0, 0.0) for k in range(n)]  # camera k strafing sideways

    # World-to-camera: identity rotation, so t = -C.
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k, c in enumerate(centers):
        extr[k, :3, 3] = -np.asarray(c, np.float32)

    # Ray-plane intersection in closed form. Ray through pixel u is (a, b, 1) with
    # a = (u - cx)/f; substituting C + s*(a, b, 1) into Z = Z0 + tilt*X gives
    # s*(1 - tilt*a) = Z0 + tilt*Cx - Cz, and Z-depth equals s because the ray's z is 1.
    a = (np.arange(HW) - HW / 2) / FOCAL
    depth = np.stack(
        [
            np.tile(((Z0 + tilt * cx - cz) / (1.0 - tilt * a))[None, :], (HW, 1)).astype(np.float32)
            for cx, _, cz in centers
        ]
    )
    return depth, np.stack([K] * n), extr


def _pairs(depth, K, extr, **kw):
    """{(i, j): PairStats} from one collected multiview pass. CPU so the controls need no GPU."""
    out = {}
    compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out, **kw)
    return {(p.idx1, p.idx2): p for p in out["pairs"]}


def _faulted_scene(n=4):
    """The standard depth fault: frame 2 scaled, everything else exact."""
    depth, K, extr = _scene(n=n)
    depth[2] *= DEPTH_FAULT
    return depth, K, extr


def _texture(n):
    """Smooth RGB with a little noise: enough structure to correlate, no aliasing under warp."""
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
    base = 120 + 60 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    imgs = np.stack([np.stack([base + 10 * c for c in range(3)], -1)] * n).astype(np.float32)
    return imgs + rng.normal(0, 2.0, imgs.shape)


def _parallax_truth_deg(depth, K, extr, i, j):
    """Median ray-to-ray angle over frame i's pixels, computed WITHOUT the measurement under test.

    An independent reimplementation, sharing the fixture and nothing else. It exists because
    every other assertion in this file consumes ``median_parallax_deg`` on both sides of a ratio
    and therefore self-normalises: a parallax scaled by a constant is invisible to all of them.
    """
    H, W = depth.shape[1:]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], -1)
    pts_cam = (np.linalg.inv(K[i]) @ pix.T).T * depth[i].reshape(-1, 1)
    c2w = np.linalg.inv(extr)
    pts_world = (c2w[i] @ np.concatenate([pts_cam, np.ones((H * W, 1))], -1).T).T[:, :3]
    v_i, v_j = pts_world - c2w[i][:3, 3], pts_world - c2w[j][:3, 3]
    cos_a = (v_i * v_j).sum(-1) / (np.linalg.norm(v_i, axis=-1) * np.linalg.norm(v_j, axis=-1))
    return float(np.median(np.rad2deg(np.arccos(np.clip(cos_a, -1.0, 1.0)))))


def _reprojection_shift_px(depth, K, extr_true, extr_faulty, i, j):
    """Median pixel motion in frame j when frame j's pose changes — the error a tracker sees.

    Measured, not assumed: frame i's pixels are unprojected once through the TRUE geometry, then
    projected into frame j under both poses and differenced. This is the only honest way to put
    a pose fault on the same pixel axis the bridge's output lives on.
    """
    H, W = depth.shape[1:]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)
    ones = np.ones((H * W, 1))
    pts_cam_i = (np.linalg.inv(K[i]) @ pix.T).T * depth[i].reshape(-1, 1)
    pts_world = (np.linalg.inv(extr_true[i]) @ np.concatenate([pts_cam_i, ones], -1).T).T[:, :3]
    pts_world_h = np.concatenate([pts_world, ones], -1)

    def project(E):
        cam = (E[j] @ pts_world_h.T).T[:, :3]
        proj = (K[j] @ cam.T).T
        return proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None), cam[:, 2]

    uv_true, z_true = project(extr_true)
    uv_bad, _ = project(extr_faulty)
    in_front = z_true > 0
    return float(np.median(np.linalg.norm(uv_true[in_front] - uv_bad[in_front], axis=1)))


########################################
# The fixture itself is a control
########################################


def test_the_fixture_produces_every_pair_and_a_real_parallax_spread():
    """Nothing below means anything if a gate has quietly emptied the collection."""
    depth, K, extr = _scene(n=4)
    # No fault injected here, so this one runs at the PRODUCTION default: the clean fixture must
    # survive the shipping tolerance, and only a faulted one needs CONTROL_REL_THRESH.
    p = _pairs(depth, K, extr)
    # ORDERED directions: the mv loop runs (i, j) and (j, i) separately, so 4 frames give 12.
    assert len(p) == 12
    assert min(v.n_pixels for v in p.values()) > 100
    # Real depth extent, which is what keeps the frusta from degenerating to slabs.
    assert depth[0].max() / depth[0].min() > 1.2
    # And real parallax spread, so no control is secretly evaluated at a single angle.
    par = [v.median_parallax_deg for v in p.values()]
    assert max(par) / min(par) > 2.0


def test_the_fixture_parallax_matches_closed_form_geometry():
    """Pins median_parallax_deg to truth, which no ratio in this file can do.

    Every other parallax assertion here feeds the same number into both sides of a ratio, so a
    parallax scaled by a constant self-normalises and survives them all — measured, a x1.5
    scaling passes every other test in this file. This compares against an independent
    reimplementation instead. The residual disagreement is the pixel SET, not the angle: the
    measurement medians over the pixels that survive its own validity and occlusion filters,
    this helper medians over all of them, which is worth ~0.5-1.6% on this fixture.
    """
    depth, K, extr = _scene(n=4)
    p = _pairs(depth, K, extr)
    for i, j in [(0, 1), (1, 2), (0, 3), (2, 0)]:
        assert (i, j) in p
        truth = _parallax_truth_deg(depth, K, extr, i, j)
        assert p[(i, j)].median_parallax_deg == pytest.approx(truth, rel=0.03)
    # Absolute pin too, so a coordinated rescale of BOTH sides still fails.
    assert p[(0, 1)].median_parallax_deg == pytest.approx(3.41, abs=0.05)


def test_a_constant_depth_fixture_is_silently_gated_out():
    """Pins WHY _scene tilts the plane. A flat fixture loses pairs without raising anything.

    This is not a control; it is the recorded reason the fixture looks the way it does. If it
    ever fails the tilt may be dropped — until then, reverting _scene to a constant depth map
    would delete the depth-scale control's evidence while leaving it green.
    """
    flat, K, extr = _scene(n=4, tilt=0.0)
    assert len(_pairs(flat, K, extr, rel_thresh=CONTROL_REL_THRESH)) == 12  # flat alone is fine

    # ...until one frame's depth moves, which slides its zero-thickness frustum off the others.
    flat[2] *= DEPTH_FAULT
    gated = _pairs(flat, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert len(gated) == 6
    assert not [k for k in gated if 2 in k]  # every pair touching frame 2 is gone


def test_the_controls_must_run_above_the_production_rel_thresh():
    """Pins CONTROL_REL_THRESH the way the tilt is pinned: the default deletes the evidence.

    Same silent-deletion trap as the frustum gate, one layer down. A from-frame-2 pair carries
    sampled < expected - tol at the production tolerance, which the measurement reads as
    OCCLUDED — absent evidence, dropped from the denominator and from the collection — so the
    fault erases its own pairs. Anyone "restoring the default" here gets a red test rather than
    a quieter one.
    """
    depth, K, extr = _faulted_scene()
    assert len(_pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)) == 12

    at_default = _pairs(depth, K, extr)  # production default, rel_thresh=0.05
    assert len(at_default) == 9
    assert not [k for k in at_default if k[0] == 2]  # every FROM-frame-2 pair deleted
    assert len([k for k in at_default if k[1] == 2]) == 3  # INTO-frame-2 pairs survive


########################################
# Depth fault
########################################


def test_control_depth_scale_moves_the_depth_measurement_on_both_sides_of_frame_2():
    """x1.1 on frame 2's depth moves BOTH directions touching frame 2, by different closed forms.

    Three buckets, all asserted, because the ordered-pair loop gives the fault two distinct
    signatures and reporting only one of them would be false:
      INTO frame 2 (i != 2, j == 2): sampled is scaled, expected is clean, so rel = +0.1 exactly.
      FROM frame 2 (i == 2):         the source point is pushed 1.1x along its ray, so expected
                                     is scaled and sampled is clean: rel = 1/1.1 - 1 = -0.0909.
      Neither:                       unreachable by the fault, so ~0.
    """
    depth, K, extr = _faulted_scene()
    p = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert len(p) == 12
    into_2 = [v.median_rel_depth_error for (i, j), v in p.items() if j == 2]
    from_2 = [v.median_rel_depth_error for (i, j), v in p.items() if i == 2]
    clean = [v.median_rel_depth_error for (i, j), v in p.items() if 2 not in (i, j)]
    assert len(into_2) == 3 and len(from_2) == 3 and len(clean) == 6

    # Tolerances from the measurement, not from what passes. Both assertions are on the MEDIAN,
    # so the median is the deviation that sets the tolerance: 0.0014 into, 0.0012 from, against
    # the abs=0.01 asserted here — roughly 7x headroom. The worst single element runs wider
    # (0.0023 into, 0.0049 from), which is why the band is not tightened to the median figure.
    assert np.median(into_2) == pytest.approx(DEPTH_FAULT - 1.0, abs=0.01)
    assert np.median(from_2) == pytest.approx(1.0 / DEPTH_FAULT - 1.0, abs=0.01)
    assert np.max(np.abs(clean)) < 0.01  # measured 0.001593, pure resampling noise


def test_control_depth_scale_moves_parallax_only_on_the_source_side():
    """Parallax is pose geometry on the TARGET side only; on the source side a depth fault leaks.

    The measurement computes parallax from world points unprojected through the SOURCE frame's
    depth, so scaling that depth slides every point along its ray and changes the subtended
    angle. Scaling the TARGET frame's depth cannot reach it at all. Measured, for a x1.1 fault:
    into-frame-2 and untouched pairs move by exactly 0.0, from-frame-2 pairs move -9.05 to
    -9.15%, against the small-angle prediction 1/1.1 - 1 = -9.09%.

    This is a real property of the design, not a wart: it says a depth fault is NOT fully absent
    from the parallax column, so source-side parallax must not be read as depth-independent.
    """
    depth, K, extr = _scene()
    before = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    depth[2] *= DEPTH_FAULT
    after = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    shared = set(before) & set(after)
    # Assert the sample BEFORE looping over it: at the production tolerance this set is 9 and
    # the from-frame-2 arm below would silently have nothing in it.
    assert len(before) == len(after) == len(shared) == 12

    untouched = [k for k in shared if 2 not in k]
    into_2 = [k for k in shared if k[1] == 2]
    from_2 = [k for k in shared if k[0] == 2]
    assert len(untouched) == 6 and len(into_2) == 3 and len(from_2) == 3

    # Arm 1 — pairs the fault cannot reach. Exact-zero BY CONSTRUCTION (identical inputs to an
    # identical computation), so this arm pins determinism and nothing more. It is kept because
    # a non-zero here would mean the fault leaked across frames entirely, but it is not evidence
    # of separation on its own; arm 3 is what carries that.
    for key in untouched:
        assert after[key].median_parallax_deg == before[key].median_parallax_deg

    # Arm 2 — the arm with real content. The fault IS in frame 2 and these pairs read frame 2,
    # yet parallax never touches the target's depth, so they too are exactly unchanged.
    for key in into_2:
        assert after[key].median_parallax_deg == before[key].median_parallax_deg

    # Arm 3 — the leak. These MUST move, by the reciprocal of the injected scale.
    for key in from_2:
        ratio = after[key].median_parallax_deg / before[key].median_parallax_deg
        # rel=0.01 is justified by measurement: the worst pair sits 7e-4 from the prediction,
        # the small-angle approximation being exact only in the limit.
        assert ratio == pytest.approx(1.0 / DEPTH_FAULT, rel=0.01)


def test_control_injected_scale_has_a_closed_form_prediction():
    """r=0.1 predicts delta_d = 0.1*d exactly, and the ratio sits at 1 for a pure depth fault."""
    depth, K, extr = _faulted_scene()
    pairs = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert (1, 2) in pairs
    p = pairs[(1, 2)]
    predicted = depth_error_in_pixels(DEPTH_FAULT - 1.0, p.median_parallax_deg, FOCAL)
    measured = depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, FOCAL)
    # Both must be real numbers: None means the pair carries under a pixel of disparity, which
    # would make the comparison below vacuous rather than passing.
    assert predicted is not None and measured is not None
    # Pin the closed form itself, delta_d = r * f * alpha. Without this line the ratio below is
    # invariant to the bridge's functional form — a bridge that ignored the residual entirely
    # would still divide to 1.0.
    assert predicted == pytest.approx(0.1 * np.deg2rad(p.median_parallax_deg) * FOCAL, rel=1e-6)
    assert measured == pytest.approx(predicted, rel=0.3)
    # A pure depth fault: the pixel motion IS the depth motion, so the ratio sits at 1.
    assert measured / predicted == pytest.approx(1.0, abs=0.3)


########################################
# Pose fault
########################################


def test_control_pose_fault_drives_the_ratio_far_above_one():
    """Pixels move while depths stay mutually consistent — the >>1 signature.

    The fault is a real translation of camera 1 along world Y (see POSE_FAULT_DY for why that
    axis): the surface is invariant under it, so the depth channel is structurally blind, while
    every projection still shifts by f*dy/Z.
    """
    depth, K, extr = _scene()
    extr_bad = extr.copy()
    extr_bad[1, 1, 3] -= POSE_FAULT_DY  # camera 1's centre moves +dy in world Y

    # Measured pixel motion, and a check that it is the size the geometry says it is.
    shift_px = _reprojection_shift_px(depth, K, extr, extr_bad, 0, 1)
    expected_shift = FOCAL * POSE_FAULT_DY / float(np.median(depth[0]))
    assert shift_px == pytest.approx(expected_shift, rel=0.1)
    assert shift_px > 2.0  # a multi-pixel fault, not a rounding artefact

    pairs = _pairs(depth, K, extr_bad, rel_thresh=CONTROL_REL_THRESH)
    assert len(pairs) == 12
    p = pairs[(0, 1)]
    # The depth channel stays as quiet as an unperturbed pair: nothing here exceeds the
    # resampling floor the clean fixture already sits at.
    assert abs(p.median_rel_depth_error) < 0.01
    equiv = depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, FOCAL)
    assert equiv is not None and equiv < 0.05

    # Pin rho itself, not just "it is big". This is the headline number the design rests on:
    # one formula reads 0.98 for a pure depth fault and ~634 here, three orders apart.
    rho = shift_px / equiv
    assert rho == pytest.approx(634.0, rel=0.25)


########################################
# Appearance fault
########################################


def test_control_depth_measurement_never_sees_appearance():
    """Appearance faults cannot reach a measurement that never reads appearance.

    Asserted structurally rather than by injecting an exposure shift, because there is nowhere
    to inject one: compute_multiview_depth_confidence takes depth, intrinsics and extrinsics and
    no image argument at all. A runtime version would have to call it twice on identical inputs
    and would therefore measure determinism, not invariance. This assertion fails the moment
    someone gives the depth measurement an appearance channel, which is the event worth catching.
    """
    params = set(inspect.signature(compute_multiview_depth_confidence).parameters)
    assert params == {
        "depth", "intrinsics", "extrinsics", "depth_masks",
        "abs_thresh", "rel_thresh", "pair_gate", "collect", "device",
    }
    assert not [p for p in params if any(w in p for w in ("image", "rgb", "color", "colour"))]


def test_control_exposure_shift_is_invisible_to_photometric_too():
    """NCC is what buys this — a raw difference would flag it as error."""
    depth, K, extr = _scene(n=4)
    images = _texture(4)
    clean = compute_photometric_ncc(images, depth, K, extr, max_separation=2)
    assert clean["available"] and clean["n_pairs"] == 5
    ncc_clean = [r["photometric_ncc"] for r in clean["pairs"]]
    # A correlation worth being invariant about: measured 0.884-0.956 on this fixture.
    assert min(ncc_clean) > 0.5

    # Gain AND offset on one frame only — the asymmetric case, since a global rescale would
    # also survive a merely shift-invariant measure.
    shifted = images.copy()
    shifted[1] = shifted[1] * 1.6 + 30.0
    after = compute_photometric_ncc(shifted, depth, K, extr, max_separation=2)
    assert after["n_pairs"] == clean["n_pairs"]  # else the zip below misaligns and truncates
    exposure_delta = max(abs(a - b["photometric_ncc"]) for a, b in zip(ncc_clean, after["pairs"]))
    assert exposure_delta < 1e-9  # measured 3.3e-16

    # Without this half the test is decoration: an NCC hardwired to a constant would pass
    # everything above. A geometric fault of comparable size must move the same number.
    extr_bad = extr.copy()
    extr_bad[1, 1, 3] -= POSE_FAULT_DY
    faulted = compute_photometric_ncc(images, depth, K, extr_bad, max_separation=2)
    assert faulted["n_pairs"] == clean["n_pairs"]  # same guard, same reason
    pose_delta = max(abs(a - b["photometric_ncc"]) for a, b in zip(ncc_clean, faulted["pairs"]))
    assert pose_delta > 0.01  # measured 0.0277
    assert pose_delta > 1e6 * exposure_delta


########################################
# Observability floor and the separation axis
########################################


def test_control_forward_motion_is_the_direction_that_falls_under_the_floor():
    """At the SAME baseline magnitude, forward motion falls under the floor and strafing clears it.

    The magnitudes are matched deliberately. Comparing a small forward step against a large
    sideways one would only restate that a shorter baseline gives less parallax, which is true
    of any scene and says nothing about direction. Both scenes below use FLOOR_STEP, and both
    are read at their BEST-observed pair — asserting on the worst pair would be trivially
    satisfiable. Measured at step 0.1: forward best 0.428 px, strafe best 1.433 px, 3.3x apart
    with the floor sitting between them.
    """
    fwd_depth, fwd_K, fwd_extr = _scene(n=3, centers=[(0.0, 0.0, FLOOR_STEP * k) for k in range(3)])
    fwd = _pairs(fwd_depth, fwd_K, fwd_extr)
    assert len(fwd) == 6
    fwd_best = max(fwd.values(), key=lambda q: q.median_parallax_deg)
    fwd_px = np.deg2rad(fwd_best.median_parallax_deg) * FOCAL

    strafe_depth, strafe_K, strafe_extr = _scene(n=3, centers=[(FLOOR_STEP * k, 0.0, 0.0) for k in range(3)])
    strafe = _pairs(strafe_depth, strafe_K, strafe_extr)
    assert len(strafe) == 6
    strafe_best = max(strafe.values(), key=lambda q: q.median_parallax_deg)
    strafe_px = np.deg2rad(strafe_best.median_parallax_deg) * FOCAL

    # The direction claim: same baseline, several times the parallax.
    assert strafe_px / fwd_px > 3.0
    # The floor claim: the derived one-pixel-of-disparity threshold separates them, and the
    # bridge declines to answer on the forward side rather than emitting an infinity.
    assert fwd_px < 1.0 < strafe_px
    assert depth_error_in_pixels(0.05, fwd_best.median_parallax_deg, FOCAL) is None
    assert depth_error_in_pixels(0.05, strafe_best.median_parallax_deg, FOCAL) is not None


def test_control_separation_axis_has_teeth():
    """Injecting error that grows with frame gap must show as a positive rho."""
    depth, K, extr = _scene(n=6)
    for k in range(6):
        depth[k] *= 1.0 + 0.02 * k  # drift: each frame slightly more scaled than the last
    # Faults injected, so CONTROL_REL_THRESH: at the production default the occlusion branch
    # deletes the widest-gap pairs, which are exactly the ones carrying the signal (measured
    # 24 pairs instead of 30, and the ones lost are the most-drifted).
    p = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert len(p) == 30
    frame_seps = np.array([abs(i - j) for (i, j) in p], dtype=np.float64)
    errs = np.array([abs(v.median_rel_depth_error) for v in p.values()])
    assert stats.spearmanr(frame_seps, errs).statistic > 0.8
