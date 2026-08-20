"""Negative controls: each measurement must move under its own fault and stay still under others.

The report aggregates three measurements chosen because their DEPENDENCIES DIFFER — epipolar
reads poses, depth cross-view reads poses+depth, photometric reads poses+depth+appearance.
Attribution works only if that separation is real, so every claim below is a fault of known
magnitude and known location, asserted to land in exactly one channel.

Two things this file deliberately does NOT do:

  * It does not assert on an empty collection. Every test that loops asserts the pair count
    first, because the pair gate can remove a whole frame's pairs silently (see
    ``test_a_constant_depth_fixture_is_silently_gated_out``) and a loop over {} passes.
  * It does not divide an invented number by a measured one. The pose control perturbs real
    extrinsics and measures the pixel motion that perturbation actually produces.
"""

import inspect

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import compute_photometric_ncc, depth_error_in_pixels
from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence

FOCAL = 30.0
HW = 24
# The world surface is the plane Z = Z0 + TILT*X. TILT is the whole reason this fixture works;
# see _scene and test_a_constant_depth_fixture_is_silently_gated_out for why it is not zero.
Z0 = 4.0
TILT = 0.3


########################################
# Fixture
########################################


def _scene(n=4, hw=HW, centers=None, tilt=TILT):
    """N cameras viewing the world plane Z = Z0 + tilt*X, depth rendered exactly per camera.

    A SLANTED plane, not a fronto-parallel one, and that is load-bearing rather than cosmetic.
    A constant depth map makes near == far, so ``_frustum_world_aabbs`` produces a zero-thickness
    slab and ``_aabbs_overlap`` — which needs overlap on every axis — rejects any pair whose
    slabs sit at different depths. Scaling one frame's depth is exactly such a displacement, so
    on a flat fixture the depth-scale control has NO pairs to measure and every assertion over
    them passes on the empty set. Measured: the flat version yields 6 pairs instead of 12, with
    all 6 pairs touching the scaled frame gone. The tilt gives each frustum real depth extent,
    which is what a real scene has, and as a side effect spreads parallax over 3.3-10.1 deg
    instead of pinning every pixel at one angle.

    Rotations stay identity and the plane is independent of Y, which buys two exact properties
    the controls below rely on: depth is a function of the pixel's x ray-component alone, and
    the surface is invariant under camera translation along world Y.
    """
    K = np.array([[FOCAL, 0, hw / 2], [0, FOCAL, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    if centers is None:
        centers = [(0.25 * k, 0.0, 0.0) for k in range(n)]  # camera k strafing to x = +0.25k

    # World-to-camera: identity rotation, so t = -C.
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k, c in enumerate(centers):
        extr[k, :3, 3] = -np.asarray(c, np.float32)

    # Ray-plane intersection in closed form. Ray through pixel u is (a, b, 1) with
    # a = (u - cx)/f; substituting C + s*(a, b, 1) into Z = Z0 + tilt*X gives
    # s*(1 - tilt*a) = Z0 + tilt*Cx - Cz, and Z-depth equals s because the ray's z is 1.
    a = (np.arange(hw) - hw / 2) / FOCAL
    depth = np.stack(
        [
            np.tile(((Z0 + tilt * cx - cz) / (1.0 - tilt * a))[None, :], (hw, 1)).astype(np.float32)
            for cx, _, cz in centers
        ]
    )
    return depth, np.stack([K] * n), extr


def _pairs(depth, K, extr, **kw):
    """{(i, j): PairStats} from one collected multiview pass. CPU so the controls need no GPU."""
    out = {}
    compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out, **kw)
    return {(p.idx1, p.idx2): p for p in out["pairs"]}


def _texture(n, hw=HW, seed=0):
    """Smooth RGB with a little noise: enough structure to correlate, no aliasing under warp."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    base = 120 + 60 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    imgs = np.stack([np.stack([base + 10 * c for c in range(3)], -1)] * n).astype(np.float32)
    return imgs + rng.normal(0, 2.0, imgs.shape)


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
    """Nothing below means anything if the pair gate has quietly emptied the collection."""
    depth, K, extr = _scene(n=4)
    p = _pairs(depth, K, extr, rel_thresh=0.5)
    # ORDERED directions: the mv loop runs (i, j) and (j, i) separately, so 4 frames give 12.
    assert len(p) == 12
    assert min(v.n_pixels for v in p.values()) > 100
    # Real depth extent, which is what keeps the frusta from degenerating to slabs.
    assert depth[0].max() / depth[0].min() > 1.2
    # And real parallax spread, so no control is secretly evaluated at a single angle.
    par = [v.median_parallax_deg for v in p.values()]
    assert max(par) / min(par) > 2.0


def test_a_constant_depth_fixture_is_silently_gated_out():
    """Pins WHY _scene tilts the plane. A flat fixture loses pairs without raising anything.

    This is not a control; it is the recorded reason the fixture looks the way it does. If it
    ever fails the tilt may be dropped — until then, reverting _scene to a constant depth map
    would delete the depth-scale control's evidence while leaving it green.
    """
    depth, K, extr = _scene(n=4)
    flat = np.full_like(depth, Z0)
    assert len(_pairs(flat, K, extr, rel_thresh=0.5)) == 12  # a flat scene alone is fine

    # ...until one frame's depth moves, which slides its zero-thickness frustum off the others.
    flat[2] *= 1.1
    gated = _pairs(flat, K, extr, rel_thresh=0.5)
    assert len(gated) == 6
    assert not [k for k in gated if 2 in k]  # every pair touching frame 2 is gone


########################################
# Depth fault
########################################


def test_control_depth_scale_moves_the_depth_measurement_by_the_injected_amount():
    """x1.1 on frame 2's depth => median_rel_depth_error ~ +0.1 on pairs INTO frame 2, and only those."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    p = _pairs(depth, K, extr, rel_thresh=0.5)
    assert len(p) == 12
    into_2 = [v.median_rel_depth_error for (i, j), v in p.items() if j == 2]
    clean = [v.median_rel_depth_error for (i, j), v in p.items() if 2 not in (i, j)]
    assert len(into_2) == 3 and len(clean) == 6
    assert np.median(into_2) == pytest.approx(0.1, abs=0.03)
    assert np.max(np.abs(clean)) < 0.01


def test_control_depth_scale_does_not_move_the_parallax_angles():
    """Parallax is pose geometry; a depth scale must not change it materially."""
    depth, K, extr = _scene()
    before = _pairs(depth, K, extr, rel_thresh=0.5)
    depth[2] *= 1.1
    after = _pairs(depth, K, extr, rel_thresh=0.5)
    shared = set(before) & set(after)
    # Assert the sample BEFORE looping over it: on a flat fixture this set is 6, and the six
    # rows it would then check are exactly the ones the fault cannot reach.
    assert len(before) == len(after) == len(shared) == 12
    clean = [k for k in shared if 2 not in k]
    assert len(clean) == 6
    for key in clean:
        assert after[key].median_parallax_deg == pytest.approx(before[key].median_parallax_deg, abs=0.05)


def test_control_injected_scale_has_a_closed_form_prediction():
    """r=0.1 predicts delta_d = 0.1*d exactly, and the ratio sits at 1 for a pure depth fault."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    pairs = _pairs(depth, K, extr, rel_thresh=0.5)
    assert (1, 2) in pairs
    p = pairs[(1, 2)]
    predicted = depth_error_in_pixels(0.1, p.median_parallax_deg, FOCAL)
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

    The fault is a real translation of camera 1 along world Y. That axis is chosen because the
    plane Z = Z0 + TILT*X contains it, so the surface is INVARIANT under the perturbation: the
    depth channel is structurally blind to this fault, which is precisely the claim. Pixels are
    not blind to it — the same translation shifts every projection by f*dy/Z.
    """
    dy = 0.4
    depth, K, extr = _scene()
    extr_bad = extr.copy()
    extr_bad[1, 1, 3] -= dy  # camera 1's centre moves +dy in world Y

    # Measured pixel motion, and a check that it is the size the geometry says it is.
    shift_px = _reprojection_shift_px(depth, K, extr, extr_bad, 0, 1)
    assert shift_px == pytest.approx(FOCAL * dy / float(np.median(depth[0])), rel=0.1)
    assert shift_px > 2.0  # a multi-pixel fault, not a rounding artefact

    pairs = _pairs(depth, K, extr_bad, rel_thresh=0.5)
    assert len(pairs) == 12
    p = pairs[(0, 1)]
    # The depth channel stays as quiet as an unperturbed pair: nothing here exceeds the
    # resampling floor the clean fixture already sits at.
    assert abs(p.median_rel_depth_error) < 0.01
    equiv = depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, FOCAL)
    assert equiv is not None and equiv < 0.05
    # rho = measured / equiv. Asserted as a product so an exactly-zero residual reads as the
    # infinite ratio it is instead of raising.
    assert shift_px > 10.0 * equiv


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
    # A correlation worth being invariant about: measured 0.88-0.96 on this fixture.
    assert min(ncc_clean) > 0.5

    # Gain AND offset on one frame only — the asymmetric case, since a global rescale would
    # also survive a merely shift-invariant measure.
    shifted = images.copy()
    shifted[1] = shifted[1] * 1.6 + 30.0
    after = compute_photometric_ncc(shifted, depth, K, extr, max_separation=2)
    assert after["n_pairs"] == clean["n_pairs"]
    exposure_delta = max(abs(a - b["photometric_ncc"]) for a, b in zip(ncc_clean, after["pairs"]))
    assert exposure_delta < 1e-9

    # Without this half the test is decoration: an NCC hardwired to a constant would pass
    # everything above. A geometric fault of comparable size must move the same number.
    extr_bad = extr.copy()
    extr_bad[1, 1, 3] -= 0.4
    faulted = compute_photometric_ncc(images, depth, K, extr_bad, max_separation=2)
    pose_delta = max(abs(a - b["photometric_ncc"]) for a, b in zip(ncc_clean, faulted["pairs"]))
    assert pose_delta > 0.01
    assert pose_delta > 1e6 * exposure_delta


########################################
# Observability floor and the separation axis
########################################


def test_control_forward_motion_falls_under_one_pixel_of_disparity():
    """Pure forward motion drives perpendicular baseline to ~0 near the epipole."""
    depth, K, extr = _scene(n=3, centers=[(0.0, 0.0, 0.05 * k) for k in range(3)])
    p = _pairs(depth, K, extr)
    assert len(p) == 6
    # The BEST-observed pair, not the worst: asserting on the minimum would be trivially true
    # of any scene and would say nothing about the ones that carry the most parallax.
    best = max(p.values(), key=lambda q: q.median_parallax_deg)
    assert np.deg2rad(best.median_parallax_deg) * FOCAL < 1.0
    # And the bridge declines to answer rather than emitting an infinity.
    assert depth_error_in_pixels(0.05, best.median_parallax_deg, FOCAL) is None

    # Contrast, or the floor would just be reporting that this fixture is small: the same three
    # cameras strafing instead of advancing clear the floor and get a real answer.
    d2, k2, e2 = _scene(n=3)
    strafe = _pairs(d2, k2, e2)
    assert len(strafe) == 6
    worst_strafe = min(strafe.values(), key=lambda q: q.median_parallax_deg)
    assert np.deg2rad(worst_strafe.median_parallax_deg) * FOCAL > 1.0
    assert depth_error_in_pixels(0.05, worst_strafe.median_parallax_deg, FOCAL) is not None


def test_control_separation_axis_has_teeth():
    """Injecting error that grows with frame gap must show as a positive rho."""
    depth, K, extr = _scene(n=6)
    for k in range(6):
        depth[k] *= 1.0 + 0.02 * k  # drift: each frame slightly more scaled than the last
    p = _pairs(depth, K, extr, rel_thresh=0.9)
    assert len(p) == 30
    frame_seps = np.array([abs(i - j) for (i, j) in p], dtype=np.float64)
    errs = np.array([abs(v.median_rel_depth_error) for v in p.values()])
    assert stats.spearmanr(frame_seps, errs).statistic > 0.8
