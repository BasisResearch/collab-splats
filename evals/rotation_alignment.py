"""Compare reconstructed rotations against an external reference whose camera frame differs.

An external attitude reference — GoPro CORI, a phone's ARKit stream, a robot's IMU — reports
orientation in *its* body frame, not the reconstruction's camera frame.  Writing that
relationship out, a reconstruction pose is

    R_est = R_world . R_true . R_camera

`R_world` is an unknown global frame change, and every trajectory metric already absorbs it
(evo's Umeyama alignment solves exactly that).  `R_camera` is an unknown *constant* offset
between the two camera conventions, and it does **not** absorb: it survives into relative
poses as a conjugation.  Feeding unfixed poses to RPE therefore reports a large rotation
error for a perfect reconstruction.  `fit_camera_offset` solves it; applying the result makes
relative rotations comparable.

Any fitted quantity can be wrong because the fit failed rather than because the
reconstruction is bad, so this module also provides a quantity that needs **no** fit at all —
`invariant_turn_angle_error_deg`.  A rotation's *angle* survives conjugation, so comparing
turn magnitudes depends on neither unknown.  It is both the honest way to choose between
reference conventions and a hard lower bound on the fitted error, which makes it a usable
self-check: land far above the floor and the fit failed.  That check is not decorative — it
caught two successive wrong fits when this code was written.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

########################################################################
# Constants
########################################################################

# The offset search is a coarse sweep of SO(3) followed by local polish of the best few.
# The objective is non-convex AND non-smooth (a median), so local descent alone lands in a
# basin that depends entirely on where it started: seeding only from the identity and the
# axis half- and quarter-turns left a 5.6 deg residual on a synthetic case whose true minimum
# is 0 deg — an error large enough to be mistaken for a bad reconstruction.  512 samples cost
# ~0.15 s and reduce that to 1e-10 deg.  Fixed random_state so results are reproducible.
_COARSE_SAMPLES = 512
_COARSE_SEED = 0
_POLISH_COUNT = 3


########################################################################
# Relative rotations
########################################################################


def relative_rotations(rotations: np.ndarray) -> np.ndarray:
    """Consecutive relative rotations R[i]^T @ R[i+1] of an (N, 3, 3) sequence."""
    return np.einsum("nij,njk->nik", rotations[:-1].transpose(0, 2, 1), rotations[1:])


def _relative_angles_rad(rotations: np.ndarray) -> np.ndarray:
    """Turn angle of each consecutive relative rotation, in radians."""
    rotvecs = Rotation.from_matrix(relative_rotations(rotations)).as_rotvec()
    return np.linalg.norm(rotvecs, axis=1)


def invariant_turn_angle_error_deg(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-pair |turn angle difference| in degrees — invariant to BOTH unknown alignments.

    A relative rotation's angle survives conjugation, so this depends on neither the world
    alignment nor the camera-frame offset and requires no fitting whatsoever.  Use it to
    choose between candidate reference conventions (a choice made by fitting would be
    circular) and as the floor below which `relative_rotation_error_deg` cannot fall.
    """
    return np.degrees(np.abs(_relative_angles_rad(est) - _relative_angles_rad(ref)))


########################################################################
# Camera-frame offset
########################################################################


def _offset_residual_deg(rotvec: np.ndarray, est: np.ndarray, ref: np.ndarray) -> float:
    """Median relative-rotation disagreement in degrees for camera offset exp(rotvec)."""
    offset = Rotation.from_rotvec(rotvec).as_matrix()
    # Conjugate the estimate's relative rotations into the reference's camera frame, then
    # measure what is left over against the reference's own relative rotations.
    conjugated = np.einsum("ij,njk,kl->nil", offset.T, relative_rotations(est), offset)
    residual = np.einsum("nij,njk->nik", conjugated.transpose(0, 2, 1), relative_rotations(ref))
    angles = np.linalg.norm(Rotation.from_matrix(residual).as_rotvec(), axis=1)
    return float(np.degrees(np.median(angles)))


def fit_camera_offset(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Solve the constant camera-frame offset by optimising the reported residual directly.

    A closed-form Kabsch fit over the relative-rotation *axes* was tried first and left a
    7.8 deg residual against a 0.29 deg invariant floor.  The axis cloud was not degenerate
    (singular values 11.6 / 8.1 / 7.0), so the closed form was not rank-limited — it was
    minimising an axis proxy rather than the quantity actually being reported.  Optimising
    the residual itself, from a coarse sweep of SO(3) rather than a handful of fixed seeds,
    reaches 0.58 deg on the same data and recovers a synthetic offset to 1e-10 deg.

    Args:
        est: (N, 3, 3) reconstruction camera-to-world rotations.
        ref: (N, 3, 3) reference camera-to-world rotations, same frames and order.

    Returns:
        (3, 3) offset to right-multiply onto `est`.
    """
    if est.shape != ref.shape:
        raise ValueError(f"pose count mismatch: est {est.shape} vs ref {ref.shape}")
    # Coarse sweep: evaluate the objective directly across SO(3) to locate the right basin.
    seeds = Rotation.random(_COARSE_SAMPLES, random_state=_COARSE_SEED).as_rotvec()
    coarse = np.array([_offset_residual_deg(seed, est, ref) for seed in seeds])
    # Polish the best few: the coarse winner is only within a few degrees, and a second and
    # third basin occasionally overtake the first once each is descended properly.
    best = min(
        (
            minimize(
                _offset_residual_deg,
                seeds[i],
                args=(est, ref),
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000},
            )
            for i in np.argsort(coarse)[:_POLISH_COUNT]
        ),
        key=lambda result: result.fun,
    )
    return Rotation.from_rotvec(best.x).as_matrix()


def relative_rotation_error_deg(
    est: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair relative-rotation error in degrees, after solving the camera-frame offset.

    Returns:
        (angles_deg (N-1,), offset (3, 3)) — the offset is returned so callers can apply it
        to the poses they write out and so it can be recorded alongside the numbers it
        produced.
    """
    offset = fit_camera_offset(est, ref)
    corrected = np.einsum("nij,jk->nik", est, offset)
    residual = np.einsum(
        "nij,njk->nik", relative_rotations(corrected).transpose(0, 2, 1), relative_rotations(ref)
    )
    angles = np.degrees(np.linalg.norm(Rotation.from_matrix(residual).as_rotvec(), axis=1))
    return angles, offset
