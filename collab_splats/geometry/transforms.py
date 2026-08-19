"""Pure-numpy camera geometry utilities shared across the pipeline.

Conventions:
  OpenCV camera axes:  X right, Y down,  Z forward  (COLMAP, VGGT-X, BA)
  OpenGL camera axes:  X right, Y up,    Z backward  (nerfstudio, splats)
"""

from __future__ import annotations

import numpy as np

########################################################################
########## Constants ###################################################
########################################################################

# Camera axis convention flip (OpenCV ↔ OpenGL). Self-inverse: applying
# twice returns to original. diag(1, -1, -1, 1).
OPENGL_TO_OPENCV: np.ndarray = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)

########################################################################
########## Geometry helpers ############################################
########################################################################


def extrinsics_to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    """Append [0,0,0,1] row to convert (N,3,4)→(N,4,4) or (3,4)→(4,4).

    Output dtype matches input dtype.
    """
    single = extrinsics.ndim == 2  # (3,4) → (4,4)
    if single:
        extrinsics = extrinsics[np.newaxis]  # (1,3,4)
    n = extrinsics.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=extrinsics.dtype), (n, 1, 1))  # (N,1,4)
    out = np.concatenate([extrinsics, bottom], axis=1)  # (N,4,4)
    return out[0] if single else out


def invert_poses(poses: np.ndarray) -> np.ndarray:
    """Closed-form SE3 inverse: (...,4,4) → (...,4,4).

    Works on any leading batch shape: (4,4), (N,4,4), (B,N,4,4).
    Uses R^T, -R^T@t — numerically exact for valid rotation matrices and
    faster than np.linalg.inv. Assumes poses are valid rigid-body transforms.
    """
    R = poses[..., :3, :3]
    t = poses[..., :3, 3:]
    R_inv = np.swapaxes(R, -1, -2)  # R^T
    t_inv = -(R_inv @ t)  # -R^T t
    out = np.zeros_like(poses)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:] = t_inv
    out[..., 3, 3] = 1.0
    return out


def extract_intrinsics(K: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from a (3,3) camera intrinsics matrix."""
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


########################################################################
########## Intrinsics estimation #######################################
########################################################################


def _compute_weighted_median(values: np.ndarray, weights: np.ndarray, max_n: int = 50_000) -> float | None:
    """Confidence-weighted median, subsampled above ``max_n`` with a seeded RNG.

    Textbook definition — sort by value, walk the cumulative weight, return the value
    at half the total mass.  Prior art for using one to reduce per-pixel focal
    estimates: github.com/PolyCam/LoGeR @ 5d7c1a7, ``run_loger.py:167``.  Returns
    ``None`` for an empty input, or for weights carrying no positive mass, so the
    caller can raise rather than invent a value.

    Values must be finite, and callers filter them before calling.  A non-finite entry
    does not poison the result visibly, it skews it: ``np.argsort`` sorts ``+inf`` and
    ``NaN`` to the tail (biasing the result upward) and ``-inf`` to the head (biasing
    it downward), so either way the return is a plausible finite number that a
    downstream ``np.isfinite`` check waves through.
    """
    if len(values) == 0:
        return None

    # A weighted median needs a full argsort, and the pooled per-pixel population is
    # H*W*N — 76.5M values at 300 frames, 255M at the 1000-frame sequences the LoGeR
    # backend exists for.  The cap bounds that; the fixed seed keeps it reproducible.
    if len(values) > max_n:
        idx = np.random.default_rng(42).choice(len(values), max_n, replace=False)
        values, weights = values[idx], weights[idx]

    # Sort by value, then walk the cumulative weight to the halfway mass.
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumw = np.cumsum(weights, dtype=np.float64)

    # All-zero weights carry no mass to bisect; searchsorted would return index 0 and
    # hand back the smallest value as if it were an estimate. Report "no estimate".
    if cumw[-1] <= 0:
        return None
    return float(values[np.searchsorted(cumw, cumw[-1] / 2.0)])


def estimate_intrinsics_from_points(
    local_points: np.ndarray, conf: np.ndarray, conf_threshold: float = 0.1
) -> np.ndarray:
    """Fit one shared pinhole K to a camera-frame pointmap by confidence-weighted median.

    This is a **fit**, not a readout: the pointmap is not guaranteed to be consistent
    with any single pinhole camera, so the returned K is the best shared pinhole
    explanation of it rather than a recovered ground truth.  Callers that need to know
    how good that explanation is should measure the reprojection residual.

    For backends whose model emits no intrinsics head.  Invert the pinhole model at
    every pixel — ``u_c = fx * X / Z``, so ``fx = u_c * Z / X`` — and reduce the pooled
    per-pixel estimates with a confidence-weighted median.  One K is returned for the
    whole batch, which is correct when every frame comes from the same physical camera
    at the same resolution.

    Prior art for the same approach: github.com/PolyCam/LoGeR @ 5d7c1a7,
    ``run_loger.py``, ``estimate_focal_lengths`` at :206 over ``_focal_from_frame`` at
    :180.  That fork also has ``_snap_square_pixels`` at :195, which we deliberately do
    **not** do — it would merge fx and fy, and a caller that rescales the two axes
    separately then un-merges the average incorrectly.

    The *vendored* fork recovers a focal too, by a different method, and is cited here so
    the PolyCam reference above is not mistaken for the only upstream precedent:
    github.com/Junyi42/LoGeR @ 7685b7a calls dust3r's ``estimate_focal_knowing_depth``
    with ``focal_mode="weiszfeld"`` and ``pp = (W // 2, H // 2)``
    (``eval/relpose/launch.py:528-534``) — one focal for both axes, unweighted IRLS rather
    than a confidence-weighted median.  It appears in the eval scripts only; the demo path
    fits nothing and hardcodes a 60 degree FOV (``loger/utils/viser_utils.py:445-449``).
    Measured on the LoGeR parity fixture, that estimator lands within 0.29% of this one
    and reproduces the model's own points slightly *worse* (see open item 3 of
    ``docs/superpowers/specs/2026-08-13-loger-feedforward-backend-design.md``).

    Args:
        local_points: (N, H, W, 3) camera-frame points, channel 2 being depth.  Note
            that a pointmap head is free to emit a per-pixel ray field not constrained
            to any pinhole K — LoGeR's, for instance, is built as ``cat([xy * z, z])``
            (github.com/Junyi42/LoGeR @ 7685b7a, ``loger/models/pi3.py:772-775``) —
            which is the reason this is an approximation.
        conf: (N, H, W) or (N, H, W, 1) per-pixel confidence, **already activated into
            [0, 1]**.  ``conf_threshold`` compares against a probability, so passing raw
            logits does not merely shift the gate — a head whose logits are all negative
            (LoGeR's measured range is -4.257..-2.019) admits *no* pixels at all and the
            fit raises.
        conf_threshold: minimum confidence for a pixel to contribute.  0.1 is a
            reasonable floor for a calibrated head and is deliberately NOT tuned to any
            one backend, but an uncalibrated head can put its whole band underneath it:
            LoGeR's measured post-sigmoid range is [0.0140, 0.1172], where 0.1 is the
            92nd percentile and keeps only 7.9% of pixels.  Callers wrapping an
            uncalibrated head should measure their own band and pass an explicit value
            rather than inherit this default.

    Returns:
        (3, 3) float32 K, shared across frames, centre-principal by construction.

    Raises:
        RuntimeError: if too few pixels survive to fit either focal.  There is no
            fallback focal on purpose.
    """
    n, h, w, _ = local_points.shape
    if conf.ndim == 4:
        conf = conf.squeeze(-1)

    # Centred pixel grid.  This line is why the function returns K and not (fx, fy):
    # every per-pixel focal below is conditioned on cx=(W-1)/2, cy=(H-1)/2, so the
    # principal point is already decided here and must not be re-chosen by a caller.
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
    )

    # Invert the pinhole model per pixel: u_c = fx * X / Z  =>  fx = u_c * Z / X.
    x, y, z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    valid = (z > 1e-3) & (np.abs(x) > 1e-6) & (np.abs(y) > 1e-6) & (conf > conf_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        fx_per_pixel = uu * z / x
        fy_per_pixel = vv * z / y

    fx_vals, fy_vals = fx_per_pixel[valid], fy_per_pixel[valid]
    weights = conf[valid]

    # Sanity bounds before the median, derived from field of view: f = 0.1 * W is a
    # ~157 degree horizontal FOV and f = 10 * W is ~6 degrees.  Real cameras live well
    # inside that; values outside it are degenerate inversions from pixels near the
    # principal axis, where X or Y is small enough that u_c * Z / X explodes.
    #
    # These bounds are also what enforce _compute_weighted_median's finite precondition,
    # and that is not incidental: the `valid` mask above cannot do it, because inf passes
    # `z > 1e-3`.  A surviving non-finite value would not poison the median visibly, it
    # would skew it — argsort sorts +inf and NaN to the tail (biasing the focal upward)
    # and -inf to the head (biasing it downward), and `u_c * Z / X` produces -inf as
    # readily as +inf as X approaches zero from below.  Either way the result stays
    # finite enough for the np.isfinite check below to wave it through.  BOTH bounds are
    # load-bearing: the upper rejects +inf, the lower rejects -inf, and NaN fails both.
    # Do not loosen either to a one-sided test without adding an explicit isfinite mask.
    ok_fx = (fx_vals > w * 0.1) & (fx_vals < w * 10)
    ok_fy = (fy_vals > h * 0.1) & (fy_vals < h * 10)
    fx = _compute_weighted_median(fx_vals[ok_fx], weights[ok_fx])
    fy = _compute_weighted_median(fy_vals[ok_fy], weights[ok_fy])

    # Fail loudly.  Upstream falls back to 1.2 * max(W, H); we do not, because a
    # plausible-but-wrong K fails silently all the way through to the mesh.
    if fx is None or fy is None or not np.isfinite(fx) or not np.isfinite(fy) or fx <= 0 or fy <= 0:
        raise RuntimeError(
            f"Pinhole intrinsics fit failed over {n} frames: "
            f"{int(valid.sum())}/{valid.size} pixels passed the validity mask "
            f"({int((conf > conf_threshold).sum())} cleared conf_threshold={conf_threshold}), "
            f"of which {int(ok_fx.sum())} survived the fx bounds and {int(ok_fy.sum())} the fy bounds "
            f"(fx={fx}, fy={fy}). No fallback focal is applied by design."
        )

    # fx and fy stay distinct.  Callers whose preprocessing rounds the two axes
    # independently (LoGeR's does, to multiples of 14) produce genuinely non-square
    # pixels, and that anisotropy belongs in K rather than being averaged away.
    return np.array(
        [[fx, 0.0, (w - 1) / 2.0],
         [0.0, fy, (h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a single 3x3 rotation matrix in degrees (trace formula)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix R such that R @ src ≈ dst.

    Args:
        src: (3,) unit vector to rotate from.
        dst: (3,) unit vector to rotate to.
    Returns:
        (3, 3) rotation matrix. Identity if src ≈ dst or antiparallel fallback.
    """
    # Normalize inputs to ensure unit vectors
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    # Compute rotation axis via cross product
    axis = np.cross(src, dst)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        # Parallel (identity) or antiparallel (180° rotation around arbitrary perp axis)
        if np.dot(src, dst) > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues for 180°: R = 2 * axis @ axis.T - I
        return -np.eye(3) + 2 * np.outer(axis, axis)

    # General case: Rodrigues' rotation formula
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(src, dst), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


########################################################################
########## Point-set alignment #########################################
########################################################################


def umeyama_se3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form SE(3) alignment via SVD (no scale).

    Args:
        source: (M, 3) points in source frame.
        target: (M, 3) corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (4, 4) float32 homogeneous T such that target ≈ T @ source.
    """
    M = source.shape[0]
    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return np.eye(4, dtype=np.float32)
    w = w / w_sum
    src = source.astype(np.float64)
    tgt = target.astype(np.float64)
    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt
    H = (src_c * w[:, None]).T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_tgt - R @ mu_src
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment via Umeyama (with scale).

    Args:
        source: (M, 3) float32/64 points in source frame.
        target: (M, 3) float32/64 corresponding points in target frame.
        weights: optional (M,) non-negative weights; uniform if None.

    Returns:
        (s, R, t): float scale, (3,3) float32 rotation, (3,) float32 translation
                   such that target ≈ s * R @ source + t.
    """
    M = source.shape[0]
    if M < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    w = np.ones(M, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    if w_sum < 1e-9:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    w = w / w_sum

    src = source.astype(np.float64)
    tgt = target.astype(np.float64)

    mu_src = (w[:, None] * src).sum(axis=0)
    mu_tgt = (w[:, None] * tgt).sum(axis=0)

    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    scale_src = float(np.sqrt((w * (src_c**2).sum(axis=1)).sum()))
    scale_tgt = float(np.sqrt((w * (tgt_c**2).sum(axis=1)).sum()))
    s = scale_tgt / scale_src if scale_src > 1e-9 else 1.0

    H = (src_c * s * w[:, None]).T @ tgt_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = (Vt.T @ D @ U.T).astype(np.float32)

    t = (mu_tgt - s * R @ mu_src).astype(np.float32)
    return s, R, t
