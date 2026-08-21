"""Video capture quality: how good is the source footage, per frame and per pair.

Measurement only — nothing here selects, rejects, or ranks frames on its own.
check_frame_quality is the one exception, and it is a gate the sampler calls,
not a decision this module makes.
"""

import logging

import cv2
import numpy as np
from skimage.measure import blur_effect

logger = logging.getLogger(__name__)


########################################################################
# Shared by the gate and the report
########################################################################

# Analysis frames are downscaled to this width before scoring — bounds LK flow
# and Laplacian cost regardless of source resolution.
_ANALYSIS_WIDTH = 480


def compute_blur_score(gray: np.ndarray) -> float:
    """Sharpness as Laplacian variance — higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _analysis_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Grayscale copy downscaled to _ANALYSIS_WIDTH for scoring."""
    scale = min(1.0, _ANALYSIS_WIDTH / frame_bgr.shape[1])
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame_bgr
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


########################################################################
# Frame quality gate — legacy policy, not part of the report.
# Its thresholds stay module constants; report functions take tuning as
# keyword arguments instead.
########################################################################

# Quality gate: Laplacian variance below this = blurred. Sharp indoor video
# sits well above 100; heavy motion blur drops below 50.
_DEFAULT_BLUR_THRESHOLD = 50.0
# Exposure bounds: mean outside this range = blown out; std below = no contrast.
_EXPOSURE_MEAN_RANGE = (20.0, 235.0)
_EXPOSURE_MIN_STD = 10.0


def check_frame_quality(
    gray: np.ndarray,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    blur_score: float | None = None,
) -> tuple[bool, dict]:
    """Quality gate: is the frame sharp enough and reasonably exposed?

    Returns (ok, metrics) where metrics holds blur_score, exposure_mean,
    exposure_std, and reject_reason (None | "blur" | "exposure").
    blur_score: pass a precomputed value to skip the Laplacian recompute.
    """
    if blur_score is None:
        blur_score = compute_blur_score(gray)
    mean, std = float(gray.mean()), float(gray.std())
    lo, hi = _EXPOSURE_MEAN_RANGE
    # Blur checked first — the first failing check names the reason
    reason = None
    if blur_score < blur_threshold:
        reason = "blur"
    elif not (lo <= mean <= hi) or std < _EXPOSURE_MIN_STD:
        reason = "exposure"
    metrics = {
        "blur_score": blur_score,
        "exposure_mean": mean,
        "exposure_std": std,
        "reject_reason": reason,
    }
    return reason is None, metrics


########################################################################
# Per frame
########################################################################


def compute_blur(gray: np.ndarray, *, h_size: int = 11) -> dict:
    """Blur measured two ways: Crete-Roffet perceptual blur and Laplacian variance.

    blur is [0, 1] and higher means blurrier; laplacian is unbounded and higher
    means sharper. They run in opposite directions on purpose. blur saturates at
    1.0 on any frame with little high-frequency content to destroy — a flat
    field, a smooth gradient and a single perfectly sharp edge all score exactly
    1.0 — so a high blur beside a high laplacian means detail is sparse, not
    that the frame is soft. That is why laplacian ships next to it, not instead.

    Args:
        gray: single-channel frame.
        h_size: width of the re-blur kernel Crete-Roffet compares against.
            Larger values report less blur; 11 is skimage's own default.
    """
    # A colour frame is a caller mistake, not a measurement: skimage reads the
    # channel axis as spatial and returns nan while cv2.Laplacian returns a
    # perfectly plausible number, so the row would read as a failed capture.
    if gray.ndim != 2:
        raise ValueError(f"compute_blur expects a 2-D single-channel frame, got shape {gray.shape}")

    # Crete-Roffet re-blurs the image and measures how little changes. A frame
    # that is already blurred barely moves, so its score rises toward 1.
    perceptual = float(blur_effect(gray, h_size=h_size))

    # Laplacian variance reuses the frame-selection gate's own metric verbatim,
    # so sharpness has exactly one implementation in the repo.
    return {"blur": perceptual, "laplacian": compute_blur_score(gray)}


def compute_exposure(gray: np.ndarray) -> dict:
    """Brightness distribution plus the fraction of pixels pinned at either end."""
    return {
        # Mean and median together: they separate when a small bright region
        # (a window, a lamp) drags the mean while most of the scene stays dark.
        "exposure_mean": float(gray.mean()),
        "exposure_median": float(np.median(gray)),
        # Contrast. A low std is a flat, textureless frame regardless of brightness.
        "exposure_std": float(gray.std()),
        # Clipped pixels are destroyed data, not merely dark or bright data:
        # 0 and 255 are the two values where the sensor recorded nothing recoverable.
        "clipped_low_frac": float((gray == 0).mean()),
        "clipped_high_frac": float((gray == 255).mean()),
    }


def compute_frame_quality(bgr: np.ndarray) -> dict:
    """Photometric measurements for one BGR frame: blur and exposure together.

    The two halves read the frame at different resolutions on purpose, and one
    of them costs something: blur is measured at _ANALYSIS_WIDTH, so **the blur
    column is not comparable across videos whose source width straddles it**.
    Wider sources are downscaled further and read sharper — measured on
    data/tutorial (1080 wide), blur is 0.2128 at 480 px against 0.2772 at
    1024 px, a 30% spread on identical frames. Videos wider than 480 px are
    mutually comparable; anything narrower is measured natively and is not.
    laplacian and every exposure column are unaffected, being native-resolution.
    """
    # Exposure reads the NATIVE-resolution gray. Downscaling averages scattered
    # saturated pixels out of existence, so clipping fractions taken from a
    # resized frame read 0.0 no matter how blown out the capture actually was.
    native_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exposure = compute_exposure(native_gray)

    # Blur reads the 480 px analysis gray, which buys throughput and pays for it
    # in the comparability noted above. Measured on one 1920x1080 tutorial frame,
    # blur_effect runs 300.4 ms natively against 39.4 ms at 853x480 — 7.6x, or
    # 12 minutes against 94 seconds over that video's 2388 frames. Cheap in time,
    # not free in value.
    blur = compute_blur(_analysis_gray(bgr))

    return {**blur, **exposure}


########################################################################
# Per pair
########################################################################


def match_orb(gray_a: np.ndarray, gray_b: np.ndarray, *, n_features: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """ORB keypoints matched mutually between two grayscale frames as Nx2 float32 arrays.

    crossCheck makes both sides injective, which is what the downstream RANSAC
    wants, but it bounds nothing about whether the two frames show the same
    scene: mutual-best still returns a full set of matches on unrelated frames.
    Measured on two independent noise images, 373 matches at a median
    displacement of 92 px, versus 539 matches at 17 px for a true 17 px shift.
    So a scene cut reads as large confident motion rather than as a failure,
    and neither the match count nor a nan reveals it. What separates them is
    descriptor distance — median Hamming 80 against 32 — which this function
    does not currently return.
    """
    # Detect and describe each frame independently — no shared state, so the
    # measurement never depends on which frames were selected before this pair.
    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    # A featureless frame yields no descriptors at all. Return empty rather than
    # raise: zero matches is a fact about the video, not an error.
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if desc_a is None or desc_b is None:
        return empty

    # ORB descriptors are binary, hence Hamming distance. crossCheck keeps only
    # mutual best matches, which removes the need for a Lowe ratio test.
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
    if not matches:
        return empty

    # Pull the pixel coordinates behind each match into two aligned Nx2 arrays
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)
    return pts_a, pts_b


def compute_translation(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Median match displacement in pixels — how far image content moved between the pair.

    Pixels of whatever grid match_orb was handed. The report feeds it
    _analysis_gray output, so the shipped column is _ANALYSIS_WIDTH pixels, and
    **it does not convert to source pixels by scaling**: ORB detects different
    keypoints at different resolutions, so the ratio is not the resize factor.
    Measured on data/tutorial (1920x1080, factor 2.25), native-over-analysis is
    2.37 on one pair and 3.28 on another. Comparable within a report, not
    across videos of differing width — the same caveat compute_frame_quality
    carries for blur, for the same reason.
    """
    # nan, not 0.0: with no matches the displacement is unknown, and 0.0 would
    # read as "the camera held perfectly still", the opposite conclusion.
    if len(pts_a) == 0:
        return float("nan")

    # Median over per-match displacement, so a handful of bad matches cannot
    # drag the number the way a mean would.
    return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))


def compute_parallax(pts_a: np.ndarray, pts_b: np.ndarray, *, ransac_thresh_px: float = 3.0) -> float:
    """One minus the homography/fundamental inlier ratio — how far the pair departs from a plane.

    A homography explains rotation-only motion and planar scenes exactly, so a
    ratio near 1 (parallax near 0) means the pair carries no depth information.
    Read it alongside translation: a flat scene under real translation also
    reads 0.

    Args:
        pts_a, pts_b: corresponding Nx2 points, as match_orb returns them.
        ransac_thresh_px: RANSAC inlier threshold, in whatever grid the points
            came from — the analysis grid for the report, same as
            translation_px. A first-order lever, not a detail: measured over 99
            tutorial pairs, 1.0 against 3.0 moves parallax by 0.19 on average
            and reorders the pairs (Spearman 0.708), while 3.0 against 5.0
            barely does (0.053, 0.945). Loosening it lets a homography explain
            more, so parallax falls monotonically.

    Returns nan below 8 correspondences. Eight is the linear 8-point algorithm's
    minimum: MAGSAC's 7-point solver does return an F at exactly 7, and OpenCV
    raises cv2.error at 6 or fewer, so the guard sets the floor and heads off
    that crash in one step.
    """
    if len(pts_a) < 8:
        return float("nan")

    # Fit both models to the same correspondences. H can only explain a plane or
    # a pure rotation; F can additionally explain translation through depth, so
    # the gap between their inlier counts IS the depth information in the pair.
    _, h_inliers = cv2.findHomography(pts_a, pts_b, cv2.USAC_MAGSAC, ransac_thresh_px)
    _, f_inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, ransac_thresh_px)
    n_h = int(h_inliers.sum()) if h_inliers is not None else 0
    n_f = int(f_inliers.sum()) if f_inliers is not None else 0

    # No F inliers means the pair is unexplained by any two-view geometry
    if n_f == 0:
        return float("nan")

    # min() guards the case where H outfits F on a degenerate pair, which would
    # otherwise push the complement negative.
    return float(1.0 - min(1.0, n_h / n_f))
