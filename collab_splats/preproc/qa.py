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
# Constants — gate-only. Report functions take tuning as keyword args.
########################################################################

# Analysis frames are downscaled to this width before scoring — bounds LK flow
# and Laplacian cost regardless of source resolution.
_ANALYSIS_WIDTH = 480

# Quality gate: Laplacian variance below this = blurred. Sharp indoor video
# sits well above 100; heavy motion blur drops below 50.
_DEFAULT_BLUR_THRESHOLD = 50.0
# Exposure bounds: mean outside this range = blown out; std below = no contrast.
_EXPOSURE_MEAN_RANGE = (20.0, 235.0)
_EXPOSURE_MIN_STD = 10.0


########################################################################
# Frame quality gate — legacy policy, not part of the report
########################################################################


def compute_blur_score(gray: np.ndarray) -> float:
    """Sharpness as Laplacian variance — higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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


def _analysis_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Grayscale copy downscaled to _ANALYSIS_WIDTH for scoring."""
    scale = min(1.0, _ANALYSIS_WIDTH / frame_bgr.shape[1])
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame_bgr
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


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
    """Photometric measurements for one BGR frame: blur and exposure together."""
    # Exposure reads the NATIVE-resolution gray. Downscaling averages scattered
    # saturated pixels out of existence, so clipping fractions taken from a
    # resized frame read 0.0 no matter how blown out the capture actually was.
    native_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exposure = compute_exposure(native_gray)

    # Blur reads the 480 px analysis gray. blur_effect costs 62 ms at 1024 px
    # against 13.8 ms at 480 px, and the score barely moves across that range
    # (0.1659 -> 0.1671), so the downscale is close to free.
    blur = compute_blur(_analysis_gray(bgr))

    return {**blur, **exposure}
