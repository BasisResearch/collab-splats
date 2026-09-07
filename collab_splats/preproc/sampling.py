"""
Keyframe selection: which frames to keep, and the three methods that pick them.

- sample_fps, sample_uniform and sample_optical_flow each take a quality report from
  preproc.qa and filter it through filter_frame_quality
- filter_frame_quality is the one place a threshold meets the report: qa measures,
  this module decides
- context_indices builds the constant-rate grid both the fps sampler and the VDA
  context stream select on
- decoding and video metadata live in preproc.video
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.stats import median_abs_deviation

from collab_splats.preproc.qa import analysis_gray
from collab_splats.preproc.video import get_video_info, iter_frames
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)


########################################################################
# Quality filter — the one place a threshold meets the report.
# qa measures; sampling decides. Nothing writes a verdict into the report.
########################################################################


def filter_frame_quality(
    report: dict,
    *,
    sharpness_k: float = 2.0,
    max_clipped_frac: float = 0.25,
) -> np.ndarray:
    """
    Per-frame usability mask over a quality report's photometry columns.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        sharpness_k: robust z-score cut on log(laplacian); larger keeps more.
        max_clipped_frac: ceiling on clipped_low_frac + clipped_high_frac.

    Returns:
        (N,) bool, True = keep, indexed by source frame index.
    """
    f = report["frames"]
    lap = np.asarray(f["laplacian"], dtype=float)
    if lap.size == 0:
        return np.zeros(0, dtype=bool)

    # Sharpness is relative: laplacian variance scales with resolution, texture and
    # content, so the cut is a robust z-score on the log rather than an absolute value.
    log_lap = np.log(np.clip(lap, 1e-6, None))
    center = float(np.median(log_lap))
    spread = float(median_abs_deviation(log_lap, scale="normal"))

    # A zero MAD is two different situations, both safe
    # - constant column: every frame equally sharp, nothing to cut
    # - non-constant column: over half the frames sit exactly on the median
    # - the cut then falls on the median itself — right answer, and divides by nothing
    if log_lap.min() == log_lap.max():
        sharp = np.ones_like(lap, dtype=bool)
    else:
        sharp = log_lap >= center - sharpness_k * spread

    # Clipping is absolute: a pixel at 0 or 255 recorded nothing recoverable.
    clipped = np.asarray(f["clipped_low_frac"], dtype=float) + np.asarray(f["clipped_high_frac"], dtype=float)

    return sharp & (clipped <= max_clipped_frac)


########################################################################
# Optical-flow selector
########################################################################


class OpticalFlowFrameSelector:
    """
    Streaming keyframe selector: motion (LK flow + rotation) and coverage scoring.

    - holds the reference keyframe between calls — score with score_frame(), promote with
      accept_frame()
    - construct one per video
    - no cv2/kornia/open3d equivalent exists for the streaming "is this frame different enough
      from the last one I kept" policy
    - the pieces it is built from are all cv2's: goodFeaturesToTrack, calcOpticalFlowPyrLK,
      estimateAffinePartial2D, calcHist/compareHist
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        *,
        rotation_threshold_deg: float = 5.0,
        lk_params: dict | None = None,
        feature_params: dict | None = None,
    ):
        self.min_disparity = min_disparity
        self.rotation_threshold_deg = rotation_threshold_deg

        # Built here, not as defaults
        # - a mutable dict default is shared across every instance in the process
        # - Python trap, not a style call
        # - params below are Lucas-Kanade sparse flow
        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Shi-Tomasi corners, the seed points that flow tracks.
        self.feature_params = feature_params or dict(maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)

        # Reference keyframe state, seeded on the first scored frame
        self.last_keyframe_gray: np.ndarray | None = None
        self.last_keyframe_pts: np.ndarray | None = None

    def score_frame(self, gray: np.ndarray) -> tuple[float, dict]:
        """
        Score a grayscale frame against the current keyframe.

        - the first frame scores 1.0 and seeds the keyframe state

        Args:
            gray: (h, w) uint8 single-channel candidate frame.

        Returns:
            (score in [0, 1], components), where components is
            {'disparity': median LK pixel motion since the last kept frame,
            'rotation': in-plane rotation against it in degrees,
            'histogram_similarity': intensity-histogram correlation with it}.
        """
        if self.last_keyframe_gray is None:
            self.accept_frame(gray)
            return 1.0, {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}

        # Motion signals from LK flow of keyframe corners into this frame
        disparity, rotation = 0.0, 0.0
        prev_pts, curr_pts = self._compute_flow(gray)
        if prev_pts is not None:
            disparity = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
            rotation = self._estimate_rotation(prev_pts, curr_pts)

        # Coverage signal: histogram correlation vs the keyframe
        hist_similarity = self._hist_similarity(gray)

        components = {
            "disparity": disparity,
            "rotation": rotation,
            "histogram_similarity": hist_similarity,
        }

        return self.combine(disparity, hist_similarity, rotation=rotation), components

    def combine(self, disparity: float, histogram_similarity: float, *, rotation: float = 0.0) -> float:
        """
        Weighted motion + coverage score in [0, 1]; >= select_threshold selects.

        Args:
            disparity: median LK pixel motion against the keyframe.
            histogram_similarity: histogram correlation against the keyframe, 1.0 = identical.
            rotation: in-plane rotation against the keyframe, degrees.

        Returns:
            Score in [0, 1], fixed 0.6 motion / 0.4 coverage weighting.
        """
        # Fixed motion/coverage weighting
        motion_weight, coverage_weight = 0.6, 0.4

        # Motion: max of the normalized translation and rotation components
        translation_score = min(disparity / max(self.min_disparity, 1e-6), 1.0)
        rotation_score = min(rotation / self.rotation_threshold_deg, 1.0)
        motion_score = max(translation_score, rotation_score)

        # Coverage: inverse histogram correlation vs the last keyframe
        coverage_score = 1.0 - histogram_similarity

        return (motion_weight * motion_score + coverage_weight * coverage_score) / (motion_weight + coverage_weight)

    def accept_frame(self, gray: np.ndarray) -> None:
        """
        Make the given grayscale frame the new reference keyframe.

        Args:
            gray: (h, w) uint8 single-channel frame to promote.
        """
        self.last_keyframe_gray = gray.copy()
        self.last_keyframe_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)

    def _compute_flow(self, gray: np.ndarray):
        """
        LK flow from keyframe corners; (None, None) if under 10 inliers survive.
        """
        if self.last_keyframe_pts is None or len(self.last_keyframe_pts) == 0:
            return None, None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_keyframe_gray, gray, self.last_keyframe_pts, None, **self.lk_params
        )
        if curr_pts is None:
            return None, None

        good_prev = self.last_keyframe_pts[status == 1]
        good_curr = curr_pts[status == 1]

        # Too few inliers -> tracking unreliable for motion estimation
        if len(good_prev) < 10:
            return None, None

        return good_prev, good_curr

    def _estimate_rotation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """
        Camera rotation angle (degrees) via RANSAC partial-affine fit.
        """
        if len(prev_pts) < 4:
            return 0.0

        try:
            M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
            if M is None:
                return 0.0
            return float(np.abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        except Exception:
            logger.debug("Rotation estimation failed", exc_info=True)
            return 0.0

    def _hist_similarity(self, gray: np.ndarray, bins: int = 64) -> float:
        """
        Histogram correlation vs the keyframe, clamped to [0, 1].
        """
        h1 = cv2.calcHist([self.last_keyframe_gray], [0], None, [bins], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [bins], [0, 256])

        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()

        return float(max(0.0, min(1.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))


########################################################################
# Selection grid — constant-rate source indices, shared by the keyframe
# sampler and the VDA context stream so the two agree by construction.
########################################################################


def context_indices(video_path: str | Path, *, target_fps: float, info: dict | None = None) -> list[int]:
    """
    Source frame indices on a constant-rate grid at target_fps.

    Args:
        video_path: source video.
        target_fps: grid rate; must be positive.
        info: a get_video_info dict, to hoist the probe out of a loop.

    Returns:
        Ascending source frame indices. Stride floors at 1 — a rate above the
        source rate cannot sample sub-frame.
    """
    # target_fps is the contract here, so an absent one is a config error, not a default
    if target_fps is None or target_fps <= 0:
        raise ValueError(f"context_indices needs a positive target_fps, got {target_fps!r}")

    # Reuse a caller's probe when given — a fresh one costs a container parse
    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]
    if total == 0:
        return []

    native_fps = info["fps"] or 30.0
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, total, step))


########################################################################
# Report-driven samplers
########################################################################


def _eligible(report: dict, *, quality: dict | None) -> np.ndarray:
    """
    Source frame indices a sampler may select from.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        quality: overrides for filter_frame_quality's thresholds.

    Returns:
        (M,) int64, ascending.
    """
    pool = np.flatnonzero(filter_frame_quality(report, **(quality or {})))

    if pool.size == 0:
        raise ValueError("no eligible frames: the quality filter rejected every frame")

    return pool


def _decode_selection(
    video_path: str,
    chosen: Sequence[int],
    *,
    report: dict,
    on_progress,
    desc: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Decode exactly the selected frames, in one pass, as RGB.

    Args:
        video_path: source video.
        chosen: ascending source frame indices.
        report: the quality report the blur_score column is read from.
        on_progress: optional (done, total) callback.
        desc: progress-bar label.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    # One ffmpeg select pass over exactly the frames we keep
    decoded = dict(iter_frames(video_path, indices=list(chosen)))

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx in progress(chosen, total=len(chosen), desc=desc, on_progress=on_progress):
        bgr = decoded.get(idx)
        if bgr is None:
            continue  # ffmpeg dropped the frame (should not happen)

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # blur_score comes from the report, not a recompute — same measurement,
        # and it is the column the record has always carried.
        records.append({"frame_idx": int(idx), "blur_score": float(laplacian[idx])})

    return frames, records


def sample_uniform(
    video_path: str,
    *,
    max_frames: int,
    report: dict,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Exactly max_frames evenly-spaced picks from the eligible pool.

    Args:
        video_path: source video.
        max_frames: how many frames to keep — the COUNT is the contract.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    if max_frames <= 0:
        return [], []

    pool = _eligible(report, quality=quality)

    # Spacing is even in POOL index, not in time: budget is not spent inside
    # footage the filter just condemned.
    if pool.size <= max_frames:
        logger.warning("max_frames=%d but only %d eligible frames; keeping the whole pool", max_frames, pool.size)
        chosen = pool.tolist()
    else:
        chosen = pool[np.linspace(0, pool.size - 1, max_frames).round().astype(int)].tolist()

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="Uniform sampling")


def sample_fps(
    video_path: str,
    *,
    fps: float,
    report: dict,
    min_frames: int | None = None,
    max_frames: int | None = None,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    One frame every 1/fps seconds, each snapped to the nearest eligible frame.

    Args:
        video_path: source video.
        fps: target rate — the SPACING is the contract, the count floats.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        min_frames: floor; a count below it re-spreads over the whole video.
        max_frames: ceiling; a count above it re-spreads over the whole video.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    # fps is the contract here, so an absent one is a config error, not a default
    if fps is None or fps <= 0:
        raise ValueError(f"sample_fps needs a positive fps, got {fps!r}")

    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []

    pool = _eligible(report, quality=quality)

    # One source of truth for the stride: a context grid built at this same rate
    # contains these targets by construction, not by coincidence
    targets = context_indices(video_path, target_fps=fps, info=info)

    # Clamp the floating count into the band by re-spreading over the pool, never by
    # truncating — truncation would hand the reconstructor half a scene
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = pool[np.linspace(0, pool.size - 1, min(bounded, pool.size)).round().astype(int)].tolist()
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            (info["fps"] or 30.0) * len(targets) / total,
        )

    # Snap each target to the nearest eligible frame, then dedup
    # - two targets either side of an excised stretch can snap to the same survivor
    # - clip to [1, pool.size - 1] keeps pos - 1 and pos in range for a one-frame pool
    # - there they collapse onto one index, so the choice is the same either way
    pos = np.clip(np.searchsorted(pool, targets), 1, pool.size - 1)
    left, right = pool[pos - 1], pool[pos]
    snapped = np.where(np.abs(targets - left) <= np.abs(right - targets), left, right)
    chosen = sorted(set(snapped.tolist()))

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="fps sampling")


def sample_optical_flow(
    video_path: str,
    *,
    report: dict,
    max_frames: int | None = None,
    min_disparity: float = 50.0,
    select_threshold: float = 0.5,
    rotation_threshold_deg: float = 5.0,
    lk_params: dict | None = None,
    feature_params: dict | None = None,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Keyframes by motion (LK disparity + rotation) and coverage (histogram diversity).

    - runs its own decode pass: LK disparity is measured against a MOVING keyframe reference,
      which the report's fixed-stride pairs cannot supply
    - blur comes from the report, never a recompute

    Args:
        video_path: source video.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        max_frames: cap on the result; None keeps every selected frame.
        min_disparity: pixel motion scoring a full translation component.
        select_threshold: score at or above which a frame is selected.
        rotation_threshold_deg: rotation scoring a full rotation component.
        lk_params: overrides for cv2.calcOpticalFlowPyrLK.
        feature_params: overrides for cv2.goodFeaturesToTrack.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their per-frame score rows.
    """
    info = get_video_info(str(video_path))
    pool = set(_eligible(report, quality=quality).tolist())
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        rotation_threshold_deg=rotation_threshold_deg,
        lk_params=lk_params,
        feature_params=feature_params,
    )

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx, bgr in progress(
        iter_frames(video_path),
        total=info["total_frames"],
        desc="Optical flow selection",
        on_progress=on_progress,
    ):
        # Pool gate first: an ineligible frame never reaches the selector, so it
        # cannot become the reference the next frames are scored against.
        if idx not in pool:
            continue

        score, components = selector.score_frame(analysis_gray(bgr))
        if score < select_threshold:
            continue

        selector.accept_frame(analysis_gray(bgr))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # frame_idx is the SOURCE video index
        records.append(
            {
                "frame_idx": int(idx),
                "blur_score": float(laplacian[idx]),
                "score": score,
                "selected": True,
                **components,
            }
        )

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames, records
