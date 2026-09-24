"""
Keyframe selection from a quality report.

- filter_frame_quality turns report columns into a keep mask; qa never decides
- sample_fps fixes spacing, sample_uniform fixes count, sample_optical_flow follows motion
- every sampler picks from frames the mask keeps; sample_fps's rescue is the one exception
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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


class _OpticalFlowSelector:
    """
    Streaming keyframe scorer: LK motion and histogram coverage against the last keyframe.

    - score() a candidate; accept() it to make it the new reference
    """

    def __init__(
        self,
        *,
        min_disparity: float,
        rotation_threshold_deg: float,
        lk_window: int = 21,
        lk_levels: int = 3,
        max_corners: int = 1000,
        min_inliers: int = 10,
        hist_bins: int = 64,
        motion_weight: float = 0.6,
    ):
        """
        Hold the scoring thresholds; no keyframe until the first accept().

        Args:
            min_disparity: mean LK displacement in pixels that scores a full translation.
            rotation_threshold_deg: in-plane rotation in degrees that scores a full rotation.
            lk_window: LK search window side in pixels.
            lk_levels: LK pyramid levels above the base image.
            max_corners: Shi-Tomasi corner cap per keyframe.
            min_inliers: fewer tracked corners than this scores zero motion.
            hist_bins: intensity-histogram bins for the coverage term.
            motion_weight: weight of motion in the score; coverage gets the rest.
        """
        self.min_disparity = min_disparity
        self.rotation_threshold_deg = rotation_threshold_deg
        self.lk_window = lk_window
        self.lk_levels = lk_levels
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.hist_bins = hist_bins
        self.motion_weight = motion_weight

        # Reference keyframe, set by accept()
        self.keyframe: np.ndarray | None = None
        self.keyframe_pts: np.ndarray | None = None

    def accept(self, gray: np.ndarray) -> None:
        """
        Make gray the reference keyframe and seed its Shi-Tomasi corners.
        """
        self.keyframe = gray.copy()
        self.keyframe_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=8, blockSize=7
        )

    def score(self, gray: np.ndarray) -> tuple[float, dict]:
        """
        Score in [0, 1] against the keyframe, plus its components; 1.0 before any accept.
        """
        if self.keyframe is None:
            return 1.0, {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}

        # Motion: mean LK displacement and in-plane rotation of the keyframe corners
        disparity, rotation = 0.0, 0.0
        prev_pts, curr_pts = self._flow(gray)
        if prev_pts is not None:
            disparity = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
            rotation = self._rotation(prev_pts, curr_pts)

        similarity = self._hist_similarity(gray)
        components = {"disparity": disparity, "rotation": rotation, "histogram_similarity": similarity}
        return self._combine(disparity, rotation, similarity), components

    def _combine(self, disparity: float, rotation: float, similarity: float) -> float:
        """
        Weighted max(translation, rotation) motion plus (1 - similarity) coverage.
        """
        translation_score = min(disparity / max(self.min_disparity, 1e-6), 1.0)
        rotation_score = min(rotation / self.rotation_threshold_deg, 1.0)
        motion = max(translation_score, rotation_score)
        return self.motion_weight * motion + (1.0 - self.motion_weight) * (1.0 - similarity)

    def _flow(self, gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        LK-tracked (keyframe, current) corners; (None, None) below min_inliers.
        """
        if self.keyframe_pts is None or len(self.keyframe_pts) == 0:
            return None, None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.keyframe,
            gray,
            self.keyframe_pts,
            None,
            winSize=(self.lk_window, self.lk_window),
            maxLevel=self.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if curr_pts is None:
            return None, None

        # (N, 1, 2) points indexed by the (N, 1) status mask -> (M, 2)
        good = status == 1
        if good.sum() < self.min_inliers:
            return None, None

        return self.keyframe_pts[good], curr_pts[good]

    def _rotation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """
        In-plane rotation in degrees from a RANSAC partial-affine fit; 0.0 when unfittable.
        """
        if len(prev_pts) < 4:
            return 0.0

        try:
            M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        except cv2.error:
            return 0.0

        return 0.0 if M is None else float(np.abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))

    def _hist_similarity(self, gray: np.ndarray) -> float:
        """
        Intensity-histogram correlation with the keyframe, clamped to [0, 1].
        """
        h1 = cv2.calcHist([self.keyframe], [0], None, [self.hist_bins], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [self.hist_bins], [0, 256])
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()
        return float(max(0.0, min(1.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))


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


def _spread(pool: np.ndarray, n: int) -> list[int]:
    """
    n picks evenly spaced by position in pool; all of pool when n >= its size.

    Args:
        pool: ascending eligible source indices.
        n: how many to pick.

    Returns:
        Ascending source indices.
    """
    if n >= pool.size:
        return pool.tolist()
    return pool[np.linspace(0, pool.size - 1, n).round().astype(int)].tolist()


def _sharpest(candidates: np.ndarray, target: int, laplacian: np.ndarray) -> int:
    """
    Candidate with the highest laplacian; ties go to the one nearest target.

    Args:
        candidates: source indices to choose from, non-empty.
        target: the grid index the slot is centered on.
        laplacian: the report's per-frame laplacian column.

    Returns:
        One source index.
    """
    rank = np.lexsort((np.abs(candidates - target), -laplacian[candidates]))
    return int(candidates[rank[0]])


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

    # One decode pass over exactly the frames we keep
    decoded = dict(iter_frames(video_path, indices=list(chosen)))
    missing = sorted(set(chosen) - decoded.keys())
    if missing:
        raise ValueError(
            f"decode of {video_path} skipped frames {missing[:5]}; "
            "the report may not match this video; delete it to recompute"
        )

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx in progress(chosen, total=len(chosen), desc=desc, on_progress=on_progress):
        frames.append(cv2.cvtColor(decoded[idx], cv2.COLOR_BGR2RGB))

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

    Raises:
        TypeError: quality names a key filter_frame_quality does not take.
        ValueError: no eligible frames, or the decode skipped a selected frame.
    """
    if max_frames <= 0:
        return [], []

    pool = _eligible(report, quality=quality)

    # Spacing is even in pool index, not time: no budget spent in condemned footage
    if pool.size <= max_frames:
        logger.warning("max_frames=%d but only %d eligible frames; keeping the whole pool", max_frames, pool.size)
    chosen = _spread(pool, max_frames)

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="Uniform sampling")


def sample_fps(
    video_path: str,
    *,
    fps: float,
    report: dict,
    min_frames: int | None = None,
    max_frames: int | None = None,
    quality: dict | None = None,
    on_empty_slot: str = "rescue",
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    One frame every 1/fps seconds, each the sharpest eligible frame in its slot.

    - the slot is +/- half the target spacing, so picks stay within half a period of the grid
    - ties break to the frame nearest the target, keeping exact spacing where sharpness is flat
    - a slot with no eligible frame is handled by on_empty_slot; picks never leave the slot

    Args:
        video_path: source video.
        fps: target rate — the SPACING is the contract, the count floats.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        min_frames: floor; a count below it re-spreads over the whole video.
        max_frames: ceiling; a count above it re-spreads over the whole video.
        quality: overrides for filter_frame_quality's thresholds.
        on_empty_slot: "rescue" keeps an all-ineligible slot's sharpest frame; "drop" skips it.
        on_progress: optional (done, total) callback.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.

    Raises:
        TypeError: quality names a key filter_frame_quality does not take.
        ValueError: fps is not positive, on_empty_slot is unknown, no frame is eligible,
            or the decode skipped a selected frame.
    """
    # fps is the contract here, so an absent one is a config error, not a default
    if fps is None or fps <= 0:
        raise ValueError(f"sample_fps needs a positive fps, got {fps!r}")

    # Unknown empty-slot policy is a config error
    if on_empty_slot not in ("rescue", "drop"):
        raise ValueError(f"on_empty_slot must be 'rescue' or 'drop', got {on_empty_slot!r}")

    native_fps = get_video_info(str(video_path))["fps"]
    pool = _eligible(report, quality=quality)
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    # Span is the report's length: metadata frame count can overcount, the report is what decoded
    total = laplacian.size

    # Constant-rate grid at fps; stride floors at 1
    targets = list(range(0, total, max(1, int(round(native_fps / fps)))))

    # Clamp the count into [min_frames, max_frames] by re-spreading, never truncating
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = _spread(pool, bounded)
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            native_fps * len(targets) / total,
        )

    # Sharpest eligible frame per slot, ties to the nearest target
    # - the gate is video-wide, so nearest-in-time alone picks an arbitrary survivor
    targets = np.asarray(targets)
    half = max(int(np.median(np.diff(targets)) // 2), 1) if targets.size > 1 else 1
    lo = np.searchsorted(pool, targets - half, side="left")
    hi = np.searchsorted(pool, targets + half, side="right")

    chosen = []
    for target, start, stop in zip(targets.tolist(), lo.tolist(), hi.tolist()):
        if start < stop:
            chosen.append(_sharpest(pool[start:stop], target, laplacian))
        elif on_empty_slot == "rescue":
            # Slot holds no eligible frame: keep its sharpest frame anyway
            window = np.arange(max(target - half, 0), min(target + half + 1, laplacian.size))
            chosen.append(_sharpest(window, target, laplacian))

    # Two targets either side of an excised stretch can land on the same survivor
    chosen = sorted(set(chosen))

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="fps sampling")


def sample_optical_flow(
    video_path: str,
    *,
    report: dict,
    max_frames: int | None = None,
    min_disparity: float = 50.0,
    select_threshold: float = 0.5,
    rotation_threshold_deg: float = 5.0,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Keyframes by motion (LK disparity + rotation) and coverage (histogram diversity).

    - runs its own decode pass: LK disparity is computed against a MOVING keyframe reference,
      which the report's fixed-stride pairs cannot supply
    - blur comes from the report, never a recompute

    Args:
        video_path: source video.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        max_frames: cap on the result; None keeps every selected frame.
        min_disparity: pixel motion scoring a full translation component.
        select_threshold: score at or above which a frame is selected.
        rotation_threshold_deg: rotation scoring a full rotation component.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and one record per kept frame:
        {frame_idx, blur_score, score, disparity, rotation, histogram_similarity}.

    Raises:
        TypeError: quality names a key filter_frame_quality does not take.
        ValueError: no eligible frames.
    """
    info = get_video_info(str(video_path))
    pool = set(_eligible(report, quality=quality).tolist())
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    selector = _OpticalFlowSelector(min_disparity=min_disparity, rotation_threshold_deg=rotation_threshold_deg)

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

        # One gray per frame; a frame is accepted once, when it is selected
        gray = analysis_gray(bgr)
        score, components = selector.score(gray)
        if score < select_threshold:
            continue

        selector.accept(gray)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        records.append({"frame_idx": int(idx), "blur_score": float(laplacian[idx]), "score": score, **components})

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames, records
