"""Keyframe selection: which frames to keep, and the sampling methods that pick them.

Sampling methods are fps, uniform, and optical_flow. Capture-quality
measurement and the quality gate live in collab_splats.preproc.qa; decoding and
video metadata live in collab_splats.preproc.video; cv2 is used here for
in-memory image operations only (LK flow, histogram comparison, BGR->RGB).
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import cv2
import numpy as np
from tqdm.auto import tqdm

from collab_splats.preproc.qa import (
    _DEFAULT_BLUR_THRESHOLD,
    _analysis_gray,
    check_frame_quality,
    compute_blur_score,
)
from collab_splats.preproc.video import (
    _iter_frames,
    _iter_selected_frames,
    get_video_info,
)

logger = logging.getLogger(__name__)


########################################################################
# Constants
########################################################################

# Uniform sampling: when a target position fails the quality gate, consider at
# most this many frames outward (each side) as a usable substitute.
_VALID_PROBE_MAX = 3

# Optical-flow selection: combined score at or above this selects the frame.
_SELECT_THRESHOLD = 0.5
# Rotation (degrees) that saturates the motion score.
_ROTATION_THRESHOLD_DEG = 5.0

# Lucas-Kanade sparse flow parameters.
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
# Shi-Tomasi corner detection parameters for flow seed points.
_FEATURE_PARAMS = dict(maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)


########################################################################
# Optical-flow selector
########################################################################


def _combine_scores(
    disparity: float,
    histogram_similarity: float,
    min_disparity: float,
    rotation: float = 0.0,
) -> float:
    """Weighted motion+coverage score in [0, 1] from raw per-frame signals.

    Single home for the selection formula — the selector and
    viz.plot_disparity_sensitivity both call this, so they can't drift apart.
    """
    # Fixed motion/coverage weighting
    motion_weight, coverage_weight = 0.6, 0.4
    # Motion: max of normalised translation and rotation components
    translation_score = min(disparity / max(min_disparity, 1e-6), 1.0)
    rotation_score = min(rotation / _ROTATION_THRESHOLD_DEG, 1.0)
    motion_score = max(translation_score, rotation_score)
    # Coverage: inverse histogram correlation vs the last keyframe
    coverage_score = 1.0 - histogram_similarity
    total = motion_weight + coverage_weight
    return (motion_weight * motion_score + coverage_weight * coverage_score) / total


class OpticalFlowFrameSelector:
    """Streaming keyframe selector: motion (LK flow + rotation) and coverage scoring.

    Holds the reference keyframe between calls — score each candidate frame
    with score_frame(); promote selected frames with accept_frame(). Construct
    fresh per video.
    """

    def __init__(self, min_disparity: float = 50.0):
        self.min_disparity = min_disparity
        # Reference keyframe state, seeded on the first scored frame
        self.last_keyframe_gray: np.ndarray | None = None
        self.last_keyframe_pts: np.ndarray | None = None

    def score_frame(self, gray: np.ndarray) -> tuple[float, dict]:
        """Score a grayscale frame against the current keyframe.

        Returns (score in [0, 1], components dict with raw disparity /
        rotation / histogram_similarity). The first frame scores 1.0 and
        seeds the keyframe state.
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
        score = _combine_scores(disparity, hist_similarity, self.min_disparity, rotation=rotation)
        return score, {
            "disparity": disparity,
            "rotation": rotation,
            "histogram_similarity": hist_similarity,
        }

    def accept_frame(self, gray: np.ndarray) -> None:
        """Make the given grayscale frame the new reference keyframe."""
        self.last_keyframe_gray = gray.copy()
        self.last_keyframe_pts = cv2.goodFeaturesToTrack(gray, **_FEATURE_PARAMS)

    def _compute_flow(self, gray: np.ndarray):
        """LK flow from keyframe corners; (None, None) if under 10 inliers survive."""
        if self.last_keyframe_pts is None or len(self.last_keyframe_pts) == 0:
            return None, None
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_keyframe_gray, gray, self.last_keyframe_pts, None, **_LK_PARAMS
        )
        if curr_pts is None:
            return None, None
        good_prev = self.last_keyframe_pts[status == 1]
        good_curr = curr_pts[status == 1]
        # Too few inliers → tracking unreliable for motion estimation
        if len(good_prev) < 10:
            return None, None
        return good_prev, good_curr

    def _estimate_rotation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """Camera rotation angle (degrees) via RANSAC partial-affine fit."""
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
        """Histogram correlation vs the keyframe, clamped to [0, 1]."""
        h1 = cv2.calcHist([self.last_keyframe_gray], [0], None, [bins], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()
        return float(max(0.0, min(1.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))


########################################################################
# Sampling
########################################################################


def _progress_reporter(
    total: int,
    desc: str,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[Callable[[int], None], Callable[[], None]]:
    """Unified progress: forward to on_progress when given, else an internal tqdm bar.

    Returns (report(done), close()) — the single progress mechanism for all loops.
    """
    if on_progress is not None:
        return (lambda done: on_progress(done, total)), (lambda: None)
    bar = tqdm(total=total or None, desc=desc, unit="frame")
    return (lambda _done: bar.update(1)), bar.close


def _uniform_targets(total: int, n: int) -> list[int]:
    """N evenly-spaced source indices spanning the whole video (endpoint-anchored)."""
    if total <= 0 or n <= 0:
        return []
    # Cap at the source length, then dedup: rounding can collide when n approaches total
    n = min(n, total)
    return np.unique(np.linspace(0, total - 1, n).round().astype(int)).tolist()


def _fps_targets(total: int, native_fps: float, fps: float) -> list[int]:
    """Source indices at a constant wall-clock interval (stride-anchored).

    Deliberately not stretched to hit the last frame: for fps the spacing is the
    contract and the count falls out, the mirror of _uniform_targets.
    """
    if total <= 0 or fps <= 0:
        return []
    # Stride floors at 1 — a requested rate above the source rate cannot sample sub-frame
    step = max(1, int(round((native_fps or 30.0) / fps)))
    return list(range(0, total, step))


def sample_frames(
    video_path: str,
    *,
    method: str = "fps",
    fps: float | None = None,
    min_frames: int | None = None,
    max_frames: int | None = None,
    min_disparity: float = 50.0,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Select frames from a video for reconstruction.

    Each method has exactly one density knob; `max_frames` is the frame budget —
    the target count for "uniform", a ceiling for the other two.

    Methods:
        "fps": one frame every 1/`fps` seconds (stride-anchored), so the baseline
            between consecutive frames is fixed regardless of video length and the
            count floats. `min_frames`/`max_frames` bound that count: outside the
            band the targets are re-spread evenly over the WHOLE video (never
            truncated) and the effective fps is logged.
        "uniform": exactly `max_frames` evenly-spaced frames spanning the video
            (endpoint-anchored) — the count is the contract and spacing falls out.
        "optical_flow": motion (LK disparity + rotation) + coverage (histogram
            diversity) scoring; frames scoring >= 0.5 are selected. Uses
            `min_disparity`; `max_frames` caps the result.

    Both target-list methods decode in a single ffmpeg `select` pass and validate
    each position against the quality gate, substituting the sharpest usable
    neighbour so the count stays exact. `blur_threshold` tunes the gate for all
    three methods (0.0 disables the blur check).

    Returns (frames, records): RGB arrays and one dict per selected frame with at
    least frame_idx (SOURCE video index) and blur_score; optical_flow adds
    disparity, rotation, histogram_similarity, score, selected.

    Raises:
        ValueError: on an unknown method, or a knob that belongs to another method.
    """
    # Reject the wrong knob for the method rather than silently ignoring it — a
    # method whose behaviour depends on which kwargs happen to be set is the defect
    # this dispatch exists to remove. Validation runs before any IO.
    if method != "fps" and fps is not None:
        raise ValueError(f"method={method!r} takes no fps= — use method='fps' to sample at a rate")
    if method != "fps" and min_frames is not None:
        raise ValueError(
            f"method={method!r} takes no min_frames= — the floor only applies to method='fps', "
            "whose count floats with video length"
        )

    if method == "fps":
        if fps is None:
            raise ValueError("method='fps' requires fps= (samples per second)")
        return _sample_fps(
            video_path,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    if method == "uniform":
        if max_frames is None:
            raise ValueError("method='uniform' requires max_frames= (the frames to spread over the video)")
        return _sample_uniform(
            video_path,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    if method == "optical_flow":
        return _sample_optical_flow(
            video_path,
            max_frames=max_frames,
            min_disparity=min_disparity,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    raise ValueError(f"Unknown method: {method!r} (expected 'fps', 'uniform' or 'optical_flow')")


def _sample_positions(
    video_path: str,
    targets: list[int],
    *,
    total: int,
    w: int,
    h: int,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
    desc: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """Decode and quality-gate one frame per target position in a single ffmpeg pass.

    Shared body of the target-list samplers (_sample_fps, _sample_uniform). Each
    position gets a validation window of neighbouring frames; the sharpest
    gate-passing frame in the window wins, falling back to the sharpest frame when
    none passes, so the returned count stays exact.
    """
    if not targets:
        return [], []
    # Validation window: radius half the target spacing, capped, and kept under
    # spacing/2 so neighbouring windows never overlap (deterministic index map).
    spacing = targets[1] - targets[0] if len(targets) > 1 else total
    radius = min(max((spacing - 1) // 2, 0), _VALID_PROBE_MAX)
    # Per-target candidate indices (clamped to range, dedup); flatten to one
    # sorted set of frames to decode in a single pass.
    windows = [sorted({min(max(t + o, 0), total - 1) for o in range(-radius, radius + 1)}) for t in targets]
    wanted = sorted({i for win in windows for i in win})
    # One ffmpeg pass → frames keyed by source index (in-C decode, ~len(wanted)
    # frames reach Python).
    frame_by_idx = dict(zip(wanted, _iter_selected_frames(video_path, wanted, w, h)))
    report, close = _progress_reporter(len(targets), desc, on_progress)
    frames: list[np.ndarray] = []
    records: list[dict] = []
    try:
        # Per position, pick the sharpest gate-passing frame in its window; fall
        # back to the sharpest frame when none passes, so the count stays exact.
        for done, win in enumerate(windows):
            best = None  # ((usable, blur), idx, bgr) — prefer usable, then sharp
            for idx in win:
                bgr = frame_by_idx.get(idx)
                if bgr is None:
                    continue
                gray = _analysis_gray(bgr)
                blur = compute_blur_score(gray)
                usable, _ = check_frame_quality(gray, blur_threshold, blur_score=blur)
                key = (usable, blur)
                if best is None or key > best[0]:
                    best = (key, idx, bgr)
            if best is None:
                continue  # window fully dropped by ffmpeg (should not happen)
            key, idx, bgr = best
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            records.append({"frame_idx": idx, "blur_score": key[1]})
            report(done + 1)
    finally:
        close()
    return frames, records


def _sample_uniform(
    video_path: str,
    *,
    max_frames: int,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Exactly max_frames evenly-spaced frames spanning the whole video."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    targets = _uniform_targets(total, max_frames)
    return _sample_positions(
        video_path,
        targets,
        total=total,
        w=info["width"],
        h=info["height"],
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="Uniform sampling",
    )


def _sample_fps(
    video_path: str,
    *,
    fps: float,
    min_frames: int | None,
    max_frames: int | None,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """One frame every 1/fps seconds; re-spread if the count falls outside the band."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    native_fps = info["fps"] or 30.0
    targets = _fps_targets(total, native_fps, fps)
    # The requested rate yields a count that floats with video length, so clamp it into
    # [min_frames, max_frames] by re-spreading over the WHOLE video — never by truncating,
    # which would drop the tail of the scene and hand the reconstructor half a video.
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))
    if bounded != requested:
        targets = _uniform_targets(total, bounded)
        effective = native_fps * len(targets) / total
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            effective,
        )
    return _sample_positions(
        video_path,
        targets,
        total=total,
        w=info["width"],
        h=info["height"],
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="fps sampling",
    )


def _iter_scored_frames(
    video_path: str,
    selector: OpticalFlowFrameSelector,
    *,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
    desc: str,
) -> Iterator[tuple[int, np.ndarray, bool, dict, float, dict]]:
    """Yield (frame_idx, frame_bgr, selected, quality, score, components) per frame.

    quality is check_frame_quality's metrics dict. Single scoring loop shared by
    _sample_optical_flow and score_frames. Gate-rejected frames yield
    selected=False with score 0.0 and never reach the selector. Selected frames
    (score >= _SELECT_THRESHOLD) become the selector's new reference keyframe.
    """
    info = get_video_info(str(video_path))
    report, close = _progress_reporter(info["total_frames"], desc, on_progress)
    try:
        for idx, frame in enumerate(_iter_frames(video_path)):
            report(idx + 1)
            gray = _analysis_gray(frame)
            # Quality gate first: unusable frames never reach the selector
            ok, quality = check_frame_quality(gray, blur_threshold)
            if not ok:
                yield idx, frame, False, quality, 0.0, {}
                continue
            score, components = selector.score_frame(gray)
            selected = score >= _SELECT_THRESHOLD
            if selected:
                selector.accept_frame(gray)
            yield idx, frame, selected, quality, score, components
    finally:
        close()


def _sample_optical_flow(
    video_path: str,
    *,
    max_frames: int | None,
    min_disparity: float,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Optical-flow keyframe selection; keeps selected frames full-res RGB."""
    selector = OpticalFlowFrameSelector(min_disparity=min_disparity)
    frames: list[np.ndarray] = []
    records: list[dict] = []
    for idx, frame, selected, quality, score, comp in _iter_scored_frames(
        video_path,
        selector,
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="Optical flow selection",
    ):
        if not selected:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # frame_idx is the SOURCE video index (fixes old positional-index bug)
        records.append(
            {"frame_idx": idx, "blur_score": quality["blur_score"], "score": score, "selected": True, **comp}
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames, records


def score_frames(
    video_path: str,
    *,
    min_disparity: float = 50.0,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Score every frame without keeping pixel data — analysis/viz workflow.

    Returns one record per frame: frame_idx, blur_score, exposure_mean,
    exposure_std, reject_reason, disparity, rotation, histogram_similarity,
    score, selected.
    """
    selector = OpticalFlowFrameSelector(min_disparity=min_disparity)
    records: list[dict] = []
    for idx, _frame, selected, quality, score, comp in _iter_scored_frames(
        video_path,
        selector,
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="Scoring frames",
    ):
        records.append(
            {
                "frame_idx": idx,
                "blur_score": quality["blur_score"],
                "exposure_mean": quality["exposure_mean"],
                "exposure_std": quality["exposure_std"],
                "reject_reason": quality["reject_reason"],
                "score": score,
                "selected": selected,
                "disparity": comp.get("disparity", 0.0),
                "rotation": comp.get("rotation", 0.0),
                "histogram_similarity": comp.get("histogram_similarity", 1.0),
            }
        )
    return records
