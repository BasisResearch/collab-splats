"""
Keyframe selection: which frames to keep, and the three methods that pick them.

sample_fps, sample_uniform and sample_optical_flow each take a quality report
from preproc.qa and filter it through filter_frame_quality — the one place a
threshold meets the report. qa measures; this module decides. Decoding and
video metadata live in preproc.video.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Callable

import cv2
import numpy as np

from collab_splats.preproc.qa import analysis_gray
from collab_splats.preproc.video import context_indices, get_video_info, iter_frames
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)


########################################################################
# Quality filter — the one place a threshold meets the report.
# qa measures; sampling decides. Nothing writes a verdict into the report.
########################################################################


def filter_frame_quality(
    report: dict,
    *,
    laplacian_min: float = 50.0,
    exposure_mean_range: tuple[float, float] = (20.0, 235.0),
    exposure_min_std: float = 10.0,
    blur_max: float | None = None,
) -> np.ndarray:
    """
    Per-frame usability mask over a quality report's photometry columns.

    - Returns one bool per frame, True = usable. Positive polarity, matching the
      name: a filter_* reports what survives.
    - Indexed by frame index directly — compute_video_quality enumerates every
      frame, so frame_idx is contiguous 0..N-1.
    - blur_max is OFF by default. Crete-Roffet `blur` saturates at 1.0 on any
      low-detail frame (a flat field and a single sharp edge both score 1.0), so
      thresholding it discards sharp frames of plain surfaces. `laplacian` is the
      sharpness gate; `blur` is there for a caller who knows the trap.

    Args:
        laplacian_min: Laplacian variance below which a frame is soft.
        exposure_mean_range: brightness band; outside is crushed or blown.
        exposure_min_std: contrast floor; below it the frame is flat.
        blur_max: optional Crete-Roffet ceiling, higher = blurrier.
    """
    f = report["frames"]
    lap = np.asarray(f["laplacian"], dtype=float)
    mean = np.asarray(f["exposure_mean"], dtype=float)
    std = np.asarray(f["exposure_std"], dtype=float)
    lo, hi = exposure_mean_range

    # Sharp enough: Laplacian variance, not Crete-Roffet blur (which saturates)
    sharp = lap >= laplacian_min

    # Exposed usably: inside the brightness band AND carrying some contrast
    exposed = (mean >= lo) & (mean <= hi) & (std >= exposure_min_std)

    usable = sharp & exposed

    # Optional perceptual-blur ceiling, off unless the caller asks for it
    if blur_max is not None:
        usable &= np.asarray(f["blur"], dtype=float) <= blur_max

    return usable


########################################################################
# Optical-flow selector
########################################################################


class OpticalFlowFrameSelector:
    """
    Streaming keyframe selector: motion (LK flow + rotation) and coverage scoring.

    Holds the reference keyframe between calls — score each candidate with
    score_frame(), promote selected frames with accept_frame(). Construct fresh
    per video.

    No cv2/kornia/open3d equivalent exists for the streaming "is this frame
    different enough from the last one I kept" policy; the pieces it is built
    from (goodFeaturesToTrack, calcOpticalFlowPyrLK, estimateAffinePartial2D,
    calcHist/compareHist) are all cv2's.
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

        # Built here, not as defaults: a mutable dict default is shared across
        # every instance in the process, which is a Python trap, not a style call.
        # Lucas-Kanade sparse flow.
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

        Returns (score in [0, 1], components). The first frame scores 1.0 and
        seeds the keyframe state. Components:

            disparity            median LK pixel motion since last kept frame
            rotation             in-plane rotation vs last kept frame, degrees
            histogram_similarity intensity-histogram correlation with last kept frame
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
        """
        # Fixed motion/coverage weighting
        motion_weight, coverage_weight = 0.6, 0.4

        # Motion: max of the normalised translation and rotation components
        translation_score = min(disparity / max(self.min_disparity, 1e-6), 1.0)
        rotation_score = min(rotation / self.rotation_threshold_deg, 1.0)
        motion_score = max(translation_score, rotation_score)

        # Coverage: inverse histogram correlation vs the last keyframe
        coverage_score = 1.0 - histogram_similarity

        return (motion_weight * motion_score + coverage_weight * coverage_score) / (motion_weight + coverage_weight)

    def accept_frame(self, gray: np.ndarray) -> None:
        """
        Make the given grayscale frame the new reference keyframe.
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
# Report-driven samplers
########################################################################


def _sample_by_quality(
    video_path: str,
    targets: list[int],
    *,
    total: int,
    report: dict,
    quality: dict | None,
    search_radius: int,
    on_progress,
    desc: str,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Pick one frame per target from its window, then decode exactly those.

    - The report supplies sharpness for every frame, so the winner is chosen
      BEFORE any decode: this touches len(targets) frames where the pre-report
      code decoded the whole union of windows, ~(2*search_radius + 1) times more.
    - Quality picks the winner WITHIN each window only. The targets come from the
      caller, so it never decides which regions of the video get sampled — a
      stretch of unusable footage still contributes its share of frames.
    - candidates restricts BOTH the target and its substitutes to a fixed index grid
      (the VDA context grid), so every keyframe is a grid member by construction. The
      window radius is then counted in grid steps, not source frames.
    """
    if not targets:
        return [], []

    # Thresholds meet the report here, once, on behalf of every caller
    usable = filter_frame_quality(report, **(quality or {}))
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    # Window radius: half the target spacing, capped at search_radius. This does NOT
    # make the windows disjoint — the no-grid path takes spacing from the FIRST target
    # gap only, and grid mode takes an average, so irregular targets still overlap.
    # The dedup pass below is what keeps the index map one-frame-one-row.
    if candidates is not None:
        # Grid mode: spacing and radius are counted in grid steps, and each target
        # snaps to the first grid member at or after it before the window is cut.
        grid = np.asarray(sorted({int(c) for c in candidates}), dtype=np.int64)
        if grid.size == 0:
            raise ValueError("_sample_by_quality: candidates is empty")

        # The grid indexes `usable`/`laplacian` directly, so an out-of-range member is
        # not a lookup error but silent corruption: a negative one wraps to the far end
        # of the video and writes a negative frame_idx no downstream row can match.
        if grid[0] < 0 or grid[-1] >= total:
            raise ValueError(
                f"_sample_by_quality: candidates span [{int(grid[0])}, {int(grid[-1])}], "
                f"outside the video's {total} frames"
            )

        spacing = grid.size / len(targets) if len(targets) > 1 else grid.size
        radius = min(max(int((spacing - 1) // 2), 0), search_radius)
        positions = np.clip(np.searchsorted(grid, targets), 0, grid.size - 1)
        windows = [grid[max(0, p - radius) : p + radius + 1].tolist() for p in positions]
    else:
        spacing = targets[1] - targets[0] if len(targets) > 1 else total
        radius = min(max((spacing - 1) // 2, 0), search_radius)
        windows = [sorted({min(max(t + o, 0), total - 1) for o in range(-radius, radius + 1)}) for t in targets]

    # Per target, prefer a usable frame and break ties on sharpness. max() over an
    # ascending range returns the FIRST maximal element, matching the old strict
    # `key > best` comparison — the tie-break is parity-critical.
    chosen: list[int] = [max(window, key=lambda i: (bool(usable[i]), float(laplacian[i]))) for window in windows]

    # Two targets can land on one frame: overlapping search windows when the target gaps
    # are irregular, or a candidate grid coarser than the frame budget. Keep the first
    # and say so rather than writing the same frame into the store twice.
    deduped = list(dict.fromkeys(chosen))
    if len(deduped) != len(chosen):
        logger.warning(
            "%d of %d targets collapsed onto an already-chosen frame (overlapping search "
            "windows, or a candidate grid too coarse for the frame budget); keeping %d unique frames",
            len(chosen) - len(deduped),
            len(chosen),
            len(deduped),
        )
        chosen = deduped

    # One ffmpeg select pass over exactly the frames we keep
    decoded = dict(iter_frames(video_path, indices=chosen))

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
    search_radius: int = 3,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Exactly max_frames evenly-spaced frames spanning the whole video.

    - The COUNT is the contract; spacing falls out of the video length.
    - The count can come in short if two targets pick the same frame; a warning
      names the cause. Duplicates are dropped, never written to the store twice.
    - report is required — a quality report from qa.compute_video_quality or
      qa.load_video_quality. quality= overrides filter_frame_quality's thresholds.
    - candidates: restrict every selected frame to this index grid (see _sample_by_quality).
    """
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0 or max_frames <= 0:
        return [], []

    # N evenly-spaced indices spanning the video, endpoint-anchored. Cap at the
    # source length then dedup: rounding collides as max_frames approaches total.
    targets = np.unique(np.linspace(0, total - 1, min(max_frames, total)).round().astype(int)).tolist()

    return _sample_by_quality(
        video_path,
        targets,
        total=total,
        report=report,
        quality=quality,
        search_radius=search_radius,
        on_progress=on_progress,
        desc="Uniform sampling",
        candidates=candidates,
    )


def sample_fps(
    video_path: str,
    *,
    fps: float,
    report: dict,
    min_frames: int | None = None,
    max_frames: int | None = None,
    search_radius: int = 3,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    One frame every 1/fps seconds; re-spread if the count falls outside the band.

    - The SPACING is the contract, so the baseline between consecutive frames is
      fixed regardless of video length and the count floats.
    - Outside [min_frames, max_frames] the targets are re-spread evenly over the
      WHOLE video, never truncated — truncation would hand the reconstructor half
      a scene.
    - The band binds on TARGETS, not on the result: the count can still come in
      under min_frames if two targets pick the same frame; a warning names the cause.
    - candidates: restrict every selected frame to this index grid (see _sample_by_quality).
    """
    # fps is the contract here, so an absent one is a config error, not a default
    if fps is None or fps <= 0:
        raise ValueError(f"sample_fps needs a positive fps, got {fps!r}")

    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []

    # One source of truth for the stride: passing the probe through means a context grid
    # built at this same rate contains these targets by construction, not by coincidence
    targets = context_indices(video_path, target_fps=fps, info=info)

    # Clamp the floating count into the band by re-spreading, never by truncating
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = np.unique(np.linspace(0, total - 1, min(bounded, total)).round().astype(int)).tolist()
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

    return _sample_by_quality(
        video_path,
        targets,
        total=total,
        report=report,
        quality=quality,
        search_radius=search_radius,
        on_progress=on_progress,
        desc="fps sampling",
        candidates=candidates,
    )


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

    - Runs its own decode pass: LK disparity is measured against a MOVING keyframe
      reference, which the report's fixed-stride pairs cannot supply. It does not
      re-measure blur or exposure — those come from the report.
    - A frame the report condemns is skipped as it arrives: the selector never
      scores it and never adopts it as its reference, so it moves to the next.
    - max_frames caps the result; frames scoring >= select_threshold are selected.
    """
    info = get_video_info(str(video_path))
    usable = filter_frame_quality(report, **(quality or {}))
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
        # Report gate first: an unusable frame never reaches the selector, so it
        # cannot become the reference the next frames are scored against.
        if idx >= len(usable) or not usable[idx]:
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
