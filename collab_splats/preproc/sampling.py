"""Video preprocessing: metadata, frame decoding, quality gating, keyframe sampling.

All decoding goes through ffmpeg/ffprobe — the only supported backend. cv2 is
used for in-memory image operations only (grayscale, resize, Laplacian, LK
flow, JPEG write), never for decode.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


########################################################################
# Constants
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
# Video metadata / decoding (ffmpeg + ffprobe only)
########################################################################


def _require_ffmpeg() -> None:
    """Raise if ffmpeg/ffprobe are missing — the only supported decode backend."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH; install ffmpeg (e.g. `apt install ffmpeg`)")


def _rotation_degrees(stream: dict) -> int:
    """CW display rotation from an ffprobe stream dict.

    Two metadata locations: legacy tags.rotate (CW), and Display Matrix side
    data (modern GoPro/iPhone; ffprobe reports CCW, convert with (-rot) % 360).
    """
    rotate = stream.get("tags", {}).get("rotate")
    if rotate:
        return int(rotate) % 360
    for sd in stream.get("side_data_list", []):
        if sd.get("side_data_type") == "Display Matrix" and sd.get("rotation") is not None:
            return int(-sd["rotation"]) % 360
    return 0


def get_video_info(video_path: str) -> dict:
    """Return video metadata via ffprobe.

    Keys: total_frames (int), fps (float), duration_s (float), width (int),
    height (int). Width/height are display dims (rotation applied), matching
    the frames the decode functions yield. All zeros if the file can't be probed.
    """
    _require_ffmpeg()
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    # -count_packets gives a reliable frame count when nb_frames is absent
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-count_packets", str(video_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe failed for %s", video_path, exc_info=True)
        return zeros
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        # Frame rate arrives as a ratio string, e.g. "30000/1001"
        num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else 0.0
        total = int(s.get("nb_frames") or s.get("nb_read_packets") or 0)
        width, height = int(s.get("width") or 0), int(s.get("height") or 0)
        # Report display dims: ffmpeg auto-rotates output, so 90/270 swaps W/H
        if _rotation_degrees(s) in (90, 270):
            width, height = height, width
        duration_s = total / fps if fps > 0 else 0.0
        return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}
    return zeros


def _iter_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield every frame as BGR uint8 HWC via one ffmpeg rawvideo pipe.

    ffmpeg applies rotation metadata itself, so yielded dims always match
    get_video_info's display dims.
    """
    info = get_video_info(str(video_path))
    w, h = info["width"], info["height"]
    if w == 0 or h == 0:
        return
    cmd = ["ffmpeg", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    try:
        # Read fixed-size frames until the pipe runs dry
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def _iter_frames_at(video_path: str, frame_indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) for the requested indices via one streaming pass.

    Indices are deduplicated and yielded in stream order; the decode stops
    after the last requested index. Streaming beats per-index seeking: exact
    for every codec, one process, no approximate-seek issues.
    """
    wanted = set(frame_indices)
    if not wanted:
        return
    last = max(wanted)
    for idx, frame in enumerate(_iter_frames(video_path)):
        if idx in wanted:
            yield idx, frame
        if idx >= last:
            break


########################################################################
# Frame quality
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


########################################################################
# Optical-flow selector
########################################################################


def _combine_scores(
    disparity: float,
    histogram_similarity: float,
    min_disparity: float,
    rotation: float = 0.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
) -> float:
    """Weighted motion+coverage score in [0, 1] from raw per-frame signals.

    Single home for the selection formula — the selector and
    viz.plot_disparity_sensitivity both call this, so they can't drift apart.
    """
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
    with score_frame(); promote selected frames with accept_frame(). Raw
    signals accumulate in .stats for viz. Construct fresh per video.
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        motion_weight: float = 0.6,
        coverage_weight: float = 0.4,
    ):
        # Validate weights; normalisation happens in _combine_scores
        if not (0 <= motion_weight <= 1 and 0 <= coverage_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if motion_weight + coverage_weight == 0:
            raise ValueError("At least one weight must be > 0")
        self.min_disparity = min_disparity
        self.motion_weight = motion_weight
        self.coverage_weight = coverage_weight
        # Reference keyframe state, seeded on the first scored frame
        self.last_keyframe_gray: np.ndarray | None = None
        self.last_keyframe_pts: np.ndarray | None = None
        # Accumulated raw signals for viz / sensitivity analysis
        self.stats: dict[str, list] = {
            "disparities": [],
            "rotations": [],
            "histogram_similarities": [],
        }

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
        # Accumulate raw signals for downstream plots
        self.stats["disparities"].append(disparity)
        self.stats["rotations"].append(rotation)
        self.stats["histogram_similarities"].append(hist_similarity)
        score = _combine_scores(
            disparity,
            hist_similarity,
            self.min_disparity,
            rotation=rotation,
            motion_weight=self.motion_weight,
            coverage_weight=self.coverage_weight,
        )
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


def _analysis_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Grayscale copy downscaled to _ANALYSIS_WIDTH for scoring."""
    scale = min(1.0, _ANALYSIS_WIDTH / frame_bgr.shape[1])
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame_bgr
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


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


def sample_frames(
    video_path: str,
    *,
    method: str = "uniform",
    max_frames: int | None = None,
    fps: float | None = None,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Select frames from a video for reconstruction.

    Methods:
        "uniform": one frame per fixed window — the sharpest usable frame in
            each. Window size comes from `fps` (samples/second), or is derived
            from `max_frames` when fps is None (falls back to 2.0 fps).
        "optical_flow": motion (LK disparity + rotation) + coverage (histogram
            diversity) scoring; frames scoring >= 0.5 are selected.
            Uses min_disparity / motion_weight / coverage_weight.

    Both methods apply the quality gate (blur + exposure); `blur_threshold`
    tunes it (0.0 disables the blur check).

    Returns (frames, records): RGB arrays and one dict per selected frame with
    at least frame_idx (SOURCE video index) and blur_score; optical_flow adds
    disparity, rotation, histogram_similarity, score, selected.
    """
    if method == "uniform":
        return _sample_uniform(
            video_path,
            fps=fps,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    if method == "optical_flow":
        return _sample_optical_flow(
            video_path,
            max_frames=max_frames,
            min_disparity=min_disparity,
            motion_weight=motion_weight,
            coverage_weight=coverage_weight,
            blur_threshold=blur_threshold,
            on_progress=on_progress,
        )
    raise ValueError(f"Unknown method: {method!r} (expected 'uniform' or 'optical_flow')")


def _sample_uniform(
    video_path: str,
    *,
    fps: float | None,
    max_frames: int | None,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Uniform windows over the video; keep the sharpest usable frame per window."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    # Window size: from fps if given, else spread max_frames over the video
    native_fps = info["fps"] or 30.0
    if fps is not None:
        interval = max(1, int(round(native_fps / fps)))
    elif max_frames:
        interval = max(1, total // max_frames)
    else:
        interval = max(1, int(round(native_fps / 2.0)))
    report, close = _progress_reporter(total, "Uniform sampling", on_progress)
    frames: list[np.ndarray] = []
    records: list[dict] = []
    best: tuple[float, int, np.ndarray] | None = None  # (blur, idx, frame)
    try:
        for idx, frame in enumerate(_iter_frames(video_path)):
            report(idx + 1)
            # Track the sharpest gate-passing frame within the current window
            gray = _analysis_gray(frame)
            blur = compute_blur_score(gray)
            usable, _ = check_frame_quality(gray, blur_threshold, blur_score=blur)
            if usable and (best is None or blur > best[0]):
                best = (blur, idx, frame)
            # Window boundary: flush the best frame and start the next window
            if (idx + 1) % interval == 0:
                if best is not None:
                    frames.append(cv2.cvtColor(best[2], cv2.COLOR_BGR2RGB))
                    records.append({"frame_idx": best[1], "blur_score": best[0]})
                best = None
                if max_frames is not None and len(frames) >= max_frames:
                    return frames, records
        # Final partial window
        if best is not None and (max_frames is None or len(frames) < max_frames):
            frames.append(cv2.cvtColor(best[2], cv2.COLOR_BGR2RGB))
            records.append({"frame_idx": best[1], "blur_score": best[0]})
    finally:
        close()
    return frames, records


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
    motion_weight: float,
    coverage_weight: float,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Optical-flow keyframe selection; keeps selected frames full-res RGB."""
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
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
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Score every frame without keeping pixel data — analysis/viz workflow.

    Returns one record per frame: frame_idx, blur_score, exposure_mean,
    exposure_std, reject_reason, disparity, rotation, histogram_similarity,
    score, selected.
    """
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
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


########################################################################
# Frame I/O
########################################################################


def load_frames(video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
    """Read specific frames by index; returns RGB arrays in index order."""
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in _iter_frames_at(video_path, frame_indices)]


def extract_frame(video_path: "str | Path", frame_idx: int) -> np.ndarray:
    """Decode exactly one frame (0-based index) via ffmpeg; returns (H, W, 3) uint8 RGB.

    Raises ValueError if frame_idx is past the end of the video.
    """
    _require_ffmpeg()
    # Delegate to the shared rotation-aware streaming decode — dims always match
    # get_video_info's display dims (ffmpeg applies rotation metadata itself)
    for _, frame in _iter_frames_at(str(video_path), [frame_idx]):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"extract_frame: frame {frame_idx} not found in {video_path}")


def extract_frame_fast(video_path: "str | Path", frame_idx: int) -> np.ndarray:
    """Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    Seeks by timestamp (frame_idx / fps) before demuxing — O(1) in frame depth, so a
    deep frame previews instantly. Exact on constant-frame-rate video; may land one
    frame off near keyframes on VFR sources. Use extract_frame where exactness matters
    (e.g. the localization run, which records frame_idx as provenance).
    """
    _require_ffmpeg()
    info = get_video_info(str(video_path))
    fps, w, h, total = info["fps"], info["width"], info["height"], info["total_frames"]
    if not fps or not w or not h:
        # Unprobeable video: fall back to the exact streaming decode.
        return extract_frame(video_path, frame_idx)
    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"extract_frame_fast: frame {frame_idx} out of range for {video_path}")
    # Seek to the frame midpoint, not its start: PTS float rounding can otherwise land
    # the demuxer just past the target timestamp and decode frame N+1 instead of N.
    seek_s = max(frame_idx - 0.5, 0) / fps
    # -ss before -i = input seek (demuxer-level); rawvideo pipe avoids a temp file.
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{seek_s:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    raw = proc.stdout
    if len(raw) < w * h * 3:
        err = proc.stderr.decode(errors="replace")[-500:]
        raise ValueError(f"extract_frame_fast: frame {frame_idx} not found in {video_path}: {err}")
    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()


def extract_frames(video_path: str, frame_indices: list[int], output_dir) -> list[Path]:
    """Save specific frames as frame_NNNNNN.jpg in output_dir; returns saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, frame in _iter_frames_at(video_path, frame_indices):
        out_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    return saved
