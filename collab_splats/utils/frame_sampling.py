from __future__ import annotations

import json
import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


########################################################################
# Helpers
########################################################################


@lru_cache(maxsize=None)
def _get_decoder_backend() -> str:
    """Return best available video decode backend: torchcodec > ffmpeg > cv2.

    Result is cached at module load time — probe runs once per process.
    """
    try:
        import torchcodec  # noqa: F401

        return "torchcodec"
    except (ImportError, RuntimeError):
        pass
    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"
    return "cv2"


def _iter_decoded_frames(
    video_path: str,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Yield BGR uint8 HWC numpy arrays for every frame, rotation already applied.

    Dispatches to torchcodec, ffmpeg, or cv2 based on _get_decoder_backend().
    """
    backend = _get_decoder_backend()

    if backend == "ffmpeg":
        rotation = _get_rotation_degrees(video_path)
        out_w, out_h = _ffmpeg_output_dims(width, height, rotation)
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_size = out_w * out_h * 3
        try:
            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                yield np.frombuffer(raw, np.uint8).reshape(out_h, out_w, 3).copy()
        finally:
            proc.stdout.close()
            proc.terminate()
            proc.wait()
        return

    if backend == "torchcodec":
        import torch
        from torchcodec.decoders import VideoDecoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        decoder = VideoDecoder(video_path, device=device)
        for frame_batch in decoder:
            # frame_batch.data: (C, H, W) uint8 RGB tensor
            rgb = frame_batch.data.permute(1, 2, 0).cpu().numpy()
            yield rgb[:, :, ::-1].copy()  # RGB → BGR to match cv2 convention
        return

    # cv2 fallback
    rotation = _get_rotation_degrees(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield _apply_rotation(frame, rotation)
    finally:
        cap.release()


def _get_rotation_degrees(video_path: str) -> int:
    """Return CW rotation degrees needed to display video correctly, via ffprobe.

    Checks two metadata locations in order:
    1. tags.rotate — older MP4 format (still used by some cameras)
    2. side_data_list Display Matrix — modern format used by GoPro, iPhone, etc.
       ffprobe reports CCW degrees; convert to CW with (-rot) % 360.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for s in json.loads(r.stdout).get("streams", []):
            if s.get("codec_type") == "video":
                # Legacy: tags.rotate (CW degrees)
                rotate = s.get("tags", {}).get("rotate")
                if rotate:
                    return int(rotate)
                # Modern: Display Matrix side data (ffprobe reports CCW, convert to CW)
                for sd in s.get("side_data_list", []):
                    if sd.get("side_data_type") == "Display Matrix":
                        rot = sd.get("rotation")
                        if rot is not None:
                            return int(-rot) % 360
    except Exception:
        logger.debug("ffprobe failed for %s", video_path, exc_info=True)
    return 0


def _apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Apply CW rotation by degrees (0/90/180/270 only)."""
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _ffmpeg_output_dims(width: int, height: int, rotation: int) -> tuple[int, int]:
    """Return (out_w, out_h) after ffmpeg auto-rotation.

    ffmpeg rotates 90/270-degree videos by default; native dims are swapped in output.
    """
    if rotation in (90, 270):
        return height, width
    return width, height


def get_video_info(video_path: str) -> dict:
    """Return basic video metadata without exposing cv2 to callers.

    Keys: total_frames (int), fps (float), duration_s (float), width (int), height (int).
    Returns zeros for all fields if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    # Report CODED (pre-rotation) dims. cv2 auto-applies display rotation by default and would
    # return swapped (portrait) dims for a 90/270 video; the ffmpeg decode paths then double-count
    # rotation via _ffmpeg_output_dims, reshaping the rawvideo buffer with W/H swapped → noise.
    # Disabling auto-orientation keeps the rotation handled in exactly one place (_ffmpeg_output_dims
    # / _apply_rotation), matching the coded dims those helpers expect.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_s = total / fps if fps > 0 else 0.0
    return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}


########################################################################
# OpticalFlowFrameSelector
########################################################################


class OpticalFlowFrameSelector:
    """Intelligent frame selection using optical flow and coverage analysis.

    Combines motion-based (Lucas-Kanade sparse optical flow) and coverage-based
    (histogram similarity) metrics to select a diverse, non-redundant subset of
    frames from a video sequence.

    Attributes:
        min_disparity: Minimum mean pixel displacement to consider as motion.
        max_features: Maximum number of features to track.
        motion_weight: Weight for motion component (normalised to sum to 1 with coverage_weight).
        coverage_weight: Weight for coverage component.
        rotation_threshold: Minimum rotation (degrees) to register as camera rotation.
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        max_features: int = 1000,
        motion_weight: float = 0.6,
        coverage_weight: float = 0.4,
        rotation_threshold: float = 5.0,
    ):
        # Validate and normalise weights to sum to 1
        if not (0 <= motion_weight <= 1 and 0 <= coverage_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        total_weight = motion_weight + coverage_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be > 0")
        self.motion_weight = motion_weight / total_weight
        self.coverage_weight = coverage_weight / total_weight

        # Thresholds for motion and rotation detection
        self.min_disparity = min_disparity
        self.max_features = max_features
        self.rotation_threshold = rotation_threshold

        # Lucas-Kanade sparse optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Shi-Tomasi corner detection parameters for feature seeding
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        # Per-keyframe state and accumulated signal stats
        self.last_keyframe_gray = None
        self.last_keyframe_pts = None
        self.last_keyframe_hist = None
        self.stats: Dict[str, list] = {
            "disparities": [],
            "rotations": [],
            "histogram_similarities": [],
        }

    def reset(self) -> None:
        """Reset selector state; call between different videos."""
        self.last_keyframe_gray = None
        self.last_keyframe_pts = None
        self.last_keyframe_hist = None
        self.stats = {
            "disparities": [],
            "rotations": [],
            "histogram_similarities": [],
        }

    def _detect_features(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect Shi-Tomasi corners as optical flow seed points."""
        return cv2.goodFeaturesToTrack(gray_image, **self.feature_params)

    def _compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute sparse Lucas-Kanade flow; return good point pairs or (None, None).

        Returns (None, None) when tracking fails or fewer than 10 inliers survive.
        """
        if prev_pts is None or len(prev_pts) == 0:
            return None, None
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **self.lk_params)
        if curr_pts is None:
            return None, None
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        # Reject tracking results with too few inliers — unreliable for motion estimation
        if len(good_prev) < 10:
            return None, None
        return good_prev, good_curr

    def _compute_disparity(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """Mean Euclidean displacement of tracked points in pixels."""
        return float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))

    def _estimate_rotation(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        method: str = "affine",
    ) -> float:
        """Estimate camera rotation angle (degrees) via RANSAC affine or homography fit."""
        if len(prev_pts) < 4:
            return 0.0
        try:
            if method == "affine":
                M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
                if M is None:
                    return 0.0
                rotation_rad = np.arctan2(M[1, 0], M[0, 0])
            else:
                H, _ = cv2.findHomography(prev_pts, curr_pts, method=cv2.RANSAC)
                if H is None:
                    return 0.0
                rotation_rad = np.arctan2(H[1, 0], H[0, 0])
            return float(np.abs(np.degrees(rotation_rad)))
        except Exception:
            logger.debug("Rotation estimation failed", exc_info=True)
            return 0.0

    def _compute_histogram_similarity(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
        bins: int = 64,
    ) -> float:
        """Histogram correlation between two grayscale images; returns value in [0, 1]."""
        hist1 = cv2.calcHist([gray1], [0], None, [bins], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [bins], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return float(max(0.0, min(1.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))))

    def _initialize_keyframe(self, gray_frame: np.ndarray) -> None:
        """Seed keyframe tracking state from a grayscale frame."""
        self.last_keyframe_gray = gray_frame.copy()
        self.last_keyframe_pts = self._detect_features(gray_frame)
        hist = cv2.calcHist([gray_frame], [0], None, [64], [0, 256])
        self.last_keyframe_hist = cv2.normalize(hist, hist).flatten()

    def compute_frame_score(
        self,
        current_frame: np.ndarray,
        return_components: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Compute combined motion+coverage score for a BGR frame.

        Returns score in [0, 1], or (score, components_dict) when return_components=True.
        First frame always scores 1.0 and seeds keyframe state.
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # First frame: seed keyframe state and unconditionally select
        if self.last_keyframe_gray is None:
            self._initialize_keyframe(gray)
            score = 1.0
            if return_components:
                return score, {
                    "motion": 1.0,
                    "coverage": 1.0,
                    "combined": 1.0,
                    "disparity": 0.0,
                    "rotation": 0.0,
                    "histogram_similarity": 1.0,
                }
            return score

        # Motion score: max of normalised translation and rotation components
        prev_pts, curr_pts = self._compute_optical_flow(self.last_keyframe_gray, gray, self.last_keyframe_pts)
        motion_score = 0.0
        rotation = 0.0
        disparity = 0.0
        if prev_pts is not None and curr_pts is not None:
            disparity = self._compute_disparity(prev_pts, curr_pts)
            rotation = self._estimate_rotation(prev_pts, curr_pts)
            translation_score = min(disparity / self.min_disparity, 1.0)
            rotation_score = min(rotation / self.rotation_threshold, 1.0)
            motion_score = max(translation_score, rotation_score)
            self.stats["disparities"].append(disparity)
            self.stats["rotations"].append(rotation)

        # Coverage score: inverse histogram similarity against last accepted keyframe
        coverage_score = 0.0
        hist_similarity = 1.0
        if self.last_keyframe_hist is not None:
            hist_similarity = self._compute_histogram_similarity(self.last_keyframe_gray, gray)
            self.stats["histogram_similarities"].append(hist_similarity)
            coverage_score = 1.0 - hist_similarity

        # Weighted combination into final selection score
        combined_score = self.motion_weight * motion_score + self.coverage_weight * coverage_score

        if return_components:
            return combined_score, {
                "motion": motion_score,
                "coverage": coverage_score,
                "combined": combined_score,
                "disparity": disparity,
                "rotation": rotation,
                "histogram_similarity": hist_similarity,
            }
        return combined_score

    def accept_frame(self, frame_bgr: np.ndarray) -> None:
        """Update keyframe reference to the given BGR frame."""
        self._initialize_keyframe(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

    def should_select_frame(
        self,
        current_frame: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Return (should_select, score, components) for a BGR frame."""
        score, components = self.compute_frame_score(current_frame, return_components=True)
        return score >= threshold, score, components


########################################################################
# Frame Selection
########################################################################


def score_all_frames(
    video_path: str,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    stride: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Score decoded frames using OpticalFlowFrameSelector.

    Returns one dict per scored frame with keys:
        frame_idx (int), disparity (float), rotation (float),
        histogram_similarity (float), score (float), selected (bool)

    stride: score every Nth decoded frame; on_progress fires for every frame regardless.
    The first scored frame always has selected=True (score=1.0).
    """
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    info = get_video_info(video_path)
    total = info["total_frames"]
    results = []
    frames_decoded = 0
    with tqdm(total=total, desc="Scoring frames", unit="frame", disable=not verbose) as pbar:
        for frame in _iter_decoded_frames(video_path, info["width"], info["height"]):
            # Score only stride-aligned frames; skip OF analysis on others
            if frames_decoded % stride == 0:
                # Downscale to 480px wide for faster OF computation
                scale = min(1.0, 480.0 / frame.shape[1])
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
                should_select, score, components = selector.should_select_frame(small)
                if should_select:
                    selector.accept_frame(small)
                results.append(
                    {
                        "frame_idx": frames_decoded,
                        "disparity": components.get("disparity", 0.0),
                        "rotation": components.get("rotation", 0.0),
                        "histogram_similarity": components.get("histogram_similarity", 1.0),
                        "score": score,
                        "selected": should_select,
                    }
                )
            frames_decoded += 1
            pbar.update(1)
            if on_progress is not None:
                on_progress(frames_decoded, total)
    return results


def _decode_fps_ffmpeg(
    video_path: str,
    targets: list[int],
    width: int,
    height: int,
    native_fps: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract evenly-spaced frames via a single ffmpeg pass with select filter.

    ffmpeg applies rotation from container metadata automatically.
    _ffmpeg_output_dims adjusts reshape dims to match rotated output.
    Returns (rgb_frames, source_indices).
    """
    if not targets:
        return [], []

    rotation = _get_rotation_degrees(video_path)
    out_w, out_h = _ffmpeg_output_dims(width, height, rotation)
    frame_size = out_w * out_h * 3
    n_targets = len(targets)

    # Infer stride from first gap; targets from sample_frames_fps are always evenly spaced
    interval = targets[1] - targets[0] if len(targets) > 1 else 1

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"select=not(mod(n\\,{interval}))",
        "-frames:v",
        str(n_targets),
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-an",
        "pipe:1",
    ]
    frames: list[np.ndarray] = []
    indices: list[int] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while len(frames) < n_targets:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(out_h, out_w, 3).copy()
            frames.append(frame)
            indices.append(targets[len(frames) - 1])
            if on_progress is not None:
                on_progress(len(frames), n_targets)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    return frames, indices


def _decode_fps_torchcodec(
    video_path: str,
    targets: list[int],
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract specific frames by index using torchcodec GPU decoder.

    torchcodec handles rotation from container metadata automatically.
    Returns (rgb_frames, targets) — indices are exact source positions.
    """
    import torch
    from torchcodec.decoders import VideoDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = VideoDecoder(video_path, device=device)
    result = decoder.get_frames_at(indices=targets)
    # result.data: (N, C, H, W) uint8 tensor, RGB
    frames = [result.data[i].permute(1, 2, 0).cpu().numpy() for i in range(result.data.shape[0])]
    if on_progress is not None:
        for i in range(1, len(frames) + 1):
            on_progress(i, len(targets))
    return frames, targets[: len(frames)]


def sample_frames_fps(
    video_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract frames at a fixed FPS rate using the best available decoder.

    Returns (frames, indices) where indices are the source frame positions.
    on_progress: called as on_progress(n_collected, n_targets) after each frame.
    max_frames: cap the number of extracted frames; None means no cap.
    """
    info = get_video_info(video_path)
    if info["total_frames"] == 0:
        return [], []

    native_fps = info["fps"] or 30.0
    total = info["total_frames"]
    interval = max(1, int(round(native_fps / fps)))
    targets = list(range(0, total, interval))
    if max_frames is not None:
        targets = targets[:max_frames]
    if not targets:
        return [], []

    # For sparse random-access, prefer ffmpeg per-frame seeking over torchcodec.
    # torchcodec get_frames_at can hang on videos with edit-list / ISOBMFF quirks;
    # ffmpeg -ss seeking is robust and out-of-process (no GIL).
    # torchcodec is kept as fallback when ffmpeg is absent.
    if shutil.which("ffmpeg") is not None:
        logger.debug("sample_frames_fps: backend=ffmpeg(seek), targets=%d", len(targets))
        return _decode_fps_ffmpeg(
            video_path,
            targets,
            info["width"],
            info["height"],
            native_fps,
            on_progress,
        )

    backend = _get_decoder_backend()
    logger.debug("sample_frames_fps: backend=%s, targets=%d", backend, len(targets))

    if backend == "torchcodec":
        return _decode_fps_torchcodec(video_path, targets, on_progress)

    # cv2 fallback: seek to each target index
    rotation = _get_rotation_degrees(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    try:
        with tqdm(targets, desc="Sampling frames", unit="frame", disable=not verbose) as pbar:
            for target in pbar:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Could not read frame %d from %s", target, video_path)
                    continue
                frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
                indices.append(target)
                if on_progress is not None:
                    on_progress(len(frames), len(targets))
    finally:
        cap.release()
    return frames, indices


def sample_frames_optical_flow(
    video_path: str,
    min_disparity: float = 50.0,
    max_frames: int = 200,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[dict]]:
    """Select keyframes using sparse Lucas-Kanade optical flow.

    Combines motion (disparity + rotation) and visual diversity (histogram
    similarity) into a 0–1 score; selects frames scoring >= 0.5.
    OF analysis runs at max 480px wide for speed; selected frames kept full-res.

    Returns (frames, scores) where scores has one dict per selected frame:
        {frame_idx, disparity, rotation, histogram_similarity, score, selected}.
    on_progress: called as on_progress(frames_decoded, total_frames).
    min_disparity: mean pixel displacement threshold. Higher = fewer frames.
    """
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    info = get_video_info(video_path)
    total = info["total_frames"]
    frames: list[np.ndarray] = []
    scores: list[dict] = []
    frames_decoded = 0
    with tqdm(total=total, desc="Optical flow selection", unit="frame", disable=not verbose) as pbar:
        for frame in _iter_decoded_frames(video_path, info["width"], info["height"]):
            if len(frames) >= max_frames:
                break
            frames_decoded += 1
            pbar.update(1)
            if on_progress is not None:
                on_progress(frames_decoded, total)
            # Score at 480px-wide scale for speed; keep full-res copy if selected
            scale = min(1.0, 480.0 / frame.shape[1])
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
            should_select, score, components = selector.should_select_frame(small)
            if should_select:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                scores.append(
                    {
                        "frame_idx": len(frames) - 1,
                        "disparity": components.get("disparity", 0.0),
                        "rotation": components.get("rotation", 0.0),
                        "histogram_similarity": components.get("histogram_similarity", 1.0),
                        "score": score,
                        "selected": True,
                    }
                )
                selector.accept_frame(small)
    return frames, scores


########################################################################
# Frame I/O
########################################################################


def load_video_frames(
    video_path: str,
    frame_indices: list[int],
) -> list[np.ndarray]:
    """Read specific frames by index from a video; return as RGB numpy arrays.

    Applies rotation metadata correction. Does not write to disk.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation_degrees(str(video_path))
    frames = []
    # Sort and deduplicate indices to minimise seeks
    for target in sorted(set(frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            logger.warning("Could not read frame %d from %s", target, video_path)
            continue
        frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_video_frames(
    video_path: str,
    frame_indices: list[int],
    output_dir,
) -> list[Path]:
    """Extract specific frames by index and save as JPEGs in output_dir.

    Returns list of saved file paths. Applies rotation metadata correction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = _get_rotation_degrees(str(video_path))
    saved: list[Path] = []
    # Sort and deduplicate indices; VideoCapture seek is approximate for some codecs
    try:
        for target in sorted(set(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:
                logger.warning("Could not read frame %d from %s", target, video_path)
                continue
            out_path = output_dir / f"frame_{target:06d}.jpg"
            cv2.imwrite(str(out_path), _apply_rotation(frame, rotation))
            saved.append(out_path)
    finally:
        cap.release()
    return saved


########################################################################
# Score I/O
########################################################################


def save_frame_scores(scores: list[dict], path) -> None:
    """Persist score_all_frames() output to JSON for later reload."""
    Path(path).write_text(json.dumps(scores))


def load_frame_scores(path) -> list[dict]:
    """Load frame scores saved by save_frame_scores()."""
    return json.loads(Path(path).read_text())


########################################################################
# Visualization
########################################################################


def plot_frame_grid(
    frames: list,
    title: str,
    n_cols: int = 6,
) -> None:
    """Display a grid of RGB frames."""
    n = len(frames)
    n_cols = min(n_cols, n)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(frames[i])
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_selection(
    total_frames: int,
    fps_indices: list | None = None,
    of_indices: list | None = None,
) -> None:
    """Vertical-line timeline of selected frame indices.

    Single set → one panel; both sets → two stacked panels for comparison.
    """
    sets = [
        (fps_indices, "FPS", "steelblue"),
        (of_indices, "Optical Flow", "darkorange"),
    ]
    active = [(idx, label, color) for idx, label, color in sets if idx is not None]
    fig, axes = plt.subplots(len(active), 1, figsize=(12, 2 * len(active)), squeeze=False)
    for ax, (indices, label, color) in zip(axes[:, 0], active):
        if indices:
            ax.vlines(indices, 0, 1, colors=color, linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xlabel("Frame index")
        ax.set_title(f"{label}  (n={len(indices) if indices else 0})", fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_frame_scores(frame_scores: list) -> None:
    """3-panel timeseries of per-frame optical flow signals: disparity / rotation / histogram similarity.

    Selected frames marked with vertical grey lines.
    """
    if not frame_scores:
        plt.subplots(3, 1, figsize=(12, 6))
        plt.show()
        return
    idxs = [d["frame_idx"] for d in frame_scores]
    selected_idxs = [d["frame_idx"] for d in frame_scores if d["selected"]]
    panels = [
        ([d["disparity"] for d in frame_scores], "Disparity (px)", "steelblue"),
        ([d["rotation"] for d in frame_scores], "Rotation (deg)", "seagreen"),
        ([d["histogram_similarity"] for d in frame_scores], "Histogram similarity", "tomato"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.plot(idxs, values, color=color, linewidth=0.8)
        # Mark each selected frame with a faint vertical line
        for x in selected_idxs:
            ax.axvline(x, color="gray", alpha=0.25, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Per-frame optical flow scores  (grey lines = selected frames)", fontsize=11)
    fig.tight_layout()
    plt.show()


def plot_disparity_sensitivity(
    frame_scores: list,
    disparity_values: list,
) -> None:
    """Approximate selected frame count vs min_disparity threshold.

    Re-thresholds precomputed frame_scores — no video re-decode needed.
    Assumes default weights (motion=0.6, coverage=0.4) and selection threshold=0.5.
    Note: approximate; true counts differ on stateful re-runs at each threshold.
    """
    # Re-score each disparity threshold using the precomputed per-frame signals
    counts = []
    for threshold in disparity_values:
        n = sum(
            1
            for d in frame_scores
            if (0.6 * min(d["disparity"] / max(threshold, 1e-6), 1.0) + 0.4 * (1.0 - d["histogram_similarity"])) >= 0.5
        )
        counts.append(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(disparity_values, counts, marker="o", color="steelblue", linewidth=1.5)
    ax.set_xlabel("min_disparity threshold (px)")
    ax.set_ylabel("Frames selected (approx.)")
    ax.set_title("Frame count vs disparity threshold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
