from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np


def _apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate frame to correct for container rotation metadata."""
    try:
        import cv2
    except ImportError:
        return frame
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def sample_frames_fps(
    video_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Extract frames at a fixed FPS rate using sequential decoding.

    on_progress: called as on_progress(frame_index, total_frames) after each
        decoded frame, where total_frames is from CAP_PROP_FRAME_COUNT.
    max_frames: stop after collecting this many frames; None means no cap.
    """
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(video_path)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(native_fps / fps)))
    frames = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            if on_progress is not None:
                on_progress(idx, total)
            idx += 1
    finally:
        cap.release()
    return frames


class OpticalFlowFrameSelector:
    """
    Intelligent frame selection using optical flow and coverage analysis.

    Combines motion-based (Lucas-Kanade sparse optical flow) and coverage-based
    (histogram similarity) metrics to select a diverse, non-redundant subset of
    frames from a video sequence.

    Attributes:
        min_disparity: Minimum mean pixel displacement to consider as motion.
        max_features: Maximum number of features to track.
        motion_weight: Weight for motion component in scoring (normalised to sum to 1).
        coverage_weight: Weight for coverage component in scoring (normalised to sum to 1).
        rotation_threshold: Minimum rotation (degrees) to detect as camera rotation.
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        max_features: int = 1000,
        motion_weight: float = 0.6,
        coverage_weight: float = 0.4,
        rotation_threshold: float = 5.0,
    ):
        import cv2

        # Validate weights
        if not (0 <= motion_weight <= 1 and 0 <= coverage_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        total_weight = motion_weight + coverage_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be > 0")

        # Normalise weights to sum to 1
        self.motion_weight = motion_weight / total_weight
        self.coverage_weight = coverage_weight / total_weight

        # Flow parameters
        self.min_disparity = min_disparity
        self.max_features = max_features
        self.rotation_threshold = rotation_threshold

        # Optical flow parameters (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        # State variables
        self.last_keyframe_gray = None
        self.last_keyframe_pts = None
        self.last_keyframe_hist = None

        # Statistics tracking
        self.stats: Dict[str, list] = {
            "disparities": [],
            "rotations": [],
            "histogram_similarities": [],
        }

    def reset(self):
        """Reset the selector state (call this between different videos)."""
        self.last_keyframe_gray = None
        self.last_keyframe_pts = None
        self.last_keyframe_hist = None
        self.stats = {
            "disparities": [],
            "rotations": [],
            "histogram_similarities": [],
        }

    def _detect_features(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect good features to track using Shi-Tomasi corner detection.

        Args:
            gray_image: Grayscale image (H, W)

        Returns:
            Array of corner points (N, 1, 2) or None if no features found.
        """
        import cv2

        pts = cv2.goodFeaturesToTrack(gray_image, **self.feature_params)
        return pts

    def _compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute sparse optical flow using Lucas-Kanade method.

        Args:
            prev_gray: Previous grayscale frame (H, W)
            curr_gray: Current grayscale frame (H, W)
            prev_pts: Points to track from previous frame (N, 1, 2)

        Returns:
            Tuple of (good_prev_pts, good_curr_pts) or (None, None) if tracking fails.
        """
        import cv2

        if prev_pts is None or len(prev_pts) == 0:
            return None, None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            return None, None

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 10:
            return None, None

        return good_prev, good_curr

    def _compute_disparity(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
    ) -> float:
        """
        Compute mean displacement (disparity) of tracked points.

        Args:
            prev_pts: Previous point positions (N, 2)
            curr_pts: Current point positions (N, 2)

        Returns:
            Mean Euclidean displacement in pixels.
        """
        displacements = np.linalg.norm(curr_pts - prev_pts, axis=1)
        return float(np.mean(displacements))

    def _estimate_rotation(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        method: str = "affine",
    ) -> float:
        """
        Estimate camera rotation between frames using point correspondences.

        Args:
            prev_pts: Previous point positions (N, 2)
            curr_pts: Current point positions (N, 2)
            method: 'affine' or 'homography' for transformation estimation.

        Returns:
            Estimated rotation angle in degrees.
        """
        import cv2

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
            return 0.0

    def _compute_histogram_similarity(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
        bins: int = 64,
    ) -> float:
        """
        Compute histogram similarity between two images using correlation.

        Args:
            gray1: First grayscale image (H, W)
            gray2: Second grayscale image (H, W)
            bins: Number of histogram bins.

        Returns:
            Similarity score between 0 (different) and 1 (identical).
        """
        import cv2

        hist1 = cv2.calcHist([gray1], [0], None, [bins], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [bins], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(max(0.0, min(1.0, similarity)))

    def _compute_normalized_entropy(self, hist: np.ndarray) -> float:
        """
        Compute normalised entropy of a histogram array.

        Args:
            hist: 1-D histogram (already normalised to sum to 1).

        Returns:
            Entropy in [0, 1] relative to uniform distribution.
        """
        probs = hist[hist > 0]
        if len(probs) == 0:
            return 0.0
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(hist))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def compute_frame_score(
        self,
        current_frame: np.ndarray,
        return_components: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute a combined score for frame selection.

        Args:
            current_frame: Current BGR frame (H, W, 3).
            return_components: If True, also return individual score components.

        Returns:
            Combined score in [0, 1], or (score, components_dict) when
            return_components=True.
        """
        import cv2

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if self.last_keyframe_gray is None:
            self._initialize_keyframe(gray)
            score = 1.0
            if return_components:
                return score, {"motion": 1.0, "coverage": 1.0, "combined": 1.0}
            return score

        prev_pts, curr_pts = self._compute_optical_flow(
            self.last_keyframe_gray, gray, self.last_keyframe_pts
        )

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

        coverage_score = 0.0
        hist_similarity = 1.0

        if self.last_keyframe_hist is not None:
            hist_similarity = self._compute_histogram_similarity(self.last_keyframe_gray, gray)
            self.stats["histogram_similarities"].append(hist_similarity)
            coverage_score = 1.0 - hist_similarity

        combined_score = self.motion_weight * motion_score + self.coverage_weight * coverage_score

        if return_components:
            components = {
                "motion": motion_score,
                "coverage": coverage_score,
                "combined": combined_score,
                "disparity": disparity,
                "rotation": rotation,
                "histogram_similarity": hist_similarity,
            }
            return combined_score, components

        return combined_score

    def _initialize_keyframe(self, gray_frame: np.ndarray):
        """Initialise keyframe tracking with the given frame."""
        import cv2

        self.last_keyframe_gray = gray_frame.copy()
        self.last_keyframe_pts = self._detect_features(gray_frame)
        self.last_keyframe_hist = cv2.calcHist([gray_frame], [0], None, [64], [0, 256])
        self.last_keyframe_hist = cv2.normalize(
            self.last_keyframe_hist, self.last_keyframe_hist
        ).flatten()

    def accept_frame(self, frame_bgr: np.ndarray) -> None:
        """Update keyframe reference after selecting a frame.

        Args:
            frame_bgr: BGR frame used for OF analysis — may be downscaled from original.
        """
        import cv2

        self._initialize_keyframe(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

    def should_select_frame(
        self,
        current_frame: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determine if current frame should be selected as a keyframe.

        Args:
            current_frame: Current BGR frame (H, W, 3).
            threshold: Selection threshold in [0, 1] (default 0.5).

        Returns:
            Tuple of (should_select, score, components).
        """
        score, components = self.compute_frame_score(current_frame, return_components=True)
        should_select = score >= threshold
        return should_select, score, components


def sample_frames_optical_flow(
    video_path: str,
    min_disparity: float = 50.0,
    max_frames: int = 200,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Select keyframes using sparse Lucas-Kanade optical flow.

    Combines motion (disparity + rotation) and visual diversity (histogram
    similarity) into a 0–1 score. Selects frames scoring >= 0.5.
    OF analysis runs at max 480px wide for speed; selected frames kept full-res.

    on_progress: called as on_progress(frames_decoded, total_frames) after
        each decoded frame, where total_frames is from CAP_PROP_FRAME_COUNT.
    min_disparity: mean pixel displacement threshold for motion detection.
        Higher = fewer frames. Typical range: 10–200px.
    """
    try:
        import cv2
    except ImportError:
        return []

    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    cap = cv2.VideoCapture(video_path)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frames_decoded = 0
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_decoded += 1
            if on_progress is not None:
                on_progress(frames_decoded, total)

            frame = _apply_rotation(frame, rotation)
            scale = min(1.0, 480.0 / frame.shape[1])
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame

            should_select, _, _ = selector.should_select_frame(small)
            if should_select:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                selector.accept_frame(small)
    finally:
        cap.release()
    return frames
