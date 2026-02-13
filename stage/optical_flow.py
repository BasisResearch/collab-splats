"""
Optical Flow-Based Video Frame Decimation and Selection

This module provides comprehensive tools for intelligent frame selection from videos
using optical flow analysis. Inspired by VGGT-SLAM's keyframe selection strategy,
it combines motion-based and coverage-based metrics to select diverse, non-redundant
frames from video sequences.

Main Components:
1. OpticalFlowFrameSelector: Main class for frame selection
2. Video utilities: Rotation detection and frame loading
3. Metrics extraction: Helper functions for analyzing selection results
4. Visualization: Functions for creating diagnostic plots

Usage:
    from optical_flow import OpticalFlowFrameSelector, extract_metrics, get_video_rotation

    # Create selector
    selector = OpticalFlowFrameSelector(
        min_disparity=50,
        motion_weight=0.6,
        coverage_weight=0.4
    )

    # Process video and save metadata
    selected_indices, metrics = selector.process_video(
        'path/to/video.mp4',
        save_selected_frames=True,
        save_metadata_json=True
    )

    # Export specific frames by index
    selector.export_frames_by_indices(
        'path/to/video.mp4',
        frame_indices=range(0, 100, 5),  # Every 5th frame from 0-100
        output_dir='output/frames'
    )

    # Extract metrics for analysis
    plot_data = extract_metrics(metrics)

Author: Based on VGGT-SLAM optical flow keyframe selection
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

# Import video utilities from preproc_utils
from preproc_utils import get_video_rotation, load_frame, rotate_frame


# ============================================================================
# Metrics Extraction
# ============================================================================

def extract_metrics(metrics: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract arrays from the metrics dictionary for easy plotting and analysis.

    The metrics structure returned by process_video() has two sources of data:
    1. metrics['stats'] - Contains raw disparity/rotation/histogram data
    2. metrics['components'] - Contains normalized scores (motion, coverage, combined)

    Note: The first frame has special handling and only contains basic scores.

    Args:
        metrics: Metrics dictionary from process_video() or process_image_directory()

    Returns:
        Dictionary containing numpy arrays of all metrics:
        - disparities: Raw pixel displacements (N-1,)
        - rotations: Rotation angles in degrees (N-1,)
        - histogram_similarities: Visual similarity scores (N-1,)
        - motion_scores: Normalized motion scores (N,)
        - coverage_scores: Normalized coverage scores (N,)
        - combined_scores: Final selection scores (N,)
    """
    # Get disparity values from stats (the ACTUAL disparity measurements in pixels)
    disparities = np.array(metrics['stats']['disparities'])
    rotations = np.array(metrics['stats']['rotations'])
    hist_similarities = np.array(metrics['stats']['histogram_similarities'])

    # Get normalized scores from components
    components = metrics['components']
    scores = metrics['scores']

    motion_scores = np.array([c.get('motion', 0) for c in components])
    coverage_scores = np.array([c.get('coverage', 0) for c in components])
    combined_scores = np.array(scores)

    return {
        'disparities': disparities,
        'rotations': rotations,
        'histogram_similarities': hist_similarities,
        'motion_scores': motion_scores,
        'coverage_scores': coverage_scores,
        'combined_scores': combined_scores,
    }


# ============================================================================
# Optical Flow Frame Selector
# ============================================================================

class OpticalFlowFrameSelector:
    """
    Intelligent frame selection using optical flow and coverage analysis.

    This class combines motion-based (optical flow) and coverage-based (visual diversity)
    metrics to select a subset of frames from a video sequence. It's designed for
    preprocessing video data for 3D reconstruction, SLAM, or other computer vision tasks.

    Key Features:
    - Lucas-Kanade sparse optical flow for motion detection
    - Histogram similarity for visual diversity
    - Rotation detection (important for drone footage)
    - Adaptive thresholding based on video statistics
    - Configurable motion vs. coverage weighting

    Attributes:
        min_disparity: Minimum mean pixel displacement to consider as motion (default: 50)
        max_features: Maximum number of features to track (default: 1000)
        motion_weight: Weight for motion component in scoring (0-1)
        coverage_weight: Weight for coverage component in scoring (0-1)
        histogram_similarity_threshold: Max histogram similarity for coverage (0-1)
        adaptive_threshold: Whether to auto-adjust disparity threshold
        rotation_threshold: Minimum rotation (degrees) to detect as camera rotation
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        max_features: int = 1000,
        motion_weight: float = 0.6,
        coverage_weight: float = 0.4,
        histogram_similarity_threshold: float = 0.85,
        adaptive_threshold: bool = True,
        rotation_threshold: float = 5.0,
        verbose: bool = True,
    ):
        """
        Initialize the optical flow frame selector.

        Args:
            min_disparity: Minimum mean pixel displacement for keyframe selection.
                Higher values = fewer frames selected. For drone footage, 30-80 is typical.
            max_features: Maximum corner points to track with optical flow.
                More features = more robust but slower. 500-2000 is typical.
            motion_weight: Importance of motion in frame selection (0-1).
                Higher = prioritize frames with camera motion.
            coverage_weight: Importance of visual diversity in frame selection (0-1).
                Higher = prioritize frames that look different from previous selections.
            histogram_similarity_threshold: Maximum allowed histogram similarity (0-1).
                Lower = require more visual diversity between frames.
            adaptive_threshold: If True, auto-adjust min_disparity based on video stats.
                Recommended for varied video content.
            rotation_threshold: Minimum rotation in degrees to detect rotation-only motion.
                Important for drone footage where camera rotates without translating.
            verbose: Whether to print progress information.
        """
        # Validate weights
        if not (0 <= motion_weight <= 1 and 0 <= coverage_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        total_weight = motion_weight + coverage_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be > 0")

        # Normalize weights to sum to 1
        self.motion_weight = motion_weight / total_weight
        self.coverage_weight = coverage_weight / total_weight

        # Flow parameters
        self.min_disparity = min_disparity
        self.max_features = max_features
        self.adaptive_threshold = adaptive_threshold
        self.rotation_threshold = rotation_threshold

        # Coverage parameters
        self.histogram_similarity_threshold = histogram_similarity_threshold

        # Optical flow parameters (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

        # State variables
        self.last_keyframe_gray = None
        self.last_keyframe_pts = None
        self.last_keyframe_hist = None

        self.verbose = verbose

        # Statistics tracking
        self.stats = {
            'disparities': [],
            'rotations': [],
            'histogram_similarities': [],
        }

    def reset(self):
        """Reset the selector state (call this between different videos)."""
        self.last_keyframe_gray = None
        self.last_keyframe_pts = None
        self.last_keyframe_hist = None
        self.stats = {
            'disparities': [],
            'rotations': [],
            'histogram_similarities': [],
        }

    def _detect_features(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect good features to track using Shi-Tomasi corner detection.

        Args:
            gray_image: Grayscale image (H, W)

        Returns:
            Array of corner points (N, 1, 2) or None if no features found
        """
        pts = cv2.goodFeaturesToTrack(gray_image, **self.feature_params)
        return pts

    def _compute_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute sparse optical flow using Lucas-Kanade method.

        Args:
            prev_gray: Previous grayscale frame (H, W)
            curr_gray: Current grayscale frame (H, W)
            prev_pts: Points to track from previous frame (N, 1, 2)

        Returns:
            Tuple of (good_prev_pts, good_curr_pts) or (None, None) if tracking fails
        """
        if prev_pts is None or len(prev_pts) == 0:
            return None, None

        # Calculate optical flow
        curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            return None, None

        # Filter points with good tracking status
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 10:  # Need minimum points for reliable estimation
            return None, None

        return good_prev, good_curr

    def _compute_disparity(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray
    ) -> float:
        """
        Compute mean displacement (disparity) of tracked points.

        Args:
            prev_pts: Previous point positions (N, 2)
            curr_pts: Current point positions (N, 2)

        Returns:
            Mean Euclidean displacement in pixels
        """
        displacements = np.linalg.norm(curr_pts - prev_pts, axis=1)
        return np.mean(displacements)

    def _estimate_rotation(
        self,
        prev_pts: np.ndarray,
        curr_pts: np.ndarray,
        method: str = 'affine'
    ) -> float:
        """
        Estimate camera rotation between frames using point correspondences.

        For drone footage, rotation is important to detect since the camera
        can rotate significantly without much translation.

        Args:
            prev_pts: Previous point positions (N, 2)
            curr_pts: Current point positions (N, 2)
            method: 'affine' or 'homography' for transformation estimation

        Returns:
            Estimated rotation angle in degrees
        """
        if len(prev_pts) < 4:
            return 0.0

        try:
            if method == 'affine':
                # Estimate affine transform
                M, inliers = cv2.estimateAffinePartial2D(
                    prev_pts, curr_pts, method=cv2.RANSAC
                )
                if M is None:
                    return 0.0

                # Extract rotation angle from affine matrix
                # M = [cos(θ)*s  -sin(θ)*s  tx]
                #     [sin(θ)*s   cos(θ)*s  ty]
                rotation_rad = np.arctan2(M[1, 0], M[0, 0])
                rotation_deg = np.abs(np.degrees(rotation_rad))

            else:  # homography
                H, inliers = cv2.findHomography(
                    prev_pts, curr_pts, method=cv2.RANSAC
                )
                if H is None:
                    return 0.0

                # Decompose homography (simplified rotation extraction)
                # For small rotations, can approximate from H
                rotation_rad = np.arctan2(H[1, 0], H[0, 0])
                rotation_deg = np.abs(np.degrees(rotation_rad))

            return rotation_deg

        except Exception:
            return 0.0

    def _compute_histogram_similarity(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
        bins: int = 64
    ) -> float:
        """
        Compute histogram similarity between two images using correlation.

        This measures visual diversity - low similarity means the frames
        look different (good for coverage).

        Args:
            gray1: First grayscale image (H, W)
            gray2: Second grayscale image (H, W)
            bins: Number of histogram bins

        Returns:
            Similarity score between 0 (different) and 1 (identical)
        """
        # Compute histograms
        hist1 = cv2.calcHist([gray1], [0], None, [bins], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [bins], [0, 256])

        # Normalize
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Compute correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return max(0.0, min(1.0, similarity))

    def _compute_feature_distribution_score(
        self,
        gray_image: np.ndarray,
        grid_size: int = 8
    ) -> float:
        """
        Compute how well features are distributed across the image.

        Better distribution = better coverage of the scene. This helps
        avoid selecting frames where all features are clustered in one area.

        Args:
            gray_image: Grayscale image (H, W)
            grid_size: Divide image into grid_size x grid_size cells

        Returns:
            Distribution score between 0 (all in one area) and 1 (well distributed)
        """
        pts = self._detect_features(gray_image)
        if pts is None or len(pts) == 0:
            return 0.0

        h, w = gray_image.shape
        cell_h = h / grid_size
        cell_w = w / grid_size

        # Count features in each grid cell
        grid_counts = np.zeros((grid_size, grid_size), dtype=int)

        for pt in pts:
            x, y = pt[0]
            cell_x = min(int(x / cell_w), grid_size - 1)
            cell_y = min(int(y / cell_h), grid_size - 1)
            grid_counts[cell_y, cell_x] += 1

        # Compute entropy (higher = better distribution)
        total = len(pts)
        probs = grid_counts.flatten() / total
        probs = probs[probs > 0]  # Remove zeros for log

        if len(probs) == 0:
            return 0.0

        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(grid_size * grid_size)  # Uniform distribution

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def compute_frame_score(
        self,
        current_frame: np.ndarray,
        return_components: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute a combined score for frame selection.

        This combines motion-based and coverage-based metrics to decide
        whether to select a frame as a keyframe.

        Args:
            current_frame: Current BGR frame (H, W, 3)
            return_components: If True, return individual score components

        Returns:
            Combined score (higher = more likely to select) or
            Tuple of (score, components_dict) if return_components=True
        """
        # Convert to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Initialize on first frame
        if self.last_keyframe_gray is None:
            self._initialize_keyframe(gray)
            score = 1.0  # Always select first frame
            if return_components:
                return score, {'motion': 1.0, 'coverage': 1.0, 'combined': 1.0}
            return score

        # Compute optical flow
        prev_pts, curr_pts = self._compute_optical_flow(
            self.last_keyframe_gray, gray, self.last_keyframe_pts
        )

        # Motion component
        motion_score = 0.0
        rotation = 0.0
        disparity = 0.0

        if prev_pts is not None and curr_pts is not None:
            # Compute disparity (translation)
            disparity = self._compute_disparity(prev_pts, curr_pts)

            # Compute rotation
            rotation = self._estimate_rotation(prev_pts, curr_pts)

            # Combine translation and rotation for motion score
            # Normalize by thresholds
            translation_score = min(disparity / self.min_disparity, 1.0)
            rotation_score = min(rotation / self.rotation_threshold, 1.0)

            # Take max (either translation OR rotation is sufficient)
            motion_score = max(translation_score, rotation_score)

            # Track statistics
            self.stats['disparities'].append(disparity)
            self.stats['rotations'].append(rotation)

        # Coverage component
        coverage_score = 0.0
        hist_similarity = 1.0

        if self.last_keyframe_hist is not None:
            # Histogram similarity (lower is better for coverage)
            hist_similarity = self._compute_histogram_similarity(
                self.last_keyframe_gray, gray
            )
            self.stats['histogram_similarities'].append(hist_similarity)

            # Convert to coverage score (invert similarity)
            coverage_score = 1.0 - hist_similarity

        # Combined score
        combined_score = (
            self.motion_weight * motion_score +
            self.coverage_weight * coverage_score
        )

        if return_components:
            components = {
                'motion': motion_score,
                'coverage': coverage_score,
                'combined': combined_score,
                'disparity': disparity,
                'rotation': rotation,
                'histogram_similarity': hist_similarity,
            }
            return combined_score, components

        return combined_score

    def _initialize_keyframe(self, gray_frame: np.ndarray):
        """Initialize keyframe tracking with the first frame."""
        self.last_keyframe_gray = gray_frame.copy()
        self.last_keyframe_pts = self._detect_features(gray_frame)
        self.last_keyframe_hist = cv2.calcHist(
            [gray_frame], [0], None, [64], [0, 256]
        )
        self.last_keyframe_hist = cv2.normalize(
            self.last_keyframe_hist, self.last_keyframe_hist
        ).flatten()

    def _update_keyframe(self, gray_frame: np.ndarray):
        """Update the reference keyframe after selection."""
        self._initialize_keyframe(gray_frame)

    def should_select_frame(
        self,
        current_frame: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determine if current frame should be selected as keyframe.

        Args:
            current_frame: Current BGR frame (H, W, 3)
            threshold: Selection threshold (default: 0.5 for combined score)

        Returns:
            Tuple of (should_select, score, components)
        """
        if threshold is None:
            threshold = 0.5  # Default threshold for normalized [0,1] score

        score, components = self.compute_frame_score(
            current_frame, return_components=True
        )

        should_select = score >= threshold

        return should_select, score, components

    def export_frames_by_indices(
        self,
        video_path: Union[str, Path],
        frame_indices: Union[List[int], range],
        output_dir: Union[str, Path],
        apply_rotation: bool = True,
    ) -> None:
        """
        Export specific frames from a video by their indices.

        Args:
            video_path: Path to input video file
            frame_indices: List or range of frame indices to export (e.g., [0, 10, 20] or range(0, 100))
            output_dir: Directory to save frames (will be cleared before writing)
            apply_rotation: Whether to apply rotation correction based on video metadata
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Convert range to list if needed
        if isinstance(frame_indices, range):
            frame_indices = list(frame_indices)

        # Remove and recreate output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Detect video rotation
        video_rotation = None
        if apply_rotation:
            video_rotation = get_video_rotation(video_path)
            if video_rotation is not None and self.verbose:
                rotation_degrees = {
                    cv2.ROTATE_90_CLOCKWISE: 90,
                    cv2.ROTATE_180: 180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE: 270
                }.get(video_rotation, 0)
                print(f"Detected video rotation: {rotation_degrees}° (will be corrected when saving frames)")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.verbose:
            print(f"\nExporting {len(frame_indices)} frames from: {video_path.name}")
            print(f"  Total video frames: {total_frames}")
            print(f"  Output directory: {output_dir}")

        # Sort indices for efficient sequential reading
        sorted_indices = sorted(frame_indices)

        # Export frames
        frames_saved = 0
        pbar = tqdm(total=len(sorted_indices), desc="Exporting frames", disable=not self.verbose)

        for idx in sorted_indices:
            if idx >= total_frames:
                if self.verbose:
                    print(f"Warning: Frame index {idx} exceeds video length ({total_frames}), skipping")
                continue

            # Load frame
            frame = load_frame(cap, idx)
            if frame is None:
                if self.verbose:
                    print(f"Warning: Failed to load frame {idx}, skipping")
                continue

            # Apply rotation if needed
            if apply_rotation and video_rotation is not None:
                frame = rotate_frame(frame, video_rotation)

            # Save frame
            output_path = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            frames_saved += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        if self.verbose:
            print(f"✓ Exported {frames_saved} frames to: {output_dir}")

    def process_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None,
        selection_threshold: float = 0.5,
        output_dir: Optional[Union[str, Path]] = None,
        save_selected_frames: bool = False,
        save_metadata_json: bool = False,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Process a video file and select keyframes.

        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None = all)
            selection_threshold: Threshold for frame selection (0-1)
            output_dir: Directory to save selected frames (if save_selected_frames=True)
            save_selected_frames: Whether to save selected frames to disk
            save_metadata_json: Whether to save frame selection metadata to JSON file

        Returns:
            Tuple of (selected_frame_indices, metrics_dict)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Reset state
        self.reset()

        # Detect video rotation from metadata
        video_rotation = get_video_rotation(video_path)
        if video_rotation is not None:
            rotation_degrees = {
                cv2.ROTATE_90_CLOCKWISE: 90,
                cv2.ROTATE_180: 180,
                cv2.ROTATE_90_COUNTERCLOCKWISE: 270
            }.get(video_rotation, 0)
            if self.verbose:
                print(f"Detected video rotation: {rotation_degrees}° (will be corrected when saving frames)")
        else:
            if self.verbose:
                print("No rotation metadata detected")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        if self.verbose:
            print(f"\nProcessing video: {video_path.name}")
            print(f"  Total frames: {total_frames}")
            print(f"  FPS: {fps:.2f}")
            print(f"  Selection threshold: {selection_threshold}")

        # Setup output directory if saving frames
        if save_selected_frames:
            if output_dir is None:
                output_dir = video_path.parent / f"{video_path.stem}_selected"
            output_dir = Path(output_dir)
            # Remove and recreate directory to ensure clean state
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Process frames
        selected_indices = []
        all_scores = []
        all_components = []

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing frames", disable=not self.verbose)

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            # Check if frame should be selected
            should_select, score, components = self.should_select_frame(
                frame, threshold=selection_threshold
            )

            all_scores.append(score)
            all_components.append(components)

            if should_select:
                selected_indices.append(frame_idx)

                # Update keyframe reference
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._update_keyframe(gray)

                # Save frame if requested
                if save_selected_frames:
                    # Apply rotation correction before saving
                    frame_to_save = rotate_frame(frame, video_rotation)
                    output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_path), frame_to_save)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        # Compute adaptive threshold if enabled
        if self.adaptive_threshold and len(self.stats['disparities']) > 0:
            median_disparity = np.median(self.stats['disparities'])
            suggested_disparity = median_disparity * 0.8  # Slightly below median

            if self.verbose:
                print(f"\nAdaptive threshold analysis:")
                print(f"  Current min_disparity: {self.min_disparity:.1f}")
                print(f"  Median disparity: {median_disparity:.1f}")
                print(f"  Suggested min_disparity: {suggested_disparity:.1f}")

        # Compile metrics
        metrics = {
            'total_frames': frame_idx,
            'selected_frames': len(selected_indices),
            'selection_rate': len(selected_indices) / frame_idx if frame_idx > 0 else 0,
            'scores': all_scores,
            'components': all_components,
            'stats': self.stats,
        }

        if self.verbose:
            print(f"\nSelection summary:")
            print(f"  Total frames: {metrics['total_frames']}")
            print(f"  Selected frames: {metrics['selected_frames']}")
            print(f"  Selection rate: {metrics['selection_rate']:.1%}")
            if save_selected_frames:
                print(f"  Saved to: {output_dir}")

        # Save metadata if requested
        if save_metadata_json:
            if output_dir is None:
                output_dir = video_path.parent / f"{video_path.stem}_selected"
                output_dir = Path(output_dir)
            else:
                output_dir = Path(output_dir)

            metadata_path = output_dir / "selection_metadata.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_metadata(
                selected_indices=selected_indices,
                metrics=metrics,
                output_path=metadata_path,
                video_path=video_path
            )

        return selected_indices, metrics

    def process_image_directory(
        self,
        image_dir: Union[str, Path],
        image_extensions: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        selection_threshold: float = 0.5,
        output_dir: Optional[Union[str, Path]] = None,
        save_selected_frames: bool = False,
        save_metadata_json: bool = False,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Process a directory of images and select keyframes.

        Args:
            image_dir: Path to directory containing images
            image_extensions: List of extensions to include (e.g., ['.jpg', '.png'])
            max_images: Maximum number of images to process
            selection_threshold: Threshold for frame selection (0-1)
            output_dir: Directory to save/copy selected images
            save_selected_frames: Whether to copy selected images to output_dir
            save_metadata_json: Whether to save frame selection metadata to JSON file

        Returns:
            Tuple of (selected_image_indices, metrics_dict)
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Default extensions
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        # Find all images
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))

        image_paths = sorted(image_paths)

        if max_images is not None:
            image_paths = image_paths[:max_images]

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        # Reset state
        self.reset()

        if self.verbose:
            print(f"\nProcessing image directory: {image_dir}")
            print(f"  Total images: {len(image_paths)}")
            print(f"  Selection threshold: {selection_threshold}")

        # Setup output directory if saving
        if save_selected_frames:
            if output_dir is None:
                output_dir = image_dir.parent / f"{image_dir.name}_selected"
            output_dir = Path(output_dir)
            # Remove and recreate directory to ensure clean state
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        selected_indices = []
        selected_paths = []
        all_scores = []
        all_components = []

        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images", disable=not self.verbose)):
            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Failed to read {img_path}")
                continue

            # Check if frame should be selected
            should_select, score, components = self.should_select_frame(
                frame, threshold=selection_threshold
            )

            all_scores.append(score)
            all_components.append(components)

            if should_select:
                selected_indices.append(idx)
                selected_paths.append(img_path)

                # Update keyframe reference
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._update_keyframe(gray)

                # Copy/save frame if requested
                if save_selected_frames:
                    import shutil
                    output_path = output_dir / img_path.name
                    shutil.copy(img_path, output_path)

        # Compile metrics
        metrics = {
            'total_images': len(image_paths),
            'selected_images': len(selected_indices),
            'selection_rate': len(selected_indices) / len(image_paths) if len(image_paths) > 0 else 0,
            'selected_paths': selected_paths,
            'scores': all_scores,
            'components': all_components,
            'stats': self.stats,
        }

        if self.verbose:
            print(f"\nSelection summary:")
            print(f"  Total images: {metrics['total_images']}")
            print(f"  Selected images: {metrics['selected_images']}")
            print(f"  Selection rate: {metrics['selection_rate']:.1%}")
            if save_selected_frames:
                print(f"  Saved to: {output_dir}")

        # Save metadata if requested
        if save_metadata_json:
            if output_dir is None:
                output_dir = image_dir.parent / f"{image_dir.name}_selected"
                output_dir = Path(output_dir)
            else:
                output_dir = Path(output_dir)

            metadata_path = output_dir / "selection_metadata.json"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_metadata(
                selected_indices=selected_indices,
                metrics=metrics,
                output_path=metadata_path,
                image_paths=image_paths
            )

        return selected_indices, metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_optical_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    max_flow_magnitude: float = 100.0,
) -> np.ndarray:
    """
    Visualize optical flow vectors on an image.

    Args:
        prev_frame: Previous BGR frame (H, W, 3)
        curr_frame: Current BGR frame (H, W, 3)
        prev_pts: Previous point positions (N, 2)
        curr_pts: Current point positions (N, 2)
        max_flow_magnitude: Maximum flow magnitude for color normalization

    Returns:
        Visualization image (H, W, 3) in BGR format
    """
    # Create output image
    vis = curr_frame.copy()

    # Draw flow vectors
    for i, (p1, p2) in enumerate(zip(prev_pts, curr_pts)):
        x1, y1 = p1.astype(int)
        x2, y2 = p2.astype(int)

        # Compute flow magnitude for color coding
        magnitude = np.linalg.norm(p2 - p1)
        normalized_mag = min(magnitude / max_flow_magnitude, 1.0)

        # Color: blue (small motion) -> green -> red (large motion)
        color_val = int(normalized_mag * 255)
        color = (255 - color_val, color_val, 0)

        # Draw line and circle
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, 2, tipLength=0.3)
        cv2.circle(vis, (x2, y2), 3, color, -1)

    return vis


def visualize_flow_between_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    selector: OpticalFlowFrameSelector
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Visualize optical flow between two frames using the selector.

    Args:
        frame1: First BGR frame (H, W, 3)
        frame2: Second BGR frame (H, W, 3)
        selector: OpticalFlowFrameSelector instance

    Returns:
        Tuple of (flow_visualization, disparity, num_features)
        flow_visualization is None if no features could be tracked
    """
    if frame1 is None or frame2 is None:
        return None, 0.0, 0

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features in first frame
    pts1 = selector._detect_features(gray1)

    if pts1 is not None and len(pts1) > 0:
        # Compute optical flow
        prev_pts, curr_pts = selector._compute_optical_flow(gray1, gray2, pts1)

        if prev_pts is not None and curr_pts is not None and len(prev_pts) > 0:
            # Visualize flow
            flow_vis = visualize_optical_flow(frame1, frame2, prev_pts, curr_pts)

            # Compute disparity
            disparity = selector._compute_disparity(prev_pts, curr_pts)

            return flow_vis, disparity, len(prev_pts)

    return frame2, 0.0, 0


def test_parameter_configuration(
    frames: List[np.ndarray],
    config_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Test a specific parameter configuration on a set of frames.

    Args:
        frames: List of BGR frames to process
        config_name: Name of the configuration (for display purposes)
        **kwargs: Parameters to pass to OpticalFlowFrameSelector

    Returns:
        Dictionary containing:
        - selected: List of selected frame indices
        - scores: Combined scores for all frames
        - motion_scores: Motion component scores
        - coverage_scores: Coverage component scores
        - disparities: Disparity values
    """
    selector_test = OpticalFlowFrameSelector(verbose=False, **kwargs)

    selected = []
    scores = []
    motion_scores = []
    coverage_scores = []
    disparities = []

    for i, frame in enumerate(frames):
        should_select, score, components = selector_test.should_select_frame(frame, threshold=0.5)
        scores.append(score)
        motion_scores.append(components.get('motion', 0))
        coverage_scores.append(components.get('coverage', 0))
        disparities.append(components.get('disparity', 0))

        if should_select:
            selected.append(i)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            selector_test._update_keyframe(gray)

    return {
        'selected': selected,
        'scores': scores,
        'motion_scores': motion_scores,
        'coverage_scores': coverage_scores,
        'disparities': disparities
    }


def create_configuration_comparison_plot(
    results: Dict[str, Dict[str, Any]],
    total_frames: int,
    configs: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (16, 12)
) -> Any:
    """
    Create a comprehensive comparison plot for different configurations.

    Args:
        results: Dictionary mapping config names to test results
        total_frames: Total number of frames tested
        configs: Dictionary mapping config names to parameter dictionaries
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(configs) + 1, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('Configuration Comparison: Frame Selection Behavior', fontsize=16, fontweight='bold')

    # Plot individual configuration results
    for idx, (config_name, result) in enumerate(results.items()):
        # Left column: Selection scores over time
        ax1 = fig.add_subplot(gs[idx, 0])
        frame_indices = np.arange(len(result['scores']))

        ax1.plot(frame_indices, result['scores'], 'b-', alpha=0.7, linewidth=2, label='Combined Score')
        ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax1.scatter(result['selected'], [result['scores'][i] for i in result['selected']],
                   c='green', s=80, marker='o', label='Selected', zorder=5,
                   edgecolors='darkgreen', linewidths=2)

        ax1.set_xlabel('Sample Frame Index', fontsize=10)
        ax1.set_ylabel('Selection Score', fontsize=10)
        selection_pct = len(result['selected'])/total_frames*100
        ax1.set_title(f"{config_name}\n({len(result['selected'])}/{total_frames} frames = {selection_pct:.1f}%)",
                     fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.1, 1.1])

        # Right column: Motion vs Coverage contribution
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.plot(frame_indices, result['motion_scores'], 'orange', alpha=0.7,
                linewidth=2, label='Motion Score')
        ax2.plot(frame_indices, result['coverage_scores'], 'purple', alpha=0.7,
                linewidth=2, label='Coverage Score')
        ax2.scatter(result['selected'], [result['motion_scores'][i] for i in result['selected']],
                   c='green', s=60, marker='o', zorder=5, edgecolors='darkgreen', linewidths=1.5)
        ax2.scatter(result['selected'], [result['coverage_scores'][i] for i in result['selected']],
                   c='green', s=60, marker='o', zorder=5, edgecolors='darkgreen', linewidths=1.5)

        ax2.set_xlabel('Sample Frame Index', fontsize=10)
        ax2.set_ylabel('Component Score', fontsize=10)
        ax2.set_title('Motion vs Coverage Components', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.1, 1.1])

    # Bottom: Comparison bar chart
    ax_compare = fig.add_subplot(gs[-1, :])
    config_names = list(results.keys())
    selection_rates = [len(results[name]['selected'])/total_frames*100 for name in config_names]
    mean_disparities = [np.mean(results[name]['disparities']) for name in config_names]

    x = np.arange(len(config_names))
    width = 0.35

    bars1 = ax_compare.bar(x - width/2, selection_rates, width, label='Selection Rate (%)',
                          color='skyblue', edgecolor='navy', linewidth=2)
    ax_compare_twin = ax_compare.twinx()
    bars2 = ax_compare_twin.bar(x + width/2, mean_disparities, width,
                                label='Mean Disparity (px)', color='lightcoral',
                                edgecolor='darkred', linewidth=2)

    ax_compare.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax_compare.set_ylabel('Selection Rate (%)', fontsize=10, color='navy', fontweight='bold')
    ax_compare_twin.set_ylabel('Mean Disparity (px)', fontsize=10, color='darkred', fontweight='bold')
    ax_compare.set_title('Configuration Comparison Summary', fontsize=12, fontweight='bold')
    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels(config_names, rotation=15, ha='right')
    ax_compare.legend(loc='upper left', fontsize=9)
    ax_compare_twin.legend(loc='upper right', fontsize=9)
    ax_compare.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_compare.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax_compare_twin.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}px', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def create_selection_summary_plot(
    metrics: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> Any:
    """
    Create a comprehensive visualization of frame selection metrics.

    Args:
        metrics: Metrics dictionary from process_video() or process_image_directory()
        output_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data
    scores = metrics['scores']
    components = metrics['components']
    stats = metrics['stats']

    frame_indices = np.arange(len(scores))
    selected_indices = [i for i, s in enumerate(scores) if s >= 0.5]

    # 1. Combined scores over time
    ax = axes[0, 0]
    ax.plot(frame_indices, scores, 'b-', alpha=0.7, label='Combined Score')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    ax.scatter(selected_indices, [scores[i] for i in selected_indices],
               c='green', s=50, label='Selected', zorder=5)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Selection Score')
    ax.set_title('Frame Selection Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Motion vs Coverage components
    ax = axes[0, 1]
    motion_scores = [c['motion'] for c in components]
    coverage_scores = [c['coverage'] for c in components]
    ax.plot(frame_indices, motion_scores, 'r-', alpha=0.7, label='Motion')
    ax.plot(frame_indices, coverage_scores, 'b-', alpha=0.7, label='Coverage')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Component Score')
    ax.set_title('Motion vs Coverage Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Disparity histogram
    ax = axes[1, 0]
    if len(stats['disparities']) > 0:
        ax.hist(stats['disparities'], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.median(stats['disparities']), color='r',
                   linestyle='--', label=f"Median: {np.median(stats['disparities']):.1f}")
        ax.set_xlabel('Disparity (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Disparity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Selection statistics
    ax = axes[1, 1]
    ax.axis('off')

    total = metrics['total_frames'] if 'total_frames' in metrics else metrics['total_images']
    selected = metrics['selected_frames'] if 'selected_frames' in metrics else metrics['selected_images']
    selection_rate = metrics['selection_rate']

    stats_text = f"""
    Selection Summary
    ─────────────────────────
    Total Frames:      {total}
    Selected Frames:   {selected}
    Selection Rate:    {selection_rate:.1%}

    Motion Statistics
    ─────────────────────────
    """

    if len(stats['disparities']) > 0:
        stats_text += f"""
    Disparity (pixels):
      Mean:     {np.mean(stats['disparities']):.1f}
      Median:   {np.median(stats['disparities']):.1f}
      Std:      {np.std(stats['disparities']):.1f}
    """

    if len(stats['rotations']) > 0:
        stats_text += f"""
    Rotation (degrees):
      Mean:     {np.mean(stats['rotations']):.1f}
      Median:   {np.median(stats['rotations']):.1f}
      Max:      {np.max(stats['rotations']):.1f}
    """

    if len(stats['histogram_similarities']) > 0:
        stats_text += f"""
    Histogram Similarity:
      Mean:     {np.mean(stats['histogram_similarities']):.3f}
      Median:   {np.median(stats['histogram_similarities']):.3f}
    """

    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to: {output_path}")

    return fig


# ============================================================================
# Export Utilities
# ============================================================================

def save_selected_indices(
    selected_indices: List[int],
    output_path: Union[str, Path]
) -> None:
    """
    Save selected frame indices to a text file.

    Args:
        selected_indices: List of frame indices
        output_path: Path to output text file
    """
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    print(f"Saved {len(selected_indices)} indices to {output_path}")


def create_copy_script(
    image_paths: List[Path],
    selected_indices: List[int],
    output_script: Union[str, Path] = "copy_selected.sh"
) -> None:
    """
    Create a bash script to copy selected images.

    Args:
        image_paths: List of all image paths
        selected_indices: Indices of images to copy
        output_script: Path to output shell script
    """
    output_script = Path(output_script)
    with open(output_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("mkdir -p selected_frames\n")
        for idx in selected_indices:
            f.write(f"cp {image_paths[idx]} selected_frames/\n")

    import os
    os.chmod(output_script, 0o755)
    print(f"Created copy script: {output_script}")


def save_metadata(
    selected_indices: List[int],
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
    video_path: Optional[Union[str, Path]] = None,
    image_paths: Optional[List[Path]] = None,
) -> None:
    """
    Save frame selection metadata to a JSON file.

    This includes all metrics, scores, selected frame indices, and optionally
    the original file paths.

    Args:
        selected_indices: List of selected frame/image indices
        metrics: Metrics dictionary from process_video() or process_image_directory()
        output_path: Path to output JSON file
        video_path: Optional path to the source video file
        image_paths: Optional list of image paths (for image directory processing)
    """
    output_path = Path(output_path)

    # Build metadata dictionary
    metadata = {
        'selected_frame_indices': selected_indices,
        'total_frames': metrics.get('total_frames', metrics.get('total_images', 0)),
        'selected_frames': metrics.get('selected_frames', metrics.get('selected_images', 0)),
        'selection_rate': metrics['selection_rate'],
        'scores': metrics['scores'],
        'components': metrics['components'],
        'stats': {
            'disparities': metrics['stats']['disparities'],
            'rotations': metrics['stats']['rotations'],
            'histogram_similarities': metrics['stats']['histogram_similarities'],
        }
    }

    # Add source file information
    if video_path is not None:
        metadata['source_video'] = str(video_path)

    if image_paths is not None:
        # Store selected image paths
        selected_image_paths = [str(image_paths[i]) for i in selected_indices]
        metadata['selected_image_paths'] = selected_image_paths
        metadata['all_image_paths'] = [str(p) for p in image_paths]

    # Add statistics summary
    if len(metrics['stats']['disparities']) > 0:
        metadata['summary_statistics'] = {
            'disparity': {
                'mean': float(np.mean(metrics['stats']['disparities'])),
                'median': float(np.median(metrics['stats']['disparities'])),
                'std': float(np.std(metrics['stats']['disparities'])),
                'min': float(np.min(metrics['stats']['disparities'])),
                'max': float(np.max(metrics['stats']['disparities'])),
            },
            'rotation': {
                'mean': float(np.mean(metrics['stats']['rotations'])) if len(metrics['stats']['rotations']) > 0 else 0.0,
                'median': float(np.median(metrics['stats']['rotations'])) if len(metrics['stats']['rotations']) > 0 else 0.0,
                'max': float(np.max(metrics['stats']['rotations'])) if len(metrics['stats']['rotations']) > 0 else 0.0,
            },
            'histogram_similarity': {
                'mean': float(np.mean(metrics['stats']['histogram_similarities'])) if len(metrics['stats']['histogram_similarities']) > 0 else 0.0,
                'median': float(np.median(metrics['stats']['histogram_similarities'])) if len(metrics['stats']['histogram_similarities']) > 0 else 0.0,
            }
        }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_path}")
