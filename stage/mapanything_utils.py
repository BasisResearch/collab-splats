"""
MapAnything Utilities for COLMAP Preprocessing

This module provides modular functions for using MapAnything to perform
image-only structure-from-motion preprocessing with conversion to COLMAP format.

Main Functions:
1. load_mapanything_model: Load MapAnything model from HuggingFace
2. load_and_preprocess_images: Load and preprocess images for MapAnything
3. run_mapanything_inference: Run MapAnything inference to get poses and depth
4. export_to_colmap: Export MapAnything outputs to COLMAP format
5. rescale_to_original_dimensions: Rescale COLMAP reconstruction to original image sizes
6. convert_to_nerfstudio_format: Convert COLMAP to nerfstudio transforms.json

Usage:
    # Load model
    model = load_mapanything_model()

    # Load images
    views, image_paths = load_and_preprocess_images(image_dir)

    # Run inference
    outputs = run_mapanything_inference(model, views)

    # Export to COLMAP
    export_to_colmap(outputs, views, image_names, output_dir, model)

    # Rescale to original dimensions
    rescale_to_original_dimensions(colmap_dir, image_paths, model_width, model_height, output_dir)

    # Convert to nerfstudio format
    convert_to_nerfstudio_format(colmap_dir, output_dir)
"""

import os
import json
import shutil
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import pycolmap
from tqdm import tqdm

from nerfstudio.process_data import colmap_utils
from nerfstudio.utils.rich_utils import CONSOLE


# ============================================================================
# Model Loading
# ============================================================================

def load_mapanything_model(
    model_name: str = "facebook/map-anything",
    device: Optional[str] = None,
    enable_optimizations: bool = True,
    verbose: bool = True,
    **kwargs
) -> Any:
    """Load MapAnything model from HuggingFace.

    Args:
        model_name: HuggingFace model name. Options:
            - "facebook/map-anything" (default): CC-BY-NC 4.0 license
            - "facebook/map-anything-apache": Apache 2.0 license
        device: Device to load model on. If None, uses CUDA if available.
        enable_optimizations: Whether to enable CUDA optimizations
        verbose: Whether to print loading information
        **kwargs: Additional arguments passed to MapAnything.from_pretrained()

    Returns:
        Loaded MapAnything model in eval mode

    Raises:
        RuntimeError: If CUDA is not available
        ImportError: If MapAnything is not installed
    """
    try:
        from mapanything.models import MapAnything
    except ImportError as e:
        raise ImportError(
            "MapAnything not installed. Please install with:\n"
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MapAnything inference but is not available")

    # Enable CUDA optimizations
    if enable_optimizations and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.95)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if verbose:
        CONSOLE.print(f"[bold cyan]Loading MapAnything model: {model_name}")
        if device == "cuda":
            CONSOLE.print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = MapAnything.from_pretrained(model_name, **kwargs).to(device)
    model.eval()

    if verbose:
        CONSOLE.print(f"[bold green]✓ Model loaded successfully")

    return model


# ============================================================================
# Image Loading and Preprocessing
# ============================================================================

def load_and_preprocess_images(
    image_dir: Union[str, Path],
    image_extensions: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    sort: bool = True,
    verbose: bool = True,
    **kwargs
) -> Tuple[List[Dict], List[Path]]:
    """Load and preprocess images from directory for MapAnything.

    Args:
        image_dir: Path to directory containing images
        image_extensions: List of file extensions to include (e.g., ["*.jpg", "*.png"])
            If None, uses default: ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        max_images: Maximum number of images to load (for debugging)
        sort: Whether to sort image paths alphabetically
        verbose: Whether to print progress information
        **kwargs: Additional arguments passed to MapAnything's load_images()

    Returns:
        Tuple of (views, image_paths) where:
            - views: List of preprocessed view dictionaries for MapAnything
            - image_paths: List of Path objects for the images

    Raises:
        ValueError: If no images are found in the directory
        ImportError: If MapAnything utilities are not installed
    """
    try:
        from mapanything.utils.image import load_images
    except ImportError as e:
        raise ImportError(
            "MapAnything not installed. Please install with:\n"
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    # Default image extensions
    if image_extensions is None:
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(ext))

    if sort:
        image_paths = sorted(image_paths)

    if max_images is not None and max_images < len(image_paths):
        # Sample image_paths at even intervals
        # image_paths = image_paths[:max_images]
        indices = np.linspace(0, len(image_paths) - 1, max_images, dtype=int)
        image_paths = [image_paths[i] for i in indices]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir} with extensions {image_extensions}")

    if verbose:
        CONSOLE.print(f"[bold cyan]Loading images from: {image_dir}")
        CONSOLE.print(f"  Found {len(image_paths)} images")

    # Convert to string paths for MapAnything
    image_path_strings = [str(p) for p in image_paths]

    # Load and preprocess with MapAnything
    views = load_images(image_path_strings, **kwargs)

    if verbose:
        CONSOLE.print(f"[bold green]✓ Images loaded and preprocessed")
        if len(views) > 0 and 'img' in views[0]:
            CONSOLE.print(f"  Image shape: {views[0]['img'].shape}")

    return views, image_paths


# ============================================================================
# MapAnything Inference
# ============================================================================

def run_mapanything_inference(
    model: Any,
    views: List[Dict],
    memory_efficient_inference: bool = True,
    minibatch_size: int = 1,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    apply_mask: bool = True,
    mask_edges: bool = True,
    apply_confidence_mask: bool = True,
    use_multiview_confidence: bool = True,
    confidence_percentile: float = 35.0,
    verbose: bool = True,
    **kwargs
) -> List[Dict]:
    """Run MapAnything inference to get camera poses, depth maps, and 3D points.

    Args:
        model: Loaded MapAnything model
        views: Preprocessed views from load_and_preprocess_images()
        memory_efficient_inference: Use memory-efficient mode
        minibatch_size: Process this many images at a time (1 = most memory efficient)
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype, "bf16" for Ampere+ GPUs, "fp16" for older GPUs
        apply_mask: Apply geometric validity mask
        mask_edges: Mask image edges for better quality
        apply_confidence_mask: Apply confidence-based point filtering
        use_multiview_confidence: Use multi-view consistency for confidence
        confidence_percentile: Confidence percentile threshold (0-100)
        verbose: Whether to print progress and timing information
        **kwargs: Additional arguments passed to model.infer()

    Returns:
        List of output dictionaries containing:
            - Camera poses
            - Depth maps
            - Confidence scores
            - 3D points
            - And other MapAnything outputs
    """
    if verbose:
        CONSOLE.print(f"[bold cyan]Running MapAnything inference")
        CONSOLE.print(f"  Frames: {len(views)}")
        CONSOLE.print(f"  Memory efficient: {memory_efficient_inference}")
        CONSOLE.print(f"  Minibatch size: {minibatch_size}")
        CONSOLE.print(f"  Mixed precision: {use_amp} ({amp_dtype})")

    # Track timing and memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()

    # Run inference
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            apply_confidence_mask=apply_confidence_mask,
            use_multiview_confidence=use_multiview_confidence,
            confidence_percentile=confidence_percentile,
            **kwargs
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    inference_time = time.time() - start_time

    if verbose:
        CONSOLE.print(f"[bold green]✓ Inference complete")
        CONSOLE.print(f"  Total time: {inference_time:.2f}s")
        CONSOLE.print(f"  Time per frame: {inference_time / len(views):.3f}s")
        CONSOLE.print(f"  FPS: {len(views) / inference_time:.2f}")

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            CONSOLE.print(f"  Peak GPU memory: {peak_memory_gb:.2f} GB")

    return outputs


# ============================================================================
# COLMAP Export - Helper Functions
# ============================================================================

def filter_points_by_spatial_extent(
    points: np.ndarray,
    colors: np.ndarray,
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    max_extent: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points to a percentile-based bounding box.

    This removes spatial outliers that fall outside the specified percentile range.
    Useful for removing noisy distant points or invalid reconstructions.

    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors
        percentile_range: (min, max) percentiles for bounding box (default: 1-99%)
            For example, (1.0, 99.0) keeps points between 1st and 99th percentiles
        max_extent: Optional absolute max extent in meters (applied after percentile filtering)
            If set, clips the bounding box to this size around the center
        verbose: Whether to print filtering information

    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if len(points) == 0:
        return points, colors

    # Compute percentile-based bounds
    pmin, pmax = percentile_range
    bbox_min = np.percentile(points, pmin, axis=0)
    bbox_max = np.percentile(points, pmax, axis=0)

    # Optionally clip to absolute max extent from center
    if max_extent is not None:
        center = (bbox_min + bbox_max) / 2
        half_extent = max_extent / 2
        bbox_min = np.maximum(bbox_min, center - half_extent)
        bbox_max = np.minimum(bbox_max, center + half_extent)

    # Filter points within bounding box
    mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)

    filtered_points = points[mask]
    filtered_colors = colors[mask]

    if verbose:
        extent = (bbox_max - bbox_min).max()
        CONSOLE.print(f"[bold yellow]Spatial filtering ({pmin}-{pmax} percentile):")
        CONSOLE.print(f"  Bounding box extent: {extent:.3f}m")
        CONSOLE.print(f"  Filtered from {len(points)} to {len(filtered_points)} points")
        removed_count = len(points) - len(filtered_points)
        removed_pct = 100 * (1 - len(filtered_points)/len(points)) if len(points) > 0 else 0
        CONSOLE.print(f"  Removed {removed_count} outliers ({removed_pct:.1f}%)")

    return filtered_points, filtered_colors


def voxel_downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_fraction: float = 0.01,
    voxel_size: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud with scene-adaptive or explicit voxel size.

    If voxel_size is provided, it is used directly. Otherwise, the voxel size
    is computed adaptively using the interquartile range (IQR) of point positions:
        voxel_size = iqr_extent * voxel_fraction

    Using IQR instead of full bounding box extent makes the method robust to
    outliers and large depth variations (e.g., landscape scenes with 1m to 1000m depth).

    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255 uint8 or 0-1 float)
        voxel_fraction: Fraction of IQR extent to use as voxel size (default: 0.01 = 1%)
        voxel_size: Explicit voxel size in meters (overrides voxel_fraction if provided)
        verbose: Whether to print downsampling information

    Returns:
        Tuple of (downsampled_points, downsampled_colors)
            - downsampled_points: (M, 3) array of downsampled 3D points
            - downsampled_colors: (M, 3) array of corresponding colors (uint8)

    Raises:
        ImportError: If open3d is not installed
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "open3d is required for voxel downsampling. "
            "Install it with: pip install open3d"
        ) from e

    if len(points) == 0:
        return points, colors

    if voxel_size is not None:
        # Use explicit voxel size
        if verbose:
            CONSOLE.print(f"  Using explicit voxel size: {voxel_size:.4f}m")
    else:
        # Compute scene extent using IQR (robust to outliers)
        q25 = np.percentile(points, 25, axis=0)
        q75 = np.percentile(points, 75, axis=0)
        iqr_extent = (q75 - q25).max()

        # Also compute full extent for reference
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        full_extent = (bbox_max - bbox_min).max()

        # Use IQR-based extent if valid, otherwise fall back to full extent
        if iqr_extent > 0:
            # Scale up IQR to approximate useful scene range
            # IQR covers ~50% of data, so multiply by 2 for better coverage
            scene_extent = iqr_extent * 2
        else:
            scene_extent = full_extent

        # Compute adaptive voxel size
        voxel_size = scene_extent * voxel_fraction

        # Ensure voxel size is positive
        if voxel_size <= 0:
            voxel_size = 0.01  # Fallback to 1cm if extent is zero

        if verbose:
            CONSOLE.print(f"  Scene extent (IQR-based): {scene_extent:.3f}m, full extent: {full_extent:.3f}m")
            CONSOLE.print(f"  Adaptive voxel size: {voxel_size:.4f}m")

    # Normalize colors to [0, 1] if needed
    if colors.dtype == np.uint8:
        colors_normalized = colors.astype(np.float64) / 255.0
    else:
        colors_normalized = colors.astype(np.float64)
        if colors_normalized.max() > 1.0:
            colors_normalized = colors_normalized / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    # Voxel downsample
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    # Extract downsampled points and colors
    downsampled_points = np.asarray(pcd_downsampled.points)
    downsampled_colors = (np.asarray(pcd_downsampled.colors) * 255).astype(np.uint8)

    if verbose:
        CONSOLE.print(f"  Downsampled from {len(points)} to {len(downsampled_points)} points")

    return downsampled_points, downsampled_colors


def backproject_points_to_frames(
    points_3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
) -> List[List[Tuple[int, float, float]]]:
    """Backproject 3D points to all frames to find Point2D observations.

    For each frame, this function:
    1. Transforms points to camera space (vectorized)
    2. Frustum culls points with z <= 0
    3. Projects to 2D with intrinsics
    4. Filters points outside image bounds

    Args:
        points_3d: (P, 3) array of 3D points in world coordinates
        extrinsics: (N, 3, 4) array of world2cam transforms [R|t]
        intrinsics: (N, 3, 3) array of camera intrinsic matrices
        image_width: Width of images in pixels
        image_height: Height of images in pixels

    Returns:
        List of length N, where each element is a list of (point3D_id, u, v) tuples
        for points that project into that frame. point3D_id is 1-indexed for COLMAP.
    """
    num_frames = extrinsics.shape[0]
    num_points = points_3d.shape[0]

    # Convert points to homogeneous coordinates (P, 4)
    points_homo = np.hstack([points_3d, np.ones((num_points, 1))])

    observations_per_frame = []

    for frame_idx in range(num_frames):
        # Get camera parameters
        ext = extrinsics[frame_idx]  # (3, 4)
        K = intrinsics[frame_idx]  # (3, 3)

        # Transform to camera space: X_cam = [R|t] @ X_world_homo
        points_cam = ext @ points_homo.T  # (3, P)

        # Frustum cull: keep points in front of camera (z > 0)
        z = points_cam[2, :]
        in_front_mask = z > 0

        if not np.any(in_front_mask):
            observations_per_frame.append([])
            continue

        # Get valid points and their indices
        valid_indices = np.where(in_front_mask)[0]
        points_cam_valid = points_cam[:, in_front_mask]  # (3, M)
        z_valid = z[in_front_mask]

        # Project to 2D: uv_homo = K @ X_cam
        uv_homo = K @ points_cam_valid  # (3, M)

        # Normalize by z to get pixel coordinates
        u = uv_homo[0, :] / z_valid
        v = uv_homo[1, :] / z_valid

        # Bounds check
        in_bounds_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

        # Collect valid observations
        frame_observations = []
        for i, (idx, uu, vv) in enumerate(
            zip(valid_indices[in_bounds_mask], u[in_bounds_mask], v[in_bounds_mask])
        ):
            # point3D_id is 1-indexed for COLMAP
            point3d_id = int(idx) + 1
            frame_observations.append((point3d_id, float(uu), float(vv)))

        observations_per_frame.append(frame_observations)

    return observations_per_frame


def _build_pycolmap_intrinsics(
    intrinsics: np.ndarray,
    camera_type: str = "PINHOLE",
) -> np.ndarray:
    """Build pycolmap camera intrinsics array from 3x3 intrinsic matrix.

    Args:
        intrinsics: (3, 3) camera intrinsic matrix
        camera_type: Camera model type ("PINHOLE" or "SIMPLE_PINHOLE")

    Returns:
        numpy array of camera parameters for pycolmap

    Raises:
        ValueError: If camera_type is not supported
    """
    if camera_type == "PINHOLE":
        # [fx, fy, cx, cy]
        return np.array([
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
        ])
    elif camera_type == "SIMPLE_PINHOLE":
        # [f, cx, cy] - use average of fx and fy
        focal = (intrinsics[0, 0] + intrinsics[1, 1]) / 2
        return np.array([
            focal,
            intrinsics[0, 2],
            intrinsics[1, 2],
        ])
    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")


def build_colmap_reconstruction(
    points_3d: np.ndarray,
    points_rgb: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: Optional[List[str]] = None,
    camera_type: str = "PINHOLE",
    skip_point2d: bool = False,
    verbose: bool = True,
) -> pycolmap.Reconstruction:
    """Build a pycolmap Reconstruction from MapAnything outputs.

    This function:
    1. Creates cameras and images with poses
    2. Adds 3D points
    3. Backprojects points to get Point2D observations (unless skip_point2d=True)
    4. Links Point2D observations to 3D points via tracks

    Args:
        points_3d: (P, 3) array of 3D points in world coordinates
        points_rgb: (P, 3) array of RGB colors (uint8)
        extrinsics: (N, 3, 4) array of world2cam transforms [R|t]
        intrinsics: (N, 3, 3) array of camera intrinsic matrices
        image_width: Width of images in pixels
        image_height: Height of images in pixels
        image_names: Optional list of image names. If None, uses "image_N.jpg"
        camera_type: Camera model type ("PINHOLE" or "SIMPLE_PINHOLE")
        skip_point2d: If True, skip Point2D backprojection for faster export
        verbose: Whether to print progress information

    Returns:
        pycolmap.Reconstruction object
    """
    num_frames = extrinsics.shape[0]
    num_points = points_3d.shape[0]

    # Generate default image names if not provided
    if image_names is None:
        image_names = [f"image_{i + 1}.jpg" for i in range(num_frames)]

    # Backproject to get Point2D observations (unless skipped)
    if skip_point2d:
        if verbose:
            CONSOLE.print("  Skipping Point2D backprojection...")
        observations_per_frame = [[] for _ in range(num_frames)]
    else:
        if verbose:
            CONSOLE.print("  Backprojecting points to frames...")
        observations_per_frame = backproject_points_to_frames(
            points_3d, extrinsics, intrinsics, image_width, image_height
        )

    # Create reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add 3D points with empty tracks (will be populated later)
    for point_idx in range(num_points):
        point3d_id = point_idx + 1  # 1-indexed
        reconstruction.add_point3D(
            points_3d[point_idx],
            pycolmap.Track(),
            points_rgb[point_idx],
        )

    # Add cameras and images
    for frame_idx in range(num_frames):
        # Build camera intrinsics
        cam_params = _build_pycolmap_intrinsics(intrinsics[frame_idx], camera_type)

        # Create camera
        camera = pycolmap.Camera(
            model=camera_type,
            width=image_width,
            height=image_height,
            params=cam_params,
            camera_id=frame_idx + 1,
        )
        reconstruction.add_camera(camera)

        # Create image with pose
        ext = extrinsics[frame_idx]  # (3, 4)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(ext[:3, :3]),
            ext[:3, 3],
        )

        image = pycolmap.Image(
            id=frame_idx + 1,
            name=image_names[frame_idx],
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        # Build Point2D list and update tracks
        points2d_list = []
        frame_observations = observations_per_frame[frame_idx]

        for point2d_idx, (point3d_id, u, v) in enumerate(frame_observations):
            # Create Point2D
            points2d_list.append(pycolmap.Point2D(np.array([u, v]), point3d_id))

            # Update track for this 3D point
            track = reconstruction.points3D[point3d_id].track
            track.add_element(frame_idx + 1, point2d_idx)

        # Set points2D on image
        if points2d_list:
            try:
                image.points2D = pycolmap.ListPoint2D(points2d_list)
                image.registered = True
            except Exception as e:
                if verbose:
                    CONSOLE.print(f"[yellow]Warning: Failed to set points2D for frame {frame_idx}: {e}")
                image.registered = False
        else:
            image.registered = True  # Still registered, just no observations

        reconstruction.add_image(image)

    # Print summary
    if verbose:
        total_observations = sum(len(obs) for obs in observations_per_frame)
        CONSOLE.print(f"  Built COLMAP reconstruction:")
        CONSOLE.print(f"    - {num_frames} images")
        CONSOLE.print(f"    - {num_points} 3D points")
        CONSOLE.print(f"    - {total_observations} Point2D observations")

    return reconstruction


def export_predictions_to_colmap_internal(
    outputs: List[Dict],
    processed_views: List[Dict],
    image_names: List[str],
    output_dir: Union[str, Path],
    voxel_fraction: float = 0.01,
    voxel_size: Optional[float] = None,
    spatial_filter_percentile: Optional[Tuple[float, float]] = None,
    spatial_filter_max_extent: Optional[float] = None,
    save_ply: bool = True,
    save_images: bool = True,
    skip_point2d: bool = False,
    verbose: bool = True,
) -> pycolmap.Reconstruction:
    """Export MapAnything predictions to COLMAP format (internal implementation).

    This is the main entry point for COLMAP export. It:
    1. Collects 3D points and colors from all views
    2. Optionally filters by spatial extent (removes outliers)
    3. Applies scene-adaptive voxel downsampling
    4. Builds COLMAP reconstruction with proper Point2D observations
    5. Saves to disk (including processed images if requested)

    Args:
        outputs: List of prediction dictionaries from model.infer()
        processed_views: List of preprocessed view dictionaries
        image_names: List of original image file names
        output_dir: Directory to save COLMAP outputs
        voxel_fraction: Fraction of IQR-based scene extent for voxel size (default: 0.01 = 1%)
        voxel_size: Explicit voxel size in meters (overrides voxel_fraction if provided)
        spatial_filter_percentile: Optional (min, max) percentile range for spatial filtering
            e.g., (1.0, 99.0) to remove top/bottom 1% outliers
        spatial_filter_max_extent: Optional absolute max extent in meters
        save_ply: Whether to save a PLY file of the point cloud
        save_images: Whether to save processed images to output_dir/images/
        skip_point2d: If True, skip Point2D backprojection for faster export
        verbose: Whether to print progress information

    Returns:
        pycolmap.Reconstruction object
    """
    try:
        import trimesh
        from mapanything.utils.geometry import closed_form_pose_inverse
    except ImportError as e:
        raise ImportError(
            "MapAnything utilities required. Install with:\n"
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    output_dir = Path(output_dir)
    num_frames = len(outputs)

    # Collect data from outputs
    all_points = []
    all_colors = []
    intrinsics_list = []
    extrinsics_list = []

    if verbose:
        CONSOLE.print(f"[bold cyan]Collecting 3D points from {num_frames} frames...")

    for i in range(num_frames):
        pred = outputs[i]

        # Get 3D points and mask
        pts3d = pred["pts3d"][0].cpu().numpy()  # (H, W, 3)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

        # Filter by valid depth (camera Z, not world Z)
        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()  # (H, W)
        valid_depth_mask = depth_z > 0
        combined_mask = mask & valid_depth_mask

        # Get colors from denormalized image
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()  # (H, W, 3) in [0, 1]
        colors = (img_no_norm * 255).astype(np.uint8)

        # Collect valid points and colors
        all_points.append(pts3d[combined_mask])
        all_colors.append(colors[combined_mask])

        # Collect camera parameters
        intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())
        # Convert cam2world to world2cam for COLMAP
        cam2world = pred["camera_poses"][0].cpu().numpy()
        world2cam = closed_form_pose_inverse(cam2world[None])[0]
        extrinsics_list.append(world2cam[:3, :4])

    # Stack camera parameters
    intrinsics = np.stack(intrinsics_list)  # (N, 3, 3)
    extrinsics = np.stack(extrinsics_list)  # (N, 3, 4)

    # Get image size from first output
    h, w = outputs[0]["pts3d"][0].shape[:2]

    # Concatenate all points and colors
    all_points_concat = np.concatenate(all_points, axis=0)
    all_colors_concat = np.concatenate(all_colors, axis=0)

    if verbose:
        CONSOLE.print(f"  Total points before filtering: {len(all_points_concat)}")

    # Spatial filtering (optional)
    if spatial_filter_percentile is not None:
        all_points_concat, all_colors_concat = filter_points_by_spatial_extent(
            all_points_concat,
            all_colors_concat,
            percentile_range=spatial_filter_percentile,
            max_extent=spatial_filter_max_extent,
            verbose=verbose,
        )

    # Voxel downsample
    if verbose:
        CONSOLE.print(f"[bold cyan]Voxel downsampling point cloud...")
    downsampled_points, downsampled_colors = voxel_downsample_point_cloud(
        all_points_concat, all_colors_concat, voxel_fraction, voxel_size, verbose=verbose
    )

    # Build COLMAP reconstruction
    if verbose:
        CONSOLE.print(f"[bold cyan]Building COLMAP reconstruction...")
    reconstruction = build_colmap_reconstruction(
        points_3d=downsampled_points,
        points_rgb=downsampled_colors,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_width=w,
        image_height=h,
        image_names=image_names,
        camera_type="PINHOLE",
        skip_point2d=skip_point2d,
        verbose=verbose,
    )

    # Save reconstruction
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ Saved COLMAP reconstruction to: {sparse_dir}")

    # Optionally save PLY file
    if save_ply:
        try:
            ply_path = sparse_dir / "points.ply"
            trimesh.PointCloud(downsampled_points, colors=downsampled_colors).export(str(ply_path))
            if verbose:
                CONSOLE.print(f"  Saved point cloud PLY to: {ply_path}")
        except Exception as e:
            if verbose:
                CONSOLE.print(f"[yellow]Warning: Failed to save PLY: {e}")

    # Optionally save processed images
    if save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_frames):
            img_no_norm = outputs[i]["img_no_norm"][0].cpu().numpy()  # (H, W, 3) in [0, 1]
            img_uint8 = (img_no_norm * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)

            # Save with original image name
            img_path = images_dir / image_names[i]
            img_pil.save(str(img_path), quality=95)

        if verbose:
            CONSOLE.print(f"  Saved {num_frames} processed images to: {images_dir}")

    return reconstruction


# ============================================================================
# COLMAP Export - High-Level Interface
# ============================================================================

def export_to_colmap(
    outputs: List[Dict],
    views: List[Dict],
    image_names: List[str],
    output_dir: Union[str, Path],
    model: Any,
    voxel_fraction: float = 0.02,
    voxel_size: Optional[float] = None,
    save_ply: bool = True,
    save_images: bool = True,
    skip_point2d: bool = False,
    verbose: bool = True,
    **kwargs
) -> Path:
    """Export MapAnything outputs to COLMAP format.

    Args:
        outputs: Outputs from run_mapanything_inference()
        views: Preprocessed views from load_and_preprocess_images()
        image_names: List of image filenames (e.g., ["image001.jpg", ...])
        output_dir: Output directory for COLMAP files
        model: MapAnything model (needed for data_norm_type)
        voxel_fraction: Voxel size as fraction of scene extent for downsampling
        voxel_size: Explicit voxel size in meters (overrides voxel_fraction if set)
        save_ply: Whether to save point cloud as PLY file
        save_images: Whether to save processed images
        skip_point2d: Whether to skip writing point2D observations
        verbose: Whether to print progress information
        **kwargs: Additional arguments passed to export_predictions_to_colmap()

    Returns:
        Path to the COLMAP sparse reconstruction directory (sparse/0/)

    Raises:
        ImportError: If MapAnything export utilities are not installed
    """
    try:
        from mapanything.utils.colmap_export import export_predictions_to_colmap
    except ImportError as e:
        raise ImportError(
            "MapAnything not installed. Please install with:\n"
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        CONSOLE.print(f"[bold cyan]Exporting to COLMAP format")
        CONSOLE.print(f"  Output directory: {output_dir}")
        CONSOLE.print(f"  Voxel downsampling: {voxel_fraction * 100:.1f}% of scene extent")

    # Export to COLMAP
    _ = export_predictions_to_colmap_internal(
        outputs=outputs,
        processed_views=views,
        image_names=image_names,
        output_dir=output_dir,
        voxel_fraction=voxel_fraction,
        voxel_size=voxel_size,
        # data_norm_type=model.encoder.data_norm_type,
        save_ply=save_ply,
        save_images=save_images,
        skip_point2d=skip_point2d,
        **kwargs
    )

    # MapAnything exports to sparse/, we need sparse/0/ for nerfstudio
    sparse_dir = output_dir / "sparse" / "0"
    if verbose:
        CONSOLE.print(f"[bold green]✓ Exported to COLMAP format")
        CONSOLE.print(f"  Output: {sparse_dir}")
        files = ["cameras.bin", "images.bin", "points3D.bin"]
        if save_ply:
            files.append("points.ply")
        for f in files:
            if (sparse_dir / f).exists():
                CONSOLE.print(f"    - {f}")

    return sparse_dir


# ============================================================================
# Rescaling to Original Dimensions
# ============================================================================

def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: List[Path],
    original_image_sizes: np.ndarray,
    image_size: Tuple[int, int],
    shared_camera: bool = False,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> Any:
    """Rescale reconstruction from model resolution to original dimensions.

    This function is adapted from nerfstudio's vggt_utils module.
    It rescales camera intrinsics and image dimensions from the model's
    fixed resolution (e.g., 336x518 for MapAnything) to the original image sizes.

    Args:
        reconstruction: pycolmap Reconstruction object
        image_paths: List of Path objects for the images
        original_image_sizes: Array of shape (N, 6) with format:
            [top_left_x, top_left_y, crop_right, crop_bottom, original_width, original_height]
            For MapAnything (which resizes without cropping), use:
            [0, 0, model_width, model_height, original_width, original_height]
        image_size: Model image size as (width, height)
        shared_camera: Whether using a single shared camera for all images
        shift_point2d_to_original_res: Whether to shift point2D observations to original resolution
        verbose: Whether to print progress information

    Returns:
        Updated pycolmap Reconstruction object with rescaled cameras
    """
    if verbose:
        sample_image = original_image_sizes[0, -2:]
        original_width, original_height = sample_image
        CONSOLE.print(
            f"[bold yellow]Rescaling reconstruction from {image_size[0]}x{image_size[1]} "
            f"to original dimensions"
        )
        CONSOLE.print(f"  Original image sizes (WxH): {int(original_width)}x{int(original_height)}")

    rescale_camera = True

    # Shared-camera state (computed once)
    shared_intrinsics = None
    shared_width = None
    shared_height = None

    for pyimageid in reconstruction.images:
        # Get image and camera objects
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]

        # Rename image to original name
        pyimage.name = image_paths[pyimageid - 1].name

        # Copy camera parameters
        pred_params = copy.deepcopy(pycamera.params)

        # Get original width/height and compute scale factors
        real_image_size = original_image_sizes[pyimageid - 1, -2:]
        scale_x = real_image_size[0] / image_size[0]
        scale_y = real_image_size[1] / image_size[1]

        # --------------------------------
        # Rescale camera intrinsics
        # --------------------------------
        # Non-shared: rescale every time
        # Shared: rescale exactly once
        if rescale_camera and (not shared_camera or shared_intrinsics is None):

            # Rescale focal length parameters
            if pycamera.model.name == "SIMPLE_PINHOLE":
                pred_params[0] *= max(scale_x, scale_y)
            elif pycamera.model.name in ("PINHOLE", "OPENCV", "RADIAL", "OPENCV_FISHEYE"):
                pred_params[0] *= scale_x  # fx
                pred_params[1] *= scale_y  # fy

            # Rescale principal point (cx, cy)
            pred_params[-2] *= scale_x
            pred_params[-1] *= scale_y

            # Apply back to camera object
            if shared_camera:
                # First image defines the shared camera
                shared_intrinsics = pred_params
                shared_width = int(real_image_size[0])
                shared_height = int(real_image_size[1])

                pycamera.params = shared_intrinsics
                pycamera.width = shared_width
                pycamera.height = shared_height
            else:
                pycamera.params = pred_params
                pycamera.width = int(real_image_size[0])
                pycamera.height = int(real_image_size[1])

        # --------------------------------
        # Ensure shared camera is consistent
        # --------------------------------
        if shared_camera and shared_intrinsics is not None:
            pycamera.params = shared_intrinsics
            pycamera.width = shared_width
            pycamera.height = shared_height

        # --------------------------------
        # Shift point2D if requested
        # --------------------------------
        if shift_point2d_to_original_res:
            top_left = original_image_sizes[pyimageid - 1, :2]

            scale_x = real_image_size[0] / image_size[0]
            scale_y = real_image_size[1] / image_size[1]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * np.array([scale_x, scale_y])

    if verbose:
        CONSOLE.print(f"[bold green]✓ Rescaled reconstruction to original dimensions")

    return reconstruction


def rescale_to_original_dimensions(
    colmap_sparse_dir: Union[str, Path],
    image_paths: List[Path],
    model_width: int,
    model_height: int,
    output_dir: Union[str, Path],
    shared_camera: bool = True,
    shift_point2d_to_original_res: bool = True,
    verbose: bool = True,
    **kwargs
) -> Path:
    """Rescale COLMAP reconstruction to original image dimensions.

    MapAnything processes images at a fixed resolution (typically 336x518).
    This function rescales the camera parameters back to the original image sizes.

    Args:
        colmap_sparse_dir: Path to COLMAP sparse reconstruction (e.g., colmap/sparse/0/)
        image_paths: List of Path objects for the original images
        model_width: Width of images used by MapAnything model
        model_height: Height of images used by MapAnything model
        output_dir: Output directory for rescaled reconstruction
        shared_camera: Whether to use a single shared camera for all images
        shift_point2d_to_original_res: Whether to shift point2D observations
        verbose: Whether to print progress information
        **kwargs: Additional arguments passed to _rescale_reconstruction_to_original_dimensions()

    Returns:
        Path to output directory containing rescaled reconstruction

    Raises:
        ValueError: If COLMAP reconstruction cannot be loaded
    """
    colmap_sparse_dir = Path(colmap_sparse_dir)
    output_dir = Path(output_dir)

    # Create output sparse directory
    output_sparse_dir = output_dir / "colmap" / "sparse" / "0"
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        CONSOLE.print(f"[bold cyan]Rescaling reconstruction to original dimensions")
        CONSOLE.print(f"  Input: {colmap_sparse_dir}")
        CONSOLE.print(f"  Output: {output_sparse_dir}")

    # Load reconstruction
    try:
        reconstruction = pycolmap.Reconstruction(str(colmap_sparse_dir))
    except Exception as e:
        raise ValueError(f"Failed to load COLMAP reconstruction from {colmap_sparse_dir}: {e}")

    if verbose:
        CONSOLE.print(f"  Loaded reconstruction:")
        CONSOLE.print(f"    Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"    Images: {len(reconstruction.images)}")
        CONSOLE.print(f"    Points3D: {len(reconstruction.points3D)}")

    # Build original_image_sizes array
    original_coords = []
    for img_path in image_paths:
        img = Image.open(img_path)
        # Format: [top_left_x, top_left_y, crop_right, crop_bottom, original_width, original_height]
        # For MapAnything, images are resized without cropping
        original_coords.append([0, 0, model_width, model_height, img.width, img.height])

    original_coords = np.array(original_coords)

    if verbose:
        CONSOLE.print(f"  Model resolution: {model_width}x{model_height}")
        sample_orig = original_coords[0, -2:]
        CONSOLE.print(f"  Original resolution (sample): {int(sample_orig[0])}x{int(sample_orig[1])}")

    # Rescale reconstruction
    reconstruction = _rescale_reconstruction_to_original_dimensions(
        reconstruction=reconstruction,
        image_paths=image_paths,
        original_image_sizes=original_coords,
        image_size=(model_width, model_height),
        shared_camera=shared_camera,
        shift_point2d_to_original_res=shift_point2d_to_original_res,
        verbose=verbose,
        **kwargs
    )

    # Write rescaled reconstruction
    reconstruction.write_binary(str(output_sparse_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ Wrote rescaled reconstruction to: {output_sparse_dir}")

    return output_dir


# ============================================================================
# Nerfstudio Format Conversion
# ============================================================================

def convert_to_nerfstudio_format(
    colmap_sparse_dir: Union[str, Path],
    output_dir: Union[str, Path],
    ply_filename: str = "sparse_pc.ply",
    copy_ply_from_colmap: bool = True,
    verbose: bool = True,
    **kwargs
) -> Path:
    """Convert COLMAP reconstruction to nerfstudio transforms.json format.

    Args:
        colmap_sparse_dir: Path to COLMAP sparse reconstruction (e.g., colmap/sparse/0/)
        output_dir: Output directory for transforms.json
        ply_filename: Filename for the point cloud PLY file
        copy_ply_from_colmap: Whether to copy existing points.ply from COLMAP dir
        verbose: Whether to print progress information
        **kwargs: Additional arguments passed to colmap_to_json()

    Returns:
        Path to the generated transforms.json file
    """
    colmap_sparse_dir = Path(colmap_sparse_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        CONSOLE.print(f"[bold cyan]Converting to nerfstudio format")
        CONSOLE.print(f"  COLMAP dir: {colmap_sparse_dir}")
        CONSOLE.print(f"  Output dir: {output_dir}")

    # Convert COLMAP to transforms.json
    colmap_utils.colmap_to_json(
        recon_dir=colmap_sparse_dir,
        output_dir=output_dir,
        **kwargs
    )

    transforms_path = output_dir / "transforms.json"

    # Load transforms to get applied_transform
    with open(transforms_path) as f:
        transforms = json.load(f)

    applied_transform = torch.tensor(transforms["applied_transform"])

    # Handle point cloud
    ply_path = output_dir / ply_filename
    colmap_ply = colmap_sparse_dir / "points.ply"

    if copy_ply_from_colmap and colmap_ply.exists():
        if verbose:
            CONSOLE.print(f"  Copying point cloud from COLMAP")
        shutil.copy(colmap_ply, ply_path)
    else:
        if verbose:
            CONSOLE.print(f"  Creating point cloud from COLMAP reconstruction")
        colmap_utils.create_ply_from_colmap(
            filename=ply_filename,
            recon_dir=colmap_sparse_dir,
            output_dir=output_dir,
            applied_transform=applied_transform,
        )

    # Update transforms.json with PLY path
    transforms["ply_file_path"] = ply_filename
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    if verbose:
        CONSOLE.print(f"[bold green]✓ Nerfstudio format conversion complete")
        CONSOLE.print(f"  transforms.json: {transforms_path}")
        CONSOLE.print(f"  Point cloud: {ply_path}")

    return transforms_path


# ============================================================================
# Cleanup Utilities
# ============================================================================

def cleanup_gpu_memory(
    model: Optional[Any] = None,
    outputs: Optional[List[Dict]] = None,
    views: Optional[List[Dict]] = None,
    verbose: bool = True
) -> None:
    """Clean up GPU memory by deleting objects and clearing cache.

    Args:
        model: MapAnything model to delete
        outputs: Inference outputs to delete
        views: Preprocessed views to delete
        verbose: Whether to print memory status
    """
    import gc

    if verbose:
        CONSOLE.print(f"[bold cyan]Cleaning up GPU memory")

    # Delete objects
    if model is not None:
        del model
    if outputs is not None:
        del outputs
    if views is not None:
        del views

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if verbose:
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            CONSOLE.print(f"[bold green]✓ GPU memory cleaned")
            CONSOLE.print(f"  Allocated: {allocated_gb:.2f} GB")
            CONSOLE.print(f"  Reserved: {reserved_gb:.2f} GB")


# ============================================================================
# End-to-End Pipeline
# ============================================================================

def run_mapanything_pipeline(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "facebook/map-anything",
    voxel_fraction: float = 0.02,
    shared_camera: bool = True,
    cleanup_after: bool = False,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Path]:
    """Run the complete MapAnything pipeline from images to nerfstudio format.

    This is a convenience function that runs all steps in sequence:
    1. Load model
    2. Load and preprocess images
    3. Run inference
    4. Export to COLMAP
    5. Rescale to original dimensions
    6. Convert to nerfstudio format

    Args:
        image_dir: Directory containing input images
        output_dir: Output directory for all results
        model_name: HuggingFace model name for MapAnything
        voxel_fraction: Voxel size fraction for point cloud downsampling
        shared_camera: Whether to use shared camera for all images
        cleanup_after: Whether to cleanup GPU memory after inference
        verbose: Whether to print progress information
        **kwargs: Additional arguments for individual pipeline steps

    Returns:
        Dictionary with paths to key outputs:
            - 'colmap_dir': COLMAP reconstruction directory
            - 'transforms_json': Nerfstudio transforms.json path
            - 'point_cloud': Point cloud PLY file path
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    # Setup directories
    preproc_dir = output_dir / "preproc"
    colmap_dir = preproc_dir / "colmap"

    if verbose:
        CONSOLE.print("[bold magenta]" + "="*70)
        CONSOLE.print("[bold magenta]MAPANYTHING PIPELINE")
        CONSOLE.print("[bold magenta]" + "="*70)
        CONSOLE.print(f"Input: {image_dir}")
        CONSOLE.print(f"Output: {output_dir}")

    # Step 1: Load model
    model = load_mapanything_model(model_name=model_name, verbose=verbose)

    # Step 2: Load images
    views, image_paths = load_and_preprocess_images(image_dir, verbose=verbose)
    image_names = [p.name for p in image_paths]

    # Get model dimensions
    model_width = views[0]['img'].shape[-1]
    model_height = views[0]['img'].shape[-2]

    # Step 3: Run inference
    outputs = run_mapanything_inference(model, views, verbose=verbose, **kwargs)

    # Step 4: Export to COLMAP
    colmap_sparse_dir = export_to_colmap(
        outputs, views, image_names, colmap_dir, model,
        voxel_fraction=voxel_fraction,
        verbose=verbose
    )

    # Step 5: Rescale to original dimensions
    rescaled_dir = rescale_to_original_dimensions(
        colmap_sparse_dir, image_paths, model_width, model_height,
        output_dir, shared_camera=shared_camera, verbose=verbose
    )

    rescaled_sparse_dir = rescaled_dir / "colmap" / "sparse" / "0"

    # Step 6: Convert to nerfstudio format
    transforms_path = convert_to_nerfstudio_format(
        rescaled_sparse_dir, preproc_dir, verbose=verbose
    )

    # Cleanup if requested
    if cleanup_after:
        cleanup_gpu_memory(model, outputs, views, verbose=verbose)

    if verbose:
        CONSOLE.print("[bold magenta]" + "="*70)
        CONSOLE.print("[bold green]✓ PIPELINE COMPLETE")
        CONSOLE.print("[bold magenta]" + "="*70)
        CONSOLE.print(f"COLMAP reconstruction: {rescaled_sparse_dir}")
        CONSOLE.print(f"Nerfstudio transforms: {transforms_path}")
        CONSOLE.print(f"Point cloud: {preproc_dir / 'sparse_pc.ply'}")

    return {
        'colmap_dir': rescaled_sparse_dir,
        'transforms_json': transforms_path,
        'point_cloud': preproc_dir / 'sparse_pc.ply',
        'preproc_dir': preproc_dir,
    }
