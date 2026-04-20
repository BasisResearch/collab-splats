"""
Code that uses VGGT-X (Visual Geometry Grounded deep Transformer - eXtended)
to estimate camera poses and depth maps for structure from motion.

VGGT-X is a memory-optimized version of VGGT with 75-85% less GPU memory usage
and 30-40% faster inference through chunked processing and mixed precision.

This module provides the following main functions:

1. run_vggt: Direct VGGT inference to COLMAP (depth-based reconstruction)
   - Runs VGGT model on images
   - Converts depth maps directly to 3D points
   - Outputs COLMAP format without bundle adjustment

2. run_vggt_ba: VGGT with feature-based tracking and bundle adjustment
   - Runs VGGT model on images for initial poses and depth
   - Uses VGGSfM feature tracking via predict_tracks()
   - Builds reconstruction with batch_np_matrix_to_pycolmap()
   - Refines with bundle adjustment
   - Outputs refined COLMAP format

3. refine_vggt_with_ba: (Legacy) Refines existing COLMAP reconstruction
   - Loads existing reconstruction and refines in-place
   - Use run_vggt_ba for the official VGGT approach

VGGT-X Features:
- Frame-wise chunked processing (adjustable chunk_size)
- Automatic mixed precision (FP16/BF16)
- CPU offloading of intermediate features
- Compiled operations for faster inference
- Same output quality as original VGGT

Requires:
- VGGT-X module from: https://github.com/Linketic/VGGT-X
- PyTorch, PIL, pycolmap
"""

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pycolmap
from tqdm import tqdm
import roma
import kornia
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt

try:
    from streamvggt.models.streamvggt import StreamVGGT
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    from streamvggt.utils.geometry import FrameDiskCache   
    _HAS_INFINITE_VGGT = True
except ImportError:
    _HAS_INFINITE_VGGT = False
    # StreamVGGT = None  # type: ignore
    # load_and_preprocess_images = None  # type: ignore
    # pose_encoding_to_extri_intri = None  # type: ignore
    # FrameDiskCache = None  # type: ignore

# Try to import vggt - it's an optional dependency
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_ratio

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
    from vggt.dependency.track_predict import predict_tracks
    _HAS_VGGT = True
except ImportError:
    _HAS_VGGT = False
    # predict_tracks = None  # type: ignore
    # create_pixel_coordinate_grid = None  # type: ignore
    # randomly_limit_trues = None  # type: ignore
    # batch_np_matrix_to_pycolmap_wo_track = None  # type: ignore

from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils.rich_utils import CONSOLE

def is_vggt_available() -> bool:
    """Check if VGGT is installed and available.

    Returns:
        True if VGGT is available, False otherwise.
    """
    return _HAS_VGGT

def run_vggt(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: str = "SIMPLE_PINHOLE",
    verbose: bool = False,
    conf_threshold: float = 50.0,
    model_name: str = "facebook/VGGT-1B",
    chunk_size: int = 256,
    scale_factor: float = 1.0,
    shared_camera: bool = True,
    max_points_for_colmap: int = 500000,
    use_global_alignment: bool = True,
    filter_outlier_cameras: bool = False,
    lambda_depth: float = 0.0,
    overwrite: bool = False,
) -> None:
    """Runs VGGT on images to estimate camera poses and depth (Facebook's feedforward mode).

    This follows Facebook's demo_colmap.py without bundle adjustment:
    VGGT inference → depth unprojection → filter points → batch_np_matrix_to_pycolmap_wo_track()

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use (not used by VGGT, kept for compatibility).
        verbose: If True, logs the output.
        conf_threshold: Confidence threshold (0-100%) for including points.
        mask_sky: If True, filter out points likely to be sky.
        mask_black_bg: If True, filter out points with very dark/black color.
        mask_white_bg: If True, filter out points with very bright/white color.
        stride: Stride for point sampling (higher = fewer points).
        model_name: HuggingFace model name for VGGT.
        chunk_size: Chunk size for VGGT-X inference.
        shared_camera: If True, use single camera model for all frames.
        use_global_alignment: If True, perform global alignment to refine poses.
        filter_outlier_cameras: If True, filter outlier cameras after reconstruction (requires use_global_alignment=True).
    """
    if not is_vggt_available():
        CONSOLE.print(
            "[bold red]Error: To use vggt sfm_tool, you must install VGGT-X!\n"
            "Please install VGGT-X from: https://github.com/Linketic/VGGT-X.git"
            "Example: pip install git+https://github.com/Linketic/VGGT-X.git"
        )
        sys.exit(1)

    # Create output directory
    output_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    vggt_ckpt_path = colmap_dir / "vggt_checkpoint.pt"

    # Run VGGT inference
    if vggt_ckpt_path.exists() and not overwrite:
        vggt_data = torch.load(vggt_ckpt_path)
    else:
        vggt_data = _run_vggt_inference(
            image_dir=image_dir,
            model_name=model_name,
            chunk_size=chunk_size,
            verbose=verbose,
        )

        torch.save(vggt_data, vggt_ckpt_path)

        if verbose:
            CONSOLE.print(f"[bold green]✓ Saved VGGT checkpoint to {vggt_ckpt_path}")
            CONSOLE.print(f"  Use this checkpoint for rapid testing of match extraction!")
            CONSOLE.print(f"  Example: python test_ransac_filtering.py --checkpoint {vggt_ckpt_path}")


    # Extract data from inference results
    images = vggt_data["images"]
    extrinsic = vggt_data["extrinsic"]
    intrinsic = vggt_data["instrinsics"] # Upsampled to native resolution
    intrinsic_downsampled = vggt_data["instrinsics_downsampled"] # Downsampled to VGGT resolution
    original_coords = vggt_data["original_coords"] # [0, 0, new_width, new_height, width, height]
    depth_map = vggt_data["depth"]
    depth_conf = vggt_data["depth_conf"]
    image_paths = vggt_data["image_paths"]

    # Use global alignment if enabled
    if use_global_alignment:
        # Optimized camera poses using feature matching
        extrinsic, intrinsic_downsampled, match_outputs = _run_global_alignment(
            images=images,
            image_paths=image_paths,
            extrinsic=extrinsic,
            intrinsic=intrinsic_downsampled,
            depth_map=depth_map,
            depth_conf=depth_conf,
            lambda_depth=lambda_depth,
            colmap_dir=colmap_dir,
            shared_camera=shared_camera,
            verbose=verbose,
        )
    else:
        match_outputs = None

    # Filter and prepare 3D points in Facebook's format (points3d, points_xyf, points_rgb)
    # Note: Facebook uses conf_threshold as a value (e.g., 5.0), not percentile
    # For compatibility, we convert percentile to value if > 1
    if conf_threshold > 1.0:
        # Interpret as percentile and convert to value
        conf_threshold_value = np.percentile(depth_conf, conf_threshold)
    else:
        # Interpret as value directly
        conf_threshold_value = conf_threshold

    if verbose:
        CONSOLE.print(f"  - Using confidence threshold: {conf_threshold_value:.4f}")

    # Scale points for better reconstruction --> taken from VGGT-X demo_colmap.py
    extrinsic[:, :3, 3] *= scale_factor
    depth_map *= scale_factor

    # Unproject depth map to point map
    points3d = unproject_depth_map_to_point_map(
        depth_map, 
        extrinsic, 
        intrinsic_downsampled
    )

    # Filter points for pycolmap reconstruction using VGGTX logic
    points3d, points_xyf, points_rgb = _filter_and_prepare_points_for_pycolmap(
        points3d=points3d,
        depth_map=depth_map,
        depth_conf=depth_conf,
        images=images,
        image_paths=image_paths,
        conf_thres_value=conf_threshold_value,
        use_global_alignment=use_global_alignment,
        max_points_for_colmap=max_points_for_colmap,
        match_outputs=match_outputs
    )

    # Grab image size from depth map (N, H, W) --> make as width and height
    image_size = np.array([depth_map.shape[2], depth_map.shape[1]])
    
    if verbose:
        CONSOLE.print(f"[bold yellow]Building pycolmap reconstruction at WxH ({image_size[0]}x{image_size[1]})...")

    # Step 1: Build reconstruction at model resolution (518x518) using intrinsic_downsampled
    # within batch_np_matrix_to_pycolmap_wo_track image_size is used as width = image_size[0] and height = image_size[1]

    reconstruction = _build_pycolmap_reconstruction_without_tracks(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsic=extrinsic,
        intrinsic=intrinsic_downsampled,
        image_paths=image_paths,
        image_size=image_size, # [W, H]
        shared_camera=shared_camera,
        camera_type=camera_model,
        verbose=verbose,
    )

    if reconstruction is None:
        CONSOLE.print("[bold red]Error: Failed to build pycolmap reconstruction!")
        sys.exit(1)

    # # Step 2: Rescale reconstruction to original dimensions
    reconstruction_resolution = (image_size[0], image_size[1]) # Reverse as it expects width and height

    reconstruction = _rescale_reconstruction_to_original_dimensions(
        reconstruction=reconstruction,
        image_paths=image_paths,
        original_image_sizes=original_coords,
        image_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
        verbose=verbose,
    )

    # Step 3: Filter outlier cameras (if enabled)
    if use_global_alignment and filter_outlier_cameras:
        if verbose:
            CONSOLE.print(f"[bold yellow]Filtering outlier cameras...")

        reconstruction, removed_ids = _filter_outlier_cameras(
            reconstruction=reconstruction,
            match_outputs=match_outputs,
            depth_conf=depth_conf,
            pose_distance_std_factor=2.0,
            min_matches=200,
            min_points_3d=250,
            verbose=verbose,
        )

        if len(removed_ids) > 0:
            CONSOLE.print(f"[bold yellow]Filtered {len(removed_ids)} outlier cameras from reconstruction")
            if verbose:
                CONSOLE.print(f"  - Removed image IDs: {removed_ids}")
    
    # Write reconstruction to binary format
    if verbose:
        CONSOLE.print(f"[bold yellow]Writing COLMAP files to {output_dir}")

    reconstruction.write_binary(str(output_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ COLMAP reconstruction complete!")
        CONSOLE.print(f"  - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"  - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"  - 3D points: {len(reconstruction.points3D)}")


def run_infinite_vggt(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: str = "SIMPLE_PINHOLE",
    verbose: bool = False,
    conf_threshold: float = 50.0,
    model_name: str = "lch01/StreamVGGT",
    total_budget: int = 1200000,
    cache_results: bool = True,
    scale_factor: float = 1.0,
    shared_camera: bool = True,
    max_points_for_colmap: int = 500000,
    use_global_alignment: bool = True,
    filter_outlier_cameras: bool = False,
    lambda_depth: float = 0.0,
    overwrite: bool = False,
) -> None:
    """Runs InfiniteVGGT on images to estimate camera poses and depth (streaming/memory-efficient mode).

    This follows the same pattern as run_vggt but uses InfiniteVGGT/StreamVGGT for more
    memory-efficient processing:
    InfiniteVGGT inference → depth unprojection → filter points → batch_np_matrix_to_pycolmap_wo_track()

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use (not used by InfiniteVGGT, kept for compatibility).
        verbose: If True, logs the output.
        conf_threshold: Confidence threshold (0-100%) for including points.
        model_name: HuggingFace model name for InfiniteVGGT (default: "lch01/StreamVGGT").
        total_budget: Total budget for memory efficiency in InfiniteVGGT.
        cache_results: If True, cache intermediate results for faster re-runs.
        scale_factor: Scale factor for 3D points and camera positions.
        shared_camera: If True, use single camera model for all frames.
        max_points_for_colmap: Maximum number of points to include in COLMAP reconstruction.
        use_global_alignment: If True, perform global alignment to refine poses.
        filter_outlier_cameras: If True, filter outlier cameras after reconstruction (requires use_global_alignment=True).
        lambda_depth: Weight for depth consistency term in global alignment.
    """
    if not is_infinite_vggt_available():
        CONSOLE.print(
            "[bold red]Error: To use InfiniteVGGT sfm_tool, you must install InfiniteVGGT!\n"
            "Please install InfiniteVGGT from: https://github.com/Linketic/InfiniteVGGT.git"
            "Example: pip install git+https://github.com/Linketic/InfiniteVGGT.git"
        )
        sys.exit(1)

    # Create output directory
    output_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    infinite_vggt_ckpt_path = colmap_dir / "infinite_vggt_checkpoint.pt"

    # Run InfiniteVGGT inference
    if infinite_vggt_ckpt_path.exists() and not overwrite:
        infinite_vggt_data = torch.load(infinite_vggt_ckpt_path)
    else:
        infinite_vggt_data = _run_infinite_vggt_inference(
            image_dir=image_dir,
            model_name=model_name,
            total_budget=total_budget,
            cache_results=cache_results,
            verbose=verbose,
        )

        torch.save(infinite_vggt_data, infinite_vggt_ckpt_path)

        if verbose:
            CONSOLE.print(f"[bold green]✓ Saved InfiniteVGGT checkpoint to {infinite_vggt_ckpt_path}")
            CONSOLE.print(f"  Use this checkpoint for rapid testing of match extraction!")
            CONSOLE.print(f"  Example: python test_ransac_filtering.py --checkpoint {infinite_vggt_ckpt_path}")


    # Extract data from inference results
    images = infinite_vggt_data["images"]
    extrinsic = infinite_vggt_data["extrinsic"]
    intrinsic = infinite_vggt_data["instrinsics"] # Upsampled to native resolution
    intrinsic_downsampled = infinite_vggt_data["instrinsics_downsampled"] # Downsampled to InfiniteVGGT resolution
    original_coords = infinite_vggt_data["original_coords"] # [0, 0, new_width, new_height, width, height]
    depth_map = infinite_vggt_data["depth"]
    depth_conf = infinite_vggt_data["depth_conf"]
    image_paths = infinite_vggt_data["image_paths"]

    # Use global alignment if enabled
    if use_global_alignment:
        # Optimized camera poses using feature matching
        extrinsic, intrinsic_downsampled, match_outputs = _run_global_alignment(
            images=images,
            image_paths=image_paths,
            extrinsic=extrinsic,
            intrinsic=intrinsic_downsampled,
            depth_map=depth_map,
            depth_conf=depth_conf,
            lambda_depth=lambda_depth,
            colmap_dir=colmap_dir,
            shared_camera=shared_camera,
            verbose=verbose,
        )
    else:
        match_outputs = None

    # Filter and prepare 3D points in Facebook's format (points3d, points_xyf, points_rgb)
    # Note: Facebook uses conf_threshold as a value (e.g., 5.0), not percentile
    # For compatibility, we convert percentile to value if > 1
    if conf_threshold > 1.0:
        # Interpret as percentile and convert to value
        conf_threshold_value = np.percentile(depth_conf, conf_threshold)
    else:
        # Interpret as value directly
        conf_threshold_value = conf_threshold

    if verbose:
        CONSOLE.print(f"  - Using confidence threshold: {conf_threshold_value:.4f}")

    # Scale points for better reconstruction --> taken from VGGT-X demo_colmap.py
    extrinsic[:, :3, 3] *= scale_factor
    depth_map *= scale_factor

    # Unproject depth map to point map
    points3d = unproject_depth_map_to_point_map(
        depth_map,
        extrinsic,
        intrinsic_downsampled
    )

    # Filter points for pycolmap reconstruction using VGGTX logic
    points3d, points_xyf, points_rgb = _filter_and_prepare_points_for_pycolmap(
        points3d=points3d,
        depth_map=depth_map,
        depth_conf=depth_conf,
        images=images,
        image_paths=image_paths,
        conf_thres_value=conf_threshold_value,
        use_global_alignment=use_global_alignment,
        max_points_for_colmap=max_points_for_colmap,
        match_outputs=match_outputs
    )

    # Grab image size from depth map (N, H, W) --> make as width and height
    image_size = np.array([depth_map.shape[2], depth_map.shape[1]])

    if verbose:
        CONSOLE.print(f"[bold yellow]Building pycolmap reconstruction at WxH ({image_size[0]}x{image_size[1]})...")

    # Step 1: Build reconstruction at model resolution using intrinsic_downsampled
    # within batch_np_matrix_to_pycolmap_wo_track image_size is used as width = image_size[0] and height = image_size[1]

    reconstruction = _build_pycolmap_reconstruction_without_tracks(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsic=extrinsic,
        intrinsic=intrinsic_downsampled,
        image_paths=image_paths,
        image_size=image_size, # [W, H]
        shared_camera=shared_camera,
        camera_type=camera_model,
        verbose=verbose,
    )

    if reconstruction is None:
        CONSOLE.print("[bold red]Error: Failed to build pycolmap reconstruction!")
        sys.exit(1)

    # Step 2: Rescale reconstruction to original dimensions
    reconstruction_resolution = (image_size[0], image_size[1]) # Reverse as it expects width and height

    reconstruction = _rescale_reconstruction_to_original_dimensions(
        reconstruction=reconstruction,
        image_paths=image_paths,
        original_image_sizes=original_coords,
        image_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
        verbose=verbose,
    )

    # Step 3: Filter outlier cameras (if enabled)
    if use_global_alignment and filter_outlier_cameras:
        if verbose:
            CONSOLE.print(f"[bold yellow]Filtering outlier cameras...")

        reconstruction, removed_ids = _filter_outlier_cameras(
            reconstruction=reconstruction,
            match_outputs=match_outputs,
            depth_conf=depth_conf,
            pose_distance_std_factor=2.0,
            min_matches=200,
            min_points_3d=250,
            verbose=verbose,
        )

        if len(removed_ids) > 0:
            CONSOLE.print(f"[bold yellow]Filtered {len(removed_ids)} outlier cameras from reconstruction")
            if verbose:
                CONSOLE.print(f"  - Removed image IDs: {removed_ids}")

    # Write reconstruction to binary format
    if verbose:
        CONSOLE.print(f"[bold yellow]Writing COLMAP files to {output_dir}")

    reconstruction.write_binary(str(output_dir))

    if verbose:
        CONSOLE.print(f"[bold green]✓ COLMAP reconstruction complete!")
        CONSOLE.print(f"  - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"  - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"  - 3D points: {len(reconstruction.points3D)}")


# def run_vggt_ba(
#     image_dir: Path,
#     colmap_dir: Path,
#     camera_model: CameraModel,
#     verbose: bool = False,
#     conf_threshold: float = 50.0,
#     mask_sky: bool = False,
#     mask_black_bg: bool = False,
#     mask_white_bg: bool = False,
#     stride: int = 1,
#     model_name: str = "facebook/VGGT-1B",
#     ba_refine_focal_length: bool = True,
#     ba_refine_principal_point: bool = False,
#     ba_refine_extra_params: bool = False,
#     shared_camera: bool = True,
#     max_query_pts: int = 2048,
#     query_frame_num: int = 5,
#     keypoint_extractor: str = "aliked+sp",
#     max_points_num: int = 2048,
#     fine_tracking: bool = True,
#     vis_thresh: float = 0.5,
#     max_reproj_error: Optional[float] = None,
#     track_resolution: int = 518,
#     use_global_alignment: bool = False,
# ) -> None:
#     """Runs VGGT-X with two mutually exclusive reconstruction approaches.

#     This function supports TWO DIFFERENT workflows (following VGGT-X patterns):

#     **Approach 1: Global Alignment (use_global_alignment=True)**
#     Following VGGT-X demo_colmap.py pattern:
#     1. VGGT-X inference for depth + poses (NO tracking head)
#     2. Global Alignment to refine poses via feature matching
#     3. Unproject depth to 3D points using GA-refined poses
#     4. Build COLMAP from depth (batch_np_matrix_to_pycolmap_wo_track)
#     5. NO bundle adjustment (GA already optimized poses)
#     → Faster, no tracking, GA-optimized poses

#     **Approach 2: Track-based Bundle Adjustment (use_global_alignment=False)**
#     Following original VGGT demo_colmap.py pattern:
#     1. VGGT-X inference for depth + poses
#     2. Feature-based tracking via predict_tracks()
#     3. Build COLMAP from tracks (batch_np_matrix_to_pycolmap)
#     4. Run COLMAP bundle adjustment to refine poses
#     → Slower, track-based, BA-optimized poses

#     VGGT-X automatically enables:
#     - Frame-wise chunking (default chunk_size=512) for memory efficiency
#     - Mixed precision (BF16 on Ampere+, FP16 on older GPUs)
#     - CPU offloading of intermediate features
#     - Compiled operations for 30-40% speedup

#     Note: Compatible with pycolmap>=0.4.0. For pycolmap>=0.6.0, all Path objects
#     are explicitly converted to strings due to stricter type checking.

#     Args:
#         image_dir: Path to the directory containing the images.
#         colmap_dir: Path to the output directory.
#         camera_model: Camera model to use (not used by VGGT, kept for compatibility).
#         verbose: If True, logs the output.
#         conf_threshold: Not used with track-based approach (kept for compatibility).
#         mask_sky: Not used with track-based approach (kept for compatibility).
#         mask_black_bg: Not used with track-based approach (kept for compatibility).
#         mask_white_bg: Not used with track-based approach (kept for compatibility).
#         stride: Not used with track-based approach (kept for compatibility).
#         model_name: HuggingFace model name for VGGT.
#         ba_refine_focal_length: If True, refine focal length during BA.
#         ba_refine_principal_point: If True, refine principal point during BA.
#         ba_refine_extra_params: If True, refine distortion params during BA.
#         shared_camera: If True, use single camera model for all frames (like Facebook demo).
#         max_query_pts: Maximum query points for track prediction. Lower values use less memory.
#             Default: 2048. With VGGT-X optimizations, this is less critical than before.
#         query_frame_num: Number of query frames for track prediction. Lower values use less memory.
#             Default: 5. With VGGT-X optimizations, this is less critical than before.
#         keypoint_extractor: Keypoint extraction method ("aliked+sp", etc.).
#         fine_tracking: Enable fine tracking in track prediction.
#             Default: True. VGGT-X's memory optimizations make this more feasible than original VGGT.
#             Note: With VGGT-X, memory usage is ~75-85% lower than original VGGT.
#         vis_thresh: Visibility threshold for filtering tracks (0-1).
#         max_reproj_error: Maximum reprojection error for track filtering (None = no limit).
#         track_resolution: Resolution for loading images for track prediction.
#         use_global_alignment: If True, use Global Alignment approach (depth-based, NO tracking).
#             If False, use Bundle Adjustment approach (track-based).
#             **These are mutually exclusive approaches:**
#             - GA (True): Faster, no tracking, depth-based reconstruction
#             - BA (False): Slower, track-based reconstruction
#             Default: False (use track-based BA).
#             Requires: roma, kornia (automatically installed with VGGT-X).
#     """
#     if not _HAS_VGGT:
#         CONSOLE.print(
#             "[bold red]Error: To use vggt sfm_tool, you must install VGGT!\n"
#             "Visit https://github.com/facebookresearch/vggt for installation instructions."
#         )
#         sys.exit(1)

#     # Create output directory
#     output_dir = colmap_dir / "sparse" / "0"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Run VGGT inference
#     vggt_data = _run_vggt_inference(
#         image_dir=image_dir,
#         model_name=model_name,
#         verbose=verbose,
#     )

#     # Extract data from inference results
#     extrinsic = vggt_data["extrinsic"]
#     intrinsic = vggt_data["intrinsic"]
#     world_points = vggt_data["world_points"]
#     colors_rgb = vggt_data["colors_rgb"]
#     depth = vggt_data["depth"]
#     depth_conf = vggt_data["depth_conf"]
#     image_paths = vggt_data["image_paths"]
#     original_width = vggt_data["original_width"]
#     original_height = vggt_data["original_height"]

#     # Aggressively free memory after VGGT inference before track prediction
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         if verbose:
#             CONSOLE.print(f"[bold yellow]Cleared GPU cache after VGGT inference")
#             # Show current memory usage
#             allocated = torch.cuda.memory_allocated() / 1024**3
#             reserved = torch.cuda.memory_reserved() / 1024**3
#             CONSOLE.print(f"  - GPU memory allocated: {allocated:.2f} GiB")
#             CONSOLE.print(f"  - GPU memory reserved: {reserved:.2f} GiB")

#             # Check for RTX 4090 and warn about float32 auto-conversion issue
#             gpu_name = torch.cuda.get_device_name(0)
#             if "4090" in gpu_name or "RTX 4090" in gpu_name:
#                 CONSOLE.print(f"[bold yellow]Note: Detected RTX 4090 GPU")
#                 CONSOLE.print(f"  - RTX 4090 may auto-convert models to float32, using 2x memory")
#                 CONSOLE.print(f"  - See: https://github.com/facebookresearch/vggt/pull/253")

#     # Global Alignment (optional): Refine poses using feature matching
#     match_outputs = None  # Will store GA match outputs if GA is used
#     if use_global_alignment:
#         if verbose:
#             CONSOLE.print(f"[bold yellow]Running Global Alignment for pose refinement...")

#         extrinsic, intrinsic, match_outputs = _run_global_alignment(
#             extrinsic=extrinsic,
#             intrinsic=intrinsic,
#             depth=depth,
#             depth_conf=depth_conf,
#             image_paths=image_paths,
#             max_query_pts=max_query_pts,
#             shared_camera=shared_camera,
#             colmap_dir=colmap_dir,
#             verbose=verbose,
#         )

#         if verbose:
#             CONSOLE.print(f"[bold green]✓ Global Alignment complete")
#             if match_outputs is not None:
#                 CONSOLE.print(f"  - Feature matches will be used for point confidence filtering")

#         # Free memory
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # ==============================================================================
#     # TWO MUTUALLY EXCLUSIVE PATHS:
#     # 1. Global Alignment (GA): Depth-based reconstruction, NO tracking
#     # 2. Bundle Adjustment (BA): Track-based reconstruction
#     # ==============================================================================

#     if use_global_alignment:
#         # ==========================================================================
#         # APPROACH 1: GLOBAL ALIGNMENT (VGGT-X demo_colmap.py pattern)
#         # ==========================================================================
#         # Depth-based reconstruction without tracking
#         # Uses GA-refined poses to unproject depth to 3D points
#         # ==========================================================================

#         if verbose:
#             CONSOLE.print(f"\n[bold cyan]Using Global Alignment approach (depth-based, NO tracking)")
#             CONSOLE.print(f"[bold yellow]Building COLMAP from depth...")

#         # Import GA utilities for conf_mask extraction
#         import sys
#         sys.path.insert(0, "/workspace/VGGT-X")
#         import utils.opt as opt_utils

#         # Extract image basenames for matching
#         image_basenames = [p.name for p in image_paths]

#         # Extract confidence mask from GA matches
#         if match_outputs is not None:
#             conf_mask = opt_utils.extract_conf_mask(match_outputs, depth_conf, image_basenames)

#             # Apply confidence threshold
#             conf_threshold_value = np.percentile(depth_conf, 0.5) if conf_threshold > 1.0 else conf_threshold
#             conf_mask = conf_mask & (depth_conf >= conf_threshold_value)

#             # Limit number of points for COLMAP (following VGGT-X pattern)
#             from vggt.utils.helper import randomly_limit_trues
#             max_points_for_colmap = 500000  # Same as VGGT-X default
#             conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

#             if verbose:
#                 num_valid_points = conf_mask.sum()
#                 total_points = conf_mask.size
#                 pct = 100.0 * num_valid_points / total_points
#                 CONSOLE.print(f"  - GA confidence filtering: {num_valid_points}/{total_points} points ({pct:.1f}%)")
#         else:
#             # Fallback if no GA matches
#             conf_threshold_value = np.percentile(depth_conf, 0.5)
#             conf_mask = depth_conf >= conf_threshold_value

#         # Use depth-based reconstruction (NO tracking)
#         use_tracking = False

#         # Unproject depth to 3D points using GA-refined poses
#         from vggt.utils.geometry import unproject_depth_map_to_point_map
#         from vggt.utils.helper import create_pixel_coordinate_grid
#         import torch.nn.functional as F

#         # Unproject depth at VGGT resolution (518x518)
#         vggt_resolution = depth.shape[1]  # Should be 518
#         points_3d = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

#         # Get RGB colors for points
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         images_for_rgb = _load_images_for_tracking(image_paths, vggt_resolution, device, verbose=False)
#         points_rgb = (images_for_rgb.cpu().numpy() * 255).astype(np.uint8)
#         points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (N, H, W, 3)
#         del images_for_rgb
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         # Create pixel coordinate grid for points_xyf
#         num_frames, height, width, _ = points_3d.shape
#         points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

#         # Apply confidence mask
#         points_3d = points_3d[conf_mask]
#         points_xyf = points_xyf[conf_mask]
#         points_rgb = points_rgb[conf_mask]

#         if verbose:
#             CONSOLE.print(f"  - Filtered to {len(points_3d)} points")
#             CONSOLE.print(f"[bold yellow]Converting to COLMAP format...")

#         # Build COLMAP reconstruction without tracks
#         reconstruction = batch_np_matrix_to_pycolmap_wo_track(
#             points_3d,
#             points_xyf,
#             points_rgb,
#             extrinsic,
#             intrinsic,
#             np.array([vggt_resolution, vggt_resolution]),
#             shared_camera=False,  # VGGT-X uses False for depth-based
#             camera_type="PINHOLE",
#         )

#         if reconstruction is None:
#             CONSOLE.print("[bold red]Error: Failed to build pycolmap reconstruction!")
#             sys.exit(1)

#         if verbose:
#             CONSOLE.print(f"  - Reconstruction built with {len(reconstruction.points3D)} points")

#         # Create original_coords array (identity mapping at original resolution)
#         original_coords = np.zeros((len(image_paths), 4), dtype=np.float32)
#         original_coords[:, 2] = original_width  # width
#         original_coords[:, 3] = original_height  # height

#         reconstruction = rename_colmap_recons_and_rescale_camera(
#             reconstruction,
#             [str(p) for p in image_paths],
#             original_coords,
#             img_size=(vggt_resolution, vggt_resolution),
#             shift_point2d_to_original_res=True,
#             shared_camera=False,
#         )

#         if verbose:
#             CONSOLE.print(f"[bold green]✓ Global Alignment reconstruction complete (NO bundle adjustment)")

#     else:
#         # ==========================================================================
#         # APPROACH 2: BUNDLE ADJUSTMENT (Original VGGT demo_colmap.py pattern)
#         # ==========================================================================
#         # Track-based reconstruction with bundle adjustment
#         # Uses predict_tracks() to get feature-based tracks
#         # ==========================================================================

#         if verbose:
#             CONSOLE.print(f"\n[bold cyan]Using Bundle Adjustment approach (track-based)")
#             CONSOLE.print(f"[bold yellow]Running feature-based track prediction...")

#         # Use VGGT's native 518x518 resolution for tracking (much more memory efficient)
#         # We'll scale the tracks to original resolution afterwards
#         vggt_resolution = 518
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#         if verbose:
#             CONSOLE.print(f"  - Using resolution {vggt_resolution}x{vggt_resolution} for track prediction")
#             CONSOLE.print(f"  - This uses ~4x less memory than 1024x1024")
#             CONSOLE.print(f"  - Tracks will be scaled to original resolution after prediction")

#         images = _load_images_for_tracking(image_paths, vggt_resolution, device, verbose)

#         # Prepare depth and 3D points for track prediction at VGGT resolution
#         depth_resized, points_3d, intrinsic_scaled = _prepare_depth_and_points_for_tracking(
#             depth_conf=depth_conf,
#             extrinsic=extrinsic,
#             intrinsic=intrinsic,
#             original_width=original_width,
#             original_height=original_height,
#             track_resolution=vggt_resolution,
#             verbose=verbose,
#         )

#         if verbose:
#             CONSOLE.print(f"  - max_query_pts: {max_query_pts}")
#             CONSOLE.print(f"  - query_frame_num: {query_frame_num}")
#             CONSOLE.print(f"  - fine_tracking: {fine_tracking}")
#             CONSOLE.print(f"  - max_points_num: {max_points_num}")

#             # Warn about memory usage if using high settings
#             if fine_tracking or max_query_pts > 1024 or query_frame_num > 3:
#                 CONSOLE.print(f"[bold yellow]Memory usage tips:")
#                 if fine_tracking:
#                     CONSOLE.print(f"  - fine_tracking=True uses significant memory. Set to False if OOM occurs.")
#                 if max_query_pts > 1024:
#                     CONSOLE.print(f"  - max_query_pts={max_query_pts} is high. Try 1024 or 512 if OOM occurs.")
#                 if query_frame_num > 3:
#                     CONSOLE.print(f"  - query_frame_num={query_frame_num} is high. Try 3 if OOM occurs.")
#                 CONSOLE.print(f"  - See: https://github.com/facebookresearch/vggt/issues/238")

#         # Predict tracks using VGGSfM
#         # Note: predict_tracks expects conf and points_3d as NUMPY ARRAYS, not torch tensors
#         # Only images should be a torch tensor (following Facebook demo pattern)
#         if verbose:
#             CONSOLE.print(f"  - images shape for predict_tracks: {images.shape}")
#             CONSOLE.print(f"  - depth_resized shape for predict_tracks: {depth_resized.shape}")
#             CONSOLE.print(f"  - points_3d shape for predict_tracks: {points_3d.shape}")

#         pred_tracks, pred_vis_scores, pred_confs, refined_points_3d, points_rgb = predict_tracks(
#             images=images,
#             conf=depth_resized,
#             points_3d=points_3d,
#             masks=None,
#             max_query_pts=max_query_pts,
#             query_frame_num=query_frame_num,
#             keypoint_extractor=keypoint_extractor,
#             max_points_num=max_points_num,
#             fine_tracking=fine_tracking,
#         )

#         # Free GPU memory from input tensors immediately
#         del images
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         use_tracking = True

#         # Convert to numpy and free GPU memory
#         if isinstance(pred_tracks, torch.Tensor):
#             pred_tracks = pred_tracks.cpu().numpy()
#         if isinstance(pred_vis_scores, torch.Tensor):
#             pred_vis_scores = pred_vis_scores.cpu().numpy()
#         if isinstance(refined_points_3d, torch.Tensor):
#             refined_points_3d = refined_points_3d.cpu().numpy()
#         if isinstance(points_rgb, torch.Tensor):
#             points_rgb = points_rgb.cpu().numpy()

#         # Clear GPU cache after converting all tensors to CPU
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         # Filter tracks by visibility threshold
#         track_mask = pred_vis_scores > vis_thresh

#         if verbose:
#             CONSOLE.print(f"  - Predicted tracks shape: {pred_tracks.shape}")
#             CONSOLE.print(f"  - Visibility scores shape: {pred_vis_scores.shape}")
#             CONSOLE.print(f"  - Valid tracks: {np.sum(track_mask)}/{track_mask.size}")
#             CONSOLE.print(f"[bold yellow]Building pycolmap reconstruction from tracks...")

#         # Build pycolmap reconstruction using batch_np_matrix_to_pycolmap
#         reconstruction = _build_pycolmap_reconstruction_from_tracks(
#             points3D=refined_points_3d,
#             extrinsic=extrinsic,
#             intrinsic=intrinsic_scaled,
#             tracks=pred_tracks,
#             image_paths=image_paths,
#             image_size=(vggt_resolution, vggt_resolution),
#             masks=track_mask,
#             shared_camera=shared_camera,
#             max_reproj_error=max_reproj_error,
#             points_rgb=points_rgb,
#             verbose=verbose,
#         )

#         if reconstruction is None:
#             CONSOLE.print("[bold red]Error: Failed to build pycolmap reconstruction!")
#             sys.exit(1)

#         if verbose:
#             CONSOLE.print(f"  - Reconstruction built with {len(reconstruction.points3D)} points")
#             CONSOLE.print(f"[bold yellow]Running bundle adjustment...")

#         # Run bundle adjustment
#         ba_options = pycolmap.BundleAdjustmentOptions()
#         ba_options.refine_focal_length = ba_refine_focal_length
#         ba_options.refine_principal_point = ba_refine_principal_point
#         ba_options.refine_extra_params = ba_refine_extra_params

#         pycolmap.bundle_adjustment(reconstruction, ba_options)

#         if verbose:
#             CONSOLE.print(f"[bold green]✓ Bundle adjustment complete")

#         # Rescale reconstruction from vggt_resolution to original dimensions
#         original_image_sizes = [(original_width, original_height)] * len(image_paths)
#         reconstruction = _rescale_reconstruction_to_original_dimensions(
#             reconstruction=reconstruction,
#             image_paths=image_paths,
#             original_image_sizes=original_image_sizes,
#             model_resolution=vggt_resolution,
#             shared_camera=shared_camera,
#             verbose=verbose,
#         )

#     # Write refined COLMAP files
#     if verbose:
#         CONSOLE.print(f"[bold yellow]Writing refined COLMAP files to {output_dir}")

#     reconstruction.write_binary(str(output_dir))

#     if verbose:
#         CONSOLE.print(f"[bold green]✓ COLMAP reconstruction with BA complete!")
#         CONSOLE.print(f"  - Cameras: {len(reconstruction.cameras)}")
#         CONSOLE.print(f"  - Images: {len(reconstruction.images)}")
#         CONSOLE.print(f"  - 3D points: {len(reconstruction.points3D)}")


# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================

def _run_vggt_inference(
    image_dir: Path,
    model_name: str,
    chunk_size: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Run VGGT inference and return all necessary data.

    This is the shared inference logic used by both run_vggt and run_vggt_ba.

    The image processing logic comes from VGGT-X's demo_colmap.py. This adjusts
    the original VGGT code to use custom image scaling + chunking.

    Returns:
        Dictionary containing:
        - extrinsic: Camera extrinsic matrices
        - intrinsic: Camera intrinsic matrices
        - world_points: 3D points from depth
        - colors_rgb: RGB colors for points
        - depth_conf: Depth confidence values
        - image_paths: Paths to input images
        - original_width/height: Original image dimensions
    """
    if not is_vggt_available():
        CONSOLE.print(
            "[bold red]Error: To use VGGT, you must install it!\n"
            "Please install VGGT-X from: https://github.com/Linketic/VGGT-X.git"
            "Example: pip install git+https://github.com/Linketic/VGGT-X.git"
        )
        sys.exit(1)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if verbose:
        CONSOLE.print(f"[bold green]Using device: {device}")
        CONSOLE.print(f"[bold yellow]Using dtype: {dtype}")

    # Load VGGT model
    if verbose:
        CONSOLE.print(f"[bold yellow]Loading VGGT-X model: {model_name}")
        CONSOLE.print(f"[bold yellow]  - Using chunk_size={chunk_size} for memory efficiency")

    # VGGT-X: Use smaller chunk_size for better memory efficiency
    model = VGGT.from_pretrained(model_name, chunk_size=chunk_size)
    model.eval()
    model = model.to(device, dtype=dtype) # Move model to device and convert to target dtype

    # --------------------------------------------------------------------------
    # Image Loading and Preprocessing Section
    # --------------------------------------------------------------------------

    # Find all image files in the directory (sorted by filename)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    if len(image_paths) == 0:
        CONSOLE.print(f"[bold red]Error: No images found in {image_dir}")
        sys.exit(1)

    if verbose:
        CONSOLE.print(f"[bold green]Found {len(image_paths)} images")

    # Prepare original images as RGB numpy arrays for later use
    original_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))

    # Prepare paths as strings for VGGT preprocessing
    image_names = [str(p) for p in image_paths]
    
    # Preprocess images for VGGT:
    #   - Keep aspect ratio;
    #   - Scale largest dimension to img_load_resolution;
    # Also outputs original (float pixel) coordinates for each processed image
    img_load_resolution = 518
    images, original_coords = load_and_preprocess_images_ratio(image_names, img_load_resolution)

    # Move tensors to device and to target dtype (match with model expectation)
    # Images are (B, 3, H, W)
    images = images.to(device, dtype=dtype)
    original_coords = original_coords.to(device)

    width, height = original_coords[0, -2:]

    # Save processed image shape (HxW) for later pose decoding and debug info
    image_shape = images.shape[-2:]

    if verbose:
        CONSOLE.print(f"[bold yellow]Running VGGT inference...")
        CONSOLE.print(f"  - Images shape: {images.shape}, dtype: {images.dtype}")
        CONSOLE.print(f"  - Model dtype: {next(model.parameters()).dtype}")

    # --------------------------------------------------------------------------

    # Run inference (no autocast needed since everything already in correct dtype)
    with torch.no_grad():
        predictions = model(images)

        # Get extrinsics and intrinsics (at resolution of processing images)
        # pose_encoding takes image shape (H, W)
        extrinsic, intrinsic_downsampled = pose_encoding_to_extri_intri(predictions["pose_enc"], image_shape)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], [width, height])

        # Squeeze and move to CPU
        extrinsic = extrinsic.cpu().float().numpy().squeeze(0)
        intrinsic = intrinsic.cpu().float().numpy().squeeze(0)
        intrinsic_downsampled = intrinsic_downsampled.cpu().float().numpy().squeeze(0)
        depth_map = predictions['depth'].squeeze(0).cpu().float().numpy()
        depth_conf = predictions['depth_conf'].squeeze(0).cpu().float().numpy()
    
    # Move model to CPU to free GPU memory
    if torch.cuda.is_available():
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
    
    if verbose:
        CONSOLE.print(f"  - extrinsic shape before squeeze: {extrinsic.shape}")
        CONSOLE.print(f"  - intrinsic shape before squeeze: {intrinsic_downsampled.shape}")

    # Clear GPU cache after all conversions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose:
        CONSOLE.print(f"[bold green]✓ VGGT inference complete")
        CONSOLE.print(f"  - Camera poses: {extrinsic.shape}")
    
    return {
        "images": images,
        "extrinsic": extrinsic,
        "instrinsics": intrinsic,
        "instrinsics_downsampled": intrinsic_downsampled,
        "depth": depth_map,
        "depth_conf": depth_conf,
        "image_paths": image_paths,
        "original_coords": original_coords.cpu().float().numpy(),
    }



def is_infinite_vggt_available() -> bool:
    """Check if InfiniteVGGT is installed and available.

    Returns:
        True if InfiniteVGGT is available, False otherwise.
    """
    return _HAS_INFINITE_VGGT


def _run_infinite_vggt_inference(
    image_dir: Path,
    model_name: str = "lch01/StreamVGGT",
    total_budget: int = 1200000,
    cache_results: bool = True,
    verbose: bool = False,
):

    if not is_infinite_vggt_available():
        CONSOLE.print(
            "[bold red]Error: To use InfiniteVGGT, you must install it!\n"
            "Please install InfiniteVGGT from: https://github.com/Linketic/InfiniteVGGT.git"
            "Example: pip install git+https://github.com/Linketic/InfiniteVGGT.git"
        )
        sys.exit(1)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if verbose:
        CONSOLE.print(f"[bold green]Using device: {device}")
        CONSOLE.print(f"[bold yellow]Using dtype: {dtype}")

    # Load VGGT model
    if verbose:
        CONSOLE.print(f"[bold yellow]Loading InfiniteVGGT model: {model_name}")
        CONSOLE.print(f"[bold yellow]  - Using total_budget={total_budget} for memory efficiency")

    # VGGT-X: Use smaller chunk_size for better memory efficiency
    model = StreamVGGT.from_pretrained(model_name, total_budget=total_budget)
    model.eval()
    model = model.to(device)  # Move model to device (keep original dtype for LayerNorm compatibility)

    # --------------------------------------------------------------------------
    # Image Loading and Preprocessing Section
    # --------------------------------------------------------------------------

    # Find all image files in the directory (sorted by filename)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    if len(image_paths) == 0:
        CONSOLE.print(f"[bold red]Error: No images found in {image_dir}")
        sys.exit(1)

    if verbose:
        CONSOLE.print(f"[bold green]Found {len(image_paths)} images")

    # Prepare original images as RGB numpy arrays for later use
    original_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))

    # Prepare paths as strings for VGGT preprocessing
    image_names = [str(p) for p in image_paths]

    # Load and preprocess images for InfiniteVGGT:
    #   - Keep aspect ratio;
    #   - Scale largest dimension to img_load_resolution;
    # Also outputs original (float pixel) coordinates for each processed image
    img_load_resolution = 518
    print(f"Loading {len(image_paths)} images...")
    images, original_coords = load_and_preprocess_images_ratio(image_names, img_load_resolution)
    print(f"✓ Images preprocessed: {images.shape}")
    print(f"  Shape: (N={images.shape[0]}, C={images.shape[1]}, H={images.shape[2]}, W={images.shape[3]})")

    # Move tensors to device (keep Float32, autocast will handle dtype conversion)
    # Images are (B, 3, H, W)
    images = images.to(device)
    original_coords = original_coords.to(device)

    # Extract width and height as Python integers (for pose_encoding_to_extri_intri)
    width = int(original_coords[0, -2].item())
    height = int(original_coords[0, -1].item())

    # Save processed image shape (HxW) for later pose decoding and debug info
    image_shape = images.shape[-2:]

    # Prepare frames for inference
    frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]

    if verbose:
        CONSOLE.print(f"[bold yellow]Running VGGT inference...")
        CONSOLE.print(f"  - Images shape: {images.shape}, dtype: {images.dtype}")
        CONSOLE.print(f"  - Model dtype: {next(model.parameters()).dtype}")
    
     # --------------------------------------------------------------------------
    # Run inference (no autocast needed since everything already in correct dtype)
    # Track timing and memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model.inference(
                frames,
                frame_writer=None,
                cache_results=cache_results
            )

        # Create matched dictionary of predictions to VGGTX
        predictions = {
            "points3d": [res['pts3d_in_other_view'].squeeze(0) for res in predictions.ress],
            "points_conf": [res['conf'].squeeze(0) for res in predictions.ress],
            "depth_map": [res['depth'].squeeze(0) for res in predictions.ress],
            "depth_conf": [res['depth_conf'].squeeze(0) for res in predictions.ress],
            "pose_enc": [res['camera_pose'].squeeze(0) for res in predictions.ress]
        }

        # Stack tensors and move to CPU (keep as tensors for pose_encoding_to_extri_intri)
        predictions = {k: torch.stack(v, dim=0).detach().cpu().float() for k, v in predictions.items()}

        # Get extrinsics and intrinsics (at resolution of processing images)
        # pose_encoding_to_extri_intri expects (B, N, 16) with batch dimension
        extrinsic, intrinsic_downsampled = pose_encoding_to_extri_intri(
            predictions["pose_enc"].unsqueeze(0),  # Add batch dimension: (N, 16) -> (1, N, 16)
            image_shape
        )
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"].unsqueeze(0),  # Add batch dimension: (N, 16) -> (1, N, 16)
            [width, height]
        )

        # Convert to numpy and squeeze batch dimension (matching VGGT format)
        extrinsic = extrinsic.squeeze(0).cpu().float().numpy()  # (1, N, 4, 4) -> (N, 4, 4)
        intrinsic = intrinsic.squeeze(0).cpu().float().numpy()  # (1, N, 3, 3) -> (N, 3, 3)
        intrinsic_downsampled = intrinsic_downsampled.squeeze(0).cpu().float().numpy()  # (1, N, 3, 3) -> (N, 3, 3)

        # Convert predictions to numpy after pose processing
        predictions = {k: v.numpy() for k, v in predictions.items()}

    torch.cuda.synchronize()
    end_time = time.time()

    inference_time = end_time - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    if verbose:
        CONSOLE.print("\n" + "="*70)
        CONSOLE.print("[bold green]INFERENCE COMPLETE")
        CONSOLE.print("[bold green]="*70)
        CONSOLE.print(f"  Total time: {inference_time:.2f}s")
        CONSOLE.print(f"  Time per frame: {inference_time / len(image_paths):.3f}s")
        CONSOLE.print(f"  Peak GPU memory: {peak_memory_gb:.2f} GB")
        CONSOLE.print(f"  Frames processed: {len(image_paths)}")
        CONSOLE.print("[bold green]="*70)

    # Return format matching _run_vggt_inference for compatibility
    return {
        "images": images,
        "extrinsic": extrinsic,
        "instrinsics": intrinsic,  # Note: typo matches _run_vggt_inference
        "instrinsics_downsampled": intrinsic_downsampled,  # Note: typo matches _run_vggt_inference
        "depth": predictions["depth_map"],  # Renamed from depth_map to match VGGT
        "depth_conf": predictions["depth_conf"],
        "image_paths": image_paths,
        "original_coords": original_coords.cpu().float().numpy(),
        # Extra InfiniteVGGT-specific data (not used by standard pipeline but useful)
        "points3d": predictions["points3d"],
        "points_conf": predictions["points_conf"],
    }

# def _build_pycolmap_reconstruction_from_tracks(
#     points3D: np.ndarray,
#     extrinsic: np.ndarray,
#     intrinsic: np.ndarray,
#     tracks: np.ndarray,
#     image_paths: List[Path],
#     image_size: Tuple[int, int],
#     masks: Optional[np.ndarray] = None,
#     shared_camera: bool = True,
#     max_reproj_error: Optional[float] = None,
#     points_rgb: Optional[np.ndarray] = None,
#     verbose: bool = False,
# ):
#     """Build a pycolmap Reconstruction from feature-based tracks using VGGT's utility.

#     This is a wrapper around batch_np_matrix_to_pycolmap from VGGT, which uses
#     feature-based tracking (VGGSfM) instead of depth-based tracks.

#     Args:
#         points3D: (P, 3) array of 3D point coordinates
#         extrinsic: Camera extrinsic matrices (N, 4, 4)
#         intrinsic: Camera intrinsic matrices (N, 3, 3)
#         tracks: (N, P, 2) array of 2D point projections across frames
#         image_paths: Paths to images
#         image_size: (width, height) tuple
#         masks: Optional (N, P) boolean array indicating valid tracks
#         shared_camera: If True, use single camera model for all frames
#         max_reproj_error: Optional threshold for reprojection error filtering
#         points_rgb: Optional (P, 3) RGB color values for points
#         verbose: If True, log progress

#     Returns:
#         pycolmap.Reconstruction object or None if failed
#     """
#     try:
#         from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
#     except ImportError:
#         CONSOLE.print(
#             "[bold red]Error: VGGT dependency not found!\n"
#             "batch_np_matrix_to_pycolmap requires: https://github.com/facebookresearch/vggt"
#         )
#         return None

#     # Note: extrinsic from pose_encoding_to_extri_intri is already (N, 3, 4) format
#     # batch_np_matrix_to_pycolmap expects (N, 3, 4) extrinsics
#     if extrinsic.shape[1] == 4 and extrinsic.shape[2] == 4:
#         # If somehow we have (N, 4, 4), convert to (N, 3, 4)
#         extrinsic_3x4 = extrinsic[:, :3, :]
#     elif extrinsic.shape[1] == 3 and extrinsic.shape[2] == 4:
#         # Already in correct (N, 3, 4) format
#         extrinsic_3x4 = extrinsic
#     else:
#         raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}. Expected (N, 3, 4) or (N, 4, 4)")

#     # Convert image_size from (width, height) to array
#     image_size_array = np.array([image_size[0], image_size[1]])

#     # Determine camera type based on shared_camera setting
#     camera_type = "SIMPLE_PINHOLE" if shared_camera else "PINHOLE"

#     if verbose:
#         CONSOLE.print(f"  - Calling batch_np_matrix_to_pycolmap with {len(points3D)} 3D points")
#         CONSOLE.print(f"  - Track shape: {tracks.shape}")
#         CONSOLE.print(f"  - Camera type: {camera_type}")

#     # Call VGGT's batch_np_matrix_to_pycolmap
#     reconstruction, valid_mask = batch_np_matrix_to_pycolmap(
#         points3d=points3D,
#         extrinsics=extrinsic_3x4,
#         intrinsics=intrinsic,
#         tracks=tracks,
#         image_size=image_size_array,
#         masks=masks,
#         max_reproj_error=max_reproj_error,
#         shared_camera=shared_camera,
#         camera_type=camera_type,
#         points_rgb=points_rgb,
#     )

#     if reconstruction is None:
#         if verbose:
#             CONSOLE.print("[bold yellow]Warning: batch_np_matrix_to_pycolmap returned None")
#         return None

#     # Add image names to reconstruction
#     for i, img_path in enumerate(image_paths):
#         if (i + 1) in reconstruction.images:
#             reconstruction.images[i + 1].name = img_path.name

#     if verbose:
#         CONSOLE.print(f"  - Created reconstruction:")
#         CONSOLE.print(f"    - Cameras: {len(reconstruction.cameras)}")
#         CONSOLE.print(f"    - Images: {len(reconstruction.images)}")
#         CONSOLE.print(f"    - Points3D: {len(reconstruction.points3D)}")
#         CONSOLE.print(f"    - Valid tracks: {np.sum(valid_mask)}/{len(valid_mask)}")

#     return reconstruction


# def _load_images_for_tracking(
#     image_paths: List[Path],
#     resolution: int,
#     device: str,
#     verbose: bool = False,
# ):
#     """Load and preprocess images at specified resolution for track prediction.

#     Args:
#         image_paths: List of paths to images
#         resolution: Target resolution (images will be resized to resolution x resolution)
#         device: Device to load tensors to ("cuda" or "cpu")
#         verbose: If True, log progress

#     Returns:
#         Tensor of shape (N, 3, H, W) with images normalized to [0, 1]
#     """
#     try:
#         import torch
#         from PIL import Image
#     except ImportError:
#         CONSOLE.print("[bold red]Error: torch and PIL are required!")
#         sys.exit(1)

#     if verbose:
#         CONSOLE.print(f"[bold yellow]Loading {len(image_paths)} images at resolution {resolution}...")

#     # Use pinned memory for faster CPU->GPU transfers if using CUDA
#     use_pinned = device == "cuda"

#     images_list = []
#     for img_path in image_paths:
#         img = Image.open(img_path).convert('RGB')
#         img = img.resize((resolution, resolution), Image.BILINEAR)
#         # Create contiguous tensor for better memory layout
#         img_array = np.array(img, dtype=np.float32)  # Use float32 directly
#         img_tensor = torch.from_numpy(img_array).permute(2, 0, 1) / 255.0

#         if use_pinned:
#             img_tensor = img_tensor.pin_memory()

#         images_list.append(img_tensor)

#     # Stack and move to device efficiently
#     images = torch.stack(images_list)
#     if device == "cuda":
#         images = images.to(device, non_blocking=True).contiguous()
#     else:
#         images = images.to(device).contiguous()

#     if verbose:
#         CONSOLE.print(f"  - Loaded images with shape: {images.shape}")

#     return images

# ============================================================================
# FACEBOOK VGGT HELPER FUNCTIONS
# These functions are sourced from Facebook's VGGT repository to match their
# exact post-model workflow for COLMAP reconstruction.
# ============================================================================

def _build_pycolmap_reconstruction_without_tracks(
    points3d: np.ndarray,
    points_xyf: np.ndarray,
    points_rgb: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_paths: List[Path],
    image_size: Tuple[int, int],
    shared_camera: bool,
    camera_type: str,
    verbose: bool,
) -> Optional[Any]:
    """Build pycolmap reconstruction without tracks using Facebook's function.

    This is a thin wrapper around Facebook's batch_np_matrix_to_pycolmap_wo_track function.

    Args:
        points3d: 3D points (P, 3)
        points_xyf: 2D points with frame indices (P, 3) - [x, y, frame_idx]
        points_rgb: RGB colors (P, 3)
        extrinsic: Camera extrinsics (N, 4, 4)
        intrinsic: Camera intrinsics (N, 3, 3)
        image_paths: Paths to images
        image_size: (width, height) of images
        shared_camera: Whether to use a single shared camera
        camera_type: Camera model type (e.g., "SIMPLE_PINHOLE")
        verbose: Whether to print progress

    Returns:
        pycolmap Reconstruction object, or None if failed
    """
    # Convert extrinsic from (N, 4, 4) to (N, 3, 4) if needed
    if extrinsic.shape[1] == 4:
        extrinsic_3x4 = extrinsic[:, :3, :]
    else:
        extrinsic_3x4 = extrinsic

    # image_size should be [width, height] for Facebook's function
    image_size_array = np.array([image_size[0], image_size[1]])

    if verbose:
        CONSOLE.print(f"  - Calling batch_np_matrix_to_pycolmap_wo_track with {len(points3d)} 3D points")
        CONSOLE.print(f"  - points_xyf shape: {points_xyf.shape}")
        CONSOLE.print(f"  - Camera type: {camera_type}")

    # Call Facebook's function
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsics=extrinsic_3x4,
        intrinsics=intrinsic,
        image_size=image_size_array, # Image size here is width height -->
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    if reconstruction is None:
        if verbose:
            CONSOLE.print("[bold yellow]Warning: batch_np_matrix_to_pycolmap_wo_track returned None")
        return None

    # Update image names to actual filenames
    for i, img_path in enumerate(image_paths):
        if (i + 1) in reconstruction.images:
            reconstruction.images[i + 1].name = img_path.name

    if verbose:
        CONSOLE.print(f"  - Created reconstruction:")
        CONSOLE.print(f"    - Cameras: {len(reconstruction.cameras)}")
        CONSOLE.print(f"    - Images: {len(reconstruction.images)}")
        CONSOLE.print(f"    - Points3D: {len(reconstruction.points3D)}")

    return reconstruction


def _rescale_reconstruction_to_original_dimensions(
    reconstruction: Any,
    image_paths: List[Path],
    original_image_sizes: List[Tuple[int, int]],
    image_size: Tuple[int, int],
    shared_camera: bool = False,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> Any:
    """Rescale reconstruction from model resolution to original dimensions.

    This is based on Facebook's rename_colmap_recons_and_rescale_camera function.
    VGGT builds reconstructions at 518x518 resolution, so we need to rescale
    camera parameters and image dimensions to match the original images.

    Args:
        reconstruction: pycolmap Reconstruction object
        image_paths: Paths to images
        original_image_sizes: List of (width, height) tuples for each image
        image_size: Image size (width, height)
        shared_camera: Whether using a single shared camera
        shift_point2d_to_original_res: Whether to shift point2d to original resolution
        verbose: Whether to print progress

    Returns:
        Updated pycolmap Reconstruction object
    """
    import copy

    if verbose:
        sample_image = original_image_sizes[0, -2:]
        original_width, original_height = sample_image
        CONSOLE.print(f"[bold yellow]Rescaling reconstruction from WxH ({image_size[0]}x{image_size[1]}) to original dimensions")
        CONSOLE.print(f"  - Original image sizes (WxH): {original_width}x{original_height}")
    
    rescale_camera = True

    # Shared-camera state (computed once)
    shared_intrinsics = None
    shared_width = None
    shared_height = None

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1].name

        pred_params = copy.deepcopy(pycamera.params)

        # Grabs original W / H
        real_image_size = original_image_sizes[pyimageid - 1, -2:]
        scale_x = real_image_size[0] / image_size[0]
        scale_y = real_image_size[1] / image_size[1]

        # --------------------------------
        # Rescale camera intrinsics
        # --------------------------------
        # Non-shared: rescale every time
        # Shared: rescale exactly once
        if rescale_camera and (not shared_camera or shared_intrinsics is None):

            # -------------------------------
            # Rescale focal length parameters
            # -------------------------------
            if pycamera.model.name == "SIMPLE_PINHOLE":
                pred_params[0] *= max(scale_x, scale_y)
            elif pycamera.model.name in ("PINHOLE", "OPENCV", "RADIAL", "OPENCV_FISHEYE"):
                pred_params[0] *= scale_x  # fx
                pred_params[1] *= scale_y  # fy

            # -------------------------------
            # Rescale principal point (cx, cy)
            # -------------------------------
            pred_params[-2] *= scale_x
            pred_params[-1] *= scale_y

            # -------------------------------
            # Apply back to camera object
            # -------------------------------
            if shared_camera:
                # First image defines the shared camera
                shared_intrinsics = pred_params
                shared_width = real_image_size[0]
                shared_height = real_image_size[1]

                pycamera.params = shared_intrinsics
                pycamera.width = shared_width
                pycamera.height = shared_height
            else:
                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

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

def _filter_outlier_cameras(
    reconstruction: Any,
    match_outputs: Optional[Dict[str, Any]] = None,
    depth_conf: Optional[np.ndarray] = None,
    extrinsic: Optional[np.ndarray] = None,
    images: Optional[torch.Tensor] = None,
    min_points_3d: int = 50,
    error_std_factor: float = 2.0,
    min_matches: int = 100,
    min_avg_depth_conf: float = 0.3,
    reprojection_error_percentile: float = 95,
    pose_distance_std_factor: float = 2.0,
    rotation_angle_threshold: float = 30.0,
    verbose: bool = False,
) -> Tuple[Any, List[int]]:
    """
    Filter outlier cameras based on multiple quality metrics.

    Inspired by CityGaussian SfM outlier detection:
    https://github.com/Linketic/CityGaussian/blob/e84c7c8774dd11d3f4189be3488e1220afa20a86/internal/utils/sfm_outlier_detection.py

    Args:
        reconstruction: pycolmap Reconstruction object
        match_outputs: Optional dictionary containing match information from global alignment
        depth_conf: Optional depth confidence array (N, H, W)
        extrinsic: Optional extrinsic matrices (N, 4, 4) for pose discontinuity detection
        images: Optional image tensor (not currently used)
        min_points_3d: Minimum number of 3D points a camera must have
        error_std_factor: Number of standard deviations above mean for error filtering
        min_matches: Minimum number of feature matches a camera must have
        min_avg_depth_conf: Minimum average depth confidence threshold
        reprojection_error_percentile: Percentile threshold for reprojection error filtering
        pose_distance_std_factor: Number of standard deviations for pose discontinuity detection
        rotation_angle_threshold: Maximum rotation angle change (degrees) between consecutive frames
        verbose: Whether to print filtering details

    Returns:
        Tuple of (filtered reconstruction, list of removed image IDs)
    """
    import copy

    filtered_reconstruction = copy.deepcopy(reconstruction)
    removed_image_ids = []

    # Metric 1: Number of 3D points each camera contributes to
    image_point_counts = {}
    for img_id in reconstruction.images:
        point_count = len([p for p in reconstruction.images[img_id].points2D if p.point3D_id != -1])
        image_point_counts[img_id] = point_count

    # Metric 2: Epipolar error per camera (if match_outputs available)
    image_epipolar_errors = {}
    if match_outputs is not None and "indexes_i_expanded" in match_outputs:
        corr_weights = match_outputs["corr_weights"]
        indexes_i = match_outputs["indexes_i_expanded"]
        indexes_j = match_outputs["indexes_j_expanded"]

        # Convert to numpy for efficiency if it's a tensor
        if isinstance(corr_weights, torch.Tensor):
            if corr_weights.ndim == 3:
                # Flatten batched weights (P, N, 1) -> flat array matching indexes
                num_matches = match_outputs["num_matches"]
                weights_list = []
                for p_idx in range(len(num_matches)):
                    weights_list.append(corr_weights[p_idx, :num_matches[p_idx], 0].cpu().numpy())
                corr_weights_flat = np.concatenate(weights_list)
            else:
                corr_weights_flat = corr_weights.cpu().numpy()
        else:
            corr_weights_flat = corr_weights

        # Compute per-camera error contribution
        for img_id in reconstruction.images:
            # Find matches involving this camera (COLMAP uses 1-indexed, matches use 0-indexed)
            mask_i = (indexes_i == img_id - 1)
            mask_j = (indexes_j == img_id - 1)
            mask = mask_i | mask_j

            if mask.sum() > 0:
                # Use match weight as error proxy (1.0 - weight = error)
                avg_error = float((1.0 - corr_weights_flat[mask]).mean())
                image_epipolar_errors[img_id] = avg_error
            else:
                image_epipolar_errors[img_id] = float('inf')

    # Metric 3: Average depth confidence (if available)
    image_depth_conf = {}
    if depth_conf is not None:
        for img_id in reconstruction.images:
            avg_conf = depth_conf[img_id - 1].mean()
            image_depth_conf[img_id] = avg_conf

    # Filter by number of matches (if match_outputs available)
    if match_outputs is not None and "indexes_i_expanded" in match_outputs:
        indexes_i = match_outputs["indexes_i_expanded"]
        indexes_j = match_outputs["indexes_j_expanded"]

        for img_id in reconstruction.images:
            # Count matches involving this camera
            mask_i = (indexes_i == img_id - 1)
            mask_j = (indexes_j == img_id - 1)
            match_count = (mask_i | mask_j).sum()

            if match_count < min_matches:
                removed_image_ids.append(img_id)
                if verbose:
                    CONSOLE.print(f"[yellow]Removing image {img_id}: only {match_count} matches (< {min_matches})")

    # Filter by 3D points
    for img_id, point_count in image_point_counts.items():
        if point_count < min_points_3d and img_id not in removed_image_ids:
            removed_image_ids.append(img_id)
            if verbose:
                CONSOLE.print(f"[yellow]Removing image {img_id}: only {point_count} 3D points (< {min_points_3d})")

    # Filter by average depth confidence
    if depth_conf is not None:
        for img_id in reconstruction.images:
            if img_id in removed_image_ids:
                continue
            avg_conf = depth_conf[img_id - 1].mean()
            if avg_conf < min_avg_depth_conf:
                removed_image_ids.append(img_id)
                if verbose:
                    CONSOLE.print(f"[yellow]Removing image {img_id}: depth conf {avg_conf:.3f} < {min_avg_depth_conf}")

    # Statistical filtering by epipolar error
    if image_epipolar_errors:
        errors = np.array(list(image_epipolar_errors.values()))
        valid_errors = errors[errors != float('inf')]

        if len(valid_errors) > 0:
            mean_error = valid_errors.mean()
            std_error = valid_errors.std()
            threshold = mean_error + error_std_factor * std_error

            for img_id, error in image_epipolar_errors.items():
                if error > threshold and img_id not in removed_image_ids:
                    removed_image_ids.append(img_id)
                    if verbose:
                        CONSOLE.print(f"[yellow]Removing image {img_id}: epipolar error {error:.3f} > {threshold:.3f}")

    # Filter by pose discontinuity (MAD from neighbors)
    if extrinsic is not None:
        camera_positions = extrinsic[:, :3, 3]
        n_neighbors = 3  # number of neighbors on each side
        num_cameras = len(camera_positions)

        for i in range(num_cameras):
            img_id = i + 1
            if img_id in removed_image_ids:
                continue

            # Select neighbor indices
            neighbor_indices = [j for j in range(max(0, i - n_neighbors), min(num_cameras, i + n_neighbors + 1))
                                if j != i]

            # Compute mean absolute deviation from neighbors
            neighbor_positions = camera_positions[neighbor_indices]
            mad = np.mean(np.linalg.norm(neighbor_positions - camera_positions[i], axis=1))

            # Compute global MAD threshold
            all_mads = []
            for k in range(num_cameras):
                neighbor_idx_k = [j for j in range(max(0, k - n_neighbors), min(num_cameras, k + n_neighbors + 1))
                                if j != k]
                neighbor_pos_k = camera_positions[neighbor_idx_k]
                all_mads.append(np.mean(np.linalg.norm(neighbor_pos_k - camera_positions[k], axis=1)))
            threshold_mad = np.mean(all_mads) + pose_distance_std_factor * np.std(all_mads)

            if mad > threshold_mad:
                removed_image_ids.append(img_id)
                if verbose:
                    CONSOLE.print(f"[yellow]Removing image {img_id}: pose MAD {mad:.3f} > {threshold_mad:.3f}")

    # Filter by rotation discontinuity
    if extrinsic is not None and len(extrinsic) > 2:
        for i in range(1, len(extrinsic) - 1):
            img_id = i + 1
            if img_id in removed_image_ids:
                continue
            R_curr = extrinsic[i, :3, :3]
            R_prev = extrinsic[i-1, :3, :3]
            R_next = extrinsic[i+1, :3, :3]

            # Relative rotations
            R_rel_prev = R_prev.T @ R_curr
            R_rel_next = R_curr.T @ R_next

            # Rotation angle (degrees)
            angle_prev = np.degrees(np.arccos(np.clip((np.trace(R_rel_prev) - 1)/2, -1, 1)))
            angle_next = np.degrees(np.arccos(np.clip((np.trace(R_rel_next) - 1)/2, -1, 1)))

            if angle_prev > rotation_angle_threshold or angle_next > rotation_angle_threshold:
                removed_image_ids.append(img_id)
                if verbose:
                    max_angle = max(angle_prev, angle_next)
                    CONSOLE.print(f"[yellow]Removing image {img_id}: rotation jump {max_angle:.1f}° > {rotation_angle_threshold}°")

    # Remove filtered images from reconstruction
    # IMPORTANT: Must call deregister_image() before deleting to maintain
    # internal registration state consistency. Without this, num_reg_images()
    # will be out of sync with the actual image count, causing write_binary()
    # to create corrupted files that fail to read with:
    # "struct.error: unpack requires a buffer of 64 bytes"
    removed_image_ids = list(set(removed_image_ids))
    for img_id in removed_image_ids:
        filtered_reconstruction.deregister_image(img_id)
        del filtered_reconstruction.images[img_id]

    # Clean up 3D points that lost observations
    points_to_remove = []
    for point3D_id in filtered_reconstruction.points3D:
        track = filtered_reconstruction.points3D[point3D_id].track
        # Remove track elements from deleted cameras
        track.elements = [elem for elem in track.elements if elem.image_id not in removed_image_ids]

        # If track has no observations left, mark for removal
        # Note: For track-free reconstructions (VGGT), tracks typically have 1 element
        # so we only remove if ALL observations are gone
        if len(track.elements) == 0:
            points_to_remove.append(point3D_id)

    for point3D_id in points_to_remove:
        del filtered_reconstruction.points3D[point3D_id]

    if verbose:
        CONSOLE.print(f"[bold green]✓ Filtered {len(removed_image_ids)} outlier cameras")
        CONSOLE.print(f"  - Removed {len(points_to_remove)} 3D points with insufficient observations")
        CONSOLE.print(f"  - Remaining: {len(filtered_reconstruction.images)} cameras, {len(filtered_reconstruction.points3D)} points")

    return filtered_reconstruction, removed_image_ids


def _filter_and_prepare_points_for_pycolmap(
    points3d: np.ndarray,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    images: torch.Tensor,
    image_paths: List[Path],
    conf_thres_value: float,
    max_points_for_colmap: int = 500000,
    use_global_alignment: bool = False,
    match_outputs: Optional[Dict[str, Any]] = None,
):
    """
    Prepares and filters 3D points, pixel coordinates, and colors for pycolmap ingestion.
    Adapts selection to use logic as in VGGT-X.

    Args:
        points3d (np.ndarray): 3D points of shape (S, H, W, 3)
        depth_map (np.ndarray): Depth map, shape (S, H, W, 1) or (S, H, W)
        depth_conf (np.ndarray): Confidence values, shape (S, H, W)
        images (torch.Tensor): Images tensor, shape (S, 3, H_raw, W_raw)
        conf_thres_value (float): Confidence threshold value
        max_points_for_colmap (int): Max number of points for COLMAP
        match_outputs, base_image_path_list: Used if use_global_alignment is True

    Returns:
        Tuple of: (filtered_points_3d, filtered_points_xyf, filtered_points_rgb)
    """

    base_image_path_list = [image_path.name for image_path in image_paths]

    num_frames, height, width, _ = points3d.shape

    # Rescale images to match depth map shape, get RGB colors
    points_rgb = F.interpolate(
        images, size=(depth_map.shape[1], depth_map.shape[2]), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().float().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (S, H, W, 3)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    if use_global_alignment:
        conf_mask = extract_conf_mask(match_outputs, depth_conf, base_image_path_list)
        conf_mask = conf_mask & (depth_conf >= conf_thres_value)
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
    else:
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    filtered_points_3d = points3d[conf_mask]
    filtered_points_xyf = points_xyf[conf_mask]
    filtered_points_rgb = points_rgb[conf_mask]

    return filtered_points_3d, filtered_points_xyf, filtered_points_rgb

def _run_global_alignment(
    images: torch.Tensor,
    image_paths: List[Path],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_conf: np.ndarray,
    colmap_dir: Path,
    max_query_pts: int = 4096,
    shared_camera: bool = False,
    verbose: bool = False,
    depth_map: Optional[np.ndarray] = None,
    lambda_depth: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run Global Alignment to refine camera poses using feature matching.

    This implements VGGT-X's Global Alignment module which:
    1. Extracts feature matches between frames
    2. Optimizes camera poses using these matches
    3. Returns refined extrinsic and intrinsic matrices

    Args:
        extrinsic: Initial camera extrinsics (N, 4, 4)
        intrinsic: Initial camera intrinsics (N, 3, 3)
        depth: Depth maps (N, H, W, 1)
        depth_conf: Depth confidence maps (N, H, W)
        image_paths: Paths to images
        max_query_pts: Maximum query points for matching
        shared_camera: Whether to use shared camera model
        colmap_dir: Output directory for saving matches
        verbose: Whether to print progress

    Returns:
        Tuple of (refined_extrinsic, refined_intrinsic)
    """
    if not is_vggt_available():
        CONSOLE.print(
            "[bold red]Error: To use Global Alignment, you must install VGGT-X!\n"
            "Visit https://github.com/Linketic/VGGT-X.git for installation instructions."
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Grab basenames from the image paths
    image_basenames = [image_path.name for image_path in image_paths]

    matches_path = colmap_dir / "matches.pt"
    # if matches_path.exists():
    #     match_outputs = torch.load(matches_path)
    # else:
    match_outputs = extract_matches(
        images=images,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        base_image_path_list=image_basenames,
        max_query_pts=max_query_pts,
    )
    torch.save(match_outputs, matches_path)

    # Run pose optimization (expects numpy arrays for extrinsic/intrinsic)
    if verbose:
        CONSOLE.print(f"  - Optimizing camera poses...")
    
    extrinsic_refined, intrinsic_refined = pose_optimization(
        match_outputs=match_outputs,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        images=images,
        depth_conf=depth_conf,
        base_image_path_list=image_basenames,
        target_scene_dir=colmap_dir,
        shared_intrinsics=shared_camera,
        depth_maps=depth_map,
        lambda_depth=lambda_depth,
    )

    # pose_optimization already returns numpy arrays, no conversion needed

    # Clean up
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return refined poses AND match_outputs for downstream filtering
    # (Following VGGT-X demo_colmap.py pattern)
    return extrinsic_refined, intrinsic_refined, match_outputs

################################################################################
######################  VGGT-X HELPER FUNCTIONS   ##############################
################################################################################

### Taken from opt.py (VGGT-X)

def make_K_cam_depth(log_focals, pps, trans, quats, min_focals, max_focals, imsizes):
    # make intrinsics
    focals = log_focals.exp().clip(min=min_focals, max=max_focals)
    K = torch.eye(4, dtype=focals.dtype, device=focals.device)[None].expand(len(trans), 4, 4).clone()
    K[:, 0, 0] = K[:, 1, 1] = focals
    K[:, 0:2, 2] = pps * imsizes
    if trans is None:
        return K

    w2cs = torch.eye(4, dtype=trans.dtype, device=trans.device)[None].expand(len(trans), 4, 4).clone()
    w2cs[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(quats, dim=1))
    w2cs[:, :3, 3] = trans

    return K, (w2cs, torch.linalg.inv(w2cs))

def get_default_lr(epipolar_err, bound1=2.5, bound2=7.5):

    assert bound2 > bound1, print("bound2 should be greater than bound1")

    if epipolar_err > bound2:
        lr_base = 1e-2
    elif epipolar_err < bound1:
        lr_base = 5e-4
    else:
        lr_base = 1e-3
    lr_end = lr_base / 10

    return lr_base, lr_end

def cosine_schedule(alpha, lr_base, lr_end=0):
    lr = lr_end + (lr_base - lr_end) * (1 + np.cos(alpha * np.pi)) / 2
    return lr

def adjust_learning_rate_by_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

def unproject_points(points_2d, depth, K, cam2w):
    """
    Unproject 2D image points with depth to 3D world coordinates.

    Args:
        points_2d: (N, 2) tensor of (x, y) image coordinates
        depth: (N, 1) or (N,) tensor of depth values
        K: (4, 4) intrinsic matrix
        cam2w: (4, 4) camera-to-world extrinsic matrix

    Returns:
        points_3d: (N, 3) tensor of 3D world coordinates
    """
    # Ensure depth has shape (N, 1)
    if depth.dim() == 1:
        depth = depth.unsqueeze(-1)

    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Unproject to camera coordinates
    x_cam = (points_2d[:, 0] - cx) * depth.squeeze(-1) / fx
    y_cam = (points_2d[:, 1] - cy) * depth.squeeze(-1) / fy
    z_cam = depth.squeeze(-1)

    # Stack to homogeneous coordinates
    points_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)  # (N, 4)

    # Transform to world coordinates: (4, 4) @ (4, N) -> (4, N) -> (N, 4)
    points_world = (cam2w @ points_cam.T).T

    return points_world[:, :3]

def project_to_depth(points_3d, K, w2cam):
    """
    Project 3D world points to depth values in camera view.

    Args:
        points_3d: (N, 3) tensor of 3D world coordinates
        K: (4, 4) intrinsic matrix
        w2cam: (4, 4) world-to-camera extrinsic matrix

    Returns:
        depth: (N, 1) tensor of depth values
    """
    # Convert to homogeneous coordinates
    points_3d_hom = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)  # (N, 4)

    # Transform to camera coordinates: (4, 4) @ (4, N) -> (4, N) -> (N, 4)
    points_cam = (w2cam @ points_3d_hom.T).T

    # Return depth (z-coordinate in camera space)
    return points_cam[:, 2:3]

def compute_depth_consistency_loss(
    corr_points_i: torch.Tensor,
    corr_points_j: torch.Tensor,
    depth_maps_tensor: torch.Tensor,
    indexes_i: list,
    indexes_j: list,
    Ks_i: torch.Tensor,
    Ks_j: torch.Tensor,
    cam2w_i: torch.Tensor,
    cam2w_j: torch.Tensor,
    w2cam_j: torch.Tensor,
    imsizes: torch.Tensor,
    corr_weight_valid: torch.Tensor,
) -> torch.Tensor:
    """
    Compute depth consistency loss for feature correspondences.

    This function validates correspondences by checking if 3D points unprojected
    from view i project to consistent depths in view j.

    Args:
        corr_points_i: (P, N, 2) correspondence points in view i (batched by pairs)
        corr_points_j: (P, N, 2) correspondence points in view j (batched by pairs)
        depth_maps_tensor: (num_images, H, W) depth maps for all images
        indexes_i: List of image indices for view i (length P)
        indexes_j: List of image indices for view j (length P)
        Ks_i: (P, 4, 4) intrinsic matrices for view i
        Ks_j: (P, 4, 4) intrinsic matrices for view j
        cam2w_i: (P, 4, 4) camera-to-world transforms for view i
        cam2w_j: (P, 4, 4) camera-to-world transforms for view j
        w2cam_j: (P, 4, 4) world-to-camera transforms for view j
        imsizes: (2,) tensor of [width, height]
        corr_weight_valid: (P, N, 1) correspondence weights (batched by pairs)

    Returns:
        Depth consistency loss (scalar tensor)
    """
    P, N = corr_points_i.shape[:2]  # P = num pairs, N = max matches per pair

    # Normalize correspondence points to [-1, 1] for grid_sample
    grid_i = corr_points_i / imsizes * 2 - 1  # (P, N, 2)
    grid_j = corr_points_j / imsizes * 2 - 1  # (P, N, 2)

    all_depth_errors = []
    all_weights = []

    # Process each image pair
    for pair_idx, (img_idx_i, img_idx_j) in enumerate(zip(indexes_i, indexes_j)):
        # Get depth maps for this image pair
        # Handle both (num_images, H, W) and (num_images, 1, H, W) shapes
        depth_i = depth_maps_tensor[img_idx_i]
        depth_j = depth_maps_tensor[img_idx_j]

        # Ensure we have (1, 1, H, W) shape for grid_sample
        if depth_i.dim() == 2:  # (H, W)
            depth_i_map = depth_i.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif depth_i.dim() == 3:  # (1, H, W)
            depth_i_map = depth_i.unsqueeze(0)  # (1, 1, H, W)
        else:  # 4D or higher
            depth_i_map = depth_i.reshape(1, 1, depth_i.shape[-2], depth_i.shape[-1])

        if depth_j.dim() == 2:  # (H, W)
            depth_j_map = depth_j.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif depth_j.dim() == 3:  # (1, H, W)
            depth_j_map = depth_j.unsqueeze(0)  # (1, 1, H, W)
        else:  # 4D or higher
            depth_j_map = depth_j.reshape(1, 1, depth_j.shape[-2], depth_j.shape[-1])

        # Get correspondences for this pair
        grid_i_pair = grid_i[pair_idx]  # (N, 2)
        grid_j_pair = grid_j[pair_idx]  # (N, 2)
        corr_pts_i_pair = corr_points_i[pair_idx]  # (N, 2)
        weights_pair = corr_weight_valid[pair_idx]  # (N, 1)

        # Sample depths at all correspondence locations for this pair
        # grid_sample expects (B, H, W, 2) format for grid
        num_corr = grid_i_pair.shape[0]

        # grid_sample with input (1, 1, H, W) and grid (1, N, 1, 2) outputs (1, 1, N, 1)
        sample_i = torch.nn.functional.grid_sample(
            depth_i_map,  # (1, 1, H, W)
            grid_i_pair.view(1, num_corr, 1, 2),
            align_corners=True,
            mode='bilinear'
        )[0, 0, :, 0]  # Extract to get (N,)

        sample_j = torch.nn.functional.grid_sample(
            depth_j_map,  # (1, 1, H, W)
            grid_j_pair.view(1, num_corr, 1, 2),
            align_corners=True,
            mode='bilinear'
        )[0, 0, :, 0]  # Extract to get (N,)

        # Get camera parameters for this pair
        K_i = Ks_i[pair_idx]  # (4, 4)
        K_j = Ks_j[pair_idx]  # (4, 4)
        c2w_i = cam2w_i[pair_idx]  # (4, 4)
        w2c_j = w2cam_j[pair_idx]  # (4, 4)

        # Unproject all points from view i to 3D
        pts_3d = unproject_points(corr_pts_i_pair, sample_i.unsqueeze(-1), K_i, c2w_i)  # (N, 3)

        # Project all 3D points to depth in view j
        projected_depths = project_to_depth(pts_3d, K_j, w2c_j)  # (N, 1)

        # Compute depth consistency error for this pair
        depth_errors = torch.abs(projected_depths.squeeze(-1) - sample_j)  # (N,)

        all_depth_errors.append(depth_errors)
        all_weights.append(weights_pair.squeeze(-1))

    # Concatenate all errors and weights
    all_depth_errors = torch.cat(all_depth_errors)  # (P*N,)
    all_weights = torch.cat(all_weights)  # (P*N,)

    # Compute weighted mean depth consistency loss
    depth_consistency_loss = (all_depth_errors * all_weights).sum() / (all_weights.sum() + 1e-8)

    return depth_consistency_loss

def l1_loss(x, y):
    return torch.linalg.norm(x - y, dim=-1)

def gamma_loss(gamma, mul=1, offset=None, clip=np.inf):
    if offset is None:
        if gamma == 1:
            return l1_loss
        # d(x**p)/dx = 1 ==> p * x**(p-1) == 1 ==> x = (1/p)**(1/(p-1))
        offset = (1 / gamma)**(1 / (gamma - 1))

    def loss_func(x, y):
        return (mul * l1_loss(x, y).clip(max=clip) + offset) ** gamma - offset ** gamma
    return loss_func

def image_pair_candidates(extrinsic, pairing_angle_threshold=30, unique_pairs=False):

    pairs, pairs_cnt = {}, 0

    # assert i_map is None or len(i_map) == len(extrinsics)

    num_images = len(extrinsic)
    
    extrinsic_tensor = torch.from_numpy(extrinsic)

    for i in range(num_images):
        
        rot_mat_i = extrinsic_tensor[i:i+1, :3, :3]
        rot_mats_j = extrinsic_tensor[i+1:, :3, :3]

        rot_mat_ij = torch.matmul(rot_mat_i, torch.linalg.inv(rot_mats_j))
        angle_rad = torch.acos((rot_mat_ij.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2)
        angle_deg = angle_rad / np.pi * 180

        i_entry = i
        j_entries = (i + 1 + torch.where(torch.abs(angle_deg) < pairing_angle_threshold)[0]).tolist()

        pairs_cnt += len(j_entries)

        if not i_entry in pairs.keys():
            pairs[i_entry] = []
        pairs[i_entry] = pairs[i_entry] + j_entries

        if not unique_pairs:
            for j_entry in j_entries:
                if not j_entry in pairs.keys():
                    pairs[j_entry] = []
                pairs[j_entry].append(i_entry)

    return pairs, pairs_cnt

def extract_conf_mask(match_outputs, depth_conf, base_image_path_list):

    conf_mask = np.zeros_like(depth_conf, dtype=bool)
    corr_points_i = np.round(match_outputs["corr_points_i"].cpu().numpy()).astype(int)
    corr_points_j = np.round(match_outputs["corr_points_j"].cpu().numpy()).astype(int)
    indexes_i = [base_image_path_list.index(img_name) for img_name in match_outputs["image_names_i"]]
    indexes_j = [base_image_path_list.index(img_name) for img_name in match_outputs["image_names_j"]]
    corr_weights = match_outputs["corr_weights"].cpu().numpy()
    for i in range(len(indexes_i)):
        single_mask = (corr_weights[i] > 0.1)
        conf_mask[indexes_i[i], corr_points_i[i, single_mask[:, 0], 1], corr_points_i[i, single_mask[:, 0], 0]] = True
    for j in range(len(indexes_j)):
        if j not in indexes_i:
            single_mask = (corr_weights[j] > 0.1)
            conf_mask[indexes_j[j], corr_points_j[j, single_mask[:, 0], 1], corr_points_j[j, single_mask[:, 0], 0]] = True
    
    return conf_mask

@torch.inference_mode()
def extract_matches(
    images: torch.Tensor,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    base_image_path_list: List[str],
    max_query_pts: int = 4096,
    batch_size: int = 128,
    err_range: float = 20,
) -> Dict[str, Any]:
    """
    Returns:
        output_dict: Dictionary containing the following keys:
            - corr_points_i: torch.Tensor of shape (P, N, 2)
                # 2D keypoint coordinates in image i (first image of each pair)
                # shape: (P, N, 2) where
                #   P: number of image pairs,
                #   N: maximum number of matches across all pairs (zero-padded)
                #   2: (x, y) pixel coordinates (pixels, top-left origin)
                # corr_points_i[p, k] = 11th matched keypoint in 6th pair, i-th image

            - corr_points_j: torch.Tensor of shape (P, N, 2)
                # 2D keypoint coordinates in image j (second image of each pair)
                # Same structure as corr_points_i
                # corr_points_j[p, k] matches corr_points_i[p, k]

            - corr_weights: torch.Tensor of shape (P, N, 1)
                # Confidence weight for each correspondence, normalized (mean ≈ 1)
                # Range: 0 (out of frame) to >1 (high consistency)
                # Used to down-weight outliers during optimization

            - image_names_i: np.ndarray of shape (P,), dtype=string
                # Filenames or paths of first image in each pair

            - image_names_j: np.ndarray of shape (P,), dtype=string
                # Filenames or paths of second image in each pair

            - num_matches: List[int], length P
                # Number of valid (non-zero-padded) matches for each image pair
                # Use corr_points_i[p, :num_matches[p]] for valid matches in pair p

            - epipolar_err: float
                # Median symmetrical epipolar error over all correspondences (pixels)

            - indexes_i_expanded: np.ndarray of shape (M,), dtype=int64
                # Camera indices for image i, for every match (expanded/repeated per match)
                # M = sum(num_matches)

            - indexes_j_expanded: np.ndarray of shape (M,), dtype=int64
                # Camera indices for image j, for every match (expanded/repeated per match)
                # M = sum(num_matches)
    """

    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=max_query_pts)

    pairs, pairs_cnt = image_pair_candidates(extrinsic, 30, unique_pairs=True)
    print("Total candidate image pairs found: ", pairs_cnt)

    indexes_i = list(range(len(base_image_path_list)-1))  # the last image 
    indexes_j = [pairs[idx_i] for idx_i in indexes_i]
    indexes_i = [np.array([idx_i] * len(indexes_j[idx_i])) for idx_i in indexes_i]
    indexes_i = np.concatenate(indexes_i).tolist()
    indexes_j = np.concatenate(indexes_j).tolist()

    matches_list = []

    for i in tqdm(range(0, len(indexes_i), batch_size), desc="Matching image pairs..."):
        indexes_i_batch = indexes_i[i:i + batch_size]
        indexes_j_batch = indexes_j[i:i + batch_size]
        
        # Extract features for the batch
        images_i = images[indexes_i_batch]
        images_j = images[indexes_j_batch]
        
        # Match features
        matches_batch = xfeat.match_xfeat_star(images_i, images_j)
        if len(images_i) == 1:
            matches_batch = [torch.concatenate([torch.tensor(matches_batch[0], device=images_i.device), 
                                                torch.tensor(matches_batch[1], device=images_i.device)], dim=-1)]
        matches_list.extend(matches_batch)

    num_matches = [len(m) for m in matches_list]

    indexes_i_expanded = []
    indexes_j_expanded = []

    for idx, n in enumerate(num_matches):
        indexes_i_expanded.append(np.array([indexes_i[idx]] * n, dtype=np.int64))
        indexes_j_expanded.append(np.array([indexes_j[idx]] * n, dtype=np.int64))
    indexes_i_expanded = np.concatenate(indexes_i_expanded)
    indexes_j_expanded = np.concatenate(indexes_j_expanded)

    image_names_i = np.array(base_image_path_list)[indexes_i_expanded]
    image_names_j = np.array(base_image_path_list)[indexes_j_expanded]

    corr_points_i = torch.cat([matches_list[k][:, :2] for k in range(len(matches_list))], dim=0).cpu()
    corr_points_j = torch.cat([matches_list[k][:, 2:] for k in range(len(matches_list))], dim=0).cpu()

    intrinsic_i = np.zeros((corr_points_i.shape[0], 4, 4), dtype=np.float32)
    intrinsic_j = np.zeros((corr_points_j.shape[0], 4, 4), dtype=np.float32)
    intrinsic_i[:, :3, :3] = intrinsic[indexes_i_expanded]
    intrinsic_j[:, :3, :3] = intrinsic[indexes_j_expanded]
    intrinsic_i[:, 3, 3] = 1.0
    intrinsic_j[:, 3, 3] = 1.0

    extrinsic_i = np.zeros((corr_points_i.shape[0], 4, 4), dtype=np.float32)
    extrinsic_j = np.zeros((corr_points_j.shape[0], 4, 4), dtype=np.float32)
    extrinsic_i[:, :3, :4] = extrinsic[indexes_i_expanded]
    extrinsic_j[:, :3, :4] = extrinsic[indexes_j_expanded]
    extrinsic_i[:, 3, 3] = 1.0
    extrinsic_j[:, 3, 3] = 1.0

    device = corr_points_i.device

    intrinsic_i_tensor = torch.FloatTensor(intrinsic_i).to(device)
    intrinsic_j_tensor = torch.FloatTensor(intrinsic_j).to(device)
    extrinsic_i_tensor = torch.FloatTensor(extrinsic_i).to(device)
    extrinsic_j_tensor = torch.FloatTensor(extrinsic_j).to(device)

    P_i = intrinsic_i_tensor @ extrinsic_i_tensor
    P_j = intrinsic_j_tensor @ extrinsic_j_tensor
    Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
    err = kornia.geometry.symmetrical_epipolar_distance(corr_points_i[:, None, :2], corr_points_j[:, None, :2], Fm, squared=False, eps=1e-08)
    
    hist, bin_edges = torch.histogram(err.cpu(), bins=100, range=(0, err_range), density=True)  # move to cpu to avoid CUDA "backend"
    corr_weights = torch.zeros_like(err)
    for i in range(len(bin_edges) - 1):
        mask = (err >= bin_edges[i]) & (err < bin_edges[i + 1])
        if torch.any(mask):
            corr_weights[mask] = (hist[i] * (bin_edges[i + 1] - bin_edges[i])) / (bin_edges[-1] - bin_edges[0])
    corr_weights /= corr_weights.mean()
    
    # set corr_weights to 0 for points outside the image frame
    out_of_frame_i = (corr_points_i[..., 0] >= images.shape[-1]) | (corr_points_i[..., 0] < 0) | \
                     (corr_points_i[..., 1] >= images.shape[-2]) | (corr_points_i[..., 1] < 0)
    out_of_frame_j = (corr_points_j[..., 0] >= images.shape[-1]) | (corr_points_j[..., 0] < 0) | \
                     (corr_points_j[..., 1] >= images.shape[-2]) | (corr_points_j[..., 1] < 0)
    corr_weights[out_of_frame_i | out_of_frame_j] = 0.0
    
    # rearrange corr_points_i_normalized and corr_points_j_normalized to (P, N, 2)
    P, N = len(num_matches), max(num_matches)
    corr_points_i_batched = torch.zeros((P, N, 2), dtype=corr_points_i.dtype, device=corr_points_i.device)
    corr_points_j_batched = torch.zeros((P, N, 2), dtype=corr_points_j.dtype, device=corr_points_j.device)
    corr_weights_batched = torch.zeros((P, N, 1), dtype=corr_weights.dtype, device=corr_weights.device)
    image_names_i_batched = np.zeros((P), dtype=image_names_i.dtype)
    image_names_j_batched = np.zeros((P), dtype=image_names_j.dtype)

    start_idx = 0
    for p in range(P):
        end_idx = start_idx + num_matches[p]
        corr_points_i_batched[p, :num_matches[p]] = corr_points_i[start_idx:end_idx]
        corr_points_j_batched[p, :num_matches[p]] = corr_points_j[start_idx:end_idx]
        corr_weights_batched[p, :num_matches[p]] = corr_weights[start_idx:end_idx]
        image_names_i_batched[p] = image_names_i[start_idx]
        image_names_j_batched[p] = image_names_j[start_idx]
        assert (image_names_i[start_idx:end_idx] == image_names_i_batched[p]).all()
        assert (image_names_j[start_idx:end_idx] == image_names_j_batched[p]).all()
        start_idx = end_idx
    
    output_dict = {
        "corr_points_i": corr_points_i_batched,
        "corr_points_j": corr_points_j_batched,
        "corr_weights": corr_weights_batched,
        "image_names_i": image_names_i_batched,
        "image_names_j": image_names_j_batched,
        "num_matches": num_matches,
        "epipolar_err": err.median().item(),
        "indexes_i_expanded": indexes_i_expanded,
        "indexes_j_expanded": indexes_j_expanded,
    }

    return output_dict

def pose_optimization(
    match_outputs: Dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    images: torch.Tensor,
    depth_conf: np.ndarray,
    base_image_path_list: List[str],
    target_scene_dir: Optional[str],
    device: str = 'cuda',
    lr_base: Optional[float] = None,
    lr_end: Optional[float] = None,
    lambda_epi: float = 1.0,
    lambda_depth: float = 0.0,
    niter: int = 300,
    shared_intrinsics: bool = True,
    depth_maps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    torch.cuda.empty_cache()
    
    if lr_base is None or lr_end is None:
        lr_base, lr_end = get_default_lr(match_outputs["epipolar_err"])
    
    with torch.no_grad():
        imsizes = torch.tensor([images.shape[-1], images.shape[-2]]).float()
        diags = torch.norm(imsizes)
        min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
        max_focals = 10 * diags

        qvec = roma.rotmat_to_unitquat(torch.tensor(extrinsic[:, :3, :3]))
        tvec = torch.tensor(extrinsic[:, :3, 3])
        log_sizes = torch.zeros(len(qvec))

        pps = torch.tensor(intrinsic[:, :2, 2]) / imsizes[None, :2]  # default principal_point would be (0.5, 0.5)
        base_focals = torch.tensor((intrinsic[:, 0, 0] + intrinsic[:, 1, 1]) / 2)

        # intrinsics parameters
        if shared_intrinsics:
            # Optimize a single set of intrinsics for all cameras. Use averages as init.
            confs = depth_conf.mean(axis=(1, 2))
            weighting = torch.tensor(confs / confs.sum())
            pps = weighting @ pps
            pps = pps.view(1, -1)
            focal_m = weighting @ base_focals
            log_focals = focal_m.view(1).log()
        else:
            log_focals = base_focals.log()

        corr_points_i = match_outputs["corr_points_i"].clone()
        corr_points_j = match_outputs["corr_points_j"].clone()
        corr_weights = match_outputs["corr_weights"].clone()
        num_matches = match_outputs["num_matches"]
        indexes_i = [base_image_path_list.index(img_name) for img_name in match_outputs["image_names_i"]]
        indexes_j = [base_image_path_list.index(img_name) for img_name in match_outputs["image_names_j"]]
        imsizes = imsizes.to(corr_points_i.device)

        # Prepare depth maps if depth consistency is enabled
        if depth_maps is not None and lambda_depth > 0:
            depth_maps_tensor = torch.from_numpy(depth_maps).float()
        else:
            depth_maps_tensor = None

    qvec = qvec.to(device)
    tvec = tvec.to(device)
    log_sizes = log_sizes.to(device)
    min_focals = min_focals.to(device)
    max_focals = max_focals.to(device)
    imsizes = imsizes.to(device)
    pps = pps.to(device)
    log_focals = log_focals.to(device)

    corr_points_i = corr_points_i.to(device)
    corr_points_j = corr_points_j.to(device)
    corr_weight_valid = corr_weights.to(device)
    corr_weight_valid = corr_weight_valid**(0.5)
    corr_weight_valid /= corr_weight_valid.mean()

    if depth_maps_tensor is not None:
        depth_maps_tensor = depth_maps_tensor.to(device)

    params = [{
        "params": [
            qvec.requires_grad_(True), 
            tvec.requires_grad_(True), 
            log_sizes.requires_grad_(True),
            log_focals.requires_grad_(True),
            pps.requires_grad_(True)
        ],
        "name": ["qvec", "tvec", "log_sizes", "log_focals", "pps"],
    }]

    optimizer = torch.optim.Adam(params, lr=1, weight_decay=0, betas=(0.9, 0.9))

    loss_list = []
    for iter in tqdm(range(niter or 1), desc="Pose Optimization..."):

        repeat_cnt = 1 if len(qvec) else shared_intrinsics
        
        K, (w2cam, cam2w) = make_K_cam_depth(log_focals.repeat(repeat_cnt), pps.repeat(repeat_cnt, 1), tvec, qvec, min_focals, max_focals, imsizes)
        
        alpha = (iter / niter)
        lr = cosine_schedule(alpha, lr_base, lr_end)
        adjust_learning_rate_by_lr(optimizer, lr)
        optimizer.zero_grad()

        # Convert index lists to tensors for proper indexing
        indexes_i_tensor = torch.tensor(indexes_i, dtype=torch.long, device=K.device)
        indexes_j_tensor = torch.tensor(indexes_j, dtype=torch.long, device=K.device)

        Ks_i = K[indexes_i_tensor]
        Ks_j = K[indexes_j_tensor]
        w2cam_i = w2cam[indexes_i_tensor]
        w2cam_j = w2cam[indexes_j_tensor]

        loss = 0.0

        # Epipolar geometry loss
        P_i = Ks_i @ w2cam_i
        P_j = Ks_j @ w2cam_j
        Fm = kornia.geometry.epipolar.fundamental_from_projections(P_i[:, :3], P_j[:, :3])
        err = kornia.geometry.symmetrical_epipolar_distance(corr_points_i, corr_points_j, Fm, squared=False, eps=1e-08)
        loss = (err * corr_weight_valid.squeeze(-1)).mean() * lambda_epi

        # Depth consistency loss (optional)
        if depth_maps_tensor is not None and lambda_depth > 0:
            cam2w_i = cam2w[indexes_i_tensor]
            cam2w_j = cam2w[indexes_j_tensor]

            depth_consistency_loss = compute_depth_consistency_loss(
                corr_points_i=corr_points_i,
                corr_points_j=corr_points_j,
                depth_maps_tensor=depth_maps_tensor,
                indexes_i=indexes_i,
                indexes_j=indexes_j,
                Ks_i=Ks_i,
                Ks_j=Ks_j,
                cam2w_i=cam2w_i,
                cam2w_j=cam2w_j,
                w2cam_j=w2cam_j,
                imsizes=imsizes,
                corr_weight_valid=corr_weight_valid,
            )

            loss += lambda_depth * depth_consistency_loss

        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
    
    if target_scene_dir is not None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title(f'Loss Curve, final loss={loss_list[-1]:.4f}')
        # plt.show()
        plt.savefig(f"{target_scene_dir}/loss_curve_pose_opt.jpg")
        plt.close()  # Close figure to free memory

    
    output_extrinsic = w2cam[:, :3, :4].detach().cpu().numpy()
    output_intrinsic = K[:, :3, :3].detach().cpu().numpy()

    return output_extrinsic, output_intrinsic