# collab_splats/pointcloud/utils.py
"""Geometric pointcloud utilities: filtering, downsampling, and coordinate conversion.

Coordinate convention used throughout:
  - Input from pycolmap uses COLMAP world (Y-down) + OpenCV camera axes (X right, Y down, Z forward).
  - All public functions that produce poses output CoordinateFrame.NERFSTUDIO (nerfstudio world frame):
    X right, Y up, Z backward camera axes; Z-up world.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses

from .base import PointcloudResult

if TYPE_CHECKING:
    from .feedforward.base import FeedforwardResult

try:
    from collab_splats.semantics.features import BaseFeatureExtractor
except ImportError:
    BaseFeatureExtractor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_DOWNSAMPLE_KWARGS: dict = {"voxel_size": 0.015, "adaptive": True}
_DEFAULT_OUTLIER_KWARGS: dict = {"nb_neighbors": 20, "std_ratio": 2.0}
_DEFAULT_DISTANCE_KWARGS: dict = {"method": "radial", "max_distance": 50.0}

# Sentinel to distinguish "use defaults" from "skip this step"
_UNSET = object()


########################################################
########## Confidence masking helpers ##################
########################################################


def _radial_mask(
    points: np.ndarray,
    max_distance: Optional[float] = None,
    n_points: Optional[int] = None,
    reference: str = "centroid",
) -> np.ndarray:
    """Compute a boolean keep-mask using Euclidean distance from a reference point.

    Args:
        points:       (N, 3) float32 world-space point coordinates.
        max_distance: Keep points within this Euclidean distance from reference.
                      At least one of max_distance / n_points must be provided.
        n_points:     Keep the N closest points to reference.
        reference:    'centroid' uses points.mean(axis=0); 'origin' uses [0,0,0].

    Returns:
        (N,) boolean array; True = keep.

    Raises:
        ValueError: If both max_distance and n_points are None, or reference is unknown.
    """
    if max_distance is None and n_points is None:
        raise ValueError("_radial_mask requires max_distance or n_points.")
    if reference == "centroid":
        ref = points.mean(axis=0)
    elif reference == "origin":
        ref = np.zeros(3)
    else:
        raise ValueError("reference must be 'centroid' or 'origin'")
    distances = np.linalg.norm(points - ref, axis=1)
    if max_distance is not None:
        mask = distances <= max_distance
    else:
        if n_points > len(points):
            raise ValueError(f"n_points ({n_points}) is greater than the number of points ({len(points)}).")
        sorted_idx = np.argsort(distances)
        mask = np.zeros(len(points), dtype=bool)
        mask[sorted_idx[:n_points]] = True
    return mask


def _bbox_mask(
    points: np.ndarray,
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    max_extent: Optional[float] = None,
) -> np.ndarray:
    """Compute a boolean keep-mask using a per-axis percentile bounding box.

    Computes per-axis min/max from percentile_range, then optionally clips the
    box to max_extent around its centre.

    Args:
        points:           (N, 3) float32 world-space point coordinates.
        percentile_range: (min_pct, max_pct) used to derive per-axis bounds.
                          E.g., (1.0, 99.0) removes the outermost 1 % on each axis.
        max_extent:       If set, clips the percentile box to this absolute size
                          around the box centre (world units).

    Returns:
        (N,) boolean array; True = keep.
    """
    pmin, pmax = percentile_range
    bbox_min = np.percentile(points, pmin, axis=0)
    bbox_max = np.percentile(points, pmax, axis=0)
    if max_extent is not None:
        center = (bbox_min + bbox_max) / 2
        half = max_extent / 2
        bbox_min = np.maximum(bbox_min, center - half)
        bbox_max = np.minimum(bbox_max, center + half)
    mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    return mask


########################################################
########## Distance and density filters ################
########################################################


def filter_distance(
    pcd,
    method: str = "radial",
    *,
    max_distance: Optional[float] = None,
    n_points: Optional[int] = None,
    reference: str = "centroid",
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    max_extent: Optional[float] = None,
    return_mask: bool = False,
):
    """Filter an Open3D point cloud by distance from a reference or bounding box.

    Args:
        pcd:              Open3D PointCloud to filter.
        method:           ``"radial"`` — Euclidean sphere filter; ``"bbox"`` — axis-aligned box.
        max_distance:     (radial) Keep points within this distance from reference.
        n_points:         (radial) Keep the N closest points.
        reference:        (radial) ``"centroid"`` or ``"origin"``.
        percentile_range: (bbox) (min_pct, max_pct) percentile bounds per axis.
        max_extent:       (bbox) Optional absolute max half-extent clamp around center.
        return_mask:      When True, return ``(pcd, mask)`` instead of just ``pcd``.

    Returns:
        Filtered Open3D PointCloud, or ``(pcd, mask)`` if ``return_mask=True``.
    """
    import open3d as o3d

    points = np.asarray(pcd.points)

    if method == "radial":
        mask = _radial_mask(points, max_distance, n_points, reference)
    elif method == "bbox":
        if len(points) == 0:
            mask = np.zeros(0, dtype=bool)
            filtered = pcd.select_by_index([])
            return (filtered, mask) if return_mask else filtered
        mask = _bbox_mask(points, percentile_range, max_extent)
    else:
        raise ValueError(f"Unknown filter_distance method: {method!r}. Use 'radial' or 'bbox'.")

    filtered = pcd.select_by_index(np.where(mask)[0])
    return (filtered, mask) if return_mask else filtered


########################################################
########## Primary cleaning pipeline ##################
########################################################


def clean_pointcloud(
    pcd,
    downsample_kwargs: Optional[dict] = _UNSET,
    outlier_kwargs: Optional[dict] = _UNSET,
    distance_kwargs: Optional[dict] = _UNSET,
) -> tuple[Any, np.ndarray]:
    """Clean an Open3D point cloud via composable filter steps.

    Applies up to three steps in order:
      1. Voxel downsampling (``voxel_downsample``)
      2. Statistical outlier removal (``pcd.remove_statistical_outlier``)
      3. Distance-based removal (``filter_distance``)

    Each step is enabled by passing a dict of kwargs (merged over module defaults)
    and disabled by passing ``None``.

    Args:
        pcd:               Open3D PointCloud to clean.
        downsample_kwargs: kwargs for ``voxel_downsample``, or None to skip.
        outlier_kwargs:    kwargs for ``remove_statistical_outlier``, or None to skip.
        distance_kwargs:   kwargs for ``filter_distance``, or None to skip.

    Returns:
        ``(cleaned_pcd, index_mapping)`` where ``index_mapping`` maps each output
        point back to its original index.
    """
    # Resolve defaults (None = skip step, _UNSET = use module defaults)
    if downsample_kwargs is _UNSET:
        downsample_kwargs = _DEFAULT_DOWNSAMPLE_KWARGS
    if outlier_kwargs is _UNSET:
        outlier_kwargs = _DEFAULT_OUTLIER_KWARGS
    if distance_kwargs is _UNSET:
        distance_kwargs = _DEFAULT_DISTANCE_KWARGS

    n_start = len(pcd.points)
    indices = np.arange(n_start)
    logger.debug("clean_pointcloud: start %d points", n_start)

    if downsample_kwargs is not None and len(pcd.points) > 0:
        n_before = len(pcd.points)
        kwargs = {**_DEFAULT_DOWNSAMPLE_KWARGS, **downsample_kwargs}
        pcd, idx_map = voxel_downsample(pcd, **kwargs)
        indices = indices[idx_map]
        logger.debug("downsample: %d → %d (%d removed)", n_before, len(pcd.points), n_before - len(pcd.points))

    if outlier_kwargs is not None and len(pcd.points) > 0:
        n_before = len(pcd.points)
        kwargs = {**_DEFAULT_OUTLIER_KWARGS, **outlier_kwargs}
        pcd, ind = pcd.remove_statistical_outlier(**kwargs)
        indices = indices[ind]
        logger.debug("outlier_removal: %d → %d (%d removed)", n_before, len(pcd.points), n_before - len(pcd.points))

    if distance_kwargs is not None and len(pcd.points) > 0:
        n_before = len(pcd.points)
        kwargs = {**_DEFAULT_DISTANCE_KWARGS, **distance_kwargs}
        pcd, mask = filter_distance(pcd, return_mask=True, **kwargs)
        indices = indices[mask]
        logger.debug("distance_removal: %d → %d (%d removed)", n_before, len(pcd.points), n_before - len(pcd.points))

    logger.debug(
        "clean_pointcloud: done %d → %d (%d total removed)",
        n_start,
        len(pcd.points),
        n_start - len(pcd.points),
    )
    return pcd, indices


def voxel_downsample(
    pcd: Any,
    voxel_size: float = 0.015,
    radius: float = 0.05,
    adaptive: bool = True,
) -> tuple[Any, np.ndarray]:
    """Voxel-downsample an Open3D point cloud, returning the cloud and index mapping.

    Args:
        pcd:        Open3D PointCloud to downsample.
        voxel_size: Base voxel size in metres (used when ``adaptive=False``).
        radius:     Radius for adaptive density estimation (used when ``adaptive=True``).
        adaptive:   When True, scale voxel_size by local point density.

    Returns:
        (downsampled_pcd, index_mapping) where index_mapping maps each voxel
        representative back to its original index.
    """
    import open3d as o3d

    points = np.asarray(pcd.points)

    # Compute adaptive voxel size scaled to point density
    if adaptive and len(points) > 0:
        # Estimate local density and adapt voxel size
        tree = o3d.geometry.KDTreeFlann(pcd)
        sample_size = min(500, len(points))
        densities = []
        for i in range(sample_size):
            [k, _, _] = tree.search_radius_vector_3d(points[i], radius * 2)
            densities.append(k)
        avg_density = float(np.mean(densities))
        effective_voxel_size = voxel_size * max(0.5, min(2.0, 50.0 / max(1e-6, avg_density)))
    else:
        effective_voxel_size = voxel_size

    min_bound = points.min(axis=0) if len(points) > 0 else np.zeros(3)
    max_bound = points.max(axis=0) if len(points) > 0 else np.zeros(3)

    # Voxel downsample with trace to preserve original point indices
    downsampled, trace_indices, _ = pcd.voxel_down_sample_and_trace(
        voxel_size=effective_voxel_size,
        min_bound=min_bound,
        max_bound=max_bound,
        approximate_class=False,
    )

    index_mapping = np.array(
        [inds[inds >= 0][0] if np.any(inds >= 0) else -1 for inds in trace_indices],
        dtype=np.int64,
    )

    return downsampled, index_mapping


########################################################
########## Legacy cleaning utilities ##################
########################################################


def clean_pcd(
    pcd,
    voxel_size: float = 0.015,
    radius: float = 0.05,
    max_distance: float = 1.0,
    downsample: bool = True,
    outlier_removal: bool = True,
    distance_removal: bool = True,
    reference: str = "centroid",
):
    """
    Enhanced cleaning with opacity and scale-based filtering.
    """
    import open3d as o3d

    indices = np.arange(len(pcd.points))

    # 3. Adaptive voxel downsampling based on point density
    if downsample:
        # Calculate local density to adapt voxel size
        points = np.asarray(pcd.points)
        if len(points) > 10000:  # For large point clouds, use adaptive voxel size
            tree = o3d.geometry.KDTreeFlann(pcd)
            densities = []
            for i in range(min(1000, len(points))):  # Sample subset for density estimation
                [k, idx, _] = tree.search_radius_vector_3d(points[i], radius * 2)
                densities.append(k)
            avg_density = float(np.mean(densities))
            adaptive_voxel_size = voxel_size * max(0.5, min(2.0, 50.0 / max(1e-6, avg_density)))
        else:
            adaptive_voxel_size = voxel_size

        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)

        logger.debug("voxel_downsample: adaptive size %.4f", adaptive_voxel_size)

        pcd, trace_indices, _ = pcd.voxel_down_sample_and_trace(
            voxel_size=adaptive_voxel_size,
            min_bound=min_bound,
            max_bound=max_bound,
            approximate_class=False,
        )

        voxel_indices = np.array([inds[inds >= 0][0] if np.any(inds >= 0) else -1 for inds in trace_indices])
        valid_mask = voxel_indices >= 0
        voxel_indices = voxel_indices[valid_mask]

        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[valid_mask])
        indices = indices[voxel_indices]

    # 4. Statistical outlier removal (more robust than radius-based)
    if outlier_removal:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        indices = indices[ind]
        logger.debug("clean_pcd: removed %d statistical outliers", len(indices) - len(ind))

    # 5. Distance-based removal
    if distance_removal:
        pcd, mask = remove_far_points(pcd, max_distance=max_distance, reference=reference, return_mask=True)
        indices = indices[mask]

    logger.debug("clean_pcd: %d points after cleaning", len(indices))
    return pcd, indices


def remove_far_points(
    pcd,
    max_distance: Optional[float] = None,
    n_points: Optional[int] = None,
    reference: str = "centroid",
    return_mask: bool = False,
):
    """
    Removes farthest points from a point cloud based on either a distance threshold
    or by keeping a fixed number of closest points to a reference point.

    Returns:
        - Point cloud with filtered points
        - (optional) Boolean mask of selected points
    """
    import open3d as o3d

    if max_distance is None and n_points is None:
        raise ValueError("You must specify either `max_distance` or `n_points`.")

    points = np.asarray(pcd.points)

    # Reference point
    if reference == "centroid":
        ref_point = np.mean(points, axis=0)
    elif reference == "origin":
        ref_point = np.zeros(3)
    else:
        raise ValueError("reference must be 'origin' or 'centroid'")

    distances = np.linalg.norm(points - ref_point, axis=1)

    if max_distance is not None:
        mask = distances <= max_distance
    else:
        if n_points is not None and n_points > len(points):
            raise ValueError("n_points is greater than the number of points in the cloud.")
        sorted_indices = np.argsort(distances)
        mask = np.zeros_like(distances, dtype=bool)
        if n_points is None:
            n_points = len(points)
        mask[sorted_indices[:n_points]] = True

    filtered_points = points[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    if pcd.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    return (filtered_pcd, mask) if return_mask else filtered_pcd


def density_filter(pcd, radius=0.03, percentile=10):
    """
    Remove points in sparse regions using local density.
    """
    import open3d as o3d

    # Find points in sparse regions using local density
    logger.debug("density_filter: estimating densities")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = []

    for i in trange(len(pcd.points), desc="Estimating point densities"):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius=radius)
        densities.append(k)

    densities = np.array(densities)
    logger.debug("density_filter: min=%d max=%d mean=%.1f", np.min(densities), np.max(densities), np.mean(densities))

    # Remove points in very sparse regions (bottom 10% by density)
    density_threshold = np.percentile(densities, percentile)  # Adjust percentage as needed
    dense_mask = densities >= density_threshold
    pcd_dense = pcd.select_by_index(np.where(dense_mask)[0])
    logger.debug(
        "density_filter: removed %d sparse points (threshold %.3f)",
        len(pcd.points) - len(pcd_dense.points),
        density_threshold,
    )


########################################################
########## Geometry: OBB + mask lifting ################
########################################################


def fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit dominant plane via RANSAC; return (R_3x3, t_3) aligning plane to Z-up.

    Uses Open3D's segment_plane on the full point cloud. No heuristic percentile —
    the dominant plane (largest inlier set) is taken as the floor.

    Args:
        points: (N, 3) float32 or float64 point cloud.
    Returns:
        R: (3, 3) rotation matrix aligning floor normal to [0, 0, 1].
        t: (3,) translation placing floor at z=0 after rotation is applied.
    """
    import open3d as o3d  # optional heavy dep

    from collab_splats.geometry.transforms import rotation_align_vectors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    n_mag = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / n_mag
    d_norm = d / n_mag  # plane: normal · x + d_norm = 0; floor at z = -d_norm after rotation

    # Ensure normal points upward (positive Z component after alignment)
    if normal[2] < 0:
        normal = -normal
        d_norm = -d_norm

    R = rotation_align_vectors(normal, np.array([0.0, 0.0, 1.0]))
    # After R, floor is at z = -d_norm. Translate by d_norm to bring to z = 0.
    t = np.array([0.0, 0.0, d_norm])
    return R.astype(np.float64), t.astype(np.float64)


def compute_obb_from_points(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute oriented bounding box for a Nx3 point cloud via PCA.

    Returns:
        center   : (3,) world-space OBB center
        extent   : (3,) box side lengths along principal axes
        rotation : (3,3) rotation matrix, columns = principal axes
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be Nx3"

    # Strip NaN/Inf rows before any computation
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    # Compute centroid and center the cloud
    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA via covariance matrix eigenvectors
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort axes by descending variance
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    rotation = eigvecs

    # Project points into PCA frame and compute axis-aligned extents
    points_local = centered @ rotation
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner

    # Map local center back to world space
    center_local = 0.5 * (min_corner + max_corner)
    center = centroid + center_local @ rotation.T

    return center, extent, rotation


def get_points_in_mask(
    frame_idx: int,
    mask: np.ndarray,
    points: np.ndarray,
    pixel_indices: np.ndarray,
) -> np.ndarray:
    """Return world-space points whose source pixel falls within a 2D mask.

    Args:
        frame_idx:     Frame to query.
        mask:          (H, W) bool array — True for pixels of interest.
        points:        (P, 3) float32 world-space point positions.
        pixel_indices: (P, 3) int32 [frame_id, row, col] per point.

    Returns:
        (M, 3) float32 — subset of points with source pixel inside mask, M <= P.
    """
    # Select only points that belong to the requested frame
    frame_mask = pixel_indices[:, 0] == frame_idx
    rows = pixel_indices[frame_mask, 1]
    cols = pixel_indices[frame_mask, 2]

    # Index mask at each point's pixel location
    in_mask = mask[rows, cols]
    return points[frame_mask][in_mask]


def voxel_downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_fraction: float = 0.01,
    voxel_size: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud with scene-adaptive or explicit voxel size (pure numpy).

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
    """
    if len(points) == 0:
        return points, colors

    if voxel_size is not None:
        # Use explicit voxel size
        if verbose:
            logger.debug("voxel_downsample_point_cloud: explicit size %.4f m", voxel_size)
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
            logger.debug(
                "voxel_downsample_point_cloud: scene extent IQR=%.3f m, full=%.3f m", scene_extent, full_extent
            )
            logger.debug("voxel_downsample_point_cloud: adaptive size %.4f m", voxel_size)

    # Pure numpy voxel downsampling
    points_float = points.astype(np.float64)

    # Compute voxel grid coordinates
    voxel_coords = np.floor(points_float / voxel_size).astype(np.int32)

    # Find unique voxels and get one representative point per voxel
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)

    # Extract downsampled points and colors using the indices
    downsampled_points = points[unique_indices]
    downsampled_colors = colors[unique_indices]

    if verbose:
        logger.debug("voxel_downsample_point_cloud: %d → %d points", len(points), len(downsampled_points))

    return downsampled_points, downsampled_colors


########################################################
########## Feature projection ##########################
########################################################


def _grid_sample_at_pixels(
    fmap: torch.Tensor,
    rows: np.ndarray,
    cols: np.ndarray,
    image_size: "tuple[int, int]",
) -> torch.Tensor:
    """Bilinear-sample fmap (D, H_p, W_p) at the given (rows, cols) in image_size frame.

    Returns (P_i, D) float32 tensor on CPU. Normalises pixel centres to [-1, 1]
    using align_corners=False convention so the same coords work for any (H_p, W_p).
    """
    H, W = image_size
    gx = torch.from_numpy(((2 * cols.astype(np.float32) + 1) / W - 1)).float()
    gy = torch.from_numpy(((2 * rows.astype(np.float32) + 1) / H - 1)).float()
    # grid_sample wants (N, H_out, W_out, 2); treat P_i points as (1, 1, P_i, 2)
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        fmap.unsqueeze(0).float(),
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    # (1, D, 1, P_i) → (P_i, D)
    return sampled.squeeze(0).squeeze(1).T.cpu()


def _sample_at_source_pixels(
    feature_maps: "list[torch.Tensor]",
    pixel_indices: np.ndarray,
    image_size: "tuple[int, int]",
) -> torch.Tensor:
    """Source-frame-only lift: each point sampled at its (frame_id, row, col) only.

    Used as a fallback for points that have zero accumulated weight in multi-view
    aggregation (never visible / always depth-inconsistent).
    """
    P = len(pixel_indices)
    D = feature_maps[0].shape[0]
    H, W = image_size
    out = torch.zeros((P, D), dtype=torch.float32)
    for i, fmap in enumerate(feature_maps):
        mask_i = pixel_indices[:, 0] == i
        if not mask_i.any():
            continue
        # Clip to model frame to handle any rounding drift from upstream
        rows = np.clip(pixel_indices[mask_i, 1], 0, H - 1)
        cols = np.clip(pixel_indices[mask_i, 2], 0, W - 1)
        out[mask_i] = _grid_sample_at_pixels(fmap, rows, cols, image_size)
    return out


def lift_features(
    feature_maps: "list[torch.Tensor]",
    result: "FeedforwardResult",
    *,
    depth_tol: float = 0.05,
) -> torch.Tensor:
    """Multi-view confidence-weighted lift of dense feature maps to per-point features.

    For each 3D point in result.points, projects into every frame, masks by
    in-bounds + depth-consistency (|z_proj - depth| / |z_proj| < depth_tol),
    weights by confidence at the projected pixel, and returns the weighted-mean
    feature. Points never visible in any frame fall back to a source-frame
    sample at their pixel_indices entry.

    Args:
        feature_maps: List of (D, H_p, W_p) per-frame dense features. Caller runs
            the extractor (and optional AE encode) before calling.
        result:       FeedforwardResult with points, pixel_indices, depth, confidence,
            extrinsics, intrinsics, model_height, model_width populated.
            Reload zarr with load_images=True before re-extracting features so
            the extractor sees the same FOV as the depth map.
        depth_tol:    Relative depth tolerance for visibility test.

    Returns:
        (P, D) float32 tensor of per-point features, aligned with result.points.
    """
    # Required fields — fail loud at function entry, not deep in the kernel
    for name in ("points", "pixel_indices", "depth", "confidence", "extrinsics", "intrinsics"):
        assert getattr(result, name) is not None, (
            f"lift_features requires result.{name}; " f"load zarr with load_images=True or run pipeline fresh"
        )
    N = result.extrinsics.shape[0]
    assert len(feature_maps) == N, f"feature_maps count ({len(feature_maps)}) != frame count ({N})"

    H, W = result.model_height, result.model_width
    P = result.points.shape[0]
    D = feature_maps[0].shape[0]
    image_size = (H, W)

    # Run the whole kernel on the GPU in float32 (was CPU float64 numpy + CPU grid_sample —
    # the dominant cost on large clouds). Everything moves to the device once; one .cpu() at the end.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pts = torch.as_tensor(np.ascontiguousarray(result.points), dtype=torch.float32, device=device)
    pts_h = torch.cat([pts, torch.ones((P, 1), dtype=torch.float32, device=device)], dim=1)  # (P, 4)

    # Ensure extrinsics are (N, 4, 4); accept (N, 3, 4) by padding
    ext_np = result.extrinsics
    if ext_np.shape[-2:] == (3, 4):
        ext_np = extrinsics_to_homogeneous(ext_np)
    ext = torch.as_tensor(np.ascontiguousarray(ext_np), dtype=torch.float32, device=device)  # (N, 4, 4)
    intr = torch.as_tensor(np.ascontiguousarray(result.intrinsics), dtype=torch.float32, device=device)  # (N, 3, 3)

    # Conf / depth → (N, H, W) float32 on device
    conf_np = result.confidence.detach().cpu().numpy() if isinstance(result.confidence, torch.Tensor) else result.confidence
    conf = torch.as_tensor(np.ascontiguousarray(conf_np), dtype=torch.float32, device=device)
    depth_np = result.depth
    if depth_np.ndim == 4:
        depth_np = depth_np[..., 0]
    depth = torch.as_tensor(np.ascontiguousarray(depth_np), dtype=torch.float32, device=device)

    features_sum = torch.zeros((P, D), dtype=torch.float32, device=device)
    weights_sum = torch.zeros((P,), dtype=torch.float32, device=device)

    pts_hT = pts_h.T  # (4, P) — reused every frame
    for i in range(N):
        fmap = feature_maps[i].to(device=device, dtype=torch.float32)  # (D, H_p, W_p)
        # Project all P points into frame i: world -> cam -> pixel
        proj = intr[i] @ (ext[i] @ pts_hT)[:3]  # (3, P)
        z = proj[2]
        safe_z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
        u = proj[0] / safe_z
        v = proj[1] / safe_z

        # Visibility: in-bounds, in-front-of-camera, depth-consistent. nan/inf coords (degenerate
        # points) → 0 for indexing/sampling; in_bounds is False there so they contribute weight 0.
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        u_safe = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        u_idx = u_safe.clamp(0, W - 1).long()
        v_idx = v_safe.clamp(0, H - 1).long()
        z_depth = depth[i, v_idx, u_idx]
        depth_ok = (z - z_depth).abs() / (z.abs() + 1e-8) < depth_tol
        w = conf[i, v_idx, u_idx] * (in_bounds & depth_ok).float()  # (P,)

        # Bilinear sample feature map at projected coords (align_corners=False, border pad)
        gx = (2 * u_safe + 1) / W - 1
        gy = (2 * v_safe + 1) / H - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, 1, P, 2)
        sampled = F.grid_sample(fmap.unsqueeze(0), grid, mode="bilinear", align_corners=False, padding_mode="border")
        sampled = sampled.squeeze(0).squeeze(1).T  # (P, D)

        features_sum += sampled * w.unsqueeze(-1)
        weights_sum += w

    # Weighted mean with eps for numerical safety
    features = features_sum / (weights_sum.unsqueeze(-1) + 1e-8)

    # Fallback: points with zero accumulated weight → source-frame sample
    zero_w = weights_sum < 1e-6
    if bool(zero_w.any()):
        zero_idx = zero_w.detach().cpu().numpy()
        fallback = _sample_at_source_pixels(feature_maps, result.pixel_indices[zero_idx], image_size)
        features[zero_w] = fallback.to(device)

    return features.detach().cpu()


def reproject_pixels(
    depth: np.ndarray,
    pixel_indices: np.ndarray,
    extrinsics_3x4: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Reproject points to world space using stored source pixels and (new) poses.

    Use this instead of re-running unproject_and_filter_points after BA — stored
    pixel_indices bypass the stochastic conf_mask subsampling so the point set
    stays aligned with pre-BA features and colors.

    Args:
        depth:          (N, H, W, 1) or (N, H, W) float32 depth maps.
        pixel_indices:  (P, 3) int32 — [frame_id, row, col] source pixel per point.
        extrinsics_3x4: (N, 3, 4) world-to-camera extrinsics (e.g. refined by BA).
        intrinsics:     (N, 3, 3) camera intrinsics.

    Returns:
        (P, 3) float32 world-space point positions.
    """
    fi = pixel_indices[:, 0]  # frame index per point  (P,)
    ri = pixel_indices[:, 1]  # pixel row per point    (P,)
    ci = pixel_indices[:, 2]  # pixel col per point    (P,)

    # depth at source pixel for each point
    if depth.ndim == 4:
        z = depth[fi, ri, ci, 0].astype(np.float64)
    else:
        z = depth[fi, ri, ci].astype(np.float64)

    # unproject source pixel to camera space with pinhole model
    fx = intrinsics[fi, 0, 0].astype(np.float64)
    fy = intrinsics[fi, 1, 1].astype(np.float64)
    cx = intrinsics[fi, 0, 2].astype(np.float64)
    cy = intrinsics[fi, 1, 2].astype(np.float64)
    x_cam = (ci - cx) * z / fx  # (P,)
    y_cam = (ri - cy) * z / fy  # (P,)

    # homogeneous camera-space coords: (P, 4)
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)

    # build (N, 4, 4) world-to-cam and invert to cam-to-world
    N = extrinsics_3x4.shape[0]
    w2c = np.zeros((N, 4, 4), dtype=np.float64)
    w2c[:, :3, :] = extrinsics_3x4
    w2c[:, 3, 3] = 1.0
    cam2world = invert_poses(w2c)  # (N, 4, 4)

    # per-point transform: cam2world[fi] @ pts_cam[p]
    pts_world = np.einsum("pij,pj->pi", cam2world[fi], pts_cam)  # (P, 4)

    return pts_world[:, :3].astype(np.float32)


########################################################
########## Cross-frame attention utilities #############
########################################################


def cross_frame_attention_ratio(
    k: torch.Tensor,
    q: torch.Tensor,
    token_offset: int = 5,
) -> float:
    """Cross-frame attention ratio between two frames' QKV tensors.

    Measures how much frame B's tokens attend to frame A relative to frame A's
    self-attention peak.  Port of VGGT-SPARK get_similarity().  Used to gate loop
    closure candidate acceptance — high ratio means the two frames share coherent
    overlapping geometry.

    Args:
        k:            (B, heads, N_tokens, head_dim) key projections.  N_tokens covers
                      both frames concatenated, so tokens_per_img = N_tokens // 2.
        q:            (B, heads, N_tokens, head_dim) query projections, same layout.
        token_offset: Skip the first N tokens per frame (camera + register tokens
                      that precede patch tokens in VGGT-style models).  Default 5.

    Returns:
        Scalar float in [0, ∞), mean of the top-25% normalised cross-frame attention
        values (mean_top_quarter aggregation, matching VGGT-SPARK get_similarity()).
        Values >= 0.85 match the VGGT-SPARK acceptance threshold calibrated on VGGT-1B.
        Returns 0.0 if token_offset >= tokens_per_img (no patch tokens to measure).
    """
    tokens_per_img = q.shape[2] // 2
    # Slice only the patch tokens from frame A (skip camera+register tokens)
    k_first = k[:, :, token_offset:tokens_per_img, :]
    if k_first.shape[2] == 0:
        return 0.0

    # Compute attention of all queries over first-frame patch keys
    attn = q @ k_first.transpose(-2, -1)  # (B, H, N_q, N_k_first)
    attn = attn.transpose(-2, -1)  # (B, H, N_k_first, N_q)
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)  # (B, N_k_first, N_q) — avg over heads

    # Split queries by destination frame to separate self- vs cross-frame attention
    attn_to_first = attn[..., :tokens_per_img]  # first-frame self-attention
    attn_to_second = attn[..., tokens_per_img:]  # cross-frame attention to second

    # Ratio: how much cross-frame attention relative to self-attention peak
    max_self = attn_to_first.max(dim=-1)[0]  # (B, N_k_first)
    normalized = attn_to_second / (max_self.unsqueeze(-1) + 1e-8)
    ratio = normalized.max(dim=1)[0]  # (B, N_second)

    # Aggregate: mean of top-25% values — matches VGGT-SPARK mean_top_quarter().
    # Previously used np.percentile(90) which gives a lower scalar and caused
    # VGGT-X scores (~0.74) to fall below the 0.85 threshold calibrated for VGGT-1B.
    ratio_np = ratio.cpu().float().numpy().ravel()
    if ratio_np.size == 0:
        return 0.0
    thresh = float(np.percentile(ratio_np, 75))
    top_vals = ratio_np[ratio_np >= thresh]
    return float(top_vals.mean())
