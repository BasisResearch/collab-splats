# collab_splats/pointcloud/utils.py
"""Geometric pointcloud utilities. Moved from collab_splats/utils/pointcloud.py."""
from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple
from tqdm import trange


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
            for i in range(
                min(1000, len(points))
            ):  # Sample subset for density estimation
                [k, idx, _] = tree.search_radius_vector_3d(points[i], radius * 2)
                densities.append(k)
            avg_density = float(np.mean(densities))
            adaptive_voxel_size = voxel_size * max(
                0.5, min(2.0, 50.0 / max(1e-6, avg_density))
            )
        else:
            adaptive_voxel_size = voxel_size

        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)

        print(f"Using adaptive voxel size: {adaptive_voxel_size}")

        pcd, trace_indices, _ = pcd.voxel_down_sample_and_trace(
            voxel_size=adaptive_voxel_size,
            min_bound=min_bound,
            max_bound=max_bound,
            approximate_class=False,
        )

        voxel_indices = np.array(
            [inds[inds >= 0][0] if np.any(inds >= 0) else -1 for inds in trace_indices]
        )
        valid_mask = voxel_indices >= 0
        voxel_indices = voxel_indices[valid_mask]

        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[valid_mask])
        indices = indices[voxel_indices]

    # 4. Statistical outlier removal (more robust than radius-based)
    if outlier_removal:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        indices = indices[ind]
        print(f"Removed {len(indices) - len(ind)} statistical outliers")

    # 5. Distance-based removal
    if distance_removal:
        pcd, mask = remove_far_points(
            pcd, max_distance=max_distance, reference=reference, return_mask=True
        )
        indices = indices[mask]

    print(f"Point cloud has {len(indices)} points after enhanced cleaning")
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
            raise ValueError(
                "n_points is greater than the number of points in the cloud."
            )
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
    print("Finding sparse regions...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = []

    for i in trange(len(pcd.points), desc="Estimating point densities"):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius=radius)
        densities.append(k)

    densities = np.array(densities)
    print(
        f"Density stats - Min: {np.min(densities)}, Max: {np.max(densities)}, Mean: {np.mean(densities):.1f}"
    )

    # Remove points in very sparse regions (bottom 10% by density)
    density_threshold = np.percentile(
        densities, percentile
    )  # Adjust percentage as needed
    dense_mask = densities >= density_threshold
    pcd_dense = pcd.select_by_index(np.where(dense_mask)[0])
    print(
        f"Removed {len(pcd.points) - len(pcd_dense.points)} sparse points (threshold: {density_threshold})"
    )

    return pcd_dense


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
        print(f"Spatial filtering ({pmin}-{pmax} percentile):")
        print(f"  Bounding box extent: {extent:.3f}m")
        print(f"  Filtered from {len(points)} to {len(filtered_points)} points")
        removed_count = len(points) - len(filtered_points)
        removed_pct = 100 * (1 - len(filtered_points)/len(points)) if len(points) > 0 else 0
        print(f"  Removed {removed_count} outliers ({removed_pct:.1f}%)")

    return filtered_points, filtered_colors


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
            print(f"  Using explicit voxel size: {voxel_size:.4f}m")
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
            print(f"  Scene extent (IQR-based): {scene_extent:.3f}m, full extent: {full_extent:.3f}m")
            print(f"  Adaptive voxel size: {voxel_size:.4f}m")

    # Pure numpy voxel downsampling
    points_float = points.astype(np.float64)

    # Compute voxel grid coordinates
    voxel_coords = np.floor(points_float / voxel_size).astype(np.int32)

    # Create unique voxel identifiers
    # Use a large prime to hash coordinates
    voxel_ids = (
        voxel_coords[:, 0] * 73856093 ^
        voxel_coords[:, 1] * 19349663 ^
        voxel_coords[:, 2] * 83492791
    )

    # Find unique voxels and get one representative point per voxel
    unique_voxels, unique_indices = np.unique(voxel_ids, return_index=True)

    # Extract downsampled points and colors using the indices
    downsampled_points = points[unique_indices]
    downsampled_colors = colors[unique_indices]

    if verbose:
        print(f"  Downsampled from {len(points)} to {len(downsampled_points)} points")

    return downsampled_points, downsampled_colors
