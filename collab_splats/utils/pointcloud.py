import open3d as o3d
import numpy as np
from typing import Optional, Union, Tuple
from tqdm import trange


def clean_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.015,
    radius: float = 0.05,
    max_distance: float = 1.0,
    downsample: bool = True,
    outlier_removal: bool = True,
    distance_removal: bool = True,
    reference: str = "centroid",
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Enhanced cleaning with opacity and scale-based filtering.
    """

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
    pcd: o3d.geometry.PointCloud,
    max_distance: Optional[float] = None,
    n_points: Optional[int] = None,
    reference: str = "centroid",
    return_mask: bool = False,
) -> Union[o3d.geometry.PointCloud, Tuple[o3d.geometry.PointCloud, np.ndarray]]:
    """
    Removes farthest points from a point cloud based on either a distance threshold
    or by keeping a fixed number of closest points to a reference point.

    Returns:
        - Point cloud with filtered points
        - (optional) Boolean mask of selected points
    """
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
