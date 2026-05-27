from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm, trange

from collab_splats.mesh.base import MeshResult
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.utils.geometry import invert_poses

try:
    import meshlib.mrmeshpy as mm
    _MM_AVAILABLE = True
except ImportError:
    mm = None
    _MM_AVAILABLE = False

logger = logging.getLogger(__name__)


def pick_indices_at_random(valid_mask, samples_per_frame):
    indices = torch.nonzero(torch.ravel(valid_mask))
    if samples_per_frame < len(indices):
        which = torch.randperm(len(indices))[:samples_per_frame]
        indices = indices[which]
    return torch.ravel(indices)


def find_depth_edges(depth_im, threshold=0.01, dilation_itr=3):
    # Accept numpy arrays — convert to tensor
    if isinstance(depth_im, np.ndarray):
        depth_im = torch.from_numpy(depth_im)

    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=depth_im.dtype, device=depth_im.device
    )
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
    depth_laplacian = (
        F.conv2d(
            (1.0 / (depth_im + 1e-6)).unsqueeze(0).unsqueeze(0).squeeze(-1),
            laplacian_kernel,
            padding=1,
        )
        .squeeze(0)
        .squeeze(0)
        .unsqueeze(-1)
    )

    edges = (depth_laplacian > threshold) * 1.0
    structure_el = laplacian_kernel * 0.0 + 1.0

    dilated_edges = edges
    for i in range(dilation_itr):
        dilated_edges = (
            F.conv2d(
                dilated_edges.unsqueeze(0).unsqueeze(0).squeeze(-1),
                structure_el,
                padding=1,
            )
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(-1)
        )
    dilated_edges = (dilated_edges > 0.0) * 1.0

    # Return as numpy bool array (H, W)
    result = dilated_edges.squeeze(-1).detach().cpu().numpy().astype(bool)
    return result


########################################################
########## Feature Aggregation Utils ###################
########################################################


def normals2vertex(mesh_vertices, points, normals, k=5, sdf_trunc=0.03):
    """
    Map point cloud normals to mesh vertices using KNN over a KDTree.
    Same as features2vertex but with normalization for unit vectors.
    """
    mesh_normals = features2vertex(mesh_vertices, points, normals, k, sdf_trunc)

    # Normalize to unit vectors (critical for normals!)
    norms = np.linalg.norm(mesh_normals, axis=1, keepdims=True)
    mesh_normals = mesh_normals / (norms + 1e-8)  # avoid division by zero

    return mesh_normals


def features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """
    Map point cloud features to mesh vertices using KNN over a KDTree.

    Returns np.ndarray (unlike original utils/mesh.py which returned torch.Tensor)

    Args:
        mesh_vertices: (M, 3) array of mesh vertex positions
        points: (N, 3) array of input point cloud
        features: (N, D) array of per-point features
        k: number of nearest neighbors to use for weighting
        sdf_trunc: truncation distance for SDF

    Returns:
        features_kNN: (M, D) array of per-vertex features
    """

    vertices = np.asarray(mesh_vertices)

    # Build tree
    tree = cKDTree(vertices)

    # Query nearest vertex for each point
    distances, indices = tree.query(points, k=k)  # shape: (N,)

    # Mask points where nearest vertex is within truncation distance
    # Use distance to closest vertex (distance[:, 0]) for truncation mask
    valid_mask = distances[:, 0] <= sdf_trunc

    if not np.any(valid_mask):
        # No points within truncation distance, return zeros
        return np.zeros((len(vertices), features.shape[1]))

    # Filter distances, indices, and features by valid points
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    features = features[valid_mask]

    # Weighting with Gaussian kernel
    sigma = np.mean(distances)  # or set manually
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)  # normalize

    # Aggregate features per vertex
    features_kNN = np.zeros((len(vertices), features.shape[1]))

    # Use a counts array to normalize contributions per vertex later
    vertex_weight_sum = np.zeros((len(vertices), 1))

    # Accumulate weighted features
    for i in trange(k, desc="Mapping features to vertices"):
        vertex_indices = indices[:, i]
        weighted_feats = features * weights[:, i : i + 1]

        # Accumulate weighted features
        np.add.at(features_kNN, vertex_indices, weighted_feats)

        # Accumulate weights for normalization
        np.add.at(vertex_weight_sum, vertex_indices, weights[:, i : i + 1])

    # Normalize aggregated features by summed weights (avoid div by zero)
    nonzero_mask = vertex_weight_sum.squeeze() > 0
    features_kNN[nonzero_mask] /= vertex_weight_sum[nonzero_mask]

    return features_kNN


########################################################
############## Mesh cleaning / repair ##################
########################################################


def clean_repair_mesh(
    mesh_path: str,
    max_hole_size: float = 3.0,
    max_edge_splits: int = 10000,
    use_largest: bool = False,  # if True, selects only the largest
):
    if not _MM_AVAILABLE:
        raise ImportError("meshlib is required for clean_repair_mesh. Install it with: pip install meshlib")

    # Load mesh
    mesh = mm.loadMesh(mesh_path)

    # Identify all connected components
    components = mm.getAllComponents(mesh)

    # Determine component sizes
    sizes = [mask.count() for mask in components]

    # Always find largest cluster
    largest_idx = max(range(len(sizes)), key=lambda i: sizes[i])

    # Add the largest component
    combined = mm.Mesh()
    combined.addPartByMask(mesh, components[largest_idx])

    # Initialize n_removed before any branch
    n_removed = 0

    # Remove the largest component from list of idxs
    if not use_largest:
        idxs = list(range(len(sizes)))
        idxs.remove(largest_idx)

        # Add the remaining components if they fall within the bounds
        combined_bounds = combined.getBoundingBox()

        ## THIS IS REALLY HACKY AND INEFFIENCT CHANGE SOMETIME
        for idx in tqdm(idxs, desc="Finding components within bounds"):
            _temp = mm.Mesh()
            _temp.addPartByMask(mesh, components[idx])

            if combined_bounds.contains(_temp.getBoundingBox()):
                combined.addPartByMask(mesh, components[idx])
            else:
                n_removed += 1

    logger.info("Removed %d components", n_removed)
    mesh = combined

    # Compute average edge length
    avg_edge_length = 0.0
    num_edges = 0

    for i in trange(
        mesh.topology.undirectedEdgeSize(), desc="Calculating average edge length"
    ):
        dir_edge = mm.EdgeId(i * 2)
        org = mesh.topology.org(dir_edge)
        dest = mesh.topology.dest(dir_edge)
        avg_edge_length += (
            mesh.points.vec[dest.get()] - mesh.points.vec[org.get()]
        ).length()
        num_edges += 1
    avg_edge_length /= num_edges

    # Fill holes
    hole_ids = mesh.topology.findHoleRepresentiveEdges()
    fill_params = mm.FillHoleParams()

    for he in tqdm(hole_ids, desc=f"Filling holes ({len(hole_ids)})"):
        if mesh.holePerimiter(he) < max_hole_size:
            new_faces = mm.FaceBitSet()
            fill_params.outNewFaces = new_faces
            mm.fillHole(mesh, he, fill_params)

            new_verts = mm.VertBitSet()
            subdiv_settings = mm.SubdivideSettings()
            subdiv_settings.maxEdgeLen = avg_edge_length
            subdiv_settings.maxEdgeSplits = max_edge_splits
            subdiv_settings.region = new_faces
            subdiv_settings.newVerts = new_verts
            mm.subdivideMesh(mesh, subdiv_settings)
            mm.positionVertsSmoothly(mesh, new_verts)
        else:
            logger.debug("Skipping hole %s of perimeter %s", he, mesh.holePerimiter(he))

    return mesh


def align_geometry_floor(
    geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
    dist_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    num_sample_points: int = 10000,
) -> tuple[Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh], np.ndarray, np.ndarray]:
    """Align point cloud or triangle mesh to the floor plane.

    Args:
        geometry: Input geometry (PointCloud or TriangleMesh)
        dist_threshold: Distance threshold for RANSAC plane fitting
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        num_sample_points: Number of points to sample from mesh surface (only used for meshes)

    Returns:
        Tuple of:
            - aligned_geometry: Aligned geometry (same type as input)
            - R: Rotation matrix (3x3 np.ndarray) used to align the geometry
            - translation: Translation vector (3,) np.ndarray to translate geometry floor to z=0
    """
    # Determine input type and get point cloud for plane detection
    is_mesh = isinstance(geometry, o3d.geometry.TriangleMesh)

    if is_mesh:
        # For mesh: sample points from surface for robust plane detection
        pcd_for_plane_detection = geometry.sample_points_uniformly(
            number_of_points=num_sample_points
        )
    else:
        # For point cloud: use directly
        pcd_for_plane_detection = geometry

    # Find the floor plane
    floor = get_floor_plane(
        pcd_for_plane_detection,
        dist_threshold=dist_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = floor

    # Normalize the normal vector
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Ensure normal points upward
    if normal[2] < 0:
        normal = -normal
        d = -d  # Flip d when we flip the normal

    # Compute rotation to align the normal with Z-axis
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)
    rotation_angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))

    if np.linalg.norm(rotation_axis) < 1e-6:
        R = np.eye(3)
    else:
        rotation_axis /= np.linalg.norm(rotation_axis)
        axis_angle = rotation_axis * rotation_angle
        # Use appropriate method based on geometry type
        if is_mesh:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        else:
            R = geometry.get_rotation_matrix_from_axis_angle(axis_angle)

    # Apply rotation to the geometry
    geometry.rotate(R, center=(0, 0, 0))

    # Recompute floor plane after rotation
    if is_mesh:
        rotated_pcd_for_plane_detection = geometry.sample_points_uniformly(
            number_of_points=num_sample_points
        )
    else:
        rotated_pcd_for_plane_detection = geometry

    new_plane = get_floor_plane(
        rotated_pcd_for_plane_detection,
        dist_threshold=dist_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    _, _, _, d_new = new_plane

    # Translate the geometry so floor is at z=0
    translation = np.array([0, 0, -d_new])
    geometry.translate(translation)

    return geometry, R, translation


def get_floor_plane(
    pcd: o3d.geometry.PointCloud,
    dist_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
):
    """
    Get the floor plane from the point cloud.
    """
    plane_model, _ = pcd.segment_plane(
        distance_threshold=dist_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return plane_model


########################################################
############## Mesh Clustering Utils ###################
########################################################


def mesh_clustering(
    mesh, similarity_values, similarity_threshold=0.8, spatial_radius=0.03
):
    """
    Clusters the mesh into connected components based on similarity values.
    """

    vertices = np.asarray(mesh.vertices)

    # Pre-filter by similarity
    valid_mask = similarity_values > similarity_threshold
    valid_indices = np.where(valid_mask)[0]
    valid_vertices = vertices[valid_indices]

    if len(valid_vertices) == 0:
        return []

    # Build KDTree on all mesh vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Build adjacency matrix efficiently
    n_valid = len(valid_indices)
    valid_set = set(valid_indices)
    adjacency = np.zeros((n_valid, n_valid), dtype=bool)

    # Map original indices to compressed indices
    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}

    # For each valid vertex, find spatial neighbors that are also valid
    for i, orig_idx in tqdm(enumerate(valid_indices), desc="Building adjacency matrix"):
        [_, neighbors, _] = kdtree.search_radius_vector_3d(
            vertices[orig_idx], spatial_radius
        )

        for neighbor_idx in neighbors:
            if neighbor_idx in valid_set and neighbor_idx != orig_idx:
                j = idx_map[neighbor_idx]
                adjacency[i, j] = True

    adjacency_sparse = csr_matrix(adjacency)
    n_components, labels = connected_components(adjacency_sparse)

    # Convert back to original indices
    clusters = []
    for cluster_id in range(n_components):
        cluster_mask = labels == cluster_id
        cluster_vertices = valid_indices[cluster_mask]
        if len(cluster_vertices) > 10:  # Minimum size filter
            clusters.append(cluster_vertices.tolist())

    clusters = [np.asarray(c) for c in clusters]
    return clusters


########################################################
############## Feedforward → Mesh ######################
########################################################


def _feedforward_to_tsdf_inputs(
    result: FeedforwardResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world_points → per-frame Z depth, load RGBs, invert extrinsics → c2w.

    Raises:
        ValueError: if result.world_points is None.
    """
    if result.world_points is None:
        raise ValueError(
            "result.world_points is None. Access creator.outputs after reconstruct() — "
            "MapAnythingCreator, VGGTXCreator, and VGGTOmegaCreator all populate "
            "world_points during _postprocess()."
        )

    world_points = result.world_points          # (N, H, W, 3)
    N, H, W, _ = world_points.shape

    depths = np.empty((N, H, W), dtype=np.float32)
    for i in range(N):
        R = result.extrinsics[i, :3, :3]        # (3, 3) world-to-cam rotation
        t = result.extrinsics[i, :3, 3]         # (3,) world-to-cam translation
        cam_pts = world_points[i] @ R.T + t     # (H, W, 3)
        depths[i] = cam_pts[..., 2].clip(0)     # Z >= 0; negatives are boundary artefacts

    rgbs = np.empty((N, H, W, 3), dtype=np.float32)
    for i, path in enumerate(result.image_paths):
        img = PILImage.open(path).convert("RGB")
        if result.original_coords is not None:
            tl_x, tl_y, cr_x, cr_y = result.original_coords[i, :4]
            # VGGTOmega and VGGTXCreator crop-mode store original-image-pixel coords;
            # cr_x = orig_w > model_W signals a crop must be applied before resize.
            if cr_x > W + 1 or cr_y > H + 1:
                img = img.crop((float(tl_x), float(tl_y), float(cr_x), float(cr_y)))
        img = img.resize((W, H), PILImage.BILINEAR)
        rgbs[i] = np.asarray(img, dtype=np.float32) / 255.0

    c2w = invert_poses(result.extrinsics).astype(np.float32)

    return depths, rgbs, c2w, result.intrinsics.copy()


def pointcloud_to_mesh(
    result: FeedforwardResult,
    output_dir: Path,
    method: str = "open3d_tsdf",
    **mesher_kwargs,
) -> MeshResult:
    """Mesh directly from a FeedforwardResult using any registered mesh method.

    All mesh creators share the same interface: create(depths, rgbs, c2w, intrinsics).
    Only "open3d_tsdf" is implemented today; Poisson variants raise NotImplementedError
    from their subclass.

    Args:
        result:          creator.outputs after reconstruct(). Requires world_points populated.
        output_dir:      Directory to write mesh output.
        method:          Registry key — "open3d_tsdf", "depth_normal_poisson", "gaussians_poisson".
        **mesher_kwargs: Forwarded to the mesh creator constructor (voxel_size, sdf_trunc, etc.).

    Returns:
        MeshResult with mesh_path pointing to the output PLY.

    Raises:
        ValueError: if result.world_points is None or method is not in the registry.
    """
    from collab_splats.mesh import get_mesh_creator

    depths, rgbs, c2w, intrinsics = _feedforward_to_tsdf_inputs(result)
    mesher = get_mesh_creator(method, Path(output_dir), **mesher_kwargs)
    return mesher.create(depths, rgbs, c2w, intrinsics)
