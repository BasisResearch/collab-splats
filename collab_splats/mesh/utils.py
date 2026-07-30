from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from collab_splats.mesh.base import MeshResult
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.geometry.transforms import invert_poses

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

    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=depth_im.dtype, device=depth_im.device)
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

    Spatial query runs on the CPU KDTree (multicore, O(N log M)); the Gaussian-weighted
    aggregation runs on the GPU in float32 via index_add_ (mirrors lift_features). Falls
    back to the CPU torch device when CUDA is unavailable.

    Returns np.ndarray (M, D), dtype matching input features.

    Args:
        mesh_vertices: (M, 3) array of mesh vertex positions
        points:        (N, 3) array of input point cloud
        features:      (N, D) array of per-point features
        k:             number of nearest neighbors used for weighting
        sdf_trunc:     truncation distance — points whose nearest vertex is farther are dropped
    """
    vertices = np.asarray(mesh_vertices)
    M = len(vertices)
    D = features.shape[1]

    # Nearest-vertex query for every point; workers=-1 uses all cores.
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k, workers=-1)

    # k=1 collapses the neighbour axis; restore it so the kernel below is uniform.
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Drop points whose closest vertex is beyond the truncation band.
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    feats = features[valid_mask]

    # Move the aggregation to the GPU (float32); one .cpu() at the end.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.as_tensor(np.ascontiguousarray(distances), dtype=torch.float32, device=device)
    idx = torch.as_tensor(np.ascontiguousarray(indices), dtype=torch.long, device=device)
    f = torch.as_tensor(np.ascontiguousarray(feats), dtype=torch.float32, device=device)

    # Gaussian kernel over neighbour distances; normalize weights per point (over k).
    sigma = d.mean()
    w = torch.exp(-(d**2) / (2 * sigma**2))
    w = w / w.sum(dim=1, keepdim=True)

    # Scatter weighted features to vertices; accumulate weights for normalization.
    acc = torch.zeros((M, D), dtype=torch.float32, device=device)
    wsum = torch.zeros((M, 1), dtype=torch.float32, device=device)
    for j in range(k):
        acc.index_add_(0, idx[:, j], f * w[:, j : j + 1])
        wsum.index_add_(0, idx[:, j], w[:, j : j + 1])

    # Normalize aggregated features by summed weights (skip untouched vertices -> stay zero).
    nz = wsum.squeeze(1) > 0
    acc[nz] /= wsum[nz]

    return acc.cpu().numpy().astype(features.dtype)


def transfer_features_to_mesh(
    result: FeedforwardResult,
    mesh: o3d.geometry.TriangleMesh,
    *,
    k: int = 5,
    sdf_trunc: float = 0.03,
) -> np.ndarray:
    """Transfer per-point features from a FeedforwardResult to mesh vertices via KNN.

    Args:
        result:    FeedforwardResult with features (P, D) and points (P, 3) populated.
        mesh:      Open3D TriangleMesh whose vertices receive the features.
        k:         Neighbors for Gaussian-weighted aggregation (passed to features2vertex).
        sdf_trunc: Truncation distance — pointcloud points farther than this from their
                   nearest vertex are excluded from aggregation.

    Returns:
        (M, D) ndarray of per-vertex features, dtype matches input features, index-aligned with mesh.vertices.
    """
    assert (
        result.features is not None
    ), "result.features is None — call lift_features() and assign result.features before transferring"
    return features2vertex(
        np.asarray(mesh.vertices),
        result.points,
        result.features,
        k=k,
        sdf_trunc=sdf_trunc,
    )


def persist_mesh_vertex_features(
    mesh_path: Path,
    points: np.ndarray,
    point_features: np.ndarray,
    *,
    k: int = 5,
    sdf_trunc: float = 0.03,
) -> np.ndarray:
    """Transfer point features to a written mesh's vertices and cache them as vertex_features.npy.

    Reads the mesh PLY at mesh_path, runs features2vertex against the supplied point
    features (already lifted/normalized by the caller), writes vertex_features.npy beside
    the mesh, and returns the (M, D) per-vertex array (index-aligned with mesh.vertices).
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    vertex_features = features2vertex(vertices, points, point_features, k=k, sdf_trunc=sdf_trunc)
    out_path = Path(mesh_path).parent / "vertex_features.npy"
    np.save(out_path, vertex_features)
    logger.info("saved mesh vertex features → %s  shape=%s", out_path, vertex_features.shape)
    return vertex_features


########################################################
############## Mesh cleaning / repair ##################
########################################################


def clean_repair_mesh(
    mesh_path: str | Path,
    max_hole_size: float = 3.0,
    max_edge_splits: int = 10000,
    use_largest: bool = False,  # if True, selects only the largest
) -> Path:
    """Drop stray components and fill small holes in a mesh on disk, rewriting it in place.

    Args:
        mesh_path: Mesh to clean. Overwritten with the result.
        max_hole_size: Fill holes whose perimeter is below this; larger ones are real openings
            (an unscanned wall, the open side of a room) and get left alone.
        max_edge_splits: Subdivision ceiling for each patch, so one huge hole cannot explode
            the triangle count.
        use_largest: Keep only the biggest component. Off by default — that also throws away
            legitimate detached geometry (furniture, objects) that sits inside the scene.
    Returns:
        The path written (same as mesh_path).
    """
    if not _MM_AVAILABLE:
        raise ImportError("meshlib is required for clean_repair_mesh. Install it with: pip install meshlib")

    mesh_path = Path(mesh_path)
    mesh = mm.loadMesh(str(mesh_path))

    # Split into connected components and seed the output with the biggest one — for a TSDF
    # scene that is the room itself; everything else is either scene content or noise.
    components = mm.getAllComponents(mesh)
    sizes = [mask.count() for mask in components]
    largest_idx = max(range(len(sizes)), key=lambda i: sizes[i])

    combined = mm.Mesh()
    combined.addMeshPart(mm.MeshPart(mesh, components[largest_idx]))

    n_removed = 0
    if not use_largest:
        # Keep every other component whose bounding box sits inside the main one, drop the rest.
        # That is the cheap separator between scene content and the floating specks TSDF leaves
        # outside the room from stray depth.
        idxs = [i for i in range(len(sizes)) if i != largest_idx]
        combined_bounds = combined.getBoundingBox()

        for idx in tqdm(idxs, desc="Finding components within bounds"):
            _temp = mm.Mesh()
            _temp.addMeshPart(mm.MeshPart(mesh, components[idx]))

            if combined_bounds.contains(_temp.getBoundingBox()):
                combined.addMeshPart(mm.MeshPart(mesh, components[idx]))
            else:
                n_removed += 1

    logger.info("Kept %d of %d components (removed %d)", len(sizes) - n_removed, len(sizes), n_removed)
    mesh = combined

    # Patch size follows the mesh's own resolution, so a fill matches the surface around it.
    # Native call, not a Python loop over edges: ~94s of pybind round-trips on a 4.7M-triangle
    # scene versus sub-millisecond, same value to 1e-8.
    avg_edge_length = mesh.averageEdgeLength()

    # Fill each small hole, then subdivide + smooth the new faces so the patch is not a flat cap.
    hole_ids = mesh.topology.findHoleRepresentiveEdges()
    fill_params = mm.FillHoleParams()
    n_filled = 0

    for he in tqdm(hole_ids, desc=f"Filling holes ({len(hole_ids)})"):
        perimeter = mesh.holePerimeter(he)
        if perimeter >= max_hole_size:
            logger.debug("Skipping hole %s of perimeter %s", he, perimeter)
            continue

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
        n_filled += 1

    logger.info("Filled %d of %d holes (max_hole_size=%s)", n_filled, len(hole_ids), max_hole_size)

    mm.saveMesh(mesh, str(mesh_path))
    return mesh_path


def align_geometry_floor(
    geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
    dist_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    num_sample_points: int = 10000,
) -> tuple[Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh], np.ndarray, np.ndarray]:
    """Align point cloud or triangle mesh to the floor plane.

    Uses fit_dominant_plane (single RANSAC call) to detect the dominant floor
    and compute rotation + translation, then applies both to the geometry.

    Args:
        geometry: Input geometry (PointCloud or TriangleMesh).
        dist_threshold: Kept for backward compatibility; not passed to fit_dominant_plane.
        ransac_n: Kept for backward compatibility; not passed to fit_dominant_plane.
        num_iterations: Kept for backward compatibility; not passed to fit_dominant_plane.
        num_sample_points: Surface sample count for mesh inputs only.
    Returns:
        Tuple of (aligned_geometry, R (3,3), t (3,)).
    """
    from collab_splats.pointcloud.utils import fit_dominant_plane

    # Sample points from mesh surface; use points directly for point clouds
    is_mesh = isinstance(geometry, o3d.geometry.TriangleMesh)
    if is_mesh:
        sample_pcd = geometry.sample_points_uniformly(number_of_points=num_sample_points)
        pts = np.asarray(sample_pcd.points)
    else:
        pts = np.asarray(geometry.points)

    # Fit dominant floor plane and get rotation + translation in one call
    R, t = fit_dominant_plane(pts)
    geometry.rotate(R, center=(0, 0, 0))
    geometry.translate(t)
    return geometry, R, t


########################################################
############## Mesh Clustering Utils ###################
########################################################


def mesh_clustering(mesh, similarity_values, similarity_threshold=0.8, spatial_radius=0.03):
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
        [_, neighbors, _] = kdtree.search_radius_vector_3d(vertices[orig_idx], spatial_radius)

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

    world_points = result.world_points  # (N, H, W, 3)
    N, H, W, _ = world_points.shape

    depths = np.empty((N, H, W), dtype=np.float32)
    for i in range(N):
        R = result.extrinsics[i, :3, :3]  # (3, 3) world-to-cam rotation
        t = result.extrinsics[i, :3, 3]  # (3,) world-to-cam translation
        cam_pts = world_points[i] @ R.T + t  # (H, W, 3)
        depths[i] = cam_pts[..., 2].clip(0)  # Z >= 0; negatives are boundary artefacts

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
