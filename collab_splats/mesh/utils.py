from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import zarr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh.base import MeshResult
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import confidence_mask

try:
    import meshlib.mrmeshnumpy as mn
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
    max_edge_splits: int = 1_000_000,
    use_largest: bool = False,  # if True, selects only the largest
) -> Path:
    """Drop stray components and fill small holes in a mesh on disk, rewriting it in place.

    Memory-flat by construction: components come from open3d's native clustering (one int
    per face) and holes are filled in a single batched meshlib call — never one bitset or
    temp mesh per component/hole. The previous meshlib getAllComponents path allocated a
    dense per-component FaceBitSet (240k components x 7.1M faces ≈ 214 GB on a TSDF scene)
    and was OOM-killed.

    Args:
        mesh_path: Mesh to clean. Overwritten with the result.
        max_hole_size: Fill holes whose perimeter is below this; larger ones are real openings
            (an unscanned wall, the open side of a room) and get left alone.
        max_edge_splits: Global subdivision budget shared by all hole patches, so patch
            refinement cannot explode the triangle count.
        use_largest: Keep only the biggest component. Off by default — that also throws away
            legitimate detached geometry (furniture, objects) that sits inside the scene.
    Returns:
        The path written (same as mesh_path).
    """
    if not _MM_AVAILABLE:
        raise ImportError("meshlib is required for clean_repair_mesh. Install it with: pip install meshlib")

    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    has_colors = mesh.has_vertex_colors()

    # Connected components via native clustering: one component id per triangle. For a TSDF
    # scene the largest component is the room itself; everything else is scene content or noise.
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    n_comp = len(cluster_sizes)
    largest = int(cluster_sizes.argmax())

    if use_largest:
        keep = np.zeros(n_comp, dtype=bool)
    else:
        # Per-component AABBs in one vectorized pass, then the cheap separator between scene
        # content and the floating specks TSDF leaves outside the room from stray depth:
        # keep every component whose bounding box sits inside the main one.
        tri_pts = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        comp_min = np.full((n_comp, 3), np.inf)
        comp_max = np.full((n_comp, 3), -np.inf)
        np.minimum.at(comp_min, cluster_ids, tri_pts.min(axis=1))
        np.maximum.at(comp_max, cluster_ids, tri_pts.max(axis=1))
        keep = np.all(comp_min >= comp_min[largest], axis=1) & np.all(
            comp_max <= comp_max[largest], axis=1
        )
    keep[largest] = True
    mesh.remove_triangles_by_mask(~keep[cluster_ids])
    mesh.remove_unreferenced_vertices()
    logger.info(
        "Kept %d of %d components (removed %d)", int(keep.sum()), n_comp, n_comp - int(keep.sum())
    )

    # Hand off to meshlib for hole filling — via arrays, not disk: meshlib's PLY round-trip
    # drops vertex colors, so colors stay behind in numpy and are reattached after.
    faces = np.asarray(mesh.triangles).astype(np.int32)
    verts = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors) if has_colors else None
    mmesh = mn.meshFromFacesVerts(faces, verts)

    # Patch size follows the mesh's own resolution, so a fill matches the surface around it.
    avg_edge_length = mmesh.averageEdgeLength()

    # Perimeter gate in Python (cheap: ~2 s for 176k holes), then ONE native batch fill —
    # a per-hole fill/subdivide/smooth loop does not finish at TSDF hole counts.
    hole_ids = mmesh.topology.findHoleRepresentiveEdges()
    small = mm.std_vector_Id_EdgeTag()
    for he in tqdm(hole_ids, desc=f"Measuring holes ({len(hole_ids)})"):
        perimeter = mmesh.holePerimeter(he)
        if perimeter < max_hole_size:
            small.append(he)
        else:
            logger.debug("Skipping hole %s of perimeter %s", he, perimeter)

    new_faces = mm.FaceBitSet()
    fill_params = mm.FillHoleParams()
    fill_params.outNewFaces = new_faces
    mm.fillHoles(mmesh, small, fill_params)

    # One subdivide + smooth over every patch at once, so fills are not flat caps.
    new_verts = mm.VertBitSet()
    subdiv_settings = mm.SubdivideSettings()
    subdiv_settings.maxEdgeLen = avg_edge_length
    subdiv_settings.maxEdgeSplits = max_edge_splits
    subdiv_settings.region = new_faces
    subdiv_settings.newVerts = new_verts
    mm.subdivideMesh(mmesh, subdiv_settings)
    mm.positionVertsSmoothly(mmesh, new_verts)
    logger.info("Filled %d of %d holes (max_hole_size=%s)", len(small), len(hole_ids), max_hole_size)

    # Back to numpy. Original vertices keep their indices through fill/subdivide/pack, so
    # colors copy straight through and only patch vertices need a nearest-neighbour lookup;
    # if meshlib ever reorders, fall back to a full NN transfer.
    mmesh.pack()
    out_verts = mn.getNumpyVerts(mmesh).astype(np.float64)
    out_faces = mn.getNumpyFaces(mmesh.topology)

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(out_verts), o3d.utility.Vector3iVector(out_faces)
    )
    if has_colors:
        out_colors = np.empty((len(out_verts), 3))
        n_orig = len(verts)
        prefix_stable = len(out_verts) >= n_orig and np.allclose(
            out_verts[:n_orig], verts, atol=1e-5
        )
        if prefix_stable:
            out_colors[:n_orig] = colors
            new_idx = np.arange(n_orig, len(out_verts))
        else:
            new_idx = np.arange(len(out_verts))
        if len(new_idx):
            _, nn = cKDTree(verts).query(out_verts[new_idx], k=1)
            out_colors[new_idx] = colors[nn]
        out.vertex_colors = o3d.utility.Vector3dVector(out_colors)

    o3d.io.write_triangle_mesh(str(mesh_path), out)
    return mesh_path


########
# Guided depth upsampling — native-resolution TSDF fusion
########


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    """Normalized box filter, the O(1) primitive of the guided filter."""
    k = 2 * radius + 1
    return cv2.boxFilter(x, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. gray-guide guided filter: edge-preserving smoothing of src steered by guide."""
    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g
    cov_gs = _box(guide * src, radius) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


def guided_upsample_depth(
    depth: np.ndarray,
    rgb_full: np.ndarray,
    crop_box: tuple[int, int, int, int],
    out_hw: tuple[int, int],
    radius: int | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """Upsample a model-res depth map into its crop region of an original-res canvas.

    Nearest-neighbour resize (never fabricates depth), then a validity-weighted guided filter
    with the original-res RGB as guide snaps depth edges to image edges. Pixels that were 0
    (masked / no observation) in the source stay exactly 0. Canvas outside the crop box is 0.

    Args:
        depth:    (h, w) float32 model-res depth, 0 = no observation
        rgb_full: (H, W, 3) uint8 original-res frame (the guide)
        crop_box: (tl_x, tl_y, cr_x, cr_y) model crop in original pixels (original_coords[:4])
        out_hw:   (H, W) output canvas size (original_coords[4:6] reversed)
    """
    tl_x, tl_y, cr_x, cr_y = (int(round(v)) for v in crop_box)
    cw, ch = cr_x - tl_x, cr_y - tl_y
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Degenerate crop box {crop_box} — original_coords are corrupt")

    # Nearest resize of depth and validity to crop size — blocky but never invents values
    depth_nn = cv2.resize(depth, (cw, ch), interpolation=cv2.INTER_NEAREST)
    valid_nn = (depth_nn > 0).astype(np.float32)

    # Gray guide in [0, 1] from the original-res crop; radius spans ~2x the upsample factor
    guide = cv2.cvtColor(rgb_full[tl_y:cr_y, tl_x:cr_x], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if radius is None:
        radius = max(1, int(np.ceil(2 * cw / depth.shape[1])))

    # Validity-weighted filtering: masked pixels contribute nothing to their neighbours
    num = _guided_filter(guide, depth_nn * valid_nn, radius, eps)
    den = _guided_filter(guide, valid_nn, radius, eps)
    filtered = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)

    # The guide must never resurrect deleted depth
    filtered[valid_nn == 0] = 0.0
    # Guided filter can undershoot slightly; depth must stay non-negative
    np.maximum(filtered, 0.0, out=filtered)

    canvas = np.zeros(out_hw, dtype=np.float32)
    canvas[tl_y:cr_y, tl_x:cr_x] = filtered
    return canvas


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
    conf_percentile: float | None = None,
    frame_store=None,
    native_intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a FeedforwardResult into (depths, rgbs, c2w, intrinsics) for TSDF fusion.

    Default: everything at model resolution, mutually pixel-aligned, straight off the forward
    pass. `conf_percentile` zeroes depth below that confidence percentile (0 = no
    observation to Open3D); if the result has no confidence (e.g. an SfM-derived
    reconstruction), the gate is skipped and fusion proceeds unmasked. `frame_store` +
    `native_intrinsics` switch to native resolution: original-res uint8 RGB from frames.zarr,
    depth guided-upsampled into the model crop region (see guided_upsample_depth), and the
    caller's original-res K (the COLMAP camera).

    Raises:
        ValueError: depth/images missing; frame_store length != frame count;
                    frame_store without native_intrinsics.
    """
    # depth is the model's own output — all three backends populate it, so a fallback
    # derivation here would be dead code (and was measurably worse: see the design doc)
    if result.depth is None:
        raise ValueError(
            "result.depth is None — cannot mesh. Access creator.outputs after reconstruct(), "
            "or load the zarr with load_depth=True."
        )

    depths = np.ascontiguousarray(result.depth, dtype=np.float32)  # (N, H, W)

    # Confidence gate BEFORE any upsampling — never amplify pixels about to be deleted.
    # Same global-percentile rule as the pointcloud path (shared confidence_mask helper),
    # so the mesh inherits exactly the filter that makes the sparse cloud look clean.
    if conf_percentile is not None:
        if result.confidence is None:
            # SfM-derived results (e.g. instantsfm) carry no per-pixel confidence — fuse
            # unmasked rather than fail; the percentile gate only applies when the model
            # produced one.
            logger.info(
                "conf_percentile=%s set but result has no confidence — fusing unmasked",
                conf_percentile,
            )
        else:
            conf = result.confidence
            if hasattr(conf, "numpy"):
                conf = conf.detach().cpu().numpy()
            dropped = ~confidence_mask(conf, conf_percentile)
            depths = depths.copy()  # copy only when mutating — the default path fuses read-only
            depths[dropped] = 0.0
            logger.info(
                "Confidence mask (p%.0f): %.1f%% of depth pixels dropped",
                conf_percentile,
                100.0 * float(dropped.mean()),
            )

    c2w = invert_poses(result.extrinsics).astype(np.float32)

    # Native-resolution path: original-res RGB + guided-upsampled depth + caller's K
    if frame_store is not None:
        if native_intrinsics is None:
            raise ValueError("frame_store requires native_intrinsics (the original-res COLMAP K)")
        n = depths.shape[0]
        if len(frame_store) != n:
            raise ValueError(
                f"Frame-count mismatch: frames.zarr has {len(frame_store)} frames but the "
                f"reconstruction has {n} — they are from different runs."
            )
        rgbs = np.ascontiguousarray(frame_store.images())  # (N, H, W, 3) uint8
        out_hw = tuple(rgbs.shape[1:3])
        # frames.zarr must be the resolution the crop boxes were computed against — a
        # same-count store at a different res would silently misplace every crop.
        expected_hw = (int(result.original_coords[0, 5]), int(result.original_coords[0, 4]))
        if out_hw != expected_hw:
            raise ValueError(
                f"frames.zarr resolution {out_hw} != reconstruction original resolution "
                f"{expected_hw} (original_coords) — they are from different preprocessing runs."
            )
        native_depths = np.zeros((n, *out_hw), dtype=np.float32)
        for i in tqdm(range(n), desc="Upsampling depth to native resolution"):
            native_depths[i] = guided_upsample_depth(
                depths[i], rgbs[i], crop_box=tuple(result.original_coords[i, :4]), out_hw=out_hw
            )
        return native_depths, rgbs, c2w, native_intrinsics.copy()

    # Model-resolution path (default): images is (N, 3, H, W) in [0, 1]
    if result.images is None:
        raise ValueError(
            "result.images is None — cannot mesh. Access creator.outputs after reconstruct(), "
            "or load the zarr with load_images=True."
        )
    imgs = result.images
    if hasattr(imgs, "numpy"):
        imgs = imgs.detach().cpu().numpy()
    rgbs = np.ascontiguousarray(imgs.transpose(0, 2, 3, 1), dtype=np.float32)  # (N, H, W, 3)

    return depths, rgbs, c2w, result.intrinsics.copy()


def _splats_to_tsdf_inputs(
    splats_zarr: Path, conf_percentile: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack splats.zarr (rendered views) into (depths, rgbs, c2w, intrinsics) for TSDF fusion.

    - `alpha` is the confidence: pixels with alpha == 0 never reach Open3D, and
      `conf_percentile` drops the lowest-alpha percentile globally (same rule as the
      feedforward path's confidence gate).
    - Poses are the zarr's `c2w` — what was actually rendered, including pose-opt deltas.
    - Already native to the training frames, so there is no upsampling path.
    """
    # Loud failure before opening: the splats stage is never auto-run by mesh()
    splats_zarr = Path(splats_zarr)
    if not splats_zarr.exists():
        raise FileNotFoundError(f"{splats_zarr} — run the splats stage first")
    store = zarr.open_group(str(splats_zarr), mode="r")

    # Alpha gate: zero alpha is "no surface rendered here"; the percentile cut mirrors the
    # feedforward path's confidence_mask so both sources filter by the same global rule
    depths = np.ascontiguousarray(store["depth"][:], dtype=np.float32)
    alpha = store["alpha"][:]
    keep = alpha > 0
    if conf_percentile is not None:
        keep &= confidence_mask(alpha, conf_percentile)
    dropped = ~keep
    depths[dropped] = 0.0
    logger.info(
        "Alpha mask (p%s): %.1f%% of depth pixels dropped",
        "none" if conf_percentile is None else f"{conf_percentile:.0f}",
        100.0 * float(dropped.mean()),
    )

    # Rendered uint8 RGB + the poses/intrinsics actually rendered, all native to the frames
    rgbs = np.ascontiguousarray(store["rgb"][:])
    c2w = store["c2w"][:].astype(np.float32)
    intrinsics = store["K"][:].astype(np.float32)
    return depths, rgbs, c2w, intrinsics


def optimize_color_map(
    mesh_path: Path,
    depths: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    iterations: int,
    depth_trunc: float,
) -> None:
    """Rigid Zhou-Koltun color map optimization — recolors mesh_path in place.

    Refines a private copy of the camera poses for photo-consistency and reassigns vertex
    colors from the refined poses. Poses are report-only: nothing is written back to COLMAP
    or the zarr — the overwritten mesh file is the only output.

    Args:
        mesh_path:   PLY to load, recolor, and overwrite (same in-place contract as
                     clean_repair_mesh).
        depths:      (N, H, W) float32 — the SAME array the TSDF fusion consumed.
        rgbs:        (N, H, W, 3) uint8, or float32 in [0, 1] (converted at this boundary).
        c2w:         (N, 4, 4) float32 cam-to-world OpenCV.
        intrinsics:  (N, 3, 3) float32.
        iterations:  rigid optimizer iteration count (upstream default is 300).
        depth_trunc: visibility cutoff — must match the fusion's depth_trunc; the option's
                     2.5 default assumes metric depth and ours is non-metric.

    Holds a full-resolution RGBD copy of every frame plus the optimizer's internal gradient
    images (~15-20 GB at 300 frames of 1080p) — do not co-schedule with other heavy stages.
    """
    # Float [0,1] RGB (model-res path) -> uint8 at the boundary; native path is already uint8
    if rgbs.dtype != np.uint8:
        rgbs = (np.clip(rgbs, 0.0, 1.0) * 255.0).astype(np.uint8)

    # RGBD list + camera trajectory from the same arrays the fusion consumed — resolution
    # consistency with the mesh is guaranteed by construction
    height, width = depths.shape[1:3]
    w2c = invert_poses(c2w)  # optimizer wants world-to-camera
    rgbd_images = []
    cam_params = []
    for i in range(depths.shape[0]):
        color = o3d.geometry.Image(np.ascontiguousarray(rgbs[i]))
        depth = o3d.geometry.Image(np.ascontiguousarray(depths[i], dtype=np.float32))
        rgbd_images.append(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
        )
        K = intrinsics[i]
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        cam.extrinsic = w2c[i]
        cam_params.append(cam)
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = cam_params

    # Run the rigid optimizer and overwrite the mesh; the refined trajectory is discarded.
    # Debug verbosity makes Open3D print its per-iteration residual so long runs are
    # observable (the optimizer is one opaque C++ call — this is the only progress signal).
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices():
        raise ValueError(f"{mesh_path} is missing or empty — nothing to optimize.")
    option = o3d.pipelines.color_map.RigidOptimizerOption(
        maximum_iteration=int(iterations),
        maximum_allowable_depth=float(depth_trunc),
    )
    logger.info("Color map optimization: %d frames, %d iterations", depths.shape[0], iterations)
    start = time.perf_counter()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, _ = o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, trajectory, option
        )
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    logger.info("Color map optimization done in %.1fs", time.perf_counter() - start)


def pointcloud_to_mesh(
    result: FeedforwardResult,
    output_dir: Path,
    method: str = "open3d_tsdf",
    conf_percentile: float | None = None,
    frame_store=None,
    native_intrinsics: np.ndarray | None = None,
    color_map_iterations: int = 0,
    **mesher_kwargs,
) -> MeshResult:
    """Mesh directly from a FeedforwardResult using any registered mesh method.

    All mesh creators share the same interface: create(depths, rgbs, c2w, intrinsics).
    Only "open3d_tsdf" is implemented today; Poisson variants raise NotImplementedError
    from their subclass.

    Args:
        result:            FeedforwardResult with populated depth + images. Access creator.outputs
                           after reconstruct(), or load_zarr(path, load_images=True).
        output_dir:        Directory to write mesh output.
        method:            Registry key — "open3d_tsdf", "depth_normal_poisson", "gaussians_poisson".
        conf_percentile:   Forwarded to _feedforward_to_tsdf_inputs — zero out depth below this
                           confidence percentile before fusion.
        frame_store:       Forwarded to _feedforward_to_tsdf_inputs — switches to native-resolution
                           fusion using this FrameStore's RGB.
        native_intrinsics: Forwarded to _feedforward_to_tsdf_inputs — original-res K required
                           alongside frame_store.
        color_map_iterations: Rigid color-map optimization iterations run on mesh.ply after
                           fusion + clean_repair (0 = off). Only "open3d_tsdf" supports it.
        **mesher_kwargs:   Forwarded to the mesh creator constructor (voxel_size, sdf_trunc, etc.).

    Returns:
        MeshResult with mesh_path pointing to the output PLY.

    Raises:
        ValueError: if result.depth/result.images are None, method is not in the registry,
                    or color_map_iterations > 0 with a non-TSDF method.
    """
    # Adapter then fuse — the splats source runs the same fuse on its own adapter's output
    depths, rgbs, c2w, intrinsics = _feedforward_to_tsdf_inputs(
        result,
        conf_percentile=conf_percentile,
        frame_store=frame_store,
        native_intrinsics=native_intrinsics,
    )
    return mesh_from_tsdf_inputs(
        depths,
        rgbs,
        c2w,
        intrinsics,
        output_dir,
        method=method,
        color_map_iterations=color_map_iterations,
        **mesher_kwargs,
    )


def mesh_from_tsdf_inputs(
    depths: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
    method: str = "open3d_tsdf",
    color_map_iterations: int = 0,
    **mesher_kwargs,
) -> MeshResult:
    """
    Fuse pre-built (depths, rgbs, c2w, intrinsics) with any registered mesher; optional colour-map pass.

    - Shared tail of both input adapters (pointcloud.zarr and splats.zarr).
    - Raises ValueError when color_map_iterations > 0 with a non-TSDF method.
    """
    from collab_splats.mesh import get_mesh_creator

    # Loud failure before any work — only the TSDF path has the depth_trunc + mesh.ply
    # contract the optimizer needs
    if color_map_iterations > 0 and method != "open3d_tsdf":
        raise ValueError(
            f"color_map_iterations requires method='open3d_tsdf', got {method!r}"
        )

    mesher = get_mesh_creator(method, Path(output_dir), **mesher_kwargs)
    mesh_result = mesher.create(depths, rgbs, c2w, intrinsics)

    # Color-map optimization AFTER create(): fusion and clean_repair both run inside it, so
    # the optimizer colors the final geometry instead of speckle about to be deleted
    if color_map_iterations > 0:
        optimize_color_map(
            mesh_result.mesh_path,
            depths,
            rgbs,
            c2w,
            intrinsics,
            iterations=color_map_iterations,
            depth_trunc=mesher.depth_trunc,
        )
    return mesh_result
