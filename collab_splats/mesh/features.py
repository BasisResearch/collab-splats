"""
Point features onto mesh vertices, and clustering of the result.

  - features2vertex: Gaussian-weighted k-NN scatter, truncated at sdf_trunc
  - mesh_clustering: connected components over high-similarity vertices within a radius
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

######## Feature transfer


def features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """
    Gaussian-weighted k-NN scatter of point features onto mesh vertices.

    Args:
        mesh_vertices: (M, 3) float vertex positions.
        points: (P, 3) float point positions in the same frame.
        features: (P, D) array of per-point features.
        k: int, nearest vertices each point contributes to.
        sdf_trunc: float, points whose nearest vertex is farther than this (world units) are dropped.
    Returns:
        (M, D) array in features.dtype; vertices no point reached are zero.
    """
    vertices = np.asarray(mesh_vertices)
    M = len(vertices)
    D = features.shape[1]

    # Nearest-vertex query for every point; workers=-1 uses all cores
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k, workers=-1)

    # k=1 collapses the neighbor axis; restore it so the kernel below is uniform
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Drop points whose closest vertex is beyond the truncation band
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    feats = features[valid_mask]

    # Move the aggregation to the GPU (float32); one .cpu() at the end
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.as_tensor(np.ascontiguousarray(distances), dtype=torch.float32, device=device)
    idx = torch.as_tensor(np.ascontiguousarray(indices), dtype=torch.long, device=device)
    f = torch.as_tensor(np.ascontiguousarray(feats), dtype=torch.float32, device=device)

    # Gaussian kernel over neighbor distances; normalize weights per point (over k)
    sigma = d.mean()
    w = torch.exp(-(d**2) / (2 * sigma**2))
    w = w / w.sum(dim=1, keepdim=True)

    # Scatter weighted features to vertices; accumulate weights for normalization
    acc = torch.zeros((M, D), dtype=torch.float32, device=device)
    wsum = torch.zeros((M, 1), dtype=torch.float32, device=device)
    for j in range(k):
        acc.index_add_(0, idx[:, j], f * w[:, j : j + 1])
        wsum.index_add_(0, idx[:, j], w[:, j : j + 1])

    # Normalize aggregated features by summed weights (untouched vertices stay zero)
    nz = wsum.squeeze(1) > 0
    acc[nz] /= wsum[nz]

    return acc.cpu().numpy().astype(features.dtype)


######## Clustering


def mesh_clustering(mesh, similarity_values, similarity_threshold=0.8, spatial_radius=0.03, min_cluster_size=10):
    """
    Group spatially connected vertices whose similarity exceeds a threshold.

    Args:
        mesh: open3d.geometry.TriangleMesh; only its vertices are read.
        similarity_values: (V,) float per-vertex similarity.
        similarity_threshold: float, vertices with similarity above this are candidates.
        spatial_radius: float, candidates within this distance (world units) are connected.
        min_cluster_size: int, clusters with fewer vertices are dropped.
    Returns:
        list of (n_i,) int arrays of vertex indices into the mesh, one per cluster.
    """
    similarity_values = np.asarray(similarity_values)
    valid = np.flatnonzero(similarity_values > similarity_threshold)
    if len(valid) == 0:
        return []

    # Sparse adjacency from all candidate pairs within the radius
    xyz = np.asarray(mesh.vertices)[valid]
    pairs = cKDTree(xyz).query_pairs(spatial_radius, output_type="ndarray")
    n = len(valid)
    adjacency = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)

    # Map component labels back to original vertex indices; drop small clusters
    clusters = []
    for label in np.unique(labels):
        members = valid[labels == label]
        if len(members) >= min_cluster_size:
            clusters.append(members)

    return clusters
