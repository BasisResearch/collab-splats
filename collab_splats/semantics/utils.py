"""
Semantic feature math: contrastive scoring, patch-token reshape, point clustering.

- store IO lives in `collab_splats.semantics.store`
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = ["cluster_points", "compute_semantic_contrast"]


########################################################
########## Contrastive scoring #########################
########################################################


def compute_semantic_contrast(
    raw_similarities: torch.Tensor,
    num_positive: int,
    temperature: float = 0.05,
    reduction: str = "max",
) -> torch.Tensor:
    """
    Contrastive score per patch or point: positive queries against negative ones.

    - no negatives (num_positive == N_queries): a plain reduction over positives
    - "max": each positive scored against all negatives, then max; distinct concepts
    - "pool": positives averaged before the softmax; synonyms read as one query

    Args:
        raw_similarities: (N_queries, N) similarities per patch or point (cosine when inputs are unit-norm).
        num_positive: rows [0:num_positive] are positive queries; the rest are negative.
        temperature: softmax temperature; lower is sharper. Unused without negatives.
        reduction: "max" or "pool".

    Returns:
        (N,) softmax scores in [0, 1]; the raw max or mean similarity when there are no negatives.

    Raises:
        ValueError: when `reduction` is neither "max" nor "pool".
    """
    if reduction not in ("max", "pool"):
        raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")

    pos = raw_similarities[:num_positive]
    neg = raw_similarities[num_positive:]

    if neg.shape[0] == 0:
        return pos.max(dim=0).values if reduction == "max" else pos.mean(dim=0)

    if reduction == "max":
        scores = []
        for p_i in pos:
            stacked = torch.cat([p_i.unsqueeze(0), neg], dim=0)
            scores.append(stacked.div(temperature).softmax(dim=0)[0])
        return torch.stack(scores).max(dim=0).values

    avg_pos = pos.mean(dim=0, keepdim=True)
    stacked = torch.cat([avg_pos, neg], dim=0)
    return stacked.div(temperature).softmax(dim=0)[0]


def cluster_points(
    points: np.ndarray,
    similarity_values: np.ndarray,
    similarity_threshold: float = 0.8,
    spatial_radius: float = 0.03,
    min_cluster_size: int = 10,
) -> list[np.ndarray]:
    """
    Group spatially connected points whose similarity exceeds a threshold.

    - any point set: pointcloud points or mesh vertices (`np.asarray(mesh.vertices)`)

    Args:
        points: (N, 3) positions.
        similarity_values: (N,) per-point similarity.
        similarity_threshold: points with similarity above this are candidates.
        spatial_radius: candidates within this distance (world units) are connected.
        min_cluster_size: clusters with fewer points are dropped.

    Returns:
        (n_i,) int arrays of point indices, one per cluster.
    """
    similarity_values = np.asarray(similarity_values)
    valid = np.flatnonzero(similarity_values > similarity_threshold)

    if len(valid) == 0:
        return []

    # Sparse adjacency from all candidate pairs within the radius
    xyz = np.asarray(points)[valid]
    pairs = cKDTree(xyz).query_pairs(spatial_radius, output_type="ndarray")
    n = len(valid)
    adjacency = csr_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)

    # Map component labels back to original point indices; drop small clusters
    clusters = []

    for label in np.unique(labels):
        members = valid[labels == label]

        if len(members) >= min_cluster_size:
            clusters.append(members)

    return clusters


########################################################################
# Shared token utilities
########################################################################


def _tokens_to_feature_map(tokens: torch.Tensor, input_h: int, input_w: int, patch_size: int) -> torch.Tensor:
    """
    Reshape (N, D) patch tokens to (D, H_p, W_p), L2-normalized along the channel dim.

    Args:
        tokens: (N, D) patch tokens, N == (input_h // patch_size) * (input_w // patch_size).
        input_h: preprocessed image height in pixels.
        input_w: preprocessed image width in pixels.
        patch_size: pixel stride of one patch token.

    Returns:
        (D, H_p, W_p) feature map with unit-norm patch vectors.

    Raises:
        ValueError: when the token count does not match the patch grid.
    """
    ph = input_h // patch_size
    pw = input_w // patch_size
    if tokens.shape[0] != ph * pw:
        raise ValueError(
            f"Expected {ph * pw} tokens for {input_h}x{input_w} (patch_size={patch_size}), got {tokens.shape[0]}"
        )
    feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)
    return F.normalize(feat, dim=0)
