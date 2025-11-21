"""
Clustering utilities for feature embedding analysis.

This module provides core utilities for:
1. K-Means clustering (with MiniBatch support and automatic K selection)
2. HDBSCAN density-based clustering
3. SpLiCE integration for interpretable concept decomposition
4. Cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. Dimensionality reduction (PCA, UMAP)
6. Semantic-first clustering strategies

Based on feature_clustering_analysis.ipynb
"""

import gc
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# UMAP import (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

# SpLiCE import
try:
    import splice
    SPLICE_AVAILABLE = True
except ImportError:
    SPLICE_AVAILABLE = False
    warnings.warn("SpLiCE not available. Install from https://github.com/AI4LIFE-GROUP/SpLiCE.git")


###############################################################################
# Optimal K Selection for K-Means
###############################################################################

def compute_elbow_curve(
    features: np.ndarray,
    k_range: range,
    use_minibatch: bool = True,
    batch_size: int = 50000,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[List[float], List[int]]:
    """
    Compute elbow curve for K-Means clustering.

    Args:
        features: Feature embeddings of shape (N, D)
        k_range: Range of K values to test (e.g., range(2, 50))
        use_minibatch: Whether to use MiniBatchKMeans (faster for large datasets)
        batch_size: Batch size for MiniBatchKMeans
        random_state: Random seed
        verbose: Whether to show progress bar

    Returns:
        Tuple of (inertias, k_values)
    """
    ClusteringClass = MiniBatchKMeans if use_minibatch else KMeans

    if verbose:
        print(f"Computing elbow curve using {ClusteringClass.__name__}...")

    inertias = []
    k_values = list(k_range)

    iterator = tqdm(k_values) if verbose else k_values

    for k in iterator:
        if use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=batch_size,
                n_init=10,
                max_iter=100,
            )
        else:
            kmeans = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                max_iter=100,
            )

        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    return inertias, k_values


def find_elbow_point(k_values: List[int], inertias: List[float]) -> int:
    """
    Find elbow point using the "kneedle" algorithm.

    Args:
        k_values: List of K values
        inertias: List of corresponding inertia values

    Returns:
        Optimal K value at the elbow
    """
    # Normalize values to [0, 1]
    k_norm = (np.array(k_values) - np.min(k_values)) / (np.max(k_values) - np.min(k_values))
    inertia_norm = (np.array(inertias) - np.min(inertias)) / (np.max(inertias) - np.min(inertias))

    # Compute distance from each point to the line connecting first and last points
    x1, y1 = k_norm[0], inertia_norm[0]
    x2, y2 = k_norm[-1], inertia_norm[-1]

    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    distances = np.abs(a * k_norm + b * inertia_norm + c) / np.sqrt(a**2 + b**2)

    # Elbow is the point with maximum distance
    elbow_idx = np.argmax(distances)
    optimal_k = k_values[elbow_idx]

    return optimal_k


def select_optimal_k(
    features: np.ndarray,
    k_range: range = range(2, 50),
    use_minibatch: bool = True,
    batch_size: int = 50000,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Automatically select optimal K using elbow method.

    Args:
        features: Feature embeddings of shape (N, D)
        k_range: Range of K values to test
        use_minibatch: Whether to use MiniBatchKMeans
        batch_size: Batch size for MiniBatchKMeans
        random_state: Random seed
        verbose: Whether to print results

    Returns:
        Dictionary with optimal K and elbow curve data
    """
    inertias, k_values = compute_elbow_curve(
        features, k_range, use_minibatch, batch_size, random_state, verbose
    )
    optimal_k = find_elbow_point(k_values, inertias)

    if verbose:
        print(f"\nOptimal K (elbow method): {optimal_k}")

    return {
        'optimal_k': optimal_k,
        'inertias': inertias,
        'k_values': k_values,
    }


###############################################################################
# K-Means Clustering
###############################################################################

def kmeans_clustering(
    features: np.ndarray,
    n_clusters: int,
    use_minibatch: bool = True,
    batch_size: int = 50000,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Any]:
    """
    Perform K-Means clustering on features.

    Args:
        features: Feature embeddings of shape (N, D)
        n_clusters: Number of clusters
        use_minibatch: Whether to use MiniBatchKMeans (recommended for >100k samples)
        batch_size: Batch size for MiniBatchKMeans
        random_state: Random seed
        verbose: Whether to print cluster statistics

    Returns:
        Tuple of (cluster_labels, kmeans_model)
    """
    ClusteringClass = MiniBatchKMeans if use_minibatch else KMeans

    if verbose:
        print(f"Performing K-means clustering with K={n_clusters} using {ClusteringClass.__name__}...")

    if use_minibatch:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init=10,
            max_iter=100,
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=100,
        )

    labels = kmeans.fit_predict(features)

    if verbose:
        print(f"\nCluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Cluster {label}: {count:,} samples ({100*count/len(labels):.1f}%)")

    return labels, kmeans


###############################################################################
# HDBSCAN Clustering
###############################################################################

def hdbscan_clustering(
    features: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: Optional[int] = None,
    metric: str = 'euclidean',
    cluster_selection_method: str = 'eom',
    verbose: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Perform HDBSCAN density-based clustering on features.

    NOTE: There is NO MiniBatch version of HDBSCAN. For very large datasets (>1M samples),
    consider subsampling or using MiniBatchKMeans instead.

    Args:
        features: Feature embeddings of shape (N, D)
        min_cluster_size: Minimum number of samples in a cluster
        min_samples: Number of samples in a neighborhood (defaults to None, which uses min_cluster_size)
        metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
        cluster_selection_method: Method for selecting clusters ('eom' or 'leaf')
        verbose: Whether to print cluster statistics

    Returns:
        Tuple of (cluster_labels, n_clusters)

    Note: Labels of -1 indicate noise points (not assigned to any cluster)
    """
    # Set default min_samples if not provided
    # Lower min_samples = more clusters found (less conservative)
    if min_samples is None:
        min_samples = max(5, min_cluster_size // 10)  # More sensitive than min_cluster_size

    if verbose:
        print(f"Performing HDBSCAN clustering...")
        print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"  metric={metric}, cluster_selection_method={cluster_selection_method}")

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )

    labels = hdbscan_model.fit_predict(features)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if verbose:
        print(f"\nHDBSCAN Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise:,} ({100*n_noise/len(labels):.1f}%)")

    return labels, n_clusters


###############################################################################
# SpLiCE Integration for CLIP Embedding Interpretation
###############################################################################

def splice_clip_embeddings(
    clip_embedding: torch.Tensor,
    splicemodel,
    vocabulary: List[str],
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Decode a single CLIP embedding using SpLiCE to get sparse concept weights.

    This function takes a CLIP embedding and decomposes it into sparse, interpretable
    concepts using the SpLiCE model.

    Args:
        clip_embedding: CLIP embedding tensor of shape (D,) or (1, D)
        splicemodel: Loaded SpLiCE model
        vocabulary: SpLiCE vocabulary (list of concept strings)
        top_k: Number of top concepts to return

    Returns:
        Dictionary mapping concept names to their weights (only non-zero weights)

    Example:
        >>> clip_emb = model.encode_image(image)  # Shape: (768,)
        >>> concepts = splice_clip_embeddings(clip_emb, splicemodel, vocabulary, top_k=5)
        >>> print(concepts)
        {'animal': 0.234, 'fur': 0.156, 'rodent': 0.089, ...}
    """
    # Ensure embedding is 2D (1, D)
    if clip_embedding.dim() == 1:
        clip_embedding = clip_embedding.unsqueeze(0)

    with torch.no_grad():
        # Get sparse weights from SpLiCE
        sparse_weights = splicemodel.encode_image(clip_embedding)

    # Get top-k concepts
    top_indices = torch.argsort(sparse_weights.squeeze(), descending=True)[:top_k]

    results = {}
    for idx in top_indices:
        concept = vocabulary[idx]
        weight = float(sparse_weights[0][idx])
        if weight > 0:  # Only include non-zero weights
            results[concept] = weight

    return results


def batch_splice_clip_embeddings(
    clip_embeddings: torch.Tensor,
    splicemodel,
    vocabulary: List[str],
    top_k: int = 10,
    aggregate: str = 'mean',
) -> Dict[str, float]:
    """
    Decode a batch of CLIP embeddings using SpLiCE and aggregate results.

    This function processes multiple CLIP embeddings (e.g., from a cluster of Gaussians)
    and aggregates their sparse concept weights.

    Args:
        clip_embeddings: Batch of CLIP embeddings of shape (N, D) where N is the
                        number of embeddings and D is the embedding dimension
        splicemodel: Loaded SpLiCE model
        vocabulary: SpLiCE vocabulary (list of concept strings)
        top_k: Number of top concepts to return after aggregation
        aggregate: Aggregation method - 'mean' (average weights across batch) or
                  'first' (use only first embedding)

    Returns:
        Dictionary mapping concept names to their aggregated weights

    Example:
        >>> # Get CLIP embeddings for a cluster
        >>> cluster_clip_embs = decoded_features[cluster_mask]  # Shape: (1000, 768)
        >>> concepts = batch_splice_clip_embeddings(
        ...     cluster_clip_embs, splicemodel, vocabulary, top_k=10, aggregate='mean'
        ... )
        >>> print(concepts)
        {'background': 0.312, 'ground': 0.189, ...}
    """
    if aggregate == 'mean':
        # Average embeddings across the batch
        avg_embedding = clip_embeddings.mean(dim=0, keepdim=True)
        return splice_clip_embeddings(avg_embedding, splicemodel, vocabulary, top_k)

    elif aggregate == 'first':
        # Use only the first embedding
        return splice_clip_embeddings(clip_embeddings[0], splicemodel, vocabulary, top_k)

    elif aggregate == 'voting':
        # Decode each embedding and aggregate concept weights
        all_weights = {}

        with torch.no_grad():
            # Get sparse weights for all embeddings at once
            sparse_weights = splicemodel.encode_image(clip_embeddings)  # (N, vocab_size)

        # Sum weights across all embeddings
        summed_weights = sparse_weights.sum(dim=0)  # (vocab_size,)

        # Get top-k concepts
        top_indices = torch.argsort(summed_weights, descending=True)[:top_k]

        results = {}
        for idx in top_indices:
            concept = vocabulary[idx]
            weight = float(summed_weights[idx] / len(clip_embeddings))  # Normalize by batch size
            if weight > 0:
                results[concept] = weight

        return results

    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}. Choose 'mean', 'first', or 'voting'")


###############################################################################
# Dimensionality Reduction
###############################################################################

def reduce_dimensions_pca(
    features: np.ndarray,
    n_components: Union[int, float] = 0.95,
    whiten: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce feature dimensionality using PCA.

    Args:
        features: Feature embeddings of shape (N, D)
        n_components: Number of components or variance ratio to preserve
                     If int: exact number of components
                     If float (0-1): preserve this much variance
        whiten: Whether to whiten the features (recommended for clustering)
        verbose: Whether to print PCA statistics

    Returns:
        Tuple of (reduced_features, pca_model)
    """
    if verbose:
        print(f"Applying PCA dimensionality reduction...")
        print(f"  Input shape: {features.shape}")

    # Standardize features before PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    reduced_features = pca.fit_transform(features_scaled)

    if verbose:
        print(f"  Output shape: {reduced_features.shape}")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  Number of components: {pca.n_components_}")

    return reduced_features, pca


def reduce_dimensions_umap(
    features: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    verbose: bool = True,
) -> Tuple[np.ndarray, Any]:
    """
    Reduce feature dimensionality using UMAP.

    UMAP preserves both local and global structure better than PCA
    for non-linear manifolds. Good for visualization and clustering.

    Args:
        features: Feature embeddings of shape (N, D)
        n_components: Target dimensionality
        n_neighbors: Number of neighbors for manifold approximation
        min_dist: Minimum distance between points in low-D space
        metric: Distance metric to use
        verbose: Whether to print UMAP statistics

    Returns:
        Tuple of (reduced_features, umap_model)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")

    if verbose:
        print(f"Applying UMAP dimensionality reduction...")
        print(f"  Input shape: {features.shape}")
        print(f"  Target components: {n_components}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=verbose,
    )

    reduced_features = reducer.fit_transform(features)

    if verbose:
        print(f"  Output shape: {reduced_features.shape}")

    return reduced_features, reducer


###############################################################################
# Cluster Quality Metrics
###############################################################################

def evaluate_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    metric_names: List[str] = ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
    sample_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.

    Metrics:
    - Silhouette Score: [-1, 1], higher is better, measures cluster cohesion vs separation
    - Davies-Bouldin Index: [0, inf), lower is better, ratio of within-cluster to between-cluster distances
    - Calinski-Harabasz Score: [0, inf), higher is better, ratio of between-cluster to within-cluster variance

    Args:
        features: Feature embeddings of shape (N, D)
        labels: Cluster labels of shape (N,)
        metric_names: List of metrics to compute
        sample_size: Sample size for expensive metrics (silhouette). None = use all
        verbose: Whether to print metrics

    Returns:
        Dictionary mapping metric names to scores
    """
    n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1) if present

    if n_clusters < 2:
        if verbose:
            print("Cannot evaluate: need at least 2 clusters")
        return {}

    results = {}

    # Sample for expensive metrics if needed
    if sample_size is not None and len(features) > sample_size:
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        features_sample = features[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        features_sample = features
        labels_sample = labels

    # Silhouette Score (expensive for large datasets)
    if 'silhouette' in metric_names:
        # Exclude noise points for silhouette
        valid_mask = labels_sample >= 0
        if valid_mask.sum() >= 2 and len(np.unique(labels_sample[valid_mask])) >= 2:
            score = silhouette_score(features_sample[valid_mask], labels_sample[valid_mask])
            results['silhouette'] = score
            if verbose:
                print(f"Silhouette Score: {score:.3f} (higher is better, [-1, 1])")

    # Davies-Bouldin Index
    if 'davies_bouldin' in metric_names:
        valid_mask = labels >= 0
        if valid_mask.sum() >= 2 and len(np.unique(labels[valid_mask])) >= 2:
            score = davies_bouldin_score(features[valid_mask], labels[valid_mask])
            results['davies_bouldin'] = score
            if verbose:
                print(f"Davies-Bouldin Index: {score:.3f} (lower is better)")

    # Calinski-Harabasz Score
    if 'calinski_harabasz' in metric_names:
        valid_mask = labels >= 0
        if valid_mask.sum() >= 2 and len(np.unique(labels[valid_mask])) >= 2:
            score = calinski_harabasz_score(features[valid_mask], labels[valid_mask])
            results['calinski_harabasz'] = score
            if verbose:
                print(f"Calinski-Harabasz Score: {score:.1f} (higher is better)")

    return results


def compare_clusterings(
    features: np.ndarray,
    clusterings: Dict[str, np.ndarray],
    sample_size: Optional[int] = 10000,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple clustering results using quality metrics.

    Args:
        features: Feature embeddings of shape (N, D)
        clusterings: Dictionary mapping clustering names to label arrays
        sample_size: Sample size for silhouette computation
        verbose: Whether to print comparison table

    Returns:
        Dictionary mapping clustering names to metric dictionaries
    """
    if verbose:
        print("\n" + "="*70)
        print("CLUSTERING COMPARISON")
        print("="*70)

    results = {}

    for name, labels in clusterings.items():
        if verbose:
            print(f"\n{name}:")
        metrics = evaluate_clustering(features, labels, sample_size=sample_size, verbose=verbose)
        results[name] = metrics

    if verbose:
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(f"{'Method':<30} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<18}")
        print("-"*70)
        for name, metrics in results.items():
            sil = f"{metrics.get('silhouette', float('nan')):.3f}" if 'silhouette' in metrics else "N/A"
            db = f"{metrics.get('davies_bouldin', float('nan')):.3f}" if 'davies_bouldin' in metrics else "N/A"
            ch = f"{metrics.get('calinski_harabasz', float('nan')):.1f}" if 'calinski_harabasz' in metrics else "N/A"
            print(f"{name:<30} {sil:<12} {db:<16} {ch:<18}")

    return results


###############################################################################
# Semantic-First Clustering Strategy
###############################################################################

def semantic_spatial_clustering(
    semantic_features: np.ndarray,
    spatial_positions: np.ndarray,
    n_clusters: int,
    semantic_weight: float = 0.8,
    use_minibatch: bool = True,
    batch_size: int = 50000,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Any]:
    """
    Perform clustering with explicit semantic-spatial weighting.

    This approach:
    1. Normalizes semantic features and spatial positions separately
    2. Combines them with explicit weighting (semantic_weight controls the balance)
    3. Performs clustering on the combined representation

    Higher semantic_weight (e.g., 0.8-0.95) prioritizes semantic coherence.
    Lower semantic_weight (e.g., 0.5-0.7) balances semantic and spatial.

    Args:
        semantic_features: Semantic feature embeddings of shape (N, D)
        spatial_positions: Spatial coordinates of shape (N, 3)
        n_clusters: Number of clusters
        semantic_weight: Weight for semantic features [0, 1], spatial gets (1 - semantic_weight)
        use_minibatch: Whether to use MiniBatchKMeans
        batch_size: Batch size for MiniBatchKMeans
        random_state: Random seed
        verbose: Whether to print statistics

    Returns:
        Tuple of (cluster_labels, kmeans_model)
    """
    if verbose:
        print(f"\nSemantic-Spatial Clustering (semantic_weight={semantic_weight:.2f})")
        print(f"  Semantic features: {semantic_features.shape}")
        print(f"  Spatial positions: {spatial_positions.shape}")

    # Normalize semantic features (L2 norm)
    from sklearn.preprocessing import normalize
    semantic_norm = normalize(semantic_features, norm='l2', axis=1)

    # Normalize spatial positions (min-max to [0, 1])
    spatial_min = spatial_positions.min(axis=0)
    spatial_max = spatial_positions.max(axis=0)
    spatial_norm = (spatial_positions - spatial_min) / (spatial_max - spatial_min + 1e-8)

    # Combine with weighting
    combined = np.concatenate([
        semantic_norm * semantic_weight,
        spatial_norm * (1 - semantic_weight)
    ], axis=1)

    if verbose:
        print(f"  Combined features: {combined.shape}")

    # Cluster on combined features
    labels, model = kmeans_clustering(
        combined, n_clusters, use_minibatch, batch_size, random_state, verbose=False
    )

    if verbose:
        print(f"\nCluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Cluster {label}: {count:,} samples ({100*count/len(labels):.1f}%)")

    return labels, model


###############################################################################
# Cluster Interpretation with SpLiCE
###############################################################################

def interpret_clusters_with_splice(
    clip_embeddings: torch.Tensor,
    cluster_labels: np.ndarray,
    splicemodel,
    vocabulary: List[str],
    top_k: int = 5,
    aggregate_method: str = 'mean',
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """
    Interpret each cluster using SpLiCE to get semantic concept labels.

    This function:
    1. Groups CLIP embeddings by cluster
    2. Aggregates embeddings within each cluster
    3. Decodes aggregated embeddings to sparse concepts
    4. Returns top concepts for each cluster

    Args:
        clip_embeddings: CLIP embeddings of shape (N, D)
        cluster_labels: Cluster assignments of shape (N,)
        splicemodel: Loaded SpLiCE model
        vocabulary: SpLiCE vocabulary (list of concept strings)
        top_k: Number of top concepts per cluster
        aggregate_method: 'mean', 'first', or 'voting'
        verbose: Whether to print cluster interpretations

    Returns:
        Dictionary mapping cluster_id -> {concept: weight}
    """
    if not SPLICE_AVAILABLE:
        raise ImportError("SpLiCE not available for cluster interpretation")

    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])  # Exclude noise
    cluster_concepts = {}

    if verbose:
        print("\n" + "="*70)
        print("SEMANTIC CLUSTER INTERPRETATION (SpLiCE)")
        print("="*70)

    for cluster_id in unique_clusters:
        # Get embeddings for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embs = clip_embeddings[cluster_mask]

        # Decode to concepts
        concepts = batch_splice_clip_embeddings(
            cluster_embs,
            splicemodel,
            vocabulary,
            top_k=top_k,
            aggregate=aggregate_method
        )

        cluster_concepts[int(cluster_id)] = concepts

        if verbose:
            n_points = cluster_mask.sum()
            pct = 100 * n_points / len(cluster_labels)
            print(f"\nCluster {cluster_id} ({n_points:,} points, {pct:.1f}%):")
            for i, (concept, weight) in enumerate(concepts.items(), 1):
                print(f"  {i}. {concept}: {weight:.3f}")

    if verbose:
        print("="*70)

    return cluster_concepts


###############################################################################
# Utility Functions
###############################################################################

def free_memory():
    """Free GPU and CPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
