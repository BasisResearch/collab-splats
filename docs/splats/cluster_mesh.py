"""
GPU-Accelerated Mesh Clustering - Refactored for Simplicity and Modularity

Key improvements:
- Uses clustering_utils.py for core functionality (no duplication)
- Semantic-first clustering with explicit weighting
- Integrated quality metrics evaluation
- Simplified post-processing pipeline
- Cleaner code structure with better separation of concerns

Installation:
    # RAPIDS cuML (requires CUDA-capable GPU)
    conda install -c rapidsai -c conda-forge -c nvidia \
        cuml=24.10 python=3.11 cudatoolkit=11.8

Usage:
    # Semantic-first clustering with quality metrics
    python cluster_mesh_refactored.py --dataset rats_001 --n-clusters 11 \
        --semantic-weight 0.85 --use-pca --evaluate --visualize

    # GPU-accelerated with comparison
    python cluster_mesh_refactored.py --dataset rats_001 --n-clusters 11 \
        --use-gpu --compare-weights --evaluate --visualize

    # HDBSCAN with automatic cluster detection
    python cluster_mesh_refactored.py --dataset rats_001 --algorithm hdbscan \
        --use-gpu --evaluate --visualize
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt
import colorsys
import pickle
import warnings
from tqdm import tqdm

# Add paths for imports
# Add collab-splats root to path for package imports
collab_splats_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(collab_splats_root))

# Add stage directory to path for clustering_utils
stage_dir = collab_splats_root / "stage"
sys.path.insert(0, str(stage_dir))

# Collab-splats imports
from collab_splats.wrapper import Splatter, SplatterConfig
from collab_splats.utils.visualization import MESH_KWARGS, VIZ_KWARGS, visualize_splat

# Import clustering utilities directly (no duplication!)
from clustering_utils import (
    # K-Means and elbow method
    select_optimal_k,
    kmeans_clustering,
    hdbscan_clustering,
    # Dimensionality reduction
    reduce_dimensions_pca,
    reduce_dimensions_umap,
    # Quality metrics
    evaluate_clustering,
    compare_clusterings,
    # Semantic-first clustering
    semantic_spatial_clustering,
    # SpLiCE interpretation
    interpret_clusters_with_splice,
    SPLICE_AVAILABLE,
    UMAP_AVAILABLE,
)

# GPU acceleration (cuML)
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.preprocessing import normalize as cu_normalize
    import cupy as cp
    CUML_AVAILABLE = True
    print("✓ RAPIDS cuML available - GPU acceleration enabled")
except ImportError:
    CUML_AVAILABLE = False
    print("⚠ RAPIDS cuML not available - falling back to CPU")

# CPU fallback
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.stats import mode
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# GPU/CPU Utility Functions
# ============================================================================

def to_gpu(array):
    """Convert numpy array to GPU array (cupy)."""
    if CUML_AVAILABLE:
        return cp.asarray(array)
    return array


def to_cpu(array):
    """Convert GPU array (cupy) to numpy array."""
    if CUML_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


# ============================================================================
# GPU-Accelerated Clustering (CuML-specific)
# ============================================================================

def gpu_kmeans_clustering(features, n_clusters, verbose=True):
    """
    GPU-accelerated K-means using cuML.

    10-100x faster than CPU for large datasets.
    """
    if not CUML_AVAILABLE:
        raise ImportError("cuML not available for GPU clustering")

    if verbose:
        print(f"GPU K-means with K={n_clusters}...")

    # Transfer to GPU
    features_gpu = to_gpu(features)

    # GPU K-means
    kmeans = cuKMeans(
        n_clusters=n_clusters,
        random_state=42,
        max_iter=1000,
        n_init=100,
        tol=1e-5,
        verbose=0
    )

    labels_gpu = kmeans.fit_predict(features_gpu)
    labels = to_cpu(labels_gpu)

    if verbose:
        print(f"GPU clustering completed in {kmeans.n_iter_} iterations")
        print_cluster_distribution(labels)

    return labels, kmeans


def gpu_hdbscan_clustering(features, min_cluster_size=500, min_samples=None, verbose=True):
    """
    GPU-accelerated HDBSCAN using cuML.

    Significantly faster than CPU HDBSCAN.
    """
    if not CUML_AVAILABLE:
        raise ImportError("cuML not available for GPU clustering")

    # Set default min_samples if not provided
    # Lower min_samples = more clusters found (less conservative)
    if min_samples is None:
        min_samples = max(5, min_cluster_size // 10)  # More sensitive than min_cluster_size

    if verbose:
        print(f"GPU HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    # Transfer to GPU
    features_gpu = to_gpu(features)

    # GPU HDBSCAN
    clusterer = cuHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='leaf',
        gen_min_span_tree=True
    )

    labels_gpu = clusterer.fit_predict(features_gpu)
    labels = to_cpu(labels_gpu)

    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = np.sum(labels == -1)

    if verbose:
        print(f"Found {n_clusters} clusters, {n_noise:,} noise points ({100*n_noise/len(labels):.1f}%)")
        print_cluster_distribution(labels)

    return labels, n_clusters


def gpu_semantic_spatial_clustering(
    semantic_features,
    spatial_positions,
    n_clusters,
    semantic_weight=0.85,
    verbose=True
):
    """
    GPU-accelerated semantic-spatial clustering.

    Combines semantic features with spatial positions, then uses GPU K-means.
    """
    if not CUML_AVAILABLE:
        if verbose:
            print("⚠ GPU requested but cuML not available, using CPU")
        return semantic_spatial_clustering(
            semantic_features, spatial_positions, n_clusters,
            semantic_weight=semantic_weight, verbose=verbose
        )

    if verbose:
        print(f"\nGPU Semantic-Spatial Clustering (semantic_weight={semantic_weight:.2f})")
        print(f"  Semantic features: {semantic_features.shape}")
        print(f"  Spatial positions: {spatial_positions.shape}")

    # Normalize on CPU (cuML normalize is finicky)
    semantic_norm = normalize(semantic_features, norm='l2', axis=1)

    # Normalize spatial
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

    # GPU clustering
    labels, model = gpu_kmeans_clustering(combined, n_clusters, verbose=verbose)

    return labels, model


def semantic_spatial_hdbscan(
    semantic_features,
    spatial_positions,
    min_cluster_size=500,
    min_samples=None,
    semantic_weight=0.85,
    verbose=True
):
    """
    HDBSCAN with semantic-spatial feature combination.

    Combines semantic features with spatial positions before clustering.
    """
    if verbose:
        print(f"\nSemantic-Spatial HDBSCAN (semantic_weight={semantic_weight:.2f})")
        print(f"  Semantic features: {semantic_features.shape}")
        print(f"  Spatial positions: {spatial_positions.shape}")

    # Normalize semantic features
    semantic_norm = normalize(semantic_features, norm='l2', axis=1)

    # Normalize spatial positions
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

    # Call imported hdbscan_clustering from clustering_utils
    labels, n_clusters = hdbscan_clustering(
        combined,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        verbose=verbose
    )

    return labels, n_clusters


def gpu_semantic_spatial_hdbscan(
    semantic_features,
    spatial_positions,
    min_cluster_size=500,
    min_samples=None,
    semantic_weight=0.85,
    verbose=True
):
    """
    GPU-accelerated HDBSCAN with semantic-spatial feature combination.
    """
    if not CUML_AVAILABLE:
        if verbose:
            print("⚠ GPU requested but cuML not available, using CPU")
        return semantic_spatial_hdbscan(
            semantic_features, spatial_positions,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            semantic_weight=semantic_weight,
            verbose=verbose
        )

    if verbose:
        print(f"\nGPU Semantic-Spatial HDBSCAN (semantic_weight={semantic_weight:.2f})")
        print(f"  Semantic features: {semantic_features.shape}")
        print(f"  Spatial positions: {spatial_positions.shape}")

    # Normalize on CPU (cuML normalize is finicky)
    semantic_norm = normalize(semantic_features, norm='l2', axis=1)

    # Normalize spatial
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

    # GPU HDBSCAN
    labels, n_clusters = gpu_hdbscan_clustering(
        combined,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        verbose=verbose
    )

    return labels, n_clusters


def spectral_clustering_features(
    positions,
    features,
    n_clusters,
    n_neighbors=30,
    spatial_weight=0.3,
    verbose=True
):
    """
    Spectral clustering combining spatial and semantic features.

    Note: No GPU version available - uses CPU sklearn implementation.
    Spectral clustering requires graph construction which is complex on GPU.
    """
    if verbose:
        print(f"\nSpectral Clustering with K={n_clusters}, n_neighbors={n_neighbors}")
        print(f"  Spatial weight: {spatial_weight:.2f}")

    n_points = len(positions)

    # Build KNN graph
    tree = KDTree(positions)
    distances, indices = tree.query(positions, k=n_neighbors+1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    if verbose:
        print("  Computing edge weights...")

    # Spatial weights
    spatial_scale = np.median(distances)
    spatial_weights = np.exp(-distances / spatial_scale).flatten()

    # Feature similarity
    feature_weights = np.zeros(n_points * n_neighbors)
    features_norm = normalize(features, norm='l2', axis=1)

    for i in range(n_points):
        neighbor_feats = features_norm[indices[i]]
        my_feat = features_norm[i]
        sims = np.dot(neighbor_feats, my_feat)
        sims = np.maximum(0, sims)
        feature_weights[i*n_neighbors:(i+1)*n_neighbors] = sims

    # Combine weights
    combined_weights = (1 - spatial_weight) * feature_weights + spatial_weight * spatial_weights

    # Build adjacency
    row_ind = np.repeat(np.arange(n_points), n_neighbors)
    col_ind = indices.flatten()
    adjacency = csr_matrix((combined_weights, (row_ind, col_ind)), shape=(n_points, n_points))
    adjacency = adjacency + adjacency.T
    adjacency.data = adjacency.data / 2

    if verbose:
        print("  Running spectral clustering...")

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        n_init=10,
        random_state=42
    )
    labels = clustering.fit_predict(adjacency)

    if verbose:
        print_cluster_distribution(labels)

    return labels


def agglomerative_clustering_features(
    positions,
    features,
    n_clusters,
    n_neighbors=30,
    spatial_weight=0.05,
    verbose=True
):
    """
    Agglomerative clustering combining spatial and semantic features.

    Note: No GPU version available - uses CPU sklearn implementation.
    """
    if verbose:
        print(f"\nAgglomerative Clustering with K={n_clusters}, connectivity={n_neighbors}")
        print(f"  Spatial weight: {spatial_weight:.2f}")

    # Combine features with spatial positions
    pos_normalized = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)
    combined_features = np.concatenate([features, pos_normalized * spatial_weight], axis=1)

    if verbose:
        print("  Building connectivity graph...")

    connectivity = kneighbors_graph(positions, n_neighbors=n_neighbors, include_self=False)

    if verbose:
        print("  Running agglomerative clustering...")

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage='ward'
    )
    labels = clustering.fit_predict(combined_features)

    if verbose:
        print_cluster_distribution(labels)

    return labels


# ============================================================================
# Helper Functions
# ============================================================================

def print_cluster_distribution(labels):
    """Print cluster size distribution."""
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Cluster {label}: {count:,} samples ({100*count/len(labels):.1f}%)")


def normalize_features_wrapper(features, method='l2', use_gpu=False):
    """Normalize features with optional GPU acceleration."""
    if use_gpu and CUML_AVAILABLE:
        features_gpu = to_gpu(features)
        normalized_gpu = cu_normalize(features_gpu, norm=method)
        return to_cpu(normalized_gpu)
    else:
        return normalize(features, norm=method, axis=1)


# ============================================================================
# Simplified Post-Processing
# ============================================================================

def spatial_smoothing_knn(positions, labels, k=20, iterations=1, verbose=True):
    """
    Lightweight spatial smoothing using KNN voting.

    Simplified from original: fewer iterations, cleaner implementation.
    """
    if verbose:
        print(f"\nSpatial smoothing (k={k}, iterations={iterations})...")

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(positions)
    smoothed_labels = labels.copy()

    for iteration in range(iterations):
        distances, indices = nbrs.kneighbors(positions)
        neighbor_indices = indices[:, 1:]  # Exclude self
        neighbor_labels = smoothed_labels[neighbor_indices]

        # Majority voting
        voted_labels, _ = mode(neighbor_labels, axis=1, keepdims=False)
        smoothed_labels = voted_labels.flatten()

        if verbose:
            n_changed = (smoothed_labels != labels).sum()
            print(f"  Iteration {iteration+1}: {n_changed:,} points reassigned")

    return smoothed_labels


def handle_hdbscan_noise(positions, labels, method='knn', k=20, verbose=True):
    """
    Handle HDBSCAN noise points by reassigning to nearest cluster.

    Simplified from original: single clean method.
    """
    noise_mask = labels == -1
    n_noise = noise_mask.sum()

    if n_noise == 0:
        return labels

    if verbose:
        print(f"\nHandling {n_noise:,} noise points ({100*n_noise/len(labels):.1f}%)...")

    clustered_mask = ~noise_mask
    noise_indices = np.where(noise_mask)[0]
    clustered_indices = np.where(clustered_mask)[0]

    # Build KD-tree of clustered points
    tree = KDTree(positions[clustered_mask])

    new_labels = labels.copy()

    # Assign each noise point to nearest cluster
    for noise_idx in tqdm(noise_indices, disable=not verbose, desc="Reassigning noise"):
        distances, neighbor_indices = tree.query([positions[noise_idx]], k=k)
        actual_neighbors = clustered_indices[neighbor_indices[0]]
        neighbor_labels = labels[actual_neighbors]

        # Weighted voting based on distance
        weights = 1.0 / (distances[0] + 1e-8)
        weighted_votes = {}
        for label, weight in zip(neighbor_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        new_labels[noise_idx] = max(weighted_votes.items(), key=lambda x: x[1])[0]

    if verbose:
        print(f"  All noise points reassigned")

    return new_labels


# ============================================================================
# Clustering Pipeline
# ============================================================================

def cluster_features(
    features,
    positions,
    n_clusters=None,
    algorithm='kmeans',
    semantic_weight=0.85,
    use_gpu=False,
    use_pca=False,
    pca_variance=0.95,
    min_cluster_size=500,
    min_samples=None,
    k_range=(5, 25),
    graph_neighbors=30,
    graph_spatial_weight=None,
    verbose=True,
):
    """
    Main clustering pipeline with all preprocessing.

    Simplified and modular approach:
    1. Optional PCA dimensionality reduction
    2. Semantic-spatial clustering with explicit weighting
    3. Quality evaluation

    Args:
        features: Semantic feature embeddings (N, D)
        positions: Spatial positions (N, 3)
        n_clusters: Number of clusters (None/'auto' for elbow method)
        algorithm: 'kmeans', 'hdbscan', 'spectral', or 'agglomerative'
        semantic_weight: Weight for semantic features [0, 1]
        use_gpu: Use GPU acceleration if available
        use_pca: Apply PCA dimensionality reduction
        pca_variance: Variance to preserve with PCA
        min_cluster_size: For HDBSCAN - minimum cluster size
        min_samples: For HDBSCAN - samples in neighborhood (None = auto)
        k_range: Range for auto K selection
        graph_neighbors: Number of neighbors for graph-based methods
        graph_spatial_weight: Spatial weight for graph methods (overrides semantic_weight)
        verbose: Print statistics

    Returns:
        Tuple of (labels, n_clusters, features_processed)
    """
    print("\n" + "="*70)
    print("CLUSTERING PIPELINE")
    print("="*70)

    # Step 1: Optional dimensionality reduction
    if use_pca:
        print("\n--- STEP 1: DIMENSIONALITY REDUCTION ---")
        features_processed, pca = reduce_dimensions_pca(
            features,
            n_components=pca_variance,
            whiten=True,
            verbose=verbose
        )
    else:
        features_processed = features
        if verbose:
            print("\n--- STEP 1: DIMENSIONALITY REDUCTION ---")
            print("Skipped (use --use-pca to enable)")

    # Step 2: Clustering
    print("\n--- STEP 2: CLUSTERING ---")

    if algorithm == 'kmeans':
        # Auto-select K if needed
        if n_clusters is None or n_clusters == 'auto':
            if verbose:
                print("Auto-selecting optimal K...")
            k_result = select_optimal_k(
                features_processed,
                k_range=range(k_range[0], k_range[1]),
                use_minibatch=True,
                verbose=verbose
            )
            n_clusters = k_result['optimal_k']
            print(f"✓ Optimal K (elbow method): {n_clusters}")

        # Semantic-spatial clustering
        if use_gpu and CUML_AVAILABLE:
            labels, model = gpu_semantic_spatial_clustering(
                features_processed, positions, n_clusters,
                semantic_weight=semantic_weight, verbose=verbose
            )
        else:
            labels, model = semantic_spatial_clustering(
                features_processed, positions, n_clusters,
                semantic_weight=semantic_weight, verbose=verbose
            )

    elif algorithm == 'spectral':
        # Spectral clustering (graph-based)
        if n_clusters is None or n_clusters == 'auto':
            raise ValueError("Spectral clustering requires explicit n_clusters (cannot use 'auto')")

        # Use graph_spatial_weight if provided, else convert from semantic_weight
        if graph_spatial_weight is None:
            graph_spatial_weight = 1.0 - semantic_weight

        if use_gpu and CUML_AVAILABLE:
            print("⚠ Note: Spectral clustering uses CPU (no cuML GPU implementation)")

        labels = spectral_clustering_features(
            positions, features_processed, n_clusters,
            n_neighbors=graph_neighbors,
            spatial_weight=graph_spatial_weight,
            verbose=verbose
        )

    elif algorithm == 'agglomerative':
        # Agglomerative clustering (hierarchical)
        if n_clusters is None or n_clusters == 'auto':
            raise ValueError("Agglomerative clustering requires explicit n_clusters (cannot use 'auto')")

        # Use graph_spatial_weight if provided, else convert from semantic_weight
        if graph_spatial_weight is None:
            graph_spatial_weight = 1.0 - semantic_weight

        if use_gpu and CUML_AVAILABLE:
            print("⚠ Note: Agglomerative clustering uses CPU (no cuML GPU implementation)")

        labels = agglomerative_clustering_features(
            positions, features_processed, n_clusters,
            n_neighbors=graph_neighbors,
            spatial_weight=graph_spatial_weight,
            verbose=verbose
        )

    elif algorithm == 'hdbscan':
        # HDBSCAN (density-based, auto # clusters)
        # Use semantic-spatial weighting for better clustering
        if use_gpu and CUML_AVAILABLE:
            labels, n_clusters = gpu_semantic_spatial_hdbscan(
                features_processed, positions,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                semantic_weight=semantic_weight,
                verbose=verbose
            )
        else:
            labels, n_clusters = semantic_spatial_hdbscan(
                features_processed, positions,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                semantic_weight=semantic_weight,
                verbose=verbose
            )

        # Handle noise points
        if np.any(labels == -1):
            labels = handle_hdbscan_noise(positions, labels, k=20, verbose=verbose)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: kmeans, spectral, agglomerative, hdbscan")

    return labels, n_clusters, features_processed


# ============================================================================
# Visualization
# ============================================================================

def generate_colors(n_clusters):
    """Generate distinct colors for clusters."""
    hues = np.linspace(0, 1, n_clusters, endpoint=False)
    return np.array([
        [int(c * 255) for c in colorsys.hsv_to_rgb(h, 0.65, 0.95)]
        for h in hues
    ], dtype=np.uint8)


def visualize_plain_mesh(pv_mesh, dataset_name, output_path):
    """Create plain mesh visualization without clustering."""
    print("\nRendering plain mesh...")
    pv.start_xvfb()

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])

    # Use original mesh colors if available, otherwise use a default color
    if "RGB" in pv_mesh.array_names:
        plotter.add_mesh(
            pv_mesh,
            scalars="RGB",
            rgb=True,
            point_size=2,
            render_points_as_spheres=True,
            ambient=0.3,
            diffuse=0.8,
            specular=0.1,
        )
    else:
        plotter.add_mesh(
            pv_mesh,
            color='lightgray',
            point_size=2,
            render_points_as_spheres=True,
            ambient=0.3,
            diffuse=0.8,
            specular=0.1,
        )

    plotter.camera_position = [
        VIZ_KWARGS.get("position", (2, 2, 1)),
        VIZ_KWARGS.get("focal_point", (0, 0, 0)),
        VIZ_KWARGS.get("view_up", (0, 0, 1)),
    ]
    plotter.camera.azimuth = VIZ_KWARGS.get("azimuth", 235)
    plotter.camera.elevation = VIZ_KWARGS.get("elevation", 15)
    plotter.camera.Zoom(VIZ_KWARGS.get("zoom", 0.9))

    for light in VIZ_KWARGS.get("lighting", []):
        plotter.add_light(pv.Light(**light))

    img = plotter.screenshot()
    plotter.close()

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'{dataset_name} - Plain Mesh', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path.name}")


def visualize_all_clusters(pv_mesh, labels, n_clusters, dataset_name, algorithm_name, output_path, metrics=None):
    """Create view showing all clusters colored."""
    print(f"\nRendering all clusters ({algorithm_name})...")
    pv.start_xvfb()

    colors = generate_colors(n_clusters)
    mesh_copy = pv_mesh.copy()
    mesh_copy["RGB"] = colors[labels]

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])

    plotter.add_mesh(
        mesh_copy,
        scalars="RGB",
        rgb=True,
        point_size=2,
        render_points_as_spheres=True,
        ambient=0.3,
        diffuse=0.8,
        specular=0.1,
    )

    plotter.camera_position = [
        VIZ_KWARGS.get("position", (2, 2, 1)),
        VIZ_KWARGS.get("focal_point", (0, 0, 0)),
        VIZ_KWARGS.get("view_up", (0, 0, 1)),
    ]
    plotter.camera.azimuth = VIZ_KWARGS.get("azimuth", 235)
    plotter.camera.elevation = VIZ_KWARGS.get("elevation", 15)
    plotter.camera.Zoom(VIZ_KWARGS.get("zoom", 0.9))

    for light in VIZ_KWARGS.get("lighting", []):
        plotter.add_light(pv.Light(**light))

    img = plotter.screenshot()
    plotter.close()

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')

    # Create title with metrics if available
    title = f'{dataset_name} - {algorithm_name} All Clusters (K={n_clusters})'
    if metrics:
        title += f'\nSilhouette: {metrics["silhouette"]:.3f} | Davies-Bouldin: {metrics["davies_bouldin"]:.3f} | Calinski-Harabasz: {metrics["calinski_harabasz"]:.1f}'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path.name}")


def visualize_individual_clusters(pv_mesh, labels, n_clusters, dataset_name, algorithm_name, output_path, grid_cols=4):
    """Create multi-panel view with each cluster highlighted individually."""
    print(f"\nRendering individual cluster panels ({algorithm_name})...")
    pv.start_xvfb()

    colors = generate_colors(n_clusters)
    grid_rows = int(np.ceil(n_clusters / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 3.5))
    axes = axes.flatten() if n_clusters > 1 else [axes]

    for cluster_id in range(n_clusters):
        print(f"  Cluster {cluster_id + 1}/{n_clusters}...")

        mesh_copy = pv_mesh.copy()

        cluster_colors = np.zeros((len(labels), 3), dtype=np.uint8)
        mask = labels == cluster_id
        cluster_colors[mask] = colors[cluster_id]
        mesh_copy["RGB"] = cluster_colors

        plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])

        plotter.add_mesh(
            mesh_copy,
            scalars="RGB",
            rgb=True,
            point_size=2,
            render_points_as_spheres=True,
            ambient=0.3,
            diffuse=0.8,
            specular=0.1,
        )

        plotter.camera_position = [
            VIZ_KWARGS.get("position", (2, 2, 1)),
            VIZ_KWARGS.get("focal_point", (0, 0, 0)),
            VIZ_KWARGS.get("view_up", (0, 0, 1)),
        ]
        plotter.camera.azimuth = VIZ_KWARGS.get("azimuth", 235)
        plotter.camera.elevation = VIZ_KWARGS.get("elevation", 15)
        plotter.camera.Zoom(VIZ_KWARGS.get("zoom", 0.9))

        for light in VIZ_KWARGS.get("lighting", []):
            plotter.add_light(pv.Light(**light))

        img = plotter.screenshot()
        plotter.close()

        axes[cluster_id].imshow(img)
        axes[cluster_id].axis('off')

        n_pts = mask.sum()
        pct = (n_pts / len(labels)) * 100
        axes[cluster_id].set_title(
            f'Cluster {cluster_id}\n({n_pts:,} vertices, {pct:.1f}%)',
            fontsize=10
        )

    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        f'{dataset_name} - {algorithm_name} Individual Cluster Segments (K={n_clusters})',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path.name}")


# ============================================================================
# Dataset Loading
# ============================================================================

DATASETS = {
    'birds_008': {
        'file_path': '/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4',
        'frame_proportion': 0.25,
    },
    'rats_001': {
        'file_path': '/workspace/fieldwork-data/rats/2024-07-11/SplatsSD/C0119.MP4',
        'frame_proportion': 0.25,
    }
}


def load_mesh_and_features(dataset_name, config):
    """Load mesh, features, and create PyVista mesh. Returns the splat directory for output."""
    print(f"\nInitializing {dataset_name}...")

    splatter_config = SplatterConfig(
        file_path=config['file_path'],
        method="rade-features",
        frame_proportion=config['frame_proportion'],
    )
    splatter = Splatter(splatter_config)

    # Check if preprocessing and features are already available
    splatter.preprocess()
    splatter.extract_features()
    splatter.mesh()

    mesh_path = splatter.config["mesh_info"]["mesh"]
    mesh_features_path = mesh_path.parent / "mesh_features.pt"

    print(f"Loading mesh: {mesh_path.name}")
    print(f"Loading features: {mesh_features_path.name}")

    mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh_features = torch.load(mesh_features_path).cpu().numpy()
    mesh_vertices = np.asarray(mesh_o3d.vertices)
    pv_mesh = pv.read(str(mesh_path))

    # The splat directory is the parent of the mesh directory
    splat_dir = mesh_path.parent.parent

    print(f"✓ Vertices: {len(mesh_vertices):,}, Features: {mesh_features.shape}")
    print(f"✓ Splat directory: {splat_dir}")

    return pv_mesh, mesh_features, mesh_vertices, splat_dir


def save_results(labels, n_clusters, dataset_name, output_dir, metadata=None):
    """Save clustering labels and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / f"{dataset_name}_labels_k{n_clusters}.npy"
    np.save(labels_path, labels)

    meta = {
        'dataset': dataset_name,
        'n_clusters': n_clusters,
        'n_vertices': len(labels),
        'distribution': {int(i): int((labels == i).sum()) for i in range(n_clusters)}
    }

    if metadata:
        meta.update(metadata)

    metadata_path = output_dir / f"{dataset_name}_metadata_k{n_clusters}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n✓ Saved: {labels_path.name}, {metadata_path.name}")


# ============================================================================
# Main Processing Function
# ============================================================================

def process_dataset(dataset_name, args):
    """Process a single dataset with simplified pipeline."""
    print(f"\n{'='*70}")
    print(f"PROCESSING: {dataset_name}")
    print(f"{'='*70}")

    config = DATASETS[dataset_name]

    # Load mesh and features
    pv_mesh, features, positions, splat_dir = load_mesh_and_features(dataset_name, config)

    # Use splat_dir as the base output directory
    output_dir = Path(splat_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create clustering subdirectory for all visualizations
    plot_dir = output_dir / "clustering"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory set to: {output_dir}")
    print(f"✓ Plot directory set to: {plot_dir}")

    # Plain mesh visualization (always create this first)
    if args.visualize:
        print("\n--- PLAIN MESH VISUALIZATION ---")
        plain_mesh_path = plot_dir / f"{dataset_name}_plain_mesh.png"
        visualize_plain_mesh(pv_mesh, dataset_name, plain_mesh_path)

    # Run KMeans clustering
    print("\n" + "="*70)
    print("RUNNING KMEANS CLUSTERING")
    print("="*70)

    kmeans_labels, kmeans_n_clusters, kmeans_features = cluster_features(
        features=features,
        positions=positions,
        n_clusters='auto',  # Use elbow method
        algorithm='kmeans',
        semantic_weight=args.semantic_weight,
        use_gpu=args.use_gpu,
        use_pca=args.use_pca,
        pca_variance=args.pca_variance,
        k_range=args.k_range,
        verbose=True,
    )

    # Optional: lightweight spatial smoothing for KMeans
    if args.smooth:
        print("\n--- POST-PROCESSING (KMeans) ---")
        kmeans_labels = spatial_smoothing_knn(
            positions, kmeans_labels,
            k=args.smooth_k,
            iterations=args.smooth_iterations,
            verbose=True
        )

    # Relabel consecutively
    unique_labels = np.unique(kmeans_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    kmeans_labels = np.array([label_map[l] for l in kmeans_labels])
    kmeans_n_clusters = len(unique_labels)

    # Save KMeans results
    save_results(kmeans_labels, kmeans_n_clusters, dataset_name, plot_dir,
                 metadata={'algorithm': 'kmeans', 'elbow_method': True})

    # Compute KMeans metrics for visualization
    kmeans_metrics_dict = None
    if args.evaluate or args.visualize:
        print("\n--- Computing KMeans metrics ---")
        kmeans_metrics = evaluate_clustering(
            features=kmeans_features,
            labels=kmeans_labels,
            sample_size=args.eval_sample_size,
            verbose=False
        )
        kmeans_metrics_dict = {
            'silhouette': kmeans_metrics['silhouette'],
            'davies_bouldin': kmeans_metrics['davies_bouldin'],
            'calinski_harabasz': kmeans_metrics['calinski_harabasz']
        }

    # KMeans visualization
    if args.visualize:
        print("\n--- KMEANS VISUALIZATION ---")

        if args.viz_all:
            path = plot_dir / f"{dataset_name}_kmeans_all_clusters_k{kmeans_n_clusters}.png"
            visualize_all_clusters(pv_mesh, kmeans_labels, kmeans_n_clusters,
                                 dataset_name, "KMeans", path, kmeans_metrics_dict)

        if args.viz_panels:
            path = plot_dir / f"{dataset_name}_kmeans_cluster_panels_k{kmeans_n_clusters}.png"
            visualize_individual_clusters(pv_mesh, kmeans_labels, kmeans_n_clusters,
                                        dataset_name, "KMeans", path, args.grid_cols)

    # Run HDBSCAN clustering
    print("\n" + "="*70)
    print("RUNNING HDBSCAN CLUSTERING")
    print("="*70)

    hdbscan_labels, hdbscan_n_clusters, hdbscan_features = cluster_features(
        features=features,
        positions=positions,
        algorithm='hdbscan',
        semantic_weight=args.semantic_weight,
        use_gpu=args.use_gpu,
        use_pca=args.use_pca,
        pca_variance=args.pca_variance,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        verbose=True,
    )

    # Optional: lightweight spatial smoothing for HDBSCAN
    if args.smooth:
        print("\n--- POST-PROCESSING (HDBSCAN) ---")
        hdbscan_labels = spatial_smoothing_knn(
            positions, hdbscan_labels,
            k=args.smooth_k,
            iterations=args.smooth_iterations,
            verbose=True
        )

    # Relabel consecutively
    unique_labels = np.unique(hdbscan_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    hdbscan_labels = np.array([label_map[l] for l in hdbscan_labels])
    hdbscan_n_clusters = len(unique_labels)

    # Save HDBSCAN results
    save_results(hdbscan_labels, hdbscan_n_clusters, dataset_name, plot_dir,
                 metadata={'algorithm': 'hdbscan'})

    # Compute HDBSCAN metrics for visualization
    hdbscan_metrics_dict = None
    if args.evaluate or args.visualize:
        print("\n--- Computing HDBSCAN metrics ---")
        hdbscan_metrics = evaluate_clustering(
            features=hdbscan_features,
            labels=hdbscan_labels,
            sample_size=args.eval_sample_size,
            verbose=False
        )
        hdbscan_metrics_dict = {
            'silhouette': hdbscan_metrics['silhouette'],
            'davies_bouldin': hdbscan_metrics['davies_bouldin'],
            'calinski_harabasz': hdbscan_metrics['calinski_harabasz']
        }

    # HDBSCAN visualization
    if args.visualize:
        print("\n--- HDBSCAN VISUALIZATION ---")

        if args.viz_all:
            path = plot_dir / f"{dataset_name}_hdbscan_all_clusters_k{hdbscan_n_clusters}.png"
            visualize_all_clusters(pv_mesh, hdbscan_labels, hdbscan_n_clusters,
                                 dataset_name, "HDBSCAN", path, hdbscan_metrics_dict)

        if args.viz_panels:
            path = plot_dir / f"{dataset_name}_hdbscan_cluster_panels_k{hdbscan_n_clusters}.png"
            visualize_individual_clusters(pv_mesh, hdbscan_labels, hdbscan_n_clusters,
                                        dataset_name, "HDBSCAN", path, args.grid_cols)

    # Evaluation - save metrics if computed
    if args.evaluate:
        print("\n--- EVALUATION: CLUSTER QUALITY METRICS ---")

        # Save KMeans metrics (already computed if visualize was enabled)
        if kmeans_metrics_dict is None:
            print("\nKMeans metrics:")
            kmeans_metrics = evaluate_clustering(
                features=kmeans_features,
                labels=kmeans_labels,
                sample_size=args.eval_sample_size,
                verbose=True
            )
        else:
            print(f"\nKMeans metrics:")
            print(f"Silhouette Score: {kmeans_metrics_dict['silhouette']:.3f} (higher is better, [-1, 1])")
            print(f"Davies-Bouldin Index: {kmeans_metrics_dict['davies_bouldin']:.3f} (lower is better)")
            print(f"Calinski-Harabasz Score: {kmeans_metrics_dict['calinski_harabasz']:.1f} (higher is better)")
            # Convert dict back to full metrics format
            kmeans_metrics = {
                'silhouette': kmeans_metrics_dict['silhouette'],
                'davies_bouldin': kmeans_metrics_dict['davies_bouldin'],
                'calinski_harabasz': kmeans_metrics_dict['calinski_harabasz']
            }

        metrics_path = plot_dir / f"{dataset_name}_kmeans_metrics_k{kmeans_n_clusters}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(kmeans_metrics, f)
        print(f"✓ Saved KMeans metrics: {metrics_path.name}")

        # Save HDBSCAN metrics (already computed if visualize was enabled)
        if hdbscan_metrics_dict is None:
            print("\nHDBSCAN metrics:")
            hdbscan_metrics = evaluate_clustering(
                features=hdbscan_features,
                labels=hdbscan_labels,
                sample_size=args.eval_sample_size,
                verbose=True
            )
        else:
            print(f"\nHDBSCAN metrics:")
            print(f"Silhouette Score: {hdbscan_metrics_dict['silhouette']:.3f} (higher is better, [-1, 1])")
            print(f"Davies-Bouldin Index: {hdbscan_metrics_dict['davies_bouldin']:.3f} (lower is better)")
            print(f"Calinski-Harabasz Score: {hdbscan_metrics_dict['calinski_harabasz']:.1f} (higher is better)")
            # Convert dict back to full metrics format
            hdbscan_metrics = {
                'silhouette': hdbscan_metrics_dict['silhouette'],
                'davies_bouldin': hdbscan_metrics_dict['davies_bouldin'],
                'calinski_harabasz': hdbscan_metrics_dict['calinski_harabasz']
            }

        metrics_path = plot_dir / f"{dataset_name}_hdbscan_metrics_k{hdbscan_n_clusters}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(hdbscan_metrics, f)
        print(f"✓ Saved HDBSCAN metrics: {metrics_path.name}")

    print(f"\n{'='*70}")
    print(f"✓ COMPLETED: {dataset_name}")
    print(f"  - Plain mesh visualization")
    print(f"  - KMeans clustering (K={kmeans_n_clusters})")
    print(f"  - HDBSCAN clustering (K={hdbscan_n_clusters})")
    print(f"  - All outputs saved to: {plot_dir}")
    print(f"{'='*70}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Refactored GPU-Accelerated Mesh Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Semantic-first clustering with PCA and quality metrics
  python cluster_mesh_refactored.py --dataset rats_001 --n-clusters 11 \\
      --semantic-weight 0.85 --use-pca --evaluate --visualize

  # GPU-accelerated with weight comparison
  python cluster_mesh_refactored.py --dataset rats_001 --n-clusters 11 \\
      --use-gpu --compare-weights --evaluate --visualize

  # HDBSCAN with automatic cluster detection
  python cluster_mesh_refactored.py --dataset rats_001 --algorithm hdbscan \\
      --use-gpu --evaluate --visualize

  # Auto K-selection with comparison
  python cluster_mesh_refactored.py --dataset birds_008 --n-clusters auto \\
      --use-pca --evaluate --visualize
        """
    )

    # Dataset
    parser.add_argument('--dataset', nargs='+', choices=list(DATASETS.keys()), required=True,
                       help='Dataset(s) to process')

    # Core clustering
    clustering = parser.add_argument_group('Clustering')
    clustering.add_argument('--algorithm',
                           choices=['kmeans', 'spectral', 'agglomerative', 'hdbscan'],
                           default='kmeans',
                           help='Clustering algorithm (default: kmeans)')
    clustering.add_argument('--n-clusters', type=str, default='8',
                           help='Number of clusters or "auto" for elbow method (kmeans/hdbscan only)')
    clustering.add_argument('--k-range', nargs=2, type=int, default=[5, 25],
                           help='K search range for auto selection')

    # Semantic weighting (NEW!)
    semantic = parser.add_argument_group('Semantic Features')
    semantic.add_argument('--semantic-weight', type=float, default=0.85,
                         help='Weight for semantic features (0-1). Higher = more semantic, less spatial. Default: 0.85')
    semantic.add_argument('--compare-weights', action='store_true',
                         help='Compare multiple semantic weights (0.70, 0.80, 0.85, 0.90, 0.95)')

    # Dimensionality reduction (NEW!)
    dimred = parser.add_argument_group('Dimensionality Reduction')
    dimred.add_argument('--use-pca', action='store_true',
                       help='Apply PCA dimensionality reduction before clustering')
    dimred.add_argument('--pca-variance', type=float, default=0.95,
                       help='Variance to preserve with PCA (default: 0.95)')

    # GPU acceleration
    gpu_group = parser.add_argument_group('GPU Acceleration')
    gpu_group.add_argument('--use-gpu', action='store_true',
                          help='Use GPU acceleration via RAPIDS cuML')

    # Graph-based clustering options
    graph_group = parser.add_argument_group('Graph-Based Clustering (Spectral, Agglomerative)')
    graph_group.add_argument('--graph-neighbors', type=int, default=30,
                            help='Number of spatial neighbors for graph construction (default: 30)')
    graph_group.add_argument('--graph-spatial-weight', type=float, default=None,
                            help='Spatial weight for graph methods (default: 1 - semantic_weight)')

    # HDBSCAN
    hdbscan_group = parser.add_argument_group('HDBSCAN Options')
    hdbscan_group.add_argument('--min-cluster-size', type=int, default=500,
                              help='Minimum cluster size for HDBSCAN (default: 500)')
    hdbscan_group.add_argument('--min-samples', type=int, default=None,
                              help='Min samples for HDBSCAN (default: min_cluster_size // 10, lower = more clusters)')

    # Post-processing (SIMPLIFIED!)
    postproc = parser.add_argument_group('Post-processing')
    postproc.add_argument('--smooth', action='store_true', default=False,
                         help='Apply lightweight spatial smoothing (1 iteration)')
    postproc.add_argument('--smooth-k', type=int, default=20,
                         help='Number of spatial neighbors for smoothing')
    postproc.add_argument('--smooth-iterations', type=int, default=1,
                         help='Number of smoothing iterations')

    # Evaluation (NEW!)
    eval_group = parser.add_argument_group('Quality Evaluation')
    eval_group.add_argument('--evaluate', action='store_true',
                           help='Compute cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)')
    eval_group.add_argument('--eval-sample-size', type=int, default=10000,
                           help='Sample size for metric computation (default: 10000)')

    # Visualization
    viz = parser.add_argument_group('Visualization')
    viz.add_argument('--visualize', action='store_true',
                    help='Create visualizations')
    viz.add_argument('--viz-all', action='store_true', default=True,
                    help='All clusters view')
    viz.add_argument('--viz-panels', action='store_true', default=True,
                    help='Individual cluster panels')
    viz.add_argument('--grid-cols', type=int, default=4,
                    help='Panel grid columns')

    # I/O
    io = parser.add_argument_group('Input/Output')
    io.add_argument('--load-labels', type=Path,
                   help='Load existing labels instead of clustering')
    io.add_argument('--output-dir', type=Path,
                   default=Path(__file__).parent / 'clustering-results',
                   help='Output directory')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert n_clusters to int if not 'auto'
    if args.n_clusters != 'auto':
        args.n_clusters = int(args.n_clusters)

    # Print configuration
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Output: {args.output_dir}")
    print(f"Algorithm: {args.algorithm}")
    print(f"N Clusters: {args.n_clusters}")
    print(f"Semantic Weight: {args.semantic_weight} (higher = more semantic)")
    print(f"GPU Acceleration: {args.use_gpu and CUML_AVAILABLE}")
    print(f"PCA Reduction: {args.use_pca} (variance={args.pca_variance if args.use_pca else 'N/A'})")
    print(f"Evaluate Quality: {args.evaluate}")
    print(f"Compare Weights: {args.compare_weights}")
    print(f"Spatial Smoothing: {args.smooth}")

    if args.use_gpu and not CUML_AVAILABLE:
        print("⚠ WARNING: GPU requested but RAPIDS cuML not available, using CPU")

    # Process each dataset
    for dataset_name in args.dataset:
        try:
            process_dataset(dataset_name, args)
        except Exception as e:
            print(f"\n❌ ERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("✓ ALL PROCESSING COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
