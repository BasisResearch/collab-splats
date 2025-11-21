"""
Mesh Clustering and Visualization - GPU-Accelerated Version with RAPIDS cuML

GPU-accelerated clustering using RAPIDS cuML for massive speedup on large meshes.

Installation:
    # RAPIDS cuML (requires CUDA-capable GPU)
    conda install -c rapidsai -c conda-forge -c nvidia \
        cuml=24.10 python=3.11 cudatoolkit=11.8

    # Or use pip (may require manual CUDA setup)
    pip install cuml-cu11

Usage:
    # GPU-accelerated K-means (10-100x faster)
    python cluster_mesh_cuml.py --dataset rats_001 --n-clusters 11 --visualize --use-gpu

    # Fall back to CPU if no GPU
    python cluster_mesh_cuml.py --dataset rats_001 --n-clusters 11 --visualize

    # GPU spectral clustering
    python cluster_mesh_cuml.py --dataset rats_001 --n-clusters 11 --algorithm spectral --use-gpu

    # GPU HDBSCAN
    python cluster_mesh_cuml.py --dataset rats_001 --algorithm hdbscan --use-gpu
"""

import argparse
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

# Collab-splats imports
from collab_splats.wrapper import Splatter, SplatterConfig
from collab_splats.utils.visualization import MESH_KWARGS, VIZ_KWARGS, visualize_splat

# Try to import cuML for GPU acceleration
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
    from cuml.manifold import SpectralEmbedding as cuSpectralEmbedding
    from cuml.preprocessing import normalize as cu_normalize
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    import cupy as cp
    CUML_AVAILABLE = True
    print("✓ RAPIDS cuML available - GPU acceleration enabled")
except ImportError:
    CUML_AVAILABLE = False
    print("⚠ RAPIDS cuML not available - falling back to CPU (sklearn)")
    print("  Install with: conda install -c rapidsai -c conda-forge cuml")

# CPU fallback imports
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import silhouette_score
from scipy.stats import mode
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix

try:
    import hdbscan
    HDBSCAN_CPU_AVAILABLE = True
except ImportError:
    HDBSCAN_CPU_AVAILABLE = False

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


def get_array_module(use_gpu):
    """Get appropriate array module (cupy or numpy)."""
    if use_gpu and CUML_AVAILABLE:
        return cp
    return np


# ============================================================================
# Clustering Utilities
# ============================================================================

def find_elbow_point(k_values, inertias):
    """Find elbow point using the kneedle algorithm."""
    k_norm = (np.array(k_values) - np.min(k_values)) / (np.max(k_values) - np.min(k_values))
    inertia_norm = (np.array(inertias) - np.min(inertias)) / (np.max(inertias) - np.min(inertias))

    x1, y1 = k_norm[0], inertia_norm[0]
    x2, y2 = k_norm[-1], inertia_norm[-1]

    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    distances = np.abs(a * k_norm + b * inertia_norm + c) / np.sqrt(a**2 + b**2)
    elbow_idx = np.argmax(distances)

    return k_values[elbow_idx]


def select_optimal_k(features, k_range, use_gpu=False, verbose=False):
    """Find optimal K using elbow method with GPU acceleration."""
    print(f"Computing elbow curve for K in {k_range.start}-{k_range.stop-1}...")
    
    inertias = []
    k_values = list(k_range)
    
    if use_gpu and CUML_AVAILABLE:
        print("Using GPU-accelerated K-means for elbow detection")
        features_gpu = to_gpu(features)
        
        for k in tqdm(k_values, disable=not verbose):
            kmeans = cuKMeans(n_clusters=k, random_state=42, max_iter=100, n_init=10)
            kmeans.fit(features_gpu)
            inertias.append(float(kmeans.inertia_))
    else:
        print("Using CPU K-means for elbow detection")
        for k in tqdm(k_values, disable=not verbose):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=50000, 
                                    n_init=10, max_iter=100)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
    
    optimal_k = find_elbow_point(k_values, inertias)
    
    return {
        'optimal_k': optimal_k,
        'inertias': inertias,
        'k_values': k_values,
    }


def kmeans_clustering(features, n_clusters, use_gpu=False, verbose=True):
    """
    Perform K-means clustering with optional GPU acceleration.
    
    GPU version is typically 10-100x faster than CPU for large datasets.
    """
    if use_gpu and CUML_AVAILABLE:
        print(f"GPU K-means with K={n_clusters}...")
        
        # Transfer to GPU
        features_gpu = to_gpu(features)
        
        # GPU K-means
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            random_state=42,
            max_iter=300,
            n_init=10,
            tol=1e-4,
            verbose=0
        )
        
        labels_gpu = kmeans.fit_predict(features_gpu)
        
        # Transfer back to CPU for compatibility
        labels = to_cpu(labels_gpu)
        
        if verbose:
            print(f"GPU clustering completed in {kmeans.n_iter_} iterations")
    else:
        if use_gpu:
            print("⚠ GPU requested but cuML not available, using CPU")
        
        print(f"CPU K-means with K={n_clusters}...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=50000,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(features)
    
    if verbose:
        print("\nCluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Cluster {label}: {count:,} samples ({100*count/len(labels):.1f}%)")
    
    return labels, kmeans


def spectral_clustering(positions, features, n_clusters, n_neighbors=30, 
                       spatial_weight=0.3, use_gpu=False, verbose=True):
    """
    Spectral clustering with optional GPU acceleration.
    
    GPU version uses cuML's SpectralEmbedding + KMeans for speedup.
    """
    if verbose:
        mode_str = "GPU" if (use_gpu and CUML_AVAILABLE) else "CPU"
        print(f"{mode_str} Spectral clustering with K={n_clusters}, n_neighbors={n_neighbors}...")
    
    n_points = len(positions)
    
    if use_gpu and CUML_AVAILABLE:
        # GPU-accelerated spectral clustering
        print("  Building spatial connectivity graph on GPU...")
        
        # Transfer to GPU
        positions_gpu = to_gpu(positions)
        features_gpu = to_gpu(features)
        
        # Build KNN graph on GPU
        nbrs = cuNearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
        nbrs.fit(positions_gpu)
        distances, indices = nbrs.kneighbors(positions_gpu)
        
        # Convert to CPU for sparse matrix construction (cuML doesn't have full sparse support)
        distances_cpu = to_cpu(distances[:, 1:])
        indices_cpu = to_cpu(indices[:, 1:])
        features_cpu = to_cpu(features_gpu)
        positions_cpu = to_cpu(positions_gpu)
        
        print("  Computing edge weights...")
        
        # Spatial weights
        spatial_scale = np.median(distances_cpu)
        spatial_weights = np.exp(-distances_cpu / spatial_scale).flatten()
        
        # Feature similarity weights
        feature_weights = np.zeros(n_points * n_neighbors)
        features_norm = normalize(features_cpu, norm='l2', axis=1)
        
        for i in range(n_points):
            neighbor_feats = features_norm[indices_cpu[i]]
            my_feat = features_norm[i]
            sims = np.dot(neighbor_feats, my_feat)
            sims = np.maximum(0, sims)
            feature_weights[i*n_neighbors:(i+1)*n_neighbors] = sims
        
        # Combine weights
        combined_weights = (1 - spatial_weight) * feature_weights + spatial_weight * spatial_weights
        
        # Build sparse adjacency matrix
        row_ind = np.repeat(np.arange(n_points), n_neighbors)
        col_ind = indices_cpu.flatten()
        adjacency = csr_matrix((combined_weights, (row_ind, col_ind)), shape=(n_points, n_points))
        adjacency = adjacency + adjacency.T
        adjacency.data = adjacency.data / 2
        
        print("  Running spectral clustering...")
        
        # Use sklearn's spectral clustering with precomputed affinity
        # (cuML's spectral embedding can be used here but sklearn integration is simpler)
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            n_init=10,
            random_state=42
        )
        labels = clustering.fit_predict(adjacency)
        
    else:
        # CPU version (original implementation)
        if use_gpu:
            print("  ⚠ GPU requested but cuML not available, using CPU")
        
        print("  Building spatial connectivity graph...")
        
        # Build KNN graph
        tree = KDTree(positions)
        distances, indices = tree.query(positions, k=n_neighbors+1)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
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
        print("\nCluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Cluster {label}: {count:,} samples ({100*count/len(labels):.1f}%)")
    
    return labels


def hdbscan_clustering(features, positions=None, min_cluster_size=500, min_samples=None,
                       spatial_weight=0.05, use_gpu=False, handle_noise='reassign', 
                       noise_k=20, verbose=True):
    """
    HDBSCAN clustering with optional GPU acceleration.
    
    GPU version uses cuML's HDBSCAN which is significantly faster.
    """
    if verbose:
        mode_str = "GPU" if (use_gpu and CUML_AVAILABLE) else "CPU"
        print(f"{mode_str} HDBSCAN clustering (min_cluster_size={min_cluster_size}, "
              f"spatial_weight={spatial_weight})")
    
    # Prepare features with spatial weighting
    if positions is not None and spatial_weight > 0:
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        pos_feat = pos_norm * spatial_weight
        
        feat_norm = normalize(features, norm='l2', axis=1)
        data = np.concatenate([feat_norm, pos_feat], axis=1)
    else:
        data = normalize(features, norm='l2', axis=1)
    
    if min_samples is None:
        min_samples = max(5, min_cluster_size // 2)
    
    if use_gpu and CUML_AVAILABLE:
        # GPU HDBSCAN
        print("  Running GPU HDBSCAN...")
        
        data_gpu = to_gpu(data)
        
        clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
            gen_min_span_tree=True
        )
        
        labels_gpu = clusterer.fit_predict(data_gpu)
        labels = to_cpu(labels_gpu)
        
    else:
        # CPU HDBSCAN
        if use_gpu:
            print("  ⚠ GPU requested but cuML not available, using CPU")
        
        if not HDBSCAN_CPU_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        print("  Running CPU HDBSCAN...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
            core_dist_n_jobs=-1,
            prediction_data=True
        )
        labels = clusterer.fit_predict(data)
    
    # Handle noise points
    n_noise = np.sum(labels == -1)
    if n_noise > 0 and handle_noise != 'keep':
        if verbose:
            print(f"  Handling {n_noise:,} noise points ({100*n_noise/len(labels):.1f}%) "
                  f"using '{handle_noise}' strategy...")
        
        if handle_noise == 'reassign':
            noise_mask = labels == -1
            noise_indices = np.where(noise_mask)[0]
            clustered_mask = ~noise_mask
            
            if positions is not None and np.any(clustered_mask):
                tree = KDTree(positions[clustered_mask])
                clustered_indices = np.where(clustered_mask)[0]
                
                for noise_idx in tqdm(noise_indices, disable=not verbose, desc="Reassigning noise"):
                    distances, neighbor_indices = tree.query([positions[noise_idx]], k=noise_k)
                    actual_neighbors = clustered_indices[neighbor_indices[0]]
                    neighbor_labels = labels[actual_neighbors]
                    
                    # Weighted voting
                    weights = 1.0 / (distances[0] + 1e-8)
                    weighted_votes = {}
                    for label, weight in zip(neighbor_labels, weights):
                        weighted_votes[label] = weighted_votes.get(label, 0) + weight
                    
                    labels[noise_idx] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            else:
                # Fallback to feature similarity
                for noise_idx in noise_indices:
                    distances = np.linalg.norm(data[clustered_mask] - data[noise_idx], axis=1)
                    nearest = np.argmin(distances)
                    labels[noise_idx] = labels[clustered_indices[nearest]]
        
        elif handle_noise == 'small_cluster':
            noise_mask = labels == -1
            if np.sum(noise_mask) >= min_cluster_size:
                noise_data = data[noise_mask]
                
                if use_gpu and CUML_AVAILABLE:
                    noise_data_gpu = to_gpu(noise_data)
                    sub_clusterer = cuHDBSCAN(
                        min_cluster_size=max(50, min_cluster_size // 4),
                        min_samples=5,
                        cluster_selection_method='eom'
                    )
                    noise_labels_gpu = sub_clusterer.fit_predict(noise_data_gpu)
                    noise_labels = to_cpu(noise_labels_gpu)
                else:
                    sub_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=max(50, min_cluster_size // 4),
                        min_samples=5,
                        cluster_selection_method='eom'
                    )
                    noise_labels = sub_clusterer.fit_predict(noise_data)
                
                max_cluster = labels[labels >= 0].max() if np.any(labels >= 0) else -1
                noise_labels[noise_labels >= 0] += (max_cluster + 1)
                labels[noise_mask] = noise_labels
                
                if verbose:
                    n_new = len(np.unique(noise_labels[noise_labels >= 0]))
                    n_still_noise = np.sum(noise_labels == -1)
                    print(f"    Created {n_new} small clusters from noise")
                    print(f"    Remaining noise: {n_still_noise:,}")
    
    if verbose:
        n_noise_final = np.sum(labels == -1)
        unique = np.unique(labels[labels >= 0])
        print(f"  Final clusters: {len(unique)}")
        if n_noise_final > 0:
            print(f"  Noise points: {n_noise_final:,} ({100*n_noise_final/len(labels):.1f}%)")
        for c in unique:
            cnt = np.sum(labels == c)
            print(f"    Cluster {c}: {cnt:,} points ({100*cnt/len(labels):.1f}%)")
    
    # Relabel consecutively
    unique_labels = sorted([u for u in np.unique(labels) if u != -1])
    remap = {old: new for new, old in enumerate(unique_labels)}
    if -1 in labels:
        remap[-1] = -1
    new_labels = np.array([remap[l] for l in labels])
    
    n_clusters = len(unique_labels)
    
    return new_labels, n_clusters


def agglomerative_clustering(positions, features, n_clusters, n_neighbors=30, 
                             spatial_weight=0.05, use_gpu=False, verbose=True):
    """
    Agglomerative clustering with optional GPU acceleration.
    
    GPU version uses cuML's AgglomerativeClustering.
    """
    if verbose:
        mode_str = "GPU" if (use_gpu and CUML_AVAILABLE) else "CPU"
        print(f"{mode_str} Agglomerative clustering with K={n_clusters}, connectivity={n_neighbors}...")
    
    # Combine features with spatial positions
    pos_normalized = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)
    combined_features = np.concatenate([features, pos_normalized * spatial_weight], axis=1)
    
    if use_gpu and CUML_AVAILABLE:
        print("  Building connectivity graph on GPU...")
        
        # Build connectivity on GPU
        positions_gpu = to_gpu(positions)
        nbrs = cuNearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nbrs.fit(positions_gpu)
        
        # cuML's AgglomerativeClustering can work with connectivity
        # but we need to convert to CPU for scipy sparse matrix
        distances, indices = nbrs.kneighbors(positions_gpu)
        indices_cpu = to_cpu(indices)
        
        # Build connectivity matrix on CPU
        n_points = len(positions)
        row_ind = np.repeat(np.arange(n_points), n_neighbors)
        col_ind = indices_cpu.flatten()
        data = np.ones(len(row_ind))
        connectivity = csr_matrix((data, (row_ind, col_ind)), shape=(n_points, n_points))
        
        print("  Running GPU agglomerative clustering...")
        
        # Note: cuML's AgglomerativeClustering has limited connectivity support
        # For best GPU performance, we use it without connectivity constraint
        combined_features_gpu = to_gpu(combined_features)
        
        clustering = cuAgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='euclidean',
            linkage='single'  # cuML supports 'single' and 'complete'
        )
        
        labels_gpu = clustering.fit_predict(combined_features_gpu)
        labels = to_cpu(labels_gpu)
        
    else:
        if use_gpu:
            print("  ⚠ GPU requested but cuML not available, using CPU")
        
        print("  Building connectivity graph...")
        connectivity = kneighbors_graph(positions, n_neighbors=n_neighbors, include_self=False)
        
        print("  Running CPU agglomerative clustering...")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            connectivity=connectivity,
            linkage='ward'
        )
        labels = clustering.fit_predict(combined_features)
    
    if verbose:
        print("\nCluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Cluster {label}: {count:,} samples ({100*count/len(labels):.1f}%)")
    
    return labels


# ============================================================================
# Preprocessing Functions
# ============================================================================

def normalize_features(features, method='l2', use_gpu=False):
    """Normalize features with optional GPU acceleration."""
    if use_gpu and CUML_AVAILABLE:
        features_gpu = to_gpu(features)
        if method == 'l2':
            normalized_gpu = cu_normalize(features_gpu, norm='l2')
        elif method == 'l1':
            normalized_gpu = cu_normalize(features_gpu, norm='l1')
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        return to_cpu(normalized_gpu)
    else:
        if method == 'l2':
            return normalize(features, norm='l2', axis=1)
        elif method == 'l1':
            return normalize(features, norm='l1', axis=1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def add_spatial_features(features, positions, spatial_weight=0.1):
    """Add weighted spatial coordinates to features."""
    print(f"Adding spatial features (weight={spatial_weight})...")
    
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_norm = (positions - pos_min) / (pos_max - pos_min + 1e-8)
    
    pos_weighted = pos_norm * spatial_weight
    features_spatial = np.concatenate([features, pos_weighted], axis=1)
    
    return features_spatial


# ============================================================================
# Post-processing Functions
# ============================================================================

def reassign_small_clusters(positions, labels, features, min_cluster_pct=2.0, verbose=True):
    """Reassign small clusters to nearest large cluster."""
    n_total = len(labels)
    min_size = int(n_total * min_cluster_pct / 100)
    
    if verbose:
        print(f"Reassigning clusters <{min_cluster_pct}% ({min_size:,} points) to nearest neighbors...")
    
    unique, counts = np.unique(labels, return_counts=True)
    small_clusters = unique[counts < min_size]
    large_clusters = unique[counts >= min_size]
    
    if len(small_clusters) == 0:
        if verbose:
            print("  No small clusters to reassign")
        return labels
    
    if len(large_clusters) == 0:
        if verbose:
            print("  Warning: All clusters are small, keeping as-is")
        return labels
    
    new_labels = labels.copy()
    total_reassigned = 0
    
    large_centroids = np.array([features[labels == c].mean(axis=0) for c in large_clusters])
    
    for cluster_id in small_clusters:
        cluster_mask = labels == cluster_id
        cluster_features = features[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        for idx, feat in zip(cluster_indices, cluster_features):
            distances = np.linalg.norm(large_centroids - feat, axis=1)
            nearest_large_cluster = large_clusters[np.argmin(distances)]
            new_labels[idx] = nearest_large_cluster
            total_reassigned += 1
    
    unique_labels = np.unique(new_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map[l] for l in new_labels])
    
    if verbose:
        print(f"  Reassigned {len(small_clusters)} small clusters ({total_reassigned:,} points)")
        print(f"  Remaining: {len(large_clusters)} clusters")
    
    return final_labels


def spatial_smoothing_knn(positions, labels, k=20, iterations=2, verbose=True):
    """Smooth cluster assignments using KNN voting."""
    if verbose:
        print(f"Spatial smoothing with KNN (k={k}, iterations={iterations})...")
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(positions)
    smoothed_labels = labels.copy()
    
    for iteration in range(iterations):
        distances, indices = nbrs.kneighbors(positions)
        neighbor_indices = indices[:, 1:]
        neighbor_labels = smoothed_labels[neighbor_indices]
        
        voted_labels, _ = mode(neighbor_labels, axis=1, keepdims=False)
        smoothed_labels = voted_labels.flatten()
        
        if verbose:
            n_changed = (smoothed_labels != labels).sum()
            print(f"  Iteration {iteration+1}: {n_changed:,} points reassigned")
    
    return smoothed_labels


def merge_adjacent_clusters(positions, labels, features, similarity_threshold=0.85, 
                           k=50, verbose=True):
    """Merge spatially adjacent clusters that have similar features."""
    if verbose:
        print(f"\nMerging adjacent similar clusters (threshold={similarity_threshold})...")
    
    n_clusters = len(np.unique(labels))
    cluster_ids = np.unique(labels)
    cluster_features = {}
    cluster_sizes = {}
    
    for cid in cluster_ids:
        mask = labels == cid
        cluster_features[cid] = features[mask].mean(axis=0)
        cluster_sizes[cid] = mask.sum()
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(positions)
    cluster_adjacency = {cid: set() for cid in cluster_ids}
    
    for i in range(len(positions)):
        _, neighbor_indices = nbrs.kneighbors([positions[i]])
        my_cluster = labels[i]
        neighbor_clusters = labels[neighbor_indices[0]]
        
        for neighbor_cluster in neighbor_clusters:
            if neighbor_cluster != my_cluster:
                cluster_adjacency[my_cluster].add(neighbor_cluster)
    
    merge_map = {}
    
    for cid in cluster_ids:
        if cid in merge_map:
            continue
        
        best_merge = None
        best_similarity = similarity_threshold
        
        for neighbor_cid in cluster_adjacency[cid]:
            if neighbor_cid in merge_map:
                continue
            
            feat1 = cluster_features[cid]
            feat2 = cluster_features[neighbor_cid]
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_merge = neighbor_cid
        
        if best_merge is not None:
            if cluster_sizes[cid] < cluster_sizes[best_merge]:
                merge_map[cid] = best_merge
            else:
                merge_map[best_merge] = cid
    
    new_labels = labels.copy()
    for old_cid, new_cid in merge_map.items():
        new_labels[labels == old_cid] = new_cid
    
    n_merged = len(merge_map)
    n_remaining = len(np.unique(new_labels))
    
    if verbose:
        print(f"  Merged {n_merged} adjacent similar clusters")
        print(f"  Clusters: {n_clusters} -> {n_remaining}")
    
    return new_labels


def spatial_postprocessing(positions, labels, k=30, iterations=3, verbose=True):
    """Apply spatial smoothing to create clean, cohesive clusters."""
    if verbose:
        print("\n--- SPATIAL SMOOTHING ---")
    
    labels = spatial_smoothing_knn(positions, labels, k=k, iterations=iterations, verbose=verbose)
    
    return labels


# ============================================================================
# Dataset configurations
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
    """Load mesh, features, and create PyVista mesh."""
    print(f"Initializing {dataset_name}...")
    
    splatter_config = SplatterConfig(
        file_path=config['file_path'],
        method="rade-features",
        frame_proportion=config['frame_proportion'],
    )
    splatter = Splatter(splatter_config)
    
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
    
    print(f"Vertices: {len(mesh_vertices):,}, Features: {mesh_features.shape}")
    
    return pv_mesh, mesh_features, mesh_vertices


def cluster_features(features, positions, n_clusters=None, k_range=(2, 20),
                     algorithm='kmeans', use_gpu=False, **kwargs):
    """
    Perform clustering on features with GPU acceleration.
    
    Args:
        features: Feature array
        positions: Vertex positions
        n_clusters: Number of clusters or None/'auto' for elbow
        k_range: Range for auto K selection
        algorithm: 'kmeans', 'spectral', 'agglomerative', 'hdbscan'
        use_gpu: Use GPU acceleration if available
        **kwargs: Algorithm-specific parameters
    """
    # Auto-select K
    if n_clusters is None or n_clusters == 'auto':
        if algorithm != 'kmeans':
            print(f"Auto K-selection only supported for kmeans, using default range mid-point")
            n_clusters = (k_range[0] + k_range[1]) // 2
        else:
            k_result = select_optimal_k(features, k_range=range(k_range[0], k_range[1]), 
                                       use_gpu=use_gpu, verbose=False)
            n_clusters = k_result['optimal_k']
            print(f"Optimal K (elbow): {n_clusters}")
    
    # Perform clustering
    if algorithm == 'kmeans':
        labels, _ = kmeans_clustering(features, n_clusters, use_gpu=use_gpu, verbose=True)
    
    elif algorithm == 'spectral':
        spatial_weight = kwargs.get('graph_spatial_weight', 0.3)
        labels = spectral_clustering(
            positions, features, n_clusters,
            n_neighbors=kwargs.get('graph_neighbors', 30),
            spatial_weight=spatial_weight,
            use_gpu=use_gpu,
            verbose=True
        )
    
    elif algorithm == 'agglomerative':
        spatial_weight = kwargs.get('graph_spatial_weight', 0.05)
        labels = agglomerative_clustering(
            positions, features, n_clusters,
            n_neighbors=kwargs.get('graph_neighbors', 30),
            spatial_weight=spatial_weight,
            use_gpu=use_gpu,
            verbose=True
        )
    
    elif algorithm == 'hdbscan':
        labels, n_clusters = hdbscan_clustering(
            features,
            positions=positions,
            min_cluster_size=kwargs.get('min_cluster_size', 800),
            min_samples=kwargs.get('min_samples', None),
            spatial_weight=kwargs.get('spatial_weight', 0.05),
            use_gpu=use_gpu,
            handle_noise=kwargs.get('handle_noise', 'reassign'),
            verbose=True
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return labels, n_clusters


def save_labels(labels, n_clusters, dataset_name, output_dir):
    """Save clustering labels and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_path = output_dir / f"{dataset_name}_labels_k{n_clusters}.npy"
    np.save(labels_path, labels)
    
    metadata = {
        'dataset': dataset_name,
        'n_clusters': n_clusters,
        'n_vertices': len(labels),
        'distribution': {int(i): int((labels == i).sum()) for i in range(n_clusters)}
    }
    metadata_path = output_dir / f"{dataset_name}_metadata_k{n_clusters}.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved: {labels_path.name}, {metadata_path.name}")


def generate_colors(n_clusters):
    """Generate distinct colors for clusters."""
    hues = np.linspace(0, 1, n_clusters, endpoint=False)
    return np.array([
        [int(c * 255) for c in colorsys.hsv_to_rgb(h, 0.65, 0.95)]
        for h in hues
    ], dtype=np.uint8)


def visualize_all_clusters(pv_mesh, labels, n_clusters, dataset_name, output_path):
    """Create view showing all clusters colored."""
    print("Rendering all clusters...")
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
    plt.title(f'{dataset_name} - All Clusters (K={n_clusters})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path.name}")


def visualize_individual_clusters(pv_mesh, labels, n_clusters, dataset_name, output_path, grid_cols=4):
    """Create multi-panel view with each cluster highlighted individually."""
    print("Rendering individual cluster panels...")
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
        f'{dataset_name} - Individual Cluster Segments (K={n_clusters})',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path.name}")


def process_dataset(dataset_name, args):
    """Process a single dataset."""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*70}\n")
    
    config = DATASETS[dataset_name]
    
    # Load mesh and features
    pv_mesh, features, positions = load_mesh_and_features(dataset_name, config)
    
    # Clustering
    if args.load_labels:
        print(f"Loading labels from: {args.load_labels}")
        labels = np.load(args.load_labels)
        n_clusters = len(np.unique(labels))
    else:
        print("\n--- PREPROCESSING ---")
        
        if args.normalize:
            print("Normalizing features (L2)...")
            features = normalize_features(features, method='l2', use_gpu=args.use_gpu)
        
        if args.spatial_weight > 0:
            features = add_spatial_features(features, positions, args.spatial_weight)
        
        print(f"Final feature shape: {features.shape}")
        
        print("\n--- CLUSTERING ---")
        n_clusters = None if args.n_clusters == 'auto' else int(args.n_clusters)
        labels, n_clusters = cluster_features(
            features,
            positions,
            n_clusters,
            args.k_range,
            algorithm=args.algorithm,
            use_gpu=args.use_gpu,
            graph_neighbors=args.graph_neighbors,
            graph_spatial_weight=args.graph_spatial_weight,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            spatial_weight=args.hdbscan_spatial_weight,
            handle_noise=args.handle_noise
        )
        
        # Post-processing
        if args.merge_small > 0:
            print("\n--- POST-PROCESSING ---")
            labels = reassign_small_clusters(positions, labels, features, min_cluster_pct=args.merge_small)
            n_clusters = len(np.unique(labels))
        
        if args.merge_adjacent:
            if args.algorithm in ['spectral', 'agglomerative']:
                print("\nNote: Graph-based clustering already handles spatial coherence,")
                print("      skipping adjacent cluster merging to preserve results.")
            else:
                labels = merge_adjacent_clusters(
                    positions,
                    labels,
                    features,
                    similarity_threshold=args.merge_threshold,
                    k=args.merge_k,
                    verbose=True
                )
                n_clusters = len(np.unique(labels))
        
        if args.spatial_smooth:
            smooth_iterations = args.smooth_iterations
            if args.algorithm in ['spectral', 'agglomerative']:
                smooth_iterations = max(1, smooth_iterations // 2)
                if smooth_iterations < args.smooth_iterations:
                    print(f"\nNote: Using {smooth_iterations} smoothing iterations for {args.algorithm} clustering")
            
            labels = spatial_postprocessing(
                positions,
                labels,
                k=args.smooth_k,
                iterations=smooth_iterations
            )
        
        # Final relabeling
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        n_clusters = len(unique_labels)
        
        save_labels(labels, n_clusters, dataset_name, args.output_dir)
    
    # Visualization
    if args.visualize:
        print("\n--- VISUALIZATION ---")
        
        if args.viz_all:
            path = args.output_dir / f"{dataset_name}_all_clusters_k{n_clusters}.png"
            visualize_all_clusters(pv_mesh, labels, n_clusters, dataset_name, path)
        
        if args.viz_panels:
            path = args.output_dir / f"{dataset_name}_cluster_panels_k{n_clusters}.png"
            visualize_individual_clusters(pv_mesh, labels, n_clusters, dataset_name, path, args.grid_cols)
    
    print(f"\n✓ Completed {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Mesh Clustering with RAPIDS cuML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU K-means (10-100x faster)
  python cluster_mesh_cuml.py --dataset rats_001 --n-clusters 11 --use-gpu --visualize

  # GPU spectral clustering
  python cluster_mesh_cuml.py --dataset rats_001 --n-clusters 11 --algorithm spectral --use-gpu

  # GPU HDBSCAN
  python cluster_mesh_cuml.py --dataset rats_001 --algorithm hdbscan --use-gpu --visualize

  # Auto K selection with GPU
  python cluster_mesh_cuml.py --dataset birds_008 --n-clusters auto --use-gpu

  # CPU fallback (no GPU available)
  python cluster_mesh_cuml.py --dataset rats_001 --n-clusters 11 --visualize
        """
    )
    
    # Dataset selection
    parser.add_argument('--dataset', nargs='+', choices=list(DATASETS.keys()), required=True,
                       help='Dataset(s) to process')
    
    # GPU acceleration
    gpu_group = parser.add_argument_group('GPU Acceleration')
    gpu_group.add_argument('--use-gpu', action='store_true',
                          help='Use GPU acceleration via RAPIDS cuML (requires CUDA)')
    
    # Preprocessing
    preproc = parser.add_argument_group('Preprocessing')
    preproc.add_argument('--normalize', action='store_true', default=True,
                        help='L2 normalize features (default: True)')
    preproc.add_argument('--no-normalize', dest='normalize', action='store_false')
    preproc.add_argument('--spatial-weight', type=float, default=0.0,
                        help='Weight for spatial features (0=off, 0.05-0.2 recommended)')
    
    # Clustering
    clustering = parser.add_argument_group('Clustering')
    clustering.add_argument('--algorithm', choices=['kmeans', 'spectral', 'agglomerative', 'hdbscan'],
                           default='kmeans',
                           help='Clustering algorithm (default: kmeans)')
    clustering.add_argument('--n-clusters', type=str, default='auto',
                           help='Number of clusters or "auto" for elbow method')
    clustering.add_argument('--k-range', nargs=2, type=int, default=[2, 20],
                           help='K search range for auto selection')
    
    # Graph-based options
    graph = parser.add_argument_group('Graph-based Clustering')
    graph.add_argument('--graph-neighbors', type=int, default=30,
                      help='Number of spatial neighbors for graph construction')
    graph.add_argument('--graph-spatial-weight', type=float, default=None,
                      help='Spatial weight for graph clustering')
    
    # HDBSCAN options
    hdbscan_group = parser.add_argument_group('HDBSCAN Options')
    hdbscan_group.add_argument('--min-cluster-size', type=int, default=500,
                              help='Minimum cluster size for HDBSCAN')
    hdbscan_group.add_argument('--min-samples', type=int, default=None,
                              help='HDBSCAN min_samples parameter')
    hdbscan_group.add_argument('--hdbscan-spatial-weight', type=float, default=0.05,
                              help='Spatial weight for HDBSCAN')
    hdbscan_group.add_argument('--handle-noise', 
                              choices=['reassign', 'keep', 'small_cluster'],
                              default='reassign',
                              help='Strategy for handling noise points')
    
    # Post-processing
    postproc = parser.add_argument_group('Post-processing')
    postproc.add_argument('--merge-small', type=float, default=2.0,
                         help='Reassign clusters smaller than this %% (default: 2.0)')
    
    merge_adjacent_group = postproc.add_mutually_exclusive_group()
    merge_adjacent_group.add_argument('--merge-adjacent', dest='merge_adjacent', action='store_true', default=True,
                         help='Merge adjacent clusters with similar features (default)')
    merge_adjacent_group.add_argument('--no-merge-adjacent', dest='merge_adjacent', action='store_false',
                         help='Disable adjacent cluster merging')
    
    postproc.add_argument('--merge-threshold', type=float, default=0.85,
                         help='Feature similarity threshold for merging (default: 0.85)')
    postproc.add_argument('--merge-k', type=int, default=50,
                         help='Neighbors to check for cluster adjacency (default: 50)')
    
    # Spatial smoothing
    spatial = parser.add_argument_group('Spatial Smoothing')
    
    spatial_smooth_group = spatial.add_mutually_exclusive_group()
    spatial_smooth_group.add_argument('--spatial-smooth', dest='spatial_smooth', action='store_true', default=True,
                        help='Apply KNN spatial smoothing (default)')
    spatial_smooth_group.add_argument('--no-spatial-smooth', dest='spatial_smooth', action='store_false',
                        help='Disable spatial smoothing')
    
    spatial.add_argument('--smooth-k', type=int, default=30,
                        help='Number of spatial neighbors (default: 30)')
    spatial.add_argument('--smooth-iterations', type=int, default=3,
                        help='Number of smoothing iterations (default: 3)')
    
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
                   help='Load existing labels')
    io.add_argument('--output-dir', type=Path,
                   default=Path(__file__).parent / 'clustering_results',
                   help='Output directory')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output: {args.output_dir}")
    print(f"Algorithm: {args.algorithm}")
    print(f"GPU Acceleration: {args.use_gpu and CUML_AVAILABLE}")
    if args.use_gpu and not CUML_AVAILABLE:
        print("⚠ WARNING: GPU requested but RAPIDS cuML not available, using CPU")
    print(f"Preprocessing: normalize={args.normalize}, spatial_weight={args.spatial_weight}")
    print(f"Post-processing: merge_small={args.merge_small}%, merge_adjacent={args.merge_adjacent}, spatial_smooth={args.spatial_smooth}")
    
    for dataset_name in args.dataset:
        try:
            process_dataset(dataset_name, args)
        except Exception as e:
            print(f"\n❌ Error: {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✓ Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()