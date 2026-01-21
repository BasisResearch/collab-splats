#!/usr/bin/env python3
"""
Semantic Dictionary Segmentation Script

This script performs semantic segmentation using a dictionary of terms with positive queries.
Each term's queries are contrasted against negative queries from other dictionary items to
segment the environment into semantic compartments.

Usage:
    python run_semantic_dictionary.py --config CONFIG_FILE [--dict DICT_FILE] [--threshold FLOAT] [--radius FLOAT]
"""

import argparse
import copy
import json
import math
import numpy as np
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from collab_splats.wrapper import Splatter
from collab_splats.utils.mesh import mesh_clustering
from collab_splats.utils.visualization import visualize_splat

# Start xvfb for headless rendering (needed for screenshot generation)
try:
    pv.start_xvfb()
    print("Started xvfb for headless rendering")
except Exception as e:
    print(f"Warning: Could not start xvfb: {e}")
    print("Screenshot generation may fail - will save PLY meshes only")


# Default semantic dictionary
DEFAULT_SEMANTIC_DICTIONARY = {
    'tree': ['green', 'leaves', 'bark', 'trunk'],
    'feeder': ['bird feeder', 'container', 'food'],
    'brush': ['leaves', 'plants', 'thicket', 'bramble'],
    'ground': ['dirt', 'gravel', 'rock', 'concrete'],
}


def load_semantic_dictionary(dict_path=None):
    """Load semantic dictionary from JSON file or use default."""
    if dict_path is None:
        return DEFAULT_SEMANTIC_DICTIONARY

    with open(dict_path, 'r') as f:
        return json.load(f)


def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap('hsv')
        return [cmap(i / n) for i in range(n)]


def query_semantic_categories(splatter, semantic_dictionary, query_dir):
    """Query mesh for each semantic category."""
    semantic_results = {}

    for category, positive_queries in semantic_dictionary.items():
        # Gather negative queries from all other categories
        negative_queries = []
        positive_queries_expanded = positive_queries + [category]

        for other_category in semantic_dictionary.keys():
            if other_category != category:
                negative_queries.append(other_category)

        print(f"\nQuerying '{category}'...")
        print(f"  Positive: {positive_queries_expanded}")
        print(f"  Negative: {len(negative_queries)} terms from other categories")

        similarity = splatter.query_mesh(
            positive_queries=positive_queries_expanded,
            negative_queries=negative_queries,
            output_fn=str(query_dir / f"query-{category}.ply")
        )

        semantic_results[category] = similarity
        print(f"  Done! Saved to {query_dir / f'query-{category}.ply'}")

    return semantic_results


def cluster_semantic_categories(semantic_dictionary, query_dir, similarity_threshold=0.95, spatial_radius=0.02):
    """Cluster all semantic categories and aggregate results."""
    all_clusters = {}
    cluster_counter = 0

    for category in semantic_dictionary.keys():
        print(f"\nProcessing '{category}'...")

        # Load query mesh
        query_mesh_path = query_dir / f"query-{category}.ply"
        mesh_cat = o3d.io.read_triangle_mesh(str(query_mesh_path))
        similarity = np.asarray(mesh_cat.vertex_colors)[:, 0]

        # Cluster
        clusters = mesh_clustering(
            mesh=mesh_cat,
            similarity_values=similarity,
            similarity_threshold=similarity_threshold,
            spatial_radius=spatial_radius,
        )

        print(f"  Found {len(clusters)} clusters")

        # Store clusters with global cluster IDs
        all_clusters[category] = []
        for cluster_vertices in clusters:
            all_clusters[category].append((cluster_counter, np.array(cluster_vertices)))
            print(f"    Cluster {cluster_counter}: {len(cluster_vertices)} vertices")
            cluster_counter += 1

    print(f"\n\nTotal clusters across all categories: {cluster_counter}")
    return all_clusters, cluster_counter


def create_categorical_mesh(all_clusters, semantic_dictionary, query_dir, cluster_dir, cluster_counter):
    """Create categorical mesh with cluster labels."""
    # Load the base mesh from query directory
    base_category = list(semantic_dictionary.keys())[0]
    base_mesh_path = query_dir / f"query-{base_category}.ply"
    categorical_mesh = o3d.io.read_triangle_mesh(str(base_mesh_path))

    # Create cluster label array (-1 means no cluster assigned)
    num_vertices = len(categorical_mesh.vertices)
    cluster_labels = -np.ones(num_vertices, dtype=np.int32)

    # Assign cluster labels
    for category_id, (category, category_clusters) in enumerate(all_clusters.items()):
        for cluster_id, vertex_indices in category_clusters:
            cluster_labels[vertex_indices] = category_id

    # Count clustered vertices
    num_clustered = (cluster_labels >= 0).sum()
    print(f"\nAssigned {num_clustered}/{num_vertices} vertices to clusters")
    if num_vertices > 0:
        print(f"Cluster labels range: {cluster_labels.min()} to {cluster_labels.max()}")
    else:
        print("Warning: No vertices found in mesh")

    # Save cluster labels to cluster directory
    np.save(cluster_dir / "cluster_labels.npy", cluster_labels)
    print(f"Saved cluster labels to {cluster_dir / 'cluster_labels.npy'}")

    return categorical_mesh, cluster_labels, num_vertices, num_clustered


def create_combined_mesh_visualization(categorical_mesh, cluster_labels, all_clusters,
                                       num_vertices, cluster_counter, cluster_dir):
    """Create and save visualization of all clusters combined."""
    # Generate colors for categories (superordinate terms), not individual clusters
    n_categories = len(all_clusters.items())
    cluster_colors = generate_distinct_colors(n_categories)
    print(f"\nGenerated {len(cluster_colors)} distinct colors for {n_categories} categories")

    # Create mesh with all clusters colored by category
    all_clusters_mesh = copy.deepcopy(categorical_mesh)
    colors_all = np.zeros((num_vertices, 3))

    # Color each vertex by its category (superordinate term)
    # cluster_labels contains category IDs (0-3), so we use those directly
    for vertex_idx in range(num_vertices):
        category_id = cluster_labels[vertex_idx]
        if category_id >= 0:
            colors_all[vertex_idx] = cluster_colors[category_id][:3]
        else:
            colors_all[vertex_idx] = [0, 0, 0]

    all_clusters_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_all)

    # Save the all-clusters mesh to cluster directory
    all_clusters_path = cluster_dir / "all_clusters.ply"
    o3d.io.write_triangle_mesh(str(all_clusters_path), all_clusters_mesh)
    print(f"Saved all-clusters mesh to {all_clusters_path}")

    return cluster_colors, colors_all


def create_individual_cluster_views(categorical_mesh, all_clusters, cluster_labels,
                                    cluster_colors, num_vertices, cluster_dir):
    """Create visualization images for each semantic category."""
    n_categories = len(all_clusters)
    print(f"\nCreating category views for {n_categories} semantic categories...")

    if n_categories == 0:
        print("No categories to visualize")
        return []

    try:
        # Convert Open3D mesh to PyVista mesh once
        vertices = np.asarray(categorical_mesh.vertices)
        faces = np.asarray(categorical_mesh.triangles)
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()

        # Create category images directory
        cluster_images_dir = cluster_dir / "cluster_views"
        cluster_images_dir.mkdir(exist_ok=True)

        # Create one image per category (showing all clusters of that category)
        for category_id, (category, category_clusters) in enumerate(all_clusters.items()):
            # Create color array: black background, all clusters of this category in category color
            colors_single = np.zeros((num_vertices, 3))

            # Color all vertices that belong to this category
            for cluster_id, vertex_indices in category_clusters:
                colors_single[vertex_indices] = cluster_colors[category_id][:3]

            # Create mesh with category colors
            pv_mesh = pv.PolyData(vertices, faces_pv)
            pv_mesh['RGB'] = colors_single

            # Save category view
            output_image_path = cluster_images_dir / f"category_{category}.png"

            visualize_splat(
                mesh=pv_mesh,
                mesh_kwargs={"scalars": "RGB", "rgb": True},
                viz_kwargs={"window_size": [1200, 1200], "off_screen": True},
                out_fn=str(output_image_path)
            )

            print(f"  Saved category view: {category} ({len(category_clusters)} clusters)")

        print(f"Successfully saved all {n_categories} category views to {cluster_images_dir}")
    except Exception as e:
        print(f"Warning: Could not create category view PNGs: {e}")
        print(f"Error details: {str(e)}")
        print("PLY meshes are still saved successfully")

    return list(all_clusters.keys())


def create_rgb_mesh_view(splatter, cluster_dir):
    """Create and save a view of the original RGB mesh (fitted splat)."""
    try:
        print("\nCreating RGB mesh screenshot (original fitted splat)...")

        # Load the original mesh with RGB colors
        mesh_path = splatter.config["mesh_info"]["mesh"]
        output_image_path = cluster_dir / "rgb_mesh_view.png"

        visualize_splat(
            mesh=str(mesh_path),
            mesh_kwargs={"scalars": "RGB", "rgb": True},
            viz_kwargs={"window_size": [2400, 2400], "off_screen": True},
            out_fn=str(output_image_path)
        )

        print(f"Saved RGB mesh view to {output_image_path}")
    except Exception as e:
        print(f"Warning: Could not create RGB mesh view PNG: {e}")
        print(f"Error details: {str(e)}")
        print("This is likely due to OpenGL/display issues in headless environment")


def create_combined_view(categorical_mesh, colors_all, cluster_dir):
    """Create and save a single combined view of all clusters using visualize_splat."""
    try:
        print("\nCreating combined semantic view screenshot...")

        # Convert Open3D mesh to PyVista mesh
        vertices = np.asarray(categorical_mesh.vertices)
        faces = np.asarray(categorical_mesh.triangles)
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
        pv_mesh = pv.PolyData(vertices, faces_pv)
        pv_mesh['RGB'] = colors_all

        # Save as image using visualize_splat to cluster directory
        output_image_path = cluster_dir / "semantic_clusters_view.png"

        visualize_splat(
            mesh=pv_mesh,
            mesh_kwargs={"scalars": "RGB", "rgb": True},
            viz_kwargs={"window_size": [2400, 2400], "off_screen": True},
            out_fn=str(output_image_path)
        )

        print(f"Saved semantic clusters view to {output_image_path}")
    except Exception as e:
        print(f"Warning: Could not create semantic clusters view PNG: {e}")
        print(f"Error details: {str(e)}")
        print("This is likely due to OpenGL/display issues in headless environment")
        print("PLY meshes are still saved successfully")


def print_cluster_summary(all_clusters, cluster_colors, num_clustered, num_vertices, cluster_counter):
    """Print a summary of all clusters."""
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)

    for category_id, (category, category_clusters) in enumerate(all_clusters.items()):
        color_rgb = [int(c * 255) for c in cluster_colors[category_id][:3]]
        print(f"\n{category.upper()}: {len(category_clusters)} cluster(s) | Category Color: RGB{tuple(color_rgb)}")
        for cluster_id, vertex_indices in category_clusters:
            print(f"  Cluster {cluster_id}: {len(vertex_indices)} vertices")

    print("\n" + "=" * 60)
    print(f"Total: {cluster_counter} individual clusters across {len(all_clusters)} categories")
    print(f"Total clustered vertices: {num_clustered}/{num_vertices} ({100*num_clustered/num_vertices:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Run semantic dictionary segmentation on a processed dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., rats_date-07112024_video-C0119)'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='/workspace/collab-splats/docs/splats/configs',
        help='Path to config directory containing base.yaml (default: /workspace/collab-splats/docs/splats/configs)'
    )
    parser.add_argument(
        '--dict',
        type=str,
        default=None,
        help='Path to semantic dictionary JSON file (optional, uses default if not provided)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Similarity threshold for clustering (default: 0.95)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=0.02,
        help='Spatial radius for clustering (default: 0.02)'
    )

    args = parser.parse_args()

    # Load semantic dictionary
    semantic_dictionary = load_semantic_dictionary(args.dict)
    print("Semantic Dictionary:")
    for category, queries in semantic_dictionary.items():
        print(f"  {category}: {queries}")

    # Load splatter from config (matches run_pipeline.py style)
    print(f"\nLoading dataset: {args.dataset}")
    print(f"Config directory: {args.config_dir}")

    splatter = Splatter.from_config_file(
        dataset=args.dataset,
        config_dir=args.config_dir
    )

    # Ensure preprocessing, features, and mesh are ready
    print("\nEnsuring preprocessing, features, and mesh are complete...")
    splatter.preprocess()
    splatter.extract_features()
    splatter.mesh()
    splatter.load_model()

    # Get mesh directory and create query and cluster subdirectories
    mesh_dir = splatter.config["mesh_info"]["mesh"].parent
    query_dir = mesh_dir / "query"
    cluster_dir = mesh_dir / "cluster"
    query_dir.mkdir(exist_ok=True)
    cluster_dir.mkdir(exist_ok=True)
    print(f"\nMesh directory: {mesh_dir}")
    print(f"Query output directory: {query_dir}")
    print(f"Cluster output directory: {cluster_dir}")

    # Query semantic categories
    print("\n" + "=" * 60)
    print("QUERYING SEMANTIC CATEGORIES")
    print("=" * 60)
    semantic_results = query_semantic_categories(splatter, semantic_dictionary, query_dir)

    # Cluster results
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)
    all_clusters, cluster_counter = cluster_semantic_categories(
        semantic_dictionary,
        query_dir,
        similarity_threshold=args.threshold,
        spatial_radius=args.radius
    )

    # Create categorical mesh
    print("\n" + "=" * 60)
    print("CREATING CATEGORICAL MESH")
    print("=" * 60)
    categorical_mesh, cluster_labels, num_vertices, num_clustered = create_categorical_mesh(
        all_clusters,
        semantic_dictionary,
        query_dir,
        cluster_dir,
        cluster_counter
    )

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # First, save the original RGB mesh view
    create_rgb_mesh_view(splatter, cluster_dir)

    cluster_colors, colors_all = create_combined_mesh_visualization(
        categorical_mesh,
        cluster_labels,
        all_clusters,
        num_vertices,
        cluster_counter,
        cluster_dir
    )

    create_combined_view(categorical_mesh, colors_all, cluster_dir)

    all_clusters_flat = create_individual_cluster_views(
        categorical_mesh,
        all_clusters,
        cluster_labels,
        cluster_colors,
        num_vertices,
        cluster_dir
    )

    # Print summary
    print_cluster_summary(all_clusters, cluster_colors, num_clustered, num_vertices, cluster_counter)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nQuery files saved to: {query_dir}")
    print(f"  - query-*.ply (individual category query meshes)")
    print(f"\nCluster files saved to: {cluster_dir}")
    print(f"  - all_clusters.ply (mesh with semantic clusters)")
    print(f"  - cluster_labels.npy (numpy array of cluster labels)")
    print(f"  - rgb_mesh_view.png (original fitted splat)")
    print(f"  - semantic_clusters_view.png (semantic segmentation view)")
    print(f"  - cluster_views/ (directory with category PNGs - one per semantic category)")


if __name__ == '__main__':
    main()
