"""
Floater removal and hole filling on a fused mesh.

  - every threshold is a fraction of scene scale, never a world distance
  - get_scene_scale: the robust extent those fractions multiply
  - remove_floaters: drop components too small or too far from the main body
  - fill_holes: triangulate boundary loops under a scene-scale bound
  - clean_repair_mesh: run both over a PLY path, rewritten in place
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


######## Scale


def get_scene_scale(vertices):
    """
    Robust scene extent: diagonal of the 1st-99th percentile bounding box.

    Args:
        vertices: (V, 3) float array of positions.
    Returns:
        float, scene extent in world units.
    """
    lo, hi = np.percentile(np.asarray(vertices), [1, 99], axis=0)
    return float(np.linalg.norm(hi - lo))


######## Cleaning


def remove_floaters(mesh, min_area_frac=6e-6, max_gap_frac=0.01, gap_kdtree_points=200_000):
    """
    Drop connected components that are tiny or far from the largest one; edits in place.

    Args:
        mesh: open3d.geometry.TriangleMesh, edited in place.
        min_area_frac: float, keep components with area >= this × scene_scale².
        max_gap_frac: float, keep components whose centroid is within this × scene_scale of the main body.
        gap_kdtree_points: int, cap on main-body vertices indexed for the gap query (stride-subsampled).
    Returns:
        the same open3d.geometry.TriangleMesh.
    """
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    if len(cluster_sizes) == 0:
        return mesh
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n_comp = len(cluster_sizes)

    # Scale comes from the largest component alone so strays cannot inflate it
    largest = int(cluster_sizes.argmax())
    main_xyz = verts[np.unique(tris[cluster_ids == largest])]
    scale = get_scene_scale(main_xyz)

    # Per-component surface area and centroid
    tri_pts = verts[tris]
    tri_area = 0.5 * np.linalg.norm(np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1)
    comp_area = np.zeros(n_comp)
    np.add.at(comp_area, cluster_ids, tri_area)
    comp_centroid_sum = np.zeros((n_comp, 3))
    np.add.at(comp_centroid_sum, cluster_ids, tri_pts.mean(axis=1))
    comp_centroid = comp_centroid_sum / cluster_sizes[:, None]

    # Gap = distance from each centroid to the (subsampled) main body
    stride = max(1, len(main_xyz) // gap_kdtree_points)
    comp_gap, _ = cKDTree(main_xyz[::stride]).query(comp_centroid, k=1)

    # Keep large-enough, close-enough components; the main body always stays
    keep = (comp_area >= min_area_frac * scale**2) & (comp_gap <= max_gap_frac * scale)
    keep[largest] = True
    mesh.remove_triangles_by_mask(~keep[cluster_ids])
    mesh.remove_unreferenced_vertices()
    logger.info(
        "remove_floaters: kept %d of %d components (removed %d) at scene_scale=%.3f",
        int(keep.sum()),
        n_comp,
        int((~keep).sum()),
        scale,
    )
    return mesh


######## Repair


def fill_holes(mesh, max_hole_frac=0.0045):
    """
    Fill boundary loops up to a size proportional to the scene; returns a new mesh.

    Args:
        mesh: open3d.geometry.TriangleMesh; vertex colors survive.
        max_hole_frac: float, hole-size bound as this × scene_scale (Open3D `hole_size` is
            diameter-like: a hole fills iff hole_size >= ~2 × its radius).
    Returns:
        open3d.geometry.TriangleMesh with the small holes triangulated.
    """
    scale = get_scene_scale(np.asarray(mesh.vertices))
    hole_size = max_hole_frac * scale

    # Open3D 0.19 fill_holes returns views into the from_legacy source
    #   - the source must stay bound until the result is read
    #   - chained from_legacy(...).fill_holes(...) frees it early
    #   - measured garbage, no error raised: positions off by 1.0, colors 3.7e19
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tmesh.fill_holes(hole_size=hole_size).to_legacy()

    logger.info(
        "fill_holes: hole_size=%.4f (%.4f × scene_scale %.3f), triangles %d -> %d",
        hole_size,
        max_hole_frac,
        scale,
        len(mesh.triangles),
        len(filled.triangles),
    )
    return filled


def clean_repair_mesh(mesh_path, min_area_frac=6e-6, max_gap_frac=0.01, max_hole_frac=0.0045):
    """
    Remove floaters, fill small holes, and overwrite the mesh file.

    Args:
        mesh_path: Path or str to a PLY; rewritten in place.
        min_area_frac: float, see remove_floaters.
        max_gap_frac: float, see remove_floaters.
        max_hole_frac: float, see fill_holes.
    Returns:
        Path to mesh_path.
    """
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    remove_floaters(mesh, min_area_frac=min_area_frac, max_gap_frac=max_gap_frac)
    mesh = fill_holes(mesh, max_hole_frac=max_hole_frac)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh_path
