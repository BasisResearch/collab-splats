"""
Mesh cleanup at full density, then preparation for UV unwrapping.

- clean_repair_mesh: remove_floaters, optional make_convex_hull, fill_holes
- remove_floaters drops small or far components; fill_holes patches interior loops, not the rim
- make_convex_hull: cut to a rounded ground hull, trim_mesh_edges, ground patch, bridge_mesh_edges
- build_volume_ground: obstacle membrane create_tsdf_mesh writes into unobserved voxels
- prepare_mesh: fill_holes, decimate_mesh (error-bounded QEM), make_manifold
- thresholds are scene-relative: fractions of get_scene_scale, or cells of the median edge
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import meshlib.mrmeshnumpy as mn
import meshlib.mrmeshpy as mm
import numpy as np
import open3d as o3d
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, cKDTree

from collab_splats.geometry.transforms import fit_dominant_plane
from collab_splats.mesh.utils import (
    face_areas,
    face_components,
    face_edge_ids,
    from_meshlib,
    to_meshlib,
)
from collab_splats.utils.image import fill_missing_pixels
from collab_splats.utils.torch_utils import get_device

logger = logging.getLogger(__name__)


########################################################################
# Entry point
########################################################################


def clean_repair_mesh(
    mesh: o3d.geometry.TriangleMesh,
    c2w: np.ndarray | None = None,
    voxel_size: float | None = None,
    min_area_frac: float = 6e-6,
    max_gap_frac: float = 0.01,
    max_hole_perimeter_ratio: float = 0.014,
    subdivide_fill: bool = True,
    use_convex_hull: bool = False,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    """
    Remove floaters and fill small holes; also returns the surface right after the floater cut.

    - the floater cut edits mesh in place; the hull and the hole fill build new meshes

    Args:
        mesh: fused mesh, edited in place by the floater cut.
        c2w: see make_convex_hull; required when use_convex_hull is on.
        voxel_size: see make_convex_hull; required when use_convex_hull is on.
        min_area_frac: see remove_floaters.
        max_gap_frac: see remove_floaters.
        max_hole_perimeter_ratio: see fill_holes.
        subdivide_fill: see fill_holes.
        use_convex_hull: run make_convex_hull between the floater cut and the hole fill.

    Returns:
        The cleaned mesh, and mesh itself after the floater cut: only fused faces, no hull or hole fill.

    Raises:
        ValueError: mesh has no vertices, or use_convex_hull is on and c2w or voxel_size is None.
    """
    # Empty fusion means the views never agreed on one surface; name it here, not in a hull step
    if not len(mesh.vertices):
        raise ValueError(
            "clean_repair_mesh: fused mesh has 0 vertices; the views do not agree on one "
            "surface (check BA dropped frames; keyframes may be too far apart for the motion)"
        )

    real = remove_floaters(mesh, min_area_frac=min_area_frac, max_gap_frac=max_gap_frac)
    cleaned = real

    # Optional hull between the floater cut and the hole fill; needs the camera poses
    if use_convex_hull:
        if c2w is None or voxel_size is None:
            raise ValueError(
                "clean_repair_mesh: use_convex_hull needs c2w and voxel_size"
            )

        cleaned = make_convex_hull(real, c2w, voxel_size)

    cleaned = fill_holes(
        cleaned,
        max_hole_perimeter_ratio=max_hole_perimeter_ratio,
        subdivide_fill=subdivide_fill,
    )
    return cleaned, real


########################################################################
# Cleanup (full density)
########################################################################


def remove_floaters(
    mesh: o3d.geometry.TriangleMesh,
    min_area_frac: float = 6e-6,
    max_gap_frac: float = 0.01,
    gap_kdtree_points: int = 200_000,
    keep_frac: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    """
    Drop connected components that are tiny or far from the largest one; edits in place.

    Args:
        mesh: mesh edited in place.
        min_area_frac: keep components with area >= this × scene_scale².
        max_gap_frac: keep components whose centroid is within this × scene_scale of the main body.
        gap_kdtree_points: cap on main-body vertices indexed for the gap query (stride-subsampled).
        keep_frac: keep components with at least this fraction of the largest one's faces.

    Returns:
        The same mesh.
    """
    # Edge-connected components with their face counts and areas
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    cluster_ids, cluster_sizes, comp_area = face_components(verts, tris)

    if len(cluster_sizes) == 0:
        return mesh

    n_comp = len(cluster_sizes)

    # Scale comes from the largest component alone so strays cannot inflate it
    largest = int(cluster_sizes.argmax())
    main_xyz = verts[np.unique(tris[cluster_ids == largest])]
    scale = get_scene_scale(main_xyz)

    # Per-component centroid
    comp_centroid_sum = np.zeros((n_comp, 3))
    np.add.at(comp_centroid_sum, cluster_ids, verts[tris].mean(axis=1))
    comp_centroid = comp_centroid_sum / cluster_sizes[:, None]

    # Gap = distance from each centroid to the (subsampled) main body
    stride = max(1, len(main_xyz) // gap_kdtree_points)
    comp_gap, _ = cKDTree(main_xyz[::stride]).query(comp_centroid, k=1)

    # Keep large-enough, close-enough components; the main body always stays
    keep = (comp_area >= min_area_frac * scale**2) & (comp_gap <= max_gap_frac * scale)
    keep &= cluster_sizes >= keep_frac * cluster_sizes.max()
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


def fill_holes(
    mesh: o3d.geometry.TriangleMesh,
    max_hole_perimeter_ratio: float = 0.014,
    subdivide_fill: bool = True,
    max_edge_splits: int = 20_000,
    max_plain_edges: int = 8,
) -> o3d.geometry.TriangleMesh:
    """
    Fill interior boundary loops under a perimeter bound: small loops plain, the rest smooth patches.

    - outer rim: a component's widest loop spanning half its extent; never filled, whatever the bound
    - scene_scale: diagonal of the 1st-99th percentile bounding box (get_scene_scale)
    - patches: all triangulated, then one subdivide + cotan smooth; fillHoleNicely per hole scales with the mesh
    - loops of at most max_plain_edges edges take meshlib's plain fillHole: a flat lid
    - vertex colors carry over; patch vertices take their nearest source vertex's color
    - a hole meshlib cannot fill stays open and is counted in the log

    Args:
        mesh: input mesh; not modified.
        max_hole_perimeter_ratio: fill an interior hole iff its perimeter < this × scene_scale.
        subdivide_fill: subdivide and smooth each patch; off leaves a flat lid, far cheaper.
        max_edge_splits: subdivision budget per patch; the batched subdivide gets this × patch count.
        max_plain_edges: loops with at most this many edges get a plain flat lid; 0 = all subdivided.

    Returns:
        New mesh with those loops closed.
    """
    verts = np.asarray(mesh.vertices)
    gate = max_hole_perimeter_ratio * get_scene_scale(verts)
    mmesh = to_meshlib(mesh)

    # Weld near-duplicate boundary vertices: a sliver edge between them crashes cotan smoothing natively
    mm.uniteCloseVertices(mmesh, 0.01 * mmesh.averageEdgeLength(), True)

    # Component of each hole: the face across its representative edge
    tris = np.asarray(mesh.triangles)
    cluster_ids, cluster_sizes, _ = face_components(verts, tris)
    holes = mmesh.topology.findHoleRepresentiveEdges()
    perimeters = np.array([mmesh.holePerimeter(hole) for hole in holes])
    hole_comp = np.array(
        [cluster_ids[mmesh.topology.right(hole).get()] for hole in holes],
        dtype=np.int64,
    )

    # Per-component bounding boxes, the yardstick for what counts as an outer rim
    tri_pts = verts[tris]
    n_comp = len(cluster_sizes)
    comp_lo = np.full((n_comp, 3), np.inf)
    comp_hi = np.full((n_comp, 3), -np.inf)
    np.minimum.at(comp_lo, cluster_ids, tri_pts.min(axis=1))
    np.maximum.at(comp_hi, cluster_ids, tri_pts.max(axis=1))

    # Walk each loop once: its edge count and its bounding-box diagonal
    points = mn.getNumpyVerts(mmesh)
    n_edges = np.zeros(len(holes), dtype=np.int64)
    extents = np.zeros(len(holes))

    for i, hole in enumerate(holes):
        loop, extents[i] = _boundary_loop(mmesh, points, hole)
        n_edges[i] = len(loop)

    # Find each component's outer rim: its widest loop, if that loop spans half the component or more
    widest = {hole_comp[i]: i for i in np.argsort(extents)}
    rims = set()

    for i in widest.values():
        comp_extent = np.linalg.norm(comp_hi[hole_comp[i]] - comp_lo[hole_comp[i]])

        if extents[i] >= 0.5 * comp_extent:
            rims.add(i)

    # Split the interior loops under the gate into plain lids and nice patches
    plain, nice = [], []

    for i, hole in enumerate(holes):
        if i in rims or perimeters[i] >= gate:
            continue

        (plain if n_edges[i] <= max_plain_edges else nice).append(hole)

    # Triangulate every nice patch; one degenerate hole must not cost the rest
    edge_len = mmesh.averageEdgeLength()
    first_new_face = mmesh.topology.faceSize()
    triangulate = mm.FillHoleNicelySettings().triangulateParams
    n_failed = _fill_each(mmesh, nice, triangulate)

    # One subdivide over all the new patches, then one cotan smooth of their new vertices to the rims
    if subdivide_fill and nice:
        patch = np.zeros(mmesh.topology.faceSize(), dtype=bool)
        patch[first_new_face:] = True
        new_verts = mm.VertBitSet()
        subdivide = mm.SubdivideSettings()
        subdivide.maxEdgeLen = edge_len
        subdivide.maxEdgeSplits = max_edge_splits * len(nice)
        subdivide.maxAngleChangeAfterFlip = np.pi / 6
        subdivide.region = mn.faceBitSetFromBools(patch)
        subdivide.newVerts = new_verts
        mm.subdivideMesh(mmesh, subdivide)
        mm.positionVertsSmoothly(mmesh, new_verts, mm.EdgeWeights.Cotan)

    # Plain lids last, so the subdivide region holds only the nice patches
    n_plain_failed = _fill_each(mmesh, plain, mm.FillHoleParams())
    n_plain = len(plain) - n_plain_failed
    n_failed += n_plain_failed
    n_filled = len(plain) + len(nice) - n_failed

    # Back to Open3D; normals are recomputed over the patched surface
    filled = from_meshlib(mmesh, mesh)

    if mesh.has_vertex_normals():
        filled.compute_vertex_normals()

    logger.info(
        "fill_holes: filled %d of %d holes (%d plain, %d outer rims kept) under perimeter %.4f "
        "(%.4f × scene_scale, %d failed), triangles %d -> %d",
        n_filled,
        len(holes),
        n_plain,
        len(rims),
        gate,
        max_hole_perimeter_ratio,
        n_failed,
        len(mesh.triangles),
        len(filled.triangles),
    )
    return filled


def trim_mesh_edges(
    mesh: o3d.geometry.TriangleMesh,
    c2w: np.ndarray,
    *,
    outline_open: int = 25,
    min_up_cos: float = 0.7,
) -> o3d.geometry.TriangleMesh:
    """
    Cut the ragged outer edge: rim-connected regions outside the smoothed top-down outline.

    - outline: top-down coverage closed, holes filled, opened, blurred, largest region
    - opening at outline_open cells removes fingers up to about twice that wide
    - a region outside the outline survives unless it is edge-connected to the outer rim
    - only faces are removed; vertices and their colors are kept as they are

    Args:
        mesh: input mesh; not modified.
        c2w: see make_convex_hull.
        outline_open: opening radius that smooths the outline, cells of the median edge length;
            also sets the blur, sigma outline_open / 3.
        min_up_cos: see make_convex_hull.

    Returns:
        New mesh with the same vertices and the trimmed faces.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimmed = o3d.geometry.TriangleMesh(mesh)

    # Top-down image of the mesh on the ground plane
    frame, center = _find_ground_plane(verts, c2w, min_up_cos=min_up_cos)
    res = _median_edge(verts, faces)
    xy = ((verts - center) @ frame.T)[:, :2]
    lo, shape = _ground_image_bounds(xy, res, margin=outline_open)

    # Faces whose centroid falls outside the smoothed outline; holes fill first so the opening keeps their rims
    outline = _keep_largest_region_fill_holes(
        cv2.morphologyEx(
            _mesh_coverage_mask(xy, faces, lo, res, shape),
            cv2.MORPH_CLOSE,
            _filled_circle(3),
        )
    )
    outline = cv2.morphologyEx(outline, cv2.MORPH_OPEN, _filled_circle(outline_open))

    # Blur and re-threshold so the outline curves smoothly between the opening's arcs
    blurred = cv2.GaussianBlur(outline.astype(np.float32), (0, 0), outline_open / 3)
    outline = _keep_largest_region_fill_holes((blurred > 127).astype(np.uint8) * 255)
    center_px = _ground_xy_to_pixel(xy[faces].mean(axis=1), lo, res)
    outside = outline[center_px[:, 1], center_px[:, 0]] == 0
    mmesh = to_meshlib(mesh)
    loop = _outer_rim_loop(mmesh)

    if not outside.any() or not loop:
        return trimmed

    # Faces on the outer rim
    rim_faces = np.array([mmesh.topology.right(edge).get() for edge in loop])

    # Group the outside faces into edge-connected regions
    outside_ids = np.flatnonzero(outside)
    pos = np.full(len(faces), -1)
    pos[outside_ids] = np.arange(len(outside_ids))
    region = face_components(verts, faces[outside_ids])[0]

    # Drop the regions that touch the outer rim
    seeds = np.unique(region[pos[rim_faces[outside[rim_faces]]]])
    drop = np.zeros(len(faces), dtype=bool)
    drop[outside_ids[np.isin(region, seeds)]] = True
    trimmed.remove_triangles_by_mask(drop)
    logger.info(
        "trim_mesh_edges: dropped %d faces in %d rim-connected regions",
        int(drop.sum()),
        len(seeds),
    )
    return trimmed


def bridge_mesh_edges(
    mesh: o3d.geometry.TriangleMesh,
    *,
    radius: float = 3.0,
    min_loop_edges: int = 40,
) -> o3d.geometry.TriangleMesh:
    """
    Bridge narrow necks of the outer rim, so inlets become interior holes fill_holes can close.

    - neck: two outer-rim vertices within radius in 3D, far apart along the rim both ways round
    - widest separation first; one meshlib makeBridge (two faces) per pass

    Args:
        mesh: input mesh; not modified.
        radius: largest 3D gap bridged, cells of the median edge length.
        min_loop_edges: smallest separation along the rim, in rim edges, that counts as a neck.

    Returns:
        New mesh with the bridges added.
    """
    gap = radius * _median_edge(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    mmesh = to_meshlib(mesh)
    topology = mmesh.topology
    points = mn.getNumpyVerts(mmesh)

    # One bridge per pass across the widest neck; skip pairs meshlib refuses
    n_bridges, refused = 0, set()
    bridged = True

    while bridged:
        loop = _outer_rim_loop(mmesh)

        if not loop:
            break

        # Neck candidates: close rim pairs, ordered by separation along the rim
        org = np.array([topology.org(edge).get() for edge in loop])
        pairs = cKDTree(points[org]).query_pairs(gap, output_type="ndarray")
        apart = np.abs(pairs[:, 0] - pairs[:, 1])
        apart = np.minimum(apart, len(loop) - apart)
        order = np.argsort(-apart)
        bridged = False

        for i, j in pairs[order[apart[order] > min_loop_edges]]:
            key = (int(org[i]), int(org[j]))

            if key in refused:
                continue

            if mm.makeBridge(topology, loop[i], loop[j]):
                n_bridges += 1
                bridged = True
                break

            refused.add(key)

    logger.info("bridge_mesh_edges: %d bridges", n_bridges)
    return from_meshlib(mmesh, mesh)


########################################################################
# Convex hull
########################################################################


def make_convex_hull(
    mesh: o3d.geometry.TriangleMesh,
    c2w: np.ndarray,
    voxel_size: float,
    *,
    hull_round: int = 20,
    outline_open: int = 25,
    bridge_radius: float = 3.0,
    min_piece_faces: int = 1000,
    min_up_cos: float = 0.7,
) -> o3d.geometry.TriangleMesh:
    """
    Cut the mesh to a rounded convex hull of its ground, trim the ragged edge, patch the ground gaps.

    - lengths are in cells: pixels of the top-down image of the mesh, one median edge length each
    - ground: first of 3 RANSAC planes facing the camera up, else the largest; up on the cameras' side
    - cut: faces outside the rounded hull, then trim_mesh_edges on the ragged outline
    - patch: obstacle membrane over the ground gaps the volume ground left, seamed to the ground rim
    - repair: make_manifold, then bridge_mesh_edges closes inlets
    - assumes a ground-dominated height field; indoor and object scenes should leave it off

    Args:
        mesh: input mesh; not modified.
        c2w: (N, 4, 4) camera-to-world poses, OpenCV axes; they set the expected up and its sign.
        voxel_size: TSDF voxel the mesh was fused at; sets the patch's ground coverage band.
        hull_round: opening radius that rounds the hull's corners, cells.
        outline_open: see trim_mesh_edges.
        bridge_radius: see bridge_mesh_edges (radius).
        min_piece_faces: pieces with fewer faces are dropped.
        min_up_cos: smallest ``|plane normal . expected up|`` that counts as ground;
            below it on every plane, the largest plane is used.

    Returns:
        New mesh: cut, trimmed, patched, manifold; a copy of the input if no piece survives the cut.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = (
        np.asarray(mesh.vertex_colors)
        if mesh.has_vertex_colors()
        else np.full((len(verts), 3), 0.5)
    )

    # Ground frame; top-down image of the mesh, with a margin for the hull's rounding
    frame, center = _find_ground_plane(verts, c2w, min_up_cos=min_up_cos)
    res = _median_edge(verts, faces)
    local = (verts - center) @ frame.T
    lo, shape = _ground_image_bounds(local[:, :2], res, margin=hull_round)
    vert_px = _ground_xy_to_pixel(local[:, :2], lo, res)

    # Hull cut: faces outside the rounded hull, then the pieces the cut strands
    inside = _create_rounded_hull_mask(
        _mesh_coverage_mask(local[:, :2], faces, lo, res, shape), hull_round
    )
    faces = faces[inside[vert_px[:, 1], vert_px[:, 0]][faces].all(axis=1)]
    faces = _drop_small_pieces(verts, faces, min_piece_faces)

    # Too fragmented for a hull: every piece is under the floor, so keep the fused surface
    if not len(faces):
        logger.warning(
            "make_convex_hull: no piece of %d+ faces survives the hull cut; mesh left uncut",
            min_piece_faces,
        )
        return o3d.geometry.TriangleMesh(mesh)

    # Outline trim, then the pieces it strands
    cut = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )
    trimmed = trim_mesh_edges(
        cut, c2w, outline_open=outline_open, min_up_cos=min_up_cos
    )
    faces = _drop_small_pieces(verts, np.asarray(trimmed.triangles), min_piece_faces)

    # Ground patch out to the hull, joined to the mesh; grid points are appended to the vertices
    grid_local, grid_colors, patch = _patch_ground_gaps(
        faces,
        local,
        colors,
        inside,
        lo,
        res,
        scale=get_scene_scale(verts),
        voxel_size=voxel_size,
    )
    all_verts = np.concatenate([verts, grid_local @ frame + center])
    all_faces = _drop_small_pieces(
        all_verts, np.concatenate([faces, patch]), min_piece_faces
    )

    # Repair, then bridge the outer rim's necks
    joined = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(all_verts), o3d.utility.Vector3iVector(all_faces)
    )
    joined.vertex_colors = o3d.utility.Vector3dVector(
        np.concatenate([colors, grid_colors])
    )
    joined = bridge_mesh_edges(make_manifold(joined), radius=bridge_radius)

    if not mesh.has_vertex_colors():
        joined.vertex_colors = o3d.utility.Vector3dVector()

    logger.info(
        "make_convex_hull: cell %.4f, triangles %d -> %d",
        res,
        len(mesh.triangles),
        len(joined.triangles),
    )
    return joined


def _create_rounded_hull_mask(coverage: np.ndarray, hull_round: int) -> np.ndarray:
    """
    Rounded convex hull of the mesh's main region, as a bool mask in the top-down image.

    - main region: closed, largest region, holes filled, then opened so thin spurs do not stretch the hull
    - corners rounded by an opening of hull_round cells
    - all False when the opening erases the whole region
    """
    solid = _keep_largest_region_fill_holes(
        cv2.morphologyEx(coverage, cv2.MORPH_CLOSE, _filled_circle(3))
    )
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, _filled_circle(15))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Thin mesh: nothing survives the opening, so there is no hull to draw
    if not contours:
        return np.zeros(coverage.shape, bool)

    hull = np.zeros_like(coverage)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(contours))], 255)
    return cv2.morphologyEx(hull, cv2.MORPH_OPEN, _filled_circle(hull_round)) > 0


def _patch_ground_gaps(
    faces: np.ndarray,
    local: np.ndarray,
    colors: np.ndarray,
    inside: np.ndarray,
    lo: np.ndarray,
    res: float,
    *,
    scale: float,
    voxel_size: float,
    kappa: float = 30.0,
    band: float = 3.0,
    near_voxels: float = 80.0,
    min_up: float = 0.5,
    rim_max_edge: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ground patch over the hull's uncovered cells, seamed to the mesh's up-facing ground rim.

    - membrane: solve_ground over the per-cell lowest mesh height, kappa / scale; ground = within band cells
    - covered: ground faces, or up-facing (|n_z| > 0.5) faces within near_voxels voxels of the membrane;
      so a low step or curb top near the membrane blocks the patch
    - rim: open-edge ground vertices next to the gap whose vertex normal has n_z >= min_up
    - grid points on gap cells, Delaunay with the rim; one Laplacian solve pinned at the rim, pulled to
      the membrane away from the mesh; rim faces over rim_max_edge cells and non-manifold faces dropped
    - inputs in the ground frame: local (V, 3) vertices, inside the hull mask, lo/res its image grid
    - returns (G, 3) local grid points, (G, 3) colors, (P, 3) faces; ids >= V index the grid points
    """
    xy, height = local[:, :2], local[:, 2]
    shape = inside.shape
    vert_px = _ground_xy_to_pixel(xy, lo, res)
    n_local = len(local)

    # Obstacle membrane under the lowest mesh height per cell; ground = vertices within band of it
    used = np.unique(faces)
    lowest = np.full(inside.size, np.inf, dtype=np.float32)
    np.minimum.at(lowest, vert_px[used, 1] * shape[1] + vert_px[used, 0], height[used])
    g = solve_ground(lowest.reshape(shape), res, kappa / scale)
    g_vert = g[vert_px[:, 1], vert_px[:, 0]]
    is_ground = np.abs(height - g_vert) < band * res

    # Covered cells: ground faces, or up-facing faces near the membrane; the rest of the hull is gap
    face_n = np.cross(
        local[faces[:, 1]] - local[faces[:, 0]], local[faces[:, 2]] - local[faces[:, 0]]
    )
    up = np.abs(face_n[:, 2]) / (np.linalg.norm(face_n, axis=1) + 1e-12) > 0.5
    near = (np.abs(height - g_vert) < near_voxels * voxel_size)[faces].all(axis=1)
    covering = faces[is_ground[faces].all(axis=1) | (up & near)]
    covered = _mesh_coverage_mask(xy, covering, lo, res, shape) > 0
    gap = inside & ~covered
    dist_mesh = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 5)
    rows, cols = np.nonzero(gap)
    n_grid = len(rows)

    # Rim anchors: open-edge ground vertices next to the gap, up-facing by area-weighted normal
    edge_ids, per_edge = np.unique(
        face_edge_ids(faces, n_local, directed=False), return_counts=True
    )
    open_edges = edge_ids[per_edge == 1]
    rim = np.unique(np.concatenate([open_edges // n_local, open_edges % n_local]))
    near_gap = cv2.dilate(gap.astype(np.uint8), _filled_circle(2)).astype(bool)
    rim = rim[near_gap[vert_px[rim, 1], vert_px[rim, 0]] & is_ground[rim]]
    vert_n = np.zeros_like(local)

    for k in range(3):
        np.add.at(vert_n, faces[:, k], face_n)

    rim = rim[
        vert_n[rim, 2] / np.maximum(np.linalg.norm(vert_n[rim], axis=1), 1e-12)
        >= min_up
    ]
    n_rim = len(rim)

    if n_grid < 3 or n_rim == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), np.int64)

    # Grid points on gap cells: membrane target, pulled harder the farther from the mesh
    grid_xy = np.stack([lo[0] + cols * res, lo[1] + rows * res], axis=1)
    pull = 1e-6 + 0.5 * np.clip(dist_mesh[rows, cols] / 20, 0, 1) ** 2

    # Delaunay over rim + grid: faces over the gap, inside the hull, short, not rim-only unless tiny
    points = np.concatenate([xy[rim], grid_xy])
    tris = Delaunay(points).simplices
    center_px = _ground_xy_to_pixel(points[tris].mean(axis=1), lo, res)
    gap_wide = cv2.dilate(gap.astype(np.uint8), _filled_circle(1)).astype(bool)
    keep = (
        gap_wide[center_px[:, 1], center_px[:, 0]]
        & inside[center_px[:, 1], center_px[:, 0]]
    )
    longest = np.linalg.norm(
        points[tris] - points[np.roll(tris, 1, axis=1)], axis=2
    ).max(axis=1)
    keep &= longest < 3 * res
    keep &= ~(tris < n_rim).all(axis=1) | (longest < 1.5 * res)
    tris = tris[keep]

    # Laplacian over the patch edges: grid unknowns, rim heights (local median) and colors pinned
    tri_edges = np.unique(face_edge_ids(tris, len(points), directed=False))
    ends = np.stack([tri_edges // len(points), tri_edges % len(points)], axis=1)
    src = np.concatenate([ends[:, 0], ends[:, 1]])
    dst = np.concatenate([ends[:, 1], ends[:, 0]])
    on_grid = src >= n_rim
    src, dst = src[on_grid] - n_rim, dst[on_grid]
    to_rim = dst < n_rim
    target = np.column_stack(
        [
            g[rows, cols].astype(np.float64),
            np.tile(np.median(colors, axis=0), (n_grid, 1)),
        ]
    )
    pinned = np.column_stack(
        [_local_median(xy[rim], height[rim], 6 * res), colors[rim]]
    )
    rhs = pull[:, None] * target
    np.add.at(rhs, src[to_rim], pinned[dst[to_rim]])
    lap = sp.csc_matrix(
        (-np.ones((~to_rim).sum()), (src[~to_rim], dst[~to_rim] - n_rim)),
        shape=(n_grid, n_grid),
    ) + sp.diags(pull + np.bincount(src, minlength=n_grid))
    solution = splu(lap.tocsc()).solve(rhs)
    grid_local = np.concatenate([grid_xy, solution[:, :1]], axis=1)
    grid_colors = np.clip(solution[:, 1:], 0, 1)

    # Drop rim faces with a long 3D edge; then wind every face up
    points3 = np.concatenate([np.column_stack([xy[rim], height[rim]]), grid_local])
    longest3 = np.linalg.norm(
        points3[tris] - points3[np.roll(tris, 1, axis=1)], axis=2
    ).max(axis=1)
    tris = tris[(longest3 < rim_max_edge * res) | ~(tris < n_rim).any(axis=1)]
    corners = points3[tris]
    down = (
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2] < 0
    )
    tris[down] = tris[down][:, [0, 2, 1]]
    patch = np.concatenate([rim, n_local + np.arange(n_grid)])[tris]
    n_verts = n_local + n_grid

    # Mesh edges between rim vertices; flip patch faces that repeat one's direction, if that fixes it
    on_rim = np.zeros(n_verts, dtype=bool)
    on_rim[rim] = True
    mesh_dir = face_edge_ids(faces, n_verts, directed=True).ravel()
    mesh_dir = mesh_dir[on_rim[mesh_dir // n_verts] & on_rim[mesh_dir % n_verts]]
    start, end = mesh_dir // n_verts, mesh_dir % n_verts
    mesh_und = np.minimum(start, end) * n_verts + np.maximum(start, end)
    mesh_ids = np.unique(mesh_dir)
    repeats = np.isin(face_edge_ids(patch, n_verts, directed=True), mesh_ids).any(
        axis=1
    )
    flipped = patch[repeats][:, [0, 2, 1]]
    ok = ~np.isin(face_edge_ids(flipped, n_verts, directed=True), mesh_ids).any(axis=1)
    patch[np.flatnonzero(repeats)[ok]] = flipped[ok]

    # Drop patch faces on an edge used more than twice, or a direction used more than once
    while True:
        all_und = np.concatenate(
            [mesh_und, face_edge_ids(patch, n_verts, directed=False).ravel()]
        )
        all_dir = np.concatenate(
            [mesh_dir, face_edge_ids(patch, n_verts, directed=True).ravel()]
        )
        _, edge_of, per_edge = np.unique(
            all_und, return_inverse=True, return_counts=True
        )
        _, dir_of, per_dir = np.unique(all_dir, return_inverse=True, return_counts=True)
        bad = (per_edge[edge_of[len(mesh_und) :].reshape(-1, 3)] > 2).any(axis=1)
        bad |= (per_dir[dir_of[len(mesh_dir) :].reshape(-1, 3)] > 1).any(axis=1)

        if not bad.any():
            break

        patch = patch[~bad]

    logger.info(
        "make_convex_hull: patch %d gap cells, %d rim anchors, %d faces",
        n_grid,
        n_rim,
        len(patch),
    )
    return grid_local, grid_colors, patch


def _local_median(xy: np.ndarray, values: np.ndarray, radius: float) -> np.ndarray:
    """
    Median of values over the points within radius of each point, in the ground plane.

    - one k-nearest query, k = the largest neighborhood (at least 2); missing slots are +inf, sort last
    """
    # Neighborhood size per point, then one k-nearest query sized to the largest
    tree = cKDTree(xy)
    count = tree.query_ball_point(xy, radius, workers=-1, return_length=True)
    bound = np.nextafter(radius, np.inf)
    _, neighbors = tree.query(
        xy, k=max(2, int(count.max())), distance_upper_bound=bound, workers=-1
    )

    # Neighbor values, +inf in the slots past each point's count, sorted per row
    found = neighbors < len(xy)
    vals = np.full(neighbors.shape, np.inf)
    vals[found] = values[neighbors[found]]
    vals.sort(axis=1)

    # Middle element, or mean of the two middle ones for an even count
    row = np.arange(len(xy))
    return (vals[row, (count - 1) // 2] + vals[row, count // 2]) / 2


########################################################################
# Volume ground
########################################################################

# Ported from the session 8e887a96 scratchpad
# - recover/sb/volground.py:build, recover/sb/proto_hull.py:solve_ground
# - spec: docs/superpowers/specs/2026-11-04-weighted-mesh-fusion-design.md


@dataclass
class VolumeGround:
    """
    Ground membrane on a top-down grid in the ground frame, plus the TSDF blocks its band touches.

    - local coords: (world - center) @ frame.T; local z is height above the ground plane
    - height and color on the coarse grid (res per cell); inside on the voxel grid, 4x its shape
    - frame: (3, 3) rows ground x, y, up; center: (3,) world origin; lo: (2,) local xy of cell (0, 0)
    - color: (Hc, Wc, 3) float32 in 0-255, as the TSDF stores it; blocks: (M, 3) int64 block coords
    """

    frame: np.ndarray
    center: np.ndarray
    lo: np.ndarray
    res: float
    voxel_size: float
    height: np.ndarray
    inside: np.ndarray
    color: np.ndarray
    blocks: np.ndarray
    sdf_trunc: float


def build_volume_ground(
    depths: np.ndarray,
    weights: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    *,
    voxel_size: float,
    sdf_trunc: float,
    kappa: float = 30.0,
    grow: int = 4,
    stride: int = 8,
) -> VolumeGround | None:
    """
    Obstacle ground under the fused depth, for create_tsdf_mesh to write into unobserved voxels.

    - points: depth with weight > 0, every stride-th pixel at its pixel center
    - per 4-voxel cell: the 10th-percentile height (>= 3 points) is the obstacle, its color the ground's
    - inside: rounded hull of known cells at voxel res, minus closed coverage (tops stay empty), grown
    - coarse cells bin by round (center lo + c res), fine/band use lo + (c + 0.5) res: prototype parity
    - membrane: solve_ground at 4 voxels per cell, kappa / scene scale
    - blocks: inside columns x heights within the membrane +- (sdf_trunc + 1 voxel)

    Args:
        depths: (N, H, W) frame depth, 0 = none; the depths create_tsdf_mesh fuses.
        weights: (N, H, W) fusion weights; pixels at 0 do not shape the ground.
        rgbs: (N, H, W, 3) uint8 frames.
        c2w: (N, 4, 4) camera-to-world poses, OpenCV axes.
        K: (N, 3, 3) frame intrinsics, unshifted: pixel u's center is u + 0.5.
        voxel_size: TSDF voxel edge.
        sdf_trunc: TSDF truncation band.
        kappa: membrane pressure before the 1 / scene-scale normalization.
        grow: dilation of the gap mask into the observed ground, voxels.
        stride: pixel step of the ground points.

    Returns:
        The ground, or None when no pixel has both depth and weight.
    """
    t0 = time.time()

    # Weighted depth points, every stride-th pixel at its center, with their colors
    points, colors = [], []
    for i in range(len(depths)):
        d = depths[i, ::stride, ::stride]
        v, u = np.nonzero((d > 0) & (weights[i, ::stride, ::stride] > 0))
        z = d[v, u]
        x = ((u * stride + 0.5) - K[i, 0, 2]) / K[i, 0, 0] * z
        y = ((v * stride + 0.5) - K[i, 1, 2]) / K[i, 1, 1] * z
        points.append(np.column_stack([x, y, z]) @ c2w[i, :3, :3].T + c2w[i, :3, 3])
        colors.append(rgbs[i, ::stride, ::stride][v, u])
    points = np.concatenate(points)
    colors = np.concatenate(colors).astype(np.float32)

    if not len(points):
        return None

    # Ground frame; coarse top-down grid at 4 voxels per cell
    frame, center = _find_ground_plane(
        points[:: max(1, len(points) // 500_000)], c2w, min_up_cos=0.7
    )
    local = (points - center) @ frame.T
    res = 4 * voxel_size
    lo, shape = _ground_image_bounds(local[:, :2], res, margin=6)
    px = _ground_xy_to_pixel(local[:, :2], lo, res)
    cell = px[:, 1] * shape[1] + px[:, 0]

    # Per-cell 10th-percentile height and its point's color; cells with < 3 points unknown
    order = np.lexsort((local[:, 2], cell))
    sorted_cells = cell[order]
    starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    counts = np.diff(np.r_[starts, len(sorted_cells)])
    pick = order[starts + (0.1 * counts).astype(int)]
    ok = counts >= 3
    lowest = np.full(shape[0] * shape[1], np.inf, np.float32)
    lowest[sorted_cells[starts][ok]] = local[pick[ok], 2]
    lowest = lowest.reshape(shape)
    cell_color = np.zeros((shape[0] * shape[1], 3), np.float32)
    cell_color[sorted_cells[starts][ok]] = colors[pick[ok]]

    known = np.isfinite(lowest)
    if not known.any():
        return None

    # Ground color everywhere: known cells filled outward, then softened
    color = np.stack(
        [
            fill_missing_pixels(
                np.where(known, cell_color[:, k].reshape(shape), 0), known
            )
            for k in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    color = cv2.GaussianBlur(color, (0, 0), 1)

    # Gap columns at voxel res: rounded hull minus closed coverage, grown into the observed ground
    fine = cv2.resize(
        known.astype(np.uint8),
        (shape[1] * 4, shape[0] * 4),
        interpolation=cv2.INTER_NEAREST,
    )
    inside = _create_rounded_hull_mask(fine, 20)
    covered = cv2.morphologyEx(fine, cv2.MORPH_CLOSE, _filled_circle(3)) > 0
    gap = cv2.dilate((inside & ~covered).astype(np.uint8), _filled_circle(grow)) > 0
    inside &= gap

    # Obstacle membrane under the lowest heights
    t_solve = time.time()
    height = solve_ground(lowest, res, kappa / get_scene_scale(points))
    t_solve = time.time() - t_solve

    # Band blocks (16-voxel Open3D blocks): inside columns x membrane +- (sdf_trunc + 1 voxel)
    rows, cols = np.nonzero(inside)
    xy = np.column_stack(
        [lo[0] + (cols + 0.5) * voxel_size, lo[1] + (rows + 0.5) * voxel_size]
    )
    span = np.arange(-sdf_trunc - voxel_size, sdf_trunc + 2 * voxel_size, voxel_size)
    heights = height[rows // 4, cols // 4][:, None] + span
    band = np.column_stack([np.repeat(xy, len(span), axis=0), heights.ravel()])
    blocks = np.floor((band @ frame + center) / (16 * voxel_size)).astype(np.int64)
    blocks = np.unique(blocks.reshape(-1, 3), axis=0)

    logger.info(
        "build_volume_ground: %d blocks, %d hull cells, solve %.1fs, total %.1fs",
        len(blocks),
        int(inside.sum()),
        t_solve,
        time.time() - t0,
    )
    return VolumeGround(
        frame=frame,
        center=center,
        lo=lo,
        res=res,
        voxel_size=voxel_size,
        height=height,
        inside=inside,
        color=color,
        blocks=blocks,
        sdf_trunc=sdf_trunc,
    )


def solve_ground(lowest: np.ndarray, res: float, kappa: float) -> np.ndarray:
    """
    Obstacle membrane: the lowest surface under every observed height, pushed up by uniform pressure.

    - fixed point: observed cells g = mean4(g) + kappa h^2 / 4, gaps mean4(g); g <= lowest everywhere
    - red-black projected SOR (omega 1.8), coarse-to-fine over a min-pooled pyramid down to 32 cells
    - per level: up to 20000 sweeps; every 50th, stop if that sweep moved g by less than 1e-4 res

    Args:
        lowest: (H, W) observed height per cell; inf or nan = unobserved.
        res: cell size, world units.
        kappa: pressure; larger lifts the membrane further into the gaps between observations.

    Returns:
        (H, W) float32 membrane height.

    Raises:
        ValueError: no cell of lowest is finite.
    """
    if not np.isfinite(lowest).any():
        raise ValueError("solve_ground: no observed cell")

    device = torch.device(get_device())
    finite = np.where(np.isfinite(lowest), lowest, np.inf).astype(np.float32)
    levels = [torch.from_numpy(finite).to(device)]

    # Min-pool pyramid down to <= 32 cells on the short side; unobserved stays inf
    while min(levels[-1].shape) > 32:
        U = levels[-1]
        H, W = U.shape
        U = F.pad(U, (0, W % 2, 0, H % 2), value=float("inf"))
        levels.append(-F.max_pool2d(-U[None, None], 2)[0, 0])

    # Coarse to fine: each level starts from the coarser solution, upsampled
    g = None
    for level in range(len(levels) - 1, -1, -1):
        U = levels[level]
        h = res * 2**level
        known = torch.isfinite(U)

        if g is None:
            g = torch.full_like(U, float(U[known].min()))
        else:
            g = F.interpolate(
                g[None, None], scale_factor=2, mode="bilinear", align_corners=False
            )[0, 0, : U.shape[0], : U.shape[1]]

        # Pressure on observed cells; red-black checkerboard
        source = torch.where(
            known, torch.full_like(U, kappa * h * h / 4), torch.zeros_like(U)
        )
        yy, xx = torch.meshgrid(
            torch.arange(U.shape[0], device=device),
            torch.arange(U.shape[1], device=device),
            indexing="ij",
        )
        red = ((yy + xx) % 2).bool()

        # Projected SOR sweeps, capped at the observations
        for sweep in range(20000):
            prev = g
            for phase in (red, ~red):
                pad = F.pad(g[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
                target = (
                    pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
                ) / 4
                target = target + source
                g = torch.where(phase, torch.minimum(g + 1.8 * (target - g), U), g)

            if sweep % 50 == 49 and float((g - prev).abs().max()) < 1e-4 * res:
                break

        logger.debug(
            "solve_ground: level %d %s, %d sweeps", level, tuple(U.shape), sweep + 1
        )

    return g.cpu().numpy()


########################################################################
# Prepare for UV unwrap
########################################################################


def decimate_mesh(
    mesh: o3d.geometry.TriangleMesh, *, max_error: float, max_faces: int | None = None
) -> tuple[o3d.geometry.TriangleMesh, float]:
    """
    QEM decimation to an absolute surface-deviation bound; vertices are removed, never moved.

    Args:
        mesh: input mesh; not modified.
        max_error: largest allowed surface deviation in world units.
        max_faces: most faces to keep, reached past max_error if need be; None = no limit.

    Returns:
        The decimated mesh and the result error in world units.
    """
    # Collapse edges with meshlib QEM under an absolute error bound, keeping vertex positions
    mmesh = to_meshlib(mesh)
    settings = mm.DecimateSettings()
    settings.maxError = float(max_error)
    settings.optimizeVertexPos = False
    settings.packMesh = True
    settings.subdivideParts = 64
    error = float(mm.decimateMesh(mmesh, settings).errorIntroduced)
    n_faces = mmesh.topology.numValidFaces()

    # Over the face limit: decimate on to max_faces with no error bound
    if max_faces is not None and n_faces > max_faces:
        settings.maxError = float(np.finfo(np.float32).max)
        settings.maxDeletedFaces = n_faces - max_faces
        error = max(error, float(mm.decimateMesh(mmesh, settings).errorIntroduced))

    return from_meshlib(mmesh, mesh), error


def make_manifold(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Repair a mesh to a clean manifold; the input is left untouched.

    - duplicate and fold-over faces pass Open3D's manifold check
    - Open3D's remove_duplicated_triangles misses a duplicate with reversed winding
    - zero-area faces (distinct ids at one position) are dropped; remove_degenerate_triangles misses them
    - bowtie vertices: one copy per extra fan, all fans found in one corner-graph pass

    Args:
        mesh: input mesh.

    Returns:
        New mesh with no degenerate, duplicate, fold-over or non-manifold parts.
    """
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_degenerate_triangles()
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    # Duplicate faces in any winding: keep the first of each sorted vertex triple
    n_in = len(f)
    tri = np.sort(f, axis=1)
    order = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    tri = tri[order]
    first = np.ones(len(f), dtype=bool)
    first[1:] = (tri[1:] != tri[:-1]).any(axis=1)
    keep = np.zeros(len(f), dtype=bool)
    keep[order[first]] = True
    f = f[keep]
    n_dup = n_in - len(f)

    # Drop fold-over faces: keep the first face using each directed edge, drop later ones
    _, first_edge = np.unique(
        face_edge_ids(f, len(v), directed=True).T.reshape(-1), return_index=True
    )
    later = np.ones(3 * len(f), dtype=bool)
    later[first_edge] = False
    fold = later.reshape(3, -1).any(axis=0)
    f = f[~fold]

    # Drop zero-area faces: two corners at one position under distinct ids
    f = f[face_areas(v, f) > 0]

    # Non-manifold edges: Open3D drops faces until each edge has at most two; skipped when none exist
    _, per_edge = np.unique(
        face_edge_ids(f, len(v), directed=False), return_counts=True
    )

    if (per_edge > 2).any():
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
        )
        mesh.remove_non_manifold_edges()
        f = np.asarray(mesh.triangles)

    # Split bowtie vertices, then rebuild with the copies appended and drop vertices no face references
    f, new_ids = _split_bowties(f, len(v))
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack([v, v[new_ids]])),
        o3d.utility.Vector3iVector(f),
    )

    if colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(
            np.vstack([colors, colors[new_ids]])
        )

    out.remove_unreferenced_vertices()
    logger.info(
        "make_manifold: dropped %d duplicate + %d fold-over faces, split %d bowtie vertices",
        n_dup,
        int(fold.sum()),
        len(new_ids),
    )
    return out


def _split_bowties(faces: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Vertex copy for every fan around a vertex after its first; returns new faces and copied ids.

    - fan: the vertex's corners joined across shared edges; one corner graph covers every vertex at once
    - fans ordered by their lowest face; copies numbered from n_verts, by vertex then fan
    """
    f = faces.astype(np.int64)
    n_faces = len(f)

    # Each edge's two corners, ordered by vertex id, so shared edges pair corners of the same vertex
    nxt = np.roll(f, -1, axis=1)
    k = np.arange(3)
    base = 3 * np.arange(n_faces)[:, None]
    lo_corner = base + np.where(f < nxt, k, (k + 1) % 3)
    hi_corner = base + np.where(f < nxt, (k + 1) % 3, k)

    # Link the matching corners across every shared edge; components are fans
    key = face_edge_ids(f, n_verts, directed=False).ravel()
    order = np.argsort(key, kind="stable")
    key = key[order]
    lo_corner = lo_corner.ravel()[order]
    hi_corner = hi_corner.ravel()[order]
    same = key[1:] == key[:-1]
    src = np.concatenate([lo_corner[:-1][same], hi_corner[:-1][same]])
    dst = np.concatenate([lo_corner[1:][same], hi_corner[1:][same]])
    graph = sp.coo_matrix(
        (np.ones(len(src), np.int8), (src, dst)), shape=(3 * n_faces, 3 * n_faces)
    )
    fan = connected_components(graph, directed=False)[1]

    # Group corners by (vertex, fan); each group's lowest face orders the fans of its vertex
    vert = f.ravel()
    corner_order = np.lexsort((fan, vert))
    group_vert = vert[corner_order]
    group_fan = fan[corner_order]
    new_group = np.ones(len(vert), dtype=bool)
    new_group[1:] = (group_vert[1:] != group_vert[:-1]) | (
        group_fan[1:] != group_fan[:-1]
    )
    starts = np.flatnonzero(new_group)
    first_face = np.minimum.reduceat(corner_order // 3, starts)
    group_vert = group_vert[starts]

    # Every fan but a vertex's first gets a copy, numbered by vertex then fan
    by_fan = np.lexsort((first_face, group_vert))
    sorted_vert = group_vert[by_fan]
    rank = np.arange(len(by_fan)) - np.searchsorted(
        sorted_vert, sorted_vert, side="left"
    )
    split = by_fan[rank > 0]
    copy_of_group = np.full(len(starts), -1)
    copy_of_group[split] = n_verts + np.arange(len(split))

    # Point each corner of a split fan at its copy
    group_of_corner = np.cumsum(new_group) - 1
    target = copy_of_group[group_of_corner]
    moved = target >= 0
    flat = f.ravel()
    flat[corner_order[moved]] = target[moved]
    return flat.reshape(-1, 3), group_vert[split]


def prepare_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    voxel_size: float,
    max_hole_perimeter_ratio: float = 3.9,
    decimate_max_error: float = 0.5,
    smooth_iterations: int = 0,
    max_faces: int | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Fill, decimate and repair a cleaned mesh into a clean manifold mesh; the input is not modified.

    - fill at full density, decimate, make_manifold; lid the pinholes that opens, repair again
    - smoothing runs last so decimation never sees it; a final repair drops the faces it folds
    - outer rims stay open whatever max_hole_perimeter_ratio (see fill_holes)
    - max_faces caps the decimated face count, past the error bound, so texturing time stays bounded

    Args:
        mesh: the cleaned mesh, the first one clean_repair_mesh returns.
        voxel_size: TSDF voxel the mesh was fused at; sets the decimation bound.
        max_hole_perimeter_ratio: patch holes with a perimeter under this × scene_scale.
        decimate_max_error: decimation bound as a multiple of voxel_size.
        smooth_iterations: volume-keeping relax passes; 0 skips smoothing.
        max_faces: most faces to keep after decimation; None = no limit.

    Returns:
        The filled, decimated, manifold mesh.
    """
    filled = fill_holes(mesh, max_hole_perimeter_ratio=max_hole_perimeter_ratio)
    decimated, err = decimate_mesh(
        filled, max_error=decimate_max_error * voxel_size, max_faces=max_faces
    )
    manifold = make_manifold(decimated)

    # Decimation and repair open pinholes; flat lids close them, then repair what the lids fold
    lidded = fill_holes(
        manifold,
        max_hole_perimeter_ratio=max_hole_perimeter_ratio,
        subdivide_fill=False,
    )
    manifold = make_manifold(lidded)

    # Volume-keeping relax moves vertices only; repair the few faces it folds
    if smooth_iterations > 0:
        mmesh = to_meshlib(manifold)
        relax = mm.MeshRelaxParams()
        relax.iterations = smooth_iterations
        mm.relaxKeepVolume(mmesh, relax)
        manifold = make_manifold(from_meshlib(mmesh, manifold))

    logger.info(
        "prepare_mesh: %d -> %d faces (decimation error %.4f)",
        len(mesh.triangles),
        len(manifold.triangles),
        err,
    )
    return manifold


########################################################################
# Helpers
########################################################################


def _find_ground_plane(
    verts: np.ndarray,
    c2w: np.ndarray,
    *,
    min_up_cos: float,
    max_planes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotation into ground coordinates (rows x, y, up) and the center it is taken about.

    - expected up: mean of the camera path's plane normal and the axis camera right vectors avoid
    - ground: first of max_planes RANSAC planes (fixed seed, earlier inliers removed) facing expected up
    - no plane faces expected up: the largest plane, with a warning
    - up points to the side the cameras are on; flipping rows 1 and 2 together keeps it right-handed
    """
    centers = c2w[:, :3, 3]
    right = c2w[:, :3, 0]

    # Expected up: path-plane normal and right-vector normal, sign-aligned and averaged
    path_up = np.linalg.eigh(np.cov(centers.T))[1][:, 0]
    right_up = np.linalg.eigh(right.T @ right)[1][:, 0]
    expected = path_up + np.sign(path_up @ right_up) * right_up
    expected /= np.linalg.norm(expected)

    # First RANSAC plane that faces expected up; walls and slopes are removed and refit
    keep = np.ones(len(verts), dtype=bool)
    cosines = []
    planes = []

    while len(planes) < max_planes and keep.sum() >= 3:
        rot, t = fit_dominant_plane(verts[keep])
        planes.append((rot, t))
        cosines.append(round(abs(float(rot[2] @ expected)), 3))

        if cosines[-1] >= min_up_cos:
            break

        keep &= np.abs(verts @ rot[2] + t[2]) >= 0.02

    # No plane faces expected up: the largest is the ground
    if cosines[-1] < min_up_cos:
        rot, t = planes[0]
        logger.warning(
            "ground plane: no plane faces the camera up (|normal . up| = %s, below %s); "
            "using the largest plane",
            cosines,
            min_up_cos,
        )

    # Up toward the cameras
    frame = rot.copy()

    if np.median(centers @ rot[2] + t[2]) < 0:
        frame[1:] *= -1

    logger.debug("ground plane: |normal . camera up| per plane %s", cosines)
    return frame, np.median(verts, axis=0)


def get_scene_scale(vertices: np.ndarray) -> float:
    """
    Robust scene extent: diagonal of the 1st-99th percentile bounding box.

    Args:
        vertices: (V, 3) positions.

    Returns:
        Scene extent in world units.
    """
    lo, hi = np.percentile(np.asarray(vertices), [1, 99], axis=0)
    return float(np.linalg.norm(hi - lo))


def _fill_each(mmesh: mm.Mesh, holes: list, params: mm.FillHoleParams) -> int:
    """
    meshlib fillHole on each hole; returns how many raised.
    """
    # A hole meshlib cannot fill raises; count it and go on
    n_failed = 0

    for hole in holes:
        try:
            mm.fillHole(mmesh, hole, params)
        except RuntimeError:
            n_failed += 1

    return n_failed


def _outer_rim_loop(mmesh: mm.Mesh) -> list:
    """
    Edges of the widest boundary loop, the outer rim, as fill_holes picks it; empty with no boundary.

    - a loop's bounding-box diagonal is at most sqrt(3) / 2 of its perimeter, so short loops are skipped
    """
    holes = list(mmesh.topology.findHoleRepresentiveEdges())
    perimeters = np.array([mmesh.holePerimeter(hole) for hole in holes])
    points = mn.getNumpyVerts(mmesh)
    widest, widest_extent = [], -1.0

    # Longest first; stop once no remaining loop can be wider than the widest so far
    for i in np.argsort(-perimeters):
        if np.sqrt(3) / 2 * perimeters[i] < widest_extent:
            break

        loop, extent = _boundary_loop(mmesh, points, holes[i])

        if extent > widest_extent:
            widest, widest_extent = loop, extent

    return widest


def _boundary_loop(
    mmesh: mm.Mesh, points: np.ndarray, hole: mm.EdgeId
) -> tuple[list, float]:
    """
    Edges of the boundary loop through one hole edge, and the bounding-box diagonal of its vertices.
    """
    loop = list(mm.trackRightBoundaryLoop(mmesh.topology, hole))
    loop_pts = points[[mmesh.topology.org(edge).get() for edge in loop]]
    return loop, float(np.linalg.norm(loop_pts.max(axis=0) - loop_pts.min(axis=0)))


def _median_edge(verts: np.ndarray, faces: np.ndarray) -> float:
    """
    Median face edge length: the pixel size of the top-down image of the mesh.
    """
    return float(
        np.median(
            np.linalg.norm(verts[faces] - verts[np.roll(faces, 1, axis=1)], axis=2)
        )
    )


def _drop_small_pieces(
    verts: np.ndarray, faces: np.ndarray, min_faces: int
) -> np.ndarray:
    """
    Faces of the edge-connected pieces with at least min_faces faces.
    """
    cluster_ids, cluster_sizes, _ = face_components(verts, faces)
    return faces[cluster_sizes[cluster_ids] >= min_faces]


########################################################################
# Top-down image of the mesh
########################################################################


def _ground_image_bounds(
    xy: np.ndarray, res: float, *, margin: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Origin and (rows, cols) size of the top-down image of the mesh, padded by margin pixels.

    - top-down image: the ground plane seen from straight above, one pixel = res on a side
    - res is the median edge length, so one pixel is about one TSDF voxel
    """
    lo = xy.min(axis=0) - margin * res
    width, rows = (
        np.ceil((xy.max(axis=0) + margin * res - lo) / res).astype(int) + 1
    ).tolist()
    return lo, (rows, width)


def _ground_xy_to_pixel(xy: np.ndarray, lo: np.ndarray, res: float) -> np.ndarray:
    """
    Pixel (col, row) of the top-down image each ground-plane xy point falls in, int32.
    """
    return ((xy - lo) / res).round().astype(np.int32)


def _mesh_coverage_mask(
    xy: np.ndarray,
    faces: np.ndarray,
    lo: np.ndarray,
    res: float,
    shape: tuple[int, int],
    n_threads: int = 4,
) -> np.ndarray:
    """
    Top-down image of the mesh as a uint8 mask: 255 where a face covers that pixel, else 0.
    """
    # Face corners on the top-down pixel grid
    corners = _ground_xy_to_pixel(xy[faces], lo, res)

    # fillPoly runs on one core: fill face chunks in threads, one image each, then OR them
    chunks = np.array_split(corners, n_threads)

    with ThreadPoolExecutor(n_threads) as pool:
        images = list(pool.map(_fill_faces, chunks, [shape] * n_threads))

    return np.bitwise_or.reduce(images)


def _fill_faces(corners: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    uint8 mask with each (3, 2) pixel-corner face filled to 255.
    """
    img = np.zeros(shape, np.uint8)
    cv2.fillPoly(img, corners, 255)  # type: ignore[call-overload, unused-ignore]
    return img


def _filled_circle(radius: int) -> np.ndarray:
    """
    Filled circle of the given radius in pixels, the shape erode/dilate grow or shrink a mask by.
    """
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )


def _keep_largest_region_fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Largest connected region of a top-down uint8 mask, with the holes inside it filled.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(
        np.uint8
    ) * 255
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled
