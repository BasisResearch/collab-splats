"""
Mesh cleanup at full density, then preparation for UV unwrapping.

- clean_repair_mesh: remove_floaters, optional make_convex_hull, fill_holes; PLY rewritten in place
- remove_floaters: drop components that are small or far from the main body
- fill_holes: patch interior loops under a perimeter bound; the outer rim stays open
- make_convex_hull: trim_mesh_edges, patch the ground out to a rounded hull, bridge_mesh_edges
- prepare_mesh: fill_holes, decimate_mesh (error-bounded QEM), make_manifold into a clean manifold mesh
- thresholds are scene-relative: fractions of get_scene_scale, or cells of the median edge
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import meshlib.mrmeshnumpy as mn
import meshlib.mrmeshpy as mm
import numpy as np
import open3d as o3d
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay, cKDTree

from collab_splats.geometry.transforms import fit_dominant_plane
from collab_splats.utils.image import fill_missing_pixels

logger = logging.getLogger(__name__)


########################
# Entry point
########################


def clean_repair_mesh(
    mesh_path: Path | str,
    min_area_frac: float = 6e-6,
    max_gap_frac: float = 0.01,
    max_hole_perimeter_ratio: float = 0.014,
    subdivide_fill: bool = True,
    use_convex_hull: bool = False,
) -> Path:
    """
    Remove floaters, fill small holes, and overwrite the mesh file.

    Args:
        mesh_path: PLY path; rewritten in place.
        min_area_frac: see remove_floaters.
        max_gap_frac: see remove_floaters.
        max_hole_perimeter_ratio: see fill_holes.
        subdivide_fill: see fill_holes.
        use_convex_hull: run make_convex_hull between the floater cut and the hole fill.

    Returns:
        Path to mesh_path.
    """
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    remove_floaters(mesh, min_area_frac=min_area_frac, max_gap_frac=max_gap_frac)

    if use_convex_hull:
        mesh = make_convex_hull(mesh)

    mesh = fill_holes(mesh, max_hole_perimeter_ratio=max_hole_perimeter_ratio, subdivide_fill=subdivide_fill)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh_path


########################
# Cleanup (full density)
########################


def remove_floaters(
    mesh: o3d.geometry.TriangleMesh,
    min_area_frac: float = 6e-6,
    max_gap_frac: float = 0.01,
    gap_kdtree_points: int = 200_000,
) -> o3d.geometry.TriangleMesh:
    """
    Drop connected components that are tiny or far from the largest one; edits in place.

    Args:
        mesh: mesh edited in place.
        min_area_frac: keep components with area >= this × scene_scale².
        max_gap_frac: keep components whose centroid is within this × scene_scale of the main body.
        gap_kdtree_points: cap on main-body vertices indexed for the gap query (stride-subsampled).

    Returns:
        The same mesh.
    """
    cluster_ids, cluster_sizes, comp_area = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(cluster_ids)
    cluster_sizes = np.asarray(cluster_sizes)
    comp_area = np.asarray(comp_area)

    if len(cluster_sizes) == 0:
        return mesh

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
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
) -> o3d.geometry.TriangleMesh:
    """
    Fill interior boundary loops under a perimeter bound with meshlib's fillHoleNicely.

    - outer rim: a component's longest loop spanning half its extent; never filled, whatever the bound
    - scene_scale: diagonal of the 1st-99th percentile bounding box (get_scene_scale)
    - subdivide_fill: patch split to the mesh's mean edge length, then smoothed to the rim's curvature
    - otherwise the hole's boundary is triangulated only: a flat lid with no new vertices
    - vertex colors carry over; patch vertices take their nearest source vertex's color
    - a hole meshlib cannot fill stays open and is counted in the log

    Args:
        mesh: input mesh; not modified.
        max_hole_perimeter_ratio: fill an interior hole iff its perimeter < this × scene_scale.
        subdivide_fill: subdivide and smooth each patch; off leaves a flat lid, far cheaper.
        max_edge_splits: cap on the subdivisions one patch may take.

    Returns:
        New mesh with those loops closed.
    """
    verts = np.asarray(mesh.vertices)
    gate = max_hole_perimeter_ratio * get_scene_scale(verts)
    mmesh = _to_meshlib(mesh)

    # Weld near-duplicate boundary vertices: a sliver edge between them crashes cotan smoothing natively
    mm.uniteCloseVertices(mmesh, 0.01 * mmesh.averageEdgeLength(), True)

    # Patch settings: subdivide to the mesh's own edge length, cotan-smooth to the hole's boundary
    settings = mm.FillHoleNicelySettings()
    settings.triangulateOnly = not subdivide_fill
    settings.smoothCurvature = True
    settings.subdivideSettings.maxEdgeLen = mmesh.averageEdgeLength()
    settings.subdivideSettings.maxEdgeSplits = max_edge_splits
    settings.smoothSettings.edgeWeights = mm.EdgeWeights.Cotan

    # Component of each hole: the face across its representative edge
    cluster_ids = np.asarray(mesh.cluster_connected_triangles()[0])
    holes = mmesh.topology.findHoleRepresentiveEdges()
    perimeters = np.array([mmesh.holePerimeter(hole) for hole in holes])
    hole_comp = np.array([cluster_ids[mmesh.topology.right(hole).get()] for hole in holes], dtype=np.int64)

    # Per-component bounding boxes, the yardstick for what counts as an outer rim
    tri_pts = verts[np.asarray(mesh.triangles)]
    n_comp = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
    comp_lo = np.full((n_comp, 3), np.inf)
    comp_hi = np.full((n_comp, 3), -np.inf)
    np.minimum.at(comp_lo, cluster_ids, tri_pts.min(axis=1))
    np.maximum.at(comp_hi, cluster_ids, tri_pts.max(axis=1))

    # Find each component's outer rim: its longest loop, if that loop spans half the component or more
    mesh_pts = mn.getNumpyVerts(mmesh)
    longest = {hole_comp[i]: i for i in np.argsort(perimeters)}
    rims = set()

    for i in longest.values():
        loop = [mmesh.topology.org(edge).get() for edge in mm.trackRightBoundaryLoop(mmesh.topology, holes[i])]
        loop_extent = np.linalg.norm(mesh_pts[loop].max(axis=0) - mesh_pts[loop].min(axis=0))
        comp_extent = np.linalg.norm(comp_hi[hole_comp[i]] - comp_lo[hole_comp[i]])

        if loop_extent >= 0.5 * comp_extent:
            rims.add(i)

    # Fill every interior loop under the gate; one degenerate hole must not cost the rest
    n_filled, n_failed = 0, 0

    for i, hole in enumerate(holes):
        if i in rims or perimeters[i] >= gate:
            continue

        try:
            mm.fillHoleNicely(mmesh, hole, settings)
            n_filled += 1
        except RuntimeError:
            n_failed += 1

    # Back to Open3D; normals are recomputed over the patched surface
    filled = _from_meshlib(mmesh, mesh)

    if mesh.has_vertex_normals():
        filled.compute_vertex_normals()

    logger.info(
        "fill_holes: filled %d of %d holes (%d outer rims kept) under perimeter %.4f (%.4f × scene_scale, %d failed), "
        "triangles %d -> %d",
        n_filled,
        len(holes),
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
    *,
    outline_open: int = 8,
    min_up_agreement: float = 0.3,
) -> o3d.geometry.TriangleMesh:
    """
    Cut the ragged outer edge: rim-connected regions outside the smoothed top-down outline.

    - outline: the mesh's coverage mask in the top-down image, closed, opened, largest region, holes filled
    - a region outside the outline survives unless it is edge-connected to the outer rim
    - only faces are removed; vertices and their colors are kept as they are

    Args:
        mesh: input mesh; not modified.
        outline_open: opening radius that smooths the outline, cells of the median edge length.
        min_up_agreement: see make_convex_hull.

    Returns:
        New mesh with the same vertices and the trimmed faces.

    Raises:
        ValueError: the mesh has no dominant ground plane.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimmed = o3d.geometry.TriangleMesh(mesh)

    # Top-down image of the mesh on the ground plane
    frame, center = _find_ground_plane(verts, faces, min_up_agreement=min_up_agreement)
    res = _median_edge(verts, faces)
    xy = ((verts - center) @ frame.T)[:, :2]
    lo, shape = _ground_image_bounds(xy, res, margin=outline_open)

    # Faces whose centroid falls outside the smoothed outline
    outline = cv2.morphologyEx(_mesh_coverage_mask(xy, faces, lo, res, shape), cv2.MORPH_CLOSE, _filled_circle(3))
    outline = _keep_largest_region_fill_holes(cv2.morphologyEx(outline, cv2.MORPH_OPEN, _filled_circle(outline_open)))
    center_px = _ground_xy_to_pixel(xy[faces].mean(axis=1), lo, res)
    outside = outline[center_px[:, 1], center_px[:, 0]] == 0
    mmesh = _to_meshlib(mesh)
    loop = _outer_rim_loop(mmesh)

    if not outside.any() or not loop:
        return trimmed

    # Faces on the outer rim
    rim_faces = np.array([mmesh.topology.right(edge).get() for edge in loop])

    # Group the outside faces into edge-connected regions
    outside_ids = np.flatnonzero(outside)
    pos = np.full(len(faces), -1)
    pos[outside_ids] = np.arange(len(outside_ids))
    outside_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces[outside_ids])
    )
    region = np.asarray(outside_mesh.cluster_connected_triangles()[0])

    # Drop the regions that touch the outer rim
    seeds = np.unique(region[pos[rim_faces[outside[rim_faces]]]])
    drop = np.zeros(len(faces), dtype=bool)
    drop[outside_ids[np.isin(region, seeds)]] = True
    trimmed.remove_triangles_by_mask(drop)
    logger.info("trim_mesh_edges: dropped %d faces in %d rim-connected regions", int(drop.sum()), len(seeds))
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
    mmesh = _to_meshlib(mesh)
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
    return _from_meshlib(mmesh, mesh)


########################
# Convex hull
########################


def make_convex_hull(
    mesh: o3d.geometry.TriangleMesh,
    *,
    hull_round: int = 20,
    outline_open: int = 8,
    rim_max_dz: float = 3.0,
    rim_max_edge: float = 3.0,
    bridge_radius: float = 3.0,
    min_piece_faces: int = 1000,
    min_up_agreement: float = 0.3,
) -> o3d.geometry.TriangleMesh:
    """
    Trim the ragged outer edge, then patch the ground out to a rounded convex hull.

    - lengths are in cells: pixels of the top-down image of the mesh, one median edge length each
    - up: the dominant plane's normal, on the side the area-weighted face normal points to
    - assumes a ground-dominated height field; indoor and object scenes should leave it off
    - necks of the outer rim are bridged, so inlets become interior holes fill_holes can close

    Args:
        mesh: input mesh; not modified.
        hull_round: opening radius that rounds the hull's corners, cells.
        outline_open: see trim_mesh_edges.
        rim_max_dz: join a rim vertex only within this of its neighbours' median height, cells.
        rim_max_edge: drop patch triangles touching the rim with a 3D edge over this, cells.
        bridge_radius: see bridge_mesh_edges (radius).
        min_piece_faces: pieces with fewer faces are dropped.
        min_up_agreement: smallest |mean face normal . plane normal| that counts as a ground.

    Returns:
        New mesh: trimmed, patched out to the hull, manifold.

    Raises:
        ValueError: the mesh has no dominant ground plane.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full((len(verts), 3), 0.5)

    # Ground frame; top-down image of the mesh, with a margin for the hull's rounding
    frame, center = _find_ground_plane(verts, faces, min_up_agreement=min_up_agreement)
    res = _median_edge(verts, faces)
    local = (verts - center) @ frame.T
    lo, shape = _ground_image_bounds(local[:, :2], res, margin=hull_round)
    vert_px = _ground_xy_to_pixel(local[:, :2], lo, res)

    # Hull cut: faces outside the rounded hull, then the pieces the cut strands
    inside = _create_rounded_hull_mask(_mesh_coverage_mask(local[:, :2], faces, lo, res, shape), hull_round)
    faces = faces[inside[vert_px[:, 1], vert_px[:, 0]][faces].all(axis=1)]
    faces = _drop_small_pieces(verts, faces, min_piece_faces)

    # Outline trim, then the pieces it strands
    cut = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    trimmed = trim_mesh_edges(cut, outline_open=outline_open, min_up_agreement=min_up_agreement)
    faces = _drop_small_pieces(verts, np.asarray(trimmed.triangles), min_piece_faces)

    # Ground patch out to the hull, joined to the mesh; grid points are appended to the vertices
    grid_local, grid_colors, patch = _connect_mesh_hull(
        faces, local, colors, inside, lo, res, rim_max_dz=rim_max_dz, rim_max_edge=rim_max_edge
    )
    all_verts = np.concatenate([verts, grid_local @ frame + center])
    all_faces = _drop_small_pieces(all_verts, np.concatenate([faces, patch]), min_piece_faces)

    # Repair, then bridge the outer rim's necks
    joined = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(all_verts), o3d.utility.Vector3iVector(all_faces))
    joined.vertex_colors = o3d.utility.Vector3dVector(np.concatenate([colors, grid_colors]))
    joined = bridge_mesh_edges(make_manifold(joined), radius=bridge_radius)

    if not mesh.has_vertex_colors():
        joined.vertex_colors = o3d.utility.Vector3dVector()

    logger.info(
        "make_convex_hull: cell %.4f, %d patch faces, triangles %d -> %d",
        res,
        len(patch),
        len(mesh.triangles),
        len(joined.triangles),
    )
    return joined


def _create_rounded_hull_mask(coverage: np.ndarray, hull_round: int) -> np.ndarray:
    """
    Rounded convex hull of the mesh's main region, as a bool mask in the top-down image.

    - main region: closed, largest region, holes filled, then opened so thin spurs do not stretch the hull
    - corners rounded by an opening of hull_round cells
    """
    solid = _keep_largest_region_fill_holes(cv2.morphologyEx(coverage, cv2.MORPH_CLOSE, _filled_circle(3)))
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, _filled_circle(15))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros_like(coverage)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(contours))], 255)
    return cv2.morphologyEx(hull, cv2.MORPH_OPEN, _filled_circle(hull_round)) > 0


def _connect_mesh_hull(
    faces: np.ndarray,
    local: np.ndarray,
    colors: np.ndarray,
    inside: np.ndarray,
    lo: np.ndarray,
    res: float,
    *,
    rim_max_dz: float,
    rim_max_edge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ground patch from the mesh's open rim out to the hull, joined to the mesh without clashes.

    - returns grid points (G, 3) in the local frame, their colors, and patch faces indexing [mesh; grid]
    - prior: per-cell min height, eroded, gaps filled by fill_missing_pixels
    - heights and colors: harmonic, rim pinned, pulled to the prior away from the mesh
    - faces wound up; a flip fixes a repeated directed edge, else faces on over-used edges are dropped
    """
    xy, height = local[:, :2], local[:, 2]
    shape = inside.shape
    vert_px = _ground_xy_to_pixel(xy, lo, res)

    # Gap: hull cells no face covers from above; distance to the mesh in cells
    covered = _mesh_coverage_mask(xy, faces, lo, res, shape) > 0
    gap = inside & ~covered
    dist_mesh = cv2.distanceTransform((~covered).astype(np.uint8), cv2.DIST_L2, 5)

    # Ground prior: per-cell min height, eroded so walls and objects sink to the ground
    empty = np.finfo(np.float32).max
    used = np.unique(faces)
    lowest = np.full(inside.size, empty, dtype=np.float32)
    np.minimum.at(lowest, vert_px[used, 1] * shape[1] + vert_px[used, 0], height[used])
    ground = cv2.erode(lowest.reshape(shape), _filled_circle(6))
    known = ground < empty

    # Empty cells take the smooth continuation of their known neighbors
    filled = fill_missing_pixels(np.where(known, ground, 0.0), known)
    prior = cv2.GaussianBlur(filled, (0, 0), 5)

    # Rim vertices: on open edges, bordering the gap
    edge_ids, per_edge = np.unique(_face_edge_ids(faces, len(local), directed=False), return_counts=True)
    open_edges = edge_ids[per_edge == 1]
    rim = np.unique(np.concatenate([open_edges // len(local), open_edges % len(local)]))
    near_gap = cv2.dilate(gap.astype(np.uint8), _filled_circle(2)).astype(bool)
    rim = rim[near_gap[vert_px[rim, 1], vert_px[rim, 0]]]

    # Height gate: a rim vertex far off its neighbours' median (wall top, spike) stays unjoined
    rim = rim[np.abs(height[rim] - _local_median(xy[rim], height[rim], 6 * res)) < rim_max_dz * res]
    rim_height = _local_median(xy[rim], height[rim], 6 * res)
    n_rim = len(rim)

    # Grid points in the gap, one cell clear of the mesh
    clear = cv2.distanceTransform(gap.astype(np.uint8), cv2.DIST_L2, 5) >= 1.0
    rows, cols = np.nonzero(clear)
    grid_xy = np.stack([lo[0] + cols * res, lo[1] + rows * res], axis=1)
    n_grid = len(grid_xy)

    # Triangulate rim + grid; keep short triangles over the gap, rim-only ones shorter still
    points = np.concatenate([xy[rim], grid_xy])
    tris = Delaunay(points).simplices
    center_px = _ground_xy_to_pixel(points[tris].mean(axis=1), lo, res)
    gap_wide = cv2.dilate(gap.astype(np.uint8), _filled_circle(1)).astype(bool)
    keep = gap_wide[center_px[:, 1], center_px[:, 0]] & inside[center_px[:, 1], center_px[:, 0]]
    longest = np.linalg.norm(points[tris] - points[np.roll(tris, 1, axis=1)], axis=2).max(axis=1)
    keep &= longest < 3 * res
    keep &= ~(tris < n_rim).all(axis=1) | (longest < 1.5 * res)
    tris = tris[keep]

    # Build a smooth height + color system on the grid, pinned to the rim and pulled to the ground prior
    tri_edges = np.unique(_face_edge_ids(tris, len(points), directed=False))
    ends = np.stack([tri_edges // len(points), tri_edges % len(points)], axis=1)
    src = np.concatenate([ends[:, 0], ends[:, 1]])
    dst = np.concatenate([ends[:, 1], ends[:, 0]])
    on_grid = src >= n_rim
    src, dst = src[on_grid] - n_rim, dst[on_grid]
    pull = 1e-6 + 0.5 * np.clip(dist_mesh[rows, cols] / 20, 0, 1) ** 2
    target = np.column_stack([prior[rows, cols].astype(np.float64), np.tile(np.median(colors, axis=0), (n_grid, 1))])
    pinned = np.column_stack([rim_height, colors[rim]])
    rhs = pull[:, None] * target
    to_rim = dst < n_rim
    np.add.at(rhs, src[to_rim], pinned[dst[to_rim]])
    laplacian = sp.csc_matrix(
        (-np.ones((~to_rim).sum()), (src[~to_rim], dst[~to_rim] - n_rim)), shape=(n_grid, n_grid)
    ) + sp.diags(pull + np.bincount(src, minlength=n_grid))

    # Solve: grid heights and colors
    solution = splu(laplacian.tocsc()).solve(rhs)
    grid_local = np.concatenate([grid_xy, solution[:, :1]], axis=1)
    grid_colors = np.clip(solution[:, 1:], 0, 1)

    # 3D edge cap on rim triangles: one reaching up a wall or to a spike is a strand
    points3 = np.concatenate([np.column_stack([xy[rim], height[rim]]), grid_local])
    longest3 = np.linalg.norm(points3[tris] - points3[np.roll(tris, 1, axis=1)], axis=2).max(axis=1)
    tris = tris[(longest3 < rim_max_edge * res) | ~(tris < n_rim).any(axis=1)]

    # Wind every face up (+z), then renumber from [rim; grid] to [mesh; grid]
    corners = points3[tris]
    down = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2] < 0
    tris[down] = tris[down][:, [0, 2, 1]]
    patch = np.concatenate([rim, len(local) + np.arange(n_grid)])[tris]
    n_verts = len(local) + n_grid

    # Flip a face that repeats a mesh directed edge, if the flip repeats none
    mesh_ids = np.unique(_face_edge_ids(faces, n_verts, directed=True))
    repeats = np.isin(_face_edge_ids(patch, n_verts, directed=True), mesh_ids).any(axis=1)
    flipped = patch[repeats][:, [0, 2, 1]]
    ok = ~np.isin(_face_edge_ids(flipped, n_verts, directed=True), mesh_ids).any(axis=1)
    patch[np.flatnonzero(repeats)[ok]] = flipped[ok]

    # Drop faces on an edge of >2 faces or a repeated direction; each pass shrinks the patch
    while True:
        all_faces = np.concatenate([faces, patch])
        _, edge_of, per_edge = np.unique(
            _face_edge_ids(all_faces, n_verts, directed=False), return_inverse=True, return_counts=True
        )
        _, dir_of, per_dir = np.unique(
            _face_edge_ids(all_faces, n_verts, directed=True), return_inverse=True, return_counts=True
        )
        bad = (per_edge[edge_of.reshape(-1, 3)[len(faces) :]] > 2).any(axis=1)
        bad |= (per_dir[dir_of.reshape(-1, 3)[len(faces) :]] > 1).any(axis=1)

        if not bad.any():
            break

        patch = patch[~bad]

    return grid_local, grid_colors, patch


########################
# Prepare for UV unwrap
########################


def decimate_mesh(mesh: o3d.geometry.TriangleMesh, *, max_error: float) -> tuple[o3d.geometry.TriangleMesh, float]:
    """
    QEM decimation to an absolute surface-deviation bound; vertices are removed, never moved.

    Args:
        mesh: input mesh; not modified.
        max_error: largest allowed surface deviation in world units.

    Returns:
        The decimated mesh and the result error in world units.
    """
    # Collapse edges with meshlib QEM under an absolute error bound, keeping vertex positions
    mmesh = _to_meshlib(mesh)
    settings = mm.DecimateSettings()
    settings.maxError = float(max_error)
    settings.optimizeVertexPos = False
    settings.packMesh = True
    settings.subdivideParts = 64
    result = mm.decimateMesh(mmesh, settings)
    return _from_meshlib(mmesh, mesh), float(result.errorIntroduced)


def make_manifold(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Repair a mesh to a clean manifold; the input is left untouched.

    - duplicate and fold-over faces pass Open3D's manifold check
    - Open3D's remove_duplicated_triangles misses a duplicate with reversed winding

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
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    n_dup = n_in - len(f)

    # Drop fold-over faces: keep the first face using each directed edge, drop later ones
    _, first_edge = np.unique(_face_edge_ids(f, len(v), directed=True).T.reshape(-1), return_index=True)
    later = np.ones(3 * len(f), dtype=bool)
    later[first_edge] = False
    fold = later.reshape(3, -1).any(axis=0)
    f = f[~fold]

    # Non-manifold edges: Open3D drops faces until each edge has at most two
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    mesh.remove_non_manifold_edges()
    f = np.asarray(mesh.triangles).copy()

    # Split bowtie vertices: each extra fan of faces around a vertex gets its own copy
    new_ids = []

    for vid in np.asarray(mesh.get_non_manifold_vertices(), dtype=np.int64):
        tris = np.flatnonzero((f == vid).any(axis=1))
        others = f[tris][f[tris] != vid].reshape(-1, 2)
        share = (others[:, None, :, None] == others[None, :, None, :]).any(axis=(2, 3))
        _, fan = connected_components(share, directed=False)

        for k in range(1, fan.max() + 1):
            fan_faces = f[tris[fan == k]]
            f[tris[fan == k]] = np.where(fan_faces == vid, len(v) + len(new_ids), fan_faces)
            new_ids.append(vid)

    # Rebuild with the copies appended, then drop vertices no face references
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack([v, v[new_ids]])), o3d.utility.Vector3iVector(f)
    )

    if colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(np.vstack([colors, colors[new_ids]]))

    out.remove_unreferenced_vertices()
    logger.info(
        "make_manifold: dropped %d duplicate + %d fold-over faces, split %d bowtie vertices",
        n_dup,
        int(fold.sum()),
        len(new_ids),
    )
    return out


def prepare_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    voxel_size: float,
    max_hole_perimeter_ratio: float = 3.9,
    decimate_max_error: float = 0.5,
    smooth_iterations: int = 0,
) -> o3d.geometry.TriangleMesh:
    """
    Fill, decimate and repair a cleaned mesh into a clean manifold mesh; the input is not modified.

    - fill at full density, decimate, make_manifold; lid the pinholes that opens, repair again
    - smoothing runs last so decimation never sees it; a final repair drops the faces it folds
    - outer rims stay open whatever max_hole_perimeter_ratio (see fill_holes)

    Args:
        mesh: cleaned mesh (clean_repair_mesh output).
        voxel_size: TSDF voxel the mesh was fused at; sets the decimation bound.
        max_hole_perimeter_ratio: patch holes with a perimeter under this × scene_scale.
        decimate_max_error: decimation bound as a multiple of voxel_size.
        smooth_iterations: Taubin smoothing passes; 0 skips smoothing.

    Returns:
        The filled, decimated, manifold mesh.
    """
    filled = fill_holes(mesh, max_hole_perimeter_ratio=max_hole_perimeter_ratio)
    decimated, err = decimate_mesh(filled, max_error=decimate_max_error * voxel_size)
    manifold = make_manifold(decimated)

    # Decimation and repair open pinholes; flat lids close them, then repair what the lids fold
    lidded = fill_holes(manifold, max_hole_perimeter_ratio=max_hole_perimeter_ratio, subdivide_fill=False)
    manifold = make_manifold(lidded)

    # Taubin smoothing moves vertices only; repair the few faces it folds
    if smooth_iterations > 0:
        smoothed = manifold.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        manifold = make_manifold(smoothed)

    logger.info(
        "prepare_mesh: %d -> %d faces (decimation error %.4f)", len(mesh.triangles), len(manifold.triangles), err
    )
    return manifold


########################
# Helpers
########################


def _find_ground_plane(
    verts: np.ndarray, faces: np.ndarray, *, min_up_agreement: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotation into ground coordinates (rows x, y, up) and the center it is taken about.

    - RANSAC plane with a fixed seed: the same mesh always gets the same frame
    - up: the side the area-weighted mean face normal points to; it shrinks as walls cancel
    - flipping rows 1 and 2 together keeps the frame right-handed
    """
    o3d.utility.random.seed(0)
    rot, _ = fit_dominant_plane(verts)

    # Agreement of the mean face normal with the plane normal; too weak means no ground
    corners = verts[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    agree = float(face_normals.sum(axis=0) @ rot[2] / np.linalg.norm(face_normals, axis=1).sum())

    if abs(agree) < min_up_agreement:
        raise ValueError(
            f"no dominant ground plane (mean face normal . plane normal = {agree:.3f}, "
            f"below {min_up_agreement}) — leave use_convex_hull off for this scene."
        )

    frame = rot.copy()

    if agree < 0:
        frame[1:] *= -1

    logger.debug("ground plane: up agreement %.3f", agree)
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


def _to_meshlib(mesh: o3d.geometry.TriangleMesh) -> mm.Mesh:
    """
    meshlib copy of an Open3D mesh's vertices and faces.
    """
    return mn.meshFromFacesVerts(
        np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.int32),
        np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float64),
    )


def _from_meshlib(mmesh: mm.Mesh, source: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Open3D mesh from a meshlib one; vertex colors from source's nearest vertex.

    - packs mmesh first: getNumpyFaces writes deleted face slots as zero rows
    """
    mmesh.pack()
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mn.getNumpyVerts(mmesh).astype(np.float64)),
        o3d.utility.Vector3iVector(mn.getNumpyFaces(mmesh.topology).astype(np.int64)),
    )

    if source.has_vertex_colors():
        _, nearest = cKDTree(np.asarray(source.vertices)).query(np.asarray(out.vertices), k=1, workers=-1)
        out.vertex_colors = o3d.utility.Vector3dVector(np.asarray(source.vertex_colors)[nearest])

    return out


def _outer_rim_loop(mmesh: mm.Mesh) -> list:
    """
    Edges of the longest boundary loop, the outer rim; empty when the mesh has no boundary.
    """
    holes = mmesh.topology.findHoleRepresentiveEdges()

    if not holes:
        return []

    outer = max(holes, key=mmesh.holePerimeter)
    return list(mm.trackRightBoundaryLoop(mmesh.topology, outer))


def _median_edge(verts: np.ndarray, faces: np.ndarray) -> float:
    """
    Median face edge length: the pixel size of the top-down image of the mesh.
    """
    return float(np.median(np.linalg.norm(verts[faces] - verts[np.roll(faces, 1, axis=1)], axis=2)))


def _face_edge_ids(faces: np.ndarray, n_verts: int, *, directed: bool) -> np.ndarray:
    """
    An id for each of a face's 3 edges, (F, 3); two faces sharing an edge get the same id.

    - id of edge a->b: a * n_verts + b, so id // n_verts and id % n_verts give its ends back
    - directed on: a->b and b->a get different ids (winding matters)
    - directed off: a->b and b->a share one id, the smaller end first
    """
    start = faces.astype(np.int64)
    end = start[:, [1, 2, 0]]

    if not directed:
        start, end = np.minimum(start, end), np.maximum(start, end)

    return start * n_verts + end


def _drop_small_pieces(verts: np.ndarray, faces: np.ndarray, min_faces: int) -> np.ndarray:
    """
    Faces of the edge-connected pieces with at least min_faces faces.
    """
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
    return faces[np.asarray(cluster_sizes)[np.asarray(cluster_ids)] >= min_faces]


def _local_median(xy: np.ndarray, values: np.ndarray, radius: float) -> np.ndarray:
    """
    Median of values over the points within radius of each point, in the ground plane.
    """
    neighbours = cKDTree(xy).query_ball_point(xy, radius, workers=-1)
    return np.array([np.median(values[ix]) for ix in neighbours])


########################
# Top-down image of the mesh
########################


def _ground_image_bounds(xy: np.ndarray, res: float, *, margin: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Origin and (rows, cols) size of the top-down image of the mesh, padded by margin pixels.

    - top-down image: the ground plane seen from straight above, one pixel = res on a side
    - res is the median edge length, so one pixel is about one TSDF voxel
    """
    lo = xy.min(axis=0) - margin * res
    width, rows = (np.ceil((xy.max(axis=0) + margin * res - lo) / res).astype(int) + 1).tolist()
    return lo, (rows, width)


def _ground_xy_to_pixel(xy: np.ndarray, lo: np.ndarray, res: float) -> np.ndarray:
    """
    Pixel (col, row) of the top-down image each ground-plane xy point falls in, int32.
    """
    return ((xy - lo) / res).round().astype(np.int32)


def _mesh_coverage_mask(
    xy: np.ndarray, faces: np.ndarray, lo: np.ndarray, res: float, shape: tuple[int, int]
) -> np.ndarray:
    """
    Top-down image of the mesh as a uint8 mask: 255 where a face covers that pixel, else 0.
    """
    img = np.zeros(shape, np.uint8)
    cv2.fillPoly(img, list(_ground_xy_to_pixel(xy[faces], lo, res)), 255)
    return img


def _filled_circle(radius: int) -> np.ndarray:
    """
    Filled circle of the given radius in pixels, the shape erode/dilate grow or shrink a mask by.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _keep_largest_region_fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Largest connected region of a top-down uint8 mask, with the holes inside it filled.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled
