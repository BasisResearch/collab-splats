"""
Mesh helpers shared across tsdf, clean and texture.

- to_meshlib / from_meshlib: Open3D <-> meshlib conversion
- face_edge_ids / adjacent_face_pairs / face_components / face_areas: per-face geometry and connectivity
- validate_views: the view-array contract for fusion and texturing
"""

from __future__ import annotations

import meshlib.mrmeshnumpy as mn
import meshlib.mrmeshpy as mm
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

########################################################################
# meshlib conversion
########################################################################


def to_meshlib(mesh: o3d.geometry.TriangleMesh) -> mm.Mesh:
    """
    meshlib copy of an Open3D mesh's vertices and faces.

    Args:
        mesh: source mesh; not modified.

    Returns:
        New meshlib mesh.
    """
    return mn.meshFromFacesVerts(
        np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.int32),
        np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float64),
    )


def from_meshlib(
    mmesh: mm.Mesh, source: o3d.geometry.TriangleMesh
) -> o3d.geometry.TriangleMesh:
    """
    Open3D mesh from a meshlib one; vertex colors from source's nearest vertex.

    - packs mmesh first: getNumpyFaces writes deleted face slots as zero rows

    Args:
        mmesh: meshlib mesh; packed in place.
        source: mesh whose vertex colors carry over, if it has any.

    Returns:
        New Open3D mesh.
    """
    # Compact deleted slots, then copy vertices and faces across
    mmesh.pack()
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mn.getNumpyVerts(mmesh).astype(np.float64)),
        o3d.utility.Vector3iVector(mn.getNumpyFaces(mmesh.topology).astype(np.int64)),
    )

    # Colors: each new vertex takes its nearest source vertex's
    if source.has_vertex_colors():
        _, nearest = cKDTree(np.asarray(source.vertices)).query(
            np.asarray(out.vertices), k=1, workers=-1
        )
        out.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(source.vertex_colors)[nearest]
        )

    return out


########################################################################
# Face connectivity
########################################################################


def face_edge_ids(faces: np.ndarray, n_verts: int, *, directed: bool) -> np.ndarray:
    """
    An id for each of a face's 3 edges, (F, 3); two faces sharing an edge get the same id.

    - id of edge a->b: a * n_verts + b, so id // n_verts and id % n_verts give its ends back
    - directed on: a->b and b->a get different ids (winding matters)
    - directed off: a->b and b->a share one id, the smaller end first

    Args:
        faces: (F, 3) vertex indices.
        n_verts: vertex count, the id radix.
        directed: keep edge direction in the id.

    Returns:
        (F, 3) int64 edge ids.
    """
    # Edge k of a face runs from corner k to corner k + 1
    start = faces.astype(np.int64)
    end = start[:, [1, 2, 0]]

    # Undirected: smaller end first, so both windings share one id
    if not directed:
        start, end = np.minimum(start, end), np.maximum(start, end)

    return start * n_verts + end


def adjacent_face_pairs(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """
    Edge-adjacent face pairs, one per consecutive pair of faces sharing an edge.

    - an edge shared by k faces gives k - 1 pairs, enough to connect them all

    Args:
        faces: (F, 3) vertex indices.
        n_verts: vertex count.

    Returns:
        (P, 2) face indices.
    """
    # Sort every face edge by id; neighbors with one id share that edge
    edge = face_edge_ids(faces, n_verts, directed=False).T.ravel()
    face = np.tile(np.arange(len(faces)), 3)
    order = np.argsort(edge, kind="stable")
    edge, face = edge[order], face[order]
    same = edge[1:] == edge[:-1]
    return np.stack([face[:-1][same], face[1:][same]], 1)


def face_components(
    verts: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Edge-connected components of a face array, as Open3D's cluster_connected_triangles returns them.

    - scipy connected_components over adjacent_face_pairs; same partition as Open3D, far faster

    Args:
        verts: (V, 3) positions.
        faces: (F, 3) vertex indices.

    Returns:
        (ids, sizes, areas)
        - ids: (F,) component of each face
        - sizes: (C,) face count per component
        - areas: (C,) surface area per component
    """
    # Face adjacency graph; its connected components are the pieces
    n_faces = len(faces)
    pairs = adjacent_face_pairs(faces, len(verts))
    graph = coo_matrix(
        (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n_faces, n_faces),
    )
    ids = connected_components(graph, directed=False)[1]

    # Face count and summed area per component
    areas = face_areas(verts, faces)

    return ids, np.bincount(ids), np.bincount(ids, weights=areas)


def face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Surface area of every triangle, half its edge cross-product norm.

    Args:
        verts: (V, 3) positions.
        faces: (F, 3) vertex indices.

    Returns:
        (F,) areas in world units squared.
    """
    # Half the norm of two edge vectors' cross product
    corners = verts[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return 0.5 * np.linalg.norm(normals, axis=1)


########################################################################
# View arrays
########################################################################


def validate_views(
    rgbs: np.ndarray, c2w: np.ndarray, K: np.ndarray, depths: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Coerce the view arrays and reject dtype, count and resolution mismatches.

    - shared by create_tsdf_mesh and create_texture_mesh
    - depths, when given, must match the rgbs resolution

    Args:
        rgbs: (N, H, W, 3) uint8 views.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at the rgbs resolution.
        depths: optional (N, H, W) depth maps.

    Returns:
        rgbs, c2w and K (float64) and depths as numpy arrays; depths stays None when not given.

    Raises:
        ValueError: dtype, view count or resolution mismatch.
    """
    # Plain numpy arrays; float64 geometry so float32 poses never meet float64 vertices
    rgbs = np.asarray(rgbs)
    c2w = np.asarray(c2w, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # Input contract: uint8 color, one view count, K at the image resolution
    if rgbs.dtype != np.uint8:
        raise ValueError(f"rgbs must be uint8 in [0, 255], got {rgbs.dtype}")

    n, h, w = rgbs.shape[:3]

    if rgbs.shape != (n, h, w, 3) or c2w.shape != (n, 4, 4) or K.shape != (n, 3, 3):
        raise ValueError(
            f"views disagree: rgbs {rgbs.shape}, c2w {c2w.shape}, K {K.shape}"
        )

    if depths is not None:
        depths = np.asarray(depths)

        if depths.shape != (n, h, w):
            raise ValueError(
                f"views disagree: depths {depths.shape}, rgbs {rgbs.shape}"
            )

    # Principal point inside the image: K and rgbs share one resolution
    cx, cy = K[:, 0, 2], K[:, 1, 2]

    if cx.min() < 0 or cx.max() > w or cy.min() < 0 or cy.max() > h:
        raise ValueError(
            f"Principal point outside the {w}x{h} image grid (cx range [{cx.min():.1f}, {cx.max():.1f}], "
            f"cy range [{cy.min():.1f}, {cy.max():.1f}]) — intrinsics and images are at different resolutions."
        )

    return rgbs, c2w, K, depths
