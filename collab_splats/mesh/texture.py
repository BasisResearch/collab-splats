"""
Texture a fused mesh.

  - decimate_mesh: error-bounded QEM decimation
  - _make_manifold: repair to what UVAtlas accepts
  - unwrap_mesh_uvs: UV atlas over the repaired mesh
  - project_images_to_texture: visibility-weighted projection of the source views
"""

from __future__ import annotations

import logging
import time

import cv2
import meshoptimizer as mo
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
import warp as wp

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses
from collab_splats.mesh.io import write_textured_ply

logger = logging.getLogger(__name__)


######## Decimation


def decimate_mesh(mesh, max_error):
    """
    QEM decimation to an absolute surface-deviation bound; vertices are removed, never moved.

    Args:
        mesh: open3d.geometry.TriangleMesh.
        max_error: float, largest allowed surface deviation in world units.
    Returns:
        (decimated open3d.geometry.TriangleMesh, float result error in world units).
    """
    # meshoptimizer simplify under an absolute error bound; target_index_count=3 = as few as allowed
    v = np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float32)
    idx = np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.uint32).ravel()
    dst = np.zeros_like(idx)
    err = np.zeros(1, dtype=np.float32)
    n = mo.simplify(
        dst,
        idx,
        v,
        target_index_count=3,
        target_error=float(max_error),
        options=mo.SIMPLIFY_ERROR_ABSOLUTE,
        result_error=err,
    )

    # Rebuild as a legacy mesh and drop the vertices no triangle references any more
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(dst[:n].reshape(-1, 3).astype(np.int32)),
    )
    out.remove_unreferenced_vertices()
    return out, float(err[0])


######## Manifold repair — UVAtlas rejects what Open3D calls manifold


def _find(parent, x):
    """
    Union-find root of x with path halving.

    Args:
        parent: dict, node -> parent node.
        x: int, node id.
    Returns:
        int, root node of x.
    """
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _split_non_manifold_vertices(mesh):
    """
    Duplicate every bowtie vertex once per extra edge-connected triangle fan.

    Args:
        mesh: open3d.geometry.TriangleMesh; not modified.
    Returns:
        (new open3d.geometry.TriangleMesh, int count of vertices split).
    """
    v = np.asarray(mesh.vertices).copy()
    f = np.asarray(mesh.triangles).copy()
    colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
    nm = np.asarray(mesh.get_non_manifold_vertices(), dtype=np.int64)
    if len(nm) == 0:
        return o3d.geometry.TriangleMesh(mesh), 0

    # Vertex -> incident-triangle index, built once from the flattened corner array
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(len(v) + 1))
    new_v, new_c = [], []
    n_split = 0
    for vid in nm:
        tris = order[starts[vid] : starts[vid + 1]] // 3

        # Union-find over incident triangles: same fan iff they share a vertex besides vid
        parent = {int(t): int(t) for t in tris}
        other = {}
        for t in tris:
            for w in f[t]:
                if w != vid:
                    other.setdefault(int(w), []).append(int(t))
        for group in other.values():
            for t in group[1:]:
                parent[_find(parent, t)] = _find(parent, group[0])
        fans = {}
        for t in tris:
            fans.setdefault(_find(parent, int(t)), []).append(int(t))

        # First fan keeps vid; every further fan is rewired to a fresh copy
        for fan in list(fans.values())[1:]:
            nid = len(v) + len(new_v)
            new_v.append(v[vid])
            if colors is not None:
                new_c.append(colors[vid])
            for t in fan:
                f[t][f[t] == vid] = nid
            n_split += 1

    if new_v:
        v = np.vstack([v, np.asarray(new_v)])
        if colors is not None:
            colors = np.vstack([colors, np.asarray(new_c)])
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    if colors is not None:
        out.vertex_colors = o3d.utility.Vector3dVector(colors)
    return out, n_split


def _drop_duplicate_and_fold_over_faces(mesh):
    """
    Drop duplicate faces in any winding and fold-overs (a directed edge owned by two faces).

    Args:
        mesh: open3d.geometry.TriangleMesh with degenerate faces already removed.
    Returns:
        (new open3d.geometry.TriangleMesh, int duplicates dropped, int fold-overs dropped).
    """
    f = np.asarray(mesh.triangles)
    n_in = len(f)
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    n_dup = n_in - len(f)

    # Directed-edge multiplicity: a manifold orientable surface uses each direction once
    #   - the first face owning a direction keeps it
    #   - any face owning a non-first copy is dropped
    de = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, first_edge = np.unique(de, axis=0, return_index=True)
    non_first = np.ones(len(de), dtype=bool)
    non_first[first_edge] = False
    fold = non_first.reshape(3, -1).any(axis=0)
    f = f[~fold]
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)), o3d.utility.Vector3iVector(f)
    )
    if mesh.has_vertex_colors():
        out.vertex_colors = mesh.vertex_colors
    return out, n_dup, int(fold.sum())


def _make_manifold(mesh):
    """
    Repair a mesh to a UVAtlas-ready one; the input is left untouched.

    Args:
        mesh: open3d.geometry.TriangleMesh.
    Returns:
        new open3d.geometry.TriangleMesh without degenerate, duplicate or fold-over faces,
        non-manifold edges or vertices, or orphan vertices.
    """
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_degenerate_triangles()
    mesh, n_dup, n_fold = _drop_duplicate_and_fold_over_faces(mesh)
    mesh.remove_non_manifold_edges()
    mesh, n_split = _split_non_manifold_vertices(mesh)
    mesh.remove_unreferenced_vertices()
    logger.info(
        "_make_manifold: dropped %d duplicate + %d fold-over faces, split %d bowtie vertices", n_dup, n_fold, n_split
    )
    return mesh


######## UV atlas


def unwrap_mesh_uvs(mesh, tex_size, parallel_partitions=16, min_faces_per_partition=1000):
    """
    Compute a UV atlas with Open3D's UVAtlas.

    Args:
        mesh: manifold open3d.geometry.TriangleMesh (see _make_manifold).
        tex_size: int, atlas edge in texels.
        parallel_partitions: int, UVAtlas partitions run in parallel (1 = single-threaded, 20+ min at 500k faces).
        min_faces_per_partition: int, floor clamping the partition count (Open3D's PCA partition
            raises on an empty one).
    Returns:
        o3d.t.geometry.TriangleMesh with triangle.texture_uvs (F, 3, 2).
    """
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    partitions = max(1, min(int(parallel_partitions), len(mesh.triangles) // min_faces_per_partition))
    tm.compute_uvatlas(size=tex_size, parallel_partitions=partitions)
    return tm


def bake_atlas_attributes(tm, tex_size):
    """
    Rasterize a UV atlas into per-texel world position and normal.

    Args:
        tm: o3d.t.geometry.TriangleMesh with triangle.texture_uvs (from unwrap_mesh_uvs).
        tex_size: int, atlas edge in texels.
    Returns:
        (positions, normals), two (tex_size, tex_size, 3) float32 arrays; zero where no triangle covers.
    """
    tm.compute_vertex_normals()
    verts = tm.vertex.positions.numpy()
    faces = tm.triangle.indices.numpy()
    vert_normals = tm.vertex.normals.numpy()

    # One vertex per triangle corner
    #   - a seam vertex carries a different UV in each triangle sharing it
    #   - a shared-vertex buffer therefore cannot express the atlas
    corner_pos = np.ascontiguousarray(verts[faces].reshape(-1, 3), dtype=np.float32)
    corner_nrm = np.ascontiguousarray(vert_normals[faces].reshape(-1, 3), dtype=np.float32)
    corner_tri = np.arange(len(corner_pos), dtype=np.int32).reshape(-1, 3)

    # UV to clip space with v flipped: nvdiffrast's first output row is the top of the atlas
    #   - measured against Open3D's bake: unflipped off by 47.8 world units, flipped by 2e-5
    uv = tm.triangle.texture_uvs.numpy().reshape(-1, 2)
    zeros = np.zeros(len(uv), dtype=np.float32)
    clip = np.stack([uv[:, 0] * 2.0 - 1.0, (1.0 - uv[:, 1]) * 2.0 - 1.0, zeros, zeros + 1.0], axis=-1)

    # Rasterize once, interpolate both attributes off the same fragment buffer
    tri = torch.as_tensor(corner_tri, device="cuda")
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(
        ctx, torch.as_tensor(clip.astype(np.float32), device="cuda")[None], tri, resolution=[tex_size, tex_size]
    )
    positions, _ = dr.interpolate(torch.as_tensor(corner_pos, device="cuda")[None], rast, tri)
    normals, _ = dr.interpolate(torch.as_tensor(corner_nrm, device="cuda")[None], rast, tri)

    # Uncovered texels keep a zero normal, which is what the projection kernel rejects on
    covered = (rast[0, ..., 3] > 0).unsqueeze(-1)
    return (positions[0] * covered).cpu().numpy(), (normals[0] * covered).cpu().numpy()


######## Projection — NVIDIA Warp, one thread per texel


@wp.kernel
def _project_kernel(
    positions: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    image: wp.array2d(dtype=wp.vec3),
    w2c: wp.mat44,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam: wp.vec3,
    mesh_id: wp.uint64,
    eps: float,
    rgb_acc: wp.array2d(dtype=wp.vec3),
    w_acc: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    p = positions[i, j]
    n = normals[i, j]
    if wp.length(n) < 0.5:
        return
    n = wp.normalize(n)

    # Back-face and grazing rejection: weight is the cosine to the camera
    to_cam = cam - p
    dist = wp.length(to_cam)
    d = to_cam / dist
    cos = wp.dot(n, d)
    if cos <= 0.0:
        return

    # Project into the image; skip texels behind the camera or outside the frame
    pc = wp.transform_point(w2c, p)
    if pc[2] <= 0.0:
        return
    u = fx * pc[0] / pc[2] + cx
    v = fy * pc[1] / pc[2] + cy
    H = image.shape[0]
    W = image.shape[1]
    if u < 0.0 or v < 0.0 or u > float(W - 1) or v > float(H - 1):
        return

    # Occlusion: anything the ray from the camera hits before the texel hides it
    q = wp.mesh_query_ray(mesh_id, cam, -d, dist - eps)
    if q.result:
        return

    # Bilinear sample, cos-weighted accumulate
    x0 = int(wp.floor(u))
    y0 = int(wp.floor(v))
    x1 = wp.min(x0 + 1, W - 1)
    y1 = wp.min(y0 + 1, H - 1)
    ax = u - float(x0)
    ay = v - float(y0)
    c = (image[y0, x0] * (1.0 - ax) + image[y0, x1] * ax) * (1.0 - ay) + (
        image[y1, x0] * (1.0 - ax) + image[y1, x1] * ax
    ) * ay
    rgb_acc[i, j] = rgb_acc[i, j] + c * cos
    w_acc[i, j] = w_acc[i, j] + cos


def _dilate_texels(albedo, filled, gutter_px):
    """
    Grow filled texels into unfilled neighbors so bilinear sampling never reads black seams.

    Args:
        albedo: (S, S, 3) float atlas.
        filled: (S, S) bool, texels some view colored.
        gutter_px: int, dilation radius in texels.
    Returns:
        (S, S, 3) float32 atlas with the gutter filled by the mean of filled neighbors.
    """
    out = albedo.astype(np.float32)
    mask = filled.astype(np.float32)
    for _ in range(gutter_px):
        num = cv2.boxFilter(out * mask[..., None], -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        den = cv2.boxFilter(mask, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        grow = (den > 0) & (mask == 0)
        out[grow] = num[grow] / den[grow][:, None]
        mask[grow] = 1.0
    return out


def project_images_to_texture(tm, rgbs, c2w, K, tex_size, occlusion_eps, gutter_px=4):
    """
    Visibility-weighted projection of images into a mesh's UV atlas.

    Args:
        tm: o3d.t.geometry.TriangleMesh with triangle.texture_uvs (from unwrap_mesh_uvs).
        rgbs: (N, H, W, 3) uint8 images.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        tex_size: int, atlas edge in texels.
        occlusion_eps: float, ray-test tolerance in world units (≈ voxel size) so a surface never occludes itself.
        gutter_px: int, texels of color dilation around every chart.
    Returns:
        (tex_size, tex_size, 3) uint8 albedo; texels no view saw are 0.
    """
    wp.init()

    # Per-texel world position and normal from the atlas
    positions, normals = bake_atlas_attributes(tm, tex_size)
    posw = wp.array(positions, dtype=wp.vec3)
    nrmw = wp.array(normals, dtype=wp.vec3)

    # BVH over the mesh for the occlusion ray test
    verts = tm.vertex.positions.numpy().astype(np.float32)
    faces = tm.triangle.indices.numpy().astype(np.int32)
    wmesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3), indices=wp.array(faces.ravel(), dtype=wp.int32))

    # Accumulate cos-weighted color over every view
    rgb_acc = wp.zeros((tex_size, tex_size), dtype=wp.vec3)
    w_acc = wp.zeros((tex_size, tex_size), dtype=float)
    c2w = np.asarray(c2w, dtype=np.float64)
    w2c = invert_poses(c2w)
    for i in range(len(rgbs)):
        image = wp.array(np.ascontiguousarray(rgbs[i], dtype=np.float32) / 255.0, dtype=wp.vec3)
        fx, fy, cx, cy = extract_intrinsics(K[i])
        wp.launch(
            _project_kernel,
            dim=(tex_size, tex_size),
            inputs=[
                posw,
                nrmw,
                image,
                wp.mat44(w2c[i].astype(np.float32)),
                fx,
                fy,
                cx,
                cy,
                wp.vec3(c2w[i, :3, 3].astype(np.float32)),
                wmesh.id,
                float(occlusion_eps),
                rgb_acc,
                w_acc,
            ],
        )

    # Normalize, dilate the gutter, quantize
    rgb = rgb_acc.numpy()
    w = w_acc.numpy()
    albedo = np.where(w[..., None] > 0, rgb / np.maximum(w[..., None], 1e-12), 0.0)
    albedo = _dilate_texels(albedo, w > 0, gutter_px)
    return (np.clip(albedo, 0, 1) * 255).astype(np.uint8)


######## Entry point


def texture_mesh(mesh_path, out_dir, rgbs, c2w, K, *, voxel_size, decimate_max_error=0.25, tex_size=8192):
    """
    Decimate, repair, unwrap and texture a fused mesh; the input file is never modified.

    Args:
        mesh_path: Path or str to the fused mesh.ply (from fuse_tsdf + clean_repair_mesh).
        out_dir: Path or str, directory to create; receives mesh.ply (with UVs) and albedo.png.
        rgbs: (N, H, W, 3) uint8 views that were fused.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        voxel_size: float, TSDF voxel the mesh was fused at, world units; sets the decimation
            bound and the occlusion tolerance.
        decimate_max_error: float, decimation bound as a multiple of voxel_size.
        tex_size: int, atlas edge in texels.
    Returns:
        Path to out_dir/mesh.ply.
    """
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    decimated, err = decimate_mesh(mesh, decimate_max_error * voxel_size)
    manifold = _make_manifold(decimated)
    t1 = time.perf_counter()
    tm = unwrap_mesh_uvs(manifold, tex_size)
    t2 = time.perf_counter()
    albedo = project_images_to_texture(tm, rgbs, c2w, K, tex_size, occlusion_eps=voxel_size)
    t3 = time.perf_counter()
    out = write_textured_ply(tm.to_legacy(), tm.triangle.texture_uvs.numpy(), albedo, out_dir)
    logger.info(
        "texture_mesh: %d -> %d faces (error %.4f) in %.1fs, uvatlas %.1fs, projection %.1fs over %d views -> %s",
        len(mesh.triangles),
        len(manifold.triangles),
        err,
        t1 - t0,
        t2 - t1,
        t3 - t2,
        len(rgbs),
        out,
    )
    return out
