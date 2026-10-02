"""
Texture a fused mesh: one albedo atlas baked from the source views.

- input: prepare_mesh output (clean.py)
- unwrap_mesh_uvs: UV atlas (Open3D UVAtlas)
- _rasterize_atlas: per-texel world position + normal (nvdiffrast)
- project_images_to_texture: per-texel color, two Warp kernel passes then fill_missing_pixels
- write_textured_obj (utils/io.py): mesh.obj + mesh.mtl + albedo.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

import nvdiffrast.torch as dr
import warp as wp

from collab_splats.geometry.transforms import extract_intrinsics, invert_poses
from collab_splats.mesh.tsdf import _validate_views
from collab_splats.utils.image import fill_missing_pixels
from collab_splats.utils.io import to_uint8_hwc, write_textured_obj

logger = logging.getLogger(__name__)


########################
# Entry point
########################


def create_texture_mesh(
    mesh: o3d.geometry.TriangleMesh,
    occluder: o3d.geometry.TriangleMesh,
    out_dir: Path | str,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    *,
    voxel_size: float,
    tex_size: int = 8192,
) -> Path:
    """
    Unwrap and texture a prepared mesh.

    - the unfilled occluder hides surfaces, so invented patches never hide a real surface
    - voxel_size is the occlusion tolerance
    - writes out_dir/mesh.obj + mesh.mtl + albedo.png via write_textured_obj

    Args:
        mesh: prepare_mesh output.
        occluder: the cleaned mesh before prepare_mesh.
        out_dir: directory to create.
        rgbs: (N, H, W, 3) uint8 views that were fused.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        voxel_size: TSDF voxel the mesh was fused at, world units.
        tex_size: atlas edge in texels.

    Returns:
        Path to out_dir/mesh.obj.
    """
    rgbs, c2w, K, _ = _validate_views(rgbs, c2w, K)
    tm = unwrap_mesh_uvs(mesh, tex_size)
    albedo = project_images_to_texture(tm, rgbs, c2w, K, tex_size, occlusion_eps=voxel_size, occluder=occluder)

    # Smooth per-vertex normals for the OBJ; without them viewers shade split corners flat
    tm.compute_vertex_normals()
    out = write_textured_obj(
        out_dir,
        tm.vertex.positions.numpy(),
        tm.triangle.indices.numpy(),
        tm.vertex.normals.numpy(),
        tm.triangle.texture_uvs.numpy(),
        albedo,
    )
    logger.info("create_texture_mesh: %d faces -> %s", len(tm.triangle.indices), out)
    return out


########################
# UV unwrap
########################


def unwrap_mesh_uvs(
    mesh: o3d.geometry.TriangleMesh,
    tex_size: int,
    parallel_partitions: int = 16,
    min_faces_per_partition: int = 1000,
    max_stretch: float = 0.1667,
) -> o3d.t.geometry.TriangleMesh:
    """
    UV atlas from Open3D's UVAtlas.

    - partition count clamped so none is empty; Open3D's PCA partition raises on an empty one
    - low max_stretch shatters a curved mesh into many small charts, each one a seam

    Args:
        mesh: manifold mesh (see make_manifold).
        tex_size: atlas edge in texels.
        parallel_partitions: UVAtlas partitions run in parallel (1 = single-threaded).
        min_faces_per_partition: fewest faces per partition.
        max_stretch: distortion in [0, 1] UVAtlas tolerates before cutting a new chart.

    Returns:
        Tensor mesh with triangle.texture_uvs (F, 3, 2).
    """
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    partitions = max(1, min(int(parallel_partitions), len(mesh.triangles) // min_faces_per_partition))
    tm.compute_uvatlas(size=tex_size, max_stretch=float(max_stretch), parallel_partitions=partitions)
    return tm


########################
# Atlas rasterization
########################


def _rasterize_atlas(tm: o3d.t.geometry.TriangleMesh, tex_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize a UV atlas into per-texel world position and normal.

    - returns two (tex_size, tex_size, 3) float32 arrays, zero where no triangle covers
    """
    tm.compute_vertex_normals()
    verts = tm.vertex.positions.numpy()
    faces = tm.triangle.indices.numpy()
    vert_normals = tm.vertex.normals.numpy()

    # One vertex per triangle corner, since a seam vertex has a different UV in each triangle
    corner_pos = np.ascontiguousarray(verts[faces].reshape(-1, 3), dtype=np.float32)
    corner_nrm = np.ascontiguousarray(vert_normals[faces].reshape(-1, 3), dtype=np.float32)
    corner_tri = np.arange(len(corner_pos), dtype=np.int32).reshape(-1, 3)

    # UV to clip space with v flipped, since nvdiffrast's first output row is the top of the atlas
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


########################
# Projection
########################


def project_images_to_texture(
    tm: o3d.t.geometry.TriangleMesh,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    tex_size: int,
    occlusion_eps: float,
    occluder: o3d.geometry.TriangleMesh | None = None,
    view_ratio: float = 1.5,
) -> np.ndarray:
    """
    Visibility-weighted projection of images into a mesh's UV atlas.

    - texels no view reaches take a push-pull fill from their seen surroundings
    - a hole-filled mesh hides its own observed surface, so pass the pre-fill mesh as occluder
    - view_ratio 1.0 is single-best-view (sharpest, seam-prone); large values average everything

    Args:
        tm: mesh with triangle.texture_uvs (from unwrap_mesh_uvs).
        rgbs: (N, H, W, 3) uint8 images.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        tex_size: atlas edge in texels.
        occlusion_eps: ray-test tolerance in world units (≈ voxel size); stops self-occlusion.
        occluder: mesh to ray-test against instead of tm.
        view_ratio: keep views whose pixel size at the texel is within this factor of the texel's best.

    Returns:
        (tex_size, tex_size, 3) uint8 albedo.
    """
    wp.init()

    # Per-texel world position and normal from the atlas
    positions, normals = _rasterize_atlas(tm, tex_size)
    posw = wp.array(positions, dtype=wp.vec3)
    nrmw = wp.array(normals, dtype=wp.vec3)

    # BVH over the mesh for the occlusion ray test; a separate occluder wins when given
    if occluder is None:
        verts = tm.vertex.positions.numpy()
        faces = tm.triangle.indices.numpy()
    else:
        verts = np.asarray(occluder.vertices)
        faces = np.asarray(occluder.triangles)
        logger.debug("occlusion tested against a separate %d-triangle mesh", len(faces))

    wmesh = wp.Mesh(
        points=wp.array(verts.astype(np.float32), dtype=wp.vec3),
        indices=wp.array(faces.astype(np.int32).ravel(), dtype=wp.int32),
    )

    # One Camera struct per view, shared by both passes
    c2w = np.asarray(c2w, dtype=np.float64)
    w2c = invert_poses(c2w)
    cameras = []

    for i in range(len(rgbs)):
        cam = _Camera()
        cam.w2c = wp.mat44(w2c[i].astype(np.float32))
        cam.center = wp.vec3(c2w[i, :3, 3].astype(np.float32))
        cam.fx, cam.fy, cam.cx, cam.cy = extract_intrinsics(K[i])
        cam.height, cam.width = int(rgbs[i].shape[0]), int(rgbs[i].shape[1])
        cameras.append(cam)

    # Pass one: finest pixel size any view achieves per texel, used to gate voters in pass two
    best_px_size = wp.full((tex_size, tex_size), value=1.0e30, dtype=float)

    for cam in cameras:
        wp.launch(
            _finest_pixel_size_kernel,
            dim=(tex_size, tex_size),
            inputs=[posw, nrmw, cam, wmesh.id, float(occlusion_eps), best_px_size],
        )

    # Pass two: sample color from the views that resolve each texel within view_ratio of its best
    rgb_acc = wp.zeros((tex_size, tex_size), dtype=wp.vec3)
    w_acc = wp.zeros((tex_size, tex_size), dtype=float)

    for rgb, cam in zip(rgbs, cameras):
        image = wp.array(np.ascontiguousarray(rgb, dtype=np.float32) / 255.0, dtype=wp.vec3)
        wp.launch(
            _project_kernel,
            dim=(tex_size, tex_size),
            inputs=[
                posw,
                nrmw,
                image,
                cam,
                wmesh.id,
                float(occlusion_eps),
                best_px_size,
                float(view_ratio),
                rgb_acc,
                w_acc,
            ],
        )

    # Normalize the weighted sum
    rgb = rgb_acc.numpy()
    w = w_acc.numpy()
    albedo = np.where(w[..., None] > 0, rgb / np.maximum(w[..., None], 1e-12), 0.0)

    # Two unrelated blacks: atlas space no chart claimed, and surface no camera ever reached
    seen = w > 0
    covered = np.linalg.norm(normals, axis=-1) > 0.5
    logger.info(
        "atlas %.1f%% surface texels | %.1f%% of surface unseen | %.1f%% of atlas blank",
        100 * covered.mean(),
        100 * float((covered & ~seen).sum()) / max(int(covered.sum()), 1),
        100 * (~seen).mean(),
    )

    # Fill everything no view reached, then quantize
    albedo = fill_missing_pixels(albedo, seen)
    return to_uint8_hwc(albedo, channels_first=False)


########################
# Projection kernels (Warp)
########################


@wp.struct
class _Camera:
    """
    One source view: world-to-camera pose, center, pinhole intrinsics and image size.
    """

    w2c: wp.mat44
    center: wp.vec3
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@wp.func
def _pixel_size_at_texel(p: wp.vec3, n: wp.vec3, camera: _Camera, mesh_id: wp.uint64, eps: float):
    """
    World size one source pixel covers at this texel, and the pixel (u, v) it projects to.

    - pixel size = dist / (fx * cos): grazing and distant views resolve the surface coarsely
    - pixel size -1 for back-faces, out-of-frustum texels and anything the mesh occludes
    """
    to_cam = camera.center - p
    dist = wp.length(to_cam)
    d = to_cam / dist
    cos = wp.dot(n, d)

    if cos <= 0.0:
        return wp.vec3(-1.0, 0.0, 0.0)

    # Project into the image; skip texels behind the camera or outside the frame
    pc = wp.transform_point(camera.w2c, p)

    if pc[2] <= 0.0:
        return wp.vec3(-1.0, 0.0, 0.0)

    u = camera.fx * pc[0] / pc[2] + camera.cx
    v = camera.fy * pc[1] / pc[2] + camera.cy

    if u < 0.0 or v < 0.0 or u > float(camera.width - 1) or v > float(camera.height - 1):
        return wp.vec3(-1.0, 0.0, 0.0)

    # Occlusion: anything the ray from the camera hits before the texel hides it
    q = wp.mesh_query_ray(mesh_id, camera.center, -d, dist - eps)

    if q.result:
        return wp.vec3(-1.0, 0.0, 0.0)

    return wp.vec3(dist / (camera.fx * cos), u, v)


@wp.func
def _bilinear(image: wp.array2d(dtype=wp.vec3), u: float, v: float):
    """
    Bilinear sample of an image at a pixel position inside the frame.
    """
    x0 = int(wp.floor(u))
    y0 = int(wp.floor(v))
    x1 = wp.min(x0 + 1, image.shape[1] - 1)
    y1 = wp.min(y0 + 1, image.shape[0] - 1)
    ax = u - float(x0)
    ay = v - float(y0)
    top = image[y0, x0] * (1.0 - ax) + image[y0, x1] * ax
    bottom = image[y1, x0] * (1.0 - ax) + image[y1, x1] * ax
    return top * (1.0 - ay) + bottom * ay


@wp.kernel
def _finest_pixel_size_kernel(
    positions: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    camera: _Camera,
    mesh_id: wp.uint64,
    eps: float,
    best_px_size: wp.array2d(dtype=float),
):
    """
    Pass one: record the finest pixel size any view achieves on each texel.
    """
    i, j = wp.tid()
    n = normals[i, j]

    if wp.length(n) < 0.5:
        return

    px_size = _pixel_size_at_texel(positions[i, j], wp.normalize(n), camera, mesh_id, eps)[0]

    if px_size > 0.0:
        wp.atomic_min(best_px_size, i, j, px_size)


@wp.kernel
def _project_kernel(
    positions: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    image: wp.array2d(dtype=wp.vec3),
    camera: _Camera,
    mesh_id: wp.uint64,
    eps: float,
    best_px_size: wp.array2d(dtype=float),
    view_ratio: float,
    rgb_acc: wp.array2d(dtype=wp.vec3),
    w_acc: wp.array2d(dtype=float),
):
    """
    Pass two: accumulate color from every view within view_ratio of the texel's best pixel size.
    """
    i, j = wp.tid()
    n = normals[i, j]

    if wp.length(n) < 0.5:
        return

    hit = _pixel_size_at_texel(positions[i, j], wp.normalize(n), camera, mesh_id, eps)
    px_size = hit[0]

    if px_size <= 0.0:
        return

    # Keep only views resolving this texel nearly as finely as the best one
    if px_size > best_px_size[i, j] * view_ratio:
        return

    # Bilinear sample, weighted by how finely this view resolves the texel
    weight = best_px_size[i, j] / px_size
    rgb_acc[i, j] = rgb_acc[i, j] + _bilinear(image, hit[1], hit[2]) * weight
    w_acc[i, j] = w_acc[i, j] + weight
