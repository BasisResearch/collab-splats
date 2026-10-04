"""
Texture a fused mesh: one albedo atlas baked from the source views.

- input: prepare_mesh output (clean.py)
- _solve_view_gains / _apply_view_gains: per-view color gains divided out before anything samples the images
- unwrap_view_charts: UV atlas from camera-projection charts (nvdiffrast face ids)
- project_images_to_texture: atlas texels, per-view depth test, two torch passes, fill_missing_pixels
- _dilate_chart_gutters: chart edges copied into their own padding
- write_textured_obj (utils/io.py): mesh.obj + mesh.mtl + albedo.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import nvdiffrast.torch as dr

from collab_splats.geometry.projection import project
from collab_splats.geometry.transforms import (
    extract_intrinsics,
    invert_poses,
    rescale_intrinsics,
)
from collab_splats.mesh.utils import adjacent_face_pairs, face_areas, validate_views
from collab_splats.utils.image import fill_missing_pixels
from collab_splats.utils.io import to_uint8_hwc, write_textured_obj

logger = logging.getLogger(__name__)


########################################################################
# Entry point
########################################################################


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
    color_correct: bool = True,
) -> Path:
    """
    Unwrap and texture a prepared mesh.

    - the unfilled occluder hides surfaces, so invented patches never hide a real surface
    - the occluder also supplies the color-gain samples: only real surfaces vote
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
        color_correct: divide a per-view, per-channel gain out of each image first.

    Returns:
        Path to out_dir/mesh.obj.
    """
    rgbs, c2w, K, _ = validate_views(rgbs, c2w, K)

    # Exposure gains solved on the occluder, divided out of a copy of the views
    if color_correct:
        gains = _solve_view_gains(np.asarray(occluder.vertices), np.asarray(occluder.triangles), rgbs, c2w, K)
        rgbs = _apply_view_gains(rgbs, gains)

    # UVs, then smooth vertex normals on a copy; without normals viewers shade split corners flat
    uvs, boxes = unwrap_view_charts(mesh, c2w, K, rgbs.shape[1:3], tex_size)
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()
    verts, faces, normals = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_normals)
    occluder_vf = (np.asarray(occluder.vertices), np.asarray(occluder.triangles))

    # Project, then grow each chart's edge into its own padding so bilinear lookups never mix charts
    albedo = project_images_to_texture(
        verts, faces, normals, uvs, rgbs, c2w, K, tex_size, occlusion_eps=voxel_size, occluder=occluder_vf
    )
    albedo = _dilate_chart_gutters(albedo, uvs, boxes)

    out = write_textured_obj(out_dir, verts, faces, normals, uvs, albedo)
    logger.info("create_texture_mesh: %d faces -> %s", len(uvs), out)
    return out


########################################################################
# Color gains
########################################################################


def _solve_view_gains(
    verts: np.ndarray,
    faces: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    *,
    n_points: int = 300_000,
    scale: float = 0.5,
    blur: int = 5,
    rel_tol: float = 0.01,
    prior: float = 1e-2,
    trim: float = 0.25,
    smooth: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Per-view, per-channel gains so gain * albedo explains every observation, (V, 3).

    - model: log observed = log gain[view] + log albedo[point], solved in log space
    - smooth: weight on the squared log-gain change between consecutive frames
    - prior: tiny pseudo-count anchoring each log gain at 0
    - second solve drops observations whose absolute log residual exceeds trim
    """
    w2c = invert_poses(c2w)
    point, view, color = _gain_samples(verts, faces, rgbs, w2c, K, n_points, scale, blur, rel_tol, seed)

    # Keep points seen in at least two views; reindex
    keep = np.bincount(point)[point] >= 2

    if not keep.any():
        logger.warning("view gains: no surface point seen in two views, gains left at 1")
        return np.ones((len(rgbs), 3))

    point, view, color = point[keep], view[keep], color[keep]
    _, point = np.unique(point, return_inverse=True)
    point = point.ravel()
    n_obs, n_views, n_points_kept = len(point), len(rgbs), int(point.max()) + 1
    log_color = np.log(color.astype(np.float64))

    # Incidence matrices and the fixed part of the normal equations
    rows = np.arange(n_obs)
    A = coo_matrix((np.ones(n_obs), (rows, view)), shape=(n_obs, n_views)).tocsr()
    B = coo_matrix((np.ones(n_obs), (rows, point)), shape=(n_obs, n_points_kept)).tocsr()
    D = np.diff(np.eye(n_views), axis=0)
    base = smooth * n_obs * (D.T @ D) / n_views + prior * np.eye(n_views) + np.ones((n_views, n_views)) / n_views

    # Solve, trim outliers, solve again
    weight = np.ones(n_obs)
    log_gain, log_albedo = _solve_log_gains(A, B, view, log_color, weight, base)
    residual = np.abs(log_color - log_gain[view] - log_albedo[point]).max(1)
    weight = (residual < trim).astype(np.float64)
    log_gain, _ = _solve_log_gains(A, B, view, log_color, weight, base)

    gains = np.exp(log_gain)
    logger.info(
        "view gains: %d obs, %.0f%% trimmed, range [%.2f, %.2f]",
        n_obs,
        100 * (1 - weight.mean()),
        gains.min(),
        gains.max(),
    )
    return gains


def _apply_view_gains(rgbs: np.ndarray, gains: np.ndarray, *, knee: float = 200.0) -> np.ndarray:
    """
    Each view divided by its gain, as a new uint8 array.

    - values above knee roll off exponentially toward 255 instead of hard-clipping
    """
    out = np.empty_like(rgbs)

    for k in range(len(rgbs)):
        img = torch.from_numpy(rgbs[k]).cuda().float() / torch.from_numpy(gains[k].astype(np.float32)).cuda()

        # Soft highlight rolloff, so brightened views do not blow out to flat white
        over = (img - knee).clamp(min=0)
        rolled = knee + (255 - knee) * (1 - torch.exp(-over / (255 - knee)))
        img = torch.where(img > knee, rolled, img)
        out[k] = img.round().clamp(0, 255).byte().cpu().numpy()

    return out


def _gain_samples(
    verts: np.ndarray,
    faces: np.ndarray,
    rgbs: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    n_points: int,
    scale: float,
    blur: int,
    rel_tol: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Depth-tested color samples of random surface points in every view.

    - points: area-weighted face centroids
    - colors read from a box-blurred image at `scale`, so residual misregistration averages out
    - clipped (> 0.97) and near-black (< 0.02) samples dropped: the gain model does not hold there
    - returns point index, view index and RGB in [0, 1] per observation
    """
    # Area-weighted face centroids
    rng = np.random.default_rng(seed)
    fv = verts[faces]
    area = face_areas(verts, faces)
    pick = rng.choice(len(faces), n_points, p=area / area.sum())
    points = torch.from_numpy(fv[pick].mean(1).astype(np.float32)).cuda()

    # Working grid and its intrinsics
    ctx = dr.RasterizeCudaContext()
    verts_h, faces_t = _to_cuda_mesh(verts, faces)
    full_hw = rgbs.shape[1:3]
    h, w = int(full_hw[0] * scale), int(full_hw[1] * scale)
    K_small = rescale_intrinsics(K, full_hw, (h, w))
    w2c_t = torch.from_numpy(w2c.astype(np.float32)).cuda()
    K_t = torch.from_numpy(K_small.astype(np.float32)).cuda()
    out_point, out_view, out_color = [], [], []

    for k in range(len(rgbs)):
        _, depth = _rasterize_view(ctx, verts_h, faces_t, w2c_t[k], K_t[k], h, w)

        # Blurred image at working scale
        img = torch.from_numpy(rgbs[k]).cuda().permute(2, 0, 1)[None].float() / 255.0
        img = F.interpolate(img, size=(h, w), mode="area")
        img = F.avg_pool2d(img, blur, stride=1, padding=blur // 2, count_include_pad=False)[0]

        # Project samples; keep in-frame ones at the rendered depth
        pixels, cam = project(points, w2c_t[k], K_t[k])
        u = pixels[:, 0].round().long()
        r = pixels[:, 1].round().long()
        framed = (cam[:, 2] > 1e-3) & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        idx = torch.nonzero(framed)[:, 0]
        d = _nearest_depth(depth, pixels[idx])
        idx = idx[(d > 0) & ((cam[idx, 2] - d).abs() < rel_tol * d)]
        color = img[:, r[idx], u[idx]].T

        # Drop clipped and near-black samples
        valid = (color.max(1).values < 0.97) & (color.min(1).values > 0.02)
        idx = idx[valid]
        out_point.append(idx.cpu().numpy())
        out_view.append(np.full(len(idx), k, dtype=np.int32))
        out_color.append(color[valid].cpu().numpy())

    return np.concatenate(out_point), np.concatenate(out_view), np.concatenate(out_color)


def _solve_log_gains(
    A: coo_matrix, B: coo_matrix, view: np.ndarray, log_color: np.ndarray, weight: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted least squares log gains with the per-point log albedo eliminated exactly.

    - A: (N, V) observation -> view, B: (N, P) observation -> point
    - base: (V, V) prior + smoothness + sum-to-zero pin
    - returns (V, 3) log gains, median-centered, and (P, 3) log albedo
    """
    Aw = A.multiply(weight[:, None]).tocsr()
    Bw = B.multiply(weight[:, None]).tocsr()
    inv_count = 1.0 / np.maximum(np.asarray(Bw.sum(0)).ravel(), 1e-9)

    # Reduced normal equations: points eliminated by Schur complement
    BtA = (B.T @ Aw).toarray()
    lhs = (A.T @ Aw).toarray() - BtA.T @ (BtA * inv_count[:, None]) + base
    rhs = A.T @ (weight[:, None] * log_color) - BtA.T @ ((Bw.T @ log_color) * inv_count[:, None])
    log_gain = np.linalg.solve(lhs, rhs)

    # Typical view is the reference, so the texture keeps a typical exposure
    log_gain -= np.median(log_gain, 0)
    log_albedo = (Bw.T @ (log_color - log_gain[view])) * inv_count[:, None]
    return log_gain, log_albedo


########################################################################
# UV unwrap: view charts
########################################################################


def unwrap_view_charts(
    mesh: o3d.geometry.TriangleMesh,
    c2w: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
    tex_size: int,
    *,
    frac: float = 0.75,
    small_px: float = 4.0,
    rel_tol: float = 0.01,
    alpha: float = 0.25,
    rounds: int = 40,
    tile_m: float = 0.5,
    pad: int = 1,
    min_owned: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    UV atlas whose charts are the source images themselves.

    - a face may take a view only if it owns its pixels in that view's face-id buffer
    - the z-buffer gives each pixel to one face, so faces in one camera chart cannot overlap
    - faces no view fully sees get one flat patch per connected group, along its mean normal
    - one atlas check: flipped or overwritten faces become single flat charts, then one repack

    Args:
        mesh: prepare_mesh output.
        c2w: (V, 4, 4) camera-to-world poses.
        K: (V, 3, 3) intrinsics at image resolution.
        image_hw: (height, width) of the views.
        tex_size: atlas edge in texels.
        frac: share of its projected area a face must own for the view to count.
        small_px: faces under this projected area use a centroid depth test instead.
        rel_tol: relative depth tolerance of that centroid test.
        alpha: smoothing lets a face switch to a neighbor's view scoring >= alpha of its best.
        rounds: smoothing rounds.
        tile_m: charts are cut into tiles of this world size, so no box spans the atlas.
        pad: gutter texels around each chart box.
        min_owned: a face owning under this share of its expected texels is overwritten.

    Returns:
        (F, 3, 2) float32 UVs in [0, 1] with v up, and (C, 4) chart boxes x, y, w, h in texels.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles)
    w2c = invert_poses(c2w)
    n_faces, n_views = len(faces), len(w2c)

    # Label: best view the face owns its pixels in, smoothed among owned views only
    scores = _face_view_scores(verts, faces, w2c, K, image_hw, frac=frac, small_px=small_px, rel_tol=rel_tol)
    unseen = (scores.max(1).values == 0).cpu().numpy()
    labels = scores.float().argmax(1)
    pairs = adjacent_face_pairs(faces, len(verts))
    labels = _smooth_labels(labels, scores, pairs, alpha=alpha, rounds=rounds).cpu().numpy()
    del scores
    torch.cuda.empty_cache()

    # Camera UVs: corner pixels in the label view
    fv = verts[faces]
    fn = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    pixels, _ = project(torch.from_numpy(fv), torch.from_numpy(w2c[labels]), torch.from_numpy(K[labels]))
    uv = pixels.numpy()

    # Unseen faces: connected unseen groups (every seen face keyed alone) become plane patches
    patch = _same_key_components(np.where(unseen, 0, 1 + np.arange(n_faces)), pairs, n_faces)
    patch_normal = np.zeros((int(patch.max()) + 1, 3))
    np.add.at(patch_normal, patch[unseen], fn[unseen])
    uv[unseen] = _plane_uv(fv[unseen], patch_normal[patch[unseen]])
    logger.info("view charts: %d of %d faces unseen", int(unseen.sum()), n_faces)

    # Pack: chart key is the label view, or the plane patch for unseen faces
    key = np.where(unseen, n_views + patch, labels)
    uvs, chart, boxes = _pack_charts(uv, key, area, pairs, tile_m, pad, tex_size)
    bad = _find_broken_faces(uvs, chart, unseen, tex_size, min_owned)

    # Broken faces become single flat charts; repack once
    if bad.any():
        uv[bad] = _plane_uv(fv[bad] - fv[bad][:, :1], fn[bad])
        key = np.where(bad, n_views + n_faces + np.arange(n_faces), key)
        uvs, _, boxes = _pack_charts(uv, key, area, pairs, tile_m, pad, tex_size)

    return uvs, boxes


def _face_view_scores(
    verts: np.ndarray,
    faces: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
    *,
    frac: float,
    small_px: float,
    rel_tol: float,
) -> torch.Tensor:
    """
    Pixels each face owns in each view's face-id buffer, (F, V) float16 cuda; 0 where it does not count.

    - counts only if in front, fully framed, and owning >= frac of its projected area (one pixel of slack)
    - sub-pixel faces own nothing: they count if the z-buffer depth at their centroid matches instead
    """
    height, width = image_hw
    ctx = dr.RasterizeCudaContext()
    verts_h, faces_t = _to_cuda_mesh(verts, faces)
    corners = faces_t.long()
    centroids = verts_h[:, :3][corners].mean(1)
    n_faces = len(faces)
    w2c_t = torch.from_numpy(w2c.astype(np.float32)).cuda()
    K_t = torch.from_numpy(K.astype(np.float32)).cuda()
    scores = torch.zeros(n_faces, len(w2c), dtype=torch.float16, device="cuda")

    for k in range(len(w2c)):
        rast, depth = _rasterize_view(ctx, verts_h, faces_t, w2c_t[k], K_t[k], height, width)
        face_id = rast[..., 3].long().flatten()
        owned = torch.bincount(face_id[face_id > 0] - 1, minlength=n_faces).float()

        # Projected area of each face; in front and fully framed only
        pixels, cam = project(verts_h[:, :3], w2c_t[k], K_t[k])
        p = pixels[corners]
        area = _signed_area(p).abs()
        in_front = (cam[corners][..., 2] > 1e-3).all(1)
        framed = ((p[..., 0] >= 0) & (p[..., 0] < width) & (p[..., 1] >= 0) & (p[..., 1] < height)).all(1)

        # Centroid depth test for faces too small to own pixels
        centroid_px, centroid_cam = project(centroids, w2c_t[k], K_t[k])
        d = _nearest_depth(depth, centroid_px)
        near = (d > 0) & ((centroid_cam[:, 2] - d).abs() < rel_tol * d)

        # Big faces must own their area; small ones must sit at the z-buffer depth
        owns = owned >= frac * area - 1.0
        valid = in_front & framed & torch.where(area >= small_px, owns, near)
        scores[:, k] = torch.where(valid, owned, 0).clamp(max=6e4).half()

    return scores


def _smooth_labels(
    labels: torch.Tensor, scores: torch.Tensor, pairs: np.ndarray, *, alpha: float, rounds: int
) -> torch.Tensor:
    """
    Majority vote: a face takes the label two of its neighbors share if it scores >= alpha of its best there.

    - only switches to a view with a nonzero score, so a face never leaves the views it owns
    - up to 3 neighbor slots per face; extra neighbors on non-manifold edges are dropped
    """
    n = len(labels)
    a = torch.from_numpy(np.concatenate([pairs[:, 0], pairs[:, 1]])).cuda()
    b = torch.from_numpy(np.concatenate([pairs[:, 1], pairs[:, 0]])).cuda()
    best = scores.max(1).values.float()

    # Neighbor table (F, 3), -1 where empty
    order = torch.argsort(a, stable=True)
    a, b = a[order], b[order]
    start = torch.searchsorted(a, torch.arange(n, device="cuda"))
    slot = torch.arange(len(a), device="cuda") - start[a]
    keep = slot < 3
    nbr = torch.full((n, 3), -1, dtype=torch.long, device="cuda")
    nbr[a[keep], slot[keep]] = b[keep]

    for _ in range(rounds):
        # Candidate: a label at least two neighbors share
        nl = torch.where(nbr >= 0, labels[nbr.clamp(min=0)], -1)
        first = (nl[:, 0] >= 0) & ((nl[:, 0] == nl[:, 1]) | (nl[:, 0] == nl[:, 2]))
        second = (nl[:, 1] >= 0) & (nl[:, 1] == nl[:, 2])
        cand = torch.where(first, nl[:, 0], torch.where(second, nl[:, 1], -1))

        # Switch where the face scores well enough in the candidate view
        ok = cand >= 0
        cand_score = torch.zeros(n, device="cuda")
        cand_score[ok] = scores[ok.nonzero()[:, 0], cand[ok]].float()
        switch = ok & (cand != labels) & (cand_score >= alpha * best) & (cand_score > 0)
        labels = torch.where(switch, cand, labels)

    return labels


def _same_key_components(key: np.ndarray, pairs: np.ndarray, n: int) -> np.ndarray:
    """
    Connected-component id per face, over edge-adjacent faces that share the same key.
    """
    p = pairs[key[pairs[:, 0]] == key[pairs[:, 1]]]
    graph = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _plane_uv(fv: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Corner coordinates in the plane perpendicular to each normal, world units, (F, 3, 2).
    """
    n = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)

    # In-plane basis from any helper axis not parallel to the normal
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(n, helper)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(n, e1)
    return np.stack([np.einsum("fcj,fj->fc", fv, e1), np.einsum("fcj,fj->fc", fv, e2)], -1)


########################################################################
# UV unwrap: packing
########################################################################


def _pack_charts(
    uv: np.ndarray, key: np.ndarray, area: np.ndarray, pairs: np.ndarray, tile_m: float, pad: int, tex_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charts from (key, adjacency), tiled, scaled to one texel density, rotated, shelf-packed.

    - returns (F, 3, 2) float32 UVs with v up, chart id per face, and (C, 4) boxes x, y, w, h
    """
    n_faces = len(key)
    uv_area = np.abs(_signed_area(uv))
    chart = _same_key_components(key, pairs, n_faces)

    # Cut charts into world-sized tiles so no box spans the atlas
    scale = _chart_world_scale(chart, uv_area, area)
    cell = np.floor(uv.mean(1) * scale[chart][:, None] / tile_m).astype(np.int64)
    _, tile = np.unique(np.stack([chart, cell[:, 0], cell[:, 1]], 1), axis=0, return_inverse=True)
    chart = _same_key_components(tile.ravel(), pairs, n_faces)
    n_charts = int(chart.max()) + 1

    # World-unit corners, centered per chart
    scale = _chart_world_scale(chart, uv_area, area)
    flat = (uv * scale[chart][:, None, None]).reshape(-1, 2)
    cc = np.repeat(chart, 3)
    mean = np.stack([np.bincount(cc, flat[:, i], n_charts) for i in range(2)], 1) / np.bincount(cc)[:, None]
    d = flat - mean[cc]

    # Rotate each chart onto its principal axes
    sxx = np.bincount(cc, d[:, 0] ** 2, n_charts)
    syy = np.bincount(cc, d[:, 1] ** 2, n_charts)
    sxy = np.bincount(cc, d[:, 0] * d[:, 1], n_charts)
    angle = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    ca, sa = np.cos(angle)[cc], np.sin(angle)[cc]
    rot = np.stack([ca * d[:, 0] + sa * d[:, 1], -sa * d[:, 0] + ca * d[:, 1]], 1).reshape(-1, 3, 2)
    cmin = np.full((n_charts, 2), np.inf)
    cmax = np.full((n_charts, 2), -np.inf)
    np.minimum.at(cmin, chart, rot.min(1))
    np.maximum.at(cmax, chart, rot.max(1))
    local = rot - cmin[chart][:, None]

    # Largest global scale whose shelf pack fits the atlas
    lo, hi = 1e-2, 1e6

    for _ in range(25):
        s = (lo * hi) ** 0.5
        size = np.ceil((cmax - cmin) * s).astype(np.int64) + 2 * pad + 1
        _, _, total = _shelf_pack(size[:, 0], size[:, 1], tex_size)
        lo, hi = (s, hi) if total <= tex_size else (lo, s)

    size = np.ceil((cmax - cmin) * lo).astype(np.int64) + 2 * pad + 1
    x, y, _ = _shelf_pack(size[:, 0], size[:, 1], tex_size)
    logger.info("view charts: %d charts at %.0f texels per world unit", n_charts, lo)

    # Atlas texel coordinates -> UV with v up (row 0 is v = 1)
    tx = local[..., 0] * lo + x[chart][:, None] + pad + 0.5
    ty = local[..., 1] * lo + y[chart][:, None] + pad + 0.5
    uvs = np.stack([tx / tex_size, 1.0 - ty / tex_size], -1).astype(np.float32)
    return uvs, chart, np.concatenate([x[:, None], y[:, None], size], 1)


def _chart_world_scale(chart: np.ndarray, uv_area: np.ndarray, world_area: np.ndarray) -> np.ndarray:
    """
    Per-chart sqrt(world area / uv area): chart coordinates times this are world units.
    """
    n = int(chart.max()) + 1
    return np.sqrt(np.bincount(chart, world_area, n) / np.maximum(np.bincount(chart, uv_area, n), 1e-12))


def _shelf_pack(w: np.ndarray, h: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Shelf packing of integer boxes, tallest first; returns x, y offsets and total height.
    """
    order = np.argsort(-h, kind="stable")
    x = np.zeros(len(w), dtype=np.int64)
    y = np.zeros(len(w), dtype=np.int64)
    cx, cy, shelf = 0, 0, 0

    for i in order:
        # Start a new shelf when the row is full
        if cx + w[i] > width:
            cx, cy, shelf = 0, cy + shelf, 0

        x[i], y[i] = cx, cy
        cx += w[i]
        shelf = max(shelf, h[i])

    return x, y, cy + shelf


def _find_broken_faces(
    uvs: np.ndarray, chart: np.ndarray, unseen: np.ndarray, tex_size: int, min_owned: float
) -> np.ndarray:
    """
    Faces one atlas raster shows flipped or overwritten, (F,) bool.

    - flipped: plane-patch faces wound against their chart's area-weighted majority
    - lost: faces owning under min_owned of their expected texels
    """
    signed = _signed_area(uvs.astype(np.float64))
    majority = np.sign(np.bincount(chart, signed))
    flipped = unseen & (np.sign(signed) != majority[chart])

    # Lost: faces another face overwrote in the atlas raster
    rast = _rasterize_uvs(uvs, tex_size)
    face_id = rast[0, ..., 3].long() - 1
    owned = torch.bincount(face_id[face_id >= 0], minlength=len(uvs)).cpu().numpy()
    del rast, face_id
    expect = np.abs(signed) * tex_size**2
    lost = (expect >= 1.0) & (owned < min_owned * expect)
    logger.info("view charts: %d flipped, %d lost faces become single charts", int(flipped.sum()), int(lost.sum()))
    return flipped | lost


########################################################################
# Projection
########################################################################


def project_images_to_texture(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    rgbs: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    tex_size: int,
    occlusion_eps: float,
    occluder: tuple[np.ndarray, np.ndarray] | None = None,
    view_ratio: float = 1.5,
) -> np.ndarray:
    """
    Visibility-weighted projection of images into a mesh's UV atlas.

    - texels no view reaches take a push-pull fill from their seen surroundings
    - a hole-filled mesh hides its own observed surface, so pass the pre-fill mesh as occluder
    - view_ratio 1.0 is single-best-view (sharpest, seam-prone); large values average everything

    Args:
        verts: (P, 3) vertex positions.
        faces: (F, 3) vertex indices.
        normals: (P, 3) vertex normals; texels facing away from a view take nothing from it.
        uvs: (F, 3, 2) per-corner UVs in [0, 1], v up (from unwrap_view_charts).
        rgbs: (N, H, W, 3) uint8 images.
        c2w: (N, 4, 4) camera-to-world poses.
        K: (N, 3, 3) intrinsics at image resolution.
        tex_size: atlas edge in texels.
        occlusion_eps: depth-test tolerance in world units (≈ voxel size); stops self-occlusion.
        occluder: (verts, faces) to depth-test against instead of the mesh.
        view_ratio: keep views whose pixel size at the texel is within this factor of the texel's best.

    Returns:
        (tex_size, tex_size, 3) uint8 albedo.
    """
    # World position and normal of every covered texel, interpolated from per-corner attributes
    rast = _rasterize_uvs(uvs, tex_size)
    covered = rast[0, ..., 3] > 0
    corners = np.concatenate([verts[faces], normals[faces]], -1).reshape(-1, 6).astype(np.float32)
    tri = torch.arange(len(corners), dtype=torch.int32, device="cuda").reshape(-1, 3)
    attr, _ = dr.interpolate(torch.from_numpy(corners).cuda()[None], rast, tri)
    texels = attr[0][covered]
    points, dirs = texels[:, :3], F.normalize(texels[:, 3:], dim=1)
    del rast, attr, texels

    # Mesh whose per-view depth render decides occlusion; a separate occluder wins when given
    occ_verts, occ_faces = (verts, faces) if occluder is None else occluder
    verts_h, faces_t = _to_cuda_mesh(occ_verts, occ_faces)
    ctx = dr.RasterizeCudaContext()
    w2c = invert_poses(c2w)
    w2c = torch.from_numpy(w2c.astype(np.float32)).cuda()
    K_t = torch.from_numpy(K.astype(np.float32)).cuda()
    height, width = rgbs.shape[1:3]

    # Pass one: finest pixel size any view achieves per texel, used to gate voters in pass two
    best = torch.full((len(points),), torch.inf, device="cuda")

    for k in range(len(rgbs)):
        _, depth = _rasterize_view(ctx, verts_h, faces_t, w2c[k], K_t[k], height, width)
        px_size, _ = _visible_pixel_size(points, dirs, depth, w2c[k], K_t[k], occlusion_eps)
        best = torch.minimum(best, px_size)

    # Pass two: bilinear color from the views within view_ratio of each texel's best
    rgb_acc = torch.zeros(len(points), 3, device="cuda")
    w_acc = torch.zeros(len(points), device="cuda")
    frame = torch.tensor([width - 1, height - 1], device="cuda")

    for k in range(len(rgbs)):
        _, depth = _rasterize_view(ctx, verts_h, faces_t, w2c[k], K_t[k], height, width)
        px_size, pixels = _visible_pixel_size(points, dirs, depth, w2c[k], K_t[k], occlusion_eps)
        keep = torch.isfinite(px_size) & (px_size <= best * view_ratio)

        # Pixel positions to grid_sample's [-1, 1] frame; corners are pixel centers
        grid = pixels[keep] / frame * 2 - 1
        image = torch.from_numpy(rgbs[k]).cuda().permute(2, 0, 1)[None].float() / 255.0
        color = F.grid_sample(image, grid[None, None], align_corners=True)[0, :, 0].T

        # Weight by how finely this view resolves the texel
        weight = best[keep] / px_size[keep]
        rgb_acc[keep] += color * weight[:, None]
        w_acc[keep] += weight

    # Normalize the weighted sum back into the atlas
    hit = w_acc > 0
    albedo = torch.zeros(tex_size, tex_size, 3, device="cuda")
    albedo[covered] = rgb_acc / w_acc.clamp(min=1e-12)[:, None] * hit[:, None]
    seen = torch.zeros(tex_size, tex_size, dtype=torch.bool, device="cuda")
    seen[covered] = hit

    # Two unrelated blacks: atlas space no chart claimed, and surface no camera ever reached
    logger.info(
        "atlas %.1f%% surface texels | %.1f%% of surface unseen | %.1f%% of atlas blank",
        100 * float(covered.float().mean()),
        100 * (1 - float(hit.float().mean())),
        100 * (1 - float(seen.float().mean())),
    )

    # Fill everything no view reached, then quantize
    albedo = fill_missing_pixels(albedo.cpu().numpy(), seen.cpu().numpy())
    return to_uint8_hwc(albedo, channels_first=False)


def _visible_pixel_size(
    points: torch.Tensor,
    dirs: torch.Tensor,
    depth: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    World size one source pixel covers at each texel, and the pixel (u, v) it projects to.

    - pixel size = dist / (fx * cos): grazing and distant views resolve the surface coarsely
    - inf for back-faces, texels behind the camera or out of frame, and texels the depth render hides
    - hidden: depth at the nearest pixel; testing all 4 bilinear pixels hides grazing ground from itself
    - depth: the occluder's (H, W) camera depth in this view, 0 where empty; w2c, K float32 cuda
    """
    height, width = depth.shape
    pixels, cam = project(points, w2c, K)

    # Facing the camera: the camera-frame normal points back along the camera-frame position
    dist = cam.norm(dim=1)
    cam_dirs = dirs @ w2c[:3, :3].T
    cos = -(cam_dirs * cam).sum(1) / dist

    # In front of the camera and inside the frame
    u, v = pixels.unbind(1)
    ok = (cos > 0) & (cam[:, 2] > 0) & (u >= 0) & (v >= 0) & (u <= width - 1) & (v <= height - 1)

    # Occluded when the depth render at the nearest pixel is nearer than the texel by more than eps
    front = _nearest_depth(depth, pixels)
    ok &= (front == 0) | (cam[:, 2] < front + eps)
    return torch.where(ok, dist / (K[0, 0] * cos), torch.inf), pixels


def _dilate_chart_gutters(albedo: np.ndarray, uvs: np.ndarray, boxes: np.ndarray, steps: int = 2) -> np.ndarray:
    """
    Texels no face covers filled from covered 8-neighbors in the same chart box, uint8.

    - steps should reach the gutter width (pad + 1)
    """
    tex_size = len(albedo)
    rast = _rasterize_uvs(uvs, tex_size)
    cov = rast[0, ..., 3] > 0
    del rast

    # Chart-box id per texel, -1 outside every box
    box = np.full((tex_size, tex_size), -1, dtype=np.int32)

    for c, (x, y, w, h) in enumerate(boxes):
        box[y : y + h, x : x + w] = c

    img = torch.from_numpy(albedo).cuda().float()
    bid = torch.from_numpy(box).cuda()

    for _ in range(steps):
        acc = torch.zeros_like(img)
        cnt = torch.zeros(cov.shape, device="cuda")

        # Sum of covered same-box neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                same_box = torch.roll(bid, (dy, dx), (0, 1)) == bid
                ok = torch.roll(cov, (dy, dx), (0, 1)) & same_box & (bid >= 0)
                acc += torch.roll(img, (dy, dx), (0, 1)) * ok[..., None]
                cnt += ok

        grow = ~cov & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        cov = cov | grow

    return img.round().clamp(0, 255).byte().cpu().numpy()


########################################################################
# Helpers
########################################################################


def _gl_projection(K: np.ndarray, width: int, height: int, near: float = 0.01, far: float = 1000.0) -> np.ndarray:
    """
    OpenCV pinhole K as an OpenGL clip matrix, (4, 4) float32.

    - NDC y = -1 is image row 0, matching nvdiffrast's first output row
    - raster pixel j is centered on u = j, the package convention (`unproject`); hence the + 1 on cx, cy
    """
    fx, fy, cx, cy = extract_intrinsics(K)
    return np.array(
        [
            [2 * fx / width, 0, (2 * cx + 1) / width - 1, 0],
            [0, 2 * fy / height, (2 * cy + 1) / height - 1, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )


def _rasterize_view(
    ctx: dr.RasterizeCudaContext,
    verts_h: torch.Tensor,
    faces: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Face-id raster and camera depth of a mesh in one view.

    - verts_h: (P, 4) homogeneous float32 cuda vertices; faces: (F, 3) int32 cuda; w2c, K float32 cuda
    - returns rast (H, W, 4), whose channel 3 is face id + 1 (0 = empty), and depth (H, W), 0 where empty
    """
    proj = torch.from_numpy(_gl_projection(K, width, height)).cuda()
    clip = verts_h @ (proj @ w2c).T
    rast, _ = dr.rasterize(ctx, clip[None], faces, resolution=[height, width])

    # Camera z interpolated over the same fragments
    z = (verts_h @ w2c[2])[:, None].contiguous()
    depth, _ = dr.interpolate(z[None], rast, faces)
    hit = rast[0, ..., 3] > 0
    return rast[0], depth[0, ..., 0] * hit


def _nearest_depth(depth: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """
    Depth render value at the pixel nearest each (u, v), clamped into the frame.
    """
    height, width = depth.shape
    rows = pixels[:, 1].round().long().clamp(0, height - 1)
    cols = pixels[:, 0].round().long().clamp(0, width - 1)
    return depth[rows, cols]


def _rasterize_uvs(uvs: np.ndarray, tex_size: int) -> torch.Tensor:
    """
    Atlas raster of (F, 3, 2) UVs with one vertex per corner, (1, tex_size, tex_size, 4) cuda.

    - channel 3 is face id + 1 (0 = empty); corner triangles are arange(3F).reshape(-1, 3)
    - v flipped, since nvdiffrast's first output row is the top of the atlas
    """
    flat = uvs.reshape(-1, 2).astype(np.float32)
    zeros = np.zeros(len(flat), dtype=np.float32)
    clip = np.stack([flat[:, 0] * 2 - 1, (1 - flat[:, 1]) * 2 - 1, zeros, zeros + 1], -1)
    tri = torch.arange(len(flat), dtype=torch.int32, device="cuda").reshape(-1, 3)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, torch.from_numpy(clip).cuda()[None], tri, resolution=[tex_size, tex_size])
    return rast


def _to_cuda_mesh(verts: np.ndarray, faces: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Homogeneous float32 vertices and int32 faces on the GPU, as nvdiffrast takes them.
    """
    v = torch.from_numpy(verts.astype(np.float32)).cuda()
    verts_h = torch.cat([v, torch.ones_like(v[:, :1])], 1)
    return verts_h, torch.from_numpy(faces.astype(np.int32)).cuda()


def _signed_area(uv: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Signed 2D area per triangle of (F, 3, 2) corners; numpy or torch.
    """
    e1 = uv[:, 1] - uv[:, 0]
    e2 = uv[:, 2] - uv[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1])
