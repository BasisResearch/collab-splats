"""
Array inputs for TSDF fusion, and textured-PLY output.

  - upsample_depths: model-res depth onto the original-res RGB grid (guided filter)
  - render_tsdf_inputs: depth + RGB + poses from a trained splat checkpoint
  - write_textured_ply: mesh + per-corner UVs + albedo atlas
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from collab_splats.preproc.frames import read_frames

logger = logging.getLogger(__name__)

# Which 2dgs depth render feeds the TSDF
# - "expected" (default) is the alpha-weighted mean, defined wherever anything contributes
# - "median" is the ray's median-transmittance surface: sharper, but blank wherever no
#   gaussian crosses the median, which is what opens holes on grazing ground
# - measured on GH010229 (853 views, voxel 0.10): median fused 3.02M vertices in 94,937
#   components against expected's 3.95M in 426,695, so median wins on fragmentation and
#   loses on coverage — expected is the visually better mesh, which is the criterion
# - 3dgs renders only the expected depth and ignores the choice
DEPTH_SOURCES = ("expected", "median")


######## Depth upsampling


def _box(x, radius):
    """
    Normalized box filter, the O(1) primitive of the guided filter.

    Args:
        x: (H, W) float32 image.
        radius: int, half window; the kernel is 2 × radius + 1.
    Returns:
        (H, W) float32 local mean.
    """
    k = 2 * radius + 1
    return cv2.boxFilter(x, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide, src, radius, eps):
    """
    He et al. gray-guide guided filter: edge-preserving smoothing of src steered by guide.

    Args:
        guide: (H, W) float32 in [0, 1].
        src: (H, W) float32 signal to smooth.
        radius: int, box half window.
        eps: float, regularizer on the guide variance.
    Returns:
        (H, W) float32 filtered src.
    """
    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g
    cov_gs = _box(guide * src, radius) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


def _guided_upsample_depth(depth, rgb_full, crop_box, radius=None, eps=1e-3):
    """
    Upsample one model-res depth map into its crop region of the original-res RGB canvas.

    Args:
        depth: (h, w) float32 model-res depth, 0 = no observation.
        rgb_full: (H, W, 3) uint8 original-res frame, the guide.
        crop_box: (tl_x, tl_y, cr_x, cr_y) model crop in original pixels.
        radius: int or None, guided-filter half window; None = ~2 × the upsample factor.
        eps: float, guided-filter regularizer.
    Returns:
        (H, W) float32 depth; masked pixels stay 0, canvas outside the crop is 0.
    """
    H, W = rgb_full.shape[:2]
    tl_x, tl_y, cr_x, cr_y = (int(round(v)) for v in crop_box)
    cw, ch = cr_x - tl_x, cr_y - tl_y
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Degenerate crop box {crop_box} — original_coords are corrupt")
    if tl_x < 0 or tl_y < 0 or cr_x > W or cr_y > H:
        raise ValueError(f"Crop box {crop_box} lies outside the {H}x{W} canvas")

    # Nearest resize of depth and validity to crop size — blocky but never invents values
    depth_nn = cv2.resize(depth, (cw, ch), interpolation=cv2.INTER_NEAREST)
    valid_nn = (depth_nn > 0).astype(np.float32)

    # Gray guide in [0, 1] from the original-res crop; radius spans ~2x the upsample factor
    guide = cv2.cvtColor(rgb_full[tl_y:cr_y, tl_x:cr_x], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if radius is None:
        radius = max(1, int(np.ceil(2 * cw / depth.shape[1])))

    # Validity-weighted filtering: masked pixels contribute nothing to their neighbors
    num = _guided_filter(guide, depth_nn * valid_nn, radius, eps)
    den = _guided_filter(guide, valid_nn, radius, eps)
    filtered = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)

    # The guide must never resurrect deleted depth, and depth must stay non-negative
    filtered[valid_nn == 0] = 0.0
    np.maximum(filtered, 0.0, out=filtered)

    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[tl_y:cr_y, tl_x:cr_x] = filtered
    return canvas


def upsample_depths(depths, rgbs, crop_boxes):
    """
    Guided-filter upsample model-res depth maps onto their original-res RGB frames.

    Args:
        depths: (N, h, w) float model-res depth, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 original-res frames; H, W set the output size.
        crop_boxes: (N, 4) [tl_x, tl_y, cr_x, cr_y] model crops in original pixels
            (original_coords[:, :4]).
    Returns:
        (N, H, W) float32 depth at frame resolution.
    """
    depths = np.asarray(depths)
    rgbs = np.asarray(rgbs)
    crop_boxes = np.asarray(crop_boxes)
    if not (len(depths) == len(rgbs) == len(crop_boxes)):
        raise ValueError(f"{len(depths)} depths, {len(rgbs)} rgbs, {len(crop_boxes)} crop boxes")

    # One guided upsample per frame into a preallocated stack
    n, H, W = len(depths), rgbs.shape[1], rgbs.shape[2]
    out = np.zeros((n, H, W), dtype=np.float32)
    for i in range(n):
        out[i] = _guided_upsample_depth(np.asarray(depths[i], dtype=np.float32), rgbs[i], crop_boxes[i])
    return out


######## Splat rendering


def _to_numpy(t):
    """
    Torch tensor or array to a host numpy array.

    Args:
        t: torch.Tensor on any device, or array-like.

    Returns:
        numpy array.
    """
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def render_tsdf_inputs(ckpt_path, images_dir=None, device="cuda", depth_source="expected"):
    """
    Render depth (+ RGB) from a trained splat checkpoint at its training cameras.

    Args:
        ckpt_path: splats/ckpt.pt written by the splats stage.
        images_dir: keyframe directory (images/ + frames.json); when given, RGB comes from
            the source frames matched by image id instead of the render.
        device: torch device for rendering.
        depth_source: which 2dgs depth to fuse, "expected" or "median"; 3dgs has only the
            expected depth and ignores this.

    Returns:
        depths (N, H, W) float32 (0 where alpha is 0), rgbs (N, H, W, 3) uint8,
        c2w (N, 4, 4) float32, K (N, 3, 3) float32.
    """
    if depth_source not in DEPTH_SOURCES:
        raise ValueError(f"depth_source must be one of {DEPTH_SOURCES}, got {depth_source!r}")
    # Imported here: gsplat needs CUDA at import, and this module must load without it
    from collab_splats.splats.rendering import load_checkpoint, render_views

    model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width) = load_checkpoint(Path(ckpt_path), device)

    # Source frames replace rendered RGB when a keyframe directory is given
    rgbs = None
    if images_dir is not None:
        rgbs = read_frames(images_dir, [int(i) for i in image_ids])
        if rgbs.shape[1:3] != (height, width):
            raise ValueError(f"{images_dir} frames are {rgbs.shape[1:3]} but the checkpoint renders {(height, width)}")

    # Render every camera; only 2dgs offers a choice, so "median" falls back to "depth"
    key = "median_depth" if depth_source == "median" else "depth"
    depths, rendered = [], []
    for view in render_views(model, camera_opt, cam_to_world, intrinsics, height, width):
        depth = _to_numpy(view[key] if key in view else view["depth"]).reshape(height, width, -1)[..., 0]
        alpha = _to_numpy(view["alpha"]).reshape(height, width, -1)[..., 0]
        depths.append(np.where(alpha > 0, depth, 0.0).astype(np.float32))

        if rgbs is None:
            rgb = _to_numpy(view["rgb"]).reshape(height, width, -1)[..., :3]
            rendered.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    if rgbs is None:
        rgbs = np.stack(rendered)

    logger.info("render_tsdf_inputs: %d views at %dx%d from %s", len(depths), height, width, ckpt_path)
    return np.stack(depths), rgbs, _to_numpy(cam_to_world).astype(np.float32), _to_numpy(intrinsics).astype(np.float32)


######## Textured PLY


def write_textured_ply(mesh, uv, albedo, out_dir):
    """
    Write a mesh with per-corner UVs and its albedo atlas as mesh.ply + albedo.png.

    Args:
        mesh: open3d.geometry.TriangleMesh (legacy); vertices are split per face corner on write.
        uv: (F, 3, 2) float texture coordinates per face corner, Open3D bake convention (v up).
        albedo: (S, S, 3) uint8 RGB atlas.
        out_dir: Path or str, directory to create; receives mesh.ply and albedo.png.
    Returns:
        Path to out_dir/mesh.ply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One vertex per face corner so each carries a single (s, t)
    faces = np.asarray(mesh.triangles)
    xyz = np.asarray(mesh.vertices)[faces].reshape(-1, 3)
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)

    # Vertex color sampled from the atlas at each corner (viewers without texture support)
    h, w = albedo.shape[:2]
    row = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
    col = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    rgb = albedo[row, col]

    # Structured arrays for plyfile: x y z red green blue s t + vertex_indices
    vertex = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("s", "f4"),
            ("t", "f4"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertex["s"], vertex["t"] = uv[:, 0], uv[:, 1]
    face = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = np.arange(len(xyz), dtype=np.int32).reshape(-1, 3)

    # plyfile writes the comment verbatim; trimesh's loader reads TextureFile from it
    mesh_path = out_dir / "mesh.ply"
    elements = [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")]
    PlyData(elements, comments=["TextureFile albedo.png"]).write(str(mesh_path))

    # Atlas is RGB; cv2 writes BGR
    cv2.imwrite(str(out_dir / "albedo.png"), np.ascontiguousarray(albedo[..., ::-1]))
    logger.info("write_textured_ply: %d faces, %dx%d atlas -> %s", len(faces), w, h, mesh_path)
    return mesh_path
