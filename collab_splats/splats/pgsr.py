"""
PGSR geometry: plane depth, its two multi-view losses' machinery, and neighbor-view selection.

- reimplemented, not vendored: neither upstream ships a license
- each function cites its own site
- upstream: `yanxian-ll/GS-SR` @ 566359be, `gssr/scene/pgsr_scene.py`,
  `gssr/utils/{point_utils,graphics_utils,mvsnet_utils}.py`
- plane rasterizer: `zju3dv/PGSR`,
  `submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu`
- every `GS-SR <file>:<line>` below is at 566359be; upstream moved them in an Oct 2025 refactor
- poses: `world_to_cam` (4, 4) OpenCV, column vectors, `x_cam = R @ x_world + t`
- rotations: upstream stores the transpose and right-multiplies, so every one here is transposed
- pixel centers: gsplat's `i + 0.5`, not the 3DGS kernel's integer centers
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


########################################
# Plane depth
########################################


def pixel_rays(height: int, width: int, intrinsics: Tensor) -> Tensor:
    """
    Camera-frame ray directions with unit z, one per pixel: `((u - cx)/fx, (v - cy)/fy, 1)`.

    - not normalized: `plane_depth` divides by the ray's z, which is 1 here
    - normalizing these: silently rescales every rendered depth

    Args:
        height: image height in pixels.
        width: image width in pixels.
        intrinsics: (C, 3, 3) camera matrices.

    Returns:
        (C, H, W, 3) camera-frame directions with z == 1.
    """
    # `pixel_grid` is flat (H*W, 2); reshape back to the image so the per-camera broadcast works
    pixels = pixel_grid(height, width, intrinsics.device, intrinsics.dtype).reshape(height, width, 2)
    grid_u, grid_v = pixels[..., 0], pixels[..., 1]

    # Broadcast the per-camera intrinsics over the pixel grid
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    ray_x = (grid_u[None] - cx) / fx
    ray_y = (grid_v[None] - cy) / fy
    return torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)


def plane_depth(normal_map: Tensor, distance_map: Tensor, intrinsics: Tensor, *, min_cosine: float = 1e-4) -> Tensor:
    """
    PGSR's unbiased depth: the ray-plane intersection of the accumulated plane, per pixel.

    - PGSR forward.cu:404
    - `depth = distance / -(n . ray)`, `ray = ((u-cx)/fx, (v-cy)/fy, 1)`
    - the minus: the normal faces the camera, so `n . x_cam <= 0`
    - both maps un-normalized alpha sums; the ratio cancels the missing `1/alpha`

    Args:
        normal_map: (C, H, W, 3) alpha-accumulated camera-frame plane normals.
        distance_map: (C, H, W, 1) alpha-accumulated plane distances.
        intrinsics: (C, 3, 3) camera matrices.
        min_cosine: ray-plane cosine floor; bounds the depth gradient near edge-on. No yaml key.

    Returns:
        (C, H, W, 1) plane depth.
    """
    rays = pixel_rays(normal_map.shape[-3], normal_map.shape[-2], intrinsics)
    projected = (normal_map * rays).sum(dim=-1, keepdim=True)

    # Clamp, not +eps: empty pixels have `projected` == 0, where +eps gives a 1e8 derivative -> inf
    denominator = (-projected).clamp(min=min_cosine)
    return distance_map / denominator


########################################
# Single-view planar loss
########################################


def image_gradient_weight(image: Tensor) -> Tensor:
    """
    Min-max normalized color-gradient magnitude, 1 on the border (GS-SR pgsr_scene.py:29-45).

    Args:
        image: (H, W, 3) image.

    Returns:
        (H, W) weight; the larger of the mean absolute horizontal and vertical central differences.
    """
    channels_first = image.permute(2, 0, 1)
    height, width = channels_first.shape[-2:]

    # Central differences, averaged over color channels, then the stronger of the two directions
    bottom, top = channels_first[..., 2:height, 1 : width - 1], channels_first[..., 0 : height - 2, 1 : width - 1]
    right, left = channels_first[..., 1 : height - 1, 2:width], channels_first[..., 1 : height - 1, 0 : width - 2]
    grad_x = (right - left).abs().mean(dim=0)
    grad_y = (top - bottom).abs().mean(dim=0)
    grad = torch.maximum(grad_x, grad_y)

    # Normalize into [0, 1] and pad the one-pixel border with the maximum weight
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return F.pad(grad[None, None], (1, 1, 1, 1), mode="constant", value=1.0).squeeze()


def erode(mask: Tensor, ksize: int = 5) -> Tensor:
    """
    Gray erosion by min-pooling with a reflected border (GS-SR pgsr_scene.py:47-56).

    Args:
        mask: (1, 1, H, W) mask.
        ksize: square pooling window.

    Returns:
        (1, 1, H, W) eroded mask; shrunk bright regions stop the weight bleeding over edges.
    """
    pad = (ksize - 1) // 2
    padded = F.pad(1 - mask, pad=[pad, pad, pad, pad], mode="reflect")
    return 1 - F.max_pool2d(padded, kernel_size=ksize, stride=1, padding=0)


def flat_region_weight(image: Tensor, ksize: int = 5) -> Tensor:
    """
    Per-pixel weight for the single-view planar loss: high on flat image regions, ~0 on edges.

    - GS-SR pgsr_scene.py:109-111
    - `(1 - gradient)^5`, eroded and detached, so it never carries gradient

    Args:
        image: (H, W, 3) image.
        ksize: erosion window.

    Returns:
        (H, W) weight.
    """
    weight = (1.0 - image_gradient_weight(image)).clamp(0, 1).detach() ** 5
    return erode(weight[None, None], ksize=ksize).squeeze()


########################################
# Patch NCC
########################################


def patch_offsets(half_patch: int, device: torch.device) -> Tensor:
    """
    Integer pixel offsets of a square patch (GS-SR graphics_utils.py:185-187).

    Args:
        half_patch: patch radius in pixels.
        device: device to build the offsets on.

    Returns:
        (1, (2h+1)^2, 2) offsets.
    """
    offsets = torch.arange(-half_patch, half_patch + 1, device=device)
    grid_v, grid_u = torch.meshgrid(offsets, offsets, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)


def patch_warp(homographies: Tensor, pixels: Tensor) -> Tensor:
    """
    Apply one homography per patch to its pixel coordinates (GS-SR graphics_utils.py:189-198).

    Args:
        homographies: (N, 3, 3) per-patch homographies.
        pixels: (N, P, 2) patch pixel coordinates.

    Returns:
        (N, P, 2) warped pixels.
    """
    homogeneous = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    warped = torch.einsum("nij,npj->npi", homographies, homogeneous)
    return warped[..., :2] / (warped[..., 2:] + 1e-10)


def lncc(reference: Tensor, neighbor: Tensor) -> tuple[Tensor, Tensor]:
    """
    Per-patch local NCC loss `1 - cc^2` clamped to [0, 2], with upstream's `< 0.9` keep mask.

    - GS-SR pgsr_scene.py:58-95
    - invariant to a per-patch gain and bias: compares structure, not exposure

    Args:
        reference: (N, P) gray reference patches, P a square number.
        neighbor: (N, P) gray neighbor patches.

    Returns:
        (loss (N, 1), keep mask (N, 1) bool).
    """
    n_patches, patch_area = neighbor.shape
    side = int(round(patch_area**0.5))

    # Sums over each patch, via a box filter evaluated only at the patch center
    reference_sq = reference.view(n_patches, 1, side, side)
    neighbor_sq = neighbor.view(n_patches, 1, side, side)
    product = (reference * neighbor).view(n_patches, 1, side, side)
    kernel = torch.ones(1, 1, side, side, device=reference.device, dtype=reference.dtype)
    padding = side // 2

    # One box filter for all five sums; reading only the center tap makes it a whole-patch sum
    def box_sum(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, stride=1, padding=padding)[:, :, padding, padding]

    reference_sum = box_sum(reference_sq)
    neighbor_sum = box_sum(neighbor_sq)
    reference_sq_sum = box_sum(reference_sq.pow(2))
    neighbor_sq_sum = box_sum(neighbor_sq.pow(2))
    product_sum = box_sum(product)

    # Covariance and variances about the patch means
    reference_mean = reference_sum / patch_area
    neighbor_mean = neighbor_sum / patch_area
    covariance = product_sum - neighbor_mean * reference_sum
    reference_var = reference_sq_sum - reference_mean * reference_sum
    neighbor_var = neighbor_sq_sum - neighbor_mean * neighbor_sum

    correlation = covariance * covariance / (reference_var * neighbor_var + 1e-8)
    ncc = torch.clamp(1 - correlation, 0.0, 2.0).mean(dim=1, keepdim=True)
    return ncc, ncc < 0.9


########################################
# Projection helpers
########################################


def unproject(depth: Tensor, world_to_cam: Tensor, intrinsics: Tensor) -> Tensor:
    """
    Lift a z-depth map to world points, one row per pixel.

    Args:
        depth: (1, H, W, 1) z-depth map.
        world_to_cam: (1, 4, 4) OpenCV pose.
        intrinsics: (1, 3, 3) camera matrix.

    Returns:
        (H*W, 3) world points, row-major.
    """
    rays = pixel_rays(depth.shape[-3], depth.shape[-2], intrinsics)
    points_cam = (rays * depth).reshape(-1, 3)
    rotation = world_to_cam[0, :3, :3]
    translation = world_to_cam[0, :3, 3]
    return (points_cam - translation) @ rotation


def project(
    points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor, *, min_depth: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """
    Project world points into a camera.

    Args:
        points_world: (N, 3) world points.
        world_to_cam: (1, 4, 4) OpenCV pose.
        intrinsics: (1, 3, 3) camera matrix.
        min_depth: perspective-divide floor; a point this close projects with unbounded gradient.

    Returns:
        (pixels (N, 2), camera-frame points (N, 3)).
    """
    rotation = world_to_cam[0, :3, :3]
    translation = world_to_cam[0, :3, 3]
    points_cam = points_world @ rotation.transpose(-1, -2) + translation
    fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
    cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]

    # Points at or behind the camera divide by ~0; unclamped, a mask's 0 * inf poisons the backward
    depth = points_cam[:, 2:3].clamp(min=min_depth)
    pixels = torch.stack([points_cam[:, 0] * fx / depth[:, 0] + cx, points_cam[:, 1] * fy / depth[:, 0] + cy], dim=-1)
    return pixels, points_cam


def _normalize_pixels(pixels: Tensor, height: int, width: int) -> Tensor:
    """
    Map pixel centers to `F.grid_sample`'s [-1, 1] coordinates, half-pixel convention.

    - `0.5 -> -1`, `W - 0.5 -> 1`, matching `align_corners=True`
    - GS-SR pgsr_scene.py:162-163 normalizes integer centers as `2x/(W-1) - 1`; the half-pixel is
      this module's documented divergence
    - all three call sites route through here — `patch_ncc`'s two patch grids and
      `sample_at_pixels`' lookups — so the conventions cannot drift apart

    Args:
        pixels: (..., 2) `(u, v)` pixel centers; leading shape is preserved.
        height: image height in pixels.
        width: image width in pixels.

    Returns:
        (..., 2) grid-sample coordinates.
    """
    grid_x = 2 * (pixels[..., 0] - 0.5) / (width - 1) - 1.0
    grid_y = 2 * (pixels[..., 1] - 0.5) / (height - 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def sample_at_pixels(image: Tensor, pixels: Tensor, height: int, width: int) -> Tensor:
    """
    Bilinearly sample a map at continuous pixel coordinates.

    Args:
        image: (1, C, H, W) map.
        pixels: (N, 2) pixel centers, normalized through `_normalize_pixels` as `patch_ncc` does.
        height: image height in pixels.
        width: image width in pixels.

    Returns:
        (N, C) sampled values.
    """
    grid = _normalize_pixels(pixels, height, width).view(1, -1, 1, 2)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1)


def to_gray(image: Tensor) -> Tensor:
    """
    ITU-R 601-2 luma, matching torchvision's Grayscale (GS-SR gssr/cameras/__init__.py:64).

    Args:
        image: (H, W, 3) image.

    Returns:
        (1, 1, H, W) gray image.
    """
    weights = torch.tensor([0.2989, 0.587, 0.114], device=image.device, dtype=image.dtype)
    return (image * weights).sum(dim=-1)[None, None]


########################################
# Neighbor-view selection
########################################


@torch.no_grad()
def select_near_views(
    world_to_cam: Tensor,
    intrinsics: Tensor,
    points: Tensor,
    height: int,
    width: int,
    num_views: int = 5,
    max_points: int = 20000,
    *,
    theta0: float = 5.0,
    sigma_below: float = 1.0,
    sigma_above: float = 10.0,
) -> list[list[int]]:
    """
    Per view, the `num_views` best-scoring neighbors by MVSNet covisibility.

    - score: `sum over co-visible points of exp(-(theta - theta0)^2 / (2 sigma^2))`
    - theta: the angle the two cameras subtend at the point
    - covisibility: upstream reads COLMAP tracks; no per-image track ids reach this boundary, so a
      point counts as seen when it projects inside the frame with positive depth

    Args:
        world_to_cam: (N, 4, 4) OpenCV poses.
        intrinsics: (N, 3, 3) camera matrices at (height, width).
        points: (P, 3) world points; the shared set the score sums over.
        height: frame height in pixels.
        width: frame width in pixels.
        num_views: neighbors to keep per view.
        max_points: score-set size; the cloud is strided down, which scales every pair alike.
        theta0: ideal subtended angle in degrees.
        sigma_below: falloff below `theta0` — sharp, a near-duplicate view adds little.
        sigma_above: falloff above — ten times gentler, a wide baseline is only harder to match.

    Returns:
        One neighbor-index list per view, best first; empty for a view with no co-visible partner.

    Ported from GS-SR @ 566359be, gssr/utils/mvsnet_utils.py:306-343.
    """
    device = world_to_cam.device
    n_views = len(world_to_cam)

    # Thin the point cloud on a fixed stride so the score is deterministic across runs
    if len(points) > max_points:
        points = points[:: max(len(points) // max_points, 1)][:max_points]

    # Visibility per view: in front of the camera and inside the frame
    rotations = world_to_cam[:, :3, :3]
    translations = world_to_cam[:, :3, 3]
    points_cam = torch.einsum("vij,pj->vpi", rotations, points) + translations[:, None]
    depths = points_cam[..., 2]
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    cx = intrinsics[:, 0, 2][:, None]
    cy = intrinsics[:, 1, 2][:, None]
    # 1e-6 is inf-hygiene, not a tunable: every pixel it rescues is dropped by `depths > 0` below
    pixel_u = points_cam[..., 0] * fx / depths.clamp(min=1e-6) + cx
    pixel_v = points_cam[..., 1] * fy / depths.clamp(min=1e-6) + cy
    visible = (depths > 0) & (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)

    # Unit directions from each point to each camera center, for the subtended angle
    centers = -torch.einsum("vij,vj->vi", rotations.transpose(-1, -2), translations)
    directions = F.normalize(centers[:, None] - points[None], dim=-1)

    # Score every pair against view i in one shot; the self-pair is masked out by its own -inf
    near_ids: list[list[int]] = []
    for view in range(n_views):
        cosine = (directions[view][None] * directions).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.rad2deg(torch.arccos(cosine))
        sigma = torch.where(theta <= theta0, sigma_below, sigma_above)
        weights = torch.exp(-((theta - theta0) ** 2) / (2 * sigma**2))
        scores = (weights * (visible[view][None] & visible)).sum(dim=-1)
        scores[view] = float("-inf")

        # Keep only neighbors that actually share surface with this view
        top = torch.topk(scores, k=min(num_views, n_views - 1))
        near_ids.append([int(index) for index, score in zip(top.indices, top.values) if score > 0])

    empty = sum(1 for ids in near_ids if not ids)
    if empty:
        logger.warning(
            "pgsr: %d/%d views have no co-visible neighbor; their multi-view loss is skipped", empty, n_views
        )
    return near_ids


########################################
# Multi-view consistency
########################################


def pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Every pixel center of an image, flattened row-major.

    - GS-SR pgsr_scene.py:120-121 builds the same grid on integer centers
    - `pixel_rays` shares this grid, so row `v * W + u` matches `unproject`'s output row

    Args:
        height: image height in pixels.
        width: image width in pixels.
        device: device to build the grid on.
        dtype: grid dtype.

    Returns:
        (H*W, 2) of `(u + 0.5, v + 0.5)`.
    """
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)


def forward_backward_noise(
    ref_plane_depth: Tensor,
    ref_world_to_cam: Tensor,
    ref_intrinsics: Tensor,
    near_plane_depth: Tensor,
    near_world_to_cam: Tensor,
    near_intrinsics: Tensor,
    *,
    min_depth: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Reprojection error of a reference pixel bounced off the neighbor's own rendered surface.

    - GS-SR pgsr_scene.py:126-139, `pts = get_points_from_depth(...)` through
      `pixel_noise = torch.norm(...)`
    - mask: point_utils.py `get_points_depth_in_depth_map`
    - gradient flows through BOTH depth maps; the neighbor render is deliberately not detached

    Args:
        ref_plane_depth: (1, H, W, 1) reference plane depth.
        ref_world_to_cam: (1, 4, 4) reference pose.
        ref_intrinsics: (1, 3, 3) reference camera matrix.
        near_plane_depth: (1, H, W, 1) neighbor plane depth.
        near_world_to_cam: (1, 4, 4) neighbor pose.
        near_intrinsics: (1, 3, 3) neighbor camera matrix.
        min_depth: floors every perspective divide on the round trip, `project` included. No yaml.

    Returns:
        `(pixel_noise (H*W,), valid (H*W,) bool)`, reference row-major. Invalid entries are zeroed,
        not non-finite; still mask before reducing.
    """
    height, width = ref_plane_depth.shape[-3], ref_plane_depth.shape[-2]
    near_height, near_width = near_plane_depth.shape[-3], near_plane_depth.shape[-2]

    # Forward: the reference plane depth as world points, then into the neighbor camera
    points_world = unproject(ref_plane_depth, ref_world_to_cam, ref_intrinsics)
    pixels_near, points_near_cam = project(points_world, near_world_to_cam, near_intrinsics, min_depth=min_depth)

    # Upstream's validity: inside the neighbor frame and safely in front of it
    valid = (
        (pixels_near[:, 0] > 0)
        & (pixels_near[:, 0] < near_width)
        & (pixels_near[:, 1] > 0)
        & (pixels_near[:, 1] < near_height)
        & (points_near_cam[:, 2] > 0.1)
    )

    # Slide each point along its neighbor-camera ray onto the neighbor's own rendered surface
    map_z = sample_at_pixels(near_plane_depth.permute(0, 3, 1, 2), pixels_near, near_height, near_width)
    points_near_cam = points_near_cam / points_near_cam[:, 2:3].clamp(min=min_depth) * map_z

    # Backward: undo the neighbor pose, then project into the reference camera
    rotation_near = near_world_to_cam[0, :3, :3]
    translation_near = near_world_to_cam[0, :3, 3]
    points_back_world = (points_near_cam - translation_near) @ rotation_near
    pixels_back, _ = project(points_back_world, ref_world_to_cam, ref_intrinsics, min_depth=min_depth)

    # Where each pixel started; `pixel_grid` and `unproject` share `pixel_rays`' row ordering
    start = pixel_grid(height, width, ref_plane_depth.device, ref_plane_depth.dtype)

    # Zero the invalid rows BEFORE the norm
    # - masking a non-finite entry out of a reduction still runs its backward: 0 * inf = NaN
    # - norm spelled out: `torch.linalg.norm` is non-differentiable at 0
    difference = torch.where(valid[:, None], pixels_back - start, torch.zeros_like(pixels_back))
    pixel_noise = torch.sqrt((difference * difference).sum(dim=-1) + 1e-12)
    return pixel_noise, valid


def patch_ncc(
    ref_gray: Tensor,
    near_gray: Tensor,
    pixels: Tensor,
    ref_normal: Tensor,
    ref_distance: Tensor,
    ref_world_to_cam: Tensor,
    ref_intrinsics: Tensor,
    near_world_to_cam: Tensor,
    near_intrinsics: Tensor,
    half_patch: int = 3,
) -> tuple[Tensor, Tensor]:
    """
    Patch NCC between a reference view and a neighbor, warped by the per-pixel plane homography.

    - GS-SR pgsr_scene.py:155-193, the `## compute Homography` / `## compute neareast frame patch` /
      `## compute loss` blocks
    - gradient: `ref_normal` and `ref_distance` only; `pixels` and the grays carry none

    Args:
        ref_gray: (1, 1, H, W) reference gray image.
        near_gray: (1, 1, H, W) neighbor gray image.
        pixels: (M, 2) reference pixel centers.
        ref_normal: (M, 3) RAW alpha-accumulated camera-frame plane normals at those pixels.
        ref_distance: (M,) RAW alpha-accumulated plane distances at those pixels.
        ref_world_to_cam: (1, 4, 4) reference pose.
        ref_intrinsics: (1, 3, 3) reference camera matrix.
        near_world_to_cam: (1, 4, 4) neighbor pose.
        near_intrinsics: (1, 3, 3) neighbor camera matrix.
        half_patch: patch radius in pixels.

    Returns:
        `lncc`'s `(ncc (M, 1), mask (M, 1) bool)` unchanged; callers flatten the mask.
    """
    ref_height, ref_width = ref_gray.shape[-2], ref_gray.shape[-1]
    near_height, near_width = near_gray.shape[-2], near_gray.shape[-1]
    n_patches = pixels.shape[0]

    # Square patch of reference pixel coordinates around each selected center
    offsets = patch_offsets(half_patch, pixels.device).to(pixels.dtype)
    ori_pixels_patch = pixels[:, None, :] + offsets

    # Reference gray at those coordinates; zero padding off-frame is upstream's (pgsr_scene.py:164)
    ref_grid = _normalize_pixels(ori_pixels_patch, ref_height, ref_width).view(1, -1, 1, 2)
    ref_patch = F.grid_sample(ref_gray, ref_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    ref_patch = ref_patch.reshape(n_patches, -1)

    # Relative pose reference -> neighbor, column vectors
    # - GS-SR pgsr_scene.py:167-168, same pair transposed
    # - R_rel = R_near R_ref^T, t_rel = t_near - R_rel t_ref
    rotation_ref = ref_world_to_cam[0, :3, :3]
    translation_ref = ref_world_to_cam[0, :3, 3]
    rotation_near = near_world_to_cam[0, :3, :3]
    translation_near = near_world_to_cam[0, :3, 3]
    relative_rotation = rotation_near @ rotation_ref.transpose(-1, -2)
    relative_translation = translation_near - relative_rotation @ translation_ref

    # Plane-induced homography, one per patch
    # - GS-SR pgsr_scene.py:177-181
    # - normal faces the camera: plane `n . x_ref = -d`, so `x_near = (R_rel - t_rel n^T / d) x_ref`
    plane_term = relative_translation[None, :, None] @ (ref_normal[:, None, :] / ref_distance[:, None, None])
    homography = near_intrinsics[0][None] @ (relative_rotation[None] - plane_term) @ torch.linalg.inv(ref_intrinsics[0])

    # Neighbor gray at the warped patch, sampled through the same normalization as the reference
    grid = patch_warp(homography, ori_pixels_patch)
    near_grid = _normalize_pixels(grid, near_height, near_width).view(1, -1, 1, 2)
    near_patch = F.grid_sample(near_gray, near_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    near_patch = near_patch.reshape(n_patches, -1)

    return lncc(ref_patch, near_patch)


########################################
# Neighbor rendering
########################################


def render_neighbor(
    model, image: np.ndarray, cam_to_world: Tensor, intrinsics: Tensor, camera_id: Tensor
) -> dict[str, Tensor]:
    """
    Render one co-visible neighbor view for the multi-view losses.

    - reads plane depth and gray only: no normals, appearance correction or background
    - NOT detached: the geometric term pulls both views' plane depths together
    - detaching it: a one-sided fit

    Args:
        model: the `Gaussians` or `Scaffold` being trained.
        image: (h, w, 3) uint8 neighbor frame, already downscaled to this step's resolution.
        cam_to_world: (1, 4, 4) neighbor pose, already pose-corrected.
        intrinsics: (1, 3, 3) neighbor camera matrix at the same resolution.
        camera_id: (1,) long neighbor view index.

    Returns:
        {"plane_depth", "gray", "world_to_cam", "intrinsics"} — what the two losses read.

    Ported from GS-SR @ 566359be, gssr/scene/pgsr_scene.py:206 (`get_train_loss_dict`).
    """
    height, width = image.shape[:2]

    # Full SH degree (step=None): only plane_depth is read back, and it carries no SH dependence
    render, _ = model.render(
        cam_to_world,
        intrinsics,
        width,
        height,
        camera_id,
        step=None,
        render_normals=False,
        render_plane=True,
    )
    gray = to_gray(torch.from_numpy(image).to(intrinsics.device).float() / 255.0)
    return {
        "plane_depth": render["plane_depth"],
        "gray": gray,
        "world_to_cam": torch.linalg.inv(cam_to_world),
        "intrinsics": intrinsics,
    }
