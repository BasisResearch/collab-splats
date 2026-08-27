"""
PGSR geometry: plane depth, its two multi-view losses' machinery, and neighbour-view selection.

Reimplemented from GS-SR (`yanxian-ll/GS-SR`, `gssr/scene/pgsr_scene.py`,
`gssr/utils/{point_utils,graphics_utils,mvsnet_utils}.py`) and the plane rasterizer it calls
(`zju3dv/PGSR`, `submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu`). Neither ships a
license, so nothing here is vendored — each function cites the upstream site it reproduces.

Frames and conventions in this module:

- poses are `world_to_cam` (4, 4) OpenCV, points are column vectors: `x_cam = R @ x_world + t`.
  Upstream stores the transpose and right-multiplies, so every rotation here is the transpose of
  the one in the cited line.
- pixel coordinates are gsplat's: the centre of pixel `i` sits at `i + 0.5`. Upstream uses the
  3DGS kernel's integer-centre convention; the half-pixel is the only difference.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Covisibility scoring (GS-SR gssr/utils/mvsnet_utils.py:306-343): a co-visible point scores best
# when the two cameras see it about THETA0 degrees apart, and the penalty for being closer than
# that is ten times sharper than for being further away.
THETA0 = 5.0
SIGMA_BELOW = 1.0
SIGMA_ABOVE = 10.0


########################################
# Plane depth
########################################


# Numerical floors for the two perspective divides. Both bound a derivative rather than fix a value:
# a plane within MIN_RAY_COSINE of edge-on and a point within MIN_DEPTH of the image plane are
# degenerate either way, and callers mask them out — the floors only keep the backward pass finite.
MIN_RAY_COSINE = 1e-4
MIN_DEPTH = 1e-6


def pixel_rays(height: int, width: int, intrinsics: Tensor) -> Tensor:
    """
    Camera-frame ray directions with unit z, one per pixel: `((u - cx)/fx, (v - cy)/fy, 1)`.

    - Returns (C, H, W, 3) for intrinsics (C, 3, 3).
    """
    device = intrinsics.device
    u = torch.arange(width, device=device, dtype=intrinsics.dtype) + 0.5
    v = torch.arange(height, device=device, dtype=intrinsics.dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    # Broadcast the per-camera intrinsics over the pixel grid
    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    ray_x = (grid_u[None] - cx) / fx
    ray_y = (grid_v[None] - cy) / fy
    return torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)


def plane_depth(normal_map: Tensor, distance_map: Tensor, intrinsics: Tensor) -> Tensor:
    """
    PGSR's unbiased depth: the ray-plane intersection of the accumulated plane, per pixel.

    - `plane_depth = distance / -(n . ray)` with `ray = ((u-cx)/fx, (v-cy)/fy, 1)`
      (PGSR forward.cu:404). The minus is the normal's orientation: it faces the camera, so
      `n . x_cam <= 0` for the points it describes.
    - Both maps are the rasterizer's alpha-weighted sums, un-normalised; the ratio makes the
      missing `1/alpha` cancel, so no division by the accumulated alpha is needed.
    - `normal_map` (C, H, W, 3), `distance_map` (C, H, W, 1) -> (C, H, W, 1).
    """
    rays = pixel_rays(normal_map.shape[-3], normal_map.shape[-2], intrinsics)
    projected = (normal_map * rays).sum(dim=-1, keepdim=True)

    # Empty pixels accumulate no normal, so `projected` is 0 there and an additive epsilon would
    # give the depth a 1e8 derivative w.r.t. the distance map — finite, but it overflows to inf once
    # the multi-view terms multiply through it. Clamping the magnitude bounds that derivative and
    # only bites where the plane is within 1e-4 of edge-on, which is not a depth anyone can use.
    denominator = (-projected).clamp(min=MIN_RAY_COSINE)
    return distance_map / denominator


########################################
# Single-view planar loss
########################################


def image_gradient_weight(image: Tensor) -> Tensor:
    """
    Min-max normalised colour-gradient magnitude of an image, 1 on the border (GS-SR pgsr_scene.py:29-45).

    - `image` is (H, W, 3); returns (H, W). Interior pixels take the larger of the mean absolute
      horizontal and vertical central differences.
    """
    channels_first = image.permute(2, 0, 1)
    height, width = channels_first.shape[-2:]

    # Central differences, averaged over colour channels, then the stronger of the two directions
    bottom, top = channels_first[..., 2:height, 1 : width - 1], channels_first[..., 0 : height - 2, 1 : width - 1]
    right, left = channels_first[..., 1 : height - 1, 2:width], channels_first[..., 1 : height - 1, 0 : width - 2]
    grad_x = (right - left).abs().mean(dim=0)
    grad_y = (top - bottom).abs().mean(dim=0)
    grad = torch.maximum(grad_x, grad_y)

    # Normalise into [0, 1] and pad the one-pixel border with the maximum weight
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return F.pad(grad[None, None], (1, 1, 1, 1), mode="constant", value=1.0).squeeze()


def erode(mask: Tensor, ksize: int = 5) -> Tensor:
    """
    Grey erosion by min-pooling with a reflected border (GS-SR pgsr_scene.py:47-56).

    - `mask` is (1, 1, H, W); shrinks bright regions so the flat-area weight does not bleed over edges.
    """
    pad = (ksize - 1) // 2
    padded = F.pad(1 - mask, pad=[pad, pad, pad, pad], mode="reflect")
    return 1 - F.max_pool2d(padded, kernel_size=ksize, stride=1, padding=0)


def flat_region_weight(image: Tensor, ksize: int = 5) -> Tensor:
    """
    Per-pixel weight for the single-view planar loss: high on flat image regions, ~0 on edges.

    - `(1 - gradient)^5`, eroded — upstream detaches it, so it never carries gradient
      (GS-SR pgsr_scene.py:110-113).
    """
    weight = (1.0 - image_gradient_weight(image)).clamp(0, 1).detach() ** 5
    return erode(weight[None, None], ksize=ksize).squeeze()


########################################
# Patch NCC
########################################


def patch_offsets(half_patch: int, device: torch.device) -> Tensor:
    """
    (1, (2h+1)^2, 2) integer pixel offsets of a square patch (GS-SR graphics_utils.py:185-187).
    """
    offsets = torch.arange(-half_patch, half_patch + 1, device=device)
    grid_v, grid_u = torch.meshgrid(offsets, offsets, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)


def patch_warp(homographies: Tensor, pixels: Tensor) -> Tensor:
    """
    Apply one homography per patch to its pixel coordinates (GS-SR graphics_utils.py:189-198).

    - `homographies` (N, 3, 3), `pixels` (N, P, 2) -> (N, P, 2).
    """
    homogeneous = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    warped = torch.einsum("nij,npj->npi", homographies, homogeneous)
    return warped[..., :2] / (warped[..., 2:] + 1e-10)


def lncc(reference: Tensor, neighbour: Tensor) -> tuple[Tensor, Tensor]:
    """
    Per-patch local NCC loss `1 - cc^2` clamped to [0, 2], with upstream's `< 0.9` keep mask.

    - Both inputs are (N, P) grey patches with P a square number (GS-SR pgsr_scene.py:58-95).
    - Invariant to a per-patch gain and bias, which is the point: it compares structure, not exposure.
    """
    n_patches, patch_area = neighbour.shape
    side = int(round(patch_area**0.5))

    # Sums over each patch, via a box filter evaluated only at the patch centre
    reference_sq = reference.view(n_patches, 1, side, side)
    neighbour_sq = neighbour.view(n_patches, 1, side, side)
    product = (reference * neighbour).view(n_patches, 1, side, side)
    kernel = torch.ones(1, 1, side, side, device=reference.device, dtype=reference.dtype)
    padding = side // 2

    def box_sum(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, stride=1, padding=padding)[:, :, padding, padding]

    reference_sum = box_sum(reference_sq)
    neighbour_sum = box_sum(neighbour_sq)
    reference_sq_sum = box_sum(reference_sq.pow(2))
    neighbour_sq_sum = box_sum(neighbour_sq.pow(2))
    product_sum = box_sum(product)

    # Covariance and variances about the patch means
    reference_mean = reference_sum / patch_area
    neighbour_mean = neighbour_sum / patch_area
    covariance = product_sum - neighbour_mean * reference_sum
    reference_var = reference_sq_sum - reference_mean * reference_sum
    neighbour_var = neighbour_sq_sum - neighbour_mean * neighbour_sum

    correlation = covariance * covariance / (reference_var * neighbour_var + 1e-8)
    ncc = torch.clamp(1 - correlation, 0.0, 2.0).mean(dim=1, keepdim=True)
    return ncc, ncc < 0.9


########################################
# Projection helpers
########################################


def unproject(depth: Tensor, world_to_cam: Tensor, intrinsics: Tensor) -> Tensor:
    """
    Lift a z-depth map to world points, one row per pixel: (H*W, 3).

    - `depth` (1, H, W, 1), `world_to_cam` (1, 4, 4), `intrinsics` (1, 3, 3).
    """
    rays = pixel_rays(depth.shape[-3], depth.shape[-2], intrinsics)
    points_cam = (rays * depth).reshape(-1, 3)
    rotation = world_to_cam[0, :3, :3]
    translation = world_to_cam[0, :3, 3]
    return (points_cam - translation) @ rotation


def project(points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor) -> tuple[Tensor, Tensor]:
    """
    Project world points into a camera. Returns (pixels (N, 2), camera-frame points (N, 3)).
    """
    rotation = world_to_cam[0, :3, :3]
    translation = world_to_cam[0, :3, 3]
    points_cam = points_world @ rotation.transpose(-1, -2) + translation
    fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
    cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]
    # Points at or behind the camera divide by ~0; clamping keeps them finite (and grad-free through
    # the clamp) so a caller's validity mask can drop them without 0 * inf poisoning the backward
    depth = points_cam[:, 2:3].clamp(min=MIN_DEPTH)
    pixels = torch.stack([points_cam[:, 0] * fx / depth[:, 0] + cx, points_cam[:, 1] * fy / depth[:, 0] + cy], dim=-1)
    return pixels, points_cam


def sample_at_pixels(image: Tensor, pixels: Tensor, height: int, width: int) -> Tensor:
    """
    Bilinearly sample a (1, C, H, W) map at continuous pixel coordinates (N, 2) -> (N, C).

    - Pixel centres are at `i + 0.5`, so the normalised grid maps `0.5 -> -1` and `W - 0.5 -> 1`.
    """
    grid_x = 2 * (pixels[:, 0] - 0.5) / (width - 1) - 1.0
    grid_y = 2 * (pixels[:, 1] - 0.5) / (height - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1)


def to_gray(image: Tensor) -> Tensor:
    """
    ITU-R 601-2 luma of an (H, W, 3) image -> (1, 1, H, W), matching torchvision's Grayscale.

    - GS-SR builds every camera's grey image this way (gssr/cameras/__init__.py:64).
    """
    weights = torch.tensor([0.2989, 0.587, 0.114], device=image.device, dtype=image.dtype)
    return (image * weights).sum(dim=-1)[None, None]


########################################
# Neighbour-view selection
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
) -> list[list[int]]:
    """
    Per view, the `num_views` best-scoring neighbours by MVSNet covisibility (GS-SR mvsnet_utils.py:306-343).

    - A pair scores `sum over co-visible points of exp(-(theta - 5)^2 / (2 sigma^2))`, where theta is
      the angle the two cameras subtend at the point and sigma is 1 below 5 degrees, 10 above: a view
      that sees the same surface from a useful baseline beats both a near-duplicate and a wide one.
    - Upstream reads covisibility from COLMAP tracks; we have no per-image track ids at this boundary,
      so a point counts as seen when it projects inside the frame with positive depth.
    - Points are subsampled to `max_points` (the score is a sum over a shared point set, so thinning it
      scales every pair the same way).
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
    pixel_u = points_cam[..., 0] * fx / depths.clamp(min=1e-6) + cx
    pixel_v = points_cam[..., 1] * fy / depths.clamp(min=1e-6) + cy
    visible = (depths > 0) & (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)

    # Unit directions from each point to each camera centre, for the subtended angle
    centres = -torch.einsum("vij,vj->vi", rotations.transpose(-1, -2), translations)
    directions = F.normalize(centres[:, None] - points[None], dim=-1)

    # Score every pair against view i in one shot; the self-pair is masked out by its own -inf
    near_ids: list[list[int]] = []
    for view in range(n_views):
        cosine = (directions[view][None] * directions).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.rad2deg(torch.arccos(cosine))
        sigma = torch.where(theta <= THETA0, SIGMA_BELOW, SIGMA_ABOVE)
        weights = torch.exp(-((theta - THETA0) ** 2) / (2 * sigma**2))
        scores = (weights * (visible[view][None] & visible)).sum(dim=-1)
        scores[view] = float("-inf")

        # Keep only neighbours that actually share surface with this view
        top = torch.topk(scores, k=min(num_views, n_views - 1))
        near_ids.append([int(index) for index, score in zip(top.indices, top.values) if score > 0])

    empty = sum(1 for ids in near_ids if not ids)
    if empty:
        logger.warning(
            "pgsr: %d/%d views have no co-visible neighbour; their multi-view loss is skipped", empty, n_views
        )
    return near_ids


########################################
# Multi-view consistency
########################################


def pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Every pixel centre of an image, flattened row-major: (H*W, 2) of `(u + 0.5, v + 0.5)`.

    - `u` is the fastest-varying axis, exactly as in `pixel_rays`, so row `v * W + u` here is the
      same pixel as row `v * W + u` of `unproject`'s output. Upstream builds the same grid on
      integer centres (GS-SR pgsr_scene.py:120-121).
    """
    u = torch.arange(width, device=device, dtype=dtype) + 0.5
    v = torch.arange(height, device=device, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([grid_u, grid_v], dim=-1).reshape(-1, 2)


def _normalise_pixels(pixels: Tensor, height: int, width: int) -> Tensor:
    """
    Map pixel centres to `F.grid_sample`'s [-1, 1] coordinates, half-pixel convention.

    - Trailing dim is 2 `(u, v)`; any leading shape is preserved.
    - Centre of pixel `i` is `i + 0.5`, so `0.5 -> -1` and `W - 0.5 -> 1`, matching
      `align_corners=True`. Upstream normalises integer centres as `2x/(W-1) - 1`
      (GS-SR pgsr_scene.py:162-163); the half-pixel shift is this module's documented divergence.
    - Both the reference and the neighbour patch sampling go through here so the two cannot drift.
    """
    grid_x = 2 * (pixels[..., 0] - 0.5) / (width - 1) - 1.0
    grid_y = 2 * (pixels[..., 1] - 0.5) / (height - 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def forward_backward_noise(
    ref_plane_depth: Tensor,
    ref_world_to_cam: Tensor,
    ref_intrinsics: Tensor,
    near_plane_depth: Tensor,
    near_world_to_cam: Tensor,
    near_intrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Reprojection error of a reference pixel bounced off the neighbour's own rendered surface.

    - Reproduces GS-SR pgsr_scene.py:126-139 (`pts = get_points_from_depth(...)` through
      `pixel_noise = torch.norm(...)`) plus the mask of point_utils.py `get_points_depth_in_depth_map`.
    - Depth maps are (1, H, W, 1), poses (1, 4, 4) `world_to_cam`, intrinsics (1, 3, 3).
    - Returns `(pixel_noise (H*W,), valid (H*W,) bool)` in reference row-major pixel order.
    - Gradient flows through BOTH depth maps; the neighbour render is deliberately not detached.
    - Entries outside `valid` are zeroed, so the tensor is finite everywhere; still mask before reducing.
    """
    height, width = ref_plane_depth.shape[-3], ref_plane_depth.shape[-2]
    near_height, near_width = near_plane_depth.shape[-3], near_plane_depth.shape[-2]

    # Forward: the reference plane depth as world points, then into the neighbour camera
    points_world = unproject(ref_plane_depth, ref_world_to_cam, ref_intrinsics)
    pixels_near, points_near_cam = project(points_world, near_world_to_cam, near_intrinsics)

    # Upstream's validity: inside the neighbour frame and safely in front of it
    valid = (
        (pixels_near[:, 0] > 0)
        & (pixels_near[:, 0] < near_width)
        & (pixels_near[:, 1] > 0)
        & (pixels_near[:, 1] < near_height)
        & (points_near_cam[:, 2] > 0.1)
    )

    # Slide each point along its neighbour-camera ray onto the neighbour's own rendered surface
    map_z = sample_at_pixels(near_plane_depth.permute(0, 3, 1, 2), pixels_near, near_height, near_width)
    points_near_cam = points_near_cam / points_near_cam[:, 2:3].clamp(min=MIN_DEPTH) * map_z

    # Backward: undo the neighbour pose, then project into the reference camera
    rotation_near = near_world_to_cam[0, :3, :3]
    translation_near = near_world_to_cam[0, :3, 3]
    points_back_world = (points_near_cam - translation_near) @ rotation_near
    pixels_back, _ = project(points_back_world, ref_world_to_cam, ref_intrinsics)

    # Where each pixel started. `pixel_grid` shares `pixel_rays`' ordering, so row `v * W + u` here
    # is the same pixel as row `v * W + u` of `unproject`'s flattening — the two line up by
    # construction, and a mismatch would show as a large near-constant noise everywhere.
    start = pixel_grid(height, width, ref_plane_depth.device, ref_plane_depth.dtype)

    # Invalid rows are zeroed BEFORE the norm, not masked after it. Masking a non-finite entry out of
    # a reduction still runs its backward: the 0 the mask contributes meets an inf local derivative
    # and the product is NaN, which then lands in every upstream parameter. `torch.linalg.norm` is
    # also non-differentiable at 0, so the norm is spelled out with an epsilon under the root.
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
    Patch NCC between a reference view and a neighbour, warped by the per-pixel plane homography.

    - Reproduces the `## compute Homography` / `## compute neareast frame patch` / `## compute loss`
      blocks of GS-SR pgsr_scene.py:155-193.
    - Grays are (1, 1, H, W); `pixels` (M, 2) reference pixel centres; `ref_normal` (M, 3) and
      `ref_distance` (M,) are the RAW alpha-accumulated camera-frame plane at those pixels.
    - Returns `lncc`'s `(ncc (M, 1), mask (M, 1) bool)` unchanged; callers flatten the mask.
    - Gradient flows through `ref_normal` and `ref_distance` — that is the whole point of this loss.
      `pixels` and the sampled grays carry none.
    """
    ref_height, ref_width = ref_gray.shape[-2], ref_gray.shape[-1]
    near_height, near_width = near_gray.shape[-2], near_gray.shape[-1]
    n_patches = pixels.shape[0]

    # Square patch of reference pixel coordinates around each selected centre
    offsets = patch_offsets(half_patch, pixels.device).to(pixels.dtype)
    ori_pixels_patch = pixels[:, None, :] + offsets

    # Reference gray at those coordinates; zero padding off-frame is upstream's (pgsr_scene.py:164)
    ref_grid = _normalise_pixels(ori_pixels_patch, ref_height, ref_width).view(1, -1, 1, 2)
    ref_patch = F.grid_sample(ref_gray, ref_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    ref_patch = ref_patch.reshape(n_patches, -1)

    # Relative pose reference -> neighbour. With `x_cam = R x_world + t` (column vectors):
    #   x_world    = R_ref^T (x_ref - t_ref)
    #   x_near     = R_near x_world + t_near
    #              = R_near R_ref^T (x_ref - t_ref) + t_near
    #              = R_rel x_ref + (t_near - R_rel t_ref)
    # so R_rel = R_near R_ref^T and t_rel = t_near - R_rel t_ref. GS-SR pgsr_scene.py:167-168 writes
    # the same pair transposed, because its `world_view_transform` stores R^T with t in the last row.
    rotation_ref = ref_world_to_cam[0, :3, :3]
    translation_ref = ref_world_to_cam[0, :3, 3]
    rotation_near = near_world_to_cam[0, :3, :3]
    translation_near = near_world_to_cam[0, :3, 3]
    relative_rotation = rotation_near @ rotation_ref.transpose(-1, -2)
    relative_translation = translation_near - relative_rotation @ translation_ref

    # Plane-induced homography, one per patch (GS-SR pgsr_scene.py:177-181). The accumulated normal
    # faces the camera, so its plane is `n . x_ref = -d`; substituting that into the relative pose
    # gives `x_near = (R_rel - t_rel n^T / d) x_ref`, which K sandwiches into pixel coordinates.
    plane_term = relative_translation[None, :, None] @ (ref_normal[:, None, :] / ref_distance[:, None, None])
    homography = near_intrinsics[0][None] @ (relative_rotation[None] - plane_term) @ torch.linalg.inv(ref_intrinsics[0])

    # Neighbour gray at the warped patch, sampled through the same normalisation as the reference
    grid = patch_warp(homography, ori_pixels_patch)
    near_grid = _normalise_pixels(grid, near_height, near_width).view(1, -1, 1, 2)
    near_patch = F.grid_sample(near_gray, near_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    near_patch = near_patch.reshape(n_patches, -1)

    return lncc(ref_patch, near_patch)
