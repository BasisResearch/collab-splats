"""
Photometric residual: a pixel carried into another frame by its depth keeps its brightness.

- port of pypose/bae @ c2c1c34 examples/rgbd_pose_refine/ (photometric.py, covis.py, geometry.py)
- poses are world-to-cam; upstream uses cam-to-world
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn.functional as F

import pypose as pp
from bae.autograd.function import map_transform

from collab_splats.utils.torch_utils import to_numpy

########################################################################
# Residual
########################################################################


@map_transform
def photometric_residual(
    pose_i: torch.Tensor,
    pose_j: torch.Tensor,
    x_i: torch.Tensor,
    K_j: torch.Tensor,
    I_i: torch.Tensor,
    I_j: torch.Tensor,
    gx_j: torch.Tensor,
    gy_j: torch.Tensor,
    uv_j: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Brightness error of a frame-i pixel carried into frame j; zero-padded to 3 wide.

    - only pose_i / pose_j are optimized; the rest is fixed at the last re-sample
    - frame-j brightness predicted from image slope (gx_j, gy_j) times pixel shift
    - @map_transform vectorizes it for the bae LM Jacobian; one row per sample

    Args:
        pose_i: SE3 world-to-cam of the source frame, 7 wide (a trailing focal is ignored).
        pose_j: SE3 world-to-cam of the target frame, same layout.
        x_i: source-camera 3D point of the sample.
        K_j: target-frame 3x3 K.
        I_i: source brightness at the sample.
        I_j: target brightness at the warped pixel, from the last re-sample.
        gx_j: target x image slope at the warped pixel.
        gy_j: target y image slope at the warped pixel.
        uv_j: warped pixel the target samples were taken at.
        weight: per-sample residual weight.

    Returns:
        [r, 0, 0] per sample.
    """
    x_w = pp.SE3(pose_i[..., :7]).Inv().Act(x_i)
    x_j = pp.SE3(pose_j[..., :7]).Act(x_w)
    uvw = torch.einsum("...ij,...j->...i", K_j, x_j)
    duv = uvw[..., :2] / uvw[..., 2:3] - uv_j
    r = (I_i - (I_j + gx_j * duv[..., 0] + gy_j * duv[..., 1])) * weight
    zero = torch.zeros_like(r)
    return torch.stack([r, zero, zero], dim=-1)


########################################################################
# Samples
########################################################################


def photometric_samples(
    world_to_cam: torch.Tensor,
    gray: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    n_neighbors: int = 16,
    max_view_angle_deg: float = 75.0,
    n_samples: int = 128,
    max_depth_diff: float = 0.1,
    consecutive_weight: float = 3.0,
    sigma: float = 0.05,
    min_depth: float = 1e-3,
    min_samples: int = 8,
    seed: int = 0,
) -> dict[str, torch.Tensor] | None:
    """
    Frame pairs and pixel samples for one image scale.

    - pairs: nearest cameras facing within max_view_angle_deg, both directions
    - pixels: drawn per frame by image slope; 128 not upstream's 2048 (256 runs out of memory)
    - pixel-corner K as upstream (center = i + 0.5); projection.py is pixel-center
    - occluded samples dropped: frame j's depth disagrees with the carried depth

    Args:
        world_to_cam: (N, 4, 4) poses.
        gray: (N, H, W) brightness.
        depth: (N, H, W) z-depth; <= min_depth is invalid.
        intrinsics: (N, 3, 3) K at this scale.
        n_neighbors: nearest cameras paired with each frame.
        max_view_angle_deg: widest angle between paired viewing directions.
        n_samples: pixels drawn per frame.
        max_depth_diff: relative depth disagreement that marks a sample occluded.
        consecutive_weight: weight of sequence-neighbor pairs relative to other pairs.
        sigma: brightness noise; weights are divided by it.
        min_depth: smallest valid depth.
        min_samples: fewest surviving samples worth returning.
        seed: seed of the pixel draw; the same seed and inputs draw the same pixels.

    Returns:
        photometric_residual inputs keyed by name, plus i_idx / j_idx frame indices, one row per
        sample; None when fewer than min_samples survive.
    """
    N, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    # Overlapping pairs: nearest camera centers facing within max_view_angle_deg
    cam_to_world = torch.linalg.inv(world_to_cam)
    centers = cam_to_world[:, :3, 3]
    forward = F.normalize(cam_to_world[:, :3, 2], dim=-1)
    max_view_angle = np.deg2rad(max_view_angle_deg)
    facing_away = forward @ forward.T <= np.cos(max_view_angle)
    dist = torch.cdist(centers, centers).fill_diagonal_(float("inf"))
    dist = dist.masked_fill(facing_away, float("inf"))
    near_dist, near_idx = torch.topk(dist, min(n_neighbors, N - 1), dim=1, largest=False)
    near_ok = torch.isfinite(near_dist)
    near_ok = to_numpy(near_ok)
    near_idx = to_numpy(near_idx)

    # Symmetric pair set, index-neighbor fallback so every frame appears
    pairs = set()
    for i in range(N):
        picked = set(near_idx[i, near_ok[i]].tolist())

        if len(picked) < 2:
            picked |= {(i - 1) % N, (i + 1) % N} - {i}

        pairs |= {(i, j) for j in picked} | {(j, i) for j in picked}

    pairs = torch.tensor(sorted(pairs), dtype=torch.long, device=device)

    # Signed central-difference gradients, one-sided at the borders
    gy, gx = torch.gradient(gray, dim=(1, 2))

    # Per-frame pixel pool drawn by image slope over valid depth, seeded; +0.5 = pixel centers (pixel-corner K)
    grad_mag = 0.5 * (gx.abs() + gy.abs())
    draw_weight = torch.where(depth > min_depth, grad_mag, 0.0).reshape(N, -1).clamp(min=1e-6)
    n_pool = min(n_samples, H * W)
    generator = torch.Generator(device=draw_weight.device)
    generator.manual_seed(seed)
    pool = torch.multinomial(draw_weight, n_pool, replacement=False, generator=generator)
    uv_pool = torch.stack([pool % W, pool // W], dim=-1).to(dtype) + 0.5
    depth_pool = torch.gather(depth.reshape(N, -1), 1, pool)

    # Every pair's pool warped into its target frame in one batch
    i_idx = pairs[:, 0].repeat_interleave(n_pool)
    j_idx = pairs[:, 1].repeat_interleave(n_pool)
    uv_i = uv_pool[pairs[:, 0]].reshape(-1, 2)
    d_i = depth_pool[pairs[:, 0]].reshape(-1)
    K_i = intrinsics[i_idx]
    xy_i = (uv_i - K_i[:, :2, 2]) / K_i[:, [0, 1], [0, 1]] * d_i[:, None]
    x_i = torch.cat([xy_i, d_i[:, None]], dim=-1)
    cam_i = pp.mat2SE3(world_to_cam[i_idx], check=False)
    cam_j = pp.mat2SE3(world_to_cam[j_idx], check=False)
    x_w = cam_i.Inv().Act(x_i)
    x_j = cam_j.Act(x_w)
    uvw = torch.einsum("mij,mj->mi", intrinsics[j_idx], x_j)
    uv_j = uvw[:, :2] / uvw[:, 2:3].clamp(min=1e-8)

    # Keep warps in front of camera j and inside its image
    z_j = x_j[:, 2]
    keep = (d_i > min_depth) & (z_j > min_depth)
    keep &= (uv_j[:, 0] >= 0) & (uv_j[:, 0] < W) & (uv_j[:, 1] >= 0) & (uv_j[:, 1] < H)
    i_idx, j_idx, x_i, uv_i, uv_j, z_j = (t[keep] for t in (i_idx, j_idx, x_i, uv_i, uv_j, z_j))

    # Bilinear samples per view: intensity at uv_i; intensity, gradients and depth at uv_j
    I_i = torch.empty(len(i_idx), dtype=dtype, device=device)
    target = torch.empty(len(i_idx), 4, dtype=dtype, device=device)
    size = uv_i.new_tensor([W, H])
    sample = functools.partial(F.grid_sample, align_corners=False, padding_mode="border")
    for view in range(N):
        src = i_idx == view
        dst = j_idx == view
        maps = torch.stack([gray[view], gx[view], gy[view], depth[view]])[None]

        if src.any():
            I_i[src] = sample(maps[:, :1], (uv_i[src] / size * 2 - 1)[None, None])[0, 0, 0]

        if dst.any():
            target[dst] = sample(maps, (uv_j[dst] / size * 2 - 1)[None, None])[0, :, 0].T

    I_j, gx_j, gy_j, D_j = target.unbind(1)

    # Occlusion check: target depth must agree with the warped point
    visible = (D_j > min_depth) & ((D_j - z_j).abs() <= max_depth_diff * z_j)

    if int(visible.sum()) < min_samples:
        return None

    # Consecutive pairs weigh more; sigma puts intensity on the pixel scale
    weight = torch.where((j_idx - i_idx).abs() == 1, consecutive_weight, 1.0).to(dtype) / sigma

    samples = {
        "i_idx": i_idx,
        "j_idx": j_idx,
        "x_i": x_i,
        "K_j": intrinsics[j_idx],
        "I_i": I_i,
        "I_j": I_j,
        "gx_j": gx_j,
        "gy_j": gy_j,
        "uv_j": uv_j,
        "weight": weight,
    }

    return {key: value[visible] for key, value in samples.items()}
