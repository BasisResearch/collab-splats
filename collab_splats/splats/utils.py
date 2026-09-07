"""
Scene-geometry, view-scheduling and per-view preparation helpers for splat training.

- Pure functions only: no model, optimizer or loss; the trainer calls them around the step loop.
"""

import random
from collections.abc import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

########################################
# Scene geometry
########################################


def compute_scene_scale(cam_to_world: Tensor, *, margin: float = 1.1) -> float:
    """
    gsplat's scene-extent proxy: largest camera distance from the camera centroid, times a margin.

    Args:
        cam_to_world: (N, 4, 4) camera-to-world poses.
        margin: multiplier on the measured spread; 1.1 matches gsplat's simple_trainer.

    Returns:
        Scene extent as a python float, in the units of `cam_to_world`.
    """
    positions = cam_to_world[:, :3, 3]
    centroid = positions.mean(0)
    spread = (positions - centroid).norm(dim=-1).max()
    return float(spread) * margin


def scene_normalization(cam_to_world: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Splatfacto's Sim3: center = mean camera position, scale = 1 / max |camera coordinate - center|.

    - upstream: nerfstudio ``center_method="poses"`` + ``auto_scale_poses``, L-inf not L2.
    - no "up" re-orientation: no loss or lr depends on the world's rotation.

    Args:
        cam_to_world: (N, 4, 4) float camera-to-world poses in world units.

    Returns:
        (center (3,) float32 in world units, scale as a python float in 1 / world units).
    """
    positions = cam_to_world[:, :3, 3]
    center = positions.mean(0)
    spread = float(np.abs(positions - center).max())
    if spread <= 0:
        raise ValueError("splats: cannot normalize a scene whose cameras coincide")
    return center.astype(np.float32), 1.0 / spread


def denormalize_cameras(cam_to_world: Tensor, center: np.ndarray, scale: float) -> None:
    """
    Undo ``scene_normalization`` on camera translations, in place.

    Args:
        cam_to_world: (N, 4, 4) poses in the normalized frame; mutated to world units.
        center: (3,) the center `scene_normalization` returned.
        scale: the scale `scene_normalization` returned.

    Returns:
        None — `cam_to_world` is modified in place.
    """
    center_t = torch.as_tensor(center, dtype=torch.float32, device=cam_to_world.device)
    with torch.no_grad():
        cam_to_world[:, :3, 3] = cam_to_world[:, :3, 3] / scale + center_t


########################################
# Coarse-to-fine views
########################################


def downscale_factor(step: int, num_downscales: int, resolution_schedule: int) -> int:
    """
    Coarse-to-fine divisor at a step: 2 ** max(0, num_downscales - step // resolution_schedule).

    Args:
        step: current training step.
        num_downscales: how many halvings the run starts at; 0 disables the schedule.
        resolution_schedule: steps between halvings.

    Returns:
        Integer divisor, 1 once the schedule has run out.
    """
    if num_downscales <= 0:
        return 1
    return 2 ** max(0, num_downscales - step // resolution_schedule)


def downscale_view(image: np.ndarray, intrinsics: Tensor, factor: int) -> tuple[np.ndarray, Tensor]:
    """
    Image (bilinear) and K scaled by 1 / factor; passthrough at factor 1.

    Args:
        image: (H, W, 3) uint8 image.
        intrinsics: (1, 3, 3) camera matrix in pixels.
        factor: integer divisor from `downscale_factor`.

    Returns:
        ((H // factor, W // factor, 3) image, (1, 3, 3) scaled intrinsics) — the inputs themselves
        at factor 1, never mutated.
    """
    if factor == 1:
        return image, intrinsics

    height, width = image.shape[:2]
    small = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)
    K_small = intrinsics.clone()
    K_small[:, :2, :] /= factor
    return small, K_small


def prepare_target(image: np.ndarray, depth: np.ndarray | None, device: str) -> dict:
    """
    One view's supervision targets as tensors on `device`.

    Args:
        image: (H, W, 3) uint8 image.
        depth: (h, w) float depth target, possibly at a different resolution, or None.
        device: torch device string.

    Returns:
        {"rgb": (1, H, W, 3) float in [0, 1], "depth": (1, H, W, 1) float or None}. Depth is
        resized nearest so that zeros (meaning "no target") stay exactly zero.
    """
    rgb = torch.from_numpy(image).to(device).float()[None] / 255.0
    if depth is None:
        return {"rgb": rgb, "depth": None}

    height, width = image.shape[:2]
    depth_nchw = torch.from_numpy(depth).to(device)[None, None]

    # Nearest, never bilinear: 0 means "no target" and must not blend into its neighbors
    depth_nchw = F.interpolate(depth_nchw, size=(height, width), mode="nearest")
    return {"rgb": rgb, "depth": depth_nchw.permute(0, 2, 3, 1)}


########################################
# View schedule
########################################


def view_order(n_views: int, *, seed: int = 42) -> Iterator[int]:
    """
    Splatfacto's view schedule: an endless stream of seeded shuffled epochs.

    - every view: max_steps / n_views (+-1) visits; with replacement, 5.7% relative sd at
      100 views / 30k steps.
    - upstream: nerfstudio-project/nerfstudio @ 50e0e3c, full_images_datamanager.py:152-161
      (seeded shuffle) and :396-399 (`pop(0)` drains and refills the epoch).
    - trap: upstream pops the front, we yield `reversed(order)` — not equal seed for seed for
      n_views > 1; `tests/splats/test_utils.py` pins ours.

    Args:
        n_views: number of training views.
        seed: RNG seed; 42 is the value every existing run was trained at.

    Yields:
        View indices in [0, n_views), forever.
    """
    rng = random.Random(seed)
    while True:
        order = list(range(n_views))
        rng.shuffle(order)
        yield from reversed(order)
