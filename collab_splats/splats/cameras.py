"""
Per-camera optimization for splat training: pose deltas and per-image color correction.

- vendored: ``rotation_6d_to_matrix`` and ``CameraOpt``'s pose half from nerfstudio-project/gsplat
  @ d2f5c0f, examples/utils.py:132-153 and :27-63 (upstream ``CameraOptModule``); ``examples/`` is
  not shipped in the wheel.
- local: the 9-vector split into translation (3) + rotation (6) embeddings for separate learning
  rates, every embedding zero-initialized, and the per-image color affine on the same module.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

# Annotation-only: the trainer imports this module, so a runtime import back would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """
    Gram-Schmidt 6D rotation representation (Zhou et al. 2019) -> (..., 3, 3) rotation matrices.

    Args:
        rotation_6d: (..., 6) float, two unnormalized basis vectors concatenated.

    Returns:
        (..., 3, 3) right-handed rotation matrices, same dtype as the input.
    """
    # First basis vector: normalized first triple
    first_triple = rotation_6d[..., :3]
    second_triple = rotation_6d[..., 3:]
    basis_x = F.normalize(first_triple, dim=-1)

    # Second basis vector: second triple with its projection onto basis_x removed
    projection = (basis_x * second_triple).sum(-1, keepdim=True) * basis_x
    basis_y = F.normalize(second_triple - projection, dim=-1)

    # Third basis vector completes the right-handed frame
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-2)


########################################
# Camera-side optimization
########################################


class CameraOpt(torch.nn.Module):
    """
    Learned per-camera SE(3) delta and per-image color affine, either half optional.

    - halves selected independently: an unselected one is `None` and its accessor is a passthrough;
      `has_pose` / `has_appearance` are the predicates callers branch on.
    - every embedding zero-initialized, so a fresh module is the identity on both counts.
    - `optimizers` / `schedulers`: one entry per selected half, so `train()` steps them blind.
    """

    def __init__(self, n_views: int, *, optimize_pose: bool = True, optimize_appearance: bool = False):
        """
        Allocate the selected halves, zero-initialized, for ``n_views`` cameras.

        - an unselected half stays `None`; `optimizers` / `schedulers` are left for `from_config`
          to fill, so a directly constructed module is inert but usable.

        Args:
            n_views: number of training views; sizes every embedding.
            optimize_pose: allocate the (n_views, 3) translation and (n_views, 6) rotation halves.
            optimize_appearance: allocate the (n_views, 6) gain/bias half.
        """
        super().__init__()

        # Which halves are on; the predicates every caller asks instead of probing the embeddings
        self.has_pose = optimize_pose
        self.has_appearance = optimize_appearance

        # Pose delta per camera: translation (3, scene units) + rotation 6D (6), split for lrs
        self.translation = None
        self.rotation = None
        if optimize_pose:
            self.translation = torch.nn.Embedding(n_views, 3)
            self.rotation = torch.nn.Embedding(n_views, 6)
            torch.nn.init.zeros_(self.translation.weight)
            torch.nn.init.zeros_(self.rotation.weight)

            # Identity rotation in 6D form; the learned rotation delta is added to it
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        # Color affine per image: 3 gain deltas + 3 biases, zero = identity
        self.appearance = None
        if optimize_appearance:
            self.appearance = torch.nn.Embedding(n_views, 6)
            torch.nn.init.zeros_(self.appearance.weight)

        # Populated by from_config; empty when the module is constructed directly (tests, inference)
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

    @classmethod
    def from_config(
        cls,
        cfg: "SplatsConfig",
        n_views: int,
        world_extent: float,
        scene_scale: float,
        lr_gamma: float,
        device: str,
        *,
        weight_decay: float = 1e-6,
    ) -> "CameraOpt":
        """
        Build the halves the config selected, with their optimizers and exponential lr decay.

        Args:
            cfg: the run's SplatsConfig; `pose_opt` / `appearance_opt` select the halves and
                `pose_lr` / `appearance_lr` set their base learning rates.
            n_views: number of training views; sizes every embedding.
            world_extent: camera extent of the UN-normalized cameras; scales the rotation lr,
                which is unit-free and must not follow the training frame.
            scene_scale: camera extent of the training frame; scales the translation lr so steps
                keep their world-unit size.
            lr_gamma: per-step multiplicative decay, shared with the model's schedulers.
            device: torch device string.
            weight_decay: Adam weight decay on the pose deltas; 1e-6 as in gsplat @ d2f5c0f
                (`pose_opt_reg` at examples/simple_trainer.py:214, applied at :532-537).

        Returns:
            A CameraOpt with one optimizer and scheduler per selected half; both lists are empty
            when neither is selected.
        """
        module = cls(
            n_views,
            optimize_pose=cfg.pose_opt,
            optimize_appearance=cfg.appearance_opt,
        ).to(device)

        # Rotation follows the world extent, translation the training frame — see the arg docs
        if module.has_pose:
            optimizer = torch.optim.Adam(
                [
                    {"params": module.rotation.parameters(), "lr": cfg.pose_lr * world_extent},
                    {"params": module.translation.parameters(), "lr": cfg.pose_lr * scene_scale},
                ],
                weight_decay=weight_decay,
            )
            module.optimizers.append(optimizer)
            module.schedulers.append(ExponentialLR(optimizer, gamma=lr_gamma))

        # Appearance is one flat embedding at one learning rate
        if module.has_appearance:
            optimizer = torch.optim.Adam(module.appearance.parameters(), lr=cfg.appearance_lr)
            module.optimizers.append(optimizer)
            module.schedulers.append(ExponentialLR(optimizer, gamma=lr_gamma))

        return module

    def camera(self, cam_to_world: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Apply the learned pose deltas on the right of camera-to-world.

        - trap: the delta is built in the embedding's dtype, so a mismatched input raises.

        Args:
            cam_to_world: (..., 4, 4) camera-to-world poses.
            camera_ids: (...) long view indices, matching `cam_to_world`'s batch shape.

        Returns:
            (..., 4, 4) refined poses — the same tensor object when the pose half is off.
        """
        if not self.has_pose:
            return cam_to_world
        assert (
            cam_to_world.shape[:-2] == camera_ids.shape
        ), f"cam_to_world batch {cam_to_world.shape[:-2]} != camera_ids {camera_ids.shape}"
        batch_shape = cam_to_world.shape[:-2]

        # Look up each camera's translation and rotation deltas
        translation_delta = self.translation(camera_ids)
        rotation_delta = self.rotation(camera_ids)
        identity_6d = self.identity.expand(*batch_shape, -1)
        rotation = rotation_6d_to_matrix(rotation_delta + identity_6d)

        # Build the 4x4 delta transform and compose it onto the input pose
        delta_transform = torch.eye(4, device=translation_delta.device, dtype=translation_delta.dtype).repeat(
            (*batch_shape, 1, 1)
        )
        delta_transform[..., :3, :3] = rotation
        delta_transform[..., :3, 3] = translation_delta
        return torch.matmul(cam_to_world, delta_transform)

    def color(self, rgb: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Apply the learned per-image affine color correction.

        Args:
            rgb: (B, H, W, 3) rendered colors.
            camera_ids: (B,) long view indices.

        Returns:
            (B, H, W, 3) rgb * (1 + gain) + bias — the same tensor object when appearance is off.
        """
        if not self.has_appearance:
            return rgb
        params = self.appearance(camera_ids)
        gain = 1.0 + params[:, None, None, :3]
        bias = params[:, None, None, 3:]
        return rgb * gain + bias

    def color_params(self, camera_ids: Tensor) -> Tensor | None:
        """
        Raw per-image color parameters for ``appearance_reg``, or None when the half is off.

        Args:
            camera_ids: (B,) long view indices.

        Returns:
            (B, 6) gain deltas then biases, or None when appearance is off.
        """
        return self.appearance(camera_ids) if self.has_appearance else None

    def denormalize(self, scale: float) -> None:
        """
        Undo scene normalization on the pose translation deltas, in place.

        - deltas are in the camera frame (`cam_to_world @ delta`), so only 1 / scale applies
        - rotations are unit-free and untouched

        Args:
            scale: the scale `utils.scene_normalization` returned.

        Returns:
            None — modified in place. No-op when the pose half is off.
        """
        if not self.has_pose:
            return
        with torch.no_grad():
            self.translation.weight /= scale
