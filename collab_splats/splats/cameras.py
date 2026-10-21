"""
Per-camera corrections learned during splat training.

- pose: a small SE(3) delta per camera
- appearance: a per-image color gain and bias
- ported from gsplat @ d2f5c0f, examples/utils.py:27-63 and :132-153 (`CameraOptModule`)
"""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

# Annotation-only: a runtime import of the trainer would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """
    Rotation matrices from the 6D representation (Zhou et al. 2019).

    Args:
        rotation_6d: (..., 6) two unnormalized basis vectors.

    Returns:
        (..., 3, 3) rotation matrices.
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
    Learned pose delta and color correction per camera; either half can be off.

    - an off half is `None` and its method passes the input through
    - zero-initialized, so a fresh module changes nothing
    """

    def __init__(
        self,
        n_views: int,
        *,
        optimize_pose: bool = True,
        optimize_appearance: bool = False,
    ):
        """
        Allocate the selected halves for `n_views` cameras.

        - no optimizers here; `from_config` adds them

        Args:
            n_views: number of training views.
            optimize_pose: learn a pose delta per camera.
            optimize_appearance: learn a color gain and bias per image.
        """
        super().__init__()

        # Which halves are on
        self.has_pose = optimize_pose
        self.has_appearance = optimize_appearance

        # Pose delta per camera: translation (3) + 6D rotation (6), separate lrs
        self.translation: torch.nn.Embedding | None = None
        self.rotation: torch.nn.Embedding | None = None

        if optimize_pose:
            self.translation = torch.nn.Embedding(n_views, 3)
            self.rotation = torch.nn.Embedding(n_views, 6)
            torch.nn.init.zeros_(self.translation.weight)
            torch.nn.init.zeros_(self.rotation.weight)

            # Identity rotation in 6D form; the learned delta is added to it
            self.register_buffer(
                "identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

        # Color per image: 3 gain deltas + 3 biases, zero = identity
        self.appearance: torch.nn.Embedding | None = None

        if optimize_appearance:
            self.appearance = torch.nn.Embedding(n_views, 6)
            torch.nn.init.zeros_(self.appearance.weight)

        # Filled by from_config
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
        Build the halves the config turns on, each with an optimizer and lr decay.

        Args:
            cfg: run config; `pose_opt` / `appearance_opt` pick the halves.
            n_views: number of training views.
            world_extent: camera extent before normalization; scales the rotation lr.
            scene_scale: camera extent after normalization; scales the translation lr.
            lr_gamma: per-step lr decay, shared with the model.
            device: torch device string.
            weight_decay: pose weight decay (gsplat @ d2f5c0f, examples/simple_trainer.py:214).

        Returns:
            The module with one optimizer and scheduler per selected half.
        """
        module = cls(
            n_views,
            optimize_pose=cfg.pose_opt,
            optimize_appearance=cfg.appearance_opt,
        ).to(device)

        # Pose: rotation lr follows the world extent, translation lr the training frame
        if module.has_pose:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": module.rotation.parameters(),
                        "lr": cfg.pose_lr * world_extent,
                    },
                    {
                        "params": module.translation.parameters(),
                        "lr": cfg.pose_lr * scene_scale,
                    },
                ],
                weight_decay=weight_decay,
            )
            module.optimizers.append(optimizer)
            module.schedulers.append(ExponentialLR(optimizer, gamma=lr_gamma))

        # Appearance: one embedding, one lr
        if module.has_appearance:
            optimizer = torch.optim.Adam(
                module.appearance.parameters(), lr=cfg.appearance_lr
            )
            module.optimizers.append(optimizer)
            module.schedulers.append(ExponentialLR(optimizer, gamma=lr_gamma))

        return module

    def camera(self, cam_to_world: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Poses with the learned delta applied: `cam_to_world @ delta`.

        Args:
            cam_to_world: (..., 4, 4) poses.
            camera_ids: (...) view indices, same batch shape.

        Returns:
            (..., 4, 4) refined poses; the input itself when pose is off.
        """
        if not self.has_pose:
            return cam_to_world

        if cam_to_world.shape[:-2] != camera_ids.shape:
            raise ValueError(
                f"cam_to_world batch {cam_to_world.shape[:-2]} != camera_ids {camera_ids.shape}"
            )

        batch_shape = cam_to_world.shape[:-2]
        assert self.translation is not None and self.rotation is not None

        # Look up each camera's translation and rotation deltas
        translation_delta = self.translation(camera_ids)
        rotation_delta = self.rotation(camera_ids)
        identity_6d = self.identity.expand(*batch_shape, -1)
        rotation = rotation_6d_to_matrix(rotation_delta + identity_6d)

        # Build the 4x4 delta and compose it onto the input pose
        delta_transform = torch.eye(
            4, device=translation_delta.device, dtype=translation_delta.dtype
        ).repeat((*batch_shape, 1, 1))
        delta_transform[..., :3, :3] = rotation
        delta_transform[..., :3, 3] = translation_delta
        return torch.matmul(cam_to_world, delta_transform)

    def color(self, rgb: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Colors with the learned per-image correction: `rgb * (1 + gain) + bias`.

        Args:
            rgb: (B, H, W, 3) rendered colors.
            camera_ids: (B,) view indices.

        Returns:
            (B, H, W, 3) corrected colors; the input itself when appearance is off.
        """
        if not self.has_appearance:
            return rgb

        assert self.appearance is not None
        params = self.appearance(camera_ids)
        gain = 1.0 + params[:, None, None, :3]
        bias = params[:, None, None, 3:]
        return rgb * gain + bias

    def color_params(self, camera_ids: Tensor) -> Tensor | None:
        """
        Raw color parameters, for the appearance regularizer.

        Args:
            camera_ids: (B,) view indices.

        Returns:
            (B, 6) gain deltas then biases; None when appearance is off.
        """
        return self.appearance(camera_ids) if self.appearance is not None else None

    def denormalize(self, scale: float) -> None:
        """
        Undo scene normalization on the translation deltas, in place.

        - rotations are scale-free and left alone

        Args:
            scale: the scale `utils.scene_normalization` returned.
        """
        if not self.has_pose:
            return

        assert self.translation is not None

        with torch.no_grad():
            self.translation.weight /= scale
