"""
Per-camera pose refinement for splat training.

Vendored from nerfstudio-project/gsplat @ d2f5c0f, examples/utils.py:
``CameraOptModule`` lines 27-63, ``rotation_6d_to_matrix`` lines 132-153.
``examples/`` is not shipped in the gsplat wheel, so the two pieces we need are copied verbatim.
Local change: the per-camera 9-vector is stored as two embeddings (translation 3, rotation 6)
so the trainer can give the two groups different learning rates — translation is in scene
units, rotation is not.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """
    Gram-Schmidt 6D rotation representation (Zhou et al. 2019) -> (..., 3, 3) rotation matrices.
    """
    # First basis vector: normalised first triple
    first_triple = rotation_6d[..., :3]
    second_triple = rotation_6d[..., 3:]
    basis_x = F.normalize(first_triple, dim=-1)

    # Second basis vector: second triple with its projection onto basis_x removed
    projection = (basis_x * second_triple).sum(-1, keepdim=True) * basis_x
    basis_y = F.normalize(second_triple - projection, dim=-1)

    # Third basis vector completes the right-handed frame
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-2)


class CameraOptModule(torch.nn.Module):
    """
    Learned per-camera SE(3) delta, applied on the right of camera-to-world.
    """

    def __init__(self, n_cameras: int):
        super().__init__()

        # Per camera: translation delta (3, scene units) + rotation delta in 6D form (6, unit-free)
        self.translation = torch.nn.Embedding(n_cameras, 3)
        self.rotation = torch.nn.Embedding(n_cameras, 6)

        # Identity rotation in 6D form; the learned rotation delta is added to it
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        """
        Reset all deltas to identity.
        """
        torch.nn.init.zeros_(self.translation.weight)
        torch.nn.init.zeros_(self.rotation.weight)

    def random_init(self, std: float):
        """
        Perturb deltas with N(0, std) noise.
        """
        torch.nn.init.normal_(self.translation.weight, std=std)
        torch.nn.init.normal_(self.rotation.weight, std=std)

    def forward(self, cam_to_world: Tensor, camera_ids: Tensor) -> Tensor:
        """
        Apply the learned deltas: cam_to_world (..., 4, 4), camera_ids (...) -> refined (..., 4, 4).

        The caller passes cam_to_world in the module's dtype (float32 by default); the delta
        transform is built in the embedding's dtype, so a dtype-mismatched cam_to_world raises.
        """
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
