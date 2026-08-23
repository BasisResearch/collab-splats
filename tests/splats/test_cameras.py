"""
CameraOptModule: zero-init is identity; random-init perturbs; 6D rotations are proper rotations.
"""

import torch

from collab_splats.splats.cameras import CameraOptModule, rotation_6d_to_matrix


def test_zero_init_leaves_poses_unchanged():
    refiner = CameraOptModule(3)
    refiner.zero_init()
    cam_to_world = torch.eye(4).expand(3, 4, 4).clone()
    cam_to_world[:, :3, 3] = torch.arange(3).float()[:, None]
    camera_ids = torch.arange(3)
    refined = refiner(cam_to_world, camera_ids)
    assert torch.allclose(refined, cam_to_world)


def test_random_init_changes_poses():
    torch.manual_seed(0)
    refiner = CameraOptModule(2)
    refiner.random_init(std=0.1)
    cam_to_world = torch.eye(4).expand(2, 4, 4).clone()
    camera_ids = torch.arange(2)
    refined = refiner(cam_to_world, camera_ids)
    assert not torch.allclose(refined, cam_to_world)


def test_rotation_6d_gives_proper_rotations():
    torch.manual_seed(0)
    rotations = rotation_6d_to_matrix(torch.randn(5, 6))
    identity = torch.eye(3).expand(5, 3, 3)
    gram = rotations @ rotations.transpose(-1, -2)
    determinants = torch.linalg.det(rotations)
    assert torch.allclose(gram, identity, atol=1e-5)
    assert torch.allclose(determinants, torch.ones(5), atol=1e-5)


def test_forward_matches_module_dtype():
    refiner = CameraOptModule(2).double()
    refiner.random_init(std=0.1)
    cam_to_world = torch.eye(4).expand(2, 4, 4).clone().to(torch.float64)
    camera_ids = torch.arange(2)
    refined = refiner(cam_to_world, camera_ids)
    assert refined.dtype == torch.float64
