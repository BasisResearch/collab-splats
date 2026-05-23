"""Shared fixtures for pointcloud tests."""
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.loop_closure import Submap


@pytest.fixture
def identity_submap_factory():
    """Returns a factory that builds an identity-perturbed Submap."""

    def _make(submap_id: int, k: int = 3) -> Submap:
        np.random.seed(0)
        poses = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
        poses[:, :3, 3] = np.random.randn(k, 3).astype(np.float32) * 0.01
        return Submap(
            submap_id=submap_id,
            frames=torch.zeros(k, 3, 64, 64),
            poses=poses,
            intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
            retrieval_vectors=torch.zeros(k, 128),
            image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
        )

    return _make
