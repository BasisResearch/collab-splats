"""Shared fixtures for pointcloud tests.

Also ensures the project-root evals/ package is importable as `evals.datasets`,
not shadowed by tests/evals/ (which pytest adds to sys.path as a package).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.loop_closure import Submap

_PROJECT_ROOT = str(Path(__file__).parents[2])


@pytest.fixture(autouse=True)
def _fix_evals_import():
    """Evict tests/evals from sys.modules so project-root evals/ is used."""
    # Insert project root before tests/ directory
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    elif sys.path[0] != _PROJECT_ROOT:
        sys.path.remove(_PROJECT_ROOT)
        sys.path.insert(0, _PROJECT_ROOT)

    # Evict any cached evals module that points to tests/evals/
    for mod in list(sys.modules):
        if mod == "evals" or mod.startswith("evals."):
            m = sys.modules[mod]
            f = getattr(m, "__file__", "") or ""
            if "/tests/evals" in f or (not f and "/collab-splats/evals" not in f):
                del sys.modules[mod]

    yield


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
