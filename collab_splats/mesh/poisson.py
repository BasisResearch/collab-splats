from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collab_splats.mesh.base import BaseMeshCreator, MeshResult


@dataclass
class DepthNormalPoisson(BaseMeshCreator):
    """Backproject rendered depth+normals → Poisson reconstruction. Not yet implemented."""

    poisson_depth: int = 9

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        normals: np.ndarray | None = None,
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError("DepthNormalPoisson not yet implemented")


@dataclass
class GaussiansPoisson(BaseMeshCreator):
    """Gaussian means+normals → Poisson reconstruction. Not yet implemented."""

    poisson_depth: int = 9

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        means: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError("GaussiansPoisson not yet implemented")
