from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshResult:
    mesh_path: Path
    pcd_path: Path | None = None
    vertex_features: np.ndarray | None = None


@dataclass
class BaseMeshCreator:
    output_dir: Path

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError(f"{type(self).__name__}.create() is not implemented")
