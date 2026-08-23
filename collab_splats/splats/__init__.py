"""
Gaussian-splat training on upstream gsplat from an existing pointcloud stage.
"""

# The gsplat commit pinned in pyproject.toml; recorded in every splats.zarr for provenance
GSPLAT_COMMIT = "d2f5c0f"

from .trainer import SplatsConfig, train  # noqa: E402

__all__ = ["GSPLAT_COMMIT", "SplatsConfig", "train"]
