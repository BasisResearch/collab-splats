"""
Gaussian-splat training on upstream gsplat from an existing pointcloud stage.
"""

# The gsplat commit pinned in pyproject.toml; asserted against pyproject by the cu121 migration test
GSPLAT_COMMIT = "d2f5c0f"

from .gaussian import Gaussians  # noqa: E402
from .rendering import load_checkpoint  # noqa: E402
from .scaffold import Scaffold  # noqa: E402
from .trainer import SplatsConfig, train  # noqa: E402

__all__ = [
    "GSPLAT_COMMIT",
    "Gaussians",
    "Scaffold",
    "SplatsConfig",
    "load_checkpoint",
    "train",
]
