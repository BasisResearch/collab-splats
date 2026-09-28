"""
Gaussian-splat training on gsplat, starting from a pointcloud.

- `train` / `SplatsConfig`: fit a `Gaussians` or `Scaffold` model
- `load_checkpoint` / `render_views`: reload a model and re-render its views
"""

# gsplat commit pinned in pyproject.toml; a test checks they match
GSPLAT_COMMIT = "d2f5c0f"

from .checkpoint import load_checkpoint  # noqa: E402
from .gaussian import Gaussians  # noqa: E402
from .rendering import render_views  # noqa: E402
from .scaffold import Scaffold  # noqa: E402
from .trainer import SplatsConfig, train  # noqa: E402

__all__ = [
    "GSPLAT_COMMIT",
    "Gaussians",
    "Scaffold",
    "SplatsConfig",
    "load_checkpoint",
    "render_views",
    "train",
]
