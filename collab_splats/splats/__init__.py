"""
Gaussian-splat training on gsplat, starting from a pointcloud.

- `train` / `SplatsConfig`: fit a `Gaussians` or `Scaffold` model
- `load_checkpoint` / `render_views`: reload a model and re-render its views
"""

# gsplat commit pinned in pyproject.toml; a test checks they match
GSPLAT_COMMIT = "d2f5c0f"

from collab_splats.splats.checkpoint import load_checkpoint  # noqa: E402
from collab_splats.splats.gaussian import Gaussians  # noqa: E402
from collab_splats.splats.rendering import render_views  # noqa: E402
from collab_splats.splats.scaffold import Scaffold  # noqa: E402
from collab_splats.splats.trainer import SplatsConfig, train  # noqa: E402

__all__ = [
    "GSPLAT_COMMIT",
    "Gaussians",
    "Scaffold",
    "SplatsConfig",
    "load_checkpoint",
    "render_views",
    "train",
]
