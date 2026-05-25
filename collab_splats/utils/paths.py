from __future__ import annotations

import logging
import os
from pathlib import Path

import collab_splats

logger = logging.getLogger(__name__)


########################################################################
# Internal helpers
########################################################################

def get_project_root() -> Path:
    """Return the collab_splats project root by walking up to pyproject.toml."""
    candidate = Path(collab_splats.__file__).parent
    while candidate != candidate.parent:
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        f"Project root not found: no pyproject.toml above {Path(collab_splats.__file__).parent}"
    )


########################################################################
# Public API
########################################################################

def get_cache_dir(dataset: str | None = None) -> Path:
    """Return the tutorial cache directory, optionally scoped to a dataset.

    Resolves to <project_root>/docs/source/.cache[/<dataset>] — the same location
    that tutorial notebooks previously reached via Path("../../.cache"). Set
    COLLAB_SPLATS_CACHE env var to override the root entirely.

    Args:
        dataset: Optional sub-directory name (e.g. "birds_c0043").

    Returns:
        Path to the cache directory (not guaranteed to exist; call .mkdir() as needed).
    """
    if "COLLAB_SPLATS_CACHE" in os.environ:
        root = Path(os.environ["COLLAB_SPLATS_CACHE"])
    else:
        root = get_project_root() / "docs" / "source" / ".cache"
    return root / dataset if dataset else root
