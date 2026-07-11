"""Notebook display helpers — visualization and debug utilities for tutorial notebooks.

These functions are intentionally print()-based and pyvista/matplotlib-dependent.
Do not import this module from core library code.

TODO(reorganize): review final home — may warrant a dedicated collab_splats/notebooks/ package
once the set of helpers stabilizes. See worklog/notes/2026-05-23-notebook-abstraction-opportunities.md.
"""

from __future__ import annotations

import numpy as np

########################################################
########## Semantics / feature helpers #################
########################################################


def feature_viz_row(
    axes,
    frame: np.ndarray,
    features,
    sim_map: np.ndarray,
    title_prefix: str = "",
    query_label: str = "",
) -> None:
    """Render PCA→RGB, similarity heatmap, and masked image into a row of 3 matplotlib axes."""
    from collab_splats.utils.visualization import (
        compute_heatmap,
        compute_masked_image,
        pca_to_rgb,
    )

    sim_title = f"{title_prefix}Similarity: {query_label}" if query_label else f"{title_prefix}Similarity"
    axes[0].imshow(pca_to_rgb(features, frame))
    axes[0].set_title(f"{title_prefix}PCA → RGB")
    axes[1].imshow(compute_heatmap(frame, sim_map))
    axes[1].set_title(sim_title)
    axes[2].imshow(compute_masked_image(frame, sim_map))
    axes[2].set_title(f"{title_prefix}Masked Image")
    for ax in axes:
        ax.axis("off")
