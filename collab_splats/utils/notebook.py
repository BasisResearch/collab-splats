"""
Notebook display helpers for the tutorial notebooks.

- matplotlib-dependent; never imported from core library code
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from matplotlib.axes import Axes

from collab_splats.utils.visualization import (
    compute_heatmap,
    compute_masked_image,
    pca_to_rgb,
)

########################################################
########## Semantics / feature helpers #################
########################################################


def feature_viz_row(
    axes: Sequence[Axes],
    frame: np.ndarray,
    features: torch.Tensor | np.ndarray,
    sim_map: np.ndarray,
    title_prefix: str = "",
    query_label: str = "",
) -> None:
    """
    PCA colors, similarity heatmap and masked image in a row of three axes.

    Args:
        axes: three matplotlib axes, filled left to right.
        frame: (H, W, 3) uint8 image.
        features: (C, pH, pW) patch features for the PCA panel.
        sim_map: (H, W) similarity in [0, 1].
        title_prefix: prepended to every panel title.
        query_label: text query named in the heatmap title.
    """
    sim_title = (
        f"{title_prefix}Similarity: {query_label}"
        if query_label
        else f"{title_prefix}Similarity"
    )
    axes[0].imshow(pca_to_rgb(features, frame))
    axes[0].set_title(f"{title_prefix}PCA → RGB")
    axes[1].imshow(compute_heatmap(frame, sim_map))
    axes[1].set_title(sim_title)
    axes[2].imshow(compute_masked_image(frame, sim_map))
    axes[2].set_title(f"{title_prefix}Masked Image")
    for ax in axes:
        ax.axis("off")
