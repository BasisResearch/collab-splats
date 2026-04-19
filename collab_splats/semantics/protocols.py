"""
Capability protocols for feature extractors.

Using typing.Protocol (PEP 544, Python 3.8+) for structural duck typing:
any class that implements the required methods satisfies the protocol without
explicit inheritance. @runtime_checkable enables isinstance() checks at runtime.

Example:
    from collab_splats.semantics.protocols import SupportsTextQuery
    if isinstance(extractor, SupportsTextQuery):
        heatmaps = extractor.compute_semantic_heatmap(image, text_pairs)
"""

from typing import Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
from PIL import Image


@runtime_checkable
class SupportsTextQuery(Protocol):
    """
    Extractor capability: text query → per-label semantic heatmaps.

    Satisfied by Talk2DinoExtractor. Dashboard uses isinstance(extractor, SupportsTextQuery)
    to show/hide the semantic query tab without hardcoding extractor types.
    """

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings to normalized embeddings (N, D)."""
        ...

    def compute_semantic_heatmap(
        self,
        image: Image.Image,
        text_pairs: Dict[str, Tuple[List[str], List[str]]],
        softmax_temp: float,
        method: str,
    ) -> Dict[str, np.ndarray]:
        """
        Generate per-label masked image overlays.

        Args:
            image: PIL Image
            text_pairs: {"label": (positive_queries, negative_queries)}
            softmax_temp: softmax temperature (lower = sharper)
            method: "standard" or "pairwise"

        Returns:
            {"label": HxWxC float32 masked image array}
        """
        ...
