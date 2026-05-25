"""SAM3 text-prompted segmentation backend.

Provides:
  SAM3Segmentation — text-prompted segmentation via facebook/sam3 (gated HF model)
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from .base import BaseSegmentation

logger = logging.getLogger(__name__)

# SAM3 is an optional heavy dependency — imported lazily so the module loads without it
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    build_sam3_image_model = None
    Sam3Processor = None


########################################################
########## SAM3 backend ################################
########################################################


@BaseSegmentation.register("sam3")
class SAM3Segmentation(BaseSegmentation):
    """SAM3 text-prompted segmentation backend.

    Requires sam3 package (facebook/sam3 on HuggingFace Hub — gated model,
    requires accepting Meta's license).

    Args:
        confidence_threshold: Minimum score for returned masks (default 0.5).
        device: Torch device string (default 'cuda').
    """

    def __init__(self, confidence_threshold: float = 0.5, device: str = "cuda"):
        if build_sam3_image_model is None or Sam3Processor is None:
            raise ImportError(
                "sam3 is not installed. Install from https://github.com/facebookresearch/sam3 "
                "after accepting the Meta license at https://huggingface.co/facebook/sam3"
            )
        sam3_model = build_sam3_image_model()
        # Sam3Processor handles device placement internally; device param reserved for future API
        self._processor = Sam3Processor(
            sam3_model, confidence_threshold=confidence_threshold
        )

    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """Auto-segment all objects without a text prompt."""
        state = self._processor.set_image(image)
        # SAM3 has no promptless auto-segment path; "object" is the generic catch-all
        output = self._processor.set_text_prompt(state=state, prompt="object")
        return output["masks"], output

    def segment_with_text(
        self,
        image,
        prompt: str,
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text-prompted segmentation.

        Args:
            image:  PIL Image.
            prompt: Text prompt, e.g. "red tractor".
            confidence_threshold: Unused (set at construction time via Sam3Processor).

        Returns:
            masks  : (N, 1, H, W) float32
            boxes  : (N, 4) float32
            scores : (N,) float32
        """
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        return output["masks"], output["boxes"], output["scores"]
