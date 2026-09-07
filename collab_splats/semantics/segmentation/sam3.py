"""
SAM3 text-prompted segmentation backend.

- SAM3Segmentation: text-prompted segmentation via facebook/sam3, a gated HF model
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from .base import BaseSegmentation

logger = logging.getLogger(__name__)

# SAM3 is an optional heavy dependency — imported lazily so the module loads without it
try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
except ImportError:
    build_sam3_image_model = None
    Sam3Processor = None


########################################################
########## SAM3 backend ################################
########################################################


@BaseSegmentation.register("sam3")
class SAM3Segmentation(BaseSegmentation):
    """
    SAM3 text-prompted segmentation backend.

    - facebook/sam3 is a gated model: request access, wait for approval, then `huggingface-cli login`.
    - Device placement is handled inside Sam3Processor — there is no device argument.

    Args:
        confidence_threshold: minimum score for returned masks.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        if build_sam3_image_model is None or Sam3Processor is None:
            raise ImportError(
                "sam3 is not installed. Request access at https://huggingface.co/facebook/sam3, "
                "wait for approval, run `huggingface-cli login`, then install from "
                "https://github.com/facebookresearch/sam3"
            )
        sam3_model = build_sam3_image_model()
        self._processor = Sam3Processor(sam3_model, confidence_threshold=confidence_threshold)

    def segment(self, image: Image.Image) -> tuple[torch.Tensor, Any]:
        """
        Auto-segment all objects without a text prompt.

        Args:
            image: the frame to segment.

        Returns:
            (masks, output) — masks (N, 1, H, W) float32; output is the raw processor dict.
        """
        state = self._processor.set_image(image)
        # SAM3 has no promptless auto-segment path; "object" is the generic catch-all
        output = self._processor.set_text_prompt(state=state, prompt="object")
        return output["masks"], output

    def segment_with_text(
        self, image: Image.Image, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Text-prompted segmentation.

        Args:
            image: PIL Image.
            prompt: text prompt, e.g. "red tractor".

        Returns:
            masks (N, 1, H, W) float32, boxes (N, 4) float32, scores (N,) float32.
        """
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        return output["masks"], output["boxes"], output["scores"]
