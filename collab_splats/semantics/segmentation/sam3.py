"""
SAM3 text-prompted segmentation backend ("sam3"); facebook/sam3 is a gated model.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from .base import BaseSegmentation

logger = logging.getLogger(__name__)


########################################################
########## SAM3 backend ################################
########################################################


@BaseSegmentation.register("sam3")
class SAM3Segmentation(BaseSegmentation):
    """
    SAM3 text-prompted segmentation backend.

    - gated model: request access to facebook/sam3, then `huggingface-cli login`
    - Sam3Processor places the model on a device itself; no device argument

    Args:
        confidence_threshold: minimum score for returned masks.

    Raises:
        ImportError: when sam3 is not installed; other import failures propagate unchanged.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        # Imported here: optional, gated dependency
        try:
            from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
            from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415
        except ModuleNotFoundError as e:
            # Only sam3 itself missing gets the install hint; a broken transitive dep re-raises
            if (e.name or "").split(".")[0] != "sam3":
                raise
            raise ImportError(
                "sam3 is not installed. Request access at https://huggingface.co/facebook/sam3, "
                "wait for approval, run `huggingface-cli login`, then install from "
                "https://github.com/facebookresearch/sam3"
            ) from e
        self._processor = Sam3Processor(
            build_sam3_image_model(), confidence_threshold=confidence_threshold
        )

    def segment(self, image: Image.Image) -> tuple[torch.Tensor, Any]:
        """
        Every object in the frame, via the generic prompt "object".

        Args:
            image: the frame to segment.

        Returns:
            (masks, output) — masks (N, 1, H, W) bool; output is the raw processor dict.
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
            masks (N, 1, H, W) bool, boxes (N, 4) float32, scores (N,) float32.
        """
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        return output["masks"], output["boxes"], output["scores"]
