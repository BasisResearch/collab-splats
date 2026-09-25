"""
MobileSAMv2 segmentation backend ("mobilesamv2").

- "object": YOLOv8 boxes prompt SAM
- "auto": SAM's automatic mask generator
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator

from collab_splats.utils.torch_utils import batch_iterator, load_torchhub_model

from .base import BaseSegmentation

logger = logging.getLogger(__name__)


########################################################
########## Model loading ###############################
########################################################


def _load_mobile_sam(
    mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2", device: str = "cpu"
) -> tuple[Any, Any, Any]:
    """
    Load the MobileSAMV2 model trio from torchhub.

    Args:
        mobilesam_encoder_name: encoder variant published by RogerQi/MobileSAMV2.
        device: torch device string for the SAM model.

    Returns:
        (mobilesamv2, ObjAwareModel, predictor) — SAM model, YOLOv8 detector, SAMPredictor.
    """
    mobilesamv2, ObjAwareModel, predictor = load_torchhub_model(
        "RogerQi/MobileSAMV2", mobilesam_encoder_name
    )
    mobilesamv2.to(device=device)
    mobilesamv2.eval()

    return mobilesamv2, ObjAwareModel, predictor


def _stack_masks(results: list[dict], height: int, width: int) -> torch.Tensor:
    """
    Stack SAM result masks as float32, (0, H, W) when there are none.

    Args:
        results: SAM dicts with a "segmentation" (H, W) array.
        height: frame rows, for the empty stack.
        width: frame columns, for the empty stack.

    Returns:
        (N, H, W) float32.
    """
    if not results:
        return torch.zeros((0, height, width), dtype=torch.float32)
    return torch.stack([torch.tensor(m["segmentation"]).to(torch.float32) for m in results])


########################################################
########## MobileSAMv2 backend #########################
########################################################


@BaseSegmentation.register("mobilesamv2")
class MobileSAMSegmentation(BaseSegmentation):
    """
    MobileSAMv2 class-agnostic segmentation.

    Args:
        strategy: "object" prompts SAM with YOLOv8 boxes; "auto" runs SAM's mask generator.
        device: torch device string.
        mobilesam_encoder_name: encoder variant to load from torchhub.
        box_batch_size: boxes per SAM decoder call ("object" only).

    Raises:
        ValueError: if strategy is neither "object" nor "auto".
    """

    def __init__(
        self,
        strategy: str = "object",
        device: str = "cpu",
        mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2",
        box_batch_size: int = 320,
    ):
        if strategy not in ("object", "auto"):
            raise ValueError(f"Strategy '{strategy}' not supported. Available: ['object', 'auto']")
        self.seg_model, self.object_model, self.predictor = _load_mobile_sam(
            mobilesam_encoder_name, device
        )
        self.strategy = strategy
        self.box_batch_size = box_batch_size

    def segment(self, image: np.ndarray) -> tuple[torch.Tensor, list[dict]]:
        """
        Segment one frame with the configured strategy.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            (masks, results): masks (N, H, W) float32, N = 0 when nothing is detected;
            results are the raw SAM dicts, one per mask.
        """
        if self.strategy == "object":
            return self._segment_object(image)
        return self._segment_auto(image)

    def _segment_auto(self, image: np.ndarray) -> tuple[torch.Tensor, list[dict]]:
        """
        SAM's automatic mask generator, no prompts.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            (masks, results), N = 0 when the generator finds nothing.
        """
        mask_generator = SamAutomaticMaskGenerator(model=self.seg_model)
        results = mask_generator.generate(image)

        return _stack_masks(results, *image.shape[:2]), results

    def _segment_object(self, image: np.ndarray) -> tuple[torch.Tensor, list[dict]]:
        """
        YOLOv8 boxes prompt SAM once per detected object.

        Args:
            image: (H, W, 3) uint8 array.

        Returns:
            (masks, results), N = 0 when the detector finds no objects.
        """
        height, width = image.shape[:2]

        obj_results = self.object_model(image)

        if not obj_results or len(obj_results[0].boxes) == 0:
            return _stack_masks([], height, width), []

        self.predictor.set_image(image)
        image_embedding = self.predictor.features
        prompt_embedding = self.seg_model.prompt_encoder.get_dense_pe()

        boxes_xyxy = obj_results[0].boxes.xyxy.cpu().numpy()
        boxes_conf = obj_results[0].boxes.conf.cpu().numpy()

        transformed_boxes = self.predictor.transform.apply_boxes(
            boxes_xyxy, self.predictor.original_size
        )
        model_device = next(iter(self.seg_model.parameters())).device
        transformed_boxes = torch.from_numpy(transformed_boxes).to(model_device)

        results = []

        for boxes_batch, conf_batch in zip(
            batch_iterator(self.box_batch_size, transformed_boxes),
            batch_iterator(self.box_batch_size, boxes_conf),
        ):
            boxes = boxes_batch[0]
            confs = conf_batch[0]
            B = boxes.shape[0]

            with torch.no_grad():
                _image_embedding = image_embedding.repeat(B, 1, 1, 1)
                _prompt_embedding = prompt_embedding.repeat(B, 1, 1, 1)

                sparse_embeddings, dense_embeddings = self.seg_model.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,
                )

                low_res_masks, iou_preds = self.seg_model.mask_decoder(
                    image_embeddings=_image_embedding,
                    image_pe=_prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )

                masks = self.predictor.model.postprocess_masks(
                    low_res_masks, self.predictor.input_size, self.predictor.original_size
                )
                masks = masks > self.seg_model.mask_threshold
                masks = masks.squeeze(1).cpu().numpy()
                iou_preds = iou_preds.squeeze(1).cpu().numpy()

            for i in range(B):
                _mask = masks[i].astype(np.uint8)
                area = int(_mask.sum())
                if area == 0:
                    continue

                y_indices, x_indices = np.where(_mask)
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                xywh = [x_min, y_min, x_max - x_min, y_max - y_min]

                results.append(
                    {
                        "segmentation": masks[i],
                        "area": area,
                        "bbox": xywh,
                        "predicted_iou": float(iou_preds[i]),
                        "point_coords": [],
                        "stability_score": float(confs[i]),
                        "crop_box": [0, 0, width, height],
                    }
                )

        return _stack_masks(results, height, width), results
