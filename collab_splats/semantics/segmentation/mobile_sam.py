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

from collab_splats.semantics.segmentation.base import BaseSegmentation
from collab_splats.utils.torch_utils import batch_iterator, load_torchhub_model

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
    Stack SAM result masks as bool, (0, H, W) when there are none.

    Args:
        results: SAM dicts with a "segmentation" (H, W) bool array.
        height: frame rows, for the empty stack.
        width: frame columns, for the empty stack.

    Returns:
        (N, H, W) bool.
    """
    if not results:
        return torch.zeros((0, height, width), dtype=torch.bool)

    # One numpy stack, then a zero-copy tensor view
    stacked = np.stack([m["segmentation"] for m in results])
    return torch.from_numpy(stacked)


def _mask_stats(masks: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel count and tight (x_min, y_min, x_max, y_max) box of each (N, H, W) bool mask.

    - computed on the masks' device, one host copy each
    - an empty mask has area 0 and a meaningless box
    """
    areas = masks.sum(dim=(1, 2))

    # First and last occupied row and column per mask
    rows = masks.any(dim=2)
    rows = rows.int()
    cols = masks.any(dim=1)
    cols = cols.int()
    y_min = rows.argmax(dim=1)
    x_min = cols.argmax(dim=1)
    y_max = rows.shape[1] - 1 - rows.flip(1).argmax(dim=1)
    x_max = cols.shape[1] - 1 - cols.flip(1).argmax(dim=1)

    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    return areas.cpu().numpy(), boxes.cpu().numpy()


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
            raise ValueError(
                f"Strategy '{strategy}' not supported. Available: ['object', 'auto']"
            )
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
            (masks, results): masks (N, H, W) bool, N = 0 when nothing is detected;
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
                    low_res_masks,
                    self.predictor.input_size,
                    self.predictor.original_size,
                )
                masks = masks > self.seg_model.mask_threshold
                masks = masks.squeeze(1)
                iou_preds = iou_preds.squeeze(1).cpu().numpy()

            # Areas and tight xyxy boxes on the device, then one copy each to the host
            areas, xyxy = _mask_stats(masks)
            masks = masks.cpu().numpy()

            for i in range(B):
                area = int(areas[i])

                if area == 0:
                    continue

                x0, y0, x1, y1 = xyxy[i]
                xywh = [x0, y0, x1 - x0, y1 - y0]

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
