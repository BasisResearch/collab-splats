"""
Segmentation utilities using MobileSAM and related models.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from mobile_sam import SamAutomaticMaskGenerator

from collab_splats.semantics.utils import batch_iterator, load_torchhub_model
from collab_splats.utils.torch_utils import RegistryMixin

logger = logging.getLogger(__name__)

# SAM3 is an optional heavy dependency — imported lazily so the module loads without it
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    build_sam3_image_model = None
    Sam3Processor = None


########################################################
########## MobileSAM Segmentation Utils ################
########################################################


def load_mobile_sam(
    mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2", device: str = "cpu"
):
    """
    Loading models from feature-splatting repo.

    Returns:
        - mobilesamv2: MobileSAM model
        - ObjAwareModel: YOLOv8 model
        - predictor: SAMPredictor object
    """
    mobilesamv2, ObjAwareModel, predictor = load_torchhub_model(
        "RogerQi/MobileSAMV2", mobilesam_encoder_name
    )
    mobilesamv2.to(device=device)
    mobilesamv2.eval()

    return mobilesamv2, ObjAwareModel, predictor


def auto_segment_image(image, mobile_sam, kwargs: dict = {}):
    """
    Automatically segments an image using MobileSAM.
    """
    mask_generator = SamAutomaticMaskGenerator(model=mobile_sam, **kwargs)
    results = mask_generator.generate(image)

    if len(results) == 0:
        return None

    masks = [torch.tensor(mask["segmentation"]).to(torch.float32) for mask in results]
    masks = torch.stack(masks)

    return masks, results


def get_object_masks(image, obj_model, kwargs: dict = {}):
    """
    Grabs object bounding boxes from an object-aware model.

    Suggested kwargs:
        - device: str = "cuda" if torch.cuda.is_available() else "cpu"
        - imgsz: int = 1024
        - conf: float = 0.25
        - iou: float = 0.5
        - verbose: bool = False
    """
    obj_results = obj_model(image, **kwargs)
    return obj_results


def object_segment_image(
    image, mobile_sam, obj_model, predictor, batch_size: int = 320
):
    """
    Uses object bounding boxes to perform segmentation over an image.

    Inputs:
        - image: np.ndarray (H, W, 3)
        - mobile_sam: MobileSAM model
        - obj_model: YOLOv8 object detector (bounding boxes computed internally)
        - predictor: SAMPredictor object
        - batch_size: boxes per decode batch (default: 320)

    Outputs:
        - sam_mask: SAM mask
    """
    height, width = image.shape[:2]

    obj_results = get_object_masks(image, obj_model)

    if not obj_results or len(obj_results[0].boxes) == 0:
        return None

    predictor.set_image(image)
    image_embedding = predictor.features
    prompt_embedding = mobile_sam.prompt_encoder.get_dense_pe()

    boxes_xyxy = obj_results[0].boxes.xyxy.cpu().numpy()
    boxes_conf = obj_results[0].boxes.conf.cpu().numpy()

    transformed_boxes = predictor.transform.apply_boxes(boxes_xyxy, predictor.original_size)
    # Fix: derive device from model rather than hardcoding "cuda"
    model_device = next(iter(mobile_sam.parameters())).device
    transformed_boxes = torch.from_numpy(transformed_boxes).to(model_device)

    results = []

    for boxes_batch, conf_batch in zip(
        batch_iterator(batch_size, transformed_boxes),
        batch_iterator(batch_size, boxes_conf),
    ):
        boxes = boxes_batch[0]
        confs = conf_batch[0]
        B = boxes.shape[0]

        with torch.no_grad():
            _image_embedding = image_embedding.repeat(B, 1, 1, 1)
            _prompt_embedding = prompt_embedding.repeat(B, 1, 1, 1)

            sparse_embeddings, dense_embeddings = mobile_sam.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )

            low_res_masks, iou_preds = mobile_sam.mask_decoder(
                image_embeddings=_image_embedding,
                image_pe=_prompt_embedding,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                simple_type=True,
            )

            masks = predictor.model.postprocess_masks(
                low_res_masks, predictor.input_size, predictor.original_size
            )
            masks = masks > mobile_sam.mask_threshold
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

    if len(results) == 0:
        return None

    masks = [torch.tensor(mask["segmentation"]).to(torch.float32) for mask in results]
    masks = torch.stack(masks)

    return masks, results


########################################################
########## Registry and abstract base ##################
########################################################


class BaseSegmentation(RegistryMixin, ABC):
    """Abstract base for segmentation backends with name-based registry.

    Register subclasses via ``@BaseSegmentation.register("name")``.
    Retrieve with ``BaseSegmentation.get("name")``.
    """

    _registry: dict[str, type["BaseSegmentation"]] = {}

    @abstractmethod
    def segment(self, image) -> tuple[torch.Tensor, Any]:
        """Class-agnostic segmentation. Returns (masks, metadata)."""

    def segment_with_text(
        self,
        image,
        prompt: str,
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text-prompted segmentation → (masks, boxes, scores).

        Raises NotImplementedError for backends that don't support text prompts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text-prompted segmentation. "
            "Use backend='sam3'."
        )


########################################################
########## MobileSAMv2 backend #########################
########################################################


@BaseSegmentation.register("mobilesamv2")
class MobileSAMSegmentation(BaseSegmentation):
    """MobileSAMv2 class-agnostic segmentation (object or auto strategy)."""

    def __init__(
        self,
        strategy: str = "object",
        device: str = "cpu",
        mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2",
    ):
        self.seg_model, self.object_model, self.predictor = load_mobile_sam(
            mobilesam_encoder_name, device
        )
        self.strategy = strategy

    def segment(self, image) -> tuple[torch.Tensor, Any]:
        if self.strategy == "object":
            return object_segment_image(
                image, self.seg_model, self.object_model, self.predictor
            )
        elif self.strategy == "auto":
            return auto_segment_image(image, self.seg_model)
        else:
            raise ValueError(
                f"Strategy '{self.strategy}' not supported. Available: ['object', 'auto']"
            )


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


########################################################
############### Aggregation Utils ######################
########################################################


def create_patch_mask(image, num_patches: int = 32):
    """
    Provided an image of given dimensions, create an array of patches.
    """
    H, W = image.shape[:2]

    patch_width = math.ceil(W / num_patches)
    patch_height = math.ceil(H / num_patches)

    total_pixels = H * W
    y_coords = torch.arange(H).unsqueeze(1).expand(-1, W).flatten()
    x_coords = torch.arange(W).unsqueeze(0).expand(H, -1).flatten()

    patch_y_indices = torch.clamp(y_coords // patch_height, 0, num_patches - 1)
    patch_x_indices = torch.clamp(x_coords // patch_width, 0, num_patches - 1)

    flatten_patch_mask = torch.zeros(
        (num_patches, num_patches, total_pixels), dtype=torch.bool
    )

    pixel_indices = torch.arange(total_pixels)
    flatten_patch_mask[patch_y_indices, patch_x_indices, pixel_indices] = True

    return flatten_patch_mask


def create_composite_mask(results, confidence_threshold=0.85):
    """
    Creates a composite mask from the results of the segmentation model.

    Inputs:
        results: list of dicts, each containing a mask and a confidence score
        confidence_threshold: float, the minimum confidence score for a mask

    Outputs:
        composite_mask: numpy array
    """
    selected_masks = []
    for mask in results:
        if mask["predicted_iou"] < confidence_threshold or mask["predicted_iou"] > 1.0:
            continue
        selected_masks.append((mask["segmentation"], mask["predicted_iou"]))

    if not selected_masks:
        return np.zeros_like(results[0]["segmentation"], dtype=np.uint8) if results else np.zeros((0, 0), dtype=np.uint8)
    masks, confs = zip(*selected_masks)

    H, W = masks[0].shape[:2]
    mask_id = np.zeros((H, W), dtype=np.uint8)

    sorted_idxs = np.argsort(confs)
    for i, idx in enumerate(sorted_idxs, start=1):
        current_mask = masks[idx - 1]
        mask_id[current_mask == 1] = i

    mask_indices = np.unique(mask_id)
    mask_indices = np.setdiff1d(mask_indices, [0])

    composite_mask = np.zeros((H, W), dtype=np.uint8)

    for i, idx in enumerate(mask_indices, start=1):
        mask = mask_id == idx
        logger.debug("Mask %d has %d pixels", i, mask.sum())
        if mask.sum() > 0 and (mask.sum() / masks[idx - 1].sum()) > 0.1:
            composite_mask[mask] = i

    return composite_mask


def mask_id_to_binary_mask(composite_mask: np.ndarray) -> np.ndarray:
    """
    Convert an image with integer mask IDs to a binary mask array.

    Args:
        composite_mask (np.ndarray): An (H, W) array where each unique positive integer
                            represents a separate object mask.

    Returns:
        np.ndarray: A (N, H, W) boolean array where N is the number of masks.
    """
    unique_ids = np.unique(composite_mask)
    unique_ids = unique_ids[unique_ids > 0]
    binary_masks = composite_mask[None, ...] == unique_ids[:, None, None]
    return binary_masks


def convert_matched_mask(labels: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    """
    Convert a mask with sequential IDs to use the matched label IDs.

    Args:
        labels: Tensor of shape (N,) containing the matched label for each mask ID
        masks: Array of shape (H,W) containing sequential mask IDs from 1 to N

    Returns:
        Array of shape (H,W) dtype=uint16 with mask IDs replaced by their matched labels
    """
    assert labels.shape[0] == np.max(masks), (
        "Number of labels must match number of unique masks"
    )

    matched_mask = np.zeros(masks.shape, dtype=np.uint16)

    for label_idx in range(labels.shape[0]):
        mask_id = label_idx + 1
        matched_label = labels[label_idx].item() + 1
        matched_mask[masks == mask_id] = matched_label

    return matched_mask  # uint16 — preserves IDs > 255


def aggregate_masked_features(
    features: torch.Tensor,
    masks: torch.Tensor,
    resolution: Tuple[int, int],
    final_resolution: Tuple[int, int],
) -> torch.Tensor:
    """
    Aggregate features based on SAM segmentation masks.

    Args:
        features (torch.Tensor): Features for the whole image (C,H,W)
        masks (torch.Tensor): Segmentation masks from SAM (N,H,W)
        resolution (Tuple[int,int]): Resolution for intermediate feature map
        final_resolution (Tuple[int,int]): Resolution for final output

    Returns:
        torch.Tensor: Aggregated feature map (C,H,W)
    """
    features = F.interpolate(
        features.unsqueeze(0), size=resolution, mode="bilinear", align_corners=False
    )[0]

    masks = F.interpolate(masks.unsqueeze(1), size=resolution, mode="nearest").bool()[:, 0]
    masks = masks.to(features.device)

    weighted_features = torch.einsum("nhw,chw->chw", masks.float(), features)
    mask_counts = masks.sum(0).float()
    aggregated_feat_map = weighted_features / (mask_counts + 1e-6).unsqueeze(0)

    aggregated_feat_map = F.interpolate(
        aggregated_feat_map.unsqueeze(0),
        size=final_resolution,
        mode="bilinear",
        align_corners=False,
    )[0]

    return aggregated_feat_map
