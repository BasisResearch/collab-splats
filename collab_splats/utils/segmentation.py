import torch
from torch.nn import functional as F
import numpy as np
import math
from mobile_sam import SamAutomaticMaskGenerator
import logging
from typing import Optional

logging.getLogger("ultralytics").setLevel(logging.WARNING)

from typing import Tuple
from .features import batch_iterator, load_torchhub_model

##TLB --> need to figure out how to use ultralytics here
# from ultralytics import SAM

# TORCH_HOME = os.getenv('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

# ULTRALYTICS_SAM_MODELS = {
#     'mobile_sam': os.path.join(TORCH_HOME, 'ultralytics', 'mobile_sam.pt'),
#     'sam2.1_t': os.path.join(TORCH_HOME, 'ultralytics', 'sam2.1_t.pt'),
#     'sam2.1_s': os.path.join(TORCH_HOME, 'ultralytics', 'sam2.1_s.pt'),
#     'sam2.1_b': os.path.join(TORCH_HOME, 'ultralytics', 'sam2.1_b.pt'),
# }


class Segmentation:
    def __init__(
        self, backend: str = "mobile_sam", strategy: str = "object", device: str = "cpu"
    ):
        if backend == "mobilesamv2":
            self.seg_model, self.object_model, self.predictor = load_mobile_sam(
                device=device
            )
        elif backend == "ultralytics":
            pass
        elif backend == "grounded_sam":
            pass
        else:
            raise ValueError(f"Backend {backend} not supported")

        self.backend = backend
        self.strategy = strategy
        
    def segment(self, image, **kwargs):
        if self.backend == "mobilesamv2":
            if self.strategy == "object":
                return object_segment_image(
                    image, self.seg_model, self.object_model, self.predictor, **kwargs
                )
            elif self.strategy == "auto":
                return auto_segment_image(image, self.seg_model, **kwargs)
            else:
                raise ValueError(f"Strategy {self.strategy} not supported")
        elif self.backend == "ultralytics":
            pass
        elif self.backend == "grounded_sam":
            pass
        # if self.method == 'object':
        #     return object_segment_image(image)
        # elif self.method == 'auto':
        #     return auto_segment_image(image)
        # else:


########################################################
########## MobileSAM Segmentation Utils ################
########################################################


def load_mobile_sam(
    mobilesam_encoder_name: str = "mobilesamv2_efficientvit_l2", device: str = "cpu"
):
    """
    Loading models from feature-splatting repo

    This contains an efficent way to load MobileSAM and a YOLOv8 model
    that can be used to perform object detection.

    Inputs:
        - mobilesam_encoder_name: str = 'mobilesamv2_efficientvit_l2'

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


def auto_segment_image(image, mobile_sam, **kwargs):
    """
    Automatically segments an image using MobileSAM. Provides
    an open alternative to the object-aware segmentation method
    """
    mask_generator = SamAutomaticMaskGenerator(model=mobile_sam, **kwargs)

    results = mask_generator.generate(image)

    if len(results) == 0:
        return None

    # Convert masks to tensor
    masks = [torch.tensor(mask["segmentation"]).to(torch.float32) for mask in results]
    masks = torch.stack(masks)

    return masks, results


def get_object_masks(image, obj_model, **kwargs):
    """
    Grabs object bounding boxes from an object-aware model

    Suggested kwargs:
        - device: str = "cuda" if torch.cuda.is_available() else "cpu"
        - imgsz: int = 1024
        - conf: float = 0.25
        - iou: float = 0.5
        - verbose: bool = False
    """
    # Set default verbose to False to reduce output
    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    
    obj_results = obj_model(image, **kwargs)
    return obj_results


def object_segment_image(
    image, mobile_sam, obj_model, predictor, batch_size: int = 320, **kwargs
):
    """
    Uses object-bounding boxes to perform segmentation over an image

    Inputs:
        - image: np.ndarray
        - mobile_sam: MobileSAM model
        - obj_results: Object-aware model results
        - predictor: SAMPredictor object

    Outputs:
        - sam_mask: SAM mask
    """

    height, width = image.shape[:2]

    # Get object detections
    obj_results = get_object_masks(image, obj_model, **kwargs)

    if not obj_results or len(obj_results[0].boxes) == 0:
        return None

    # Setup predictor
    predictor.set_image(image)
    image_embedding = predictor.features
    prompt_embedding = mobile_sam.prompt_encoder.get_dense_pe()

    # Get boxes and confs
    boxes_xyxy = obj_results[0].boxes.xyxy.cpu().numpy()
    boxes_conf = obj_results[0].boxes.conf.cpu().numpy()

    # Convert boxes to original resolution
    transformed_boxes = predictor.transform.apply_boxes(
        boxes_xyxy, predictor.original_size
    )
    transformed_boxes = torch.from_numpy(transformed_boxes).to("cuda")

    results = []

    # Step 3: Loop through boxes in batches
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

            # Bounding box from mask (not just original box)
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
                    "point_coords": [],  # since we're using box prompts
                    "stability_score": float(
                        confs[i]
                    ),  # reuse detector conf if nothing else
                    "crop_box": [0, 0, width, height],  # no cropping here
                }
            )

    if len(results) == 0:
        return None

    # Convert masks to tensor
    masks = [torch.tensor(mask["segmentation"]).to(torch.float32) for mask in results]
    masks = torch.stack(masks)

    return masks, results


########################################################
############### Aggregation Utils ######################
########################################################


def create_patch_mask(image, num_patches: int = 32):
    """
    Provided an image of given dimensions, create an array of patches.
    """
    # Get image dimensions
    H, W = image.shape[:2]

    # Get patch dimensions
    patch_width = math.ceil(W / num_patches)
    patch_height = math.ceil(H / num_patches)

    # Create flattened coordinates
    total_pixels = H * W
    y_coords = torch.arange(H).unsqueeze(1).expand(-1, W).flatten()
    x_coords = torch.arange(W).unsqueeze(0).expand(H, -1).flatten()

    # Calculate patch indices for all pixels at once
    patch_y_indices = torch.clamp(y_coords // patch_height, 0, num_patches - 1)
    patch_x_indices = torch.clamp(x_coords // patch_width, 0, num_patches - 1)

    # Create sparse representation
    flatten_patch_mask = torch.zeros(
        (num_patches, num_patches, total_pixels), dtype=torch.bool
    )

    # Use indexing to set values
    pixel_indices = torch.arange(total_pixels)
    flatten_patch_mask[patch_y_indices, patch_x_indices, pixel_indices] = True

    return flatten_patch_mask


def create_composite_mask(results, confidence_threshold=0.85):
    """
    Creates a composite mask from the results of the segmentation model.

    Inputs:
        results: list of dicts, each containing a mask and a confidence score
        confidence_threshold: float, the minimum confidence score for a mask to be included in the composite mask

    Outputs:
        composite_mask: numpy array, the composite mask
    """

    selected_masks = []
    for mask in results:
        # Errors seems to happen above 1.0
        if mask["predicted_iou"] < confidence_threshold or mask["predicted_iou"] > 1.0:
            continue

        selected_masks.append((mask["segmentation"], mask["predicted_iou"]))

    # Store the masks and confidences
    masks, confs = zip(*selected_masks)

    # Create empty image to store mask ids
    H, W = masks[0].shape[:2]
    mask_id = np.zeros((H, W), dtype=np.uint8)

    sorted_idxs = np.argsort(confs)
    for i, idx in enumerate(sorted_idxs, start=1):
        current_mask = masks[idx - 1]
        mask_id[current_mask == 1] = i

    # Find mask indices after having calculated overlap based on ranked confidence
    mask_indices = np.unique(mask_id)
    mask_indices = np.setdiff1d(mask_indices, [0])  # remove 0 item

    composite_mask = np.zeros((H, W), dtype=np.uint8)
    current_label = 1  # ensures consecutive numbering

    # Rewriting to ensure consecutive numbering (e.g., not corresponding to the indices of the masks)
    for idx in mask_indices:
        mask = mask_id == idx

        if mask.sum() > 0 and (mask.sum() / masks[idx - 1].sum()) > 0.1:
            composite_mask[mask] = current_label
            current_label += 1

    return composite_mask


def mask_id_to_binary_mask(composite_mask: np.ndarray) -> np.ndarray:
    """
    Convert an image with integer mask IDs to a binary mask array.

    Args:
        mask_id (np.ndarray): An (H, W) array where each unique positive integer
                            represents a separate object mask.

    Returns:
        np.ndarray: A (N, H, W) boolean array where N is the number of masks and each
                    slice contains a binary mask.
    """
    unique_ids = np.unique(composite_mask)

    # This could be redundant since the background is already removed in making the composite mask
    unique_ids = unique_ids[unique_ids > 0]  # Ignore background (assumed to be 0)

    binary_masks = composite_mask[None, ...] == unique_ids[:, None, None]
    return binary_masks


def convert_matched_mask(labels: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    """
    Convert a mask with sequential IDs to use the matched label IDs.

    Args:
        labels: Tensor of shape (N,) containing the matched label for each mask ID
        masks: Array of shape (H,W) containing sequential mask IDs from 1 to N

    Returns:
        Array of shape (H,W) with mask IDs replaced by their matched labels
    """
    # Validate input - number of labels should match max mask ID
    assert labels.shape[0] == np.max(masks), (
        "Number of labels must match number of unique masks"
    )

    # Create output array with uint16 to handle potential large label values
    matched_mask = np.zeros(masks.shape, dtype=np.uint16)

    # Replace each mask ID with its matched label
    # Add 1 since mask IDs start at 1 but label indices start at 0
    for label_idx in range(labels.shape[0]):
        mask_id = label_idx + 1
        matched_label = labels[label_idx].item() + 1
        matched_mask[masks == mask_id] = matched_label

    # Keep as uint16 since we expect large label values
    return matched_mask


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
        obj_resolution (int): Resolution for intermediate feature map
        final_resolution (int): Resolution for final output

    Returns:
        torch.Tensor: Aggregated feature map (C,H,W)
    """

    # Interpolate CLIP features to object size
    features = F.interpolate(
        features.unsqueeze(0), size=resolution, mode="bilinear", align_corners=False
    )[0]

    # Interpolate masks to object size
    masks = F.interpolate(masks.unsqueeze(1), size=resolution, mode="nearest").bool()[
        :, 0
    ]

    masks = masks.to(features.device)

    # Use einsum for feature aggregation
    # Shape: (n_masks, h, w) * (c, h, w) -> (c, h, w)
    weighted_features = torch.einsum("nhw,chw->chw", masks.float(), features)

    # Count number of masks per pixel
    mask_counts = masks.sum(0).float()

    # Normalize features by mask counts, avoiding division by zero
    aggregated_feat_map = weighted_features / (mask_counts + 1e-6).unsqueeze(0)

    # Resize to final dimensions
    aggregated_feat_map = F.interpolate(
        aggregated_feat_map.unsqueeze(0),
        size=final_resolution,
        mode="bilinear",
        align_corners=False,
    )[0]

    return aggregated_feat_map

########################################################
############### Visualization ##########################
########################################################

def get_n_different_colors(n: int, seed: int = 42) -> np.ndarray:
    return np.random.randint(1, 256, (n, 3), dtype=np.uint8)

def visualize_mask(mask: np.ndarray, colors: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualize a mask where each unique mask ID gets a unique color.
    Background (0) stays black.
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # ignore background

    if colors is None:
        colors = get_n_different_colors(len(unique_ids))

    for i, mask_id in enumerate(unique_ids):
        color_mask[mask == mask_id] = colors[i]

    return color_mask