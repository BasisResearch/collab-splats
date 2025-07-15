import os
import torch
from torch.nn import functional as F
import numpy as np

from mobile_sam import SamAutomaticMaskGenerator
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
    def __init__(self, backend: str = 'mobile_sam', strategy: str = 'object', device: str = "cpu"):
        if backend == 'mobilesamv2':
            self.seg_model, self.object_model, self.predictor = load_mobile_sam(device=device)
        elif backend == 'ultralytics':
            pass
        elif backend == 'grounded_sam':
            pass
        else:
            raise ValueError(f"Backend {backend} not supported")

        self.backend = backend
        self.strategy = strategy

    def segment(self, image):
        if self.backend == 'mobilesamv2':
            if self.strategy == 'object':
                return object_segment_image(image, self.seg_model, self.object_model, self.predictor)
            elif self.strategy == 'auto':
                return auto_segment_image(image, self.seg_model)
            else:
                raise ValueError(f"Strategy {self.strategy} not supported")
        elif self.backend == 'ultralytics':
            pass
        elif self.backend == 'grounded_sam':
            pass
        # if self.method == 'object':
        #     return object_segment_image(image)
        # elif self.method == 'auto':
        #     return auto_segment_image(image)
        # else:

########################################################
########## MobileSAM Segmentation Utils ################
########################################################

def load_mobile_sam(mobilesam_encoder_name: str = 'mobilesamv2_efficientvit_l2', device: str = "cpu"):
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
    mobilesamv2, ObjAwareModel, predictor = load_torchhub_model("RogerQi/MobileSAMV2", mobilesam_encoder_name)
    mobilesamv2.to(device=device)
    mobilesamv2.eval()

    return mobilesamv2, ObjAwareModel, predictor

def auto_segment_image(image, mobile_sam, kwargs: dict = {}):
    """
    Automatically segments an image using MobileSAM. Provides
    an open alternative to the object-aware segmentation method
    """
    mask_generator = SamAutomaticMaskGenerator(
        model=mobile_sam,
        **kwargs
    )

    results = mask_generator.generate(image)

    # Convert masks to tensor
    masks = [torch.tensor(mask['segmentation']).to(torch.float32) for mask in results]
    masks = torch.stack(masks)
    
    return masks, results

def get_object_masks(image, obj_model, kwargs: dict = {}):
    """
    Grabs object bounding boxes from an object-aware model

    Suggested kwargs:
        - device: str = "cuda" if torch.cuda.is_available() else "cpu"
        - imgsz: int = 1024
        - conf: float = 0.25
        - iou: float = 0.5
        - verbose: bool = False
    """
    obj_results = obj_model(image, **kwargs)
    return obj_results

def object_segment_image(image, mobile_sam, obj_model, predictor, batch_size: int = 320):
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
    obj_results = get_object_masks(image, obj_model)
    
    if not obj_results or len(obj_results[0].boxes) == 0:
        return []
    

    # Setup predictor
    predictor.set_image(image)
    image_embedding = predictor.features
    prompt_embedding = mobile_sam.prompt_encoder.get_dense_pe()

    # Get boxes and confs
    boxes_xyxy = obj_results[0].boxes.xyxy.cpu().numpy()
    boxes_conf = obj_results[0].boxes.conf.cpu().numpy()

    # Convert boxes to original resolution
    transformed_boxes = predictor.transform.apply_boxes(boxes_xyxy, predictor.original_size)
    transformed_boxes = torch.from_numpy(transformed_boxes).to("cuda")

    results = []

    # Step 3: Loop through boxes in batches
    for boxes_batch, conf_batch in zip(batch_iterator(batch_size, transformed_boxes), batch_iterator(batch_size, boxes_conf)):
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

            results.append({
                "segmentation": masks[i],
                "area": area,
                "bbox": xywh,
                "predicted_iou": float(iou_preds[i]),
                "point_coords": [],  # since we're using box prompts
                "stability_score": float(confs[i]),  # reuse detector conf if nothing else
                "crop_box": [0, 0, width, height],  # no cropping here
            })

    # Convert masks to tensor
    masks = [torch.tensor(mask['segmentation']).to(torch.float32) for mask in results]
    masks = torch.stack(masks)
    
    return masks, results

########################################################
########## Feature Aggregation Utils ###################
########################################################

def aggregate_masked_features(features: torch.Tensor, masks: torch.Tensor, resolution: Tuple[int, int], final_resolution: Tuple[int, int]) -> torch.Tensor:
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
    features = F.interpolate(features.unsqueeze(0), 
                                size=resolution,
                                mode='bilinear',
                                align_corners=False)[0]

    # Interpolate masks to object size 
    masks = F.interpolate(masks.unsqueeze(1),
                        size=resolution,
                        mode='nearest').bool()[:,0]

    masks = masks.to(features.device)

    # Use einsum for feature aggregation
    # Shape: (n_masks, h, w) * (c, h, w) -> (c, h, w)
    weighted_features = torch.einsum('nhw,chw->chw', masks.float(), features)
    
    # Count number of masks per pixel
    mask_counts = masks.sum(0).float()
    
    # Normalize features by mask counts, avoiding division by zero
    aggregated_feat_map = weighted_features / (mask_counts + 1e-6).unsqueeze(0)
    
    # Resize to final dimensions
    aggregated_feat_map = F.interpolate(aggregated_feat_map.unsqueeze(0),
                                    size=final_resolution,
                                    mode='bilinear',
                                    align_corners=False)[0]
    
    return aggregated_feat_map