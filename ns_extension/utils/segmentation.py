import torch
from mobile_sam import SamAutomaticMaskGenerator
from torch.nn import functional as F
from typing import Tuple
from .features import batch_iterator, load_torchhub_model

class Segmentation:
    def __init__(self, backend: str = 'mobile_sam', strategy: str = 'object', device: str = "cpu"):
        if backend == 'mobilesamv2':
            self.mobilesamv2, self.object_model, self.predictor = load_mobile_sam(device=device)
        else:
            raise ValueError(f"Backend {backend} not supported")

        self.backend = backend
        self.strategy = strategy

    def segment(self, image):

        if self.backend == 'mobilesamv2':
            if self.strategy == 'object':
                return object_segment_image(image, self.mobilesamv2, self.object_model, self.predictor)
            elif self.strategy == 'auto':
                return auto_segment_image(image, self.mobilesamv2)
            else:
                raise ValueError(f"Strategy {self.strategy} not supported")

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

    masks = mask_generator.generate(image)

    # Convert masks to tensor
    masks = [torch.tensor(mask['segmentation']).to(torch.float32) for mask in masks]
    masks = torch.stack(masks)
    
    return masks

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
    # Grab the object masks
    obj_results = get_object_masks(image, obj_model)

    if not obj_results:
        return None

    # Set image for predictor
    predictor.set_image(image)

    # Apply bounding boxes to each object
    input_boxes = obj_results[0].boxes.xyxy
    input_boxes = input_boxes.cpu().numpy()
    input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
    input_boxes = torch.from_numpy(input_boxes).cuda()

    # Get features for each image
    image_embedding = predictor.features
    image_embedding = torch.repeat_interleave(image_embedding, batch_size, dim=0)
    prompt_embedding = mobile_sam.prompt_encoder.get_dense_pe()
    prompt_embedding = torch.repeat_interleave(prompt_embedding, batch_size, dim=0)

    sam_mask = []

    # Use bounding boxes to perform segmentation
    for (boxes,) in batch_iterator(batch_size, input_boxes):
        with torch.no_grad():
            image_embedding = image_embedding[0:boxes.shape[0],:,:,:]
            prompt_embedding = prompt_embedding[0:boxes.shape[0],:,:,:]
            sparse_embeddings, dense_embeddings = mobile_sam.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,)
            low_res_masks, _ = mobile_sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=prompt_embedding,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                simple_type=True,
            )
            low_res_masks = predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
            sam_mask_pre = (low_res_masks > -1) * 1.0
            sam_mask.append(sam_mask_pre.squeeze(1))

        del low_res_masks, sam_mask_pre

    sam_mask = torch.cat(sam_mask)
    return sam_mask

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