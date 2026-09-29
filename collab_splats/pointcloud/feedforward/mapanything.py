"""
MapAnything feedforward backend: metric depth and camera poses in one forward pass.

- crop box and mask follow https://github.com/facebookresearch/map-anything @ c845b8f
- crop box: mapanything/utils/cropping.py:193, 231-240, 441-447
- upstream mask: mapanything/utils/inference.py:401-402
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)

from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud.feedforward.base import (
    BaseFeedforwardCreator,
    capture_qk,
    center_crop_coords,
)

logger = logging.getLogger(__name__)


########################################################################
# Constants
########################################################################

# Map our resize_mode names to the names MapAnything's image loader uses
_MA_RESIZE_MODE_MAP: dict[str, str] = {
    "fixed": "fixed_mapping",
    "longest_side": "longest_side",
    "square": "square",
}


########################################################################
# Creator
########################################################################


@BaseFeedforwardCreator.register("mapanything")
@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    """
    Pointcloud from MapAnything's per-image metric depth and camera poses.

    - one forward pass predicts depth and poses for all frames jointly
    - the multiview mask is ANDed with upstream's confidence mask, never swapped for it

    Attributes:
        model_name: HuggingFace id for MapAnything.from_pretrained.
        confidence_percentile: learned-confidence percentile cutoff (0-100), always applied.
        min_views: other views that must agree to keep a pixel; 0 is off.
        mv_rel_thresh: depth tolerance as a fraction of the expected depth.
        minibatch_size: views per inference step; lower it on OOM.
        resize_mode: "fixed" (aspect-ratio lookup table), "longest_side" or "square".
        resolution: lookup-table size for "fixed" (518 or 512); target px otherwise.
    """

    # MapAnything adds no extra tokens before the image tokens in its attention blocks
    _lc_token_offset: ClassVar[int] = 0

    # Loop-closure settings tuned for this model (see docs/parity.md)
    default_verify_match_ratio: ClassVar[float] = 1.46
    _lc_layer_index: ClassVar[int] = 4

    model_name: str = "facebook/map-anything"
    confidence_percentile: float = 35.0
    minibatch_size: int = 1
    resize_mode: str = "fixed"  # "fixed" (aspect-ratio lookup table), "longest_side", "square"
    resolution: int = 518  # lookup-table size for "fixed", target size in pixels otherwise

    # The full sequence of views, prepared for the model by _preprocess
    _processed_views: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Reject an unknown resize_mode.
        """
        if self.resize_mode not in _MA_RESIZE_MODE_MAP:
            raise ValueError(f"resize_mode must be one of {sorted(_MA_RESIZE_MODE_MAP)}, got {self.resize_mode!r}")

    def _load_model(self, device: str) -> Any:
        """
        Load the pretrained HuggingFace model in eval mode.
        """
        model = MapAnything.from_pretrained(self.model_name)
        model = model.to(device)
        model.eval()

        return model

    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        """
        Load frames as MapAnything view dicts and keep a model-ready copy on CPU.

        - returns (views, original_coords): view dicts and (N, 6) float32 crop boxes
        - crop boxes ignore EXIF orientation; frame-store PNGs carry none
        """
        # Load and resize the frames with MapAnything's own loader
        loader_paths = [str(p) for p in paths]
        upstream_mode = _MA_RESIZE_MODE_MAP[self.resize_mode]

        if self.resize_mode == "fixed":
            views = load_images(loader_paths, resize_mode=upstream_mode, resolution_set=self.resolution)
        else:
            views = load_images(loader_paths, resize_mode=upstream_mode, size=self.resolution)

        model_h: int = views[0]["img"].shape[-2]
        model_w: int = views[0]["img"].shape[-1]

        # Compute each frame's crop box the same way MapAnything's loader resizes and crops it
        rows = []

        for p in paths:
            w, h = Image.open(p).size
            s = max(model_w / w, model_h / h) + 1e-8  # same scale formula as MapAnything's loader
            resized_wh = (int(w * s), int(h * s))
            box = center_crop_coords((w, h), resized_wh, (model_w, model_h), (s, s))
            rows.append(box)

        original_coords = np.array(rows, dtype=np.float32)

        # Check the views and convert them to the model's input format, leaving them on the CPU
        validated = validate_input_views_for_inference(views)
        self._processed_views = preprocess_input_views_for_inference(validated)

        return views, original_coords

    def _forward(self, model: Any, views: Any) -> dict[str, np.ndarray]:
        """
        One MapAnything forward pass, stacked into the base raw dict.

        - full sequence: reuses the preprocessed views, upstream mask applied
        - loop-closure window: preprocessed here, unmasked
        """
        device = next(model.parameters()).device
        device_type = device.type

        # Use the cached full-sequence views, or prepare a loop-closure window of views here
        masked = views is self.views and self._processed_views is not None

        if masked:
            self._processed_views = _views_to(self._processed_views, device)
            forward_views = self._processed_views
            logger.debug("  → %d images, minibatch_size=%d", len(forward_views), self.minibatch_size)
        else:
            window = preprocess_input_views_for_inference(validate_input_views_for_inference(views))
            forward_views = _views_to(window, device)
            logger.debug("  → %d images (LC window), minibatch_size=%d", len(forward_views), self.minibatch_size)

        # Run the model in bfloat16 on the GPU, while postprocessing later stays in float32
        with torch.no_grad():
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
                preds = model.forward(
                    forward_views,
                    memory_efficient_inference=True,
                    minibatch_size=self.minibatch_size,
                )

        return self._stack_predictions(preds, forward_views, masked=masked)

    def _stack_predictions(self, preds: list[dict], views: list[dict], *, masked: bool) -> dict[str, np.ndarray]:
        """
        Postprocess predictions upstream's way, then stack them into the base raw dict.

        - masked: upstream's edge and confidence mask is stored under "mask"
        - extrinsic is (N, 3, 4) w2c; images are (N, 3, H, W) RGB in [0, 1]
        """
        # Convert the bfloat16 pointmaps to float32 so MapAnything's postprocessing can use them
        for pred in preds:
            pred["pts3d_cam"] = pred["pts3d_cam"].float()
            pred["pts3d"] = pred["pts3d"].float()

        # Let MapAnything's postprocessing compute camera poses and intrinsics
        if masked:
            processed = postprocess_model_outputs_for_inference(
                preds,
                views,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=True,
                confidence_percentile=self.confidence_percentile,
            )
        else:
            processed = postprocess_model_outputs_for_inference(preds, views, apply_mask=False)

        # Stack each view's outputs into arrays, turning camera-to-world poses into world-to-camera
        raw = {
            "extrinsic": np.stack([invert_poses(p["camera_poses"][0].cpu().float().numpy())[:3] for p in processed]),
            "intrinsics": np.stack([p["intrinsics"][0].cpu().float().numpy() for p in processed]),
            "depth": np.stack([p["depth_z"][0].cpu().float().numpy() for p in processed]),
            "depth_conf": np.stack([p["conf"][0].cpu().float().numpy() for p in processed]),
            "images": np.stack([p["img_no_norm"][0].cpu().float().numpy().transpose(2, 0, 1) for p in processed]),
        }

        # Keep MapAnything's valid-pixel mask, but only for the full sequence
        if masked:
            raw["mask"] = np.stack([p["mask"][0, ..., 0].cpu().numpy().astype(bool) for p in processed])

        return raw

    def extract_intermediate_features(self, frames: torch.Tensor, layer_index: int) -> dict[str, Any]:
        """
        Attention q/k from one cross-frame block, plus poses and pointmaps, for a frame pair.

        - one forward pass gives both the attention and the predictions
        - poses are w2c; frame 0 sits at identity

        Args:
            frames: (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: info_sharing self-attention block to tap; -1 = last.

        Returns:
            {"q", "k": (B, heads, N_tokens, head_dim), "poses": (2, 4, 4) float32 w2c,
            "world_points": (2, H, W, 3), "conf": (2, H, W)}.
        """
        # Record the attention queries and keys of the chosen block during one forward pass
        attn = self.model.info_sharing.self_attention_blocks[layer_index].attn

        with capture_qk(attn.qkv, attn.num_heads) as captured, torch.no_grad():
            # Wrap each frame as a MapAnything view dict using the loader's dinov2 normalization
            raw_views = [{"img": f.unsqueeze(0), "data_norm_type": ["dinov2"]} for f in frames.cpu()]

            # Views are built from CPU frames, so move them to the model device
            views = _views_to(preprocess_input_views_for_inference(raw_views), next(self.model.parameters()).device)
            preds = self.model.forward(views, memory_efficient_inference=False, minibatch_size=1)

        # Convert pointmaps to float32 and postprocess them, the same as _stack_predictions
        with torch.no_grad():
            for pred in preds:
                pred["pts3d_cam"] = pred["pts3d_cam"].float()
                pred["pts3d"] = pred["pts3d"].float()

            processed = postprocess_model_outputs_for_inference(preds, views, apply_mask=False)

        # Turn camera-to-world poses into world-to-camera
        captured["poses"] = np.stack([invert_poses(p["camera_poses"][0].cpu().float().numpy()) for p in processed])

        # Pointmaps and confidence for loop-closure scale estimation
        captured["world_points"] = np.stack([p["pts3d"][0].cpu().float().numpy() for p in processed])  # (2, H, W, 3)
        captured["conf"] = np.stack([p["conf"][0].cpu().float().numpy() for p in processed])  # (2, H, W)

        return captured


########################################################################
# Helpers
########################################################################


def _views_to(views: list[dict[str, Any]], device: torch.device | str) -> list[dict[str, Any]]:
    """
    Move every tensor in a list of MapAnything view dicts to `device`.

    - returns new view dicts; non-tensor values pass through
    """
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in view.items()} for view in views]
