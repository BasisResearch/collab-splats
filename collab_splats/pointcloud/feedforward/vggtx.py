"""
VGGT-X feedforward backend: crop-mode preprocessing and creator.

- crop box follows https://github.com/Linketic/VGGT-X @ 26d1b95, vggt/utils/load_fn.py:211-251
- CUDA only: vggt.models runs a CUDA warmup at import, so _load_model imports it after a CUDA check
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from collab_splats.geometry.projection import unproject
from collab_splats.geometry.transforms import extrinsics_to_homogeneous
from collab_splats.pointcloud.feedforward.base import (
    BaseFeedforwardCreator,
    _decode_depth_head,
    capture_qk,
    center_crop_coords,
)

########################################################################
# Creator
########################################################################


@BaseFeedforwardCreator.register("vggtx")
@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """
    Pointcloud via VGGT-X feedforward pose + depth estimation.

    - predicts camera poses and per-frame depth jointly, then unprojects to a cloud

    Attributes:
        model_name: HuggingFace id for VGGT.from_pretrained.
        chunk_size: attention chunk size; lower it on OOM for long sequences.
        conf_threshold: depth-confidence percentile cutoff (0-100); 35.0 keeps the top 65%.
    """

    # Loop-closure settings for VGGT-X, calibrated in docs/parity.md
    _lc_layer_index: ClassVar[int] = 10  # layer 20, used for other VGGT models, does not detect loops here
    default_verify_match_ratio: ClassVar[float] = 1.17
    _lc_token_offset: ClassVar[int] = 5

    model_name: str = "facebook/VGGT-1B"
    chunk_size: int = 256
    conf_threshold: float = 35.0

    def _load_model(self, device: str) -> Any:
        """
        VGGT-X from HuggingFace in eval mode, bf16 on Ampere+ and fp16 otherwise.

        - imports vggt.models here: its import-time CUDA warmup would break CPU-only imports of the package
        """
        # VGGT-X's mlp.py compiles gelu on CUDA at import; refuse first with a clear message
        if not torch.cuda.is_available():
            raise RuntimeError("the vggtx backend needs a CUDA GPU; VGGT-X runs CUDA kernels at import")

        from vggt.models.vggt import VGGT

        # Use bfloat16 on Ampere or newer GPUs, float16 otherwise
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Load the pretrained weights and move the model to the device in eval mode
        model = VGGT.from_pretrained(self.model_name, chunk_size=self.chunk_size)
        model.eval()
        model = model.to(device, dtype=dtype)

        return model

    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        """
        Frame files through upstream VGGT crop mode, the preprocessing it trained on.

        - returns (N, 3, H, 518) images, H <= 518, and (N, 6) original_coords
        """
        # Compute each frame's crop box the same way upstream crop mode does
        rows = []

        for p in paths:
            w, h = Image.open(p).size
            new_h = round(h * (518 / w) / 14) * 14
            crop_h = min(new_h, 518)
            box = center_crop_coords((w, h), (518, new_h), (518, crop_h), (518 / w, new_h / h))
            rows.append(box)

        original_coords = np.array(rows, dtype=np.float32)

        # Load and crop the images with upstream's own preprocessing
        images = load_and_preprocess_images([str(p) for p in paths], mode="crop")

        return images, original_coords

    def _forward(self, model: Any, views: Any) -> dict:
        """
        One VGGT-X forward, decoded to CPU float32 arrays.

        - views: (N, 3, H, W) images from _preprocess
        - raw keys: 'images', 'extrinsic' (N, 3, 4) w2c, model-res 'intrinsics', 'depth', 'depth_conf'
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Move the images to the model's device and dtype
        images = views.to(device, dtype=dtype)

        # Run the model under autocast without tracking gradients
        with torch.no_grad():
            with torch.autocast(device.type, dtype=dtype):
                predictions = model(images.unsqueeze(0))

        # Convert the predictions into poses, intrinsics and depth maps
        return {"images": images, **_decode_depth_head(predictions, images.shape[-2:], pose_encoding_to_extri_intri)}

    def extract_intermediate_features(self, frames: torch.Tensor, layer_index: int) -> dict[str, Any]:
        """
        Hook aggregator.global_blocks[layer_index].attn.qkv on a 2-frame forward.

        - the same forward gives poses and geometry, so _verify_loop_candidate needs no second one

        Args:
            frames: (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: global block to tap; -1 = last.

        Returns:
            "q", "k" (B, heads, tokens, head_dim), "poses" (2, 4, 4) w2c,
            "world_points" (2, H, W, 3) and "conf" (2, H, W).
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # Add a batch dimension and move the frames to the model's device and dtype
        batch = frames.unsqueeze(0).to(device, dtype=dtype)

        # Capture attention queries and keys at the chosen layer during one forward pass
        attn = self.model.aggregator.global_blocks[layer_index].attn

        with capture_qk(attn.qkv, attn.num_heads) as captured, torch.no_grad():
            predictions = self.model(batch)

        # Decode poses and depth from that same forward pass
        raw = _decode_depth_head(predictions, frames.shape[-2:], pose_encoding_to_extri_intri)

        # Unproject each frame's depth into world points
        depth = torch.from_numpy(raw["depth"][..., 0])
        world_to_cam = torch.from_numpy(raw["extrinsic"])
        intrinsics = torch.from_numpy(raw["intrinsics"])
        world_points = unproject(depth, world_to_cam, intrinsics)

        # Return the poses and geometry alongside the captured queries and keys
        captured["poses"] = extrinsics_to_homogeneous(raw["extrinsic"])
        captured["world_points"] = world_points.numpy()
        captured["conf"] = raw["depth_conf"]

        return captured
