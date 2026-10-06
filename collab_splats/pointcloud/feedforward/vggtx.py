"""
VGGT-X feedforward backend: crop-mode preprocessing and creator.

- crop box follows https://github.com/Linketic/VGGT-X @ 26d1b95, vggt/utils/load_fn.py:211-251
- CUDA only: vggt.models runs a CUDA warmup at import, so _load_model imports it after a CUDA check
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from collab_splats.geometry.projection import unproject
from collab_splats.geometry.transforms import extrinsics_to_homogeneous
from collab_splats.pointcloud.feedforward.base import (
    BaseFeedforwardCreator,
    _decode_depth_head,
    _frame_sizes,
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
        Frames through upstream VGGT crop mode, the preprocessing it trained on.

        - returns (N, 3, H, 518) images, H <= 518, and (N, 6) original_coords
        """
        # Take the handed-off arrays only when they cover every path
        arrays = self._get_path_frames(paths)

        # Read the files when nothing was handed off, else preprocess the arrays
        if arrays is None:
            files = [str(p) for p in paths]
            images = load_and_preprocess_images(files, mode="crop")
        else:
            images = self._preprocess_arrays(arrays)

        # Each frame's crop box from its original size
        sizes = _frame_sizes(paths, arrays)
        original_coords = _crop_boxes(sizes)

        return images, original_coords

    def _preprocess_arrays(self, arrays: list[np.ndarray], *, workers: int = 8) -> torch.Tensor:
        """
        RGB arrays through upstream crop mode on a thread pool, minus the file open.

        - upstream's alpha blend is a no-op on (H, W, 3) uint8: no alpha, already RGB
        - upstream does the steps inline with no reusable helper; the bit-exact test guards drift
        """
        # Resize, tensorize and crop every frame
        with ThreadPoolExecutor(workers) as pool:
            images = list(pool.map(_preprocess_frame, arrays))

        # Mixed shapes get upstream's white padding to a common size
        shapes = {(im.shape[1], im.shape[2]) for im in images}

        if len(shapes) > 1:
            images = _pad_to_largest(images, shapes)

        return torch.stack(images)

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


########################################################################
# Helpers
########################################################################


def _crop_boxes(sizes: list[tuple[int, int]]) -> np.ndarray:
    """
    (N, 6) float32 crop boxes for (w, h) frames, the way upstream crop mode resizes and crops.
    """
    rows = []

    for w, h in sizes:
        new_h = round(h * (518 / w) / 14) * 14
        crop_h = min(new_h, 518)
        box = center_crop_coords((w, h), (518, new_h), (518, crop_h), (518 / w, new_h / h))
        rows.append(box)

    return np.array(rows, dtype=np.float32)


def _preprocess_frame(rgb: np.ndarray) -> torch.Tensor:
    """
    One frame resized to width 518 (height a multiple of 14), then center-cropped to 518 tall.
    """
    # Bicubic resize to width 518, height snapped to the 14 px patch
    image = Image.fromarray(rgb)
    width, height = image.size
    new_height = round(height * (518 / width) / 14) * 14
    resized = image.resize((518, new_height), Image.Resampling.BICUBIC)
    tensor = to_tensor(resized)

    # Center crop the height down to 518
    if new_height > 518:
        start_y = (new_height - 518) // 2
        tensor = tensor[:, start_y : start_y + 518, :]

    return tensor


def _pad_to_largest(images: list[torch.Tensor], shapes: set[tuple[int, int]]) -> list[torch.Tensor]:
    """
    (3, H, W) images padded white, centered, to the largest of shapes' heights and widths.
    """
    # Pad each smaller image evenly on both sides, the extra pixel going after
    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)
    padded = []

    for im in images:
        pad_h = max_h - im.shape[1]
        pad_w = max_w - im.shape[2]

        if pad_h > 0 or pad_w > 0:
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            im = torch.nn.functional.pad(im, pad, mode="constant", value=1.0)

        padded.append(im)

    return padded
