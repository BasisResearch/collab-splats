"""
VGGT-Omega feedforward backend: pose and depth from Omega checkpoints.

- default checkpoint: VGGT_OMEGA_DEFAULT_FILENAME (512-res) from VGGT_OMEGA_HF_REPO
- crop box follows https://github.com/facebookresearch/vggt-omega @ 39a0cb8, vggt_omega/utils/load_fn.py:68-82
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from collab_splats.geometry.projection import unproject
from collab_splats.geometry.transforms import extrinsics_to_homogeneous
from collab_splats.pointcloud.feedforward.base import (
    BaseFeedforwardCreator,
    _decode_depth_head,
    capture_qk,
    center_crop_coords,
)
from collab_splats.utils.torch_utils import load_hf_weights

########################################################################
# Constants
########################################################################

VGGT_OMEGA_HF_REPO = "facebook/VGGT-Omega"
VGGT_OMEGA_DEFAULT_FILENAME = "vggt_omega_1b_512.pt"

########################################################################
# Helpers
########################################################################


def _load_chunked(paths: list[str], resolution: int, mode: str, *, workers: int = 8) -> torch.Tensor:
    """
    Upstream load_and_preprocess_images over contiguous path chunks on a thread pool.

    - PIL releases the GIL in decode and resize, so threads scale
    - upstream pads mixed-shape frames to one size across the whole list; chunks of differing
      shapes fall back to one upstream call so the output always matches upstream exactly
    """
    size = -(-len(paths) // workers)
    chunks = [paths[i : i + size] for i in range(0, len(paths), size)]

    # Each chunk through upstream's own crop and resize
    def _load(chunk: list[str]) -> torch.Tensor:
        """
        One chunk through upstream preprocessing.
        """
        return load_and_preprocess_images(chunk, image_resolution=resolution, mode=mode)

    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(_load, chunks))

    # Mixed shapes need upstream's list-wide padding
    if len({part.shape[1:] for part in parts}) > 1:
        return _load(paths)

    return torch.cat(parts)


########################################################################
# Creator
########################################################################


@BaseFeedforwardCreator.register("vggt_omega")
@dataclass
class VGGTOmegaCreator(BaseFeedforwardCreator):
    """
    Pointcloud via VGGT-Omega feedforward pose + depth estimation.

    - predicts camera poses and per-frame depth jointly, then unprojects to a cloud

    Attributes:
        model_path: local checkpoint file; None downloads the default from HuggingFace.
        resolution: target image resolution; 512 matches the default checkpoint.
        resize_mode: "balanced" (token count near resolution^2) or "max_size" (longest side = resolution).
    """

    # Loop-closure settings for VGGT-Omega, calibrated in docs/parity.md
    _lc_layer_index: ClassVar[int] = 13
    default_verify_match_ratio: ClassVar[float] = 1.55
    _lc_token_offset: ClassVar[int] = 5  # VGGT's 5 leading tokens work better here than Omega's own 17

    model_path: str | None = None
    resolution: int = 512
    resize_mode: str = "balanced"  # resize mode passed to load_and_preprocess_images

    def __post_init__(self) -> None:
        """
        Reject an unknown resize_mode.
        """
        if self.resize_mode not in {"balanced", "max_size"}:
            raise ValueError(f"resize_mode must be one of {{'balanced', 'max_size'}}, got {self.resize_mode!r}")

    def _load_model(self, device: str) -> Any:
        """
        VGGT-Omega from model_path or HuggingFace, in eval mode with fp32 params.

        - params stay fp32; the aggregator autocasts internally
        - bf16 params would crash the heads' fp32 LayerNorms
        """
        # Use the local checkpoint if given, otherwise download it from HuggingFace
        ckpt = self.model_path or load_hf_weights(VGGT_OMEGA_HF_REPO, VGGT_OMEGA_DEFAULT_FILENAME)

        # Load the weights and move the model to the device in eval mode
        model = VGGTOmega()
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        model = model.to(device)

        return model

    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        """
        Frame files through Omega's crop and resize, plus each frame's crop box.

        - both resize modes first center-crop h / w into [0.5, 2.0]
        - returns (N, 3, H, W) images and (N, 6) original_coords
        """
        # Compute each frame's aspect-ratio crop box, before any resizing
        rows = []

        for p in paths:
            w, h = Image.open(p).size
            crop_w, crop_h = w, h

            if h / w < 0.5:
                crop_w = 2 * h
            elif h / w > 2.0:
                crop_h = 2 * w

            box = center_crop_coords((w, h), (w, h), (crop_w, crop_h), (1, 1))
            rows.append(box)

        original_coords = np.array(rows, dtype=np.float32)

        # Crop and resize the images with upstream's own preprocessing
        images = _load_chunked([str(p) for p in paths], self.resolution, self.resize_mode)

        return images, original_coords

    def _forward(self, model: Any, views: Any) -> dict:
        """
        One VGGT-Omega forward, decoded to CPU float32 arrays.

        - views: (N, 3, H, W) images from _preprocess
        - raw keys: 'images', 'extrinsic' (N, 3, 4) w2c, model-res 'intrinsics', 'depth', 'depth_conf'
        """
        device = next(model.parameters()).device

        # Move the images to the model's device (the model adds the batch dimension itself)
        images = views.to(device)

        # Run the model, which handles mixed precision internally
        with torch.no_grad():
            predictions = model(images)

        # Convert the predictions into poses, intrinsics and depth maps, as upstream's demo does
        return {"images": images, **_decode_depth_head(predictions, views.shape[-2:], encoding_to_camera)}

    def extract_intermediate_features(self, frames: torch.Tensor, layer_index: int) -> dict[str, Any]:
        """
        Hook inter_frame_blocks[layer_index].attn.qkv on a 2-frame forward.

        - the same forward gives poses and geometry, so _verify_loop_candidate needs no second one

        Args:
            frames: (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: inter-frame block to tap; -1 = last.

        Returns:
            "q", "k" (B, heads, tokens, head_dim), "poses" (2, 4, 4) w2c,
            "world_points" (2, H, W, 3) and "conf" (2, H, W).
        """
        device = next(self.model.parameters()).device

        # Move the frames to the model's device only, since the model handles precision itself
        images = frames.to(device)

        # Capture attention queries and keys at the chosen layer during one forward pass
        attn = self.model.aggregator.inter_frame_blocks[layer_index].attn

        with capture_qk(attn.qkv, attn.num_heads) as captured, torch.no_grad():
            predictions = self.model(images)

        # Decode poses and depth from that same forward pass
        raw = _decode_depth_head(predictions, frames.shape[-2:], encoding_to_camera)

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
