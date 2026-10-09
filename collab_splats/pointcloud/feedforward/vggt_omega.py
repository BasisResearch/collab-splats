"""
VGGT-Omega feedforward backend: pose and depth from Omega checkpoints.

- default checkpoint: VGGT_OMEGA_DEFAULT_FILENAME (512-res) from VGGT_OMEGA_HF_REPO
- crop box follows https://github.com/facebookresearch/vggt-omega @ 39a0cb8, vggt_omega/utils/load_fn.py:68-82
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import (
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
    _max_size_target_shape,
    _pad_images_to_common_size,
    load_and_preprocess_images,
)
from vggt_omega.utils.pose_enc import encoding_to_camera

from collab_splats.geometry.projection import unproject_frames
from collab_splats.geometry.transforms import extrinsics_to_homogeneous
from collab_splats.pointcloud.feedforward.base import (
    BaseFeedforwardCreator,
    _decode_depth_head,
    _frame_sizes,
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


def _crop_boxes(sizes: list[tuple[int, int]]) -> np.ndarray:
    """
    (N, 6) float32 crop boxes for (w, h) frames, upstream's center crop of h / w into [0.5, 2.0].
    """
    rows = []

    for w, h in sizes:
        crop_w, crop_h = w, h

        if h / w < 0.5:
            crop_w = 2 * h
        elif h / w > 2.0:
            crop_h = 2 * w

        box = center_crop_coords((w, h), (w, h), (crop_w, crop_h), (1, 1))
        rows.append(box)

    return np.array(rows, dtype=np.float32)


def _preprocess_frame(
    rgb: np.ndarray,
    *,
    target_shape: Callable[[float, int, int], tuple[int, int]],
    resolution: int,
) -> torch.Tensor:
    """
    One frame through upstream's aspect-ratio crop and bicubic resize, as a [0, 1] tensor.

    - target_shape is upstream's balanced or max_size (h, w) rule at patch size 16
    """
    # Upstream aspect-ratio crop, then bicubic resize to the target grid
    image = Image.fromarray(rgb)
    image = _crop_to_supported_aspect_ratio(image)
    width, height = image.size
    aspect = height / max(width, 1)
    target_h, target_w = target_shape(aspect, resolution, 16)
    resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

    return to_tensor(resized)


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
    _lc_token_offset: ClassVar[int] = (
        5  # VGGT's 5 leading tokens work better here than Omega's own 17
    )

    model_path: str | None = None
    resolution: int = 512
    resize_mode: str = "balanced"  # resize mode passed to load_and_preprocess_images

    def __post_init__(self) -> None:
        """
        Reject an unknown resize_mode.
        """
        if self.resize_mode not in {"balanced", "max_size"}:
            raise ValueError(
                f"resize_mode must be one of {{'balanced', 'max_size'}}, got {self.resize_mode!r}"
            )

    def _load_model(self, device: str) -> Any:
        """
        VGGT-Omega from model_path or HuggingFace, in eval mode with fp32 params.

        - params stay fp32; the aggregator autocasts internally
        - bf16 params would crash the heads' fp32 LayerNorms
        """
        # Use the local checkpoint if given, otherwise download it from HuggingFace
        ckpt = self.model_path or load_hf_weights(
            VGGT_OMEGA_HF_REPO, VGGT_OMEGA_DEFAULT_FILENAME
        )

        # Build the model on the device, skipping CPU random init
        with torch.device(device):
            model = VGGTOmega()

        # Load the weights memory-mapped from the checkpoint
        state_dict = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state_dict)

        # Move the legacy-constructor buffers that ignore the device context; a no-op for the rest
        model = model.to(device)
        model.eval()

        return model

    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        """
        Frames through Omega's crop and resize, plus each frame's crop box.

        - both resize modes first center-crop h / w into [0.5, 2.0]
        - returns (N, 3, H, W) images and (N, 6) original_coords
        """
        # Take the handed-off arrays only when they cover every path
        arrays = self._get_path_frames(paths)

        # Read the files when nothing was handed off, else preprocess the arrays
        if arrays is None:
            images = self._preprocess_files(paths)
        else:
            images = self._preprocess_arrays(arrays)

        # Each frame's aspect-ratio crop box from its original size
        sizes = _frame_sizes(paths, arrays)
        original_coords = _crop_boxes(sizes)

        return images, original_coords

    def _preprocess_files(self, paths: list[Path], *, workers: int = 8) -> torch.Tensor:
        """
        Frame files through upstream's loader, on contiguous path chunks in a thread pool.

        - PIL releases the GIL in decode and resize, so threads scale
        - chunks of differing shapes redo one upstream call, which pads them like upstream does
        """
        # Split the paths into one contiguous chunk per worker
        files = [str(p) for p in paths]
        n = len(files)
        size = -(-n // workers)
        chunks = [files[i : i + size] for i in range(0, n, size)]

        # Crop and resize each chunk with upstream's own preprocessing
        load = partial(
            load_and_preprocess_images,
            image_resolution=self.resolution,
            mode=self.resize_mode,
        )

        with ThreadPoolExecutor(workers) as pool:
            parts = list(pool.map(load, chunks))

        # Mixed shapes need upstream's list-wide padding
        if len({part.shape[1:] for part in parts}) > 1:
            return load(files)

        return torch.cat(parts)

    def _preprocess_arrays(
        self, arrays: list[np.ndarray], *, workers: int = 8
    ) -> torch.Tensor:
        """
        RGB arrays through upstream's per-image crop, resize and padding, on a thread pool.

        - mirrors load_and_preprocess_images step for step, minus the file open
        - upstream's _load_rgb_image is a no-op on (H, W, 3) uint8: no alpha, already RGB, no EXIF
        - patch size 16, upstream's default
        - imports upstream's private load_fn helpers; the bit-exact test guards drift
        """
        # Crop, resize and tensorize every frame
        target_shape = (
            _balanced_target_shape
            if self.resize_mode == "balanced"
            else _max_size_target_shape
        )
        preprocess = partial(
            _preprocess_frame, target_shape=target_shape, resolution=self.resolution
        )

        with ThreadPoolExecutor(workers) as pool:
            images = list(pool.map(preprocess, arrays))

        # Mixed shapes get upstream's padding to a common size
        shapes = {(im.shape[1], im.shape[2]) for im in images}

        if len(shapes) > 1:
            images = _pad_images_to_common_size(images, shapes)

        return torch.stack(images)

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
        return {
            "images": images,
            **_decode_depth_head(predictions, views.shape[-2:], encoding_to_camera),
        }

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int
    ) -> dict[str, Any]:
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
        world_points = unproject_frames(
            raw["depth"][..., 0], raw["extrinsic"], raw["intrinsics"]
        )

        # Return the poses and geometry alongside the captured queries and keys
        captured["poses"] = extrinsics_to_homogeneous(raw["extrinsic"])
        captured["world_points"] = world_points
        captured["conf"] = raw["depth_conf"]

        return captured
