"""
MapAnything feedforward backend: metric depth and camera poses in one forward pass.

- crop box and mask follow https://github.com/facebookresearch/map-anything @ c845b8f
- crop box: mapanything/utils/cropping.py:193, 231-240, 441-447
- upstream mask: mapanything/utils/inference.py:401-402
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

from mapanything.models import MapAnything
from mapanything.utils.cropping import crop_resize_if_necessary
from mapanything.utils.image import (
    IMAGE_NORMALIZATION_DICT,
    find_closest_aspect_ratio,
    load_images,
)
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)

from collab_splats.geometry.transforms import invert_poses
from collab_splats.pointcloud.feedforward.base import (
    BaseFeedforwardCreator,
    _frame_sizes,
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
        conf_percentile: learned-confidence percentile cutoff (0-100), applied by upstream's mask.
        min_views: other views that must agree to keep a pixel; 0 is off.
        mv_rel_thresh: depth tolerance as a fraction of the expected depth.
        minibatch_size: views per inference step; lower it on OOM.
        resize_mode: "fixed" (aspect-ratio lookup table), "longest_side" or "square".
        resolution: lookup-table size for "fixed" (518 or 512); target px otherwise.
    """

    # MapAnything adds no extra tokens before the image tokens in its attention blocks
    _lc_token_offset: ClassVar[int] = 0

    # Loop-closure settings tuned for this model (see decision 027)
    default_verify_match_ratio: ClassVar[float] = 1.46
    _lc_layer_index: ClassVar[int] = 4

    model_name: str = "facebook/map-anything"
    conf_percentile: float = 35.0
    minibatch_size: int = 1
    resize_mode: str = (
        "fixed"  # "fixed" (aspect-ratio lookup table), "longest_side", "square"
    )
    resolution: int = (
        518  # lookup-table size for "fixed", target size in pixels otherwise
    )

    # The full sequence of views, prepared for the model by _preprocess
    _processed_views: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Reject an unknown resize_mode.
        """
        if self.resize_mode not in _MA_RESIZE_MODE_MAP:
            raise ValueError(
                f"resize_mode must be one of {sorted(_MA_RESIZE_MODE_MAP)}, got {self.resize_mode!r}"
            )

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

        - crop boxes ignore EXIF orientation; frame-store PNGs carry none
        - returns (views, original_coords): view dicts and (N, 6) float32 crop boxes
        """
        # Take the handed-off arrays only when they cover every path
        arrays = self._get_path_frames(paths)

        # Read the files when nothing was handed off, else preprocess the arrays
        if arrays is None:
            views = self._preprocess_files(paths)
        else:
            views = self._preprocess_arrays(arrays)

        # Each frame's crop box from its original size and the model grid
        model_h: int = views[0]["img"].shape[-2]
        model_w: int = views[0]["img"].shape[-1]
        sizes = _frame_sizes(paths, arrays)
        original_coords = _crop_boxes(sizes, model_w, model_h)

        # Check the views and convert them to the model's input format, leaving them on the CPU
        validated = validate_input_views_for_inference(views)
        self._processed_views = preprocess_input_views_for_inference(validated)

        return views, original_coords

    def _preprocess_files(self, paths: list[Path]) -> list[dict[str, Any]]:
        """
        Frame files as view dicts through MapAnything's own loader.
        """
        loader_paths = [str(p) for p in paths]
        upstream_mode = _MA_RESIZE_MODE_MAP[self.resize_mode]

        # The fixed mode picks from a lookup table; the others take a target size
        if self.resize_mode == "fixed":
            return load_images(
                loader_paths, resize_mode=upstream_mode, resolution_set=self.resolution
            )

        return load_images(
            loader_paths, resize_mode=upstream_mode, size=self.resolution
        )

    def _preprocess_arrays(
        self, arrays: list[np.ndarray], *, workers: int = 8
    ) -> list[dict[str, Any]]:
        """
        RGB arrays as load_images view dicts, resized on a thread pool, minus the file open.

        - mirrors mapanything.utils.image.load_images for fixed_mapping, square and longest_side
        - upstream's exif_transpose and RGB convert are no-ops on (H, W, 3) uint8 arrays
        - patch size 14 and dinov2 normalization, load_images' defaults
        - the target-size step is inline upstream; the bit-exact test guards drift
        """
        # One target size for every frame, from the mean aspect ratio
        aspect_ratios = [rgb.shape[1] / rgb.shape[0] for rgb in arrays]
        average = sum(aspect_ratios) / len(aspect_ratios)
        size = self.resolution

        if self.resize_mode == "fixed":
            target_size = find_closest_aspect_ratio(average, size)
        elif self.resize_mode == "square":
            target_size = (round(size // 14) * 14, round(size // 14) * 14)
        elif average >= 1:
            target_size = (size, round((size // 14) / average) * 14)
        else:
            target_size = (round((size // 14) * average) * 14, size)

        # Normalize the way load_images does for dinov2
        norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        to_tensor = TF.ToTensor()
        normalize = TF.Normalize(mean=norm.mean, std=norm.std)
        img_norm = TF.Compose([to_tensor, normalize])

        # Resize and normalize every frame
        preprocess = partial(
            _preprocess_frame, target_size=target_size, img_norm=img_norm
        )

        with ThreadPoolExecutor(workers) as pool:
            processed = list(pool.map(preprocess, arrays))

        # One view dict per frame, keyed as load_images keys them
        return [
            {
                "img": img,
                "true_shape": true_shape,
                "idx": i,
                "instance": str(i),
                "data_norm_type": ["dinov2"],
            }
            for i, (img, true_shape) in enumerate(processed)
        ]

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
            logger.debug(
                "  → %d images, minibatch_size=%d",
                len(forward_views),
                self.minibatch_size,
            )
        else:
            window = preprocess_input_views_for_inference(
                validate_input_views_for_inference(views)
            )
            forward_views = _views_to(window, device)
            logger.debug(
                "  → %d images (LC window), minibatch_size=%d",
                len(forward_views),
                self.minibatch_size,
            )

        # Run the model in bfloat16 on the GPU, while postprocessing later stays in float32
        with torch.no_grad():
            with torch.autocast(
                device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")
            ):
                preds = model.forward(
                    forward_views,
                    memory_efficient_inference=True,
                    minibatch_size=self.minibatch_size,
                )

        return self._stack_predictions(preds, forward_views, masked=masked)

    def _stack_predictions(
        self, preds: list[dict], views: list[dict], *, masked: bool
    ) -> dict[str, np.ndarray]:
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
                confidence_percentile=self.conf_percentile,
            )
        else:
            processed = postprocess_model_outputs_for_inference(
                preds, views, apply_mask=False
            )

        # Stack each view's outputs into arrays, turning camera-to-world poses into world-to-camera
        raw = {
            "extrinsic": np.stack(
                [
                    invert_poses(p["camera_poses"][0].cpu().float().numpy())[:3]
                    for p in processed
                ]
            ),
            "intrinsics": np.stack(
                [p["intrinsics"][0].cpu().float().numpy() for p in processed]
            ),
            "depth": np.stack(
                [p["depth_z"][0].cpu().float().numpy() for p in processed]
            ),
            "depth_conf": np.stack(
                [p["conf"][0].cpu().float().numpy() for p in processed]
            ),
            "images": np.stack(
                [
                    p["img_no_norm"][0].cpu().float().numpy().transpose(2, 0, 1)
                    for p in processed
                ]
            ),
        }

        # Keep MapAnything's valid-pixel mask, but only for the full sequence
        if masked:
            raw["mask"] = np.stack(
                [p["mask"][0, ..., 0].cpu().numpy().astype(bool) for p in processed]
            )

        return raw

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int
    ) -> dict[str, Any]:
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
            raw_views = [
                {"img": f.unsqueeze(0), "data_norm_type": ["dinov2"]}
                for f in frames.cpu()
            ]

            # Views are built from CPU frames, so move them to the model device
            views = _views_to(
                preprocess_input_views_for_inference(raw_views),
                next(self.model.parameters()).device,
            )
            preds = self.model.forward(
                views, memory_efficient_inference=False, minibatch_size=1
            )

        # Convert pointmaps to float32 and postprocess them, the same as _stack_predictions
        with torch.no_grad():
            for pred in preds:
                pred["pts3d_cam"] = pred["pts3d_cam"].float()
                pred["pts3d"] = pred["pts3d"].float()

            processed = postprocess_model_outputs_for_inference(
                preds, views, apply_mask=False
            )

        # Turn camera-to-world poses into world-to-camera
        captured["poses"] = np.stack(
            [
                invert_poses(p["camera_poses"][0].cpu().float().numpy())
                for p in processed
            ]
        )

        # Pointmaps and confidence for loop-closure scale estimation
        captured["world_points"] = np.stack(
            [p["pts3d"][0].cpu().float().numpy() for p in processed]
        )  # (2, H, W, 3)
        captured["conf"] = np.stack(
            [p["conf"][0].cpu().float().numpy() for p in processed]
        )  # (2, H, W)

        return captured


########################################################################
# Helpers
########################################################################


def _views_to(
    views: list[dict[str, Any]], device: torch.device | str
) -> list[dict[str, Any]]:
    """
    Move every tensor in a list of MapAnything view dicts to `device`.

    - returns new view dicts; non-tensor values pass through
    """
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in view.items()}
        for view in views
    ]


def _crop_boxes(sizes: list[tuple[int, int]], model_w: int, model_h: int) -> np.ndarray:
    """
    (N, 6) float32 crop boxes for (w, h) frames, the way MapAnything's loader resizes and crops.
    """
    rows = []

    for w, h in sizes:
        s = (
            max(model_w / w, model_h / h) + 1e-8
        )  # same scale formula as MapAnything's loader
        resized_wh = (int(w * s), int(h * s))
        box = center_crop_coords((w, h), resized_wh, (model_w, model_h), (s, s))
        rows.append(box)

    return np.array(rows, dtype=np.float32)


def _preprocess_frame(
    rgb: np.ndarray, *, target_size: tuple[int, int], img_norm: TF.Compose
) -> tuple[torch.Tensor, np.ndarray]:
    """
    One frame through upstream's Lanczos resize and center crop, then normalized.

    - returns the (1, 3, H, W) image and its (1, 2) int32 true_shape
    """
    image = Image.fromarray(rgb)
    resized = crop_resize_if_necessary(image, resolution=target_size)[0]
    normalized = img_norm(resized)

    return normalized[None], np.array([resized.size[::-1]], dtype=np.int32)
