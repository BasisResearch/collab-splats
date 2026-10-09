"""
Sky segmentation over the skywater SegFormer in PyTorch, plus a cached per-scene mask stack.

- SkyWaterSegmentation ("skywater"): MiT-B2 fine-tuned on ADE20K sky/water/person
- sky_masks: (N, H, W) mask stack over a cached sky-probability PNG per frame, order-preserving
- used instead of VGGT's skyseg (facebookresearch/vggt @ a288dd0f14786c93483e45524328726ab7b1b4ce,
  visual_util.py:365-434), whose per-image min-max rescale invents sky on dim frames
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

import segmentation_models_pytorch as smp

from collab_splats.preproc import frames
from collab_splats.semantics.segmentation.base import BaseSegmentation
from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD, open_image
from collab_splats.utils.torch_utils import (
    batch_iterator,
    full_fp32_matmul,
    get_device,
    load_hf_weights,
)

logger = logging.getLogger(__name__)

########################################################
########## Sky backend #################################
########################################################


@BaseSegmentation.register("skywater")
class SkyWaterSegmentation(BaseSegmentation):
    """
    Sky mask from the skywater SegFormer MiT-B2, fine-tuned on ADE20K.

    - water and person are segmented too but unused; the mesh stage masks sky alone
    - the probability is resized to the frame BEFORE thresholding, so the mask boundary
      aliases onto the frame grid and not the model's square one
    - weights are the repo's smp checkpoint, of which its ONNX file is an export

    Args:
        threshold: probability above which a pixel is sky.
        repo_id: Hugging Face repo holding config.json and model.safetensors.
        input_size: the network's fixed square input edge; aspect ratio is not preserved.
        sky_class: sky's index along the (background, sky, water, person) logit axis.
        device: torch device; None picks one with get_device.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        repo_id: str = "Realcat/skywater_seg",
        input_size: int = 384,
        sky_class: int = 1,
        device: str | None = None,
    ) -> None:
        # Build the SegFormer from the repo config; the checkpoint replaces the ImageNet init
        config_path = load_hf_weights(repo_id, "config.json")

        with open(config_path) as f:
            config = json.load(f)

        model = smp.Segformer(
            encoder_name=config["encoder_name"],
            encoder_weights=None,
            decoder_segmentation_channels=config["decoder_channels"],
            in_channels=config["in_channels"],
            classes=config["classes"],
        )

        # The checkpoint wraps the smp model as `_model`; strip that prefix, then load strictly
        weights_path = load_hf_weights(repo_id, "model.safetensors")
        state_dict = load_file(weights_path)
        consume_prefix_in_state_dict_if_present(state_dict, "_model.")
        model.load_state_dict(state_dict)

        self._device = device or get_device()
        self._model = model.to(self._device).eval()
        self._threshold = threshold
        self._input_size = input_size
        self._sky_class = sky_class

    def segment(self, image: np.ndarray | Image.Image) -> tuple[torch.Tensor, dict]:
        """
        Sky mask for one frame.

        Args:
            image: the frame to segment, coerced through `utils.image.open_image`.

        Returns:
            (mask, metadata) — mask (H, W) bool at input resolution, True where sky;
            metadata carries 'raw', the (H, W) float32 probability map.
        """
        return self.segment_batch([image])[0]

    def segment_batch(
        self, images: Sequence[np.ndarray | Image.Image | Path | str]
    ) -> list[tuple[torch.Tensor, dict]]:
        """
        Sky masks for several frames in one forward pass.

        - frames may differ in size: each is squared for the model, then resized back to its own

        Args:
            images: the frames to segment, each coerced through `utils.image.open_image`.

        Returns:
            One (mask, metadata) per image, in order, shaped as `segment` returns them.
        """
        rgbs = [np.asarray(open_image(image).convert("RGB")) for image in images]

        # The model's own preprocessing: square resize, /255, ImageNet normalize, NCHW
        squares = np.stack(
            [cv2.resize(rgb, (self._input_size, self._input_size)) for rgb in rgbs]
        )
        squares = squares.astype(np.float32) / 255.0
        squares = (squares - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(
            IMAGENET_STD, np.float32
        )
        squares = np.ascontiguousarray(squares.transpose(0, 3, 1, 2))
        batch = torch.from_numpy(squares).to(self._device)

        # Full fp32 forward: cuDNN's default TF32 convs drift the probability ~1e-3 from the ONNX export
        precision = torch.backends.cudnn.flags(enabled=True, allow_tf32=False)

        # Softmax over the class axis: the output is logits; skyseg's min-max rescale stays dropped
        with torch.inference_mode(), precision, full_fp32_matmul():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)[:, self._sky_class].cpu().numpy()

        # Resize each probability to its own frame BEFORE thresholding
        results = []

        for rgb, prob in zip(rgbs, probs):
            raw = cv2.resize(
                prob, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            results.append((torch.from_numpy(raw > self._threshold), {"raw": raw}))

        return results


########################################################
########## Cached sky-probability stack ################
########################################################


def sky_masks(
    images_dir: Path | str,
    idxs: Sequence[int] | None = None,
    cache_dir: Path | str | None = None,
    threshold: float = 0.5,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Sky masks for a keyframe directory, segmenting only what is not already cached.

    - the cache is keyed by frame index only: a changed threshold is re-applied on read,
      but a changed model reuses stale PNGs
    - cached PNGs store sky probability x 255; old 0/255 PNGs read as the same mask at
      any threshold, so re-thresholding one means deleting the cache dir first
    - the backend's own `threshold` does not affect this function; `threshold=` below
      is the only knob

    Args:
        images_dir: keyframe directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices in the order wanted; None takes every frame in
            filename order, exactly as `frames.read_frames` does.
        cache_dir: directory of cached sky-probability PNGs; None uses images_dir's
            sibling sky/.
        threshold: sky probability above which a pixel is sky; applied on read, so
            changing it needs no re-segmentation.
        batch_size: uncached frames decoded and segmented per forward pass.

    Returns:
        (N, H, W) bool, True where sky, in the order `idxs` names.

    Raises:
        ValueError: when threshold is outside [0, 1), which marks every pixel sky or none.
        FileNotFoundError: when no frame resolves: idxs is None over an empty images_dir, or idxs is empty.
        KeyError: when idxs names a frame_idx the directory does not hold.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"sky_masks: threshold must be in [0, 1), got {threshold}")

    images_dir = Path(images_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else images_dir.parent / "sky"

    # Resolve wanted frames up front, so a warm cache rejects junk too
    paths = frames.frame_paths(images_dir, idxs)

    if not paths:
        raise FileNotFoundError(f"sky_masks: no frame images in {images_dir}")

    wanted = [frames.frame_idx_from_path(p) for p in paths]
    path_of = dict(zip(wanted, paths))

    # Segment only the cache misses, batch_size frames per pass, never every miss in RAM at once
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in wanted if not (cache_dir / f"frame_{i:06d}.png").exists()]

    if todo:
        model = BaseSegmentation.get("skywater")()

        for (chunk,) in batch_iterator(batch_size, todo):
            results = model.segment_batch([path_of[idx] for idx in chunk])

            for idx, (_, meta) in zip(chunk, results):
                prob8 = np.rint(np.clip(meta["raw"], 0.0, 1.0) * 255).astype(np.uint8)
                cv2.imwrite(str(cache_dir / f"frame_{idx:06d}.png"), prob8)

        logger.info(
            "sky_masks: segmented %d of %d frames into %s",
            len(todo),
            len(wanted),
            cache_dir,
        )

    # Threshold on read, so hits and misses share one path and a new threshold reuses the cache
    probs = frames.read_frames(cache_dir, wanted, gray=True)
    masks = np.empty(probs.shape, dtype=bool)

    # Per-frame threshold keeps the float temporary to one frame
    for row, prob in enumerate(probs):
        masks[row] = prob / 255.0 > threshold

    return masks
