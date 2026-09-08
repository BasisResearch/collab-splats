"""
Sky segmentation over the skywater ONNX SegFormer, plus a cached per-scene mask stack.

- SkyWaterSegmentation ("skywater"): MiT-B2 fine-tuned on ADE20K sky/water/person
- sky_masks: cached (N, H, W) mask stack for a keyframe directory, order-preserving
- replaces VGGT's skyseg (facebookresearch/vggt @ a288dd0f14786c93483e45524328726ab7b1b4ce,
  visual_util.py:365-434), which measured as a brightness detector on our scenes
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from collab_splats.preproc import frames
from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD, open_image
from collab_splats.utils.torch_utils import load_hf_weights

from .base import BaseSegmentation

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

    Args:
        threshold: probability above which a pixel is sky.
        repo_id: Hugging Face repo holding the ONNX file.
        filename: ONNX file within that repo.
        input_size: the network's fixed square input edge; aspect ratio is not preserved.
        sky_class: sky's index along the (background, sky, water, person) logit axis.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        repo_id: str = "Realcat/skywater_seg",
        filename: str = "skywater_segformer_b2_fp32.onnx",
        input_size: int = 384,
        sky_class: int = 1,
    ) -> None:
        # CUDA first when the wheel provides it; this container's onnxruntime is CPU-only,
        # and naming an unavailable provider only warns, so filtering keeps the log honest
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()
        ]
        self._session = ort.InferenceSession(str(load_hf_weights(repo_id, filename)), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
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
        rgb = np.asarray(open_image(image).convert("RGB"))
        height, width = rgb.shape[:2]

        # The model's own preprocessing: square resize, /255, ImageNet normalize, NCHW
        square = cv2.resize(rgb, (self._input_size, self._input_size)).astype(np.float32) / 255.0
        square = (square - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
        tensor = square.transpose(2, 0, 1)[None].astype(np.float32)

        # Softmax the (1, 4, h, w) output: it is class logits, not probabilities
        # - max-subtracted, since a large logit overflows exp()
        # - upstream skyseg's per-image min-max rescale stays dropped: it stretches a dim
        #   frame's maximum to 1.0 and invents sky where the model reported none
        logits = np.asarray(self._session.run(None, {self._input_name: tensor})[0])[0].astype(np.float32)
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        prob = (exp / exp.sum(axis=0))[self._sky_class]

        raw = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(raw > self._threshold), {"raw": raw}


########################################################
########## Cached mask stack ###########################
########################################################


def sky_masks(
    images_dir: Path | str,
    idxs: Sequence[int] | None = None,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """
    Sky masks for a keyframe directory, segmenting only what is not already cached.

    - inference costs ~0.2 s/frame on CPU, so the PNG cache is load-bearing
    - cached PNGs store 255 for sky, inverted from skyseg's 255-is-NOT-sky files

    Args:
        images_dir: keyframe directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices in the order wanted; None takes every frame in
            filename order, exactly as `frames.read_frames` does.
        cache_dir: directory of cached mask PNGs; None uses images_dir's sibling sky/.

    Returns:
        (N, H, W) bool, True where sky, in the order `idxs` names.

    Raises:
        FileNotFoundError: when images_dir holds no frames.
        KeyError: when idxs names a frame_idx the directory does not hold.
    """
    images_dir = Path(images_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else images_dir.parent / "sky"

    paths = frames.frame_paths(images_dir)
    if not paths:
        raise FileNotFoundError(f"sky_masks: no frame images in {images_dir}")

    # Resolve the wanted source indices up front, so the contract holds whether or not the
    # cache is warm — validating only the uncached ones would let a warm cache accept junk
    by_idx = {frames.frame_idx_from_path(p): p for p in paths}
    wanted = [int(i) for i in idxs] if idxs is not None else list(by_idx)
    missing = [i for i in wanted if i not in by_idx]
    if missing:
        raise KeyError(f"sky_masks: frame_idx {missing[:5]} not in {images_dir}")

    # Segment only the cache misses, one frame at a time
    # - segment() coerces a path through open_image, so no decode step is needed here
    # - frames.read_frames would np.stack every miss at once: 853 full-res frames is ~10 GB,
    #   on top of whatever the caller's fusion buffers already hold
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in wanted if not (cache_dir / f"frame_{i:06d}.png").exists()]
    if todo:
        model = BaseSegmentation.get("skywater")()
        for idx in todo:
            mask, _ = model.segment(by_idx[idx])
            cv2.imwrite(str(cache_dir / f"frame_{idx:06d}.png"), np.asarray(mask, dtype=np.uint8) * 255)
        logger.info("sky_masks: segmented %d of %d frames into %s", len(todo), len(wanted), cache_dir)

    # Read every mask back from the cache, so a hit and a miss return the identical array
    return np.stack([cv2.imread(str(cache_dir / f"frame_{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in wanted])
