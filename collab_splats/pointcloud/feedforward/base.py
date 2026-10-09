"""
Abstract template-method pipeline and shared helpers for feedforward creators.

- BaseFeedforwardCreator: the pipeline each backend subclasses; returns a PointcloudResult
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image

from collab_splats.geometry.projection import (
    multiview_depth_confidence,
    unproject_frames,
)
from collab_splats.geometry.transforms import extrinsics_to_homogeneous
from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.utils import confidence_mask, cross_frame_attention_ratio
from collab_splats.utils.torch_utils import RegistryMixin, get_device, pytorch_gc

logger = logging.getLogger(__name__)


########################################################################
# Abstract pipeline
########################################################################


@dataclass
class BaseFeedforwardCreator(BasePointcloudCreator, RegistryMixin):
    """
    Template-method pipeline shared by every feedforward creator.

    - _reconstruct steps: load_model -> setup_inference -> run_inference -> _postprocess
    - BasePointcloudCreator.create_pointcloud then cleans, caps and exports
    - the base owns device choice, CUDA cache release and step state
    - subclasses implement the abstract methods; each docstring states its contract

    Attributes:
        conf_percentile: depth-confidence percentile (0-100); pixels strictly above it are kept.
        min_views: other views that must agree with a pixel's depth; 0 turns the filter off.
        mv_rel_thresh: multiview agreement tolerance, as a fraction of depth.
        frames: preproc's RGB uint8 frames by file name; None, any path missing, or loger reads the files.
    """

    # Registry of feedforward backends, filled in as each backend module is imported
    _registry: ClassVar[dict[str, type["BaseFeedforwardCreator"]]] = {}

    # Loop-closure settings, overridden by each backend that supports loop closure
    default_verify_match_ratio: ClassVar[float | None] = None
    _lc_layer_index: ClassVar[int | None] = None
    _lc_token_offset: ClassVar[int | None] = None

    conf_percentile: float = 50.0

    # Settings for the multiview depth filter, which is off when min_views is 0
    min_views: int = 0
    mv_rel_thresh: float = 0.01

    # Preproc's frames, set by Reconstructor when this process decoded them
    frames: dict[str, np.ndarray] | None = field(default=None, repr=False)

    # State filled in as the pipeline runs
    model: Any = field(default=None, init=False, repr=False)
    views: Any = field(default=None, init=False, repr=False)
    image_paths: list[Path] | None = field(default=None, init=False, repr=False)
    original_coords: np.ndarray | None = field(default=None, init=False, repr=False)
    raw_outputs: Any = field(default=None, init=False, repr=False)

    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        Load, preprocess, forward and unproject the frames into a filtered cloud.
        """
        self.load_model()
        self.setup_inference(paths)
        self.run_inference()

        return self._postprocess(self.raw_outputs)

    def load_model(self) -> None:
        """
        Load the backend's model onto cuda when available, else cpu, once.

        - a loaded model stays where it is
        """
        if self.model is not None:
            return

        # Load and time the subclass's model
        device = get_device()
        t0 = time.perf_counter()
        logger.info("Loading model (%s)...", device)
        self.model = self._load_model(device)
        logger.info("  done in %.1fs", time.perf_counter() - t0)

    def setup_inference(self, paths: list[Path]) -> None:
        """
        Preprocess the frames into the backend's model input.

        - sets views, image_paths (frame stems) and original_coords

        Args:
            paths: frame image paths, reconstruction order; handed-off frames are matched by name.
        """
        # Name each frame by its file stem and let the backend prepare the model inputs
        t0 = time.perf_counter()
        self.image_paths = [Path(p.stem) for p in paths]
        self.views, self.original_coords = self._preprocess(paths)
        logger.info(
            "Preprocessed %d images in %.1fs", len(paths), time.perf_counter() - t0
        )

    def _list_frames(self, images_dir: Path) -> list[Path]:
        """
        Frame paths named by the handed-off frames, else listed from images_dir.

        - preproc may still be writing images/, so a handoff never lists the directory
        """
        if self.frames is None:
            return super()._list_frames(images_dir)

        return [images_dir / name for name in sorted(self.frames)]

    def _get_path_frames(self, paths: list[Path]) -> list[np.ndarray] | None:
        """
        The handed-off frame for every path, in order, or None to read the files.

        - None when nothing was handed off, or with a warning when any path is missing
        """
        if self.frames is None:
            return None

        # A partial handoff is ignored in favor of reading every file
        missing = sum(p.name not in self.frames for p in paths)

        if missing:
            logger.warning(
                "frames handed off but %d of %d paths missing; reading files",
                missing,
                len(paths),
            )
            return None

        return [self.frames[p.name] for p in paths]

    def run_inference(self) -> None:
        """
        Run the forward pass into raw_outputs, then free the CUDA cache.
        """
        t0 = time.perf_counter()
        logger.info("Running inference...")
        self.raw_outputs = self._forward(self.model, self.views)

        # Free forward-pass memory before postprocessing
        pytorch_gc()

        logger.info("  done in %.1fs", time.perf_counter() - t0)

    @abstractmethod
    def _load_model(self, device: str) -> Any:
        """
        Load the pretrained model onto `device` in eval mode.

        - holds no GPU state outside the returned model
        """
        ...

    @abstractmethod
    def _preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]:
        """
        Model input batch from the frames at paths.

        - a backend may read the handed-off arrays (_get_path_frames) instead of the files
        - returns (views, original_coords)
        - original_coords: (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]
        """
        ...

    @abstractmethod
    def _forward(self, model: Any, views: Any) -> Any:
        """
        Forward pass under torch.no_grad().

        - takes _load_model's model and _preprocess's views
        - returns the raw outputs _postprocess reads
        """
        ...

    def _multiview_mask(
        self, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Keep-mask of pixels that min(min_views, seen) other views agree with; all-True when off.

        - a pixel no other view sees is kept
        - depth (N, H, W); intrinsics (N, 3, 3) on the depth grid; extrinsics (N, 4, 4) w2c
        """
        # Skip the filter entirely when it is turned off
        if self.min_views == 0:
            return np.ones(depth.shape, dtype=bool)

        # Keep a pixel when enough of the other views that see it agree on its depth
        agree, seen = multiview_depth_confidence(
            depth, intrinsics, extrinsics, rel_thresh=self.mv_rel_thresh
        )

        return agree >= np.minimum(self.min_views, seen)

    def _postprocess(self, raw_outputs: dict[str, Any]) -> PointcloudResult:
        """
        Unproject raw depth-head outputs into the filtered cloud and dense per-pixel fields.

        - reads images (N, 3, H, W) in [0, 1], extrinsic (N, 3, 4) w2c, model-grid intrinsics (N, 3, 3)
        - reads depth (N, H, W) or (N, H, W, 1), depth_conf (N, H, W), optional keep-mask `mask`
        - keeps pixels with depth > 0 and in `mask`, or above the conf_percentile cutoff without one
        - the multiview filter, when on, is ANDed on top
        - intrinsics left None; PointcloudResult derives the full-res K
        """
        # Unpack the camera poses, intrinsics and depth, dropping any trailing depth channel
        extrinsic = raw_outputs["extrinsic"]
        intrinsic = raw_outputs["intrinsics"]
        extrinsic_4x4 = extrinsics_to_homogeneous(extrinsic)
        depth_conf = raw_outputs["depth_conf"]
        depth = raw_outputs["depth"]

        if depth.ndim == 4:
            depth = depth.squeeze(-1)

        model_h, model_w = int(depth.shape[1]), int(depth.shape[2])

        # Unproject the depth into a world point for every pixel, instead of using the backend's point maps
        world_points = unproject_frames(depth, extrinsic, intrinsic)

        # Keep pixels with positive depth that pass the backend's mask, or the confidence cutoff if it has none
        valid = depth > 0

        if "mask" in raw_outputs:
            valid &= raw_outputs["mask"]
        else:
            valid &= confidence_mask(depth_conf, self.conf_percentile)

        # Optionally also require the depth to agree across views
        valid &= self._multiview_mask(depth, intrinsic, extrinsic_4x4)

        # Collect the colors and pixel indices of the kept pixels
        images = torch.as_tensor(raw_outputs["images"]).cpu().float().numpy()
        colors_grid = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        pixel_indices = np.stack(np.where(valid), axis=1).astype(np.int32)

        assert self.image_paths is not None and self.original_coords is not None

        return PointcloudResult(
            points=world_points[valid].astype(np.float32),
            colors=colors_grid[valid],
            pixel_indices=pixel_indices,
            extrinsics=extrinsic_4x4,
            intrinsics=None,
            model_intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=torch.from_numpy(images),
            confidence=torch.from_numpy(depth_conf),
            world_points=world_points,
            depth=depth,
        )

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int
    ) -> dict[str, Any]:
        """
        Q/K at one cross-frame block, plus joint poses and geometry, for LC verification.

        - one _forward gives both, so _verify_loop_candidate needs no second pass
        - world points are the depth unprojected, as _postprocess builds them

        Args:
            frames: (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: cross-frame block to tap; -1 = last.

        Returns:
            {"q", "k": (B, heads, tokens, head_dim), "poses": (2, 4, 4) w2c,
            "world_points": (2, H, W, 3), "conf": (2, H, W)}.

        Raises:
            NotImplementedError: the backend does not support loop closure.
        """
        # Capture attention queries and keys at the chosen block during one forward pass
        attn = self._lc_attn(layer_index)

        with capture_qk(attn.qkv, attn.num_heads) as captured:
            raw = self._forward(self.model, frames)

        # Unproject each frame's depth into world points
        depth = raw["depth"].reshape(raw["depth"].shape[:3])
        world_points = unproject_frames(depth, raw["extrinsic"], raw["intrinsics"])

        # Return the poses and geometry alongside the captured queries and keys
        captured["poses"] = extrinsics_to_homogeneous(raw["extrinsic"])
        captured["world_points"] = world_points
        captured["conf"] = raw["depth_conf"]

        return captured

    def _lc_attn(self, layer_index: int) -> torch.nn.Module:
        """
        Attention module (with .qkv and .num_heads) that loop closure taps; the base has none.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support loop closure"
        )

    def _verify_loop_candidate(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        verify_match_ratio: float,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Accept or reject a loop candidate by cross-frame attention between two frames.

        - frame1, frame2: (C, H, W) preprocessed frames
        - returns (accepted, lc_data); lc_data is None on reject
        - lc_data: "poses" (2, 4, 4) w2c, "world_points" (2, H, W, 3), "conf" (2, H, W)
        """
        # Run both frames through the model and capture attention at the calibrated layer
        assert self._lc_layer_index is not None and self._lc_token_offset is not None
        device = next(self.model.parameters()).device
        features = self.extract_intermediate_features(
            torch.stack([frame1, frame2]).to(device), self._lc_layer_index
        )

        # Score the pair by the mean of the top quarter of per-token attention ratios
        ratios = cross_frame_attention_ratio(
            features["k"], features["q"], token_offset=self._lc_token_offset
        )
        thresh = np.percentile(
            ratios, 75
        )  # ignore background tokens that match nothing
        ratio = float(ratios[ratios >= thresh].mean())
        accepted = ratio >= verify_match_ratio
        logger.info(
            "LC verify: ratio=%.4f threshold=%.4f accepted=%s",
            ratio,
            verify_match_ratio,
            accepted,
        )

        if not accepted:
            return False, None

        # The pair is a loop, so return its poses and geometry
        return True, {
            "poses": features["poses"],
            "world_points": features["world_points"],
            "conf": features["conf"],
        }


########################################################################
# Helpers
########################################################################


@contextmanager
def capture_qk(
    qkv: torch.nn.Module, num_heads: int
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Capture q and k from a fused QKV projection for the duration of the block.

    - the hook is removed in a finally block, so a raising forward still cleans up

    Args:
        qkv: the Linear producing (B, T, 3 * heads * head_dim).
        num_heads: attention heads of that block.

    Yields:
        dict filled with "q", "k" as (B, heads, T, head_dim) after the forward runs.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: torch.nn.Module, _inp: Any, out: torch.Tensor) -> None:
        # Split the fused qkv output into separate per-head q, k and v tensors
        B, N, C3 = out.shape
        hd = (C3 // 3) // num_heads
        split = out.detach().reshape(B, N, 3, num_heads, hd).permute(2, 0, 3, 1, 4)
        captured["q"], captured["k"] = split[0], split[1]

    # Attach the hook and always remove it afterward, even if the forward pass fails
    handle = qkv.register_forward_hook(_hook)

    try:
        yield captured
    finally:
        handle.remove()


def center_crop_coords(
    orig_wh: tuple[int, int],
    resized_wh: tuple[int, int],
    crop_wh: tuple[int, int],
    scale: tuple[float, float],
) -> list[float]:
    """
    Centered crop of a resized frame, as a box in original pixels.

    - resize orig -> resized, then center-crop resized -> crop; the box undoes both
    - crop offset (resized - crop) // 2 in the resized grid, as the upstream loaders do

    Args:
        orig_wh: original frame (width, height).
        resized_wh: frame size after the resize, before the crop.
        crop_wh: model input size after the crop.
        scale: (sx, sy) resize factors, original -> resized.

    Returns:
        [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
    """
    ow, oh = orig_wh
    rw, rh = resized_wh
    cw, ch = crop_wh
    sx, sy = scale

    # Find the crop's top-left corner, then convert the box back to original pixels
    left, top = (rw - cw) // 2, (rh - ch) // 2

    return [left / sx, top / sy, (left + cw) / sx, (top + ch) / sy, ow, oh]


def _decode_depth_head(
    predictions: dict, hw: tuple[int, int], decode: Callable
) -> dict[str, np.ndarray]:
    """
    Decode a VGGT-family forward's poses and depth to CPU float32 arrays.

    - predictions: 'pose_enc', 'depth', 'depth_conf', each with batch dim 1
    - hw: model-res (H, W)
    - decode: the backend's pose decoder, (pose_enc, hw) -> (extrinsic, intrinsic)
    - returns 'extrinsic' (N, 3, 4) w2c and model-res 'intrinsics' (N, 3, 3)
    - returns 'depth' (N, H, W, 1) and 'depth_conf' (N, H, W)
    """
    ext_t, intr_t = decode(predictions["pose_enc"].detach(), hw)

    return {
        "extrinsic": ext_t.cpu().float().numpy().squeeze(0),
        "intrinsics": intr_t.cpu().float().numpy().squeeze(0),
        "depth": predictions["depth"].squeeze(0).cpu().float().numpy(),
        "depth_conf": predictions["depth_conf"].squeeze(0).cpu().float().numpy(),
    }


def _frame_sizes(
    paths: list[Path], arrays: list[np.ndarray] | None
) -> list[tuple[int, int]]:
    """
    Each frame's (w, h): from the handed-off arrays when given, else a header read per file.
    """
    if arrays is None:
        return [Image.open(p).size for p in paths]

    return [(a.shape[1], a.shape[0]) for a in arrays]
