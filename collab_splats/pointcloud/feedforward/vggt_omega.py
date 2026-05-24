"""VGGT-Omega feedforward backend: inference utilities and creator.

Provides:
  VGGT_OMEGA_HF_REPO            — default HuggingFace repo for checkpoint download
  VGGT_OMEGA_DEFAULT_FILENAME   — default checkpoint filename (512-res)
  VGGT_OMEGA_DEFAULT_RESOLUTION — default image resolution for inference
  _compute_omega_original_coords — compute original_coords for Omega's center-crop transform
  VGGTOmegaCreator              — feedforward creator using VGGT-Omega depth + pose estimation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _raw_to_world_points,
)
from .vggtx import unproject_and_filter_points
from collab_splats.utils.geometry import extrinsics_to_homogeneous

logger = logging.getLogger(__name__)

########################################################################
########## Constants ###################################################
########################################################################

VGGT_OMEGA_HF_REPO = "facebook/VGGT-Omega"
VGGT_OMEGA_DEFAULT_FILENAME = "vggt_omega_1b_512.pt"
VGGT_OMEGA_DEFAULT_RESOLUTION = 512

########################################################################
########## Inference utilities #########################################
########################################################################

def _compute_omega_original_coords(image_paths: list[Path]) -> np.ndarray:
    """Compute original_coords after Omega's center-crop aspect-ratio enforcement.

    Mirrors the crop logic in vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio
    so that _rescale_reconstruction_to_original_dimensions can invert the transform.

    Returns:
        (N, 6) float32 array [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h] per image.
    """
    # Must match vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio exactly
    _MIN_AR = 0.5
    _MAX_AR = 2.0

    coords = []
    for p in image_paths:
        # Read image dimensions without decoding pixels
        with Image.open(p) as img:
            orig_w, orig_h = img.size
        ar = orig_h / max(orig_w, 1)

        # Default: no crop
        tl_x, tl_y = 0.0, 0.0
        cr_x, cr_y = float(orig_w), float(orig_h)

        # Center-crop height for tall images (AR > _MAX_AR)
        if ar > _MAX_AR:
            crop_h = orig_w * _MAX_AR
            tl_y = (orig_h - crop_h) / 2
            cr_y = tl_y + crop_h

        # Center-crop width for wide images (AR < _MIN_AR)
        elif ar < _MIN_AR:
            crop_w = orig_h / _MIN_AR
            tl_x = (orig_w - crop_w) / 2
            cr_x = tl_x + crop_w

        coords.append([tl_x, tl_y, cr_x, cr_y, float(orig_w), float(orig_h)])

    return np.array(coords, dtype=np.float32)


########################################################################
########## Creator #####################################################
########################################################################

@dataclass
class VGGTOmegaCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-Omega feedforward pose + depth estimation.

    Uses VGGT-Omega to jointly predict camera poses and per-frame depth maps,
    which are then unprojected to a 3D point cloud.  Supports BundleAdjustment
    and LoopClosure wrappers via the standard BaseFeedforwardCreator interface.

    Attributes:
        camera_model:    pycolmap camera model.  Defaults to ``"PINHOLE"`` because
                         Omega predicts separate fx/fy via FoV encoding.
        model_path:      Local path to a ``vggt_omega_1b_512.pt`` checkpoint.
                         ``None`` → auto-download from HuggingFace on first run.
        model_repo:      HuggingFace repo ID for checkpoint download.
        model_filename:  Checkpoint filename to download from ``model_repo``.
        image_resolution: Target resolution for ``load_and_preprocess_images``.
                          512 for the standard checkpoint, 256 for text-aligned.
        conf_threshold:  Depth confidence percentile cutoff (0–100).  Points
                         below this percentile are discarded.  50.0 = top 50%.
    """

    camera_model: str = "PINHOLE"
    model_path: str | None = None
    model_repo: str = VGGT_OMEGA_HF_REPO
    model_filename: str = VGGT_OMEGA_DEFAULT_FILENAME
    image_resolution: int = VGGT_OMEGA_DEFAULT_RESOLUTION
    conf_threshold: float = 50.0

    def _load_model(self, device: str) -> Any:
        """Load VGGT-Omega from local path or HuggingFace, move to device."""
        # Choose dtype based on GPU capability: bfloat16 for Ampere+, float16 for older
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Resolve checkpoint path — local file or download from HuggingFace
        if self.model_path is not None:
            ckpt_path = Path(self.model_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            ckpt_path = Path(hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_filename,
            ))

        # Instantiate model, load checkpoint weights, move to device in eval mode
        model = VGGTOmega()
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()
        model = model.to(device, dtype=dtype)
        return model

    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        """Load and preprocess images from directory into VGGT-Omega input format."""
        # Collect and sort image paths; reject non-image extensions
        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Compute crop transform for each image (replicated from Omega's load_fn)
        original_coords = _compute_omega_original_coords(image_paths)

        # Load and preprocess images to model resolution via Omega's balanced resize
        image_names = [str(p) for p in image_paths]
        images = load_and_preprocess_images(image_names, image_resolution=self.image_resolution)

        return images, image_paths, original_coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run VGGT-Omega on the preprocessed image tensor; return raw predictions dict."""
        device = next(model.parameters()).device
        image_shape = views.shape[-2:]  # (H_model, W_model)
        orig_w, orig_h = self.original_coords[0, -2:]

        # Move images to model device; VGGTOmega adds the batch dim internally
        images = views.to(device)

        # Model forward handles bf16/f16 autocast internally
        with torch.no_grad():
            predictions = model(images)

        # Decode poses at model resolution (for BA track extraction)
        ext_ds, intr_ds = encoding_to_camera(predictions["pose_enc"], image_shape)
        # Decode poses at original image resolution (for final COLMAP output)
        ext, intr = encoding_to_camera(predictions["pose_enc"], (int(orig_h), int(orig_w)))

        # Move to CPU float32 for downstream numpy ops; squeeze the batch dim (always 1)
        extrinsic = ext.cpu().float().numpy().squeeze(0)         # (N, 3, 4)
        intrinsic = intr.cpu().float().numpy().squeeze(0)        # (N, 3, 3)
        intrinsic_ds = intr_ds.cpu().float().numpy().squeeze(0)  # (N, 3, 3)
        depth = predictions["depth"].squeeze(0).cpu().float().numpy()       # (N, H, W)
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()  # (N, H, W)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,
            "intrinsics_downsampled": intrinsic_ds,
            "depth": depth,
            "depth_conf": depth_conf,
        }

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """Unproject depth maps to world-space points and build FeedforwardResult."""
        extrinsic = raw_outputs["extrinsic"]   # (N, 3, 4) at original resolution
        intrinsic = raw_outputs["intrinsics"]  # (N, 3, 3) at original resolution

        # Unproject depth maps to filtered world-space points and per-point colors
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
        )

        # Resolve model spatial dimensions; handle (N, H, W, 1) and (N, H, W) depth formats
        depth = raw_outputs["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(-1)
        model_h, model_w = int(depth.shape[1]), int(depth.shape[2])

        # Populate BA fields: subsampled world-point grid for track extraction
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
        if world_pts_flat is not None:
            world_points = world_pts_flat.reshape(world_pts_flat.shape[0], model_h, model_w, 3)
        else:
            world_points = None

        # Populate BA fields: depth confidence map and preprocessed images
        conf = torch.from_numpy(raw_outputs["depth_conf"])
        images = raw_outputs["images"]

        extrinsic_4x4 = extrinsics_to_homogeneous(extrinsic)

        # LC merged outputs carry deduped global poses — one entry per input frame
        extrinsic_4x4_out = raw_outputs.get("extrinsic_global_4x4", extrinsic_4x4)

        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            features=None,
            extrinsics=extrinsic_4x4_out,
            intrinsics=intrinsic,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=images,
            conf=conf,
            world_points=world_points,
            depth=raw_outputs["depth"].squeeze(-1) if raw_outputs["depth"].ndim == 4 else raw_outputs["depth"],
        )

    def _reproject_ba(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses."""
        # Re-run depth unprojection with refined extrinsics and intrinsics
        pts3d, colors, _pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsics_3x4,
            intrinsic=intrinsics,
            conf_threshold=self.conf_threshold,
        )
        return pts3d, colors  # pixel_indices unused; post-BA uses stored indices

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Hook inter_frame_blocks[layer_index].attn.qkv; return {q, k, poses}."""
        device = next(self.model.parameters()).device

        # VGGTOmega manages bf16/f16 autocast internally — device-only cast matches _forward
        images = frames.to(device)

        # Register per-call hook on inter-frame attention block's QKV projection
        block = self.model.aggregator.inter_frame_blocks[layer_index]
        C_nh = block.attn.num_heads
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, _inp, out: torch.Tensor) -> None:
            B, N, C3 = out.shape
            hd = (C3 // 3) // C_nh
            qkv = out.detach().reshape(B, N, 3, C_nh, hd).permute(2, 0, 3, 1, 4)
            captured["q"], captured["k"] = qkv[0], qkv[1]

        hook = block.attn.qkv.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                predictions = self.model(images)
        finally:
            # Always remove the hook — no persistent state left on the model
            hook.remove()

        # Decode (2, 4, 4) camera extrinsics from Omega pose encoding
        image_shape = (frames.shape[-2], frames.shape[-1])
        ext_3x4, _ = encoding_to_camera(
            predictions["pose_enc"].detach(), image_shape
        )
        ext_3x4 = ext_3x4.cpu().float().numpy().squeeze(0)  # (2, 3, 4)
        captured["poses"] = extrinsics_to_homogeneous(ext_3x4)  # (2, 4, 4)
        return captured
