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
from typing import Any, ClassVar

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from collab_splats.geometry.transforms import extrinsics_to_homogeneous

from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _decode_verify_geometry,
    _raw_to_world_points,
    compute_multiview_depth_confidence,
    frames_as_pil_source,
    multiview_mask,
)
from .vggtx import unproject_and_filter_points

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


def _compute_omega_original_coords(sizes: list[tuple[int, int]]) -> np.ndarray:
    """Compute original_coords after Omega's center-crop aspect-ratio enforcement.

    Mirrors the crop logic in vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio
    so that _rescale_reconstruction_to_original_dimensions can invert the transform.

    Args:
        sizes: per-image ``(orig_w, orig_h)`` original-image dimensions.

    Returns:
        (N, 6) float32 array [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h] per image.
    """
    # Must match vggt_omega.utils.load_fn._crop_to_supported_aspect_ratio exactly
    _MIN_AR = 0.5
    _MAX_AR = 2.0

    coords = []
    for orig_w, orig_h in sizes:
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
        camera_model:          pycolmap camera model.  Defaults to ``"PINHOLE"`` because
                               Omega predicts separate fx/fy via FoV encoding.
        model_path:            Local path to a ``vggt_omega_1b_512.pt`` checkpoint.
                               ``None`` → auto-download from HuggingFace on first run.
        model_repo:            HuggingFace repo ID for checkpoint download.
        model_filename:        Checkpoint filename to download from ``model_repo``.
        resolution:            Target image resolution.  ``None`` → auto (512 for the
                               standard checkpoint, 256 for text-aligned).  Explicit
                               values pass through unchanged regardless of
                               ``enable_text_alignment``.
        resize_mode:           Image resize strategy passed as ``mode=`` to
                               ``load_and_preprocess_images``.  ``"balanced"`` (default):
                               smart crop/pad preserving aspect ratio.  ``"max_size"``:
                               resize longest side to ``resolution``, no crop.
        conf_threshold:        Depth confidence percentile cutoff (0–100).  Points
                               below this percentile are discarded.  50.0 = top 50%.
        enable_text_alignment: Load the text-aligned checkpoint variant via
                               ``VGGTOmega(enable_alignment=True)``.  Auto-sets
                               ``resolution=256`` when ``resolution`` is ``None``.
    """

    # LC verify calibration — chess d5 clean-negative sweep, 2026-07-10.
    # 21 SLAM-confirmed positives vs 20 GT-clean negatives (camera centers
    # > half scene diameter apart AND viewing dirs > 90°, seed 42), all 24
    # inter_frame_blocks hooked in one forward per pair. Layer 13 separates
    # perfectly (AUC 1.000; positives min 1.8385, negatives max 1.2620);
    # threshold = midpoint 1.55. Old layer 16 / 1.16 had AUC 0.824 and
    # rejected 6/21 true loops while passing 6/20 clean negatives.
    # Note: production token_offset=5 (inherited) technically miscounts
    # Omega's 17 special tokens (1 camera + 16 register; patch-16 backbone),
    # but the offset-17 re-sweep is near-identical at layer 13 (mid 1.5505),
    # so the inherited offset is kept. max_jump_ratio=0.3 enables geometric
    # sanity check (default inf disables it) for repetitive chess-texture scenes.
    _lc_layer_index: ClassVar[int] = 13
    default_verify_match_ratio: ClassVar[float] = 1.55
    default_max_jump_ratio: ClassVar[float] = 0.3

    camera_model: str = "PINHOLE"
    model_path: str | None = None
    model_repo: str = VGGT_OMEGA_HF_REPO
    model_filename: str = VGGT_OMEGA_DEFAULT_FILENAME
    resolution: int | None = None  # None → auto (512 standard, 256 text-aligned); explicit overrides
    resize_mode: str = "balanced"  # mode= passed to load_and_preprocess_images
    conf_threshold: float = 50.0
    use_multiview_confidence: bool = False
    # min_views: "at least K other views agree". K=1 is the old mv_conf_threshold=0.0.
    min_views: int = 1
    # abs_thresh stays 0.0 — VGGT depth is non-metric, so a fixed-unit tolerance is
    # meaningless and would break the scale invariance the shared function relies on.
    mv_conf_abs_thresh: float = 0.0
    mv_conf_rel_thresh: float = 0.05
    enable_text_alignment: bool = False  # VGGTOmega(enable_alignment=True); sets resolution=256 when None

    def __post_init__(self) -> None:
        if self.resize_mode not in {"balanced", "max_size"}:
            raise ValueError(f"resize_mode must be one of {{'balanced', 'max_size'}}, got {self.resize_mode!r}")
        if self.resolution is None:
            self.resolution = 256 if self.enable_text_alignment else 512
            logger.debug("VGGTOmegaCreator: resolved resolution=%d", self.resolution)

    def _load_model(self, device: str) -> Any:
        """Load VGGT-Omega from local path or HuggingFace, move to device in fp32.

        Params stay float32 on purpose. VGGTOmega runs its aggregator under an internal
        autocast (bf16/fp16) but disables autocast for CameraHead/DenseHead, which cast
        their inputs to fp32. Casting params to bf16 mismatches those fp32 head inputs and
        crashes the head LayerNorms ("expected scalar type Float but found BFloat16").
        """
        # Resolve checkpoint path — local file or download from HuggingFace
        if self.model_path is not None:
            ckpt_path = Path(self.model_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            ckpt_path = Path(
                hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_filename,
                )
            )

        # Instantiate model, load checkpoint weights, move to device in eval mode (fp32 params)
        model = VGGTOmega(enable_alignment=self.enable_text_alignment)
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()
        model = model.to(device)
        return model

    def _preprocess(self, frames: Any, frame_idxs: list[int]) -> tuple[Any, list[Path], np.ndarray]:
        """Preprocess in-memory frames into VGGT-Omega input format."""
        # Stable synthetic labels — no files on disk; the store/decoder is the sole IO path
        image_paths = [Path(f"frame_{idx:06d}") for idx in frame_idxs]

        # Compute crop transform for each image from its own dims (replicated from Omega's load_fn)
        original_coords = _compute_omega_original_coords([(int(f.shape[1]), int(f.shape[0])) for f in frames])

        # Run Omega's resize in-memory (bit-identical to path load); .png names satisfy
        # loaders' extension checks while PIL.Image.open is intercepted.
        loader_names = [f"{p.name}.png" for p in image_paths]
        with frames_as_pil_source(frames):
            images = load_and_preprocess_images(loader_names, image_resolution=self.resolution, mode=self.resize_mode)

        return images, image_paths, original_coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run VGGT-Omega on the preprocessed image tensor; return raw predictions dict."""
        device = next(model.parameters()).device
        image_shape = views.shape[-2:]  # (H_model, W_model)

        # Move images to model device; VGGTOmega adds the batch dim internally
        images = views.to(device)

        # Model forward handles bf16/f16 autocast internally
        with torch.no_grad():
            predictions = model(images)

        # Decode poses at model resolution only — matches upstream demo_gradio.run_model.
        # Original-res decode removed: no downstream consumer requires it and it was the
        # root cause of cx > model_W in result.intrinsics.
        ext, intr = encoding_to_camera(predictions["pose_enc"], image_shape)

        # Move to CPU float32 for downstream numpy ops; squeeze the batch dim (always 1)
        extrinsic = ext.cpu().float().numpy().squeeze(0)  # (N, 3, 4)
        intrinsic = intr.cpu().float().numpy().squeeze(0)  # (N, 3, 3) at model-res
        depth = predictions["depth"].squeeze(0).cpu().float().numpy()  # (N, H, W, 1)
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()  # (N, H, W)

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,  # model-res K
            "intrinsics_downsampled": intrinsic,  # alias — _raw_to_world_points expects this key
            "depth": depth,
            "depth_conf": depth_conf,
        }

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """Unproject depth maps to world-space points and build FeedforwardResult."""
        extrinsic = raw_outputs["extrinsic"]  # (N, 3, 4) at model resolution
        intrinsic = raw_outputs["intrinsics"]  # (N, 3, 3) at model resolution

        # Optionally compute geometric cross-view depth consistency mask
        mv_mask = None
        if self.use_multiview_confidence:
            depth_np = raw_outputs["depth"]
            if depth_np.ndim == 4:
                depth_np = depth_np.squeeze(-1)  # (N, H, W)
            extr_4x4 = extrinsics_to_homogeneous(extrinsic)
            mv_conf = compute_multiview_depth_confidence(
                depth_np,
                intrinsic,
                extr_4x4,
                abs_thresh=self.mv_conf_abs_thresh,
                rel_thresh=self.mv_conf_rel_thresh,
            )
            mv_mask = multiview_mask(mv_conf, depth_np > 0, min_views=self.min_views)

        # Unproject depth maps to filtered world-space points and per-point colors
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=raw_outputs["intrinsics_downsampled"],
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
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
            points=pts3d,
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
            confidence=conf,
            world_points=world_points,
            depth=raw_outputs["depth"].squeeze(-1) if raw_outputs["depth"].ndim == 4 else raw_outputs["depth"],
        )

    def _reproject(
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
            max_points=self.max_points,
        )
        return pts3d, colors  # pixel_indices unused; post-BA uses stored indices

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Hook inter_frame_blocks[layer_index].attn.qkv; return {q, k, poses, world_points, conf}."""
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

        # Decode camera extrinsics + intrinsics from Omega pose encoding
        image_shape = (frames.shape[-2], frames.shape[-1])
        ext_t, intr_t = encoding_to_camera(predictions["pose_enc"].detach(), image_shape)
        ext_3x4 = ext_t.cpu().float().numpy().squeeze(0)  # (2, 3, 4) w2c
        intrinsic = intr_t.cpu().float().numpy().squeeze(0)  # (2, 3, 3)
        captured["poses"] = extrinsics_to_homogeneous(ext_3x4)  # (2, 4, 4)

        # Decode geometry from the SAME forward — shared verify-geometry helper
        captured["world_points"], captured["conf"] = _decode_verify_geometry(
            predictions["depth"], predictions["depth_conf"], ext_3x4, intrinsic
        )
        return captured
