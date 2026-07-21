"""VGGT-X feedforward backend: inference utilities and creator.

Provides:
  VGGTX_IMG_LOAD_RESOLUTION    — fixed inference resolution for VGGT-X
  unproject_and_filter_points  — depth → world-space point cloud with confidence filtering
  VGGTXCreator                 — feedforward creator using VGGT-X depth + pose estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image as PILImage
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from collab_splats.geometry.transforms import extrinsics_to_homogeneous

# Global alignment is parked — the call site below is disabled. Re-enable both when
# comparing VGGT-X native alignment against the LM bundle adjustment (bae-vggt-parity):
#   from collab_splats.geometry.global_alignment import run_global_alignment
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _decode_verify_geometry,
    _raw_to_world_points,
    compute_multiview_depth_confidence,
    console,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# VGGT-X target inference resolution (width, px). Matches upstream training default.
# load_and_preprocess_images(mode="crop") resizes width → this value, then center-crops
# height to the same value when height > target_size.
VGGTX_IMG_LOAD_RESOLUTION: int = 518


# ── Preprocessing helpers ──────────────────────────────────────────────────────


def _compute_vggtx_crop_coords(image_paths: list[Path], target_size: int = VGGTX_IMG_LOAD_RESOLUTION) -> np.ndarray:
    """Compute original_coords for VGGTX upstream crop mode.

    Upstream ``load_and_preprocess_images(mode="crop")`` resizes width→target_size then
    center-crops height to target_size when height > target_size.  This function computes
    the crop window in original-image pixel space so downstream consumers (TSDF RGB loader,
    COLMAP rescale) can invert the transform.

    Returns:
        (N, 6) float32 array ``[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`` per image.
        cr_x always equals orig_w (full width used).
        For landscape images (no height crop): tl_y=0, cr_y=orig_h.
        For portrait images (height cropped): tl_y and cr_y mark the kept strip.
    """
    coords = []
    for p in image_paths:
        with PILImage.open(p) as img:
            orig_w, orig_h = img.size  # PIL: (width, height)

        # Upstream: resize width → target_size, maintain AR; round height to div-by-14
        scale = target_size / orig_w
        new_h_raw = orig_h * scale
        new_h = round(new_h_raw / 14) * 14  # divisible-by-14 rounding used upstream

        if new_h > target_size:
            # Height crop applied — map crop boundaries back to original-image pixels
            start_y_resized = (new_h - target_size) // 2
            tl_y = start_y_resized / scale
            cr_y = (start_y_resized + target_size) / scale
        else:
            tl_y = 0.0
            cr_y = float(orig_h)

        coords.append([0.0, tl_y, float(orig_w), cr_y, float(orig_w), float(orig_h)])

    return np.array(coords, dtype=np.float32)


# ── Inference utilities ────────────────────────────────────────────────────────


def unproject_and_filter_points(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    images: Any,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    conf_threshold: float = 50.0,
    max_points: int = 500_000,
    extra_mask: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject depth to world-space points and filter by confidence.

    Args:
        depth:          (N, H, W, 1) float32 depth maps.
        depth_conf:     (N, H, W) float32 confidence maps.
        images:         (N, 3, H, W) tensor or array of preprocessed images.
        extrinsic:      (N, 3, 4) or (N, 4, 4) camera extrinsics.
        intrinsic:      (N, 3, 3) camera intrinsics.
        conf_threshold: Percentile cutoff (>1.0) or raw threshold (≤1.0).
                        Points below this confidence are discarded.
        max_points:     Maximum number of output points; excess are randomly subsampled.
        extra_mask:     Optional (N, H, W) boolean array; pixels where False are excluded
                        before subsampling (e.g. from compute_multiview_depth_confidence).

    Returns:
        pts3d:          (P, 3) float32 world-space points.
        colors:         (P, 3) uint8 RGB.
        pixel_indices:  (P, 3) int32 — [frame_id, row, col] source pixel for each point.
    """
    # Upstream `unproject_depth_map_to_point_map` returns points in the first-camera-anchored
    # world frame. If we ever adopt an SL(4) per-submap loop-closure layer (VGGT-SLAM-style),
    # this must switch to per-camera-local points — see ROADMAP "Future considerations".
    points3d = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    if hasattr(images, "cpu"):
        images_np = images.cpu().float().numpy()
    else:
        images_np = np.asarray(images, dtype=np.float32)
    colors_np = images_np.transpose(0, 2, 3, 1)

    # conf_threshold > 1.0 is treated as a percentile; ≤ 1.0 as a raw value
    if conf_threshold > 1.0:
        threshold_val = float(np.percentile(depth_conf, conf_threshold))
    else:
        threshold_val = float(conf_threshold)

    conf_mask = depth_conf >= threshold_val

    if extra_mask is not None:
        conf_mask = conf_mask & extra_mask

    n_true = int(conf_mask.sum())
    if n_true > max_points:
        conf_mask = randomly_limit_trues(conf_mask, max_points)

    pts_out = points3d[conf_mask].astype(np.float32)
    colors_out = (colors_np[conf_mask] * 255).astype(np.uint8)
    # np.where on the final conf_mask (after randomly_limit_trues applied) gives the
    # exact (frame_id, row, col) that produced each surviving point.
    pixel_indices = np.stack(np.where(conf_mask), axis=1).astype(np.int32)  # (P, 3)

    return pts_out, colors_out, pixel_indices


# ── Creator ───────────────────────────────────────────────────────────────────


@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-X feedforward pose + depth estimation.

    Uses VGGT-X (Video Grounded Gaussian Transformer) to jointly predict camera
    poses and per-frame depth maps.  Depth maps are then unprojected to a 3D
    point cloud.

    Attributes:
        camera_model:         pycolmap camera model. Defaults to
                              ``"SIMPLE_PINHOLE"`` because VGGT-X predicts a
                              single focal length (not separate fx/fy).
        model_name:           HuggingFace model ID loaded via
                              ``VGGT.from_pretrained``.
        chunk_size:           Attention chunk size for memory-efficient inference.
                              Reduce if OOM on long sequences.
        conf_threshold:       Depth confidence percentile cutoff (0–100).
                              Points whose confidence is below this percentile
                              are discarded.  35.0 = keep the top 65 %.
    """

    # LC verify calibration — chess d5 full-layer sweep, 2026-07-09.
    # 21 SLAM-confirmed positives vs 20 GT-clean negatives (camera centers
    # > half scene diameter apart AND viewing dirs > 90°, seed 42).
    # Layer 10 separates perfectly (AUC 1.000; positives min 1.2746, negatives
    # max 1.0658); threshold = midpoint 1.17 (±0.104 margin to both sides).
    # The base-class layer 20 does NOT discriminate for VGGT-X (AUC 0.42 vs
    # clean negatives). VGGTSPARKCreator overrides both (native similarity).
    _lc_layer_index: ClassVar[int] = 10
    default_verify_match_ratio: ClassVar[float] = 1.17

    camera_model: str = "SIMPLE_PINHOLE"
    model_name: str = "facebook/VGGT-1B"
    chunk_size: int = 256
    conf_threshold: float = 35.0
    use_multiview_confidence: bool = False
    mv_conf_threshold: float = 0.0

    def _load_model(self, device: str) -> Any:
        """Load VGGT-X from HuggingFace and move to device.

        Uses bfloat16 on Ampere+ GPUs (compute capability >= 8), float16 otherwise.

        Args:
            device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).

        Returns:
            VGGT model in eval mode on the requested device.
        """
        # Choose dtype based on GPU capability: bfloat16 for Ampere+, float16 for older
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        # Load pretrained model and move to device in eval mode
        model = VGGT.from_pretrained(self.model_name, chunk_size=self.chunk_size)
        model.eval()
        model = model.to(device, dtype=dtype)
        return model

    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        """Load and preprocess images using upstream VGGT crop mode.

        Resizes width to 518px then center-crops height to 518px when height > 518px.
        Matches the training preprocessing used by VGGT and VGGT-SLAM/SPARK.
        Stores the crop window in original-image pixel coordinates in ``original_coords``
        so the TSDF RGB loader and COLMAP rescale can invert the transform.

        Args:
            image_dir: Directory containing ``.png``/``.jpg``/``.jpeg`` images.

        Returns:
            (images, image_paths, original_coords) where original_coords is (N, 6)
            float32 ``[0, tl_y, orig_w, br_y, orig_w, orig_h]`` in original-image pixels.

        Raises:
            FileNotFoundError: If no supported images are found in image_dir.
        """
        # Collect and sort image paths; reject non-image extensions
        image_dir = Path(image_dir)
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Compute crop window in original-image pixel space for downstream consumers
        original_coords = _compute_vggtx_crop_coords(image_paths, VGGTX_IMG_LOAD_RESOLUTION)

        # Load and preprocess using upstream crop mode — matches VGGT training default
        image_names = [str(p) for p in image_paths]
        images = load_and_preprocess_images(image_names, mode="crop")

        return images, image_paths, original_coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        """Run VGGT-X on the preprocessed image tensor; return raw predictions dict.

        Args:
            model: Loaded VGGT model (from _load_model).
            views: (N, 3, H, W) float tensor from _preprocess.
            **kwargs: Unused; present for interface compatibility.

        Returns:
            dict with keys: ``images``, ``extrinsic``, ``intrinsics``,
            ``intrinsics_downsampled`` (alias of intrinsics), ``depth``, ``depth_conf``.
        """
        images = views
        device = next(model.parameters()).device
        device_type = device.type
        # Match aggregator's internal dtype selection: bf16 on Ampere+, fp16 otherwise.
        # Model params are fp32 but aggregator overrides dtype unconditionally at line 221;
        # passing fp32 images causes camera/register tokens to be cast to fp32 before that
        # override, making assembled tokens fp32 while aggregator asserts bf16 — assertion fails.
        dtype = (
            torch.bfloat16
            if device_type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )

        # Cast images to model dtype before forward; aggregator asserts tokens.dtype == dtype
        # at line 270 and token assembly happens before autocast can override it.
        images = images.to(device, dtype=dtype)

        # int() cast: guards pose_encoding_to_extri_intri against numpy-2 float32 scalars
        # being assigned into CUDA tensors. Lets us use upstream VGGT-X with no local patch.
        image_shape = tuple(int(x) for x in images.shape[-2:])

        # bf16/f16 autocast scoped to model forward only; downstream numpy ops need float32.
        with torch.no_grad():
            with torch.autocast(device_type, dtype=dtype):
                predictions = model(images.unsqueeze(0))

        # Decode pose encoding at model resolution only — matches VGGT-SLAM upstream.
        # Original-res decode removed: it fed wrong K to the BA wrapper via raw["intrinsics"].
        extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(predictions["pose_enc"], image_shape)

        # Move predictions to CPU float32 for downstream processing
        extrinsic = extrinsic_t.cpu().float().numpy().squeeze(0)  # (N, 3, 4)
        intrinsic = intrinsic_t.cpu().float().numpy().squeeze(0)  # (N, 3, 3) model-res
        depth_map = predictions["depth"].squeeze(0).cpu().float().numpy()
        depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()

        return {
            "images": images,
            "extrinsic": extrinsic,
            "intrinsics": intrinsic,  # model-res K
            "intrinsics_downsampled": intrinsic,  # alias — _raw_to_world_points expects this key
            "depth": depth_map,
            "depth_conf": depth_conf,
        }

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        """Unproject depth maps to world-space points and build FeedforwardResult.

        Optionally runs global alignment to refine poses.  Lifts semantic features
        if ``extractor_name`` is set.  Populates BA fields (world_points, conf, images)
        so the BundleAdjustment wrapper can refine poses.

        Args:
            raw_outputs: Dict from _forward containing depth, extrinsics, images.
            **kwargs:    Unused.

        Returns:
            FeedforwardResult with pts3d, colors, extrinsics, and BA fields populated.
        """
        extrinsic = raw_outputs["extrinsic"]
        intrinsic = raw_outputs.get("intrinsics_downsampled", raw_outputs.get("intrinsics"))

        # Global alignment (feature matching + joint BA) is parked. Re-enable with the
        # import at the top of this file to compare against LM BA (bae-vggt-parity):
        # extrinsic, intrinsic = run_global_alignment(
        #     raw_outputs, extrinsic, intrinsic, self.image_paths,
        # )

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
                abs_thresh=0.0,
                rel_thresh=0.05,
            )
            mv_mask = mv_conf > self.mv_conf_threshold

        # Unproject depth maps to filtered world-space points and per-point colors
        pts3d, colors, pixel_indices = unproject_and_filter_points(
            depth=raw_outputs["depth"],
            depth_conf=raw_outputs["depth_conf"],
            images=raw_outputs["images"],
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            conf_threshold=self.conf_threshold,
            max_points=self.max_points,
            extra_mask=mv_mask,
        )

        # Model spatial dimensions used to reshape world-point grid for BA
        model_h = int(raw_outputs["depth"].shape[1])
        model_w = int(raw_outputs["depth"].shape[2])

        # Populate BA fields: subsampled world-point grid for track extraction.
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
        if world_pts_flat is not None:
            world_points = world_pts_flat.reshape(world_pts_flat.shape[0], model_h, model_w, 3)
        else:
            world_points = None
        # Populate BA fields: depth confidence and preprocessed images
        conf = torch.from_numpy(raw_outputs["depth_conf"])
        images = raw_outputs["images"]

        extrinsic_4x4 = extrinsics_to_homogeneous(extrinsic)

        # LC merged outputs carry the deduped global poses so that
        # FeedforwardResult.extrinsics has exactly one entry per input frame,
        # not per submap window frame (which includes overlapping frames).
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
        """Re-derive world-space points using bundle-adjusted camera poses.

        Called by the BundleAdjustment wrapper after refining extrinsics.
        Re-runs depth unprojection under the new poses so pts3d stay consistent
        with the refined camera geometry.

        Args:
            raw_outputs:     Raw predictions dict from _forward.
            extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices.
            intrinsics:      (N, 3, 3) refined camera intrinsics.

        Returns:
            (pts3d, colors) — (P, 3) float32 and (P, 3) uint8.
        """
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
        """Hook aggregator.global_blocks[layer_index].attn.qkv; return {q, k, poses, world_points, conf}.

        Runs a 2-frame VGGT-X forward with a per-call hook on the QKV projection of the
        specified global attention block.  Captures q and k tensors, then decodes the
        VGGT-X pose encoding to fresh (2, 4, 4) camera extrinsics — so
        _verify_loop_candidate gets accurate relative poses without a second forward pass.

        The hook is removed in a finally block — guaranteed cleanup even if the forward
        raises.  No persistent state is left on the model or its layers.

        Args:
            frames:      (2, C, H, W) preprocessed frames (float16/32 on CPU or GPU).
            layer_index: Which global attention block to tap.  -1 = last (default,
                         matches VGGT-SPARK).  Valid range: [-len(blocks), len(blocks)-1].
            **kwargs:    Unused (kept for interface compatibility with MapAnything).

        Returns:
            dict with keys:
              "q":     (B, heads, N_tokens, head_dim) query projections
              "k":     (B, heads, N_tokens, head_dim) key projections
              "poses": (2, 4, 4) float32 np.ndarray — decoded camera extrinsics
              "world_points": (2, H, W, 3) float32 np.ndarray — unprojected depth
              "conf": (2, H, W) float32 np.ndarray — depth confidence
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        # Add batch dimension; move to model device + dtype for the forward pass
        batch = frames.unsqueeze(0).to(device, dtype=dtype)

        # Register a per-call hook on the QKV projection of the chosen block
        block = self.model.aggregator.global_blocks[layer_index]
        C_nh = block.attn.num_heads
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, _inp, out):
            # out: (B, N, 3*C) — split into q/k/v, reshape to (B, heads, N, head_dim)
            B, N, C3 = out.shape
            hd = (C3 // 3) // C_nh
            qkv = out.detach().reshape(B, N, 3, C_nh, hd).permute(2, 0, 3, 1, 4)
            captured["q"], captured["k"] = qkv[0], qkv[1]

        hook = block.attn.qkv.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                predictions = self.model(batch)
        finally:
            # Always remove the hook — no persistent state left on the model
            hook.remove()

        # Decode camera extrinsics + intrinsics from VGGT-X pose encoding
        image_shape = (int(frames.shape[-2]), int(frames.shape[-1]))
        ext_t, intr_t = pose_encoding_to_extri_intri(predictions["pose_enc"].detach(), image_shape)
        ext_3x4 = ext_t.cpu().float().numpy().squeeze(0)  # (2, 3, 4) w2c
        intrinsic = intr_t.cpu().float().numpy().squeeze(0)  # (2, 3, 3)
        captured["poses"] = extrinsics_to_homogeneous(ext_3x4)  # (2, 4, 4)

        # Decode geometry from the SAME forward — shared verify-geometry helper
        captured["world_points"], captured["conf"] = _decode_verify_geometry(
            predictions["depth"], predictions["depth_conf"], ext_3x4, intrinsic
        )
        return captured
