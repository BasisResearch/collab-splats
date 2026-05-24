"""MapAnything feedforward backend: inference utilities and creator.

Provides:
  MapAnythingCreator — feedforward creator using MapAnything depth + pose estimation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from vggt.utils.helper import randomly_limit_trues

# timm 0.6.x compat: uniception (mapanything dep) imports `from timm.layers import DropPath`
# which does not exist in timm<0.9. Re-export it from timm.models.layers before the import.
import timm.layers as _tl
import timm.models.layers as _tml
if not hasattr(_tl, "DropPath"):
    _tl.DropPath = _tml.DropPath
del _tl, _tml

from mapanything.models import MapAnything
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)

from .base import BaseFeedforwardCreator, FeedforwardResult, console


# ── Inference utilities ────────────────────────────────────────────────────────

def _reproject_mapanything(
    raw_outputs: list[dict],
    extrinsics_3x4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-project MapAnything camera-frame points to world frame using given extrinsics.

    Args:
        raw_outputs:    list[dict] from _forward(); each dict contains pts3d_cam,
                        mask, depth_z, img_no_norm (float32, unmasked).
        extrinsics_3x4: (N, 3, 4) world-to-camera matrices.

    Returns:
        (pts3d, colors) — (P, 3) float32 world-space points and (P, 3) uint8 RGB.
    """
    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for i, pred in enumerate(raw_outputs):
        # Extract camera-frame points and validity components
        pts3d_cam = pred["pts3d_cam"][0].cpu().numpy()                  # (H, W, 3)
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)   # (H, W)
        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()          # (H, W)

        # Combine validity mask with positive-depth check
        combined_mask = mask & (depth_z > 0)

        # Refined world2cam → cam2world for re-projection into world frame
        ext_4x4 = extrinsics_to_homogeneous(extrinsics_3x4[i])   # (4, 4)
        cam2world = invert_poses(ext_4x4)                          # (4, 4)

        # Apply mask and transform camera-frame points to world frame
        pts_flat = pts3d_cam[combined_mask]                                       # (K, 3)
        pts_world = (cam2world[:3, :3] @ pts_flat.T + cam2world[:3, 3:]).T       # (K, 3)

        # Extract colors for surviving pixels
        img_no_norm = pred["img_no_norm"][0].cpu().numpy()                        # (H, W, 3)
        colors = (img_no_norm[combined_mask] * 255).astype(np.uint8)              # (K, 3)

        all_pts.append(pts_world.astype(np.float32))
        all_colors.append(colors)

    return np.concatenate(all_pts, axis=0), np.concatenate(all_colors, axis=0)


# ── Creator ───────────────────────────────────────────────────────────────────

@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    """Pointcloud via MapAnything feedforward depth + pose estimation.

    Uses Facebook's MapAnything model to jointly predict per-image depth maps
    and camera poses in a single forward pass without requiring any SfM.

    Attributes:
        model_name:               HuggingFace model ID to load via
                                  ``MapAnything.from_pretrained``.
        confidence_percentile:    Mask out pixels whose multiview confidence
                                  score falls below this percentile (0–100).
                                  Higher = more aggressive masking, fewer points.
        use_multiview_confidence: When True, uses cross-view consistency scores
                                  to mask unreliable depth predictions.
                                  Set False to keep all pixels regardless of
                                  inter-frame agreement.
        minibatch_size:           Number of images processed per inference step.
                                  Reduce if running out of GPU memory.
    """

    model_name: str = "facebook/map-anything"
    confidence_percentile: float = 35.0
    use_multiview_confidence: bool = True
    minibatch_size: int = 1
    _processed_views: Any = field(default=None, init=False, repr=False)

    def _load_model(self, device: str) -> Any:
        # Load pretrained model, move to device, set eval mode
        model = MapAnything.from_pretrained(self.model_name)
        model = model.to(device)
        model.eval()
        return model

    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        # Collect and sort image paths; reject non-image extensions
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix in exts)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Load images via MapAnything's loader; derive model resolution from first image
        views = load_images([str(p) for p in image_paths])
        model_h: int = views[0]["img"].shape[-2]
        model_w: int = views[0]["img"].shape[-1]

        # Read original image dimensions; open each file once to avoid double I/O
        original_coords = np.array(
            [
                [0, 0, model_w, model_h, w, h]
                for p in image_paths
                for img in [PILImage.open(p)]
                for w, h in [(img.width, img.height)]
            ],
            dtype=np.float32,
        )

        # Validate views meet MapAnything input requirements, then convert to the
        # internal format model.forward() expects (ray directions, metric scale, etc.).
        # Kept on CPU here; transferred to model device in _forward.
        validated = validate_input_views_for_inference(views)
        self._processed_views = preprocess_input_views_for_inference(validated)

        return views, image_paths, original_coords

    def _forward(self, model: Any, views: Any, **kwargs: Any) -> list[dict]:
        console.log(f"  → {len(self._processed_views)} images, minibatch_size={self.minibatch_size}")
        device = next(model.parameters()).device
        device_type = device.type

        # Transfer preprocessed views to model device; kept on CPU in _preprocess
        # to avoid holding GPU memory during image loading and validation.
        for view in self._processed_views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device)

        # bf16 autocast scoped to model forward only; postprocessing requires float32
        # to avoid F.grid_sample dtype mismatch (torch 2.4 enforces strict matching).
        with torch.no_grad():
            with torch.autocast(device_type, dtype=torch.bfloat16,
                                 enabled=(device_type == "cuda")):
                return model.forward(
                    self._processed_views,
                    memory_efficient_inference=True,
                    minibatch_size=self.minibatch_size,
                )

    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        model_h: int = self._processed_views[0]["img"].shape[-2]
        model_w: int = self._processed_views[0]["img"].shape[-1]

        # Cast bf16 tensors to float32 before postprocessing. model.forward() runs
        # under bf16 autocast; postprocess_model_outputs_for_inference calls
        # F.grid_sample which requires matching dtypes (torch 2.4 strict enforcement).
        for pred in raw_outputs:
            pred["pts3d_cam"] = pred["pts3d_cam"].float()
            pred["pts3d"] = pred["pts3d"].float()

        # Apply confidence and edge masking via MapAnything's postprocess utility
        processed = postprocess_model_outputs_for_inference(
            raw_outputs,
            self._processed_views,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            use_multiview_confidence=self.use_multiview_confidence,
            confidence_percentile=self.confidence_percentile,
        )

        # Build per-frame masks + point/color grids in one pass — mirrors VGGTX conf_mask pattern.
        # postprocess_model_outputs_for_inference already baked confidence + edge masking into
        # pred["mask"], so combined_mask = mask & (depth_z > 0) is the full validity mask.
        masks, pts3d_grid, colors_grid = [], [], []
        images_list, conf_list, depth_list = [], [], []
        extrinsics_list, intrinsics_list = [], []

        for pred in processed:
            m = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)       # (H, W)
            dz = pred["depth_z"][0].squeeze(-1).cpu().numpy()                # (H, W)
            masks.append(m & (dz > 0))
            depth_list.append(dz)
            pts3d_grid.append(pred["pts3d"][0].cpu().numpy())                # (H, W, 3)
            colors_grid.append(
                (pred["img_no_norm"][0].cpu().numpy() * 255).astype(np.uint8)
            )                                                                  # (H, W, 3)
            images_list.append(pred["img_no_norm"][0].cpu().permute(2, 0, 1))  # (C, H, W)
            if pred.get("conf") is not None:
                c = pred["conf"][0]
                conf_list.append(c[0] if c.ndim == 3 else c)
            cam2world = pred["camera_poses"][0].cpu().numpy()
            extrinsics_list.append(invert_poses(cam2world)[:3, :4])
            intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())

        combined_mask = np.stack(masks)           # (N, H, W) bool
        stacked_pts3d = np.stack(pts3d_grid)      # (N, H, W, 3)
        stacked_colors = np.stack(colors_grid)    # (N, H, W, 3)

        # Apply cross-frame random subsampling — same as VGGTX randomly_limit_trues on conf_mask
        if int(combined_mask.sum()) > self.max_points:
            combined_mask = randomly_limit_trues(combined_mask, self.max_points)

        pts3d = stacked_pts3d[combined_mask].astype(np.float32)
        colors = stacked_colors[combined_mask]
        pixel_indices = np.stack(np.where(combined_mask), axis=1).astype(np.int32)  # (P, 3)

        _world_points = stacked_pts3d                        # full (N, H, W, 3) grid for BA
        _images = torch.stack(images_list)                   # (N, C, H, W)
        _conf = torch.stack(conf_list) if conf_list else None
        _depth = np.stack(depth_list).astype(np.float32)    # (N, H, W) depth_z values
        extrinsics = np.stack(extrinsics_list)               # (N, 3, 4)
        intrinsics = np.stack(intrinsics_list)               # (N, 3, 3)

        # Convert extrinsics to 4×4 homogeneous form
        extrinsics_4x4 = extrinsics_to_homogeneous(extrinsics)

        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            pixel_indices=pixel_indices,
            extrinsics=extrinsics_4x4,
            intrinsics=intrinsics,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=_images,
            conf=_conf,
            world_points=_world_points,
            depth=_depth,
        )

    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs: Any
    ) -> dict[str, Any]:
        """Hook info_sharing.self_attention_blocks[layer_index].attn.qkv; return {q, k}.

        Wraps the 2 input frames into MapAnything's view format, runs a forward pass
        with a per-call hook on the cross-frame self-attention block at ``layer_index``,
        then removes the hook.  MapAnything has no decodable pose encoding, so only
        q and k are returned — _verify_loop_candidate returns None for fresh poses.

        The hook is removed in a finally block — guaranteed cleanup even if the forward
        raises.  No persistent state is left on the model or its layers.

        Args:
            frames:      (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: Which self_attention_block to tap.  -1 = last (default).
            **kwargs:    minibatch_size (int, default 1),
                         memory_efficient_inference (bool, default False).

        Returns:
            dict with keys:
              "q": (B, heads, N_tokens, head_dim) query projections
              "k": (B, heads, N_tokens, head_dim) key projections
        """
        minibatch_size = kwargs.get("minibatch_size", 1)
        memory_efficient = kwargs.get("memory_efficient_inference", False)

        # Register a per-call hook on the QKV projection of the chosen block
        block = self.model.info_sharing.self_attention_blocks[layer_index]
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
                # Wrap frames into MapAnything's {"img": (1,C,H,W)} view dicts,
                # preprocess them, then run a standard forward pass
                raw_views = [{"img": f.unsqueeze(0)} for f in frames.cpu()]
                views = preprocess_input_views_for_inference(raw_views)
                self.model.forward(
                    views,
                    memory_efficient_inference=memory_efficient,
                    minibatch_size=minibatch_size,
                )
        finally:
            # Always remove the hook — no persistent state left on the model
            hook.remove()

        return captured

    def _reproject(
        self, raw_outputs: Any, extrinsics_3x4: np.ndarray, intrinsics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-derive world-space points using bundle-adjusted camera poses.

        MapAnything predicts pts3d_cam (camera-frame XYZ) directly — not a scalar
        depth map. After BA refines world2cam extrinsics, pts3d (world-frame) is
        stale because it has the original predicted poses baked in. pts3d_cam is
        pose-independent, so we transform it with the refined cam2world instead.

        Args:
            raw_outputs:    list[dict] from _forward(); each dict contains pts3d_cam,
                            mask, depth_z, img_no_norm (float32, unmasked).
            extrinsics_3x4: (N, 3, 4) refined world-to-camera matrices from BA.
            intrinsics:     (N, 3, 3) refined camera intrinsics (unused for point
                            reprojection; stored in FeedforwardResult by wrapper).

        Returns:
            (pts3d, colors) — (P, 3) float32 world-space points and (P, 3) uint8 RGB.
        """
        return _reproject_mapanything(raw_outputs, extrinsics_3x4)
