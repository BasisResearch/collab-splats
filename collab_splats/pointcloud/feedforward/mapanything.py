"""MapAnything feedforward backend: inference utilities and creator.

Provides:
  collect_pts3d_from_outputs — extract pts3d/colors/extrinsics/intrinsics from processed outputs
  MapAnythingCreator         — feedforward creator using MapAnything depth + pose estimation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from PIL import Image as PILImage

# timm 0.6.x compat: uniception (mapanything dep) imports `from timm.layers import DropPath`
# which does not exist in timm<0.9. Re-export it from timm.models.layers before the import.
import timm.layers as _tl
import timm.models.layers as _tml
if not hasattr(_tl, "DropPath"):
    _tl.DropPath = _tml.DropPath
del _tl, _tml

from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import load_images
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)

from ..utils import voxel_downsample
from .base import BaseFeedforwardCreator, FeedforwardResult, _extrinsics_3x4_to_4x4, console


# ── Inference utilities ────────────────────────────────────────────────────────

def collect_pts3d_from_outputs(
    outputs: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pts3d, colors, extrinsics, intrinsics from MapAnything output dicts.

    Returns:
        pts3d:      (P, 3) float32 world-space points (all frames concatenated)
        colors:     (P, 3) uint8 RGB
        extrinsics: (N, 3, 4) float32 world2cam [R|t]
        intrinsics: (N, 3, 3) float32 K matrices
    """
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    extrinsics_list: list[np.ndarray] = []

    for pred in outputs:
        pts3d = pred["pts3d"][0].cpu().numpy()
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)

        depth_z = pred["depth_z"][0].squeeze(-1).cpu().numpy()
        valid_depth_mask = depth_z > 0
        combined_mask = mask & valid_depth_mask

        img_no_norm = pred["img_no_norm"][0].cpu().numpy()
        colors = (img_no_norm * 255).astype(np.uint8)

        all_points.append(pts3d[combined_mask])
        all_colors.append(colors[combined_mask])

        intrinsics_list.append(pred["intrinsics"][0].cpu().numpy())

        cam2world = pred["camera_poses"][0].cpu().numpy()
        world2cam = closed_form_pose_inverse(cam2world[None])[0]
        extrinsics_list.append(world2cam[:3, :4])

    pts3d_all = np.concatenate(all_points, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)
    intrinsics = np.stack(intrinsics_list)
    extrinsics = np.stack(extrinsics_list)

    return pts3d_all, colors_all, extrinsics, intrinsics


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

        # Extract pts3d, colors, extrinsics, intrinsics from processed per-frame dicts
        pts3d, colors, extrinsics, intrinsics = collect_pts3d_from_outputs(processed)

        # Populate BA fields: per-frame images, confidence, and world-point grid
        _images = torch.stack(
            [p["img_no_norm"][0].cpu().permute(2, 0, 1) for p in processed]
        )
        if processed[0].get("conf") is not None:
            conf_list = [p["conf"][0] for p in processed]
            _conf = torch.stack([c[0] if c.ndim == 3 else c for c in conf_list])
        else:
            _conf = None
        _world_points = np.stack(
            [p["pts3d"][0].cpu().numpy() for p in processed]
        )  # (N, H, W, 3)

        # Voxel downsample to reduce point cloud density
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(pts3d)
        _pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        _pcd, _ = voxel_downsample(_pcd, adaptive=False)
        pts3d = np.asarray(_pcd.points, dtype=np.float32)
        colors = (np.asarray(_pcd.colors) * 255).astype(np.uint8)

        # Convert extrinsics to 4×4 homogeneous form
        extrinsics_4x4 = _extrinsics_3x4_to_4x4(extrinsics)

        return FeedforwardResult(
            pts3d=pts3d,
            colors=colors,
            extrinsics=extrinsics_4x4,
            intrinsics=intrinsics,
            image_paths=self.image_paths,
            original_coords=self.original_coords,
            model_width=model_w,
            model_height=model_h,
            images=_images,
            conf=_conf,
            world_points=_world_points,
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

    def _reproject_ba(
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
            ext_4x4 = np.concatenate([extrinsics_3x4[i], [[0, 0, 0, 1]]], axis=0)  # (4, 4)
            cam2world = closed_form_pose_inverse(ext_4x4[None])[0]                   # (4, 4)

            # Apply mask and transform camera-frame points to world frame
            pts_flat = pts3d_cam[combined_mask]                                       # (K, 3)
            pts_world = (cam2world[:3, :3] @ pts_flat.T + cam2world[:3, 3:]).T       # (K, 3)

            # Extract colors for surviving pixels
            img_no_norm = pred["img_no_norm"][0].cpu().numpy()                        # (H, W, 3)
            colors = (img_no_norm[combined_mask] * 255).astype(np.uint8)              # (K, 3)

            all_pts.append(pts_world.astype(np.float32))
            all_colors.append(colors)

        return np.concatenate(all_pts, axis=0), np.concatenate(all_colors, axis=0)
