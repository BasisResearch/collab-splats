"""
Splat-stage artifacts: splats.ply, ckpt.pt, splats.zarr (every view re-rendered) and the quality report.

Renders stream into splats.zarr one view at a time: holding every render on the host first costs
~23 B/px, about 14 GB for 300 views at 1080p inside the 46.6 GB container cap.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss
from torch import Tensor

from collab_splats.splats import GSPLAT_COMMIT
from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.rendering import render_view
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)


def render_all_views(
    cfg,
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    images: np.ndarray,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    store: zarr.Group,
) -> list[dict]:
    """
    Re-render every training view at full SH over black, streaming each render into ``store``.

    - Writes rgb/depth/normal/alpha (one chunk per view) and c2w/K (one chunk each) to the group.
    - With a pose refiner the stored c2w holds the refined poses: what was actually rendered.
    - Returns per-frame psnr/ssim.
    """
    n_views, height, width = images.shape[:3]
    device = cam_to_world.device

    # Per-view chunks so downstream stages read frames independently; filled inside the loop
    per_view_arrays = {
        "rgb": ((n_views, height, width, 3), np.uint8),
        "depth": ((n_views, height, width), np.float32),
        "normal": ((n_views, height, width, 3), np.float32),
        "alpha": ((n_views, height, width), np.float32),
    }
    for name, (shape, dtype) in per_view_arrays.items():
        per_view_chunks = (1, *shape[1:])
        store.create_array(name, shape=shape, dtype=dtype, chunks=per_view_chunks)
    cam_to_world_out = cam_to_world.cpu().numpy().copy()
    per_frame = []

    with torch.no_grad():
        for view in progress(range(n_views), desc="splats render"):
            # Refined pose if poses were optimised
            view_cam_to_world = cam_to_world[view : view + 1]
            view_intrinsics = intrinsics[view : view + 1]
            if pose_refiner is not None:
                camera_id = torch.tensor([view], device=device)
                view_cam_to_world = pose_refiner(view_cam_to_world, camera_id)
                refined_pose = view_cam_to_world[0]
                cam_to_world_out[view] = refined_pose.cpu().numpy()

            # Render and score against the training frame
            render, _ = render_view(
                cfg.primitive,
                gaussians,
                view_cam_to_world,
                view_intrinsics,
                width,
                height,
                cfg.sh_degree,
                absgrad=False,
            )
            rendered_rgb = render["rgb"].clamp(0, 1)
            view_image = images[view]
            target_rgb = torch.from_numpy(view_image).to(device).float()[None] / 255.0
            rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
            target_nchw = target_rgb.permute(0, 3, 1, 2)
            mse = F.mse_loss(rendered_rgb, target_rgb).item()
            ssim_distance = ssim_loss(rendered_nchw, target_nchw).item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            per_frame.append({"image_id": view, "psnr": psnr, "ssim": 1.0 - ssim_distance})

            # Stream the render into its chunk
            rgb_uint8 = (rendered_rgb[0] * 255).round().byte()
            store["rgb"][view] = rgb_uint8.cpu().numpy()
            store["depth"][view] = render["depth"][0, ..., 0].cpu().numpy()
            store["normal"][view] = render["normal"][0].cpu().numpy()
            store["alpha"][view] = render["alpha"][0, ..., 0].cpu().numpy()

    # Cameras are tiny: one chunk each rather than 2N chunk files
    intrinsics_out = intrinsics.cpu().numpy()
    store.create_array("c2w", data=cam_to_world_out, chunks=cam_to_world_out.shape)
    store.create_array("K", data=intrinsics_out, chunks=intrinsics_out.shape)
    return per_frame


def write_splat_outputs(
    cfg,
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    images: np.ndarray,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    out_dir: Path,
    train_seconds: float,
    final_losses: dict[str, float],
) -> None:
    """
    Write splats.ply, ckpt.pt, splats.zarr and splats_quality_report.json to out_dir.

    - ``final_losses`` is the last training step's single-view snapshot, not an average.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)

    # splats.ply: standard 3DGS ply (raw log-scales / logit-opacities, as every viewer expects)
    ply_path = str(out_dir / "splats.ply")
    export_splats(
        means=gaussians["means"],
        scales=gaussians["scales"],
        quats=gaussians["quats"],
        opacities=gaussians["opacities"],
        sh0=gaussians["sh0"],
        shN=gaussians["shN"],
        format="ply",
        save_to=ply_path,
    )

    # ckpt.pt: raw parameters + pose deltas + config — everything needed to re-render or continue
    splats_cpu = {name: param.detach().cpu() for name, param in gaussians.items()}
    pose_adjust = None if pose_refiner is None else pose_refiner.state_dict()
    checkpoint = {"splats": splats_cpu, "pose_adjust": pose_adjust, "config": config_dict}
    torch.save(checkpoint, out_dir / "ckpt.pt")

    # splats.zarr: renders streamed per view, provenance in attrs
    store = zarr.open_group(out_dir / "splats.zarr", mode="w")
    per_frame = render_all_views(cfg, gaussians, pose_refiner, images, cam_to_world, intrinsics, store)
    image_ids = list(range(len(images)))
    store.attrs.update(
        image_ids=image_ids,
        primitive=cfg.primitive,
        pose_opt=cfg.pose_opt,
        gsplat_commit=GSPLAT_COMMIT,
        config=config_dict,
    )

    # splats_quality_report.json: same shape as the other stage reports (summary + per_frame)
    mean_psnr = float(np.mean([frame["psnr"] for frame in per_frame]))
    mean_ssim = float(np.mean([frame["ssim"] for frame in per_frame]))
    n_gaussians = int(len(gaussians["means"]))
    report = {
        "summary": {
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "n_gaussians": n_gaussians,
            "seconds": round(train_seconds, 1),
            "final_losses": final_losses,
            "config": config_dict,
        },
        "per_frame": per_frame,
    }
    report_path = out_dir / "splats_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(
        "splats: %d gaussians, psnr %.2f, ssim %.3f, %.0fs -> %s",
        n_gaussians,
        mean_psnr,
        mean_ssim,
        train_seconds,
        out_dir,
    )
