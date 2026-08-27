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
from collab_splats.splats.appearance import AppearanceModule
from collab_splats.splats.cameras import CameraOptModule
from collab_splats.splats.rendering import (
    SH_DC_NORMALISER,
    render_gaussians,
    render_view,
)
from collab_splats.utils.progress import progress

logger = logging.getLogger(__name__)


def render_all_views(
    cfg,
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    appearance: AppearanceModule | None,
    images: np.ndarray,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    store: zarr.Group,
    anchor_field=None,
) -> list[dict]:
    """
    Re-render every training view at full SH over black, streaming each render into ``store``.

    - Writes rgb/depth/normal/alpha (one chunk per view) and c2w/K (one chunk each) to the group.
    - 2DGS additionally writes ``median_depth`` (the RaDe-GS surface depth); 3DGS has none.
    - With a pose refiner the stored c2w holds the refined poses: what was actually rendered.
    - With an appearance module each view gets its learned colour correction (train views only).
    - With ``anchor_field`` the view's Gaussians are decoded from the anchors instead (post-activation
      RGB, so there is no SH degree to render at): these renders ARE the scaffold model, unlike the ply.
    - Returns per-frame psnr/ssim, plus ``n_decoded`` per frame under ``anchor_field``.
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

    # 2DGS also renders a median (surface) depth; mesh.splat_depth chooses which one TSDF fuses
    writes_median_depth = cfg.primitive == "2dgs"
    if writes_median_depth:
        per_view_arrays["median_depth"] = ((n_views, height, width), np.float32)

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
            camera_id = torch.tensor([view], device=device)
            if pose_refiner is not None:
                view_cam_to_world = pose_refiner(view_cam_to_world, camera_id)
                refined_pose = view_cam_to_world[0]
                cam_to_world_out[view] = refined_pose.cpu().numpy()

            # Render and score against the training frame
            if anchor_field is not None:
                decoded, _ = anchor_field.decode(
                    cfg.primitive, view_cam_to_world, view_intrinsics, width, height, camera_id
                )
                render, _ = render_gaussians(
                    cfg.primitive,
                    decoded,
                    view_cam_to_world,
                    view_intrinsics,
                    width,
                    height,
                    sh_degree=None,
                    absgrad=False,
                )
            else:
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
            rendered_rgb = render["rgb"]
            if appearance is not None:
                rendered_rgb = appearance(rendered_rgb, camera_id)
            rendered_rgb = rendered_rgb.clamp(0, 1)
            view_image = images[view]
            target_rgb = torch.from_numpy(view_image).to(device).float()[None] / 255.0
            rendered_nchw = rendered_rgb.permute(0, 3, 1, 2)
            target_nchw = target_rgb.permute(0, 3, 1, 2)
            mse = F.mse_loss(rendered_rgb, target_rgb).item()
            ssim_distance = ssim_loss(rendered_nchw, target_nchw).item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            frame = {"image_id": view, "psnr": psnr, "ssim": 1.0 - ssim_distance}

            # Scaffold's primitive count is a per-view quantity — frustum culling and the opacity
            # gate decide it every frame — so it is measured here rather than inferred from anchors
            if anchor_field is not None:
                frame["n_decoded"] = int(len(decoded["means"]))
            per_frame.append(frame)

            # Stream the render into its chunk
            rgb_uint8 = (rendered_rgb[0] * 255).round().byte()
            store["rgb"][view] = rgb_uint8.cpu().numpy()
            store["depth"][view] = render["depth"][0, ..., 0].cpu().numpy()
            store["normal"][view] = render["normal"][0].cpu().numpy()
            store["alpha"][view] = render["alpha"][0, ..., 0].cpu().numpy()
            if writes_median_depth:
                store["median_depth"][view] = render["median_depth"][0, ..., 0].cpu().numpy()

    # Cameras are tiny: one chunk each rather than 2N chunk files
    intrinsics_out = intrinsics.cpu().numpy()
    store.create_array("c2w", data=cam_to_world_out, chunks=cam_to_world_out.shape)
    store.create_array("K", data=intrinsics_out, chunks=intrinsics_out.shape)
    return per_frame


def bake_anchor_gaussians(
    anchor_field, cam_to_world: Tensor, intrinsics: Tensor, width: int, height: int
) -> dict[str, Tensor]:
    """
    Decode each anchor once at its mean observed view direction, for a static viewer-loadable ply.

    - Anchors seen by no training camera fall back to the direction of the nearest camera.
    - Colours are baked into the degree-0 SH band; there are no higher bands to write.
    - Lossy by construction: the trained model is view-dependent. splats.zarr renders are not affected.
    """
    device = anchor_field.params["anchors"].device
    anchors = anchor_field.params["anchors"].detach()

    # Accumulate the unit direction to every camera that can see each anchor
    direction_sum = torch.zeros_like(anchors)
    seen_count = torch.zeros(len(anchors), device=device)
    for view in range(len(cam_to_world)):
        visible = anchor_field.visible_anchors(
            cam_to_world[view : view + 1], intrinsics[view : view + 1], width, height
        )
        to_camera = anchors - cam_to_world[view, :3, 3]
        unit = to_camera / to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction_sum[visible] += unit[visible]
        seen_count[visible] += 1

    # Unseen anchors: use the nearest camera's direction rather than dropping them from the ply
    unseen = seen_count == 0
    if bool(unseen.any()):
        camera_centres = cam_to_world[:, :3, 3]
        nearest = torch.cdist(anchors[unseen], camera_centres).argmin(dim=1)
        to_nearest = anchors[unseen] - camera_centres[nearest]
        direction_sum[unseen] = to_nearest / to_nearest.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        seen_count[unseen] = 1

    mean_direction = direction_sum / seen_count[:, None]
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mean_distance = (anchors - cam_to_world[:, :3, 3].mean(dim=0)).norm(dim=-1, keepdim=True)

    # One decode at that direction, bypassing the frustum filter so every anchor is written
    with torch.no_grad():
        features = torch.cat([anchor_field.params["anchor_feat"].detach(), mean_direction, mean_distance], dim=-1)
        camera_id = torch.zeros(1, dtype=torch.long, device=device)
        neural_opacity, cov, colour = anchor_field.mlps(features, camera_id)

        n_offsets = anchor_field.cfg.n_offsets
        scaling = torch.exp(anchor_field.params["scaling"].detach())
        offsets = anchor_field.params["offsets"].detach()
        keep = (neural_opacity > 0).reshape(-1)

        # An all-closed decode would write an empty ply, which gsplat's export_splats cannot
        # serialise (its shN reshape needs at least one splat). Keep the most opaque offset,
        # mirroring the same guard in AnchorField.decode.
        if not bool(keep.any()):
            keep = torch.zeros_like(keep)
            keep[neural_opacity.reshape(-1).argmax()] = True

        means = (anchors[:, None, :] + offsets * scaling[:, None, :3]).reshape(-1, 3)[keep]
        cov = cov.reshape(-1, 7)[keep]
        scales = scaling[:, 3:6].repeat_interleave(n_offsets, dim=0)[keep] * torch.sigmoid(cov[:, :3])
        quats = F.normalize(cov[:, 3:7], dim=-1)
        opacities = neural_opacity.reshape(-1)[keep]
        colors = colour.reshape(-1, 3)[keep]

    # The ply writer wants the raw forms every viewer expects: log scales, logit opacities, SH DC
    return {
        "means": means,
        "scales": torch.log(scales.clamp_min(1e-12)),
        "quats": quats,
        "opacities": torch.logit(opacities.clamp(1e-4, 1 - 1e-4)),
        "sh0": ((colors - 0.5) / SH_DC_NORMALISER).unsqueeze(1),
        "shN": torch.zeros(len(means), 0, 3, device=means.device),
    }


def write_splat_outputs(
    cfg,
    gaussians: torch.nn.ParameterDict,
    pose_refiner: CameraOptModule | None,
    appearance: AppearanceModule | None,
    images: np.ndarray,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    out_dir: Path,
    train_seconds: float,
    final_losses: dict[str, float],
    anchor_field=None,
) -> None:
    """
    Write splats.ply, ckpt.pt, splats.zarr and splats_quality_report.json to out_dir.

    - ``final_losses`` is the last training step's single-view snapshot, not an average.
    - ``anchor_field`` (scaffold only) carries the anchors and MLP heads: the ply is baked from them,
      the checkpoint gains the heads, and every render decodes rather than reading `gaussians`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)

    # splats.ply: standard 3DGS ply (raw log-scales / logit-opacities, as every viewer expects).
    # Scaffold has no static Gaussians, so the ply is baked at each anchor's mean observed view
    # direction — lossy, and marked as such in the zarr provenance.
    ply_path = str(out_dir / "splats.ply")
    ply_baked = anchor_field is not None
    ply_source = (
        bake_anchor_gaussians(anchor_field, cam_to_world, intrinsics, images.shape[2], images.shape[1])
        if ply_baked
        else gaussians
    )
    export_splats(
        means=ply_source["means"],
        scales=ply_source["scales"],
        quats=ply_source["quats"],
        opacities=ply_source["opacities"],
        sh0=ply_source["sh0"],
        shN=ply_source["shN"],
        format="ply",
        save_to=ply_path,
    )

    # ckpt.pt: raw parameters + pose/appearance state + config — everything needed to re-render or continue
    splats_cpu = {name: param.detach().cpu() for name, param in gaussians.items()}
    pose_adjust = None if pose_refiner is None else pose_refiner.state_dict()
    appearance_state = None if appearance is None else appearance.state_dict()
    checkpoint = {
        "splats": splats_cpu,
        "pose_adjust": pose_adjust,
        "appearance": appearance_state,
        "config": config_dict,
    }

    # Scaffold's decode heads are half the model: without them the anchors cannot be rendered
    if anchor_field is not None:
        checkpoint["mlps"] = anchor_field.mlps.state_dict()
        checkpoint["voxel_size"] = anchor_field.voxel_size
    torch.save(checkpoint, out_dir / "ckpt.pt")

    # splats.zarr: renders streamed per view, provenance in attrs
    store = zarr.open_group(out_dir / "splats.zarr", mode="w")
    per_frame = render_all_views(
        cfg, gaussians, pose_refiner, appearance, images, cam_to_world, intrinsics, store, anchor_field=anchor_field
    )
    image_ids = list(range(len(images)))
    store.attrs.update(
        image_ids=image_ids,
        primitive=cfg.primitive,
        representation=cfg.representation,
        n_anchors=(0 if anchor_field is None else len(anchor_field.params["anchors"])),
        ply_baked=ply_baked,
        pose_opt=cfg.pose_opt,
        gsplat_commit=GSPLAT_COMMIT,
        config=config_dict,
    )

    # splats_quality_report.json: same shape as the other stage reports (summary + per_frame)
    mean_psnr = float(np.mean([frame["psnr"] for frame in per_frame]))
    mean_ssim = float(np.mean([frame["ssim"] for frame in per_frame]))
    # Primitive count: gaussians for vanilla, anchors for scaffold (the decoded count is per view)
    n_gaussians = int(len(gaussians["anchors" if anchor_field is not None else "means"]))
    summary = {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "n_gaussians": n_gaussians,
        "seconds": round(train_seconds, 1),
        "final_losses": final_losses,
        "config": config_dict,
    }

    # Scaffold: n_gaussians above counts ANCHORS, so the rendered primitive count needs its own
    # number — the mean over views of what the decode actually handed the rasterizer
    if anchor_field is not None:
        summary["n_decoded_mean"] = float(np.mean([frame["n_decoded"] for frame in per_frame]))
    report = {"summary": summary, "per_frame": per_frame}
    report_path = out_dir / "splats_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Scaffold counts anchors, so the log names the unit and carries the decoded mean beside it
    unit = "gaussians" if anchor_field is None else "anchors"
    decoded_note = "" if anchor_field is None else f" ({summary['n_decoded_mean']:.0f} decoded/view)"
    logger.info(
        "splats: %d %s%s, psnr %.2f, ssim %.3f, %.0fs -> %s",
        n_gaussians,
        unit,
        decoded_note,
        mean_psnr,
        mean_ssim,
        train_seconds,
        out_dir,
    )
