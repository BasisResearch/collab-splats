"""
Rasterization and the splat stage's artifacts.

- ``render_gaussians``: the one call into gsplat
- ``render_views``: re-render a trained model view by view
- ``write_outputs``: splats.ply / ckpt.pt / splats_quality_report.json
- ``load_checkpoint``: that checkpoint back into a render-only model
- 3DGS path: ``gsplat.rasterization``, antialiased
- 2DGS path: ``gsplat.rasterization_2dgs``, plus a distortion map and RaDe-GS median depth
- 2DGS normals arrive WORLD-frame; rotated into camera space here
- depth normals: finite-differenced at an identity pose, both primitives and both depths
- ``render_plane``: PGSR's planar signals, 3DGS-only — no 2DGS counterpart upstream
- upstream: every ``GS-SR <file>:<line>`` below is yanxian-ll/GS-SR @ 566359be
- GS-SR refactored ``pgsr_scene.py`` Oct 2025: same constructs sit elsewhere on its current main
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat
from torch import Tensor

from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.pgsr import plane_depth as compute_plane_depth
from collab_splats.utils.progress import progress

# Annotation-only: importing any of these back at runtime would be circular
if TYPE_CHECKING:
    from collab_splats.splats.gaussian import Gaussians
    from collab_splats.splats.scaffold import Scaffold
    from collab_splats.splats.trainer import SplatsConfig

logger = logging.getLogger(__name__)


def gaussian_normals_in_camera_frame(
    quats: Tensor, scales: Tensor, means: Tensor, world_to_cam: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Per-Gaussian camera-frame normal (shortest scale axis, flipped to face the camera) and mean.

    Args:
        quats: (N, 4) rotations.
        scales: (N, 3) axis lengths; `argmin` over them is non-differentiable, so no gradient
            reaches `scales` through the normal. `quats` and `means` do receive one.
        means: (N, 3) world-frame centers.
        world_to_cam: (4, 4) view matrix.

    Returns:
        normals: (N, 3), camera frame.
        means_cam: (N, 3), the positions the normals were flipped against — what PGSR's plane
            distance `|n . x_cam|` needs.
    """
    # Shortest axis of each Gaussian is its normal direction in world space
    unit_quats = F.normalize(quats, dim=-1)
    rotations = normalized_quat_to_rotmat(unit_quats)
    shortest_axis = scales.argmin(dim=-1)
    gaussian_index = torch.arange(len(rotations), device=rotations.device)
    normals_world = rotations[gaussian_index, :, shortest_axis]

    # Rotate normals and positions into the camera frame
    rotation_w2c = world_to_cam[:3, :3]
    translation_w2c = world_to_cam[:3, 3]
    normals_cam = normals_world @ rotation_w2c.T
    means_cam = means @ rotation_w2c.T + translation_w2c

    # A normal pointing away from the camera (positive dot with the view ray) is flipped
    faces_away = (normals_cam * means_cam).sum(-1, keepdim=True) > 0
    return torch.where(faces_away, -normals_cam, normals_cam), means_cam


def render_gaussians(
    primitive: str,
    decoded: dict[str, Tensor],
    cam_to_world: Tensor,
    intrinsics: Tensor,
    width: int,
    height: int,
    sh_degree: int | None,
    absgrad: bool,
    render_normals: bool = True,
    render_plane: bool = False,
) -> tuple[dict[str, Tensor], dict]:
    """
    Rasterize already-activated Gaussians for one camera.

    Args:
        primitive: "3dgs" or "2dgs".
        decoded: post-activation means/quats/scales/opacities/colors; other keys ignored.
        cam_to_world: (1, 4, 4) — one camera per call.
        intrinsics: (1, 3, 3).
        width: render width, px.
        height: render height, px.
        sh_degree: SH degree, or None for precomputed (N, 3) RGB — the scaffold path.
        absgrad: accumulate absolute screen-space gradients, for densification.
        render_normals: False omits `normal`/`depth_normal` on 3DGS, and only the
            finite-differenced depth normals on 2DGS — the rasterizer's own come free.
        render_plane: PGSR planar signals; 3DGS-only (raises on 2DGS), forces `render_normals` on.

    Returns:
        (render, gsplat strategy info). Maps are (1, H, W, C), camera frame.

        - `rgb`, `alpha`, `depth`: always
        - `normal`: unit on 3DGS; on 2DGS the alpha-weighted accumulated normal, left non-unit to
          mirror upstream's 2DGS trainer — consistency is alpha^2-weighted there
        - `depth_normal`, 2DGS `depth_normal_median`: finite-differenced at an identity pose
        - 2DGS only: `distortion`, `median_depth`
        - `render_plane` only: `plane_normal`, `plane_distance`, `plane_depth`, `plane_depth_normal`
        - `plane_normal`/`plane_distance` stay RAW: plane depth is their ratio, so 1/alpha cancels
        - plane path leaves `normal`/`depth_normal`/`depth` bit-for-bit unchanged
    """
    # No 2dgs-pgsr upstream: the plane signals have no 2DGS definition to reproduce
    if render_plane and primitive == "2dgs":
        raise ValueError(
            "render_plane is 3DGS-only: PGSR builds on the vanilla/3dgs rasterizer and there is no "
            "2dgs-pgsr upstream. Use primitive='3dgs' or turn plane rendering off."
        )

    # Plane signals share the extra-signal channels with the normals, so planes force that pass on
    render_normals = render_normals or render_plane

    assert cam_to_world.shape[0] == 1, "render_gaussians renders one camera at a time"

    # The five tensors the rasterizer takes; everything else in `decoded` is for the caller
    means = decoded["means"]
    quats = decoded["quats"]
    scales = decoded["scales"]
    opacities = decoded["opacities"]
    colors = decoded["colors"]
    world_to_cam = torch.linalg.inv(cam_to_world)
    shared_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_to_cam,
        Ks=intrinsics,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=absgrad,
        render_mode="RGB+ED",
    )

    # 2DGS precomputed colors must arrive as (C, N, 3)
    # - rasterization_2dgs concatenates the depth channel onto them per camera
    # - SH coefficients (N, K, 3) and the 3DGS kernel broadcast on their own
    if primitive == "2dgs" and sh_degree is None and colors.dim() == 2:
        shared_kwargs["colors"] = colors[None].expand(len(intrinsics), -1, -1)

    # Depth normals are finite-differenced in camera space (identity pose) for both primitives
    identity_pose = torch.eye(4, device=cam_to_world.device)[None]

    # 2DGS rasterizer outputs, and the frame they arrive in
    # - normals: WORLD frame, rotated into the camera frame here
    # - both depth normals: finite-differenced in camera space
    if primitive == "2dgs":
        rgb_depth, alpha, normal_world, _depth_normal_world, distortion, median_depth, info = rasterization_2dgs(
            **shared_kwargs, distloss=True
        )
        rgb = rgb_depth[..., :3]
        depth = rgb_depth[..., 3:4]
        rotation_w2c = world_to_cam[:, :3, :3]
        normal_cam = torch.einsum("cij,chwj->chwi", rotation_w2c, normal_world)

        # RaDe-GS median depth, kept unconditionally
        # - median Gaussian per ray, not the alpha-weighted expectation
        # - sparse (one Gaussian per ray gets gradient), sharper across depth discontinuities
        # - the mesh stage fuses from it, and the rasterizer already returned it
        render = {
            "rgb": rgb,
            "alpha": alpha,
            "depth": depth,
            "median_depth": median_depth,
            "normal": normal_cam,
            "distortion": distortion,
        }

        # Depth normals only feed normal_consistency_loss
        # - off its schedule they are dead tensors held live through backward
        # - so the trainer gates render_normals on that same loss_active
        if render_normals:
            render["depth_normal"] = depth_to_normal(depth, identity_pose, intrinsics)
            render["depth_normal_median"] = depth_to_normal(median_depth, identity_pose, intrinsics)
        return render, info

    # 3DGS without normals: plain rgb+depth pass (no extra-signal channels through the kernel)
    if not render_normals:
        rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased")
        render = {"rgb": rgb_depth[..., :3], "alpha": alpha, "depth": rgb_depth[..., 3:4]}
        return render, info

    # 3DGS normals ride as an extra per-Gaussian signal
    # - padded to 4 channels: the compiled kernel takes 8 total (rgb + depth + 4), not 7
    first_world_to_cam = world_to_cam[0]
    normals_cam, means_cam = gaussian_normals_in_camera_frame(quats, scales, means, first_world_to_cam)

    # PGSR's plane distance |n . x_cam| claims the 4th channel
    # - upstream: GS-SR gssr/scene/pgsr_scene.py:296-302
    # - without plane rendering it stays the zero pad, so the ordinary path is untouched
    if render_plane:
        fourth_channel = (normals_cam * means_cam).sum(-1, keepdim=True).abs()
    else:
        fourth_channel = torch.zeros_like(normals_cam[:, :1])
    extra_signals = torch.cat([normals_cam, fourth_channel], dim=-1)

    rgb_depth, alpha, info = rasterization(**shared_kwargs, rasterize_mode="antialiased", extra_signals=extra_signals)
    rgb = rgb_depth[..., :3]
    depth = rgb_depth[..., 3:4]
    rendered_signals = info["render_extra_signals"]
    rendered_normals = rendered_signals[..., :3]
    render = {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "normal": F.normalize(rendered_normals, dim=-1),
        "depth_normal": depth_to_normal(depth, identity_pose, intrinsics),
    }
    if not render_plane:
        return render, info

    # Both accumulated maps stay raw
    # - upstream: GS-SR gssr/scene/pgsr_scene.py:316-318, `rendered_normal` is the un-normalized sum
    # - plane depth is their ratio, so the missing 1/alpha cancels
    plane_distance = rendered_signals[..., 3:4]
    render["plane_normal"] = rendered_normals
    render["plane_distance"] = plane_distance
    render["plane_depth"] = compute_plane_depth(rendered_normals, plane_distance, intrinsics)

    # plane_depth_normal stays pure geometry: camera frame, unit length
    # - upstream scales it by `rendered_alpha.detach()` (GS-SR gssr/scene/pgsr_scene.py:320)
    # - the alpha weighting is left to the loss instead
    render["plane_depth_normal"] = depth_to_normal(render["plane_depth"], identity_pose, intrinsics)
    return render, info


########################################################################################
# Outputs
########################################################################################


# `@torch.no_grad()` as a DECORATOR, never a `with` inside the body
# - grad mode is process-global
# - a generator suspended at a `yield` inside a `with` has not exited it
# - a partial consumer (zip, bare next, break) would then leave autograd off process-wide
@torch.no_grad()
def render_views(
    model,
    camera_opt: CameraOpt,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int,
) -> Iterator[dict[str, Tensor]]:
    """
    Re-render every view from a trained model, one at a time.

    - a generator, not a list: 300 renders at 1080p cost ~23 B/px, ~14 GB held at once

    Args:
        model: a `Gaussians` or `Scaffold` in eval use — no strategy, no optimizers.
        camera_opt: the run's `CameraOpt`; its color affine is applied to `rgb`. Its pose half is
            NOT applied — `cam_to_world` is already corrected, so it would double-apply.
        cam_to_world: (N, 4, 4), output world frame.
        intrinsics: (N, 3, 3) at (height, width).
        height: render height, px.
        width: render width, px.

    Yields:
        One render dict per view: `rgb` (1, H, W, 3) in [0, 1], `depth`, `alpha`, `normal`, plus
        `median_depth` for 2dgs.
    """
    device = cam_to_world.device
    for view in range(len(cam_to_world)):
        camera_id = torch.tensor([view], device=device)
        render, _ = model.render(
            cam_to_world[view : view + 1],
            intrinsics[view : view + 1],
            width,
            height,
            camera_id,
            step=None,
        )
        render["rgb"] = camera_opt.color(render["rgb"], camera_id).clamp(0, 1)
        yield render


def write_outputs(
    cfg: "SplatsConfig",
    model,
    refine,
    images: np.ndarray,
    image_ids: list[int],
    cam_to_world: Tensor,
    intrinsics: Tensor,
    out_dir: Path,
    seconds: float,
    final_losses: dict[str, float],
    *,
    training_cam_to_world: Tensor,
) -> None:
    """
    Write splats.ply, ckpt.pt and splats_quality_report.json to out_dir.

    Args:
        cfg: the run's SplatsConfig; serialized into the checkpoint and the report.
        model: the trained `Gaussians` or `Scaffold`, already denormalized.
        refine: the run's `CameraOpt`; only its color affine is checkpointed, the pose deltas are
            already folded into `cam_to_world`.
        images: (N, H, W, 3) uint8 training frames, scored against the re-renders.
        image_ids: frame indices, in render order.
        cam_to_world: (N, 4, 4) pose-corrected, output world frame.
        intrinsics: (N, 3, 3) at the training resolution.
        out_dir: the stage's output directory; created if absent.
        seconds: wall-clock training time, for the report summary.
        final_losses: the last step's single-view loss snapshot, not an average.
        training_cam_to_world: (N, 4, 4), the same cameras WITHOUT the learned pose deltas — the
            poses the ply is baked against. See the export call below.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)
    n_views, height, width = images.shape[:3]

    # splats.ply: standard 3DGS ply, raw log-scales / logit-opacities
    # - scaffold has no static Gaussians: its export bakes one set from the anchors, lossy
    # - which is why a scaffold ply and a scaffold render do not match

    # The export takes the TRAINING poses, not the corrected ones
    # - raw cameras decide anchor visibility and each anchor's baked mean view direction
    # - the checkpoint and re-renders below take the corrected poses; do not collapse the two
    export_splats(
        **model.export_gaussians(training_cam_to_world, intrinsics, width, height),
        format="ply",
        save_to=str(out_dir / "splats.ply"),
    )

    # ckpt.pt: self-contained — model, cameras, image size and config, everything a re-render needs
    checkpoint = model.checkpoint()
    checkpoint["splats"] = {name: param.detach().cpu() for name, param in checkpoint["splats"].items()}
    checkpoint["config"] = config_dict
    checkpoint["cam_to_world"] = cam_to_world.detach().cpu()
    checkpoint["intrinsics"] = intrinsics.detach().cpu()
    checkpoint["image_ids"] = list(image_ids)
    checkpoint["image_size"] = (height, width)
    checkpoint["appearance"] = None if refine.appearance is None else refine.appearance.state_dict()
    torch.save(checkpoint, out_dir / "ckpt.pt")

    # splats_quality_report.json: per-view psnr/ssim scored in memory as the renders stream past
    per_frame = []
    renders = render_views(model, refine, cam_to_world, intrinsics, height, width)
    for view, render in zip(image_ids, progress(renders, desc="splats render", total=n_views)):
        target = torch.from_numpy(images[view]).to(render["rgb"].device).float()[None] / 255.0
        mse = F.mse_loss(render["rgb"], target).item()
        ssim_distance = ssim_loss(render["rgb"].permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)).item()
        per_frame.append(
            {
                "image_id": view,
                "psnr": 10 * np.log10(1.0 / max(mse, 1e-12)),
                "ssim": 1.0 - ssim_distance,
                **model.frame_report(render),
            }
        )

    mean_psnr = float(np.mean([frame["psnr"] for frame in per_frame]))
    mean_ssim = float(np.mean([frame["ssim"] for frame in per_frame]))
    summary = {
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "n_gaussians": model.n_primitives,
        "seconds": round(seconds, 1),
        "final_losses": final_losses,
        "config": config_dict,
    }
    # Extra per-view keys are summarized as <name>_mean
    # - scaffold's decoded primitive count is the only one today
    # - n_gaussians cannot stand in for it: that counts anchors
    for name in sorted(set(per_frame[0]) - {"image_id", "psnr", "ssim"}):
        summary[f"{name}_mean"] = float(np.mean([frame[name] for frame in per_frame]))

    report = {"summary": summary, "per_frame": per_frame}
    (out_dir / "splats_quality_report.json").write_text(json.dumps(report, indent=2))

    # The log names the model's own primitive unit
    # - scaffold counts ANCHORS, and its rendered count is per-view
    # - decoded mean carried beside it when the model reported one
    unit = model.primitive_unit
    decoded_mean = summary.get("n_decoded_mean")
    decoded_note = "" if decoded_mean is None else f" ({decoded_mean:.0f} decoded/view)"
    logger.info(
        "splats: %d %s%s, psnr %.2f, ssim %.3f, %.0fs -> %s",
        model.n_primitives,
        unit,
        decoded_note,
        mean_psnr,
        mean_ssim,
        seconds,
        out_dir,
    )


def load_checkpoint(
    path: Path, device: str
) -> tuple["Gaussians | Scaffold", CameraOpt, Tensor, Tensor, list[int], tuple[int, int]]:
    """
    Rebuild a render-only model and its cameras from a ckpt.pt.

    Args:
        path: the checkpoint written by `write_outputs`.
        device: torch device string for the model and cameras.

    Returns:
        (model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width)).

        - model: no optimizers, no strategy — it renders and exports, it does not train
        - camera_opt: the color affine only; the identity when the run had none

    Raises:
        ValueError: the checkpoint names a representation that is not in `REPRESENTATIONS`.
    """
    # Local import: the trainer imports this module, so module level would be circular
    from collab_splats.splats.trainer import MODEL_CLASSES, REPRESENTATIONS

    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    config = ckpt["config"]

    # An unknown representation raises rather than rebuilding a vanilla model
    # - the message carries the checkpoint path and the allow-list; a bare KeyError carries neither
    representation = config["representation"]
    if representation not in MODEL_CLASSES:
        raise ValueError(f"{path}: splats.representation must be one of {REPRESENTATIONS}, got '{representation}'")

    model = MODEL_CLASSES[representation].from_checkpoint(ckpt, device)

    # Color affine only: the pose deltas are already baked into the saved cam_to_world
    camera_opt = CameraOpt(
        len(ckpt["image_ids"]),
        optimize_pose=False,
        optimize_appearance=ckpt["appearance"] is not None,
    ).to(device)
    if ckpt["appearance"] is not None:
        camera_opt.appearance.load_state_dict(ckpt["appearance"])

    cam_to_world = ckpt["cam_to_world"].to(device).float()
    intrinsics = ckpt["intrinsics"].to(device).float()
    height, width = ckpt["image_size"]
    return model, camera_opt, cam_to_world, intrinsics, list(ckpt["image_ids"]), (int(height), int(width))
