"""
Writing the splat stage's outputs and loading a checkpoint back.

- `write_outputs`: splats.ply, ckpt.pt and splats_quality_report.json
- `load_checkpoint`: ckpt.pt back into a render-only model and its cameras
- `MODEL_CLASSES`: representation name -> model class
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.exporter import export_splats
from gsplat.losses import ssim_loss
from torch import Tensor

from collab_splats.splats.cameras import CameraOpt
from collab_splats.splats.gaussian import Gaussians
from collab_splats.splats.rendering import render_views
from collab_splats.splats.scaffold import Scaffold
from collab_splats.utils.io import write_json
from collab_splats.utils.progress import progress

# Annotation-only: a runtime import of the trainer would be circular
if TYPE_CHECKING:
    from collab_splats.splats.trainer import SplatsConfig

logger = logging.getLogger(__name__)

# Representation name -> model class; also the config's allow-list
MODEL_CLASSES = {"vanilla": Gaussians, "scaffold": Scaffold}
REPRESENTATIONS = tuple(MODEL_CLASSES)


def write_outputs(
    cfg: "SplatsConfig",
    model: Gaussians | Scaffold,
    refine: CameraOpt,
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
    Write splats.ply, ckpt.pt and splats_quality_report.json to `out_dir`.

    - the report scores a re-render of every training view (psnr, ssim)

    Args:
        cfg: run config; saved into the checkpoint and the report.
        model: the trained model, in world units.
        refine: the run's `CameraOpt`; only its color correction is saved.
        images: (N, H, W, 3) uint8 training frames.
        image_ids: source frame index per row of `images`.
        cam_to_world: (N, 4, 4) poses with the learned deltas applied.
        intrinsics: (N, 3, 3) at the training resolution.
        out_dir: output directory; created if absent.
        seconds: training wall-clock time.
        final_losses: the last step's losses.
        training_cam_to_world: (N, 4, 4) poses without the learned deltas; used for the ply.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)
    n_views, height, width = images.shape[:3]

    # splats.ply: standard 3DGS ply
    # - scaffold bakes one Gaussian set from its anchors, so its ply is lossy

    # The ply takes the training poses; the checkpoint and re-renders take the corrected ones
    export_splats(
        **model.export_gaussians(training_cam_to_world, intrinsics, width, height),
        format="ply",
        save_to=str(out_dir / "splats.ply"),
    )

    # ckpt.pt: model, cameras, image size and config, enough to re-render
    checkpoint = model.checkpoint()
    checkpoint["splats"] = {name: param.detach().cpu() for name, param in checkpoint["splats"].items()}
    checkpoint["config"] = config_dict
    checkpoint["cam_to_world"] = cam_to_world.detach().cpu()
    checkpoint["intrinsics"] = intrinsics.detach().cpu()
    checkpoint["image_ids"] = list(image_ids)
    checkpoint["image_size"] = (height, width)
    checkpoint["appearance"] = None if refine.appearance is None else refine.appearance.state_dict()
    torch.save(checkpoint, out_dir / "ckpt.pt")

    # splats_quality_report.json: per-view psnr and ssim
    per_frame = []
    renders = render_views(model, refine, cam_to_world, intrinsics, height, width)

    # Index images by row; image_ids maps each row to its source frame
    for row, render in enumerate(progress(renders, desc="splats render", total=n_views)):
        target = torch.from_numpy(images[row]).to(render["rgb"].device).float()[None] / 255.0
        mse = F.mse_loss(render["rgb"], target).item()
        ssim_distance = ssim_loss(render["rgb"].permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)).item()
        per_frame.append(
            {
                "image_id": image_ids[row],
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

    # Extra per-view keys (e.g. scaffold's decoded count) are summarized as <name>_mean
    for name in sorted(set(per_frame[0]) - {"image_id", "psnr", "ssim"}):
        summary[f"{name}_mean"] = float(np.mean([frame[name] for frame in per_frame]))

    report = {"summary": summary, "per_frame": per_frame}
    write_json(out_dir / "splats_quality_report.json", report)

    # Log in the model's own unit (scaffold counts anchors), plus decoded count if any
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
) -> tuple[Gaussians | Scaffold, CameraOpt, Tensor, Tensor, list[int], tuple[int, int]]:
    """
    Load a render-only model and its cameras from a ckpt.pt.

    - the model has no optimizers; it can render and export, not train
    - `camera_opt` holds only the color correction, if the run learned one

    Args:
        path: checkpoint written by `write_outputs`.
        device: torch device string.

    Returns:
        (model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width)).

    Raises:
        ValueError: the checkpoint's representation is unknown.
    """
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    config = ckpt["config"]

    # Unknown representation: raise with the path and the allow-list
    representation = config["representation"]

    if representation not in MODEL_CLASSES:
        raise ValueError(f"{path}: splats.representation must be one of {REPRESENTATIONS}, got '{representation}'")

    model = MODEL_CLASSES[representation].from_checkpoint(ckpt, device)

    # Color correction only: pose deltas are already in the saved cam_to_world
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
