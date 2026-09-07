"""
Train each splat primitive straight from a pointcloud.zarr and tabulate the trainer's own quality report.

Metric: the trainer's held-in PSNR / SSIM over the training views, Gaussian count and wall-clock,
read back from splats_quality_report.json. No COLMAP is needed — the zarr carries poses, K,
points, colors and depth. Optional GT depth error (--seq, 7-Scenes) re-renders ckpt.pt and
scores its depth where alpha > 0.5 after median alignment, via the eval_multiview_conf helpers.

Resolution: like the pipeline's splats stage, training uses the NATIVE frames from the scene's
images/ directory (found beside pointcloud.zarr, or via --images-dir) with the zarr's model-res K
rescaled to native size; --model-res trains on the model-res images stored in the zarr instead
(ablation row).
Depth targets stay model-res either way — the trainer nearest-resizes them to the frame size.

CLI/tmux only (GPU training). Results under evals/results/ (gitignored).

Usage:
  python evals/scripts/eval_splats.py --zarr data/outputs/pointcloud.zarr \
      --out evals/results/splats_tutorial --primitives 3dgs 2dgs --max-steps 30000 [--model-res]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.utils import confidence_mask
from collab_splats.preproc import frames as fr
from collab_splats.splats.rendering import load_checkpoint, render_views
from collab_splats.splats.trainer import SplatsConfig, train
from evals.scripts.eval_multiview_conf import load_7scenes_depth, median_align, retained_error

logger = logging.getLogger(__name__)

DEFAULT_CONF_PERCENTILE = 20.0  # matches base.yaml mesh.conf_percentile
ALPHA_THRESHOLD = 0.5  # rendered pixels below this have no surface to score
AUTO_IMAGES_DIR = "auto"  # sentinel: use <zarr>.parent / images when it exists, else model res


@dataclass
class SplatInputs:
    """
    Everything ``train()`` needs, pulled from one pointcloud.zarr.
    """

    images: np.ndarray  # (N, H, W, 3) uint8
    world_to_cam: np.ndarray  # (N, 4, 4) float32
    intrinsics: np.ndarray  # (N, 3, 3) float32, at the images' resolution
    points: np.ndarray  # (P, 3) float32
    colors: np.ndarray  # (P, 3) uint8
    depth_targets: np.ndarray | None  # (N, h, w) float32 model-res, 0 = no target
    resolution: tuple[int, int]  # (H, W) of images


def _model_res_images(result: FeedforwardResult) -> np.ndarray:
    """
    The zarr's own model-res images as (N, H, W, 3) uint8.
    """
    # CHW float -> HWC uint8, normalizing the [0,1] vs [0,255] backend drift first
    images = result.images.numpy().transpose(0, 2, 3, 1)
    if images.max() <= 1.0:
        images = images * 255.0
    return images.round().clip(0, 255).astype(np.uint8)


def _native_images_and_intrinsics(
    result: FeedforwardResult, images_dir: Path, zarr_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Native frames from images/ in ``result.image_paths`` order, with K rescaled model-res -> native.

    - Mirrors ``Reconstructor.splats()`` for the frames and ``_rescale_reconstruction_to_original_dimensions``
      for K: per frame ``scale = orig / model`` from ``original_coords[i, -2:]``; no crop offset.
    - Raises ValueError when the images/ resolution differs from the zarr's recorded original size.
    """
    # Frames looked up by the frame index encoded in each image name; read_frames returns them
    # in the order the indices are given, so the stack stays aligned with result.image_paths
    frame_indices = [fr.frame_idx_from_path(path) for path in result.image_paths]
    images = fr.read_frames(images_dir, idxs=frame_indices)
    frame_height, frame_width = images.shape[1:3]

    # Every frame's recorded original size must be the images/ size, else K would be wrong
    original_sizes = result.original_coords[:, -2:]
    orig_w = original_sizes[:, 0]
    orig_h = original_sizes[:, 1]
    if np.any(orig_w != frame_width) or np.any(orig_h != frame_height):
        recorded = sorted({(int(h), int(w)) for h, w in zip(orig_h, orig_w)})
        raise ValueError(
            f"{images_dir} frames are (H, W) ({frame_height}, {frame_width}) but {zarr_path} "
            f"records original sizes (H, W) {recorded}"
        )

    # Per-frame anisotropic rescale of fx, cx (x) and fy, cy (y) from model res to native
    scale_x = (orig_w / result.model_width).astype(np.float32)
    scale_y = (orig_h / result.model_height).astype(np.float32)
    intrinsics = result.intrinsics.astype(np.float32).copy()
    intrinsics[:, 0, 0] *= scale_x
    intrinsics[:, 0, 2] *= scale_x
    intrinsics[:, 1, 1] *= scale_y
    intrinsics[:, 1, 2] *= scale_y
    return images, intrinsics


def inputs_from_pointcloud_zarr(
    path: Path,
    conf_percentile: float | None = DEFAULT_CONF_PERCENTILE,
    images_dir: Path | str | None = AUTO_IMAGES_DIR,
) -> SplatInputs:
    """
    Load a pointcloud.zarr into trainer inputs.

    - images_dir: ``AUTO_IMAGES_DIR`` uses ``path.parent / images`` when it exists (native
      resolution, as the pipeline's splats stage does); an explicit Path forces it; None trains on
      the zarr's model-res images. K always matches the chosen images' resolution.
    - model-res images: stored (N, 3, H, W) float; VGGT writes [0, 255] while MapAnything writes
      [0, 1] (the known scale drift between backends), so a store whose max is <= 1 is rescaled.
    - depth targets: the model's depth masked by the learned-confidence percentile (0 = no target),
      or unmasked when ``conf_percentile`` is None. Always model-res; the trainer nearest-resizes.
    """
    path = Path(path)
    load_images = images_dir is None or images_dir == AUTO_IMAGES_DIR
    result = FeedforwardResult.load_zarr(path, load_images=load_images, load_world_points=False)

    # Resolve the sentinel: native when a sibling images/ exists, else fall back to model res
    if images_dir == AUTO_IMAGES_DIR:
        candidate = path.parent / "images"
        images_dir = candidate if candidate.is_dir() else None
        if images_dir is None:
            logger.info("no images/ beside %s; training at model resolution", path)

    # Images + matching K at the chosen resolution
    if images_dir is None:
        images = _model_res_images(result)
        intrinsics = result.intrinsics.astype(np.float32)
    else:
        images, intrinsics = _native_images_and_intrinsics(result, Path(images_dir), path)
    height, width = images.shape[1:3]
    source = "model-res zarr images" if images_dir is None else f"native frames from {images_dir}"
    logger.info(
        "training images: %s at %dx%d (model res %dx%d)", source, width, height, result.model_width, result.model_height
    )

    # Depth supervision from the model's own depth, zeroed where confidence is in the dropped tail
    depth_targets = None
    if result.depth is not None:
        depth_targets = result.depth.astype(np.float32)
        if conf_percentile is not None and result.confidence is not None:
            keep = confidence_mask(result.confidence.numpy(), conf_percentile)
            depth_targets = np.where(keep, depth_targets, 0.0).astype(np.float32)

    return SplatInputs(
        images=images,
        world_to_cam=result.extrinsics.astype(np.float32),
        intrinsics=intrinsics,
        points=result.points.astype(np.float32),
        colors=result.colors.astype(np.uint8),
        depth_targets=depth_targets,
        resolution=(int(height), int(width)),
    )


def summarise_run(out_dir: Path) -> dict:
    """
    One summary row from a run directory's splats_quality_report.json.
    """
    report_path = Path(out_dir) / "splats_quality_report.json"
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    config = summary["config"]
    max_steps = config["max_steps"]
    return {
        "primitive": config["primitive"],
        "max_steps": max_steps,
        "psnr": summary["psnr"],
        "ssim": summary["ssim"],
        "n_gaussians": summary["n_gaussians"],
        "seconds": summary["seconds"],
        "ms_per_step": 1000.0 * summary["seconds"] / max(max_steps, 1),
        "final_losses": summary["final_losses"],
    }


def depth_vs_gt(out_dir: Path, seq: Path) -> dict:
    """
    Rendered-depth error against 7-Scenes GT where alpha > ALPHA_THRESHOLD, after median alignment.
    """
    # Render rather than read a stored render: ckpt.pt is the only artifact the stage writes.
    # Unlike the mesh adapter this one keeps the whole stack — the median alignment below is
    # global across frames, so it cannot be computed one view at a time.
    model, camera_opt, cam_to_world, intrinsics, image_ids, (height, width) = load_checkpoint(
        Path(out_dir) / "ckpt.pt", "cuda"
    )
    n_frames = len(image_ids)
    depth = np.empty((n_frames, height, width), dtype=np.float32)
    alpha = np.empty((n_frames, height, width), dtype=np.float32)
    renders = render_views(model, camera_opt, cam_to_world, intrinsics, height, width)
    for view, render in enumerate(renders):
        depth[view] = render["depth"][0, ..., 0].cpu().numpy()
        alpha[view] = render["alpha"][0, ..., 0].cpu().numpy()

    # GT on the rendered pixel grid (nearest — never interpolate across depth discontinuities)
    color_paths = sorted(seq.glob("*.color.png"))[:n_frames]
    gt = load_7scenes_depth(color_paths)
    if gt.shape[1:] != depth.shape[1:]:
        height, width = depth.shape[1:]
        gt = np.stack([np.asarray(Image.fromarray(g).resize((width, height), Image.NEAREST)) for g in gt])

    # Non-metric render -> GT units, then score only covered pixels
    keep = (alpha > ALPHA_THRESHOLD) & (depth > 0)
    both = keep & (gt > 0)
    scale = median_align(depth, gt, both)
    stats = retained_error(depth * scale, gt, keep)
    stats["scale"] = scale
    return stats


def _markdown_table(rows: list[dict]) -> str:
    """
    Rows as a markdown table for pasting into the measured report.
    """
    lines = [
        "| primitive | res | steps | psnr | ssim | gaussians | seconds | ms/step |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        height, width = row["resolution"]
        lines.append(
            f"| {row['primitive']} | {width}x{height} | {row['max_steps']} | {row['psnr']:.2f} | {row['ssim']:.4f} "
            f"| {row['n_gaussians']} | {row['seconds']:.1f} | {row['ms_per_step']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", type=Path, required=True, help="pointcloud.zarr from the backend run")
    ap.add_argument("--out", type=Path, required=True, help="output dir; one subdir per primitive")
    ap.add_argument("--primitives", nargs="+", default=["3dgs", "2dgs"])
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument(
        "--conf-percentile",
        type=float,
        default=DEFAULT_CONF_PERCENTILE,
        help="learned-confidence percentile below which depth targets are dropped",
    )
    ap.add_argument("--seq", type=Path, default=None, help="7-Scenes sequence dir with .depth.png (optional)")
    ap.add_argument("--pose-opt", action="store_true")
    ap.add_argument("--images-dir", type=Path, default=None, help="images/ override (default: beside --zarr)")
    ap.add_argument("--model-res", action="store_true", help="ignore images/; train on the zarr's model-res images")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Inputs are shared across primitives; only the config differs per run
    # --model-res wins over --images-dir; neither means the auto sentinel (sibling images/)
    images_dir = None if args.model_res else (args.images_dir or AUTO_IMAGES_DIR)
    inputs = inputs_from_pointcloud_zarr(args.zarr, conf_percentile=args.conf_percentile, images_dir=images_dir)
    n_frames, height, width, _ = inputs.images.shape
    n_points = inputs.points.shape[0]
    logger.info("%d frames at %dx%d, %d init points", n_frames, width, height, n_points)

    rows = []
    for primitive in args.primitives:
        out_dir = args.out / primitive
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = SplatsConfig(primitive=primitive, max_steps=args.max_steps, pose_opt=args.pose_opt)
        logger.info("training %s for %d steps -> %s", primitive, args.max_steps, out_dir)
        train(
            cfg,
            inputs.images,
            inputs.world_to_cam,
            inputs.intrinsics,
            inputs.points,
            inputs.colors,
            out_dir,
            depth_targets=inputs.depth_targets,
        )

        # Read the trainer's own report back; GT depth is an optional extra column
        row = summarise_run(out_dir)
        row["resolution"] = list(inputs.resolution)
        if args.seq is not None:
            row["depth_vs_gt"] = depth_vs_gt(out_dir, args.seq)
        rows.append(row)
        logger.info(
            "%s: psnr=%.2f ssim=%.4f gaussians=%d seconds=%.1f ms/step=%.2f",
            row["primitive"],
            row["psnr"],
            row["ssim"],
            row["n_gaussians"],
            row["seconds"],
            row["ms_per_step"],
        )

    summary_path = args.out / "summary.json"
    summary = {"zarr": str(args.zarr), "resolution": list(inputs.resolution), "rows": rows}
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", summary_path)
    print(_markdown_table(rows))


if __name__ == "__main__":
    main()
