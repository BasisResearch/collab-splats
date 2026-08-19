"""Does BA's objective have its minimum at the true poses?

Runs the package's own BundleAdjustment twice on one reconstruction:

  A. initialised at the model's poses  (reproduces the shipping `ba` condition)
  B. initialised at ground truth, mapped into the reconstruction frame by Sim(3)

If B lowers the loss below its starting value while its ATE rises from ~0, BA is walking
away from the truth to satisfy its objective — the objective is wrong, not the optimizer.
If B's converged loss is far below A's and its ATE stays near 0, the truth is the better
optimum and A simply failed to reach it — an optimizer problem.

No edits to collab_splats/geometry/bundle_adjustment.py; both runs are plain `ba.refine`.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path("evals").resolve()))
sys.path.insert(0, str(Path("evals/scripts").resolve()))

from datasets import get_dataset  # noqa: E402

from collab_splats.geometry import BundleAdjustment, BundleAdjustmentConfig  # noqa: E402
from collab_splats.geometry.bundle_adjustment import (  # noqa: E402
    _compute_tracks_cache_key,
    _scale_intrinsics_to_model,
)
from collab_splats.geometry.loop_closure.eval import ate_translation  # noqa: E402
from collab_splats.geometry.transforms import umeyama_sim3  # noqa: E402
from collab_splats.pointcloud import get_creator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ba_start_at_gt")

SEQ = Path("data/7scenes/chess/seq-01")
OUT = Path("evals/results/ba_start_at_gt")
BACKBONE = "vggt_omega"
MAX_FRAMES = 100

# Stable image dir: eval.py symlinks into mkdtemp, and _compute_tracks_cache_key hashes
# image_paths, so a temp dir makes the track cache unhittable across runs. A fixed dir
# lets run B reuse run A's extraction.
IMG_DIR = OUT / "images"
CACHE = OUT / "cache"
# Extraction already done by the ba_convergence_chess run — same images, same knobs
SRC_TRACKS = Path("evals/results/ba_convergence_chess/cache/vggt_omega/tracks.zarr")


def cam_centres(poses_w2c: np.ndarray) -> np.ndarray:
    """(N,4,4) world-to-cam -> (N,3) camera centres in world."""
    R, t = poses_w2c[:, :3, :3], poses_w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


def gt_into_recon_frame(gt_w2c: np.ndarray, model_w2c: np.ndarray) -> np.ndarray:
    """Express GT poses in the reconstruction's frame via Sim(3) on camera centres.

    Reprojection is invariant to a global similarity, so this changes nothing about how
    well the tracks are explained — it only puts both pose sets in one frame so ATE and
    the landmark cloud stay comparable.
    """
    s, R_a, t_a = umeyama_sim3(cam_centres(gt_w2c), cam_centres(model_w2c))
    R_a, t_a = R_a.astype(np.float64), t_a.astype(np.float64)
    R, t = gt_w2c[:, :3, :3].astype(np.float64), gt_w2c[:, :3, 3].astype(np.float64)
    # X_recon = s R_a X_gt + t_a  =>  R' = R R_a^T,  t' = s t - R' t_a
    R_new = R @ R_a.T
    t_new = s * t - np.einsum("nij,j->ni", R_new, t_a)
    out = np.tile(np.eye(4, dtype=np.float64), (len(gt_w2c), 1, 1))
    out[:, :3, :3], out[:, :3, 3] = R_new, t_new
    logger.info("GT -> recon Sim(3): scale %.6f", s)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = get_dataset("7scenes")(SEQ, max_frames=MAX_FRAMES)
    logger.info("dataset: %d frames from %s", len(ds.images), SEQ)

    # Symlink into the stable dir under eval.py's naming
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(ds.images):
        link = IMG_DIR / f"{i:06d}.png"
        if not link.exists():
            link.symlink_to(src.resolve())

    # One inference pass; both BA runs share the reconstruction and the track cache
    creator = get_creator(BACKBONE)()
    creator.load_model()
    creator.setup_inference(IMG_DIR)
    creator.run_inference()
    creator.postprocess()
    result = creator.outputs
    logger.info("reconstruction ready: %d frames", len(result.images))

    gt = ds.gt_poses.astype(np.float64)
    model_poses = result.extrinsics.astype(np.float64)
    gt_recon = gt_into_recon_frame(gt, model_poses)

    # Sanity: GT expressed in the recon frame must score ~0 ATE against GT
    ate_gt_start = ate_translation(gt_recon.astype(np.float32), ds.gt_poses)
    logger.info("GT-in-recon-frame ATE against GT (must be ~0): %.3e", ate_gt_start["rmse"])

    # max_reproj_error=None skips the reprojection filter. Required for a fair A/B:
    # the filter gates on how well the *model-frame* landmarks reproject, so at GT poses
    # it cuts 99% of observations and the two runs would optimise different problems.
    no_filter = "--no-filter" in sys.argv
    cfg = BundleAdjustmentConfig(
        tracks_cache_dir=CACHE / BACKBONE,
        **({"max_reproj_error": None} if no_filter else {}),
    )
    logger.info("reprojection filter: %s", "OFF (same observation set for both runs)" if no_filter else "ON")

    # Adopt the existing extraction instead of repeating it. The cache key hashes
    # image_paths, and the original run symlinked into mkdtemp, so the on-disk tracks are
    # unreachable by key alone — same images, same extraction knobs, different temp paths.
    # Re-stamp the key after checking the arrays actually match this reconstruction.
    dst = CACHE / BACKBONE / "tracks.zarr"
    if not dst.exists() and SRC_TRACKS.exists():
        src = zarr.open(str(SRC_TRACKS), mode="r")
        n_src = src["tracks"].shape[0]
        if n_src != len(result.images):
            raise SystemExit(f"track cache has {n_src} frames, reconstruction has {len(result.images)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SRC_TRACKS, dst)
        zarr.open(str(dst), mode="r+").attrs["cache_key"] = _compute_tracks_cache_key(result, cfg)
        logger.info("adopted existing track cache (%d frames, %d points) from %s",
                    n_src, src["tracks"].shape[1], SRC_TRACKS)

    # Dump BA's exact inputs so a points-only evaluation uses the real model-resolution K
    # and pose arrays rather than reconstructing them. _scale_intrinsics_to_model is a
    # no-op for backends that already store model-res K, but read it, never assume it.
    ba_probe = BundleAdjustment(cfg)
    tracks, vis_scores, pts3d_tracks = ba_probe._load_or_extract_tracks(result)
    intr_model, *_ = _scale_intrinsics_to_model(result.intrinsics, result.images, result.original_coords)
    np.savez(
        OUT / "ba_inputs.npz",
        poses_model=model_poses, poses_gt_recon=gt_recon, intrinsics_model=intr_model,
        pts3d_tracks=pts3d_tracks, tracks=tracks, vis_scores=vis_scores,
    )
    logger.info("dumped BA inputs: K fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                intr_model[0, 0, 0], intr_model[0, 1, 1], intr_model[0, 0, 2], intr_model[0, 1, 2])

    report: dict = {"ate_gt_start_check": ate_gt_start["rmse"]}

    for name, start_poses in (("A_model_start", model_poses), ("B_gt_start", gt_recon)):
        ba = BundleAdjustment(cfg)
        start = replace(result, extrinsics=start_poses.astype(np.float32))
        ate_start = ate_translation(start.extrinsics.astype(np.float32), ds.gt_poses)
        logger.info("=== %s === starting ATE %.6f", name, ate_start["rmse"])

        refined = ba.refine(start)
        hist = ba._last_loss_history[-1] if ba._last_loss_history else []
        ate_end = ate_translation(refined.extrinsics.astype(np.float32), ds.gt_poses)

        # Pose movement from the starting point, in camera-centre distance
        moved = float(np.linalg.norm(cam_centres(refined.extrinsics.astype(np.float64))
                                     - cam_centres(start_poses), axis=1).mean())
        report[name] = {
            "ate_start": ate_start["rmse"],
            "ate_end": ate_end["rmse"],
            "loss_first": hist[0] if hist else None,
            "loss_last": hist[-1] if hist else None,
            "mean_centre_movement": moved,
        }
        logger.info(
            "%s: ATE %.6f -> %.6f | loss %.6e -> %.6e | mean centre movement %.6f",
            name, ate_start["rmse"], ate_end["rmse"],
            hist[0] if hist else float("nan"), hist[-1] if hist else float("nan"), moved,
        )

    (OUT / ("report_nofilter.json" if no_filter else "report.json")).write_text(json.dumps(report, indent=2))

    a, b = report["A_model_start"], report["B_gt_start"]
    print("\n================ VERDICT INPUTS ================")
    print(f"A model-start : ATE {a['ate_start']:.6f} -> {a['ate_end']:.6f}   "
          f"loss {a['loss_first']:.6e} -> {a['loss_last']:.6e}")
    print(f"B GT-start    : ATE {b['ate_start']:.6f} -> {b['ate_end']:.6f}   "
          f"loss {b['loss_first']:.6e} -> {b['loss_last']:.6e}")
    print(f"loss at truth (B start) vs BA's converged loss (A end): "
          f"{b['loss_first']:.6e} vs {a['loss_last']:.6e}")
    print(f"report: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
