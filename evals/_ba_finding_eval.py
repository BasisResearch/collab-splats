"""Ad-hoc A/B harness for BA findings (vis_thresh, track budget, fine_tracking).

Runs VGGTX inference ONCE on a 7scenes seq-01 slice, then sweeps BA variants
that re-extract tracks + re-run BA from the cached raw_outputs. Tracks are
cached by (max_query_pts, query_frame_num, fine_tracking) so duplicate extractions
across variants are skipped. Variant E (reference-matched) is omitted: prior
50-frame run showed it is BA-degenerate (loss plateau, AUC=0).

Usage:
    python evals/_ba_finding_eval.py --seq_dir evals/data/7scenes/chess/chess/seq-01 \
        --max_frames 100 --out evals/results/_ba_findings.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustmentConfig,
    extract_tracks_vggsfm,
    run_bundle_adjustment,
)
from collab_splats.pointcloud.feedforward import VGGTXCreator, _raw_to_world_points
from collab_splats.pointcloud.feedforward.base import _extrinsics_3x4_to_4x4
from collab_splats.geometry.loop_closure.eval import (
    ate_translation,
    auc_at_threshold,
    rpe,
)
from datasets import get_dataset


def _prepare_image_dir(image_paths, root: Path) -> Path:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="ba_finding_", dir=str(root)))
    for i, src in enumerate(image_paths):
        (tmp / f"{i:06d}.png").symlink_to(src.resolve())
    return tmp


def _extract_tracks_cached(
    raw_outputs: dict,
    extrinsic_3x4: np.ndarray,
    *,
    max_query_pts: int,
    query_frame_num: int,
    fine_tracking: bool,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    key = (max_query_pts, query_frame_num, fine_tracking)
    if key in cache:
        tracks, vis_scores, pts3d_kp = cache[key]
        return tracks, vis_scores, pts3d_kp, 0.0

    model_h, model_w = int(raw_outputs["depth"].shape[1]), int(
        raw_outputs["depth"].shape[2]
    )
    wp_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
    N = extrinsic_3x4.shape[0]
    world_pts = wp_flat.reshape(N, model_h, model_w, 3)
    conf = torch.from_numpy(raw_outputs["depth_conf"])

    t0 = time.perf_counter()
    tracks, vis_scores, pts3d_kp = extract_tracks_vggsfm(
        raw_outputs["images"],
        conf,
        world_pts,
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        fine_tracking=fine_tracking,
    )
    t_track = time.perf_counter() - t0
    cache[key] = (tracks, vis_scores, pts3d_kp)
    return tracks, vis_scores, pts3d_kp, t_track


def _ba_variant(
    *,
    raw_outputs: dict,
    extrinsic_3x4: np.ndarray,
    intrinsic: np.ndarray,
    label: str,
    max_query_pts: int,
    query_frame_num: int,
    fine_tracking: bool,
    vis_thresh: float | None,
    max_reproj_error: float,
    gt_poses_4x4: np.ndarray,
    cache: dict,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_h, model_w = int(raw_outputs["depth"].shape[1]), int(
        raw_outputs["depth"].shape[2]
    )

    tracks, vis_scores, pts3d_kp, t_track = _extract_tracks_cached(
        raw_outputs,
        extrinsic_3x4,
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        fine_tracking=fine_tracking,
        cache=cache,
    )

    n_tracks_raw = int((vis_scores > 0).sum())
    if vis_thresh is None:
        vis_mask = vis_scores.astype(bool)
    else:
        vis_mask = vis_scores > vis_thresh
    n_tracks_kept = int(vis_mask.sum())

    t0 = time.perf_counter()
    cfg = BundleAdjustmentConfig()
    _, refined_ext_3x4, _ = run_bundle_adjustment(
        pts3d_kp,
        extrinsic_3x4,
        intrinsic,
        tracks,
        vis_mask,
        image_size=(model_h, model_w),
        max_reproj_error=max_reproj_error,
        lm_steps=cfg.lm_steps,
        shared_camera=cfg.shared_camera,
        min_inliers_per_frame=cfg.min_inliers_per_frame,
    )
    t_ba = time.perf_counter() - t0

    refined_4x4 = _extrinsics_3x4_to_4x4(refined_ext_3x4)
    ate = ate_translation(refined_4x4, gt_poses_4x4)
    rp = rpe(refined_4x4, gt_poses_4x4)
    auc = auc_at_threshold(refined_4x4, gt_poses_4x4)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "label": label,
        "max_query_pts": max_query_pts,
        "query_frame_num": query_frame_num,
        "fine_tracking": fine_tracking,
        "vis_thresh": vis_thresh,
        "max_reproj_error": max_reproj_error,
        "n_tracks_raw": n_tracks_raw,
        "n_tracks_kept": n_tracks_kept,
        "ate_rmse_m": ate["rmse"],
        "ate_median_m": ate["median"],
        "rpe_trans_m": rp["trans_rmse"],
        "rpe_rot_deg": rp["rot_rmse_deg"],
        "auc_30": auc["auc_30"],
        "t_track_s": round(t_track, 2),
        "t_ba_s": round(t_ba, 2),
        "peak_gb": round(peak_gb, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="7scenes")
    ap.add_argument("--seq_dir", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dataset = get_dataset(args.dataset)(args.seq_dir, max_frames=args.max_frames)
    print(f"Loaded {len(dataset.images)} frames")
    gt_poses_4x4 = dataset.gt_poses.astype(np.float32)

    tmp_dir = _prepare_image_dir(dataset.images, root=Path("/tmp"))

    print("Running VGGT-X inference once...")
    creator = VGGTXCreator()
    creator.load_model()
    creator.setup_inference(tmp_dir)
    t0 = time.perf_counter()
    creator.run_inference()
    print(f"  VGGT-X inference: {time.perf_counter()-t0:.1f}s")

    raw = creator.raw_outputs
    extrinsic_3x4 = raw["extrinsic"]
    intrinsic = raw.get("intrinsics_downsampled", raw.get("intrinsics"))

    extrinsic_4x4 = _extrinsics_3x4_to_4x4(extrinsic_3x4)
    base_ate = ate_translation(extrinsic_4x4, gt_poses_4x4)
    base_rpe = rpe(extrinsic_4x4, gt_poses_4x4)
    base_auc = auc_at_threshold(extrinsic_4x4, gt_poses_4x4)
    baseline_metrics = {
        "label": "baseline_no_ba",
        "ate_rmse_m": base_ate["rmse"],
        "ate_median_m": base_ate["median"],
        "rpe_trans_m": base_rpe["trans_rmse"],
        "rpe_rot_deg": base_rpe["rot_rmse_deg"],
        "auc_30": base_auc["auc_30"],
    }
    print(f"BASELINE (no BA): ATE={base_ate['rmse']:.4f}m RPE={base_rpe['trans_rmse']:.4f}m AUC30={base_auc['auc_30']:.1f}")

    # Order variants so cache hits maximise: shared (mqp, qfn, fine_tracking) runs consecutively.
    variants = [
        # A: current defaults (ours: max_reproj=4, no vis_thresh)
        dict(label="A_current_defaults", max_query_pts=2048, query_frame_num=5,
             fine_tracking=False, vis_thresh=None, max_reproj_error=4.0),
        # A2: ours but with reference max_reproj=8
        dict(label="A2_reproj8", max_query_pts=2048, query_frame_num=5,
             fine_tracking=False, vis_thresh=None, max_reproj_error=8.0),
        # B: reference vis_thresh=0.2 + reproj=8 + ours budget/fine
        dict(label="B_vis0p2_reproj8", max_query_pts=2048, query_frame_num=5,
             fine_tracking=False, vis_thresh=0.2, max_reproj_error=8.0),
        # F2: reference filter cascade (vis_thresh=0.2, reproj=8, 4096/8) WITHOUT fine_tracking.
        # F (with fine_tracking=True) skipped — 50-frame run showed catastrophic BA + ~30min track time.
        dict(label="F2_ref_no_fine", max_query_pts=4096, query_frame_num=8,
             fine_tracking=False, vis_thresh=0.2, max_reproj_error=8.0),
    ]

    results = [baseline_metrics]
    cache: dict = {}
    for v in variants:
        print(f"\n--- {v['label']} ---")
        m = _ba_variant(
            raw_outputs=raw,
            extrinsic_3x4=extrinsic_3x4,
            intrinsic=intrinsic,
            gt_poses_4x4=gt_poses_4x4,
            cache=cache,
            **v,
        )
        print(
            f"  kept {m['n_tracks_kept']}/{m['n_tracks_raw']} | "
            f"ATE={m['ate_rmse_m']:.4f}m RPE={m['rpe_trans_m']:.4f}m AUC30={m['auc_30']:.1f} | "
            f"t_track={m['t_track_s']}s t_ba={m['t_ba_s']}s peak={m['peak_gb']}GB"
        )
        results.append(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
