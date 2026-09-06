"""Measure geometric verification on a reconstructed scene against 7-Scenes ground truth.

Reports (all distributions median/p90/p99 + out10 where applicable):
  Tier 1: per-pair rotation / translation-direction errors — estimated-vs-model,
          and (with --gt_dir) model-vs-GT for the same pairs.
  Tier 2: triangulated-vs-model relative depth agreement at track pixels (scale-free,
          reference-free control), triangulated-vs-GT and model-vs-GT after one global
          median scale, and out10 for each.
  Yield:  point count, track lengths, per-frame survival (from verification.json).
  Negative control: --perturb_deg rotates every 5th pose; those frames must be flagged.

Usage (tmux, never a notebook):
  /opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py \
      --backend_dir data/outputs/<scene>/vggt_omega \
      --gt_dir /data/7scenes/chess/seq-01 \
      --extractor xfeat --out evals/results/verification/chess_xfeat.json
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from collab_splats.geometry.verification import (
    _pair_pose_errors,
    verify_reconstruction,
)
from collab_splats.localization.extractors import LocalMatcher
from collab_splats.localization.localizer import load_localization_db
from collab_splats.pointcloud.feedforward.base import (
    FeedforwardResult,
    build_pycolmap_reconstruction,
)
from collab_splats.preproc import frames as fr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_7SCENES_INVALID_DEPTH = 65535  # sentinel in 7-Scenes depth PNGs (uint16 millimetres)


def _rot_x(deg: float) -> np.ndarray:
    """Rotation about x, degrees."""
    a = np.radians(deg)
    return np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float64
    )


def _recon_to_arrays(recon: pycolmap.Reconstruction):
    """Extract (extrinsics (N,3,4), intrinsics (N,3,3), names, W, H) from a reconstruction."""
    ids = sorted(recon.images)
    extr, intr, names = [], [], []
    for iid in ids:
        im = recon.images[iid]
        cam = recon.cameras[im.camera_id]
        extr.append(im.cam_from_world().matrix())  # (3, 4)
        fx, fy, cx, cy = cam.params  # PINHOLE
        intr.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
        names.append(im.name)
    cam0 = recon.cameras[recon.images[ids[0]].camera_id]
    return np.stack(extr).astype(np.float32), np.stack(intr), names, cam0.width, cam0.height


def _perturb(recon: pycolmap.Reconstruction, deg: float) -> tuple[pycolmap.Reconstruction, list[str]]:
    """Rebuild the reconstruction with every 5th pose rotated by deg (negative control)."""
    extr, intr, names, w, h = _recon_to_arrays(recon)
    bad = list(range(0, len(names), 5))
    for i in bad:
        extr[i, :3, :3] = (_rot_x(deg) @ extr[i, :3, :3]).astype(np.float32)
    out = build_pycolmap_reconstruction(
        pts3d=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        extrinsics=extr,
        intrinsics=intr,
        image_width=w,
        image_height=h,
        image_names=names,
    )
    return out, [names[i] for i in bad]


def _load_gt_poses(gt_dir: Path, frame_indices) -> dict[str, np.ndarray]:
    """7-Scenes frame-XXXXXX.pose.txt (c2w) -> w2c 4x4, keyed by reconstruction image name."""
    poses = {}
    for fi in frame_indices:
        p = gt_dir / f"frame-{int(fi):06d}.pose.txt"
        c2w = np.loadtxt(p).reshape(4, 4)
        poses[f"frame_{int(fi):06d}"] = np.linalg.inv(c2w)
    return poses


def _gt_depth(gt_dir: Path, frame_idx: int) -> np.ndarray:
    """7-Scenes depth PNG in metres, invalid pixels = nan.

    Kinect depth is registered to color only approximately — the GT columns are a noise
    floor, not truth; triangulated_vs_model_rel is the primary (reference-free) signal.
    """
    d = cv2.imread(str(gt_dir / f"frame-{int(frame_idx):06d}.depth.png"), cv2.IMREAD_UNCHANGED)
    d = d.astype(np.float64)
    d[d == _7SCENES_INVALID_DEPTH] = np.nan
    d[d == 0] = np.nan
    return d / 1000.0


def _dist(v) -> dict | None:
    """median/p90/p99 (+out10 as the fraction > 0.10 for relative errors)."""
    v = np.asarray(list(v), dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "out10": float(np.mean(v > 0.10)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend_dir", type=Path, required=True, help="e.g. .../<scene>/vggt_omega")
    ap.add_argument("--gt_dir", type=Path, default=None, help="7-Scenes seq dir (poses + depth)")
    ap.add_argument("--extractor", default="xfeat", help="vismatch model name (LocalMatcher)")
    ap.add_argument("--overlap", type=int, default=10, help="sequential pairing window")
    ap.add_argument("--perturb_deg", type=float, default=0.0, help="negative control rotation")
    ap.add_argument("--out", type=Path, required=True, help="results JSON path")
    args = ap.parse_args()

    # ── Load reconstruction (pose authority) + feature cache + model depth ──
    recon = pycolmap.Reconstruction()
    recon.read(str(args.backend_dir / "colmap" / "sparse" / "0"))
    features, ids, _ = load_localization_db(
        args.backend_dir / "pointcloud.zarr", args.extractor
    )
    ff = FeedforwardResult.load_zarr(args.backend_dir / "pointcloud.zarr")
    images_dir = args.backend_dir.parent / "images"
    frame_indices = np.array([fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)])

    perturbed_names: list[str] = []
    if args.perturb_deg:
        recon, perturbed_names = _perturb(recon, args.perturb_deg)

    # ── Run verification into a results-local dir (never into the scene's colmap/) ──
    work_dir = args.out.parent / (args.out.stem + "_work")
    matcher = LocalMatcher(args.extractor)
    result = verify_reconstruction(
        recon=recon, features=features, matcher=matcher, output_dir=work_dir, overlap=args.overlap
    )
    report: dict = {
        "backend_dir": str(args.backend_dir),
        "extractor": args.extractor,
        "overlap": args.overlap,
        "perturb_deg": args.perturb_deg,
        "perturbed_frames": perturbed_names,
        "summary": result.summary,
        "frame_stats": result.frame_stats,
    }

    # ── Tier 1 vs GT: same pairs, three comparisons ──
    if args.gt_dir is not None:
        gt_w2c = _load_gt_poses(args.gt_dir, frame_indices)
        name_to_image = {recon.images[i].name: recon.images[i] for i in recon.images}
        est_vs_model, model_vs_gt = [], []
        for p in result.pair_stats:
            if np.isnan(p.rot_error_deg):
                continue
            # GT relative pose for the same pair, in w2c convention
            T_rel = gt_w2c[p.name2] @ np.linalg.inv(gt_w2c[p.name1])
            gt_rel = pycolmap.Rigid3d(
                pycolmap.Rotation3d(T_rel[:3, :3]), T_rel[:3, 3]
            )
            im1, im2 = name_to_image[p.name1], name_to_image[p.name2]
            model_rel = im2.cam_from_world() * im1.cam_from_world().inverse()
            model_vs_gt.append(_pair_pose_errors(model_rel, gt_rel)[0])
            est_vs_model.append(p.rot_error_deg)
        # Two columns tell the story together: if estimated-vs-model tracks model-vs-GT
        # pair-by-pair, the epipolar estimate is seeing the same pose errors GT sees.
        report["tier1"] = {
            "model_vs_gt_rot_deg": _dist(model_vs_gt),
            "estimated_vs_model_rot_deg": _dist(est_vs_model),
        }

    # ── Tier 2 depth accuracy at track pixels ──
    verified = result.reconstruction
    name_to_pos = {f"frame_{int(fi):06d}": k for k, fi in enumerate(frame_indices)}
    model_h, model_w = ff.depth.shape[1:3]
    cam0 = recon.cameras[recon.images[sorted(recon.images)[0]].camera_id]
    sx, sy = model_w / cam0.width, model_h / cam0.height  # original-res px -> model grid

    tri_vs_model, tri_vs_gt_raw, model_vs_gt_raw = [], [], []
    for point in verified.points3D.values():
        for el in point.track.elements:
            image = verified.images[el.image_id]
            pos = name_to_pos[image.name]
            px = image.points2D[el.point2D_idx].xy  # original-res [x, y]
            # Triangulated depth: point through this frame's (model) pose
            z_tri = (image.cam_from_world() * point.xyz)[2]
            # Model depth: nearest sample on the model-resolution grid
            mx, my = int(round(px[0] * sx)), int(round(px[1] * sy))
            if not (0 <= mx < model_w and 0 <= my < model_h):
                continue
            z_model = float(ff.depth[pos, my, mx])
            if z_model <= 0 or z_tri <= 0:
                continue
            tri_vs_model.append(abs(z_tri - z_model) / z_model)
            if args.gt_dir is not None:
                gt = _gt_depth(args.gt_dir, frame_indices[pos])
                gy, gx = int(round(px[1])), int(round(px[0]))
                if 0 <= gy < gt.shape[0] and 0 <= gx < gt.shape[1] and np.isfinite(gt[gy, gx]):
                    tri_vs_gt_raw.append((z_tri, gt[gy, gx]))
                    model_vs_gt_raw.append((z_model, gt[gy, gx]))

    report["tier2_depth"] = {"triangulated_vs_model_rel": _dist(tri_vs_model)}
    if tri_vs_gt_raw:
        # One GLOBAL median scale (backbones are non-metric); per-frame scaling would
        # hide exactly the per-frame pose errors we are trying to see.
        s = float(np.median([g / z for z, g in model_vs_gt_raw]))
        report["tier2_depth"]["global_scale"] = s
        report["tier2_depth"]["triangulated_vs_gt_rel"] = _dist(
            abs(z * s - g) / g for z, g in tri_vs_gt_raw
        )
        report["tier2_depth"]["model_vs_gt_rel"] = _dist(
            abs(z * s - g) / g for z, g in model_vs_gt_raw
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    logger.info("Report: %s", args.out)
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report.get("tier2_depth", {}), indent=2))


if __name__ == "__main__":
    main()
