# evals/eval_vggt_slam_comparison.py
"""7-Scenes comparison: baseline / lc_se3 / lc_sl4 / vggt_slam_oob.

Usage (run in tmux — GPU + memory intensive):
    /opt/conda/envs/nerfstudio/bin/python evals/eval_vggt_slam_comparison.py \
        --scene chess --seq seq-01 --data_root /data/7scenes

Requires evo: pip install evo
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

VGGT_SLAM_ARGS = [
    "--max_loops", "1",
    "--min_disparity", "50",
    "--conf_threshold", "25",
    "--lc_thres", "0.95",
    "--submap_size", "16",
    "--skip_dense_log",
]


def load_7scenes_images(scene_dir: Path) -> list[Path]:
    """Return sorted list of RGB image paths from a 7-Scenes sequence dir."""
    frames = sorted(scene_dir.glob("frame-*.color.png"))
    if not frames:
        raise FileNotFoundError(f"No frame-*.color.png in {scene_dir}")
    return frames


def load_7scenes_gt_poses(scene_dir: Path) -> np.ndarray:
    """Load ground-truth camera-to-world poses from 7-Scenes .pose.txt files.
    Returns (N, 4, 4) float32 world-to-cam (inverted from the stored cam-to-world).
    """
    pose_files = sorted(scene_dir.glob("frame-*.pose.txt"))
    poses = []
    for pf in pose_files:
        mat = np.loadtxt(pf, dtype=np.float64).reshape(4, 4)
        poses.append(np.linalg.inv(mat).astype(np.float32))
    return np.stack(poses)


def compute_ate(est_poses: np.ndarray, gt_poses: np.ndarray, tmp_dir: Path) -> float:
    """Write TUM-format trajectories and call evo_ape; return ATE RMSE (m)."""
    def _to_tum(poses: np.ndarray, path: Path) -> None:
        with open(path, "w") as f:
            for i, P in enumerate(poses):
                t = P[:3, 3]
                from scipy.spatial.transform import Rotation
                q = Rotation.from_matrix(P[:3, :3]).as_quat()  # xyzw
                f.write(f"{i} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    est_path = tmp_dir / "est.txt"
    gt_path = tmp_dir / "gt.txt"
    _to_tum(est_poses, est_path)
    _to_tum(gt_poses, gt_path)

    result = subprocess.run(
        ["evo_ape", "tum", str(gt_path), str(est_path), "--align", "--correct_scale",
         "--no_warnings", "--save_results", str(tmp_dir / "ape.zip")],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "rmse" in line.lower():
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    log.warning("evo_ape output: %s", result.stdout[-500:])
    return float("nan")


def run_baseline(image_paths: list[Path], output_dir: Path) -> np.ndarray:
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    creator = VGGTXCreator(image_paths=image_paths)
    result = creator.reconstruct(image_dir=image_paths[0].parent, output_dir=output_dir)
    return result.extrinsics  # (N, 4, 4)


def run_lc(
    image_paths: list[Path],
    output_dir: Path,
    manifold: str,
) -> np.ndarray:
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure.closure import LoopClosureConfig
    base = VGGTXCreator(image_paths=image_paths)
    cfg = LoopClosureConfig(manifold=manifold)
    lc = LoopClosure(base=base, config=cfg)
    result = lc.reconstruct(image_dir=image_paths[0].parent, output_dir=output_dir)
    return result.extrinsics


def run_vggt_slam_oob(image_paths: list[Path], output_dir: Path) -> np.ndarray:
    """Run VGGT-SLAM out-of-the-box via subprocess; parse TUM log -> extrinsics."""
    from collab_splats.pointcloud.loop_closure.graph import decompose_camera

    # Rename images to %06d.png format (issue #43 workaround)
    renamed_dir = output_dir / "renamed_frames"
    renamed_dir.mkdir(exist_ok=True)
    import shutil
    for i, src in enumerate(image_paths):
        dst = renamed_dir / f"{i:06d}.png"
        if not dst.exists():
            shutil.copy(src, dst)

    log_path = output_dir / "vggt_slam_poses.txt"
    slam_main = Path(__file__).parents[1] / "third_party" / "VGGT-SLAM" / "main.py"
    cmd = [
        sys.executable, str(slam_main),
        "--image_folder", str(renamed_dir),
        "--log_results", "--log_path", str(log_path),
    ] + VGGT_SLAM_ARGS
    log.info("Running VGGT-SLAM: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Parse TUM-format log: timestamp tx ty tz qx qy qz qw
    from scipy.spatial.transform import Rotation
    poses = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]])  # xyzw
            R = Rotation.from_quat(q).as_matrix().astype(np.float32)
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = R
            mat[:3, 3] = t.astype(np.float32)
            poses.append(mat)
    return np.stack(poses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--seq", default="seq-01")
    parser.add_argument("--data_root", type=Path, default=Path("/data/7scenes"))
    parser.add_argument("--output_dir", type=Path, default=Path("evals/results/sl4_comparison"))
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "lc_se3", "lc_sl4", "vggt_slam_oob"])
    args = parser.parse_args()

    scene_dir = args.data_root / args.scene / args.seq
    output_dir = args.output_dir / args.scene / args.seq
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_7scenes_images(scene_dir)
    gt_poses = load_7scenes_gt_poses(scene_dir)
    log.info("Loaded %d frames from %s", len(image_paths), scene_dir)

    results: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if "baseline" in args.conditions:
            log.info("Running baseline...")
            est = run_baseline(image_paths, output_dir / "baseline")
            results["baseline"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("baseline ATE RMSE: %.4f m", results["baseline"])

        if "lc_se3" in args.conditions:
            log.info("Running lc_se3...")
            est = run_lc(image_paths, output_dir / "lc_se3", manifold="se3")
            results["lc_se3"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("lc_se3 ATE RMSE: %.4f m", results["lc_se3"])

        if "lc_sl4" in args.conditions:
            log.info("Running lc_sl4...")
            est = run_lc(image_paths, output_dir / "lc_sl4", manifold="sl4")
            results["lc_sl4"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("lc_sl4 ATE RMSE: %.4f m", results["lc_sl4"])

        if "vggt_slam_oob" in args.conditions:
            log.info("Running vggt_slam_oob...")
            est = run_vggt_slam_oob(image_paths, output_dir / "vggt_slam_oob")
            results["vggt_slam_oob"] = compute_ate(est, gt_poses[:len(est)], tmp_path)
            log.info("vggt_slam_oob ATE RMSE: %.4f m", results["vggt_slam_oob"])

    results_file = output_dir / "ate_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n=== ATE RMSE (m) — %s/%s ===", args.scene, args.seq)
    for cond, ate in results.items():
        log.info("  %-20s %.4f", cond, ate)
    log.info("Saved to %s", results_file)


if __name__ == "__main__":
    main()
