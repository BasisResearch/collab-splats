"""Subprocess wrapper around VGGT-SLAM's main.py.

Runs VGGT-SLAM using the current Python interpreter (reconstruction env,
Python 3.11 — satisfies VGGT-SLAM's SL(4)/GTSAM requirement).

Two modes, selected by ``--max_loops``:
  * ``--max_loops 0`` (default) — the published no-LC baseline. Writes the dense
    TUM trajectory.
  * ``--max_loops >0`` — loop-closure run. In addition to the TUM, writes
    ``selected_frames.txt`` (the frames fed to VGGT-SLAM, for eval.py
    ``--keyframe_list`` parity).

ATE is not computed here — drop the resulting TUM into a results dir and let
``eval.py`` / ``eval_compare`` score it against GT as a comparison row.

On success, copies the output TUM file to output_tum and removes the
corresponding .pending sentinel from evals/baselines/vggt_slam/ if present.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

REPO_ROOT = Path(__file__).resolve().parents[2]
VGGTSLAM_DIR = REPO_ROOT / "third_party" / "VGGT-SLAM"
BASELINES_DIR = REPO_ROOT / "evals" / "baselines" / "vggt_slam"


def _prepare_image_dir(
    image_dir: Path,
    max_frames: int | None = None,
    image_list: Path | None = None,
) -> tuple[Path, list[Path]]:
    """Symlink images into a temp dir as 000000.png, 000001.png, ... (sorted).

    VGGT-SLAM requires sequentially named images; 7-Scenes uses
    frame-NNNNNN.color.png which causes it to silently fail or produce garbage.
    Excludes depth images (*.depth.png) so 7-Scenes dirs are handled correctly.
    Returns the temp dir and the ordered list of source image paths fed to it.
    """
    all_imgs = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    # Exclude depth images: any file whose stem ends in '.depth' (e.g. frame-000000.depth.png)
    images = [p for p in all_imgs if not p.stem.endswith(".depth")]
    # Optional frame-universe restriction (TUM GT-gap parity): keep only listed basenames
    if image_list is not None:
        allowed = {ln.strip() for ln in Path(image_list).read_text().splitlines() if ln.strip()}
        images = [p for p in images if p.name in allowed]
    if max_frames is not None:
        images = images[:max_frames]
    tmp = Path(tempfile.mkdtemp(prefix="vggtslam_imgs_"))
    for i, src in enumerate(images):
        (tmp / f"{i:06d}.png").symlink_to(src.resolve())
    return tmp, images


def run_vggt_slam(
    image_dir: Path,
    output_tum: Path,
    submap_size: int = 16,
    max_loops: int = 0,
    max_frames: int | None = None,
    min_disparity: float = 50.0,
    conf_threshold: float = 25.0,
    lc_thres: float = 0.95,
    image_list: Path | None = None,
    python: str | None = None,
) -> Path:
    """Run VGGT-SLAM on image_dir; write trajectory to output_tum.

    Args:
        image_dir: Directory of input images (sorted, no GT required).
        output_tum: Destination path for the TUM trajectory file.
        submap_size: VGGT-SLAM submap window size (default 16).
        max_loops: Max loop closures per submap (0 = disable LC entirely). When
            >0, also writes selected_frames.txt (for eval.py --keyframe_list parity).
        max_frames: Cap number of input frames (None = all). Match eval_gt.py --max_frames.
        min_disparity: Optical-flow keyframe threshold; 0 = use all frames (default 50).
        conf_threshold: VGGT-SLAM init confidence threshold (default 25).
        lc_thres: DINO-SALAD retrieval threshold for LC candidates (default 0.95).
        image_list: File of allowed frame basenames (one per line) restricting the
            frame universe — required for TUM parity runs.
        python: Python binary to use. Defaults to sys.executable.

    Returns:
        output_tum path on success.
    """
    if not VGGTSLAM_DIR.is_dir():
        raise FileNotFoundError(
            f"VGGT-SLAM submodule not found at {VGGTSLAM_DIR}. "
            "Run: git submodule update --init third_party/VGGT-SLAM"
        )
    py = python or sys.executable
    # Resolve to absolute so the path is valid regardless of subprocess CWD
    log_path = output_tum.resolve().with_suffix(".vggtslam.txt")
    # Rename images to sequential 000000.png naming — VGGT-SLAM silently
    # breaks on non-standard filenames (e.g. frame-000000.color.png).
    tmp_img_dir, source_images = _prepare_image_dir(image_dir, max_frames=max_frames, image_list=image_list)
    cmd = [
        py,
        str(VGGTSLAM_DIR / "main.py"),
        "--image_folder",
        str(tmp_img_dir),
        "--max_loops",
        str(max_loops),
        "--min_disparity",
        str(min_disparity),
        "--conf_threshold",
        str(conf_threshold),
        "--lc_thres",
        str(lc_thres),
        "--submap_size",
        str(submap_size),
        "--log_results",
        "--skip_dense_log",
        "--log_path",
        str(log_path),
    ]
    output_tum.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, cwd=VGGTSLAM_DIR)
    finally:
        shutil.rmtree(tmp_img_dir, ignore_errors=True)
    if not log_path.is_file():
        raise RuntimeError(
            f"VGGT-SLAM finished but log not found at {log_path}. " "Check --log_path handling in VGGT-SLAM main.py."
        )
    shutil.copy2(log_path, output_tum)
    # Remove .pending sentinel for this sequence if present
    results_seq = output_tum.parent.name  # e.g. "chess_seq01"
    sentinel = BASELINES_DIR / f"{results_seq}.pending"
    if sentinel.is_file():
        sentinel.unlink()
        print(f"  Removed sentinel: {sentinel}")

    # LC parity output: the keyframe list actually fed to VGGT-SLAM, so eval.py
    # --keyframe_list can run our pipeline on the same frames. Guarded on max_loops>0
    # so the plain no-LC baseline stays a bare TUM. ATE is scored downstream by eval.py.
    if max_loops > 0:
        kf_path = output_tum.parent / "selected_frames.txt"
        kf_path.write_text("\n".join(str(p.resolve()) for p in source_images))
        print(f"  Keyframes ({len(source_images)}) → {kf_path}")

    return output_tum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--image_dir",
        "--seq_dir",
        dest="image_dir",
        type=Path,
        required=True,
        help="Directory of input images for VGGT-SLAM",
    )
    ap.add_argument("--output", "--out_tum", dest="output", type=Path, required=True, help="Output TUM trajectory path")
    ap.add_argument("--submap_size", type=int, default=16, help="VGGT-SLAM submap size (default 16)")
    ap.add_argument(
        "--max_loops",
        type=int,
        default=0,
        help="Max loop closures per submap; 0 disables LC (default 0). "
        ">0 also writes selected_frames.txt (for eval.py --keyframe_list parity)",
    )
    ap.add_argument("--max_frames", type=int, default=None, help="Limit input to first N frames (default: all)")
    ap.add_argument(
        "--min_disparity", type=float, default=50.0, help="Optical-flow keyframe threshold; 0 = all frames (default 50)"
    )
    ap.add_argument(
        "--conf_threshold", type=float, default=25.0, help="VGGT-SLAM init confidence threshold (default 25)"
    )
    ap.add_argument(
        "--lc_thres", type=float, default=0.95, help="DINO-SALAD retrieval threshold for LC candidates (default 0.95)"
    )
    ap.add_argument(
        "--image_list",
        type=Path,
        default=None,
        help="File of allowed frame basenames (one per line). Restricts the "
        "frame universe — required for TUM parity runs.",
    )
    ap.add_argument("--python", type=str, default=None, help="Python binary (default: sys.executable)")
    args = ap.parse_args()
    run_vggt_slam(
        args.image_dir,
        args.output,
        submap_size=args.submap_size,
        max_loops=args.max_loops,
        max_frames=args.max_frames,
        min_disparity=args.min_disparity,
        conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        image_list=args.image_list,
        python=args.python,
    )
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
