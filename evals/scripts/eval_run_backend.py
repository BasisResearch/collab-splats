"""Reconstruct a 7-Scenes sequence with one feedforward backend and dump pointcloud.zarr.

Feeds the sweep in ``eval_multiview_conf.py``. Multiview confidence is deliberately OFF here:
the zarr must hold unfiltered depth so the sweep can apply every (rel_thresh, min_views)
setting itself.

Frame order is the contract with the sweep — the first ``--max-frames`` ``*.color.png`` in
filename order, no quality gate, no resampling. The sweep pairs GT depth to zarr frames by
that same order.

CLI/tmux only. Results under evals/results/ (gitignored).

Usage:
  python evals/scripts/eval_run_backend.py --backend vggt_omega \
      --seq data/7scenes/chess/seq-01 --out evals/results/mv_vggt_omega --max-frames 60
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTOmegaCreator,
    VGGTXCreator,
)

logger = logging.getLogger(__name__)

CREATORS = {
    "vggt_omega": VGGTOmegaCreator,
    "vggtx": VGGTXCreator,
    "mapanything": MapAnythingCreator,
}


def stage_color_frames(seq: Path, staged: Path, max_frames: int) -> list[Path]:
    """Symlink the first max_frames *.color.png into their own dir.

    The legacy image-dir decode globs every .png, which would swallow the .depth.png
    siblings sitting next to them.
    """
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    paths = sorted(seq.glob("*.color.png"))[:max_frames]
    if not paths:
        raise FileNotFoundError(f"no *.color.png under {seq}")
    for p in paths:
        (staged / p.name).symlink_to(p.resolve())
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(CREATORS))
    ap.add_argument("--seq", type=Path, required=True, help="7-Scenes sequence dir")
    ap.add_argument("--out", type=Path, required=True, help="output dir for pointcloud.zarr")
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Stage frames, then run the backend with mv filtering off
    args.out.mkdir(parents=True, exist_ok=True)
    staged = args.out / "images_staged"
    paths = stage_color_frames(args.seq, staged, args.max_frames)
    logger.info("staged %d frames from %s", len(paths), args.seq)

    creator = CREATORS[args.backend](use_multiview_confidence=False)
    result = creator.run(staged)

    zarr_path = args.out / "pointcloud.zarr"
    result.save_zarr(zarr_path)
    logger.info("wrote %s  depth=%s", zarr_path, result.depth.shape)


if __name__ == "__main__":
    main()
