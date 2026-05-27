#!/usr/bin/env python
"""Run our VGGT-X LC pipeline on a sequence, dump per-boundary solver internals + TUM.

Monkey-patches wrappers.run_pose_graph_optimization to inject debug_out.
The debug_out list is populated by run_pose_graph_optimization with per-boundary dicts:
  {submap_id, T, scale, H_w, H_overlap, H_opt, corrected_proj}

Usage:
    python evals/runners/our_solver_dump.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --out_json evals/results/parity_harness/our_internals.json \\
        --out_tum evals/results/parity_harness/our_lc.tum \\
        --max_frames 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization as _orig_rpgo
from collab_splats.pointcloud.wrappers import LoopClosure
import collab_splats.pointcloud.wrappers as _wrappers_mod
from collab_splats.pointcloud.utils import cross_frame_attention_ratio as _orig_cfar
import collab_splats.pointcloud.utils as _utils_mod
import collab_splats.pointcloud.feedforward.base as _base_mod

logger = logging.getLogger(__name__)


def _write_tum(path: Path, poses_w2c: np.ndarray) -> None:
    """Write TUM file: timestamp tx ty tz qx qy qz qw (c2w, 8-col format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, w2c in enumerate(poses_w2c):
        c2w = np.linalg.inv(w2c.astype(np.float64))
        t = c2w[:3, 3]
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [qx, qy, qz, qw]
        lines.append(
            f"{i:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _to_serializable(entry: dict) -> dict:
    """Convert numpy arrays in a debug_out entry to JSON-serializable lists."""
    out = {}
    for k, v in entry.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def run_dump(
    seq_dir: Path,
    out_json: Path,
    out_tum: Path,
    max_frames: int = 200,
    submap_size: int = 16,
    conf_threshold: float = 25.0,
) -> None:
    """Run our LC pipeline with debug_out capture, write JSON + TUM."""
    _debug_out: list[dict] = []
    # Capture cross_frame_attention_ratio calls from _verify_loop_candidate.
    # Same metric as VGGT-SPARK image_match_ratio (port of get_similarity()).
    _our_similarity_log: list[float] = []

    def _patched_cfar(k, q, token_offset: int = 5) -> float:
        result = _orig_cfar(k, q, token_offset)
        _our_similarity_log.append(float(result))
        logger.info("our cross_frame_attention_ratio: %.4f (threshold 0.85)", float(result))
        return result

    def _patched_rpgo(*args, **kwargs):
        # Inject debug_out list so the optimizer captures per-boundary internals
        kwargs["debug_out"] = _debug_out
        return _orig_rpgo(*args, **kwargs)

    # Patch in wrappers module namespace (where the bare call lives)
    _wrappers_mod.run_pose_graph_optimization = _patched_rpgo
    # Patch cross_frame_attention_ratio in both utils and base (base imports it directly)
    _utils_mod.cross_frame_attention_ratio = _patched_cfar
    _base_mod.cross_frame_attention_ratio = _patched_cfar

    try:
        # Collect 7-Scenes color images sorted by filename
        image_paths = sorted(
            p for p in seq_dir.iterdir()
            if p.suffix.lower() == ".png" and "color" in p.name
        )[:max_frames]

        if not image_paths:
            raise FileNotFoundError(f"No *.color.png found in {seq_dir}")

        # Symlink images into a temp dir (creator.reconstruct expects a flat image directory)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for i, src in enumerate(image_paths):
                (tmp / f"{i:06d}.png").symlink_to(src.resolve())

            cfg = LoopClosureConfig(submap_size=submap_size, conf_threshold=conf_threshold)
            base = get_creator("vggtx")()
            creator = LoopClosure(base, config=cfg)

            with tempfile.TemporaryDirectory() as out_dir:
                creator.reconstruct(tmp, Path(out_dir))

        # creator.outputs proxies to base.outputs; extrinsics is (N, 4, 4) w2c
        final_poses = creator.outputs.extrinsics
        # Add sequential index to each captured boundary entry
        for i, entry in enumerate(_debug_out):
            entry["boundary_idx"] = i
    finally:
        # Always restore originals to avoid polluting other code in the same process
        _wrappers_mod.run_pose_graph_optimization = _orig_rpgo
        _utils_mod.cross_frame_attention_ratio = _orig_cfar
        _base_mod.cross_frame_attention_ratio = _orig_cfar

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seq_dir": str(seq_dir),
            "max_frames": max_frames,
            "submap_size": submap_size,
            "conf_threshold": conf_threshold,
        },
        "boundaries": [_to_serializable(e) for e in _debug_out],
        "final_poses": final_poses.tolist(),
        # cross_frame_attention_ratio values per _verify_loop_candidate call.
        # Compare against vggt_spark_similarity.json image_match_ratio scores.
        # Same algorithm (port of VGGT-SPARK get_similarity()), different model weights.
        "similarity_scores": {
            "model": "VGGT-X",
            "metric": "cross_frame_attention_ratio",
            "threshold": 0.85,
            "note": "port of VGGT-SPARK get_similarity() via VGGT-X QKV hooks",
            "scores": _our_similarity_log,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("Written %d boundaries → %s", len(_debug_out), out_json)

    _write_tum(out_tum, final_poses)
    logger.info("Written TUM → %s", out_tum)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Run our LC pipeline and dump per-boundary solver internals to JSON + TUM."
    )
    parser.add_argument("--seq_dir", required=True, help="Path to sequence directory (7-Scenes style)")
    parser.add_argument(
        "--out_json",
        default="evals/results/parity_harness/our_internals.json",
        help="Output path for per-boundary internals JSON",
    )
    parser.add_argument(
        "--out_tum",
        default="evals/results/parity_harness/our_lc.tum",
        help="Output path for TUM trajectory file",
    )
    parser.add_argument("--max_frames", type=int, default=200, help="Cap on number of input frames")
    parser.add_argument("--submap_size", type=int, default=16, help="Submap window size")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Confidence threshold for scale estimation")
    args = parser.parse_args()

    run_dump(
        seq_dir=Path(args.seq_dir),
        out_json=Path(args.out_json),
        out_tum=Path(args.out_tum),
        max_frames=args.max_frames,
        submap_size=args.submap_size,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
