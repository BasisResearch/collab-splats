#!/usr/bin/env python3
"""Extract keyframes from one or more videos and run the reconstruction pipeline.

Top-level entry point for turning raw video(s) into a reconstructed scene. Point it
at a single video, several videos, or directories of videos.

Pipeline steps per video (executed by Reconstructor.run_pipeline):
  1. keyframe extraction   — sample sharp, well-exposed frames from the video
  2. vggt_omega pointcloud — feed-forward 3D reconstruction (poses + depth + points)
  3. talk2dino semantics   — 2D features lifted to 3D, autoencoder-compressed
  4. localization database  — per-frame local-feature cache for camera localization

preproc + pointcloud always run. semantics, mesh, and localize run only when
enabled in the config (semantics.enabled / mesh.enabled / localization.enabled).
Override the set explicitly with --stages.

Usage:
    # Single video
    python docs/examples/run_pipeline.py --output-root /workspace/outputs scene.MP4

    # Several videos + a directory (dirs are globbed for *.mp4/*.mov/*.avi)
    python docs/examples/run_pipeline.py --output-root /workspace/outputs \\
        /data/birds/C0043.MP4 /data/rats/

    # Turn on localization + semantics via a shared override YAML
    python docs/examples/run_pipeline.py --output-root /workspace/outputs \\
        --config my_overrides.yaml /data/birds/*.MP4   # your own YAML, merged over base.yaml

    # Specific steps only
    python docs/examples/run_pipeline.py --output-root /workspace/outputs \\
        --stages preproc,pointcloud,localize scene.MP4

Each video V is written to  <output-root>/<session-date>/<V-stem>/  when a date-like
dir (YYYY-MM-DD) appears in V's path, else  <output-root>/<V-stem>/ . A per-scene
run_config.yaml records the exact settings used. One video's failure does not abort
the batch; the process exits non-zero if any video failed.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from collab_splats.wrapper.batch import DEFAULT_CONFIG_DIR, collect_videos, run_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Parse args and run the pipeline over the given videos / directories."""
    parser = argparse.ArgumentParser(
        description="Extract keyframes and reconstruct one or more videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="+", metavar="VIDEO|DIR", help="Video files and/or directories of videos.")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        dest="output_root",
        help="Parent dir; each video lands in <output-root>/[<date>/]<stem>/.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional shared override YAML merged over base.yaml."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        dest="config_dir",
        help=f"Directory holding base.yaml. Default: {DEFAULT_CONFIG_DIR}",
    )
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help="Steps to run: preproc,pointcloud,semantics,mesh,localize. " "Default: config-enabled steps.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run steps even if outputs already exist.")
    parser.add_argument(
        "--keep-viewer",
        action="store_true",
        dest="keep_viewer",
        help="keep the viser viewer alive after reconstruction for browser inspection",
    )
    args = parser.parse_args()

    override_config = None
    if args.config:
        with open(args.config) as f:
            override_config = yaml.safe_load(f)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    videos = collect_videos(args.paths)
    if not videos:
        logger.error("No videos found in: %s", args.paths)
        sys.exit(2)

    code = run_all(
        videos=videos,
        output_root=args.output_root,
        config_dir=args.config_dir,
        override_config=override_config,
        stages=stages,
        overwrite=args.overwrite,
        keep_viewer=args.keep_viewer,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
