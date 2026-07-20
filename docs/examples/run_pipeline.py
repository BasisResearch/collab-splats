#!/usr/bin/env python3
"""Extract keyframes from one or more videos and run the reconstruction pipeline.

Top-level entry point for turning raw video(s) into a reconstructed scene. Point it
at a single video, several videos, or directories of videos.

Pipeline steps per video (executed by Reconstructor.run_pipeline):
  1. keyframe extraction   — sample sharp, well-exposed frames from the video
  2. vggt_omega pointcloud — feed-forward 3D reconstruction (poses + depth + points)
  3. talk2dino semantics   — 2D features lifted to 3D, autoencoder-compressed
  4. localization database  — per-frame local-feature cache for camera localization

preprocess + pointcloud always run. semantics, mesh, and localize run only when
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
        --config configs/minimal.yaml /data/birds/*.MP4

    # Specific steps only
    python docs/examples/run_pipeline.py --output-root /workspace/outputs \\
        --stages preprocess,pointcloud,localize scene.MP4

Each video V is written to  <output-root>/<session-date>/<V-stem>/  when a date-like
dir (YYYY-MM-DD) appears in V's path, else  <output-root>/<V-stem>/ . A per-scene
run_config.yaml records the exact settings used. One video's failure does not abort
the batch; the process exits non-zero if any video failed.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.wrapper.config import ConfigLoader
from collab_splats.wrapper.reconstructor import Reconstructor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs"
_VIDEO_EXTS = {".mp4", ".mov", ".avi"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def collect_videos(paths):
    """Expand each path (a video file or a directory of videos) into a flat video list."""
    videos = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # Non-recursive: only videos directly inside the directory
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in _VIDEO_EXTS)
            if not found:
                logger.warning("No videos (%s) in directory: %s", sorted(_VIDEO_EXTS), p)
            videos.extend(found)
        else:
            videos.append(p)
    return videos


def scene_output_dir(video, output_root):
    """Derive <output-root>/<session-date>/<stem>, matching the live outputs/ layout.

    Session date = first YYYY-MM-DD dir in the video's parents (dashes -> underscores).
    Falls back to <output-root>/<stem> when no date dir is present.
    """
    output_root = Path(output_root)
    stem = Path(video).stem
    for parent in Path(video).parents:
        if _DATE_RE.fullmatch(parent.name):
            return output_root / parent.name.replace("-", "_") / stem
    return output_root / stem


def build_scene_config(video, output_root, config_dir, override_config=None):
    """Build a per-video config from base.yaml with input/output paths set."""
    loader = ConfigLoader(config_dir)
    config = merge({}, loader.base_config)  # copy of base.yaml
    if override_config:
        config = merge({}, config, override_config)  # shared --config overrides
    config["input_path"] = str(video)
    config["output_path"] = str(scene_output_dir(video, output_root))
    return config


def run_scene(video, output_root, config_dir, override_config, stages, overwrite):
    """Run the full pipeline for a single video. Returns the scene output path."""
    config = build_scene_config(video, output_root, config_dir, override_config)
    r = Reconstructor(config)

    # Persist run_config.yaml for reproducibility before running any stage
    output_path = Path(r.config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    run_cfg = output_path / "run_config.yaml"
    if not run_cfg.exists() or overwrite:
        with open(run_cfg, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)

    # Steps 1-4 (see module docstring) run here, governed by config + --stages
    r.run_pipeline(stages=stages, overwrite=overwrite)
    return output_path


def run_all(videos, output_root, config_dir, override_config, stages, overwrite):
    """Run every video; continue past failures. Returns the process exit code."""
    results = []
    for video in videos:
        video = Path(video)
        logger.info("=== Video: %s ===", video.name)
        try:
            out = run_scene(video, output_root, config_dir, override_config, stages, overwrite)
            results.append((video.name, "OK", str(out)))
        except Exception as exc:  # isolate one video's failure from the batch
            logger.exception("Video failed: %s", video.name)
            results.append((video.name, "FAIL", str(exc)))

    # Summary
    logger.info("==== Summary ====")
    for name, status, info in results:
        logger.info("%s: %s (%s)", status, name, info)
    failed = [n for n, s, _ in results if s == "FAIL"]
    return 1 if failed else 0


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
        help="Steps to run: preprocess,pointcloud,semantics,mesh,localize. " "Default: config-enabled steps.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run steps even if outputs already exist.")
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
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
