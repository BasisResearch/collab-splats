#!/usr/bin/env python3
"""Run the minimal reconstruction pipeline over one or more scene videos.

Usage:
    # One scene
    python docs/examples/run_scenes.py --output-root /workspace/outputs scene.MP4

    # Many scenes (shell glob), semantics + localization on via a shared override YAML
    python docs/examples/run_scenes.py \\
        --output-root /workspace/outputs \\
        --config configs/minimal.yaml \\
        /data/birds/*.MP4

    # Only some stages
    python docs/examples/run_scenes.py --output-root /workspace/outputs \\
        --stages preprocess,pointcloud,localize scene.MP4

Each video V maps to output_path = <output-root>/<V-stem>/. Config is base.yaml
(optionally deep-merged with --config). One scene's failure does not abort the batch;
the process exits non-zero if any scene failed.
"""

import argparse
import logging
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


def build_scene_config(video, output_root, config_dir, override_config=None):
    """Build a per-scene config from base.yaml with input/output paths set."""
    loader = ConfigLoader(config_dir)
    config = merge({}, loader.base_config)  # copy of base.yaml
    if override_config:
        config = merge({}, config, override_config)  # shared --config overrides
    config["input_path"] = str(video)
    config["output_path"] = str(Path(output_root) / Path(video).stem)
    return config


def run_scene(video, output_root, config_dir, override_config, stages, overwrite):
    """Run the pipeline for a single video. Returns the scene output path."""
    config = build_scene_config(video, output_root, config_dir, override_config)
    r = Reconstructor(config)

    # Persist run_config.yaml for reproducibility (same behavior as reconstruct.py)
    output_path = Path(r.config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    run_cfg = output_path / "run_config.yaml"
    if not run_cfg.exists() or overwrite:
        with open(run_cfg, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)

    r.run_pipeline(stages=stages, overwrite=overwrite)
    return output_path


def run_all(videos, output_root, config_dir, override_config, stages, overwrite):
    """Run every scene; continue past failures. Returns process exit code."""
    results = []
    for video in videos:
        video = Path(video)
        logger.info("=== Scene: %s ===", video.name)
        try:
            out = run_scene(video, output_root, config_dir, override_config, stages, overwrite)
            results.append((video.name, "OK", str(out)))
        except Exception as exc:  # isolate one scene's failure from the batch
            logger.exception("Scene failed: %s", video.name)
            results.append((video.name, "FAIL", str(exc)))

    # Summary
    logger.info("==== Summary ====")
    for name, status, info in results:
        logger.info("%s: %s (%s)", status, name, info)
    failed = [n for n, s, _ in results if s == "FAIL"]
    return 1 if failed else 0


def main():
    """Parse args and run the pipeline over the given scene videos."""
    parser = argparse.ArgumentParser(
        description="Run the minimal reconstruction pipeline over one or more scene videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("videos", nargs="+", metavar="VIDEO", help="One or more video files.")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        dest="output_root",
        help="Parent dir; each scene lands in <output-root>/<video-stem>/.",
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
        help="Stages to run: preprocess,pointcloud,semantics,mesh,localize. " "Default: config-enabled stages.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run stages even if outputs already exist.")
    args = parser.parse_args()

    override_config = None
    if args.config:
        with open(args.config) as f:
            override_config = yaml.safe_load(f)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    code = run_all(
        videos=args.videos,
        output_root=args.output_root,
        config_dir=args.config_dir,
        override_config=override_config,
        stages=stages,
        overwrite=args.overwrite,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
