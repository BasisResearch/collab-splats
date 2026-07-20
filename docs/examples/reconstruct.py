#!/usr/bin/env python3
"""Re-run the Reconstructor pipeline from a saved config file.

Use this to reproduce an exact run from a `run_config.yaml` (written by
`run_pipeline.py` into every scene's output dir). The config already carries
`input_path` and `output_path`, so no video path is needed here.

To run NEW videos, use `run_pipeline.py` instead — this script is only for
reproducing / tweaking an already-produced config.

Usage:
    # Reproduce an exact saved run
    python docs/examples/reconstruct.py \\
        --config /workspace/outputs/2024_02_06/C0043/run_config.yaml

    # Reproduce, but override config values (dotted key=value)
    python docs/examples/reconstruct.py \\
        --config /workspace/outputs/2024_02_06/C0043/run_config.yaml \\
        pointcloud.backend=vggtx semantics.extractor=dinov2

    # Send the tweaked run to a separate output dir
    python docs/examples/reconstruct.py \\
        --config /workspace/outputs/2024_02_06/C0043/run_config.yaml \\
        output_path=/workspace/outputs/2024_02_06/C0043_ba \\
        pointcloud.bundle_adjustment=true

    # Run specific stages only
    python docs/examples/reconstruct.py \\
        --config .../run_config.yaml --stages preprocess,pointcloud,localize

    # Force re-run even if outputs already exist
    python docs/examples/reconstruct.py --config .../run_config.yaml --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from mergedeep import merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse args and re-run the Reconstructor pipeline from a saved config."""
    parser = argparse.ArgumentParser(
        description="Re-run the Reconstructor pipeline from a saved run_config.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a YAML config file (e.g. a saved run_config.yaml).",
    )
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help="Comma-separated stages to run: preprocess,pointcloud,semantics,mesh,localize. "
        "Default: all enabled stages in config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run stages even if output already exists on disk.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dotted-path config overrides, e.g. pointcloud.backend=vggtx semantics.enabled=true",
    )

    args = parser.parse_args()

    from collab_splats.wrapper.config import parse_cli_overrides
    from collab_splats.wrapper.reconstructor import Reconstructor

    # Load the saved config and apply any KEY=VALUE overrides
    logger.info("Loading config from file: %s", args.config)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.overrides:
        config = merge({}, config, parse_cli_overrides(args.overrides))
    r = Reconstructor(config)

    # Write run_config.yaml to output dir before running (for auditability)
    output_path = Path(r.config["output_path"])
    run_cfg_path = output_path / "run_config.yaml"
    if not run_cfg_path.exists() or args.overwrite:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(run_cfg_path, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)
        logger.info("Config saved: %s", run_cfg_path)
    else:
        logger.info("Config exists (use --overwrite to replace): %s", run_cfg_path)

    # Parse stages
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    logger.info("Running pipeline: stages=%s overwrite=%s", stages or "auto", args.overwrite)
    r.run_pipeline(stages=stages, overwrite=args.overwrite)
    logger.info("Pipeline complete. Output: %s", r.backend_dir)


if __name__ == "__main__":
    main()
