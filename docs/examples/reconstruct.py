#!/usr/bin/env python3
"""CLI entry point for the Reconstructor pipeline.

Usage:
    # Run with a named dataset (config from configs/datasets/)
    python docs/examples/reconstruct.py --dataset birds_c0043

    # Run specific stages only
    python docs/examples/reconstruct.py --dataset birds_c0043 --stages preprocess,pointcloud

    # Build the localization database (local-feature cache) as part of the run
    python docs/examples/reconstruct.py --dataset birds_c0043 \\
        localization.enabled=true --stages preprocess,pointcloud,localize

    # Override any config value (dotted key=value)
    python docs/examples/reconstruct.py --dataset birds_c0043 \\
        pointcloud.backend=vggtx \\
        semantics.extractor=dinov2

    # Experiment variant: send output to a separate dir
    python docs/examples/reconstruct.py --dataset birds_c0043 \\
        output_path=/workspace/outputs/birds_c0043_ba \\
        pointcloud.bundle_adjustment=true

    # Re-run from a saved run_config.yaml (exact reproducibility)
    python docs/examples/reconstruct.py \\
        --config /workspace/outputs/birds_c0043/run_config.yaml

    # Force re-run even if outputs already exist
    python docs/examples/reconstruct.py --dataset birds_c0043 --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs"


def main() -> None:
    """Parse args and run Reconstructor pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Reconstructor pipeline from config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config source: exactly one of --dataset or --config
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        metavar="NAME",
        help="Dataset name — loads configs/datasets/<NAME>.yaml",
    )
    source.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="Direct path to a YAML config file (e.g. a saved run_config.yaml)",
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        dest="config_dir",
        help=f"Config directory for --dataset lookups. Default: {DEFAULT_CONFIG_DIR}",
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

    # Parse KEY=VALUE overrides
    overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    # Build Reconstructor from either a dataset template or a direct YAML
    if args.config:
        # Direct YAML path (e.g. a saved run_config.yaml for exact reproducibility)
        logger.info("Loading config from file: %s", args.config)
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if overrides:
            from mergedeep import merge

            config = merge({}, config, overrides)
        r = Reconstructor(config)
    else:
        # Named dataset: merge base.yaml + datasets/<name>.yaml + overrides
        logger.info("Loading config: dataset=%s config_dir=%s", args.dataset, args.config_dir)
        r = Reconstructor.from_config_file(
            dataset=args.dataset,
            config_dir=args.config_dir,
            overrides=overrides,
        )

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
