"""
Minimal pipeline runner using YAML configs.

Usage:
    # Run with default settings
    python run_pipeline.py --dataset birds_date-02062024_video-C0043

    # Override any config value
    python run_pipeline.py --dataset birds_date-02062024_video-C0043 --set method=rade-gs frame_proportion=0.1

    # Override nested values
    python run_pipeline.py --dataset birds_date-02062024_video-C0043 --set preprocess.sfm_tool=colmap

    # Overwrite existing outputs
    python run_pipeline.py --dataset birds_date-02062024_video-C0043 --overwrite

    # List available datasets
    python run_pipeline.py --list-datasets

Examples:
    # Use birds dataset with high quality settings
    python run_pipeline.py --dataset birds_date-02062024_video-C0043 --set training.pipeline.model.random-scale=1.0

    # Use ants dataset with colmap preprocessing
    python run_pipeline.py --dataset ants_date-11162025_video-GH010210 --set preprocess.sfm_tool=colmap
"""
import argparse
from pathlib import Path
from collab_splats.wrapper import Splatter, ConfigLoader, parse_cli_overrides


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    config_dir = script_dir / "configs"

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run splatting pipeline with YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", type=str, help="Dataset config name")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Override config (e.g., --set method=rade-gs preprocess.sfm_tool=colmap)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    parser.add_argument(
        "--list-datasets", action="store_true", help="List available datasets"
    )

    args = parser.parse_args()

    # Initialize loader
    loader = ConfigLoader(config_dir)

    # Handle list command
    if args.list_datasets:
        datasets = loader.list_datasets()
        print("Available datasets:")
        for d in datasets:
            print(f"  - {d}")
        return

    # Require dataset
    if not args.dataset:
        parser.error("--dataset is required (or use --list-datasets)")

    # Parse overrides
    overrides = parse_cli_overrides(args.set) if args.set else None

    # Create splatter from config
    splatter = Splatter.from_config_file(
        dataset=args.dataset,
        config_dir=config_dir,
        overrides=overrides,
    )

    # Run pipeline
    splatter.preprocess()
    splatter.extract_features() #overwrite=args.overwrite)
    splatter.mesh(overwrite=args.overwrite)
    # splatter.run_pipeline(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
