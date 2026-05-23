"""Run the full collab-splats pipeline for C0043 from scratch.

Stages:
  1. Archive existing output dir (timestamped)
  2. Preprocess with hloc SfM
  3. Train rade-features (maskclip + dinov2)
  4. Mesh with Open3DTSDFFusion
"""
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

FIELDWORK_ROOT = Path("/workspace/fieldwork-data/birds/2024-02-06")
OUTPUT_DIR = FIELDWORK_ROOT / "environment" / "C0043"
CONFIG_DIR = Path(__file__).parent.parent / "docs/splats/configs"
DATASET = "birds_date-02062024_video-C0043"


def archive_output_dir(output_dir: Path) -> Path:
    """Move output_dir to a timestamped archive sibling. Returns archive path."""
    if not output_dir.exists():
        raise FileNotFoundError(f"Nothing to archive at {output_dir}")
    if output_dir.is_symlink():
        raise ValueError(f"{output_dir} is a symlink — refusing to archive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = output_dir.parent / f"{output_dir.name}_archived_{timestamp}"
    shutil.move(str(output_dir), str(archive_path))
    return archive_path


def run_stage(name: str, fn, **kwargs) -> None:
    """Run a pipeline stage with banner, wall-clock timing, and error exit."""
    print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
    t0 = time.time()
    try:
        result = fn(**kwargs)
        print(f"\n[OK] {name} completed in {time.time() - t0:.1f}s")
        return result
    except Exception:
        traceback.print_exc()
        print(f"\n[FAIL] {name} failed after {time.time() - t0:.1f}s")
        sys.exit(1)


def main() -> None:
    from collab_splats.wrapper.splatter import Splatter

    # Archive existing output before any pipeline work
    if OUTPUT_DIR.exists():
        print(f"Archiving {OUTPUT_DIR} ...")
        archive_path = archive_output_dir(OUTPUT_DIR)
        print(f"Archived → {archive_path}")
    else:
        print(f"No existing output at {OUTPUT_DIR} — skipping archive.")

    # Load full config from YAML hierarchy
    # Resolves: method=rade-features, frame_proportion=0.25,
    #           preprocess.sfm_tool=hloc, training kwargs, meshing kwargs
    splatter = Splatter.from_config_file(dataset=DATASET, config_dir=CONFIG_DIR)

    # Stage 1: Preprocess — ns-process-data video --sfm-tool hloc
    run_stage(
        "Preprocess (hloc SfM)",
        splatter.preprocess,
        kwargs=splatter._preprocess_config,
        overwrite=True,
    )

    # Stage 2: Train — ns-train rade-features (maskclip + dinov2 features)
    run_stage(
        "Train rade-features (maskclip + dinov2)",
        splatter.extract_features,
        kwargs=splatter._training_config,
        overwrite=True,
    )

    # Stage 3: Mesh — Open3DTSDFFusion depth fusion
    mesher_config = (splatter._meshing_config or {}).copy()
    mesher_type = mesher_config.pop("mesher_type", "Open3DTSDFFusion")
    run_stage(
        "Mesh (Open3DTSDFFusion)",
        splatter.mesh,
        mesher_type=mesher_type,
        mesher_kwargs=mesher_config,
        overwrite=True,
    )

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Splat : {splatter.config.get('model_path')}")
    print(f"Mesh  : {splatter.config.get('mesh_info', {}).get('mesh')}")


if __name__ == "__main__":
    main()
