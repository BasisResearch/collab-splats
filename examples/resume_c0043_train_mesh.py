"""Resume C0043 pipeline from training stage (preprocess already done)."""
import sys
import time
import traceback
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "docs/splats/configs"
DATASET = "birds_date-02062024_video-C0043"


def run_stage(name: str, fn, **kwargs):
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

    splatter = Splatter.from_config_file(dataset=DATASET, config_dir=CONFIG_DIR)

    # Populate preproc_data_path without re-running preprocess
    splatter.preprocess(kwargs=splatter._preprocess_config, overwrite=False)

    run_stage(
        "Train rade-features (maskclip + dinov2)",
        splatter.extract_features,
        kwargs=splatter._training_config,
        overwrite=True,
    )

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
