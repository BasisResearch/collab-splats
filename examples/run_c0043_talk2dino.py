"""Run C0043 rade-features training with talk2dino as feature backend.

Reuses existing SfM preprocessed data (skips ns-process-data).
Outputs to C0043_talk2dino/ alongside the existing C0043/ run.
"""
import sys
import time
import traceback
from pathlib import Path

FIELDWORK_ROOT = Path("/workspace/fieldwork-data/birds/2024-02-06")
CONFIG_DIR = Path(__file__).parent.parent / "docs/splats/configs"
DATASET = "birds_date-02062024_video-C0043"

# Separate output dir — don't overwrite existing maskclip/samclip run
OUTPUT_DIR = FIELDWORK_ROOT / "environment" / "C0043_talk2dino"


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

    # Point at existing SfM output — no re-run needed
    splatter.config["preproc_data_path"] = (
        FIELDWORK_ROOT / "environment" / "C0043" / "preproc"
    )
    # Separate output so we don't overwrite the existing samclip run
    splatter.config["output_path"] = OUTPUT_DIR

    # Override training config: use talk2dino as main feature extractor.
    # Keep regularization_features=dinov2 (default) for structural regularization.
    training_kwargs = {
        **(splatter._training_config or {}),
        "pipeline.datamanager.main-features": "talk2dino",
    }

    run_stage(
        "Train rade-features (talk2dino)",
        splatter.extract_features,
        kwargs=training_kwargs,
        overwrite=True,
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"Splat : {splatter.config.get('model_path')}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
