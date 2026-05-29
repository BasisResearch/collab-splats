from pathlib import Path

import yaml

# ── Edit these two variables to point at your data ───────────────────────────
BASE_DIR   = Path("/workspace/outputs")
DATASET    = "birds_c0043"
MAX_FRAMES = 30

# ── Derived paths (do not edit) ───────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / DATASET
CACHE_DIR  = OUTPUT_DIR          # alias — notebooks use CACHE_DIR
IMAGES     = OUTPUT_DIR / "images"

# Infer VIDEO_PATH from run_config.yaml; None if not found or file missing
def _infer_video_path(output_dir: Path) -> Path | None:
    """Read video_path from run_config.yaml in the output directory."""
    config_file = output_dir / "run_config.yaml"
    if not config_file.exists():
        return None
    try:
        cfg = yaml.safe_load(config_file.read_text())
        raw = cfg.get("video_path") or cfg.get("input_path")
        if raw:
            p = Path(raw)
            return p if p.exists() else None
    except Exception:
        return None

VIDEO_PATH = _infer_video_path(OUTPUT_DIR)
