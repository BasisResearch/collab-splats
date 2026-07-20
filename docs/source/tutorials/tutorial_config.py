from pathlib import Path

import yaml

# ── Edit these two variables to point at your data ───────────────────────────
BASE_DIR = Path("/workspace/outputs")
DATASET = "2024_02_06/C0043"  # <session-date>/<video-stem>
MAX_FRAMES = 30

# ── Derived paths (do not edit) ───────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / DATASET
CACHE_DIR = OUTPUT_DIR  # alias — notebooks read canonical artifacts here
FRAMES = OUTPUT_DIR / "frames"  # pipeline-written keyframes (read-only for tutorials)
TUTORIAL_CACHE = BASE_DIR / "tutorial_cache" / DATASET  # notebook scratch output (never synced)


def _infer_video_path(output_dir: Path) -> Path | None:
    """Resolve the scene's source video: run_config.yaml keys, then *.mp4 glob."""
    config_file = output_dir / "run_config.yaml"
    if config_file.exists():
        try:
            cfg = yaml.safe_load(config_file.read_text())
            # video_ref is rclone-relative — resolve its basename against the scene dir
            raw = cfg.get("video_path") or cfg.get("input_path") or cfg.get("video_ref")
            if raw:
                candidate = output_dir / Path(raw).name
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    # Fallback: a video file sitting directly in the scene dir
    matches = sorted(output_dir.glob("*.MP4")) + sorted(output_dir.glob("*.mp4"))
    return matches[0] if matches else None


VIDEO_PATH = _infer_video_path(OUTPUT_DIR)
