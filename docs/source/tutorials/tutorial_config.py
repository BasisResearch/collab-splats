from pathlib import Path

# ── Edit these two variables to point at your data ───────────────────────────
BASE_DIR   = Path("/workspace/outputs")
DATASET    = "birds_c0043"
MAX_FRAMES = 30

# ── Derived paths (do not edit) ───────────────────────────────────────────────
VIDEO_PATH = BASE_DIR / DATASET / "video.mp4"
OUTPUT_DIR = BASE_DIR / DATASET
CACHE_DIR  = OUTPUT_DIR          # alias — notebooks use CACHE_DIR
IMAGES     = OUTPUT_DIR / "images"
