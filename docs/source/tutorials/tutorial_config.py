from pathlib import Path

from collab_splats.utils.paths import get_cache_dir

DATASET = "birds_c0043"
MAX_FRAMES = 30  # cap for light tutorial runs

CACHE_DIR = get_cache_dir(DATASET)
IMAGES = CACHE_DIR / "images"
