import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "docs/source/tutorials/tutorial_config.py"


def _load():
    """Exec tutorial_config.py in a fresh namespace and return its globals."""
    return runpy.run_path(str(CONFIG))


def test_paths_are_repo_relative_and_correct():
    ns = _load()
    assert ns["VIDEO_PATH"] == REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
    assert ns["QUERY_IMAGE"] == REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"
    assert ns["OUTPUT_DIR"] == REPO_ROOT / "data/outputs"
    assert ns["FRAMES_ZARR"] == REPO_ROOT / "data/outputs/frames.zarr"
    assert ns["RECON"] == REPO_ROOT / "data/outputs/feedforward.zarr"
    assert ns["TUTORIAL_CACHE"] == REPO_ROOT / "data/outputs/tutorial_cache"
    assert ns["MAX_FRAMES"] == 30


def test_no_retired_names():
    ns = _load()
    for gone in ("DATASET", "BASE_DIR", "FRAMES", "_infer_video_path"):
        assert gone not in ns, f"{gone} should be removed"
