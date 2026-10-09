"""
Committed inputs and the shared scene every tutorial page builds on.

- load with `%run ../tutorial.py`
- one scene dir for all pages: a page runs only the stages it needs that are not on disk yet
- delete data/tutorial_scene/ after pulling code changes; done() checks existence only
- pyvista backend set at load: static when headless (nbconvert), interactive trame otherwise
"""

import copy
import os
import shutil
from pathlib import Path

import pyvista as pv

from collab_splats.reconstructor import Reconstructor

########################################
# Committed inputs (read-only)
########################################

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_PATH = REPO_ROOT / "data/tutorial/tutorial_example-video.mp4"
QUERY_IMAGE = REPO_ROOT / "data/tutorial/tutorial_example-frame.jpg"
REF_FRAME = REPO_ROOT / "data/tutorial/tutorial_ref-frame.jpg"

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"missing {VIDEO_PATH}; see data/tutorial/README.md")

# Static renders under nbconvert, interactive in a live kernel
pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")

########################################
# Shared scene
########################################

SCENE_DIR = REPO_ROOT / "data/tutorial_scene"

# Small profile merged over configs/base.yaml (vggt_omega, feedforward textured mesh)
SCENE_CONFIG = {
    "preproc": {"max_frames": 96},
    "splats": {
        "enabled": True,
        "representation": "scaffold",
        "primitive": "2dgs",
        "max_steps": 3000,
        # Halve the downscale every 300 steps: full res from step 600, not never
        "resolution_schedule": 300,
        # Scaffold + 2dgs losses per base.yaml, start steps scaled to 3000
        "losses": {
            "opacity_reg": {"weight": 0},
            "scale_reg": {"weight": 0},
            "distortion": {"weight": 0.01, "start": 300},
            "normal_consistency": {"weight": 0.05, "start": 700},
        },
    },
    "semantics": {"extractor": "talk2dino", "max_epochs": 20},
}


def tutorial_scene(*stages: str, extractor: str | None = None) -> Reconstructor:
    """
    The shared tutorial scene, with the named stages built if they are not on disk yet.

    - stages already done are dropped before run, so a named leaf never hits the overwrite refusal
    - dependencies already on disk are reused by Reconstructor.run

    Args:
        stages: stage names this page needs, e.g. "preproc", "pointcloud", "mesh".
        extractor: semantics extractor; each one writes its own store, so pages never collide.

    Returns:
        A Reconstructor over SCENE_DIR.
    """
    config = copy.deepcopy(SCENE_CONFIG)
    config["input_path"] = str(VIDEO_PATH)
    config["output_path"] = str(SCENE_DIR)

    if extractor is not None:
        config["semantics"]["extractor"] = extractor

    scene = Reconstructor(config)
    missing = [s for s in stages if not scene.done(s)]

    if missing:
        scene.run(stages=missing)

    return scene


def work_dir(page: str) -> Path:
    """
    An empty scratch dir for one page's longhand experiments, under the shared scene.

    Args:
        page: short page label, used as the directory name.

    Returns:
        SCENE_DIR/work/<page>, emptied and recreated.
    """
    path = SCENE_DIR / "work" / page
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path
