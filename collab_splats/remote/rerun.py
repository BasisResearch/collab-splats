"""Route a scene's inputs for a re-run: curated video, or an already-processed scene pulled back.

A stage set containing only LEAF_STAGES (nothing depends on them) can be re-run against processed
data. Anything upstream rebuilds from the curated video, so a run can never leave a stale
downstream artifact beside a freshly-regenerated one.
"""

import logging
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.wrapper.reconstructor import LEAF_STAGES

logger = logging.getLogger(__name__)


def _is_rerun(stages) -> bool:
    """True when every requested stage is a leaf, so processed outputs are enough to run it."""
    return bool(stages) and set(stages) <= LEAF_STAGES


def discover_scenes(source, stages) -> list[str]:
    """Scene ids for an --all run: the processed bucket for a leaf re-run, curated otherwise."""
    # Listing curated for a leaf re-run would hand the batch every scene that has no processed
    # outputs and fail each one on a FileNotFoundError it could have avoided asking for.
    if _is_rerun(stages):
        return source.list_processed_scenes()
    return source.list_scenes()


def prepare_scene(source, scene: str, scene_dir: Path, stages, override_config, on_line=None):
    """Fetch a scene's inputs; return (video, override_config) for batch.run_scene.

    Leaf-only stage sets re-run from environments-processed and return video=None; anything else
    starts from the curated video and passes override_config through untouched.
    """
    # Route on the dependency graph, not a hardcoded list.
    if not _is_rerun(stages):
        return source.fetch_video(scene, scene_dir, on_line=on_line), override_config

    if not source.has_processed(scene):
        raise FileNotFoundError(f"{scene} has no processed outputs — run the full pipeline first")

    # No excludes: PULL_EXCLUDES is the viewer's default and drops depth/images, which is exactly
    # what meshing reads. Correct-by-construction beats a per-stage member table that silently
    # starves a stage when what it reads changes.
    source.pull_processed(scene, scene_dir, on_line=on_line)

    # The pulled run_config is the only record of which backend produced this scene, and the
    # backend names the subdir every artifact path is built from — no config, no run.
    run_cfg = Path(scene_dir) / "run_config.yaml"
    if not run_cfg.exists():
        raise FileNotFoundError(f"{scene}: pulled scene has no run_config.yaml; backend is unknowable")
    with open(run_cfg) as f:
        pulled = yaml.safe_load(f)
    backend = pulled["pointcloud"]["backend"]

    # Retargeting must be typed, and a typed one that disagrees with the data is a mistake.
    asked = (override_config or {}).get("pointcloud", {}).get("backend")
    if asked and asked != backend:
        raise ValueError(f"{scene} was built with backend '{backend}', --config asks for '{asked}'")

    # The pulled config carries provenance for every stage NOT being re-run; dropping the re-run
    # stages' sections lets base.yaml + --config supply fresh params for exactly those.
    cfg = merge({}, pulled)
    for stage in stages:
        # Stage name == config section, except localize → localization.
        cfg.pop("localization" if stage == "localize" else stage, None)
    logger.info("%s: re-run %s from processed (backend=%s)", scene, ",".join(stages), backend)
    return None, merge(cfg, override_config or {})
