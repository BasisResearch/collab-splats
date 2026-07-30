"""Per-scene batch helpers shared by the local and remote pipeline drivers.

Discovery (`collect_videos`), output-dir derivation (`scene_output_dir`), per-scene
config assembly (`build_scene_config`), single-scene execution (`run_scene`) and the
failure-isolating batch loop (`run_all`). The local driver
(`docs/examples/run_pipeline.py`) derives each scene's output dir from a date-like
parent dir; remote drivers pass an explicit `name` — a curated GCS directory named
`YYYY_MM_DD-PARENTFOLDER-VIDEONAME` already *is* the scene id.
"""

import logging
import re
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.wrapper.reconstructor import DEFAULT_CONFIG_DIR, Reconstructor

logger = logging.getLogger(__name__)

########
# Constants
########

VIDEO_EXTS = {".mp4", ".mov", ".avi"}
# Local session dirs use hyphens (2026-07-20/); underscore scene ids are handled via name=
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


########
# Input discovery
########


def collect_videos(paths):
    """Expand each path (a video file or a directory of videos) into a flat video list."""
    videos = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # Non-recursive: only videos directly inside the directory
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
            if not found:
                logger.warning("No videos (%s) in directory: %s", sorted(VIDEO_EXTS), p)
            videos.extend(found)
        else:
            videos.append(p)
    return videos


########
# Output-dir derivation
########


def scene_output_dir(video, output_root, name: str | None = None) -> Path:
    """Derive a scene's output dir. An explicit name wins over date-parent derivation.

    name wins when given — remote scenes are named by their curated dir, which already
    encodes the date, so they get a flat output dir rather than a derived two-level one.
    Otherwise: <output-root>/<session-date>/<stem>, where session date is the first
    YYYY-MM-DD dir in the video's parents (dashes -> underscores), falling back to
    <output-root>/<stem> when no date dir is present.
    """
    output_root = Path(output_root)
    if name is not None:
        return output_root / name
    stem = Path(video).stem
    for parent in Path(video).parents:
        if _DATE_RE.fullmatch(parent.name):
            return output_root / parent.name.replace("-", "_") / stem
    return output_root / stem


########
# Per-scene config
########


def build_scene_config(video, output_root, override_config=None, name: str | None = None) -> dict:
    """Build a per-video override dict. Reconstructor merges base.yaml defaults itself."""
    # Only carry the shared --config overrides plus per-video paths; defaults come from base.yaml
    config = merge({}, override_config) if override_config else {}
    config["input_path"] = str(video)
    config["output_path"] = str(scene_output_dir(video, output_root, name=name))
    return config


########
# Single-scene runner
########


def run_scene(
    video,
    output_root,
    config_dir,
    override_config,
    stages,
    overwrite,
    name: str | None = None,
):
    """Run the full pipeline for a single video. Returns (output_path, Reconstructor)."""
    config = build_scene_config(video, output_root, override_config, name=name)
    r = Reconstructor(config, config_dir=config_dir or DEFAULT_CONFIG_DIR)

    # Persist run_config.yaml for reproducibility before running any stage
    output_path = Path(r.config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    run_cfg = output_path / "run_config.yaml"
    if not run_cfg.exists() or overwrite:
        with open(run_cfg, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)

    # preproc/pointcloud/semantics/mesh/localize run here, governed by config + stages
    r.run_pipeline(stages=stages, overwrite=overwrite)
    return output_path, r


########
# Batch driver
########


def run_all(videos, output_root, config_dir, override_config, stages, overwrite, keep_viewer=False):
    """Run every video; continue past failures. Returns the process exit code.

    When keep_viewer, blocks after the batch on the last successfully-reconstructed
    scene's viser Viewer (if pointcloud.viz.enabled produced one), so the final scene
    stays browsable. No viewer (viz disabled, or every video failed) logs a note and
    returns normally rather than crashing.
    """
    results = []
    last_reconstructor = None
    for video in videos:
        video = Path(video)
        logger.info("=== Video: %s ===", video.name)
        try:
            out, r = run_scene(video, output_root, config_dir, override_config, stages, overwrite)
            results.append((video.name, "OK", str(out)))
            last_reconstructor = r
        except Exception as exc:  # isolate one video's failure from the batch
            logger.exception("Video failed: %s", video.name)
            results.append((video.name, "FAIL", str(exc)))

    # Summary
    logger.info("==== Summary ====")
    for name, status, info in results:
        logger.info("%s: %s (%s)", status, name, info)
    failed = [n for n, s, _ in results if s == "FAIL"]

    # Optionally keep the last scene's viser viewer alive for browser inspection
    if keep_viewer:
        viewer = getattr(last_reconstructor, "viewer", None)
        if viewer is not None:
            logger.info("--keep-viewer: viser server staying up — inspect the scene in a browser (Ctrl-C to exit).")
            viewer.serve_forever()
        else:
            logger.info(
                "--keep-viewer set but no viewer was created (pointcloud.viz.enabled is false, "
                "or no video succeeded); nothing to keep alive."
            )

    return 1 if failed else 0
