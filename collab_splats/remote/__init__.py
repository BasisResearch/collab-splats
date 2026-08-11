"""Remote (GCS via rclone) access to curated inputs and processed outputs."""

from collab_splats.remote.rerun import discover_scenes, prepare_scene
from collab_splats.remote.sources import (
    CURATED_BUCKET,
    PROCESSED_BUCKET,
    PULL_EXCLUDES,
    PUSH_EXCLUDES,
    SCENE_ID_RE,
    SceneSource,
    parse_rclone_percent,
)

__all__ = [
    "CURATED_BUCKET",
    "PROCESSED_BUCKET",
    "PULL_EXCLUDES",
    "PUSH_EXCLUDES",
    "SCENE_ID_RE",
    "SceneSource",
    "discover_scenes",
    "parse_rclone_percent",
    "prepare_scene",
]
