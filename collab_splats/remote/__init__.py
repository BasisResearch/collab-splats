"""Remote (GCS via rclone) access to curated inputs and processed outputs."""

# rerun.py is deliberately NOT re-exported here. It imports LEAF_STAGES from
# wrapper.reconstructor, which pulls torch and pyvista — and dashboard/operation_log.py imports
# this package on the fast-bind path, which must stay light. Import it as
# `from collab_splats.remote.rerun import ...` from code that already loads the pipeline.
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
    "parse_rclone_percent",
]
