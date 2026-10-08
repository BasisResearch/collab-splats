"""
SfM backends for `pointcloud.method: sfm`: InstantSfM (global), COLMAP and hloc (incremental).

- every creator is a BaseSfmCreator: create_pointcloud(images_dir, out_dir, model_dir) -> PointcloudResult
- the COLMAP model is written to model_dir, image names as stems
- SFM_CREATORS maps the `backend` config key to its creator
"""

from collab_splats.pointcloud.sfm.colmap import ColmapCreator
from collab_splats.pointcloud.sfm.hloc import HlocCreator
from collab_splats.pointcloud.sfm.instantsfm import InstantSfMCreator

SFM_CREATORS = {
    "instantsfm": InstantSfMCreator,
    "colmap": ColmapCreator,
    "hloc": HlocCreator,
}

__all__ = ["SFM_CREATORS", "ColmapCreator", "HlocCreator", "InstantSfMCreator"]
