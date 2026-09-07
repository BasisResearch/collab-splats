"""
Meshing from depth + RGB arrays.

  - tsdf.fuse_tsdf: integrate views into a mesh
  - clean.clean_repair_mesh: drop floaters, fill small holes
  - texture.texture_mesh: decimate, unwrap, project the views into a UV atlas
  - features: per-vertex feature transfer and clustering
"""

from __future__ import annotations

from collab_splats.mesh.clean import clean_repair_mesh
from collab_splats.mesh.features import features2vertex, mesh_clustering
from collab_splats.mesh.texture import texture_mesh
from collab_splats.mesh.tsdf import fuse_tsdf

__all__ = [
    "clean_repair_mesh",
    "features2vertex",
    "fuse_tsdf",
    "mesh_clustering",
    "texture_mesh",
]
