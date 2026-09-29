"""
Meshing from depth + RGB arrays.

- tsdf: fuse depth + RGB views into a mesh
- clean: drop floaters, fill small holes; decimate and repair for UV unwrap
- texture: unwrap, project the views into a UV atlas
- features: per-vertex feature transfer and clustering
"""

from __future__ import annotations

from collab_splats.mesh.clean import clean_repair_mesh
from collab_splats.mesh.features import features2vertex, mesh_clustering
from collab_splats.mesh.texture import create_texture_mesh
from collab_splats.mesh.tsdf import create_tsdf_mesh

__all__ = [
    "clean_repair_mesh",
    "create_texture_mesh",
    "create_tsdf_mesh",
    "features2vertex",
    "mesh_clustering",
]
