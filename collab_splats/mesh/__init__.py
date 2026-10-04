"""
Meshing from depth + RGB arrays.

- tsdf: fuse depth + RGB views into a mesh
- clean: drop floaters, fill small holes; prepare_mesh for mesh.ply
- texture: unwrap, project the views into a UV atlas
- utils: meshlib conversion, face connectivity, the view-array contract
"""

from __future__ import annotations

from collab_splats.mesh.clean import clean_repair_mesh, prepare_mesh
from collab_splats.mesh.texture import create_texture_mesh
from collab_splats.mesh.tsdf import create_tsdf_mesh

__all__ = [
    "clean_repair_mesh",
    "create_texture_mesh",
    "create_tsdf_mesh",
    "prepare_mesh",
]
