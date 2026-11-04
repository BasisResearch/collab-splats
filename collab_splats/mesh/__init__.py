"""
Meshing from depth + RGB arrays.

- tsdf: fuse depth + RGB views into a mesh
- clean: drop floaters, fill small holes; prepare_mesh for mesh.ply
- texture: unwrap, project the views into a UV atlas
- utils: meshlib conversion, face connectivity, the view-array contract
"""

from __future__ import annotations

from collab_splats.mesh.clean import (
    VolumeGround,
    build_volume_ground,
    clean_repair_mesh,
    prepare_mesh,
    solve_ground,
)
from collab_splats.mesh.texture import create_texture_mesh
from collab_splats.mesh.tsdf import (
    calibrate_confidence,
    compute_tsdf_voxel_size,
    create_tsdf_mesh,
    lift_weights,
    pixel_weights,
)

__all__ = [
    "VolumeGround",
    "build_volume_ground",
    "calibrate_confidence",
    "clean_repair_mesh",
    "compute_tsdf_voxel_size",
    "create_texture_mesh",
    "create_tsdf_mesh",
    "lift_weights",
    "pixel_weights",
    "prepare_mesh",
    "solve_ground",
]
