# 025 — Open3D TSDF reads K as pixel-corner

Date: 2026-10-08 · Status: accepted · Branch: `audit/mesh` · Amends: 024

## Context

- 024 listed Open3D as pixel-center; its `VoxelBlockGrid.integrate` floors projected u, v into [0, W−1], i.e. pixel-corner (pixel i spans [i, i + 1))
- proof (audit `verify_scratch/o3d_conv.py`): depth valid on cols 0..19, integrated voxels span u ∈ [0, 19.9]; center would give [−0.5, 19.5)
- feedforward source handed center K to the corner reader: every view fused 0.5 px off
- splats source handed the checkpoint's gsplat corner K to both sinks: TSDF right, texture (nvdiffrast, center) 0.5 px off

## Decision

- `create_tsdf_mesh` takes pixel-center K and shifts +0.5 once before Open3D, like `to_colmap` and the gsplat trainer
- `render_tsdf_inputs` returns pixel-center K: the checkpoint's corner K shifted −0.5
- `ckpt.pt` keeps the gsplat corner K it renders with

## Consequences

- both mesh sources now hand create_tsdf_mesh and create_texture_mesh the same convention
- meshes fused before this sit 0.5 px off per view; a mesh re-run fixes it (no legacy guard)
- `tests/mesh/test_tsdf.py::test_create_tsdf_mesh_reads_k_as_pixel_center` pins the fused footprint edge at u = −0.5
