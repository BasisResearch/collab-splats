# Mesh texture bake — design

Date: 2026-08-29
Status: approved, not implemented

## Problem

`scaffold2dgs_500f_50k_mesh/mesh_clean.ply` (1.63M verts / 3.05M faces, vertex colours only)
has three visual-quality limits, all consequences of the TSDF + marching-cubes output shape:

1. **Uniform tessellation.** Edge length p50 0.20 / p90 0.28 scene units everywhere. Flat
   walls and floor spend the same triangle density as curved poles. Nothing decimates.
2. **Staircase on curved surfaces.** Voxel-quantised surfaces on poles and pipes; no
   smoothing pass exists.
3. **Colour resolution = vertex density.** RGB lives on vertices, so colour detail is capped
   at the voxel grid. This is the dominant limit.

Downstream consumers (collab-data `track_reprojection`, Blender replica, demo viewer) read a
vertex-coloured PLY only. collab-data ships `bake.py` (`d059661`) that flattens a textured
OBJ to `<stem>_baked.ply`, subdividing until `target_faces` so texture detail survives as
vertex density. Our stage must therefore emit a Polycam-shaped OBJ + textures, and the baked
PLY, while leaving `mesh.ply` exactly as it is today.

## Decisions taken during brainstorming

- **Not a new stage.** Texturing is the final step of the `mesh` stage, gated by
  `mesh.texture.enabled`. Same skip-check (`mesh.ply`), same `overwrite` semantics.
- **`mesh.ply` untouched.** New artefacts land in `backend_dir/texture/<source>/`; every
  existing reader (Reconstructor skip-check, dashboard, viser, remote push) keeps working.
- **Two texture sources, one code path**, so they can be compared on the same geometry:
  `splats` (rendered RGB in `splats.zarr`, view-consistent, pose-opt deltas included) and
  `frames` (raw `frames.zarr` pixels, sharpest but exposure/pose seams). Cameras always come
  from `splats.zarr` (`c2w`, `K`); only the image array differs.
- **Error-bounded decimation, not a face budget.** QEM stops when the next collapse would
  move the surface more than `decimate_max_error * voxel_size` (absolute deviation, scene
  units). Marching cubes cannot place a surface more accurately than ~`voxel_size`, so
  collapses well under that remove quantisation noise, not signal. Expressed in voxel units
  so it means the same thing on a metric scene (voxel 0.0025) and a scaffold scene (0.2) —
  the pipeline's rule that thresholds are fractions of a scene scale, never world distances.
  A number, not a percentile: the simplifier takes a bound; the report measures the
  resulting deviation percentiles.
- **Libraries.** Decimation: **meshoptimizer** (new dependency; pure-C wheel, no CUDA).
  It is the simplifier with a true absolute-error stop (`SIMPLIFY_ERROR_ABSOLUTE` +
  `target_error`, returns `result_error`) and is what production glTF/engine toolchains use.
  Open3D's legacy `simplify_quadric_decimation(maximum_error=...)` was tried first and
  rejected by measurement: its error is an accumulated quadric sum that grows with region
  size (a noisy plane never flattens, a sphere over-collapses), so it is not a distance.
  pymeshlab's QEM stops only on face count. Smoothing: Open3D legacy `filter_smooth_taubin`
  (only maintained shrink-free smoother in an installed lib). UV atlas + projection: Open3D
  tensor API (only numpy-native option). nvdiffrast is out of scope until measured seams
  justify it.
- **Approach 1 of 3** (Open3D blend, no seam solving, no view selection, no inpainting).
  Measure first; escalate only on evidence.

## Pipeline

Runs inside `mesh_from_tsdf_inputs` after fusion → `clean_repair` → `optimize_color_map`,
so it textures the final geometry.

```
mesh.ply ──► geometry pass ──► UV atlas ──► albedo / normal projection ──► OBJ+MTL+PNG
                                                                           └─► bake-back ──► mesh_baked.ply
```

### Geometry pass (legacy `o3d.geometry.TriangleMesh`)

1. `remove_non_manifold_edges`, `remove_degenerate_triangles`, `remove_unreferenced_vertices`
   — QEM needs a clean edge structure; TSDF output has neither.
2. `filter_smooth_taubin(number_of_iterations=smooth_iterations)` — shrink-free. Default 0
   (off): it rounds box edges and text as readily as poles. Runs before decimation so QEM sees
   the smoothed surface.
3. `meshoptimizer.simplify(target_index_count=3, target_error=decimate_max_error*voxel_size,
   options=SIMPLIFY_ERROR_ABSOLUTE)` — the tiny index target makes the error bound the only
   stop; vertices are never moved, only removed. `decimate_max_error: null` skips decimation.
4. `compute_vertex_normals` — needed by the OBJ writer.

Measured on `scaffold2dgs_500f_50k_mesh/mesh_clean.ply` (3.05M faces, voxel 0.2), 2.2 s per
run; deviation = original vertices → decimated surface (Open3D `RaycastingScene`):

| bound (voxels) | faces | p90 | p99 | max | > 1 voxel |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 497k (16%) | 0.033 | 0.066 | 0.73 | 0.09% |
| 0.5 | 232k (7.6%) | 0.060 | 0.138 | 2.15 | 0.48% |
| 1.0 | 74k (2.4%) | 0.127 | 0.343 | 2.67 | 3.4% |

`result_error` tracks p99 within ~1.5x. Default 0.25.

### Texture pass (tensor `o3d.t.geometry.TriangleMesh`)

1. `compute_uvatlas(size=tex_size)` on the decimated mesh.
2. Images: `splats` → `splats.zarr/rgb`; `frames` → `FrameStore.image(i)` for each row `i` of
   `splats.zarr.attrs["image_ids"]`, with `K` rescaled from render to frame resolution when
   they differ. Cameras: `splats.zarr/c2w` inverted to extrinsics, `splats.zarr/K`.
3. `project_images_to_albedo(images, K, w2c, tex_size)` → `albedo.png`.
4. Normal map: same projector on `splats.zarr/normal` (world-frame, `(n+1)/2*255` uint8,
   renormalised after decode) → `normal_world.png`. **World-space, not tangent-space** — a
   `map_bump` reader that assumes tangent space will shade wrongly; the name and the MTL
   comment say so. Splat normals serve both sources (frames carry none). Skipped when the
   store has no `normal` array.
5. `o3d.t.io.write_triangle_mesh("mesh.obj")`; MTL carries `map_Kd albedo.png` and
   `map_bump normal_world.png`.

All views integrate — 500 frames at 1920x1080 is ~3 GB uint8 per array, well inside the
46 GB cap. `frames` guard: `frames.zarr` image shape must equal `splats.zarr/rgb` shape
(splats may have trained on undistorted or resized frames), else raise pointing at
`source: splats`.

### Bake-back

`bake_vertex_colors` copied into `collab_splats/mesh/texture.py` with attribution
(collab-data `d059661`, `collab_data/track_reprojection/bake.py`) — collab-data is not a
dependency of this repo. Subdivides until `bake_target_faces`, samples albedo at each vertex
UV, writes `mesh_baked.ply`. Normal map does not survive into the PLY (vertex-colour contract).

### Report — `texture_report.json`

- faces / vertices: fused, after cleanup, after decimation, baked
- decimation: bound in voxels and scene units, meshoptimizer `result_error`, and measured
  deviation (fused vertices → decimated surface) p50 / p90 / p99 / max plus fraction beyond
  one voxel — the table above, per run
- UV atlas: chart count, texel fill fraction (non-zero alpha in the albedo)
- wall-time per step

## Config

Nested under the existing `mesh:` block in `configs/base.yaml`:

```yaml
mesh:
  ...
  texture:
    enabled: false
    source: splats            # splats | frames — images only; cameras always from splats.zarr
    decimate_max_error: 0.25  # max surface deviation in voxel_size units; null = no decimation
    smooth_iterations: 0      # Taubin passes before decimation; 0 = off
    tex_size: 8192            # albedo / normal map side in texels
    bake_target_faces: 500000 # bake-back subdivides until at least this many faces
```

`source: splats` and `frames` both require `splats.zarr` on disk (cameras + normals); the
mesh stage already refuses to run `mesh.source: splats` without it, and the texture step
raises the same error for either source.

## Code

- `collab_splats/mesh/texture.py` (new, ~250 lines): `texture_mesh(mesh_path, out_dir,
  images, normals, c2w, intrinsics, voxel_size, cfg) -> Path` plus private helpers for the
  three passes, the report, and the ported `bake_vertex_colors`. Imports at top; block
  comments per step.
- `collab_splats/mesh/utils.py`: `mesh_from_tsdf_inputs` gains `texture: dict | None` and
  calls `texture_mesh` last. The splats/frames image loading lives in a small
  `_texture_images(splats_zarr, frames_zarr, source)` next to `_splats_to_tsdf_inputs`.
- `collab_splats/wrapper/reconstructor.py`: `mesh()` passes `mesh_cfg["texture"]` through
  `_run_tsdf_mesh`; both source branches reach it.
- `configs/base.yaml`, `configs/README.md`: the block above.
- `docs/examples/texture_mesh.py`: standalone entry — `mesh.ply` path + `splats.zarr`
  (+ `frames.zarr`) + config → `texture/<source>/`; how the measurement runs on
  `/workspace/outputs/scaffold2dgs_500f_50k_mesh/mesh_clean.ply` without a Reconstructor.
- New dependency: `meshoptimizer` in `pyproject.toml`. `Pillow` (already present) writes PNGs.

## Outputs

```
backend_dir/
  mesh.ply                       # unchanged
  texture/<source>/
    mesh.obj  mesh.mtl           # decimated, UV-mapped
    albedo.png  normal_world.png
    mesh_baked.ply               # collab-data contract
    texture_report.json
```

## Testing

`tests/mesh/test_texture.py`, flat functions:

- Synthetic textured cube: one rendered view (Open3D offscreen) of a solid-colour cube →
  full chain → face count ≤ fused count, albedo non-black at the visible face's texels, baked
  PLY vertex colours on that face match the source colour within 8/255.
- Error bound: noisy plane (σ = 0.25 bound) and sphere decimate with measured p99 deviation
  ≤ 1.5x the bound, and the plane ends with fewer faces than the sphere.
- `decimate_max_error: null` leaves face count unchanged.
- `frames` source with mismatched frame shape raises.
- Bake-back: the two collab-data tests ported (texel orientation, subdivision count).
- Missing `normal` array → `normal_world.png` absent, no error.

## Measurement plan (before defaults are final)

On `scaffold2dgs_500f_50k_mesh` (voxel 0.2):

1. Both sources at defaults. Renders of OBJ (textured) vs `mesh_clean.ply` from 5 scene
   cameras; side-by-side crops on a wall, a pole, and text.
2. `decimate_max_error` at 0.25 / 0.5 / 1.0: renders of the textured OBJ on the pole crop
   (face counts and deviation already tabulated above).
3. `smooth_iterations` at 0 / 5 / 10 on the pole crop.

Results and the chosen defaults are appended to this spec; nothing ships as default without
a row in that table.

## Out of scope

- Seam-aware view selection, texture inpainting, differentiable baking (nvdiffrast).
- Novel-view splat renders as a texture source (approach C in brainstorming) — follow-up if
  texel coverage on oblique surfaces is poor.
- Any change to `mesh.ply` or its readers.
