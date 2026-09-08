# Mesh Module

`collab_splats.mesh` turns posed depth + RGB into a triangle mesh. Every entry point takes
**plain arrays**, never a `FeedforwardResult` or a `PointcloudResult` — the caller composes
them, because only the caller knows which resolution grid it is on.

| File | Responsibility |
| --- | --- |
| `io.py` | Getting arrays in and out: `upsample_depths`, `render_tsdf_inputs`, `write_textured_ply` |
| `tsdf.py` | `fuse_tsdf` — integrate views into a TSDF volume, write `mesh.ply` |
| `clean.py` | `get_scene_scale`, `remove_floaters`, `fill_holes`, `clean_repair_mesh` |
| `texture.py` | `decimate_mesh`, `unwrap_mesh_uvs`, `bake_atlas_attributes`, `project_images_to_texture`, `texture_mesh` |
| `features.py` | `features2vertex`, `mesh_clustering` |

---

## Quickstart

The pipeline runs this stage for you (`mesh:` in the yaml, `--stages mesh`). Direct use:

```python
from pathlib import Path

from collab_splats.mesh import clean_repair_mesh, fuse_tsdf

mesh_path = fuse_tsdf(
    depths,       # (n, h, w) float32, 0 = no observation
    rgbs,         # (n, h, w, 3) uint8
    c2w,          # (n, 4, 4) camera-to-world
    K,            # (n, 3, 3) at the depth resolution
    Path("scene/mesh"),
    voxel_size=0.0025,
    depth_trunc=1.5,
)
clean_repair_mesh(mesh_path)   # rewrites mesh.ply in place
```

`voxel_size` and `depth_trunc` are in **world units, and the world has no fixed scale**. The
values above are `base.yaml`'s, tuned for feedforward backbones whose depth is normalized to
roughly unit scale. A COLMAP-scale reconstruction (`pointcloud.method: sfm`) is typically one
to two orders of magnitude larger — on GH010229 the median depth is 16.5 and the camera
trajectory spans 145, so `depth_trunc=1.5` truncates every sample and fusion returns an empty
mesh with only an `[Open3D WARNING] Write PLY failed: mesh has 0 vertices.` on stderr. Measure
first: `np.percentile(depths[depths > 0], [50, 95])`, put `depth_trunc` near p95, and scale
`voxel_size` by the same ratio.

`fuse_tsdf` raises rather than fusing quietly wrong: float RGB is rejected outright, and a
principal point outside the depth grid raises, because pairing one grid's depth with the other
grid's `K` collapses the mesh instead of failing (the 2026-08-11 regression). An out-of-scale
`depth_trunc` is NOT in that set — Open3D treats it as a legitimately empty volume.

`sdf_trunc` defaults to `4 × voxel_size` and is a keyword for callers doing parity work, not a
tuning knob; the pipeline never sets it.

---

## Composing the inputs

**From a feedforward reconstruction, at frame resolution.** Model-resolution depth is
guided-upsampled into the original frames so the COLMAP camera is the right `K` to fuse with:

```python
import numpy as np

from collab_splats.geometry.transforms import invert_poses
from collab_splats.mesh.io import upsample_depths
from collab_splats.pointcloud.utils import confidence_mask
from collab_splats.preproc import read_frames

depth = np.asarray(ff.depth)
depth = np.where(confidence_mask(np.asarray(ff.confidence), 20), depth, 0.0)
rgbs = read_frames(images_dir)
depths = upsample_depths(depth, rgbs, np.asarray(ff.original_coords)[:, :4])
c2w = invert_poses(result.extrinsics)   # COLMAP poses, matched with result.intrinsics
```

`result` is the `PointcloudResult`, `ff` the `FeedforwardResult` loaded from the same
`pointcloud.zarr`. The poses and `K` both come from `result` because the depth was lifted onto
the original frame grid — mixing one grid's depth with the other grid's `K` is the collapse bug
above.

`confidence_mask` is skipped when the reconstruction carries no confidence array — `sfm` does
not produce one, and the arrays are absent rather than zero-filled.

**At model resolution** (notebooks, evals): use `ff.depth`, `ff.images` transposed to
`(n, h, w, 3)` and scaled to uint8, `invert_poses(ff.extrinsics)` and `ff.intrinsics`. No lift,
no COLMAP camera. `02_pointcloud/feedforward_mesh.ipynb` is this path end to end.

**From a trained splat.** `render_tsdf_inputs` renders every training camera out of the splats
stage's `ckpt.pt`. Renders come out at frame resolution carrying the poses they were rendered
with, pose-opt deltas included — nothing to lift, nothing to re-pose, and `splats.zarr` is not
an input:

```python
from collab_splats.mesh.io import render_tsdf_inputs

depths, rgbs, c2w, K = render_tsdf_inputs(scene_dir / "splats" / "ckpt.pt", images_dir)
```

Depth is zeroed where alpha is 0. `depth_source` picks which 2dgs render to fuse: `"expected"`
(the default) takes the alpha-weighted `depth`, defined wherever anything contributes at all, at
the cost of smearing across depth discontinuities; `"median"` takes `median_depth`, the ray's
median-transmittance surface — sharper, but blank wherever no gaussian crosses the median, which
is what opens holes on grazing ground. Measured on GH010229 (853 views, voxel 0.10) `"median"`
fuses fewer vertices in far fewer components (3.02M / 94,937 against 4.0M / 426,695) yet the
`"expected"` mesh is the better one to look at, so fragmentation is not the metric to pick on.
A 3dgs checkpoint renders only `depth` and ignores the choice. Passing `images_dir` swaps the rendered RGB for the source
keyframes matched by image id, which is what the pipeline does; omit it to fuse the render's
own color. `gsplat` is imported inside the function, so `collab_splats.mesh.io` still loads on
a machine without CUDA.

---

## Cleaning

`clean_repair_mesh(mesh_path)` rewrites `mesh.ply` in place: drop floating components, then
fill small holes. **Every threshold is a fraction of the mesh's own extent**, never a world
distance, so one default works on a metric scan and on a scale-free feedforward reconstruction
alike. `get_scene_scale` is that extent: the 1st-to-99th-percentile diagonal of the vertex
cloud, which ignores the stray far component that would otherwise set the scale.

- `min_area_frac` (`6e-6`) — components smaller than this fraction of total surface area go
- `max_gap_frac` (`0.01`) — a component further than this fraction of scene scale from the main
  body goes, however large it is
- `max_hole_frac` (`0.0045`) — holes whose boundary is shorter than this fraction of scene
  scale are filled; anything bigger is a real opening (a doorway, the missing back of the
  scene) and is left alone

Cleaning is not optional in the pipeline and has no config key. `remove_floaters` and
`fill_holes` are public for callers who want one without the other.

---

## Texturing

`texture.py` bakes per-view color into a single albedo atlas, opt-in via `mesh.texture: true`:

1. `decimate_mesh` — `meshoptimizer` simplification to an error bound expressed in voxels, so
   the budget follows the fusion resolution rather than a triangle count
2. `_make_manifold` — split non-manifold vertices, drop degenerate, duplicate and fold-over
   faces. UVAtlas rejects a mesh that fails any of these
3. `unwrap_mesh_uvs` — Open3D UVAtlas, partitioned for parallelism
4. `bake_atlas_attributes` — rasterize the atlas with nvdiffrast into per-texel world position
   and normal
5. `project_images_to_texture` — an NVIDIA Warp kernel per texel: ray-cast for occlusion
   (`wp.mesh_query_ray`), accumulate cosine-weighted bilinear samples from every view that sees
   it, then dilate into the gutter
6. `write_textured_ply` — Open3D's PLY writer cannot emit UVs, so this writes the PLY with
   `plyfile` and a `comment TextureFile albedo.png` line

`texture_mesh(mesh_path, out_dir, ...)` writes `out_dir/mesh.ply` + `out_dir/albedo.png` and
never modifies the fused mesh it reads; the pipeline passes `<backend>/texture/`.

Open3D's own projection path was tried first and does not work here: it is CPU-only (24 s per
view, OOM-killed at 8 views) and its image-resolution depth test lets occluded texels through.
Occlusion is Warp's BVH with a per-view depth buffer; only the atlas rasterization is
nvdiffrast's.

---

## Vertex features

`features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03)` transfers per-point
semantic features onto mesh vertices by inverse-distance-weighted k-NN, zeroing vertices with
no neighbor inside `sdf_trunc`. `mesh_clustering` then groups high-scoring vertices into
spatially connected clusters. Both are used by the dashboard's semantic query view and by
`06_mesh/splats_mesh.ipynb`.
