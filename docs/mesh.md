# Mesh Module

`collab_splats.mesh` turns posed depth + RGB into a triangle mesh. Every entry point takes
**plain arrays**, never a `FeedforwardResult` or a `PointcloudResult` — the caller composes
them, because only the caller knows which resolution grid it is on.

| File | Responsibility |
| --- | --- |
| `tsdf.py` | `create_tsdf_mesh` — integrate views into a TSDF volume, write `mesh.ply` |
| `clean.py` | `get_scene_scale`, `remove_floaters`, `make_convex_hull` (+ `trim_mesh_edges`, `bridge_mesh_edges`), `fill_holes`, `clean_repair_mesh`; unwrap prep `decimate_mesh`, `make_manifold` |
| `texture.py` | `unwrap_mesh_uvs`, `project_images_to_texture`, `create_texture_mesh` |
| `features.py` | `features2vertex`, `mesh_clustering` |

---

## Quickstart

The pipeline runs this stage for you (`mesh:` in the yaml, `--stages mesh`). Direct use:

```python
from pathlib import Path

from collab_splats.mesh import clean_repair_mesh, create_tsdf_mesh

mesh_path = create_tsdf_mesh(
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

`create_tsdf_mesh` raises rather than fusing quietly wrong: float RGB is rejected outright, and a
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
from collab_splats.pointcloud.utils import confidence_mask
from collab_splats.preproc import read_frames
from collab_splats.utils.image import upsample_depths

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

**From a trained splat.** `render_tsdf_inputs` (in `collab_splats.splats.checkpoint`) renders every training camera out of the splats
stage's `ckpt.pt`. Renders come out at frame resolution carrying the poses they were rendered
with, pose-opt deltas included — nothing to lift, nothing to re-pose, and `splats.zarr` is not
an input:

```python
from collab_splats.splats.checkpoint import render_tsdf_inputs

depths, rgbs, c2w, K, image_ids = render_tsdf_inputs(scene_dir / "splats" / "ckpt.pt", images_dir)
```

`image_ids` is the source frame index behind each row, in the checkpoint's own render order —
the same values `frames.read_frames` takes, so anything wanting a per-view artifact (sky masks,
per-frame scores) can line it up without assuming the rows are in filename order.

Depth is zeroed where alpha is 0. `depth_source` picks which 2dgs render to fuse: `"expected"`
(the default) takes the alpha-weighted `depth`, defined wherever anything contributes at all, at
the cost of smearing across depth discontinuities; `"median"` takes `median_depth`, the ray's
median-transmittance surface — sharper, but blank wherever no gaussian crosses the median, which
is what opens holes on grazing ground. Measured on GH010229 (853 views, voxel 0.10) `"median"`
fuses fewer vertices in far fewer components (3.02M / 94,937 against 4.0M / 426,695) yet the
`"expected"` mesh is the better one to look at, so fragmentation is not the metric to pick on.
A 3dgs checkpoint renders only `depth` and ignores the choice. Passing `images_dir` swaps the rendered RGB for the source
keyframes matched by image id, which is what the pipeline does; omit it to fuse the render's
own color.

### Masking sky

Sky has no surface, but every depth source assigns it one — so it fuses as a backdrop and
seeds floaters. `mesh.mask_sky` zeroes depth wherever the sky segmenter fires, before fusion,
on both mesh sources:

```python
from collab_splats.semantics.segmentation import sky_masks

masks = sky_masks(images_dir, idxs=image_ids)   # (N, H, W) bool, True where sky
depths[masks] = 0.0
```

`idxs` takes SOURCE frame indices in the order wanted, so the splats arm passes
`render_tsdf_inputs`' `image_ids` and the feedforward arm passes nothing (filename order,
what `frames.read_frames` gives). Masks cache as PNGs under `<scene>/sky/`, so a re-run of
the stage pays for inference once.

The backend is `skywater`, a SegFormer MiT-B2 registered as `BaseSegmentation.get("skywater")`.
Masking applies to depth only, never to the splats stage's depth targets — that asymmetry with
`mesh.conf_percentile` is deliberate.

---

## Cleaning

`clean_repair_mesh(mesh_path)` rewrites `mesh.ply` in place: drop floating components, then
patch small holes with meshlib's `fillHoleNicely` (subdivided to the mesh's edge length and
smoothed to the hole boundary's curvature, so a patch does not read as a flat fan). **Every threshold is
relative to the mesh's own extent**, never a world distance, so one default works on a metric
scan and on a scale-free feedforward reconstruction alike. `get_scene_scale` is that extent: the 1st-to-99th-percentile diagonal of the vertex
cloud, which ignores the stray far component that would otherwise set the scale.

- `min_area_frac` (`6e-6`) — components smaller than this fraction of total surface area go
- `max_gap_frac` (`0.01`) — a component further than this fraction of scene scale from the main
  body goes, however large it is
- `max_hole_perimeter_ratio` (`0.014`) — holes whose perimeter is shorter than this × scene
  scale are filled; anything bigger is a real opening (a doorway, the missing back of the
  scene) and is left alone. On GH010229 (scene scale 181.4) that is a 2.5-unit perimeter
- `subdivide_fill` (`True`) — subdivide and smooth each patch. `False` triangulates the hole's
  boundary only: a flat lid, 8 s instead of 183 s on GH010229, barely visible on holes this small

Whatever the bound, a component's outer rim is never filled: its longest loop, when that loop
spans at least half the component's bounding box. A sheet's rim spans all of it; a hole in a
closed surface spans a fraction, so a sphere with one hole still gets it filled.

Cleaning is not optional in the pipeline. `remove_floaters`, `make_convex_hull` and
`fill_holes` are public for callers who want one without the others, as are the hull's
`trim_mesh_edges` and `bridge_mesh_edges`.

### Convex hull (`mesh.use_convex_hull`, off by default)

`clean_repair_mesh(mesh_path, use_convex_hull=True)` runs `make_convex_hull` between
`remove_floaters` and `fill_holes`. It trims the ragged outer edge of a fused scene and patches
the ground out to a rounded convex hull, so the outline is smooth and the edge has no strands:

1. **Frame** — the dominant plane from `geometry.fit_dominant_plane` (fixed RANSAC seed) is the
   ground; up is the side the area-weighted mean face normal points to (TSDF normals face free
   space). No cameras are needed
2. **Hull cut** — faces outside the rounded convex hull of the mesh's top-down coverage go
3. **Outline trim** (`trim_mesh_edges`) — faces outside the smoothed outline that connect to
   the outer rim go; interior patches stay
4. **Ground patch** — the uncovered hull cells get a triangulated sheet, pinned to the rim and
   pulled toward the eroded ground height away from the mesh; rim triangles reaching up a wall
   or a spike are dropped
5. **Join** — patch faces that would make a non-manifold or fold-over edge are dropped, then
   `make_manifold`
6. **Neck bridges** (`bridge_mesh_edges`) — rim vertices close in 3D but far apart along the outer loop are bridged,
   so inlets become interior holes that `fill_holes` then closes

Every length is in pixels of the top-down image, one pixel per median edge length (about one TSDF voxel). It assumes a
ground-dominated height field: leave it off indoors and for single objects. A mesh without a
dominant ground raises `ValueError` rather than inventing one.

---

## Texturing

`texture.py` bakes per-view color into a single albedo atlas, opt-in via `mesh.texture: true`:

1. `fill_holes` at full density with `max_hole_perimeter_ratio` (`3.9`) — the outer rims stay
   open by rule, so the bound only stops the largest interior openings (on GH010229, one hole
   at 4.0 × scene scale)
2. `decimate_mesh` — `meshoptimizer` simplification to an error bound expressed in voxels, so
   the budget follows the fusion resolution rather than a triangle count
3. `make_manifold` — split non-manifold vertices, drop degenerate, duplicate and fold-over
   faces. UVAtlas rejects a mesh that fails any of these
4. `unwrap_mesh_uvs` — Open3D UVAtlas, partitioned for parallelism
5. `_rasterize_atlas` — rasterize the atlas with nvdiffrast into per-texel world position
   and normal
6. `project_images_to_texture` — an NVIDIA Warp kernel per texel: ray-cast for occlusion
   (`wp.mesh_query_ray`) against the unfilled input, accumulate bilinear samples weighted by pixel size at the texel,
   from the views that resolve it nearly as finely as its best view, then `fill_missing_pixels` (push-pull)
   fill every texel no view reached
7. `write_textured_obj` (`utils/io.py`, trimesh) — `mesh.obj` + `mesh.mtl` + `albedo.png`, smooth vertex normals (`vn`),
   and a white diffuse (`Kd 1 1 1`); trimesh's `Kd 0.4` default darkens the texture in every
   viewer

Each step runs once. Filling after decimation instead leaves fold-over faces UVAtlas rejects
(`0x80004005`), and the order was measured against the alternatives on GH010229 (16.73 dB
reprojection PSNR, the best of four).

`create_texture_mesh(mesh_path, out_dir, ...)` writes `out_dir/mesh.obj` + `mesh.mtl` +
`albedo.png` and never modifies the fused mesh it reads; the pipeline passes `<backend>/texture/`.

Open3D's own projection path was tried first and does not work here: it is CPU-only (24 s per
view, OOM-killed at 8 views) and its image-resolution depth test lets occluded texels through.
Occlusion is Warp's BVH with a per-view depth buffer; only the atlas rasterization is
nvdiffrast's.

---

## Vertex features

`features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03)` transfers per-point
semantic features onto mesh vertices by Gaussian-weighted k-NN, zeroing vertices with
no neighbor inside `sdf_trunc`. `mesh_clustering` then groups high-scoring vertices into
spatially connected clusters. Both are used by the dashboard's semantic query view and by
`06_mesh/splats_mesh.ipynb`.
