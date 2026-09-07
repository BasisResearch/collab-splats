# Mesh module cleanup — design

**Date:** 2026-09-06
**Scope:** `collab_splats/mesh/`, `tests/mesh/`, the mesh stage of `collab_splats/wrapper/reconstructor.py`, `collab_splats/dashboard/{pipeline,viewer}.py`, `configs/base.yaml` `mesh:` block, `configs/README.md`, `docs/source/api/mesh.rst`, `docs/source/conf.py`, `docs/source/tutorials/06_mesh/splats_mesh.ipynb`, `evals/scripts/analyze_splats.py`, `pyproject.toml`, `CLAUDE.md` tree line, `docs/mesh.md` (new), CHANGELOG.
**Branch:** `clean/mesh`, forked at `6f060dfa` like the other `clean/*` branches. Execution waits until every `clean/*` branch is rebased onto the reorganised trunk; see "Branch coordination".
**Status:** approved design, awaiting plan.

## Goal

Reduce overengineering in the mesh module: one function per concern, arrays in and paths
out, no coupling to result types, no module constants, every docstring says what the function
does, what it takes and what it returns. Fold the `feat/mesh-texture-bake` texture pass in,
replacing its CPU projection with a GPU one. Behaviour of the shipped path (feedforward
source, native resolution, confidence gate, clean_repair) is preserved; see Verification.

Frozen public surface: the `mesh:` yaml block below, `mesh.ply` as the fused mesh filename,
`vertex_features.npy` beside it, and the texture output directory layout. Everything else in
the package may be renamed, moved or deleted.

## Audit findings that drive the design

| # | Finding | Resolution |
|---|---------|-----------|
| 1 | `utils.py` (989 lines) mixes five concerns: view inputs, guided upsampling, cleaning, feature transfer, clustering, plus two orchestration wrappers | Split into `io`, `tsdf`, `clean`, `texture`, `features`; `utils.py` deleted |
| 2 | `BaseMeshCreator` / `MeshResult` / `REGISTRY` / `get_mesh_creator` / `poisson.py` serve one concrete mesher; the Poisson classes raise `NotImplementedError`; REGISTRY has duplicate keys | One function `fuse_tsdf`; the rest deleted |
| 3 | meshlib is used for hole filling only, drags a colour round-trip through numpy, a subdivide + smooth pass and a `max_edge_splits` budget | Open3D tensor `fill_holes`; meshlib dependency, its docs mock and the stale pytest ignore go |
| 4 | `optimize_color_map` defaults to 300 iterations in yaml, but every real caller (notebook, evals, reconstructor test) passes 0 and it OOMs at 300 native frames | Deleted; the texture pass is the colour path |
| 5 | `_depth_edge_mask`, `splat_max_depth_frac/grad`: built, measured harmful (grad 0.05 deleted whole surfaces), ship off | Deleted |
| 6 | `pick_indices_at_random`, `find_depth_edges`, `transfer_features_to_mesh`, `persist_mesh_vertex_features`, `normals2vertex`, `align_geometry_floor`: no callers outside tests or one three-line dashboard site | Deleted |
| 7 | Adapters are typed on `FeedforwardResult`; the sfm path already fakes one; `PointcloudResult` on `clean/pointcloud` holds poses and K and cannot be meshed through them | Mesh takes arrays; callers compose from whichever object holds poses |
| 8 | `splats.zarr` (mesh input) is being retired by `clean/splats`; `frames.zarr` (native-res RGB) is already retired by `clean/preproc` | Splats input renders from `ckpt.pt`; native RGB reads `images/` via `preproc.frames` |
| 9 | `mesh_clustering` builds a dense `(n, n)` bool adjacency in a Python loop | `cKDTree.query_pairs` → sparse → `connected_components` |
| 10 | Texture branch bakes and projects with Open3D: `bake_vertex_attr_textures` and `project_images_to_albedo` are both CPU, 24 s/view, OOM past two 8192² views — it never ran this pipeline | nvdiffrast rasterizes the atlas, a Warp kernel with BVH occlusion projects, one image resident at a time (occlusion stays Warp's ray cast — a depth buffer is what Open3D got wrong) |
| 11 | `mesh:` yaml block has 14 keys; six are always default or derivable | Six keys; the rest derived or function defaults |
| 12 | Module constants (`_GAP_KDTREE_POINTS`, `_MIN_FACES_PER_PARTITION`, `_MM_AVAILABLE`) and inline literals (`> 10` cluster floor) | Every knob is a keyword argument with a default |
| 13 | Inline imports (`get_mesh_creator`, dashboard viewer, reconstructor mesh stage) | All imports at top |
| 14 | Texture branch writes four outputs in two formats: OBJ + MTL + PNG through trimesh, plus a subdivided vertex-colour `mesh_baked.ply` (`bake_vertex_colors`, ~60 lines); PLY is the mesh format everywhere else in the repo | One textured PLY (`s`/`t` UVs + vertex colours + `comment TextureFile albedo.png`, written with plyfile) beside `albedo.png`; OBJ writer and bake-back deleted |
| 15 | World-normal texture map (`project_normals`, a second accumulator, 2dgs renders only): no consumer in the repo | Dropped; the view stack is the same four arrays for every source |

## File layout

```
collab_splats/mesh/
  __init__.py   re-exports fuse_tsdf, clean_repair_mesh, texture_mesh, features2vertex, mesh_clustering
  io.py         view stacks in, meshes out: upsample_depths, render_tsdf_inputs, write_textured_ply   ~150
  tsdf.py       fuse_tsdf                                                                    ~80
  clean.py      get_scene_scale, remove_floaters, fill_holes, clean_repair_mesh              ~120
  texture.py    decimate_mesh, unwrap_mesh_uvs, project_images_to_texture, texture_mesh      ~280
  features.py   features2vertex, mesh_clustering                                             ~100
```

Deleted: `mesh/base.py`, `mesh/poisson.py`, `mesh/utils.py`, `tests/mesh/test_feature_transfer.py`,
`tests/mesh/test_utils_ground_plane.py`.

`io.py` is fine as `collab_splats.mesh.io`: absolute imports mean no clash with the stdlib
module. It shadows `io` only for a script run with its cwd inside `collab_splats/mesh/`.

## Contract: the view stack

Every consumer takes the same four arrays. Nothing in `mesh/` imports `FeedforwardResult`,
`PointcloudResult` or a zarr store.

```
depths  (N, H, W)     float32, world units, 0 = no observation
rgbs    (N, H, W, 3)  uint8
c2w     (N, 4, 4)     float32 camera-to-world
K       (N, 3, 3)     float32, at the depth/rgb resolution
```

`fuse_tsdf` checks N, H, W agree across the four arrays, that `rgbs` is uint8, and that every
principal point lies inside the `(W, H)` grid — the check that catches original-res K paired
with model-res depth (the 2026-08-11 regression class).

## Module details

### io.py — view stacks in, meshes out

```python
def upsample_depths(depths, rgbs, crop_boxes) -> np.ndarray
```
- `depths` `(N, h, w)` model-res, `rgbs` `(N, H, W, 3)` uint8 photos, `crop_boxes` `(N, 4)`
  = `original_coords[:, :4]` (tl_x, tl_y, cr_x, cr_y in photo pixels).
- Returns `(N, H, W)` float32: each depth map guided-filtered by its photo and placed into
  its crop box on a zero canvas. Today's `guided_upsample_depth` body becomes the private
  per-frame helper; `_box` and `_guided_filter` stay private.
- Raises `ValueError` on N mismatch or a crop box outside the canvas.

```python
def render_tsdf_inputs(ckpt_path, images_dir=None, device="cuda")
    -> tuple[depths, rgbs, c2w, K]
```
- `splats.rendering.load_checkpoint(ckpt_path, device)` gives the model, appearance module,
  `cam_to_world`, `intrinsics`, `image_ids`, `(height, width)`; `render_views(...)` yields one
  render dict per view. Each is copied into preallocated stacks and dropped; renders are
  never all resident.
- Depth key is derived from the checkpoint's `primitive`: `median_depth` for 2dgs (the
  measured 2dgs lever), `depth` for 3dgs. No `splat_depth` argument.
- `alpha == 0` pixels get depth 0.
- `images_dir` swaps the rendered `rgb` for the photo at `image_ids[i]` via
  `preproc.frames.read_frames(images_dir, [idx])`: sharp texture source, rendered geometry.
  Splats train coarse-to-native, so renders and photos are the same resolution — no resize.
- Poses are the checkpoint's (pose-refined). No frame-count cross-check against COLMAP.
- Dropped relative to `_splats_to_tsdf_inputs`: `conf_percentile` (alpha median 0.998 on
  GH010229, a measured no-op — `remove_floaters` does that work), `max_depth_frac`,
  `max_depth_grad`.

```python
def write_textured_ply(mesh, uv, albedo, out_dir) -> Path
```
- `mesh` legacy `TriangleMesh`, `uv` `(F, 3, 2)` float32 per-corner atlas coordinates
  (`tm.triangle.texture_uvs` from `unwrap_mesh_uvs`), `albedo` `(S, S, 3)` uint8.
- Splits vertices per face corner (UV seams need it), samples `albedo` at each corner's UV
  for its vertex colour, writes `out_dir/mesh.ply` with plyfile — `x y z red green blue s t`
  per vertex, `vertex_indices` per face, header `comment TextureFile albedo.png` — and saves
  `albedo.png` beside it. Returns the PLY path.
- Reader support (probed 2026-09-06): trimesh and MeshLab load the atlas from the comment
  (viser goes through trimesh); Blender reads `s`/`t`; Open3D reads vertices, colours and
  faces and ignores the UVs, so every existing vertex-colour consumer reads the same file.
  Open3D's PLY writer cannot emit UVs and trimesh's emits `s`/`t` without the texture link,
  hence plyfile (already a dependency).

### tsdf.py — one function

```python
def fuse_tsdf(depths, rgbs, c2w, K, out_dir, voxel_size, depth_trunc, sdf_trunc=None) -> Path
```
- `sdf_trunc=None` → `4 * voxel_size`.
- Guards from "Contract" above, then `ScalableTSDFVolume(RGB8)`, per-view
  `RGBDImage.create_from_color_and_depth(depth_scale=1.0, depth_trunc=..., convert_rgb_to_intensity=False)`,
  `integrate`, `extract_triangle_mesh`, `write_triangle_mesh(out_dir / "mesh.ply")`.
- Today's loop minus the float-RGB branch and `depth_scale` (always 1.0, never passed).
- Does not clean. Callers chain `clean_repair_mesh` (item 4 of the request: cleaning and
  fusing are separate functions).

### clean.py — cleaning and repairing, separately

```python
def get_scene_scale(vertices) -> float
def remove_floaters(mesh, min_area_frac=6e-6, max_gap_frac=0.01, gap_kdtree_points=200_000) -> TriangleMesh
def fill_holes(mesh, max_hole_frac=0.0045) -> TriangleMesh
def clean_repair_mesh(mesh_path, min_area_frac=6e-6, max_gap_frac=0.01, max_hole_frac=0.0045) -> Path
```
- `get_scene_scale`: p1–p99 AABB diagonal; today's `_scene_scale`.
- `remove_floaters`: `cluster_connected_triangles` → scale from the largest component only →
  per-component area and centroid-to-main-body gap (KD-tree over a strided subsample of at
  most `gap_kdtree_points` points) → keep `area >= min_area_frac * scale**2` and
  `gap <= max_gap_frac * scale`, largest always kept → `remove_triangles_by_mask`,
  `remove_unreferenced_vertices`. Today's `_select_components` folded in.
- `fill_holes`: `scale = get_scene_scale(mesh.vertices)` of the mesh it is given (after
  `remove_floaters` that is the main body, so it matches the scale the floater pass used), then
  `o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes(hole_size=max_hole_frac * scale).to_legacy()`.
  Open3D keeps vertex colours through the fill (probed 2026-09-06), so the numpy colour
  round-trip goes. Fills are flat caps; at under half a percent of the scene diameter that is
  invisible, so the subdivide + smooth pass and `max_edge_splits` go.
- **Threshold recalibration, still scale-relative.** Today fills holes with perimeter below
  `0.014 * scale`. Open3D's `hole_size` is diameter-like (probe: a hole is filled iff
  `hole_size` ≳ 2 × its radius). For a round hole, diameter = perimeter / π, so
  `0.014 / π ≈ 0.0045`. The docstring states the unit.
- `clean_repair_mesh`: read → `remove_floaters` → `fill_holes` → write in place → return the
  path. Reconstructor and dashboard entry point.

### texture.py — texture pass with nvdiffrast + Warp

```python
def decimate_mesh(mesh, max_error) -> tuple[TriangleMesh, float]
def unwrap_mesh_uvs(mesh, tex_size, parallel_partitions=16, min_faces_per_partition=1000) -> o3d.t.geometry.TriangleMesh
def bake_atlas_attributes(tm, tex_size) -> tuple[np.ndarray, np.ndarray]
def project_images_to_texture(tm, rgbs, c2w, K, tex_size, gutter_px=4) -> np.ndarray
def texture_mesh(mesh_path, out_dir, rgbs, c2w, K, *, voxel_size, decimate_max_error=0.25, tex_size=8192) -> Path
```
- `decimate_mesh`: meshoptimizer `simplify(..., options=SIMPLIFY_ERROR_ABSOLUTE)`. Kept over
  Open3D's `simplify_quadric_decimation` because Open3D's `maximum_error` is a quadric sum,
  not a distance (measured 2026-08-29: meshoptimizer abs-error 0.05 → p99 0.066 in scene
  units, 2.9 s on 3.05M faces). Returns the mesh and meshoptimizer's result error.
- `unwrap_mesh_uvs`: `compute_uvatlas(size=tex_size, parallel_partitions=p)` with
  `p = max(1, min(parallel_partitions, faces // min_faces_per_partition))`. Returns the tensor
  mesh carrying `texture_uvs`.
- `bake_atlas_attributes`: rasterizes the atlas with nvdiffrast and interpolates world position
  and normal off the same fragment buffer. Vertices are expanded one per triangle corner, since a
  vertex on a chart seam carries a different UV in each triangle sharing it. UV maps to clip space
  as `(2u - 1, 2(1 - v) - 1)` — nvdiffrast's first output row is the top of the atlas, and getting
  this wrong still fills the atlas, just with the wrong surface (measured 47.8 world units of
  position error unflipped, 2e-5 flipped). Texels no triangle covers keep a zero normal, which is
  what the projection kernel already rejects on.
  Replaces Open3D's `bake_vertex_attr_textures`, which is CPU: measured at 2048², 0.007 s versus
  1.21 s, agreeing to 1e-5 world units once Open3D's `margin` is set to 0. That margin defaults to
  2.0 and is a gutter, not geometry — it inflates Open3D's apparent coverage from 0.526 to 0.891
  by extrapolating positions outside every triangle. The gutter this pipeline ships is
  `_dilate_texels` after projection, so nothing is lost by baking the exact triangle coverage.
- `project_images_to_texture`:
  1. `bake_atlas_attributes(tm, tex_size)` gives per-texel world position and mesh normal; texels
     off every chart are invalid.
  2. `wp.Mesh(points, indices)` built once on the GPU. Accumulators `rgb_acc (S, S, 3)`,
     `w_acc (S, S)` float32: 1.1 GB at 8192².
  3. Per view: upload that view's image alone; launch one `@wp.kernel` over texels. Per
     texel: skip invalid; skip back-facing (`dot(n, cam - p) <= 0`); project with `w2c` and
     `K`, skip if behind the camera or outside the image; occlusion test
     `wp.mesh_query_ray(mesh, cam, dir, dist - eps)` — a hit before the texel means something
     is in front, skip; else bilinear-sample, weight by `cos(angle)`, accumulate.
  4. `albedo = rgb_acc / w_acc`; texels with `w_acc == 0` are filled by dilating from filled
     neighbours up to `gutter_px`, so bilinear lookup at chart borders does not bleed black.
     Returns `(S, S, 3)` uint8.
  5. Kernels are defined at module level in `texture.py` — Warp refuses kernels built from
     strings. `wp.init()` runs at first call, not at import.
  - Expected cost: 500 views × 67M texels at roughly 100 ms per view on the A40 → about a
    minute. Open3D has no comparable figure: it OOMs at two views, so there is no baseline to
    beat, only a capability it does not have. Wall clock on identical work measured 1.72 / 2.24 /
    13.61 s across three runs (Warp module load, BVH build, JIT cache state) — never quote a
    speedup from it.
- `texture_mesh`: read `mesh_path` → `decimate_mesh(max_error=decimate_max_error * voxel_size)`
  → `unwrap_mesh_uvs` → `project_images_to_texture` → `io.write_textured_ply`. Writes
  `out_dir/{mesh.ply, albedo.png}`. Returns the PLY path. Timings go to `logger.info`; no
  `texture_report.json`.
- The textured PLY carries vertex colours sampled from the atlas, so the vertex-colour
  consumers `bake_vertex_colors` served (collab-data `d059661`) read this file directly, at
  the decimated mesh's vertex density. Anyone wanting denser vertex colours has the fused
  `mesh.ply` one directory up. The subdivision pass goes.
- Dropped from the branch: `bake_vertex_colors` and `bake_target_faces`, `export_obj`,
  `project_normals` and the normal map, `smooth_iterations` (always 0), `max_views` (moot at
  Warp speed), `texture.source` (the texture pass consumes the views the mesh was fused from,
  so poses always match the mesh's frame), `uv_parallel_partitions` as a config knob
  (`unwrap_mesh_uvs` keeps it as a kwarg), `_project`, `_geometries`, `_to_color` (folded
  in), the report file.

### features.py — point features onto vertices

```python
def features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03) -> np.ndarray
def mesh_clustering(mesh, similarity_values, similarity_threshold=0.8, spatial_radius=0.03,
                    min_cluster_size=10) -> list[np.ndarray]
```
- `features2vertex`: body unchanged (cKDTree k-NN, vertices whose nearest point is beyond
  `sdf_trunc` get zeros, Gaussian-weighted `index_add_` on the GPU). Gains the three-part
  docstring; it has none today.
- `mesh_clustering`: same output, new adjacency. `valid = similarity_values > similarity_threshold`;
  `pairs = cKDTree(xyz[valid]).query_pairs(spatial_radius, output_type="ndarray")`;
  `csr_matrix` from the pairs; `connected_components`; clusters with at least
  `min_cluster_size` members, as index arrays into the original vertices. The dense
  `(n_valid, n_valid)` bool matrix, the Python loop and the tqdm bar go.

## Configuration

`configs/base.yaml`, 14 keys → 6:

```yaml
mesh:
  enabled: true
  source: feedforward      # feedforward | splats
  voxel_size: 0.0025       # TSDF voxel, world units; sdf_trunc = 4 × voxel_size
  depth_trunc: 1.5         # ignore depth beyond this, world units
  conf_percentile: 20      # drop depth below this confidence percentile (null = off); feedforward only
  texture: false           # decimate + UV atlas + project the fused views; writes mesh/texture/
```

| Key | Fate | Basis |
|---|---|---|
| `sdf_trunc` | derived, `4 × voxel_size` | both tuned configs on record are exactly 4× (0.01/0.0025 omega, 0.8/0.2 sfm) |
| `clean_repair` | always on | shipped default since 2026-08-27; callers wanting raw fusion call `fuse_tsdf` alone |
| `native_resolution` | always on in the reconstructor | shipped default; the model-res composition remains available to the dashboard and notebook |
| `splat_depth` | derived from the checkpoint's `primitive` | `median_depth` exists only for 2dgs and is the measured 2dgs lever (240k → 44.8k raw components). Deliberate default change: trunk ships `expected` for both primitives; 2dgs meshes now fuse median depth |
| `color_map_iterations`, `splat_max_depth_frac`, `splat_max_depth_grad`, `mesher` | deleted | findings 4, 5; `mesher` was already stale in the README |
| `texture.decimate_max_error`, `tex_size`, `uv_parallel_partitions`, `bake_target_faces`, `smooth_iterations`, `max_views`, `source` | `texture_mesh` keyword defaults (0.25 voxel, 8192) or dropped | never varied per scene |

`voxel_size` and `depth_trunc` remain absolute world-unit numbers — the one remaining
world-scale trap (omega 0.0025 vs sfm 0.2). A scale-relative voxel would be a behaviour change
needing measurement; listed under Follow-ups, not in scope.

## Call sites

### Reconstructor `_run_tsdf_mesh`

```python
recon = PointcloudResult.from_colmap(...)                     # pose + K authority, already loaded
if source == "feedforward":
    ff = FeedforwardResult.load_zarr(pointcloud_zarr, load_images=False, load_world_points=False)
    depths = ff.depth
    if conf_percentile is not None and ff.confidence is not None:
        depths[~confidence_mask(ff.confidence, conf_percentile)] = 0   # pointcloud.utils
    rgbs = read_frames(images_dir)                                    # preproc.frames
    depths = upsample_depths(depths, rgbs, ff.original_coords[:, :4])
    c2w, K = invert_poses(recon.extrinsics), recon.intrinsics
else:
    depths, rgbs, c2w, K = render_tsdf_inputs(splats_dir / "ckpt.pt", images_dir)
mesh_path = fuse_tsdf(depths, rgbs, c2w, K, out_dir, voxel_size, depth_trunc)
clean_repair_mesh(mesh_path)
if texture:
    texture_mesh(mesh_path, out_dir / "texture", rgbs, c2w, K, voxel_size=voxel_size)
```
- All imports at the top of `reconstructor.py`.
- Frame-count check between `recon` and the zarr stays for the feedforward source (they can
  come from different runs); the splats source carries its own cameras.
- `source: splats` looks for `splats/ckpt.pt` where it looked for `splats.zarr`.
- `tests/wrapper/test_reconstructor.py`: `test_base_yaml_mesh_has_fidelity_keys` asserts
  `conf_percentile is None`, `native_resolution is False`, `color_map_iterations == 0` — all
  three stale since the yaml moved on 2026-08-27; it is rewritten to assert the six-key block.
  `test_mesh_clean_repair_defaults_off`, `test_mesh_forwards_clean_repair_from_config` and
  `test_run_tsdf_mesh_passes_clean_repair_to_the_fusion` go with the key. `get_mesh_creator`
  patches become `fuse_tsdf` patches.

### Dashboard

- `pipeline.py`: in-memory result, model-res composition (`rgbs` from `result.images` as
  uint8 HWC, `K = result.intrinsics`), `fuse_tsdf`, `clean_repair_mesh` when its flag is set,
  then three lines for vertex features: read the PLY, `features2vertex`, `np.save(vertex_features.npy)`.
- `viewer.py`: the inline `features2vertex` import moves to the top.

### Notebook and evals

- `06_mesh/splats_mesh.ipynb`: import block becomes `from collab_splats.mesh import ...`
  plus `render_tsdf_inputs` / `upsample_depths` from `mesh.io`; `color_map_iterations=0`
  cells removed; `splats.zarr` → `ckpt.pt`; `FrameStore` → `read_frames`.
- `evals/scripts/analyze_splats.py`: the `color_map_iterations` key goes. The rest of that
  file is `clean/splats`' (its `analyze_normals` renders from the checkpoint).

## Dependencies

- `pyproject.toml`: remove `meshlib>=3.1` and `--ignore=tests/test_meshlib.py` (the file does
  not exist); add `warp-lang==1.14.0` (transitive today, `uv.lock` L10167),
  `meshoptimizer==0.2.30a0` (pinned on the texture branch; not in the venv) and `nvdiffrast`
  pinned to commit `253ac4fc`, used for the atlas rasterization only — it publishes no wheel and
  builds its CUDA extension at first use,
  so it needs a real `nvcc` on PATH. The pip `nvidia-cuda-nvcc-cu12` wheels are ptxas-only; the
  apt `cuda-nvcc-12-1` that `setup.sh` installs for gsplat is what satisfies this. `plyfile` and
  `trimesh` are already dependencies; `mesh/` uses plyfile only.
- Both new packages need installing into the shared venv; that is a user-approved step at
  execution time, and the pyproject/uv.lock change is one small commit at the end — that file
  is where the concurrent uv work collides.
- `docs/source/conf.py`: `meshlib` leaves the autodoc mock list; `warp`, `meshoptimizer` and `nvdiffrast` join it.

## Docs

- `docs/source/api/mesh.rst`: automodules `mesh.io`, `mesh.tsdf`, `mesh.clean`,
  `mesh.texture`, `mesh.features`.
- `configs/README.md` mesh table: the six keys above; `mesher`, `color_map_iterations`,
  `splat_max_*`, `native_resolution`, `clean_repair`, `splat_depth`, `sdf_trunc` rows go.
- `docs/mesh.md` (new, user-facing, mirrors the code tree like `docs/splats.md`): inputs,
  the five functions, the texture output layout.
- `CLAUDE.md` tree line: `mesh/  # TSDF fusion, cleaning, texturing, vertex features (io, tsdf, clean, texture, features)`.
- `docs/superpowers/CHANGELOG.md` entry; `docs/known-test-failures.md` mesh rows updated.

## Tests

`tests/mesh/` mirrors the package:

| File | Covers |
|---|---|
| `test_io.py` | `upsample_depths` placement + guards; `render_tsdf_inputs` with `load_checkpoint`/`render_views` mocked (2dgs → `median_depth`, alpha-0 → depth 0, `images_dir` swap); `write_textured_ply` header carries `s`/`t` and the `TextureFile` comment, trimesh reload has UVs + image, Open3D reload has the sampled vertex colours. Replaces `test_splats_adapter.py`. |
| `test_tsdf.py` | synthetic plane fused to a mesh; shape/dtype/principal-point guards raise; `sdf_trunc` default. |
| `test_clean.py` | two-component fixture: floater dropped, main kept, scale-relative (same mesh scaled 100× gives the same keep mask); holed sphere: hole filled, vertex colours preserved, oversize hole left open. |
| `test_texture.py` | one triangle + one synthetic image → texel colour matches; an occluder triangle blocks; back-facing view contributes nothing; gutter dilation fills the border; `decimate_mesh` reduces faces within `max_error`. |
| `test_features.py` | `features2vertex` weights, `sdf_trunc` zeroing; `mesh_clustering` equals brute-force clustering on a 200-vertex mesh, `min_cluster_size` honoured. |

`test_absent_confidence.py` (gate skipped when confidence is None) moves to `tests/wrapper`,
where the gate now lives. `test_feature_transfer.py` and `test_utils_ground_plane.py` are
deleted with their functions. Test functions stay flat.

## Verification

1. **Fusion parity.** On GH010229 (`/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229_undist_r7_500f`), feedforward source. Trunk run with `clean_repair: false, color_map_iterations: 0` (its other defaults: native on, `conf_percentile: 20`, `sdf_trunc: 0.01 = 4 × 0.0025`); new path stopped after `fuse_tsdf`, before `clean_repair_mesh`. Same vertex and triangle counts, vertex arrays match to 1e-6 — the integration loop is unchanged, so anything else is a bug.
2. **Cleaning parity.** `clean_repair_mesh` on the 7.1M-face mesh: components kept identical to trunk; holes filled within ±5% of trunk's count; wall-clock at or under meshlib's 69 s. Open3D `fill_holes` at TSDF hole counts is unmeasured (the probe was one sphere) — this gate is where it gets measured.
3. **Texture fidelity, absolute.** Not parity: Open3D cannot run this pipeline (CPU-only, OOM past two views), so it is reported and asserted on nowhere. The atlas is rendered back through one raycast of the same mesh and scored against the source images: PSNR ≥ 20 dB and ≥ 0.95 of visible pixels coloured (measured 23.03 dB @ 0.997 at 2048²). The bake swap is gated separately and directly — `bake_atlas_attributes` against Open3D's bake at `margin=0`, p99 position error < 1e-3 world units. Mutual-agreement scoring was tried and abandoned: the two implementations reject different texels, so their textures disagree at 11–16 dB while both score 23 dB against the photographs.
4. **Suite.** `tests/mesh`, `tests/wrapper`, `tests/dashboard` from the `clean/mesh` worktree with the printed `collab_splats.__file__` proof line; failure set diffed against the control run on the base commit, SKIP count included.

## Branch coordination

- `clean/preproc` already rewrote the native-resolution path in `mesh/utils.py` (58 lines) plus the notebook and `tests/mesh/*` for `images/` + `frames.json`. `clean/mesh` supersedes that hunk; it is the only other `clean/*` branch touching `mesh/`.
- `clean/splats` (spec `2026-09-05-splats-cleanup-design.md`, plan `2026-09-06-splats-cleanup.md`) retires `splats.zarr`. Its Task 17 edits `mesh/utils.py` and Task 18 edits the mesh notebook. Ask that branch to drop Task 17 and the mesh-notebook half of Task 18: `clean/mesh` owns `mesh/*` and `06_mesh/*`, and `render_tsdf_inputs` is written against the spec's `load_checkpoint` / `render_views` interface.
- Merge order: `clean/preproc` and `clean/splats` before `clean/mesh`. If executed in parallel, `render_tsdf_inputs` and its integration test are the only pieces that wait on `clean/splats` landing; everything else builds against the base.
- Execution starts only once `clean/mesh` is rebased onto the reorganised trunk with the other `clean/*` branches.

## Docstring format

Every function in `mesh/`:

```
"""
What it does, one line.

Args:
    name: what it is (shape, dtype, units).
Returns:
    what it is (shape, dtype).
"""
```

Rationale, measurements and history move to block comments at the code they explain, or to
the CHANGELOG entry. No paragraphs in docstrings.

## Follow-ups (not in scope)

- Scale-relative `voxel_size` (fraction of the camera-trajectory extent), so one default holds across omega-world and sfm-world scenes as the clean thresholds already do. Needs a measured comparison on both worlds.
- Dashboard display of the atlas (viser renders a textured trimesh, and trimesh loads the textured PLY; the dashboard shows vertex colours today).
