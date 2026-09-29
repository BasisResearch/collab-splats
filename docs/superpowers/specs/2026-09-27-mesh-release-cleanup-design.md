# Mesh release cleanup — design

Date: 2026-09-27 · Branch: `clean/mesh-release` off `clean/final` (`93cacd75`) · Status: layout approved 2026-09-27

Rules: [017-release-cleanup-rules.md](../decisions/017-release-cleanup-rules.md) — binding,
not restated here. Reference runs: [preproc](2026-09-24-preproc-release-cleanup-design.md),
[semantics](2026-09-24-semantics-release-cleanup-design.md),
[geometry](2026-09-25-geometry-release-cleanup-design.md).

## Goal

Make `collab_splats/mesh/` release-ready: brief docs, no dead code, tunables as kwargs,
one clear chain per output — arrays in, caller composes.

## Decisions (from brainstorm, user-approved 2026-09-27)

- **Bands are out.** `fuse_tsdf_bands`, `check_bands`, `CULL_VOXELS`, `depth_min` and the
  `mesh.bands` config key are deleted; one TSDF volume only.
- **Renames (verb form):** `fuse_tsdf` → `create_tsdf_mesh`, `_check_views` →
  `_validate_tsdf_inputs`, `texture_mesh` → `create_texture_mesh`,
  `bake_atlas_attributes` → `_rasterize_atlas`.
- **No module constants** — every tunable is a function kwarg.
- **No `simplify.py`.** `clean.py` gets two `########` sections: cleanup at full density,
  and preparation for UV unwrap.
- **`make_manifold`** splits bowtie vertices with a per-vertex scipy `connected_components`
  over the fan's shared-edge graph — no union-find helpers. Measured faces identical to
  the union-find version (0.67 s vs 2.38 s at 1.79M triangles).
- **meshlib** (`fillHoleNicely`) replaces the Open3D tensor hole fill used for texturing;
  declared in `pyproject.toml` (license approved).
- **Texture output is OBJ only:** `texture/mesh.obj + mesh.mtl + albedo.png`, written by
  trimesh with a white diffuse (`Kd 1 1 1`; trimesh defaults to `Kd 0.4`, which viewers
  multiply into the texture). trimesh's PLY export drops the texture — never used.
  `mesh/mesh.ply` stays the cleaned, untextured mesh.
- **Unseen fill always on;** `_dilate_texels`, `gutter_px` and `fill_unseen` deleted —
  gutter 4 and 0 produce bit-identical atlases once the fill runs (Phase 0).
- **xatlas unwrap deleted;** UVAtlas only.
- **Decimation default kept:** `decimate_max_error=0.25` × `voxel_size`.
- **`io.py` dissolved:** `upsample_depths` → `utils/image.py`; `render_tsdf_inputs` →
  `splats/rendering.py` (removes its inline import); `_to_numpy` deleted (tensors are
  always torch at the new site); textured PLY/OBJ writers replaced by trimesh export.
- **`features.py` layout unchanged** (annotations and docstrings only).
- **Isolation:** worktree `.worktrees/mesh-release`, `third_party/*` symlinked; every gate
  runs as `cd <wt> && PYTHONPATH=<wt> python ...` and prints `collab_splats.__file__`.

## Texture chain order (Phase 0)

Measured on GH010229 v018 (`mesh_clean.ply`, 1.79M triangles), reprojection PSNR/SSIM
over 86 held views; scripts in `/workspace/scratch/mesh-release/ab_order/`.

| arm | order | faces | unseen | PSNR | SSIM |
|---|---|---|---|---|---|
| A | v49 chain: fill at 230k → decimate → clean | 64,884 | 8.4% | 16.70 | 0.525 |
| B | decimate → manifold → fill → unwrap | UVAtlas `0x80004005` (fill leaves fold-over faces) | | | |
| B45r | B + second make_manifold | 75,003 | 8.0% | 16.63 | 0.517 |
| Cdef | C: fill (full density) → decimate (0.25 × voxel) → manifold | 224,018 | 9.3% | 16.73 | 0.525 |
| C65 | C with 65k target | 61,921 | 9.8% | 16.64 | 0.524 |

- order C runs every step once; its occluder is the cleaned, unfilled input mesh
- the order is decided with the user AFTER every other cleanup part lands (user, 2026-09-27);
  until then `create_texture_mesh` keeps today's order (decimate → manifold → unwrap → project)

## Target layout

- `tsdf.py` — `create_tsdf_mesh(depths, rgbs, c2w, K, out_dir, *, voxel_size, depth_trunc,
  sdf_trunc=None)`; `_validate_tsdf_inputs`. Integration and write inlined.
- `clean.py`
  - `######## Cleanup (full density)`: `get_scene_scale`, `remove_floaters`, `fill_holes`
    (meshlib, conversion inline), `clean_repair_mesh`
  - `######## Prepare for UV unwrap`: `decimate_mesh(mesh, *, max_error)`, `make_manifold`
- `texture.py` — `unwrap_mesh_uvs`, `project_images_to_texture` (occluder is an Open3D
  mesh), `create_texture_mesh`; private `_rasterize_atlas`, one `wp.struct` Camera,
  `_view_footprint` returning footprint + (u, v), `_bilinear`, the two warp kernels,
  `_fill_unseen`.
- `features.py` — unchanged API.

## Round 1 — prose only

No code change. Proof: AST equal after deleting every docstring statement on both sides
and stripping comments, plus one sanity mutation showing the check can fail.

- `__init__.py` — module docstring names what each module holds.
- `clean.py:116` — Open3D view-lifetime comment loses "measured"; 3-line run to header +
  bullets.
- `features.py` — docstrings to contract (types move to signatures in Round 2).
- `io.py:23` — 8-line `DEPTH_SOURCES` comment loses the GH010229 measurement (moves here:
  median 3.02M vertices / 94,937 components vs expected 3.95M / 426,695); `:304` weld
  comment to header + bullets.
- `texture.py:299` — drop "measured"; `:426`, `:543` prose pairs to header + bullets;
  `colour-sum` → `color-sum`.
- `tsdf.py` — `fuse_tsdf` docstring to contract; prose pairs at `:23`, `:88`, `:97`,
  `:149` are all in bands code and go with it in Round 2.

## Round 2 — code, one commit per logical change

1. **Annotations** on every public def in the package (110 contract hits today).
2. **tsdf:** delete bands (`fuse_tsdf_bands`, `check_bands`, `CULL_VOXELS`, `depth_min`),
   config key `mesh.bands`, reconstructor branch + validation, tests; inline
   `_integrate_views` and `_write_mesh`; rename `_check_views` → `_validate_tsdf_inputs`.
3. **tsdf rename** `fuse_tsdf` → `create_tsdf_mesh` (identifier-only commit).
4. **io dissolve:** move `upsample_depths` (+ `_box`, `_guided_filter`,
   `_guided_upsample_depth`) to `utils/image.py`; move `render_tsdf_inputs` to
   `splats/rendering.py` with `DEPTH_SOURCES` as an inline check; delete `_to_numpy`;
   RGB uint8 via `to_uint8_hwc`.
5. **clean:** `fill_holes` → meshlib `fillHoleNicely` gated by perimeter fraction; add the
   unwrap-prep section (`decimate_mesh`, `make_manifold` with the cc bowtie split).
6. **texture:** delete xatlas, `_dilate_texels`, `gutter_px`, `fill_unseen`; occluder
   takes an Open3D mesh; one Camera struct; rename `bake_atlas_attributes` →
   `_rasterize_atlas`; final uint8 via `to_uint8_hwc` (±1 level, approved).
7. **create_texture_mesh:** trimesh OBJ writer with white Kd; delete
   `write_textured_ply` / `write_textured_obj`. Chain order: separate step after discussion.
8. **meshlib** declared in `pyproject.toml`.

## Caller sweep

Checked with `git grep` over `collab_splats configs scripts tests evals docs/source`.

| changed | outside callers | effect |
|---|---|---|
| `fuse_tsdf` → `create_tsdf_mesh` | `wrapper/reconstructor.py`, `dashboard/pipeline.py`, `evals/scripts/analyze_splats.py`, tests (`tests/mesh`, `tests/wrapper` patch strings, `tests/dashboard`, `tests/integration`) | updated |
| bands delete | `reconstructor.py` (:86 branch, :931 validation), `configs/base.yaml`, `configs/README.md`, `tests/wrapper` | `mesh.bands` key removed |
| `texture_mesh` → `create_texture_mesh` | `reconstructor.py:730`, `tests/wrapper/test_reconstructor.py:804` | `texture/` holds OBJ + MTL + PNG, no PLY |
| `render_tsdf_inputs`, `upsample_depths` move | `reconstructor.py:39` | import path only |

## Tutorial impact (not edited here)

- `06_mesh/splats_mesh.ipynb`, `02_pointcloud/feedforward_mesh.ipynb` call `fuse_tsdf`
  → `create_tsdf_mesh`.

## Testing

- **Baseline** on the untouched worktree before any edit: `tests/mesh`, `tests/wrapper`,
  `tests/test_docstring_contract.py`, one package per pytest call; plus `tests/utils`,
  `tests/splats`, `tests/dashboard` for the moves. Pass condition: branch failures ⊆
  control failures.
- **Gate per commit:** same set, `__file__` proof, no `| tail`, no `--tb=no`.
- **Texture parity:** after the order decision, `create_texture_mesh` on the v018 inputs
  reproduces the chosen Phase 0 arm's PSNR within 0.05 dB (`reproj_eval.py`).
- **End:** add `mesh` to `PACKAGES` and `RELEASED`; CLAUDE.md enforced-packages line and
  mesh/ architecture line updated; `docs/source/api/mesh.rst` updated.

## Out of scope

- Any notebook; TSDF fusion quality; watertight sealing; multiview depth filtering.
