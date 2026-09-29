# Mesh convex hull cleanup — design

Date: 2026-09-28 · Branch: `clean/mesh-release` · Status: design approved in brainstorm 2026-09-28, spec awaiting review

Parent: [mesh release cleanup](2026-09-27-mesh-release-cleanup-design.md) — its rules bind here
(every tunable a kwarg, no module constants, `clean.py` keeps its two sections).

## Goal

A built-in cleaning step that trims the ragged outer edge of a fused mesh and joins a
manifold ground patch out to a rounded convex hull, so the textured mesh has no holes, no
strands and a smooth outline. Opt-in; camera-free; the mesh is the only input.

## Decisions (user-approved 2026-09-28)

- **Lives in `mesh/clean.py`**, Cleanup section — no new module.
- **Public name `make_convex_hull(mesh)`**; `clean_repair_mesh(..., use_convex_hull=False)`
  calls it between `remove_floaters` and `fill_holes`.
- **No cameras, no footprint.** Up direction and grid size come from the mesh itself.
- **`fit_dominant_plane` moves to geometry**, unchanged, and supplies the ground plane.
- **Dropped after ablation:** footprint-based coarse trim, e3 sheet-edge smoothing.
- **Kept after ablation:** neck bridges.

## Ablation (GH010229, scratch `hull_join4.py`, metrics after `fill_holes(3.9)`)

| arm | interior holes | max edge | tall faces | outline xy / hull |
|---|---|---|---|---|
| v3 (cameras + footprint + e3 + bridges) | 2 | 2.75 | 3,972 | 561 / 442 (1.27) |
| v3, no bridges | 2 | 2.75 | 3,984 | 655 / 442 (1.48) |
| geometry only | 2 | 2.67 | 2,877 | 813 / 444 (1.83) |
| geometry + e3 | 2 | 3.19 | 3,069 | 945 / 444 (2.13) |
| footprint, no e3 | 2 | 2.48 | 2,583 | 779 / 444 (1.75) |
| **geometry + bridges (chosen)** | **2** | **2.67** | **2,934** | **590 / 444 (1.33)** |

- bridges are the one clear lever (−28% outline); footprint −4%, e3 negative
- unseeded RANSAC shifted the frame between runs; ~10% outline differences may be noise
- the textured v3 mesh reached 0 interior holes (post-decimation lids close the last 2)

## Code

### A. `fit_dominant_plane` → `collab_splats/geometry/transforms.py`

- moved verbatim from `pointcloud/utils.py` (production callers: none; tests only)
- sits beside `rotation_align_vectors`; exported from `collab_splats.geometry`
- deleted from `pointcloud/utils.py`, no re-export
- tests move from `tests/pointcloud/test_pointcloud_utils.py` to `tests/geometry/`

### B. `make_convex_hull` in `collab_splats/mesh/clean.py`

```python
def make_convex_hull(
    mesh: o3d.geometry.TriangleMesh,
    *,
    hull_round: int = 20,
    outline_open: int = 8,
    rim_max_dz: float = 3.0,
    rim_max_edge: float = 3.0,
    bridge_radius: float = 3.0,
    min_piece_faces: int = 1000,
    min_up_agreement: float = 0.3,
) -> o3d.geometry.TriangleMesh:
```

- every length kwarg is in grid cells; cell = median mesh edge length (≈ one TSDF voxel), so all are scene-relative
- returns a new mesh; the input is not modified

Steps, one private helper each where the block exceeds a few lines:

1. **Frame** (`_find_ground_plane`). `res` = median edge length. Plane from `fit_dominant_plane` under a fixed
   Open3D RANSAC seed; up = plane normal, flipped to agree with the area-weighted mean face
   normal (TSDF normals face free space). Agreement = summed face normals · plane normal over
   summed face areas; `|agreement| < min_up_agreement` raises `ValueError` — no dominant ground.
2. **Hull cut** (`_create_rounded_hull_mask`). Raster the mesh top-down at `res`; close, keep the largest
   blob, open, convex hull, round by `hull_round` cells. Faces outside are removed; pieces
   under `min_piece_faces` dropped.
3. **Outline trim** (public `trim_mesh_edges`, its own frame and cell). Mesh outline opened by `outline_open` cells, holes filled; faces outside
   it that are edge-connected to the outer rim are removed (interior patches stay).
4. **Ground patch** (`_connect_mesh_hull`, with step 5).
   - ground prior: per-cell min height, eroded, empty cells filled by multi-scale normalized Gaussian, blurred
   - rim vertices: open edges bordering the gap, within `rim_max_dz` cells of their neighbours' median height
   - Delaunay over rim + grid points in the gap, xy edges under 3 cells
   - heights and colors: harmonic solve, rim pinned, pull to the prior growing with distance from the mesh
   - rim triangles with a 3D edge over `rim_max_edge` cells are dropped
5. **Manifold join** (end of `_connect_mesh_hull`). Flip sheet faces that repeat a mesh directed edge;
   drop sheet faces on any edge with > 2 faces or a repeated directed edge until none remain.
6. **Pieces + repair.** Pieces under `min_piece_faces` dropped; `make_manifold`.
7. **Neck bridges** (public `bridge_mesh_edges`), last, on the repaired mesh. meshlib `makeBridge` between
   outer-loop vertices within `bridge_radius` cells in 3D and > `min_loop_edges` (40) loop edges apart, widest
   separation first; turns inlets into interior holes that `fill_holes` closes.

### C. `clean_repair_mesh`

```python
def clean_repair_mesh(mesh_path, ..., use_convex_hull: bool = False) -> Path:
    remove_floaters(mesh, ...)
    if use_convex_hull:
        mesh = make_convex_hull(mesh)
    mesh = fill_holes(mesh, ...)
```

- config key `mesh.use_convex_hull` (default false) passed through by the Reconstructor mesh stage
- the height-field assumption means indoor / object scenes should leave it off

### D. Docs

- `clean.py` module docstring: one bullet for `make_convex_hull` (≤ 6 bullets total)
- `docs/mesh.md`: the flag, what it does, when to leave it off

## Tests (`tests/mesh/test_clean.py`, flat functions, synthetic scene)

Fixture: ground plane with a box on it and a jagged, partly slit outer edge, TSDF-like
edge length, normals up.

- no interior holes after `make_convex_hull` then `fill_holes`
- zero non-manifold edges; one main piece
- outer loop xy length shorter than the input's
- no sheet face with a 3D edge over `rim_max_edge` cells touching the rim (rim heights jittered, bridges off)
- a vertical-wall-only mesh raises `ValueError`
- the up flip: the same scene rotated upside down gives the same result, rotated
- input mesh unchanged
- `clean_repair_mesh(use_convex_hull=False)` output byte-identical to before
- geometry: moved `fit_dominant_plane` tests pass from `tests/geometry/`

## Out of scope

- camera footprint trim, e3 edge smoothing (measured, dropped)
- watertight sealing (never; see mesh memory)
- auto-detecting whether a scene is ground-dominated beyond the `ValueError` guard

## Commits (shown for approval before committing)

1. `refactor(geometry): move fit_dominant_plane to geometry.transforms`
2. `feat(mesh): make_convex_hull cleanup behind use_convex_hull`
3. `docs(mesh): convex hull cleanup`
