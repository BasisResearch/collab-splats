# clean_repair scalability — design

**Date:** 2026-08-18
**Status:** validated by prototype on the failing mesh (see Measured viability)
**Scope:** `clean_repair_mesh` in `collab_splats/mesh/utils.py` — internals only. Signature, call sites (`Open3DTSDFFusion.create`, dashboard), config surface (`mesh.clean_repair` boolean + three tuning fields on the creator) all unchanged.

## Problem

`mesh.clean_repair: true` gets the pipeline SIGKILLed (cgroup OOM, 46.6 GB cap) on meshes
produced by the current base.yaml TSDF parameters (`voxel_size: 0.0025`, `sdf_trunc: 0.01`,
`depth_trunc: 1.5`, fps 2.0). Measured on the killed run's mesh
(`2026_07_15-Goprosplat-GH010229`, 4.81 M verts / 7.12 M faces):

1. **`mm.getAllComponents(mesh)` is O(n_components × n_faces) memory.** meshlib returns one
   dense `FaceBitSet` (n_faces bits) per component. The mesh has **240,371 components**
   (TSDF speckle) → 240,371 × 7,121,592 bits ≈ **214 GB**. This is the allocation the kernel
   kills. Old params (voxel 0.005, sdf_trunc 0.02) produced ~4× fewer faces and far fewer
   speckle components, which is why it used to survive — the cost is roughly quadratic in
   resolution, and the new params crossed the cliff.
2. **The bounds loop constructs a full `mm.Mesh()` per component** — 240 k mesh
   constructions; hours even if memory were fixed.
3. **Hole filling is a per-hole Python loop** (fill → subdivide → smooth per hole). The mesh
   has **176,162 holes**; the v1 prototype did not finish 176 k iterations in 10 min.
4. **Latent bug found during investigation:** the meshlib `loadMesh`/`saveMesh` round-trip
   **strips vertex colors** (verified: output PLY header has only x/y/z). Every mesh
   `clean_repair` has ever rewritten lost its TSDF vertex colors.

So: yes, code functionality — three O(n_components)/O(n_holes) scaling defects plus a color
loss bug — triggered by parameters that multiplied both counts.

## Design

Rewrite `clean_repair_mesh` internals on open3d (already the module's primary dep, preserves
colors natively, C++ clustering) with meshlib kept **only** for hole filling, driven through
`mrmeshnumpy` instead of file I/O. Six steps:

1. **Load with open3d** (`read_triangle_mesh`) — vertex colors come along.
2. **Components:** `mesh.cluster_connected_triangles()` → per-triangle cluster id +
   cluster sizes. Native, edge-connectivity (same incidence as the old
   `getAllComponents(... PerEdge)`), one int per face instead of a bitset per component.
3. **Bounds filter, vectorized:** per-component AABBs via `np.minimum.at`/`np.maximum.at`
   over per-face AABBs (one pass, no per-component meshes). Keep the largest component plus
   every component whose AABB lies inside the largest's AABB — identical rule to today.
   `use_largest=True` keeps only the largest, as today. Then
   `remove_triangles_by_mask` + `remove_unreferenced_vertices` (colors follow automatically).
4. **Hole filling, batched:** convert with `mrmeshnumpy.meshFromFacesVerts` (no disk
   round-trip). `findHoleRepresentiveEdges` → filter by `holePerimeter < max_hole_size`
   (Python loop is fine: 176 k perimeter calls ≈ 1.7 s) → **one** `mm.fillHoles(mesh, edges,
   params)` call collecting all patch faces in a single `outNewFaces` bitset → **one**
   `subdivideMesh` over that region (`maxEdgeLen = averageEdgeLength()`) → **one**
   `positionVertsSmoothly` over all new verts.
5. **Colors back:** `mmesh.pack()`, export verts/faces via `mrmeshnumpy`. Original vertices
   keep their indices (verified: vertex prefix is byte-stable through fill/subdivide/pack),
   so original colors are copied through directly; patch vertices get nearest-neighbour
   colors via `cKDTree` (already imported in utils.py). Fallback if the prefix check ever
   fails: full NN transfer. Skipped entirely when the input mesh has no vertex colors
   (the test fixtures don't).
6. **Save with open3d** to the same path — in-place rewrite, colors in the PLY.

### Semantics changes (deliberate, documented in the docstring)

- **`max_edge_splits` becomes a global subdivision budget** (was per-hole — meaningless
  under batch fill). Default raised `10_000 → 1_000_000`, in both `clean_repair_mesh` and
  the `Open3DTSDFFusion.clean_max_edge_splits` field (the one value-level change outside
  utils.py). Its job is unchanged: stop
  subdivision from exploding the triangle count; `maxEdgeLen` remains the real control.
  The prototype hit a 1 M ceiling exactly, so the ceiling is live at this scale.
- **Vertex colors are now preserved** (previously silently stripped — strictly a fix).
- Everything else — in-place rewrite, keep-inside-bounds rule, `use_largest`, large holes
  left open — behaves exactly as the existing tests assert.

### Dependencies

None new: open3d, numpy, scipy `cKDTree`, meshlib all already imported by `mesh/utils.py`.
`tqdm` stays for the perimeter scan.

## Measured viability (prototype, scratchpad `proto_clean_v2.py`, on the killed mesh)

| Step | Time | Peak RSS |
|---|---|---|
| load (colors) | 2 s | 1.0 GB |
| cluster + AABB filter (240,371 → 190,588 comps kept) | 30 s | 3.9 GB |
| perimeter scan (176,162 holes, 176,141 small) | 2 s | 3.9 GB |
| batch `fillHoles` (+1,021,780 faces) | 18 s | 3.9 GB |
| subdivide + smooth (+1,000,000 verts, ceiling hit) | 11 s | 4.5 GB |
| color reattach + save | 4 s | 4.5 GB |
| **total** | **66 s** | **4.5 GB** |

Old implementation on the same mesh: ~214 GB required at step 2 → SIGKILL at 46.6 GB.
Output: 4.91 M verts / 8.94 M tris, PLY header carries `uchar red/green/blue`,
original-vertex prefix byte-stable (`allclose` at 1e-5).

## Testing

- Existing three `clean_repair_mesh` tests in `tests/mesh/test_utils.py` pass unchanged —
  they are the behavior contract (drop out-of-bounds specks, keep in-bounds detached
  geometry, fill small holes, leave large holes, `use_largest`).
- New test: colored input mesh → colors present in the rewritten PLY, patch verts colored
  (guards the latent color-strip bug against regression).
- New test: colorless input mesh → no crash (guards the reattach branch).
- No new test data: extend `_holed_sphere_with_strays` with `paint_uniform_color`.

## Implementation principles

- Reuse: o3d + cKDTree already in the module; delete the per-component and per-hole loops
  and the meshlib load/save path they served.
- No new config, no new filenames, no compat shims — internals swap only.
- `getAllComponents`-style meshlib APIs must not reappear: any per-component data structure
  has to be O(n_faces) total, not O(n_components × n_faces).
