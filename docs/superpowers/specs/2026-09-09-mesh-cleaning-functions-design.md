# Mesh cleaning functions — design

**Date:** 2026-09-09
**Status:** approved, not yet implemented
**Branch:** `clean/final`

## Problem

The GH010229 texture bake is produced by a scratch chain that the library cannot express. The
primitives exist in `collab_splats/mesh/`, but the recipe does not, and one shipped primitive is a
measured regression:

- `clean.fill_holes` wraps Open3D `fill_holes`. On a splat TSDF it closes 38% of the boundary and
  splits 387 components into 1,405. The scratch fork gates on hole **perimeter** and fills with
  meshlib instead.
- Decimation is reachable only as an error bound. The texture stage needs a triangle **count**.
- Nothing in the library bridges a snaking boundary loop, so a ground hole that has merged into the
  outer rim cannot be closed by any per-hole gate.
- `_make_manifold` is private to `texture.py`, but the chain needs it as its own step.

## Scope

**In:** four functions in `collab_splats/mesh/clean.py`. Pure library work — unit-testable, no
config schema changes, no `Reconstructor` changes.

**Out, tracked separately:** the memmap rewrite of `render_tsdf_inputs` (RAM-only `np.stack` over
853 views is ~12 GB and will not survive the 46.6 GB cap), lever passthrough on `texture_mesh`
(`occluder`, `view_ratio`, `fill_unseen`, `max_stretch`, unwrap backend), staged artifacts and
`MeshConfig` fields in `_run_tsdf_mesh`, and wiring `write_textured_obj`. Runs stay driven by
scratch scripts until those land.

## Module layout

`clean.py` becomes the whole *fused mesh → solid decimated surface* module. `decimate_mesh` and
`make_manifold` move in from `texture.py` and are re-exported there, so `texture_mesh` and
`tests/mesh/test_texture.py` are untouched.

A separate `repair.py` was considered and rejected: it splits a chain whose steps only make sense in
sequence, and `decimate_mesh` fits neither module name. The result is ~330 lines.

`clean.py`'s module docstring is rewritten — "Floater removal and hole filling" no longer describes it.

## Functions

### `fill_holes(mesh, max_hole_frac=0.014, max_edge_splits=20_000)`

Replaces the Open3D implementation.

- meshlib `fillHoleNicely` per hole, gated on **perimeter** `< max_hole_frac * scene_scale`
- `smoothCurvature=True`, `subdivideSettings.maxEdgeLen = mesh.averageEdgeLength()`,
  `smoothSettings.edgeWeights = EdgeWeights.Cotan` — a patch follows the surface instead of capping
  it flat
- `pack()` before `getNumpyFaces`, so meshlib's deleted face slots never leave as zero rows
- vertex colours carried: original indices are prefix-stable through fill/subdivide/pack, so they
  copy straight across; only patch vertices need the `cKDTree` nearest-neighbour lookup

**Breaking, approved.** `max_hole_frac` keeps its name but changes meaning (Open3D's diameter-like
`hole_size` → perimeter) and its default moves 0.0045 → 0.014. Every `clean_repair_mesh` caller
changes output, including `Reconstructor._run_tsdf_mesh`. That is the point — the current behaviour
is the regression above.

The `from_legacy` binding-hazard comment is deleted along with the code it warns about: with Open3D's
tensor `fill_holes` gone, the use-after-free it documents cannot occur.

### `decimate_mesh(mesh, max_error=None, target_triangles=None)`

Moved from `texture.py`, one mode added. Exactly one argument required; `ValueError` otherwise.

Both modes are the same `meshoptimizer.simplify` call with a different binding constraint:

| mode | `target_index_count` | `target_error` |
|---|---|---|
| `max_error` (today) | `3` | `max_error`, `SIMPLIFY_ERROR_ABSOLUTE` |
| `target_triangles` | `3 * target_triangles` | `inf` |

Returns `(mesh, result_error)` as now. Vertices are removed, never moved.

### `make_manifold(mesh)`

`_make_manifold` promoted public. Behaviour unchanged, plus one addition: drop all-zero face rows, so
it is safe on anything the meshlib functions hand back.

Its docstring gains the ordering rule: **it removes faces, so it opens boundary.** Run it before
fill and bridge, never after.

### `bridge_holes(mesh, max_hole_frac=3.9, neck_mult=2.0, ndot=0.3, loop_sep=20, max_rounds=6)`

New. Welding two boundary vertices pinches the surface and meshlib un-pinches it, so this bridges
instead: two triangles across a short neck split one loop into two and stay manifold.

Per round:

1. Walk missing half-edges into boundary loops.
2. `cKDTree` pairs of boundary vertices within `neck_mult *` the median boundary edge length.
3. Reject a pair on the same loop within `loop_sep` steps along the rim — otherwise the "neck" is
   just the next step along the same boundary.
4. Reject a pair whose area-weighted vertex normals dot `<= ndot` — otherwise it bridges two
   surfaces that face away from each other.
5. Shortest necks first; each quad's four vertices are used at most once.
6. Emit `(p, gp, q)` and `(q, gq, p)`, then call `fill_holes(max_hole_frac)`.

Stops when a round bridges zero. On mesh_v018 it converged at round 2: 1,410 bridges cut 2,031 holes
down to 409, all of which filled, and round 2 found no neck left to cut.

**Its gate is loose on purpose, and that is why it is a separate default from `fill_holes`.** By this
point the mesh is decimated and every ordinary hole is meant to close; what remains is one snaking
rim that no per-hole gate can select. On mesh_v018 the gate admitted 2,031 of 2,032 loops — every
loop except the 7,088 m snake. 3.9 is a perimeter of roughly four scene diagonals: large-sounding,
but a rim that wanders the whole site is genuinely longer than the site is wide.

## Invariants

- every threshold is a fraction of scene scale, never a world distance
- meshes in, meshes out; no file IO outside `clean_repair_mesh`
- vertex colours survive every meshlib round-trip
- `pack()` before every `getNumpyFaces`

`clean_repair_mesh` keeps its signature and structure; only its `fill_holes` call and that one
default change.

## Recipe ordering

The chain these functions compose, as measured on mesh_v018:

| step | call | tris |
|---|---|---|
| fuse | `fuse_tsdf` | 2,753,909 |
| floaters + fill | `clean_repair_mesh` | 1,791,014 |
| decimate | `decimate_mesh(max_error=0.0425)` then `make_manifold` | 241,162 |
| gated fill | `fill_holes(max_hole_frac=3.9)` | 372,503 |
| bridge | `bridge_holes()` | 469,829 |

`make_manifold` runs inside the decimate step, before any fill, because it removes faces and so
reopens boundary. 241,162 is the count after it.

**The two fills are not the same call.** `clean_repair_mesh` fills at `0.014 * scale` (2.54 m) on the
full-density mesh, closing only small holes. The post-decimation fill runs at ~3.9 (700 m), closing
everything except the one snake. Running the loose gate at full density is what exhausts the memory
cap — the curvature solve scales with mesh density, and a multi-kilometre rim in a 1.77M-triangle
mesh is too large a system. The tight gate at full density is fine, as the measurement below shows.
Decimate before widening the gate.

## Measurements

Per-hole `fillHoleNicely` on the raw fused mesh (`mesh_v018/mesh_raw.ply`, 2,004,884 verts /
2,753,909 tris; 1,674,848 tris after floaters, scene scale 181.414): **19,217 of 20,734 holes under a
2.54 m perimeter gate, ~175 s total.** Per-hole cost drifts mildly with fill count (7.8 ms over the
first 1,000, 9.1 ms by 17,500) but stays bounded — the run is not quadratic.

This settles the one open question in the design. `clean823` had noted that a per-hole
fill/subdivide/smooth loop "does not finish at TSDF hole counts" and used a batched `fillHoles` plus
one global subdivide instead. That does not hold for `fillHoleNicely` at this scale, so `fill_holes`
has **one code path** and no batched fallback.

## Testing — `tests/mesh/test_clean.py`

**fill_holes**
- a punched plane closes
- an oversized hole is left alone
- vertex colours are present on the result and unchanged on the original vertices
- no all-zero face rows in the output

**decimate_mesh**
- the three existing `max_error` tests move over unchanged
- `target_triangles` hits the requested count within tolerance
- passing both arguments raises; passing neither raises

**bridge_holes**
- a figure-eight boundary — two holes joined by a thin neck — is split and closed
- a mesh with no boundary is a no-op
- two rims facing opposite ways are not bridged (the `ndot` guard gets its own case)
- returns at `max_rounds` without spinning

**make_manifold**
- all-zero face rows are dropped
- existing bowtie and duplicate-face cases still pass

Each guard gets its own case. A test that would pass with the guard deleted is not a test of it.
