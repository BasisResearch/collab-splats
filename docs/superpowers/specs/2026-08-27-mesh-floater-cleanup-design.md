# Mesh floater cleanup — design

Date: 2026-08-27
Status: implemented

## Problem

Meshes fused from the scaffold splat runs carry heavy junk: components floating off the
ground plane, speckle across the surface, and geometry from blown-out depth. Reported
against `splats_scaffold2dgs_gssr_v2` (GH010229_undist_r7) and `splats_scaffold3dgs_gssr_v2`
(GH010229_undist).

## Measured

Both meshes, and their `splats.zarr` renders in `/tmp/scaffold_mesh/{2dgs,3dgs}_gssr_v2/`.

| | 2dgs v2 | 3dgs v2 |
| --- | --- | --- |
| faces | 4,966,913 | 4,576,621 |
| connected components | 240,461 | 211,184 |
| largest component | 50.1% of faces | 49.7% |
| components < 10 faces | 72.1% | 71.4% |
| faces in components < 500 | 44.3% | 44.0% |
| kept by the current AABB rule | **240,440** | **211,150** |

Four independent findings:

1. **The component selector is a no-op.** `clean_repair_mesh` keeps any component whose AABB
   sits inside the largest component's AABB (`mesh/utils.py:268-276`). The largest component
   is the scene, so its AABB is the scene AABB — every interior speckle passes. It removes 21
   and 34 components respectively.
2. **No size or area floor exists at all.** Half the mesh is in components the rule never
   examines.
3. **The alpha gate in `_splats_to_tsdf_inputs` filters nothing.** Measured alpha: min 0.088,
   median 0.998, fraction below 0.5 = 0.01%. `keep = alpha > 0` admits every pixel, and
   `mesh.conf_percentile: 20` cuts at alpha ~= 0.978 — an arbitrary slice through good pixels,
   not a confidence filter.
4. **Rendered depth is unbounded.** Cameras span 141 / 134 units; depth runs p50 16, p99 195,
   max 497. Beyond 80 units from a camera, 76-81% of faces sit in tiny components.

Component isolation separates junk cleanly. `gap` (component centroid to the nearest point on
the largest component) has median 5.79 / 6.35 units against a scene scale of 210 / 193.

Rejected on measurement, recorded so they are not re-proposed:

- **Multi-view depth consistency** — already built, measured and reverted (decision 016,
  `a7a2e44e`). The reason applies harder on the splats source: a floater is genuine 3D geometry
  the model built, so every view renders it at the same depth. Cross-view agreement can only
  catch view-dependent depth error.
- **SfM sparse-point support** — separates junk from real (small-component faces sit at median
  6.09 from the nearest SfM point vs 1.65 for main-component faces), but 104k points over a
  200-unit scene is too sparse for a per-face gate: radius 2.0 would delete 45% of the main
  component.
- **Cross-source gate against `pointcloud.zarr` VDA depth** — viable in principle (median
  disagreement 0.096 / 0.087) but per-frame median disagreement swings 0.04 to 0.41, and
  `depth_scales` in the zarr attrs jumps to 9-12 on frames 165-175 against ~2.5-4 either side.
  VDA's own alignment is unstable, so any tight threshold deletes good geometry. Deferred.
- **Grazing-angle mask** — 23% of pixels at `|cos(n, ray)| < 0.2`. Too blunt; oblique ground
  and walls are real surfaces here.

## Change

`clean_repair_mesh` in `collab_splats/mesh/utils.py`. Every threshold is a fraction of a
scene scale derived from the mesh, so no default carries a world-scale assumption — the class
of bug that has already produced the metric `mesh.depth_trunc: 1.5` on a scene running ~78x
that scale.

```
scene_scale = p1-p99 AABB diagonal of the largest component's vertices
keep = area >= min_area_frac * scene_scale**2  AND  gap <= max_gap_frac * scene_scale
```

- `gap` = component centroid to the nearest point on the largest component, via a KD-tree over
  a stride-subsampled 200k of its vertices; the stride is derived from the vertex count.
- p1-p99 rather than the raw AABB: far-field tendrils inflate the raw diagonal to 265 / 225
  against a true 210 / 193.
- The largest component is always kept. The hole-fill tail is unchanged.
- `use_largest` is deleted, along with `Open3DTSDFFusion.clean_use_largest`. It is a boolean
  mode that duplicates what the two thresholds already express — a large `min_area_frac` keeps
  only the main component — and no caller sets it. The thresholds are the interface.
- The selector moves into `_select_components(mesh, ...) -> bool[n_components]` so it is
  testable alone and a second selector can be added later without touching the fill path.

Defaults, chosen from the sweep and consistent across both scenes:

| param | default | on these scenes |
| --- | --- | --- |
| `min_area_frac` | `6e-6` | 0.27 / 0.22 u^2 |
| `max_gap_frac` | `0.01` | 2.1 / 1.9 u |

Result: 2,661 / 3,006 components kept, 54.1% / 53.8% of faces.

`max_hole_size` becomes `max_hole_frac`, applied as `max_hole_frac * scene_scale`. It is the
same world-scale hardcode in the same function: a 3.0 perimeter means something different in a
210-unit scene than in a metric one. `0.014` reproduces today's behaviour at this scale.
`Open3DTSDFFusion.clean_max_hole_size` renames to `clean_max_hole_frac` with the same default.
No caller sets any of the `clean_*` fields today — `Reconstructor` passes only `clean_repair` —
so the rename and the `use_largest` deletion are contained to `mesh/utils.py`, `mesh/tsdf.py`
and the existing tests.

## Verification

A scratchpad script runs the cleaner over both meshes and writes `mesh_clean.ply` beside each
`mesh.ply`. The originals are never rewritten: `mesh.ply` is the name `Reconstructor`'s
skip-check, the dashboard and the remote push all read, and in-place rewriting would destroy
the baseline. Reported: components and faces before and after. The user downloads the two PLYs
and judges visually.

## Tests

Flat functions in `tests/mesh/test_utils.py`. The existing
`test_clean_repair_mesh_drops_out_of_bounds_components_and_fills_holes` is rewritten — the
out-of-bounds rule it named no longer exists. `test_clean_repair_mesh_use_largest_keeps_only_the_main_component`
is deleted with the flag. The remaining two are updated for `max_hole_frac`.

- a small isolated component is dropped
- a large component attached to the main body is kept
- the largest component is kept regardless of the thresholds
- `max_hole_frac` scales with the mesh, so the same value fills the same holes on a mesh scaled
  up by a constant factor

## Out of scope

Deliberately excluded, each with its reason above: multi-view depth consistency, grazing-angle
masking, normal-consistency face pruning, the cross-source VDA gate, and training-side changes
(scale/opacity regularisation, depth loss). The last is the only thing that removes near
blow-out rather than masking it, and it is excluded to protect the GS-SR parity these runs
exist to measure.

## Follow-up: pre-fusion depth cuts, built and measured off

Built in `_splats_to_tsdf_inputs` as `max_depth_frac` (scene-relative far cut) and
`max_depth_grad` (relative depth-jump mask), plus `splat_depth: median` for 2dgs. Both cuts
ship **off**. The first shipped defaults were 0.75 / 0.05; the ablation below says neither
earns a default, and 0.05 visibly deleted patches of good mesh.

Ablation, both scenes, voxel 0.2 / sdf 0.8 / depth_trunc 100, 2dgs on `median_depth`:

| scene | arm | raw comps | cleaned comps | cleaned faces |
| --- | --- | --- | --- | --- |
| 2dgs | none | 44,791 | 447 | 1,447,369 |
| 2dgs | far 0.75 | 44,791 | 447 | 1,447,369 |
| 2dgs | far 0.75 + grad 0.3 | 42,069 | 433 | 1,445,940 |
| 3dgs | none | 246,457 | 1,691 | 2,354,800 |
| 3dgs | far 0.75 | 232,540 | 1,790 | 2,376,717 |
| 3dgs | far 0.75 + grad 0.3 | 231,795 | 1,785 | 2,375,586 |

Three findings behind the off defaults:

1. **`depth_trunc` is the tighter far gate.** `0.75 * extent` is 102.2 / 97.2 against a
   `depth_trunc` of 100, so on 2dgs the far cut removed only pixels Open3D already discarded —
   the two `2dgs far` rows are identical to `none` to the digit. Depth runs p50 15.8, p90 63.0,
   p95 92.5, p99 226.1: the blow-out sits past `depth_trunc` and real room surface fills
   everything below it, so any frac low enough to slip under `depth_trunc` deletes 8-15% of
   live pixels of real geometry. The knob is a backstop for configs where `depth_trunc` is set
   beyond the trajectory, not a filter for this one.
2. **`max_depth_grad: 0.05` masks surfaces, not silhouettes.** 65.5% (2dgs) / 53.9% (3dgs) of
   masked pixels sit in blobs over 10,000 px, largest single blob 3.3% / 5.5% of a frame. A
   3x3 median pre-filter moves that by ~3 points, so the blobs are coherent structure rather
   than speckle. The threshold sits below the scene's own grazing-angle gradient: at fx ~= 1000
   a plane 88 degrees off the ray steps 0.03 of its depth per pixel. 3dgs becomes edge-like at
   0.3 (0.22% masked, no blob over 10k); 2dgs `median_depth` is rougher and still shows 22k-px
   blobs there.
3. **At a safe threshold the cut is redundant.** grad 0.3 removes 14 components on 2dgs and
   none on 3dgs. `clean_repair` already removes what it removes.

What actually cleaned these meshes: the component selector above, plus `splat_depth: median`
on 2dgs, which drops raw components 240,461 -> 44,791 before any cleaning. 3dgs has no median
render, which is why 1,691 speckle components survive there against 447 on 2dgs.

Shipped beside each `mesh.ply` as `mesh_filtered.ply`: 2dgs 447 comps / 1,447,369 faces,
main body 96.3%; 3dgs 1,691 / 2,354,800, main body 95.5%.

## Risks

1. On these scenes the rule lands within 1.4% of faces of largest-component-only. If the
   cleaned meshes render indistinguishably from that, the finding is that component topology
   carries little signal here and the lever moves entirely to the pre-fusion follow-up.
2. Near blow-out survives this change by construction. It is a splat-field defect — the
   Gaussians are in the wrong place — and no component-topology rule reaches geometry welded
   into the main component. Stated up front so the result is not read as a tuning failure.
3. `min_area_frac` and `max_gap_frac` are calibrated on two scenes at one voxel size. TSDF
   triangle area tracks `voxel_size**2`, so a different `mesh.voxel_size` shifts what
   `min_area_frac` means in triangles. Re-check before treating the defaults as universal.
