# Mesh floater cleanup — design

Date: 2026-08-27
Status: approved, not yet implemented

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

## Follow-up, not designed yet

If floaters survive the cleaner: pre-fusion filtering in `_splats_to_tsdf_inputs` — a
scene-relative far-depth cut (4% of pixels at 0.75x the camera-trajectory diagonal), a relative
depth-gradient mask (3.5% at 0.05), and `splat_depth: median` for the 2dgs run, where expected
and median depth disagree by more than 5% on 9% of pixels. Scoped only after the cleaner's
result is seen.

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
