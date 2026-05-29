# Multi-Backbone Feed-Forward 3D → Mesh Pipeline Audit

## Summary
- **Backbones supported:** VGGT (VGGTXCreator), VGGT-Ω (VGGTOmegaCreator), MapAnything (MapAnythingCreator)
- **Integration pattern:** (d) selector — registry picks one backbone per scene; outputs never mixed
- **Default backbone:** `vggt_omega` (in `evals/eval_gt.py` CLI `--backbone` default)
- **Meshing method(s) detected:** Open3D `ScalableTSDFVolume` (`open3d_tsdf`); Poisson variants are registered stubs that raise `NotImplementedError`
- **Overall compliance:** PARTIAL
- **Critical issues:** 7
- **Backbone-specific issues:** VGGT: 5 | VGGT-Ω: 5 | MapAnything: 3
- **Cross-cutting issues:** 2

---

## Backbone Inventory

- **VGGT entry point:** `collab_splats/pointcloud/feedforward/vggtx.py` — `VGGTXCreator`
  Outputs consumed: `depth` (N,H,W,1) + `depth_conf` (N,H,W) for pointcloud; `world_points` (N,H,W,3) via `_raw_to_world_points` for BA + TSDF.
  Camera: `pose_encoding_to_extri_intri` → world-to-cam (3,4).

- **VGGT-Ω entry point:** `collab_splats/pointcloud/feedforward/vggt_omega.py` — `VGGTOmegaCreator`
  Outputs consumed: `depth` + `depth_conf`; `world_points` via `_raw_to_world_points`. Reuses `unproject_and_filter_points` from `vggtx.py`.
  Camera: `encoding_to_camera` → world-to-cam (3,4). Has `enable_text_alignment` flag but the embedding is NOT used for masking.

- **MapAnything entry point:** `collab_splats/pointcloud/feedforward/mapanything.py` — `MapAnythingCreator`
  Outputs consumed: `pts3d` (world-frame, H,W,3) + `depth_z` (validity) + `mask` (edge+confidence baked by `postprocess_model_outputs_for_inference`).
  Camera: `pred["camera_poses"]` (cam-to-world) → `invert_poses` → world-to-cam stored in result.
  **Metric-or-similarity:** Up-to-similarity — no calibration/pose priors passed; `MapAnything.from_pretrained("facebook/map-anything")` unconditionally.

---

## Data Flow

**VGGT / VGGT-Ω (shared path):**
1. `_preprocess` → load images at 518/512 px
2. `_forward` → VGGT/VGGTOmega inference → `pose_encoding_to_extri_intri` / `encoding_to_camera` → `depth`, `depth_conf`, `extrinsic` (world-to-cam, 3×4), `intrinsics` (orig + downsampled)
3. `_postprocess` → `unproject_and_filter_points(depth, depth_conf, ..., conf_threshold=50.0)` → global-percentile mask → `pts3d`, `colors`; `_raw_to_world_points` → `world_points` (N,H,W,3) at model res
4. `_feedforward_to_tsdf_inputs(result)` → re-project `world_points` via extrinsics → depths (N,H,W); `invert_poses(extrinsics)` → c2w
5. `Open3DTSDFFusion.create(depths, rgbs, c2w, intrinsics)` → TSDF integration → `extract_triangle_mesh()` → PLY

**MapAnything:**
1. `_preprocess` → `load_images` (aspect-ratio lookup, ~518 px default)
2. `_forward` → `model.forward(..., memory_efficient_inference=True, minibatch_size=1)` → list of per-view dicts
3. `_postprocess` → `postprocess_model_outputs_for_inference(..., apply_mask=True, mask_edges=True, apply_confidence_mask=True, confidence_percentile=35.0)` → per-view masks baked; `pts3d` (world) extracted via `invert_poses(camera_poses)` rotation; `world_points = stacked_pts3d`
4–5: same TSDF path as above

---

## Detailed Findings

### Per-Backbone Matrix

| Check | VGGT | VGGT-Ω | MapAnything |
|---|---|---|---|
| **A1** raw outputs only | COMPLIANT | COMPLIANT | COMPLIANT |
| **A2** correct extrinsics inversion | COMPLIANT (vggtx.py:467, utils.py:67) | COMPLIANT (same path) | COMPLIANT (mapanything.py:67, 256) |
| **A3** single geometry source | COMPLIANT (depth only; world_points_conf unused) | COMPLIANT (no world_points head) | COMPLIANT (pts3d factored geometry) |
| **A4** confidence head matched | COMPLIANT (depth_conf with depth) | COMPLIANT (depth_conf with depth) | COMPLIANT (MA's internal conf with pts3d) |
| **A5** batch alignment | PARTIAL (optional global_alignment, off by default; LC wrapper handles multi-submap) | PARTIAL (same) | COMPLIANT (single joint inference via MA model) |
| **A6** MapAnything metric claim | N/A | N/A | COMPLIANT (no priors → similarity; not claimed metric) |
| **B7** scene-relative voxel (similarity) | NONCOMPLIANT (fixed 0.01 m) | NONCOMPLIANT (fixed 0.01 m) | NONCOMPLIANT (fixed 0.01 m) |
| **B8** metric voxel only for metric | N/A | N/A | N/A (no metric output produced) |
| **B9** voxel respects depth resolution | NONCOMPLIANT (no scene_depth/800 check) | NONCOMPLIANT | NONCOMPLIANT |
| **C10** per-image percentile | NONCOMPLIANT (global across all N frames) | NONCOMPLIANT (same fn) | PARTIAL (MA library applies per-view internally) |
| **C11** conf as fusion weight | NONCOMPLIANT (binary mask only) | NONCOMPLIANT (binary mask) | NONCOMPLIANT (boolean mask) |
| **C12** integrator supports weights | NONCOMPLIANT (`ScalableTSDFVolume` has no per-pixel weight API) | NONCOMPLIANT (shared) | NONCOMPLIANT (shared) |
| **D13** sky masking | NOT FOUND | NOT FOUND (text_alignment loaded but not used for masking) | NOT FOUND |
| **D14** dynamic-object masking | NOT FOUND | NOT FOUND | NOT FOUND |
| **D15** edge-discontinuity filter | NOT FOUND | NOT FOUND | PARTIAL (`mask_edges=True` in MA postprocess; criterion unverified) |
| **D16** far-field truncation | PARTIAL (fixed 20 m, not 8× median depth) | PARTIAL (same) | PARTIAL (same) |
| **E17** meshing method | Open3D TSDF (`ScalableTSDFVolume`) | Open3D TSDF | Open3D TSDF |
| **E18** Poisson flagged for outdoor | N/A (Poisson stubs, NotImplementedError) | N/A | N/A |
| **E19** TSDF min_weight post-filter | NONCOMPLIANT (no min_weight filter on extract_triangle_mesh) | NONCOMPLIANT (shared) | NONCOMPLIANT (shared) |
| **E20** density-trim after Poisson | N/A | N/A | N/A |
| **E21** camera-derived normal orientation | NOT FOUND | NOT FOUND | NOT FOUND |
| **E22** visibility-aware Delaunay | N/A | N/A | N/A |
| **F23** mesh in input world frame | COMPLIANT | COMPLIANT | COMPLIANT |
| **F24** reproducible params via config | COMPLIANT (dataclass fields, `**mesher_kwargs`) | COMPLIANT | COMPLIANT |
| **F25** sanity check / reprojection | NOT FOUND | NOT FOUND | NOT FOUND |
| **F26** large-scene scale strategy | PARTIAL (`ScalableTSDFVolume` sparse hashing; no out-of-core; LoopClosure uses submaps) | PARTIAL | PARTIAL |

---

### Non-Compliant / Partial Evidence

**B7/B9 — Fixed voxel_size=0.01 (all backbones):**
`mesh/tsdf.py:34`: `voxel_size: float = 0.01`. All three backbones are up-to-similarity; scene scale is arbitrary. A 10-metre indoor scene might reconstruct at 0.3-unit scale, making 0.01-unit voxels 300× too fine or coarse depending on normalization. No scene_depth/800 guard anywhere.

**C10 — Global confidence percentile (VGGT, VGGT-Ω):**
`feedforward/vggtx.py:77`: `np.percentile(depth_conf, conf_threshold)` where `depth_conf` is (N,H,W) — flattened across all frames. A single high-confidence frame raises the global threshold, aggressively discarding good pixels from other frames. Fix: compute per-frame percentile, apply per-frame mask.

**C11/C12 — Binary mask + TSDF ignores confidence entirely:**
`vggtx.py:81`: `conf_mask = depth_conf >= threshold_val` — boolean, no gradient. `tsdf.py:71-84`: TSDF integration loop takes `depth_o3d` with no weight image. `ScalableTSDFVolume` has no `weights` parameter. All depth measurements are fused with equal weight regardless of confidence.

**D13/D14 — No sky or dynamic masking (all backbones):**
Zero references to sky, person, vehicle, or bicycle masking in `collab_splats/mesh/` or any `_postprocess`. For outdoor scenes this causes sky-plane artifacts and ghost geometry from moving objects.

**D15/D16 — Partial edge/far-field masking:**
`mesh/utils.py`: no `|∇d| < 5%×d` filter. `tsdf.py:36`: `depth_trunc: float = 20.0` — fixed absolute; meaningless for similarity-scale outputs where scene units are arbitrary.

**E19 — No min_weight post-filter:**
`tsdf.py`: `mesh = volume.extract_triangle_mesh()` with no minimum observation count. Voxels visited by one frame produce the same mesh triangles as voxels visited by 20 frames. Standard practice is to pass `weight_threshold` or filter `tsdf.voxel_grid_` post-extract.

**D13 / VGGTOmegaCreator text_alignment unused for masking:**
`vggt_omega.py:117-118`: `enable_text_alignment` loads the text-aligned checkpoint variant. The `text_alignment_embedding` output is never accessed in `_forward` or `_postprocess`. This is free sky/semantic signal that isn't exploited.

---

### Cross-Cutting Findings (G1–G8)

- **G1 [COMPLIANT]:** All three store world-to-cam in `FeedforwardResult.extrinsics`. VGGT/Ω via `pose_encoding_to_extri_intri`/`encoding_to_camera` (native OpenCV convention). MapAnything inverts `camera_poses` (cam-to-world) before storing. `_feedforward_to_tsdf_inputs` inverts uniformly via `invert_poses`. No convention mismatch.

- **G2 [COMPLIANT]:** Pattern (d) selector — each backbone has its own class with separate threshold fields (`conf_threshold=50.0` for VGGT/Ω vs `confidence_percentile=35.0` for MapAnything). No single numeric threshold shared across backbones.

- **G3 [N/A]:** Pattern (d) — outputs never fused across backbones. No scale-alignment hazard.

- **G4 [PARTIAL]:** `_feedforward_to_tsdf_inputs` (mesh/utils.py:449) error message reads "MapAnythingCreator and VGGTXCreator both populate world_points" — omits VGGTOmegaCreator. Code is correct (all three populate it), but the docstring is wrong and may mislead callers. `_raw_to_world_points` (base.py:281) correctly uses `raw.get("depth_conf")` as optional — no missing-field crash.

- **G5 [COMPLIANT]:** Pattern (d) — one backbone per scene, one world frame per reconstruction. No cross-scene composition in pipeline code.

- **G6 [COMPLIANT]:** Pattern (d) — scale derived from single backbone per scene. No re-derivation hazard.

- **G7 [COMPLIANT]:** Each backend stores model-resolution `world_points` and model-resolution `intrinsics` (via `intrinsics_downsampled`). `_feedforward_to_tsdf_inputs` derives depths from `world_points` using stored extrinsics (no intrinsics needed for re-projection), and TSDF receives the matching intrinsics. No resolution mismatch.

- **G8 [PARTIAL]:** `vggt_omega` is the default backbone in `evals/eval_gt.py` CLI but no README or docstring explains why (presumably: better depth quality or lower memory vs. VGGT-X, which needs investigation). The three backbones are not interchangeable (different confidence scales, resolution, no metric MA), but no justification is documented.

---

## Critical Issues (Ranked)

1. **Fixed voxel_size=0.01 for similarity-scale outputs (all backbones):** TSDF geometry is correct only if scene units happen to be metres. For all three up-to-similarity backbones the scale is arbitrary — mesh resolution is unpredictable and untunable without knowing the scene's implicit scale unit.
   Fix: auto-derive `voxel_size = median_scene_depth / 500` from `result.world_points` in `_feedforward_to_tsdf_inputs`.

2. **Global confidence percentile (VGGT, VGGT-Ω):** Single `np.percentile` over (N,H,W) treats all frames as one pool; high-confidence frames pollute the threshold for low-confidence ones.
   Fix: loop per-frame in `unproject_and_filter_points`; compute `np.percentile(depth_conf[i], threshold)` per frame `i`.

3. **No sky / dynamic masking (all backbones):** Outdoor scenes will always fuse sky planes and moving objects into the mesh.
   Fix: integrate a lightweight segmenter (SegFormer-sky or SAM2 fast) as a pre-filter; for VGGT-Ω, use `text_alignment_embedding` (already loaded) as the sky signal.

4. **No TSDF min_weight post-filter (all):** Single-view voxels produce the same mesh triangles as well-observed voxels.
   Fix: after `extract_triangle_mesh()`, filter vertices/triangles with `voxel_grid` observation counts ≤ `min_weight` (or use `extract_triangle_mesh(weight_threshold=N)`).

5. **Confidence treated as binary mask, TSDF has no weighting (all):** Low-confidence depth near the mask boundary contributes equally to TSDF as high-confidence interior points.
   Fix: pass depth-conf as a weight map to a custom integration or use VDBFusion which supports per-voxel weights.

6. **Fixed depth_trunc=20 m (not scene-relative, all backbones):** For similarity-scale scenes the absolute 20 m cap is meaningless — the scene might have internal units of 0.5–50× metres.
   Fix: compute `depth_trunc = 8 * np.median(depths[depths > 0])` inside `_feedforward_to_tsdf_inputs`.

7. **G4 docstring omits VGGTOmegaCreator in `_feedforward_to_tsdf_inputs` error message (mesh/utils.py:449):** Minor but misleads users of `pointcloud_to_mesh` about which backends are supported.
   Fix: update error string to include VGGTOmegaCreator.

---

## Files Reviewed

| Path | Role | Backbone(s) |
|---|---|---|
| `collab_splats/pointcloud/feedforward/vggtx.py` | VGGTXCreator, `unproject_and_filter_points` | VGGT |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | VGGTOmegaCreator | VGGT-Ω |
| `collab_splats/pointcloud/feedforward/mapanything.py` | MapAnythingCreator, `_reproject_mapanything` | MapAnything |
| `collab_splats/pointcloud/feedforward/base.py` | `FeedforwardResult`, `BaseFeedforwardCreator`, `_raw_to_world_points` | all |
| `collab_splats/mesh/tsdf.py` | `Open3DTSDFFusion` | all (shared) |
| `collab_splats/mesh/poisson.py` | `DepthNormalPoisson`, `GaussiansPoisson` (stubs) | all (shared) |
| `collab_splats/mesh/base.py` | `BaseMeshCreator`, `MeshResult` | all (shared) |
| `collab_splats/mesh/utils.py` | `_feedforward_to_tsdf_inputs`, `pointcloud_to_mesh` | all (shared) |
| `collab_splats/mesh/__init__.py` | mesh REGISTRY | all |
| `collab_splats/pointcloud/__init__.py` | creator REGISTRY | all |
| `collab_splats/pointcloud/wrappers.py` | `BundleAdjustment`, `LoopClosure` wrappers | all |
| `evals/eval_gt.py` | default backbone reference (`--backbone`, default=`vggt_omega`) | VGGT, VGGT-Ω |
| `worklog/specs/2026-05-14-feedforward-mesh-design.md` | feedforward→mesh spec | all |
| `worklog/STATE.md` | project state, in-flight work | — |

---

## Not Reviewed / Out of Scope

- `collab_splats/pointcloud/bundle_adjustment.py` — BA correctness is out of scope for mesh audit; only how BA-refined poses flow into TSDF was checked.
- `collab_splats/pointcloud/loop_closure/` — LC/SL4 pose-graph correctness previously audited; only the mesh integration point (FeedforwardResult.extrinsics after LC) was spot-checked.
- `third_party/VGGT-SLAM/` — reference implementation only; no mesh code.
- `collab_splats/nerfstudio/` — separate GS-based pipeline; shares no code with feedforward mesh path.
- MapAnything's `postprocess_model_outputs_for_inference` internals (library code, not vendored).

---

## Top 3 Highest-Impact Changes

1. **Auto-derive `voxel_size` from scene depth** — `mesh/utils.py:_feedforward_to_tsdf_inputs`: add `voxel_size = median_depth / 500` and pass to mesher; affects all three backbones. Without this, mesh quality is luck-dependent on the hidden similarity scale.

2. **Per-frame confidence percentile** — `feedforward/vggtx.py:unproject_and_filter_points`: compute `np.percentile(depth_conf[i], conf_threshold)` inside the per-frame loop instead of once over the full array; affects VGGT and VGGT-Ω.

3. **VGGT-Ω: use `text_alignment_embedding` for sky masking** — `feedforward/vggt_omega.py:_postprocess`: extract sky-token similarity from `text_alignment_embedding` (already loaded via `enable_text_alignment`) to build a per-pixel sky mask before TSDF integration; affects VGGT-Ω only, free signal with no additional model cost.
