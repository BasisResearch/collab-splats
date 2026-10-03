# Texture: view-chart unwrap + per-view color correction

Status: landed on `clean/final` (2026-10-03); unwrap is the V7 face-id design below

## Goal

Two changes to `collab_splats/mesh/texture.py`. Both go in that file and nowhere else in the package.

1. **Fast unwrap.** Replace UVAtlas with a view-chart unwrap. UVAtlas is deleted, not kept as an option.
   - UVAtlas takes ~30 min of the ~45 min texture stage on GH010229 (1.1M faces).
   - View charts unwrapped the same mesh in 11 s, and projection fell from 122 s to 61 s.
2. **Color correction.** Before projection, solve one gain per view and per channel, then divide it out of each image.
   - Removes the exposure seams (for example the light/dark band across the grass) that the weighted blend leaves.

Out of scope:
- Pose refinement. It is measured, but deferred to its own effort.
- Re-fusing the TSDF with refined poses.
- Two-band blending.
- Grey fill on surfaces that no view sees.

## Evidence (scratch prototypes, GH010229 VGGT-Omega)

Setup: 264 training views; every 10th frame held out (30 views). Same mesh and UVs across arms unless noted. Scored with refined poses.

| Arm | PSNR | Exposure-fair PSNR | Color PSNR | Sharpness | Unwrap + project |
|---|---|---|---|---|---|
| UVAtlas | 14.00 | 14.28 | 17.54 | 0.98 | ~30 min + 126 s |
| UVAtlas + gain v2 | — | 14.61 | 18.27 | 0.93 | ~30 min + 8 s gain + 123 s |
| View charts v5 | 13.91 | 14.24 | 17.46 | 1.08 | 11 s + 61 s |

Columns:
- **Exposure-fair PSNR** fits one gain per channel from the render to the photo before scoring.
- **Color PSNR** blurs both images 16× first, then applies the same gain fit.
- **Sharpness** is relative to the source images.

Sources:
- Prototypes: `$S/view_charts.py` and `$S/color_gain.py`, where `$S` is the session scratchpad.
- Report: https://claude.ai/artifact/ATsTyyR5Z4CFcxXj58uysd

**Grey patches: v5 failed, V7 (face-id ownership) fixed them.**

- v5 cause: a loose 2% depth test let hidden and back-facing faces take a camera; 17% of UV area overlapped, so faces lost all their texels and `fill_missing_pixels` painted them grey.
- The tiny-chart merge first proposed here was dropped: the cause was overlap, not chart size.
- V6 (fold-repair loop) fixed the overlap but took ~82 s per loop.
- V7 needs no loop: the face-id buffer makes camera charts injective by construction.

| Arm (all + gain v2) | Exposure-fair PSNR | Color PSNR | Grey (Z) pixel share | Unwrap | Total |
|---|---|---|---|---|---|
| UVAtlas | 14.658 | 18.30 | — | ~30 min | ~32 min |
| v5 view charts | 14.626 | 18.38 | 2.05% | 11 s | — |
| V7 face-id charts | 14.623 | 18.24 | 0.09% | 23 s | ~90 s |

- 474k of 1.1M faces (43%) are seen by no view at full ownership: per-view checks show they are genuinely hidden (view 55: 185k framed, 27k own pixels).
- V7 texel density is ~12% below UVAtlas (1552 vs 1762 texels/m), spent on single-face charts.

## Design

### Pipeline in `create_texture_mesh`

```
_validate_views
-> [color_correct] _solve_view_gains(occluder, rgbs, w2c, K) -> _apply_view_gains(rgbs, gains)
-> unwrap_view_charts
-> project_images_to_texture (torch depth test; Warp removed)
-> _dilate_chart_gutters
-> write_textured_obj
```

Gains are solved on the occluder (the cleaned, unfilled mesh): only real surfaces should vote. They are applied to `rgbs` before anything samples the images, so the view-chart labels and the projection both see corrected colors.

New keyword argument on `create_texture_mesh`: `color_correct: bool = True`. Unwrap tunables are `unwrap_view_charts` keyword defaults, not module constants.

`create_texture_mesh` gains no new required argument: the occluder, poses and K are already passed in.

### Color correction

**`_solve_view_gains(verts, faces, rgbs, c2w, K, *, n_points=300_000, scale=0.5, blur=5, rel_tol=0.01, prior=1e-2, trim=0.25, smooth=10.0) -> np.ndarray`**

Returns (V, 3) gains.

Observations:
- Sample area-weighted face centroids.
- Depth-test them in each view against an nvdiffrast depth render at half resolution.
- Read color from a box-blurred image.
- Drop clipped (> 0.97) and near-black (< 0.02) samples.

Model: `log observed = log gain[view] + log albedo[point]`. Points are eliminated exactly, so there is one (V, V) solve per channel. The system includes:
- `smooth * n * DᵀD / V`: penalizes the gain change between consecutive frames.
- A tiny `prior`.
- A sum-to-zero pin.

Solve steps:
1. Solve, then drop observations with an absolute log residual above `trim`.
2. Solve again.
3. Subtract the median log gain, so the typical view is the reference and the texture keeps a typical exposure.

**`_apply_view_gains(rgbs, gains, *, knee=200.0) -> np.ndarray`**
- Divides each view by its gain on the GPU, one view at a time, into a new array.
- Values above `knee` roll off exponentially toward 255 instead of clipping.

Why these defaults (measured in the sweep):
- With `smooth=0`, gains spanned 0.35–1.94 and 11.7% of pixels clipped, which darkened and blew out spots.
- With `smooth=10`, gains span 0.59–1.57, the jump between neighboring frames at p95 is 0.026, and 9.9% of pixels clip before the knee.

Cost: 8 s on 264 views.

The camera projection matrix for the depth render comes from a private `_gl_projection(K, w, h)` in texture.py. The prototype imported it from `compare.py`.

Known quirk, recorded and not tuned here: `trim=0.25` drops 49% of observations on GH010229. The fit holds (residual std 0.55 → 0.11), but the trim is aggressive.

### View-chart unwrap

**`unwrap_view_charts(mesh, c2w, K, image_hw, tex_size, *, frac=0.75, small_px=4.0, rel_tol=0.01, alpha=0.25, rounds=40, tile_m=0.5, pad=1, min_owned=0.5) -> tuple[np.ndarray, np.ndarray]`**

Returns (F, 3, 2) UVs and (C, 4) chart boxes; `create_texture_mesh` passes them with plain vertex/face/normal arrays, and the gutter dilation draws its own box map.

1. **Score.** Rasterize the mesh itself in each view with nvdiffrast; face id per pixel. A face counts in a view if it is in front, fully framed, and either owns at least `frac` of its projected area (one pixel slack) or, under `small_px`, sits at the z-buffer depth at its centroid within `rel_tol`. Score = owned pixels, (F, V) float16.
2. **Label.** Argmax score, then neighbor majority vote restricted to views with score >= `alpha` × best and > 0.
3. **UVs.** Corner pixel coordinates in the label view (`geometry.projection.project`, batched per face).
4. **Unseen faces.** Connected groups get one plane patch each, projected along the group's area-summed normal.
5. **Pack.** Charts = connected faces sharing a key (view, patch, or single face); tiled at `tile_m`; scaled to one texel density; rotated to principal axes; shelf-packed; global scale binary-searched to fit.
6. **One check.** Rasterize atlas face ids: plane faces wound against their chart's majority (flipped), or faces owning under `min_owned` of their expected texels (lost), become single flat charts. Repack once. No loop.

**`_dilate_chart_gutters(albedo, uvs, boxes, steps=2)`** copies each chart's edge texels outward into its own padding.

Reused package code: `geometry.projection.project`, `geometry.transforms.rescale_intrinsics` / `invert_poses` / `extract_intrinsics`, `mesh.clean._face_edge_ids`, scipy `connected_components`. New private helpers: `_gl_projection`, `_rasterize_view` (shared by gains and scores), `_rasterize_uvs` (shared by the check, the gutter mask and the projection's texel raster).

### Projection without Warp

Added after the gate: `project_images_to_texture` drops its four Warp kernels and the `warp-lang` pin.

- Occlusion: a per-view nvdiffrast depth render of the occluder (`_rasterize_view`) replaces the BVH ray cast.
- A texel is hidden when its nearest raster pixel (`_nearest_depth`, rounded; `_gl_projection` centers pixel j on u = j, the package convention) holds a surface nearer by more than `occlusion_eps`.
- Rejected: testing all four bilinear pixels. Grazing ground hid itself (unseen 34.0% → 42.7%, exposure-fair PSNR −0.03 dB).
- Pixel size, view gating (`view_ratio`) and weights are unchanged; bilinear sampling is `F.grid_sample`.
- Takes `(verts, faces, normals, uvs)` arrays and an optional `(verts, faces)` occluder, not an Open3D tensor mesh; texel positions and normals stay on the GPU (`_rasterize_atlas` folded in).
- GH010229 prototype (nearest-pixel test): exposure-fair PSNR 14.623 → 14.623, color PSNR 18.245 → 18.245, sharpness 0.943 → 0.971.
- `bae` still depends on `warp-lang`, so the package stays installed; only our direct pin goes.

### Smaller OBJ

`write_textured_obj` writes the OBJ itself instead of through trimesh:

- each position and normal once; faces index `v`/`vt`/`vn` separately; UVs deduplicated
- 6 decimals instead of trimesh's 8
- MTL is `Kd 1.0 1.0 1.0` + `map_Kd` only

### Config

No new config keys. `mesh.texture: true` now means view charts plus color correction.

`create_texture_mesh(..., color_correct=False)` turns color correction off for A/B runs.

`reconstructor.mesh` is unchanged.

### UVAtlas removal

- Delete `unwrap_mesh_uvs` and its tests in `tests/mesh/test_texture.py`.
- Update the module docstring and `docs/mesh.md` (lines 11 and 226).
- `tests/mesh/test_clean.py:386,405` call Open3D `compute_uvatlas` directly to check that `prepare_mesh` output is UVAtlas-safe (the PCA partition raises on bad input). With UVAtlas gone, that check guards nothing. Delete those two cases unless they also assert manifoldness that nothing else covers; check during the plan.

## Gate before the unwrap change lands

Run the gate on the branch, before the commit that deletes UVAtlas.

- GH010229, in scratch only; never write to `/workspace/outputs/ocr_viewer/GH010229`.
- 264 training views, 30 held-out views.
- Color correction on in both arms.
- The UVAtlas reference is the existing `$S/arms/O6g_ours_gain_v2` (UVAtlas + gain v2); no new 30-min run needed.

1. **Visual:** no flat grey patches on the held-out crops that failed v5 (container above the truck, grass at the bottom right). Shown to the user side by side with UVAtlas.
2. **Scores:** exposure-fair PSNR at least UVAtlas − 0.1 dB, and color PSNR at least UVAtlas − 0.1 dB.
3. **Coverage:** no more than UVAtlas's share of grey-filled texels on seen surface.
4. **Time:** unwrap plus projection under 3 min.

If the gate fails:
- The unwrap change does not land.
- Color correction lands alone, on UVAtlas.
- The failure goes back to the user with crops; no fallback option is shipped.

## Tests (`tests/mesh/test_texture.py`, flat functions)

**Color:**
- `_solve_view_gains` recovers known per-view gains, to 1e-2 in log space, on a synthetic textured plane rendered from several views with injected gains.
- A single view (no point seen twice) returns unit gains.
- `_apply_view_gains` leaves values below the knee at exact division and never exceeds 255.

**Unwrap** (two stacked planes, the lower one fully hidden):
- Every face gets UVs in [0, 1].
- Seen (camera chart) and hidden (flat patch) faces share one texel density, within 2%.
- Texel-center count matches UV area within 5%: no face is drawn over another.

**Integration:** `create_texture_mesh` writes mesh.obj, mesh.mtl and albedo.png on the existing small fixture.

`tests/test_docstring_contract.py` already covers `mesh`. New public functions need full docstrings; private ones need a summary only.

## Order

Done as one change: gains, V7 unwrap, UVAtlas removal (`unwrap_mesh_uvs`, its test, the `compute_uvatlas` checks in `tests/mesh/test_clean.py`, UVAtlas prose in `clean.py` and `docs/mesh.md`).
