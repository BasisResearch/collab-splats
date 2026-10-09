# Tutorial unification

Make the ten tutorial pages read and look like one tutorial: plain short prose, every plot
explained, one plotting toolkit (`collab_splats.utils.visualization` and the package's existing
viz modules), upright 3D views, and Talk2DINO as the semantics extractor.

Source: user review notes, 2026-10-09.

## Goals

- Every section opens with one sentence on what it shows and why it matters.
- Every code cell has a one-line lead-in; every plot says what to look for.
- Every cell earns its place: print-only and demo cells are merged or cut; cell code stays short
  and plain.
- Every metric is defined in plain words the first time it appears.
- Variable names say what they hold.
- All 3D views (point clouds and meshes) are upright, side-on, with small time-colored camera
  frustums.
- All heatmaps use viridis.
- Pages reuse package viz functions; no page-local plotting helpers in `tutorial.py`.

## Non-goals

- No new reconstruction, meshing or semantics behavior.
- No change to `VIZ_KWARGS` defaults (the dashboard depends on them).
- No change to `configs/base.yaml`; the extractor switch is tutorial-only.
- No new depth-warp figure in localization.

## Shared conventions

### Package viz functions the pages use

- 3D: `visualize_splat`, `apply_view`, `create_camera_frustum_pyvista`, `plot_reprojection`
- 2D: `compute_heatmap`, `apply_viridis`, `compute_masked_image`, `overlay_masks`,
  `feature_viz_row`
- preproc: `plot_frame_extremes` and the other `collab_splats.preproc.viz` plots
- localization: `plot_correspondences`, `plot_inlier_distribution`

### Library changes (`collab_splats/utils/visualization.py`)

1. New `camera_view(extrinsics) -> dict`
   - input: (N, 4, 4) w2c OpenCV poses
   - up: mean of the cameras' up vectors (`-c2w[:, :3, 1]`), normalized; fixes the upside-down
     point clouds and meshes (pyvista's default view assumes y-up, OpenCV is y-down)
   - focal point: centroid of the camera centers, shifted along the mean forward direction by
     half the trajectory extent (bounding-box diagonal of the centers)
   - position: focal point pushed sideways (perpendicular to mean forward and up) and slightly
     up, at a distance scaled to the trajectory extent
   - returns a `viz_kwargs` dict that `apply_view` / `visualize_splat` accept
2. `visualize_splat` frustums
   - already colored viridis by frame order when `camera_kwargs` has no `"color"`; pages switch
     from hand-rolled frustum loops to it and stop passing a color
   - add a scalar bar titled "frame order" when it colors by time
   - small frustums come from the existing `"scale"` key: pages pass about 0.005 x scene extent
     (far plane = 5 x scale, so about 2.5% of the scene) instead of today's 0.02-0.05
   - extra camera sets (feedforward cameras in refinement, the query in localization) are added
     to the returned plotter with `create_camera_frustum_pyvista` in one fixed color, plus
     `add_legend`

### Library change (`collab_splats/preproc/viz.py`)

- `plot_frame_extremes`: one row (n highest, then n lowest of the column) instead of two rows;
  new `max_width` keyword (pixels, default about 900) caps the saved PNG width.

### Library fix, only if needed (`collab_splats/localization/viz.py`)

- `plot_correspondences`: if the executed page shows the portrait reference as a thin strip
  next to the landscape query, scale both panels to a shared size budget inside the function, so
  the dashboard gets it too. If the page looks fine, no change.

### Heatmaps

- Viridis everywhere a scalar is shown as color: lifting 3D heat (today `turbo`), OCR 2D and 3D
  probe heat (today `magma`), feature similarity heat (`compute_heatmap`, already viridis).
- Categorical maps (OCR best-word labels, `overlay_masks`) keep categorical colors.
- Unobserved vertices are grey, not the low end of the colormap.

### Prose rules

- Short sentences; no implementation trivia (zarr internals, hashes, CLI flags) unless the
  reader needs it to act.
- "In a pipeline run" closing sections stay, trimmed to the yaml block and one or two lines.

### Semantics extractor

- `tutorial.py` `SCENE_CONFIG["semantics"]["extractor"]`: `maskclip` -> `talk2dino`.
- Talk2DINO is used for feature extraction (shown first, MaskCLIP second), segmentation
  pooling, lifting and query.

## Per-page changes

1. **Preprocessing**: frame extremes as one capped row; prose about half; define blur, clipped
   fraction, parallax and match count in one plain line each.
2. **Reconstruction**: point cloud via `visualize_splat` + `camera_view` with small time-colored
   frustums; drop the intrinsics table and crop-box detail (one sentence on the two Ks); define
   - multiview agreement: share of pixels whose depth another view confirms within 5%
   - relative depth residual: depth difference / depth; 0 means the views agree
   - parallax vs error: wider-baseline pairs should show lower depth error
   
   The comparison figure keeps three panels, each with a title, axis labels and legend.
3. **Refinement**: simpler prose; renamed variables (`ff_result`, `ba_result`, ...);
   `track_source="xfeat"` set explicitly; new BA loss curve from `ba.loss_history` (log y,
   labeled axes); before/after cameras via `visualize_splat` (BA time-colored, feedforward grey)
   with a legend; trajectory plot with start/end markers and a legend; print-only cells merged
   (about 14 -> 10 cells), each with a lead-in.
4. **Train splats**: simpler prose; drop the `sh_degree` rule demo cell; keep report, weakest
   views and the render grid with per-view PSNR in titles.
5. **Mesh**: drop the texture atlas image; fused/prepared and vertex-color/texture views upright
   via `camera_view`; small sky-mask thumbnail on one frame; texture section to two or three
   sentences.
6. **Feature extraction**: Talk2DINO row first, then MaskCLIP; plain explanation of what the PCA
   colors and the similarity heatmap show.
7. **Segmentation**: pool Talk2DINO features; new sky-mask section (`sky_masks` on the committed
   frame, `overlay_masks`, sky share printed).
8. **Lifting and query**: lift Talk2DINO features; viridis heat, grey unobserved, upright view;
   prompts "tree" and "trash can".
9. **OCR lens**: keyframe figure in the feature-extraction style: frame, probe heat upsampled
   and blended in viridis, best-word labels over the frame with a legend, one shared extent; 3D
   probe heat matches the lifting page (viridis, grey unobserved, upright).
10. **Localization**: `plot_correspondences(..., warp_corners=True)` with inliers only (each
    frame's outline drawn in the other, as in the xfeat demo notebook); full-cloud
    `plot_reprojection` kept unchanged; scene view via `visualize_splat` + `camera_view`,
    keyframes time-colored, query red.

## Testing and gates

- Unit tests: `camera_view` (up vector opposite OpenCV y, output keys), `visualize_splat`
  scalar bar present when coloring by time, `plot_frame_extremes` (one row, width cap); a
  `plot_correspondences` test only if that fix lands.
- Docstring contract and import-style tests stay green.
- Full sweep on a fresh scene (extractor change reruns semantics), all ten pages rc=0, plus the
  cold-start localization run.
- Docs tests, sphinx build, notebooks under 32 MiB total.

## Landing

- This spec is its own commit on `clean/tutorials`.
- The implementation lands as one commit: library changes, tests, page edits, executed
  notebooks, and the earlier uncommitted page fixes (preprocessing quality filter, refinement BA
  config, mesh depth_trunc cell).
- `clean/tutorials` is fast-forwarded onto `clean/final`; the main checkout's uncommitted
  `README.md` and dead-code spec edits are preserved. No push.
