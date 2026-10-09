# Tutorial unification

Make the ten tutorial pages read and look like one tutorial: plain short prose, every plot
explained, one plotting toolkit (`collab_splats.utils.visualization` and the package's existing
viz modules), upright 3D views, and Talk2DINO as the semantics extractor.

Source: user review notes, 2026-10-09.

## Goals

- Every section opens with one sentence on what it shows and why it matters.
- Every code cell has a one-line lead-in; every plot says what to look for.
- Every metric is defined in plain words the first time it appears.
- Variable names say what they hold.
- All 3D views are upright, side-on, with small time-colored camera frustums.
- All heatmaps use viridis.
- Pages reuse package viz functions; no page-local plotting helpers in `tutorial.py`.

## Non-goals

- No new reconstruction, meshing or semantics behavior.
- No change to `VIZ_KWARGS` defaults (the dashboard depends on them).
- No new depth-warp figure in localization.

## Shared conventions

### Library extensions (`collab_splats/utils/visualization.py`)

1. `camera_view(extrinsics) -> dict`
   - input: (N, 4, 4) w2c OpenCV poses
   - up: mean of the cameras' up vectors (`-c2w[:, :3, 1]`), normalized
   - focal point: centroid of the camera centers, shifted along the mean forward direction by
     half the trajectory extent (bounding-box diagonal of the centers)
   - position: focal point pushed sideways (perpendicular to mean forward and up) and slightly
     up, at a distance scaled to the trajectory extent
   - returns a `viz_kwargs` dict that `apply_view` / `visualize_splat` accept
2. `visualize_splat` `camera_kwargs["cmap"]`
   - when set, frustum i is colored by i / (N - 1) through that colormap, with a scalar bar
     titled "frame order"
   - when absent, the current single color is kept
3. Heatmaps: every page heatmap goes through viridis (`apply_viridis` or `cmap="viridis"`),
   including pyvista scalar heat on meshes and the OCR keyframe figure.

### Library extension (`collab_splats/preproc/viz.py`)

- `plot_frame_extremes`: one row of `2n` frames (highest blur first, then lowest), figure width
  capped so the saved PNG is about 900 px wide.

### Library fix (`collab_splats/localization/viz.py`)

- `plot_correspondences`: if a portrait reference next to a landscape query renders as a thin
  strip, scale both panels to a shared height and width budget. Applied in the function so the
  dashboard gets it too.

### Prose rules

- Short sentences; no implementation trivia (zarr internals, hashes, CLI flags) unless the
  reader needs it to act.
- "In a pipeline run" closing sections stay, trimmed to the yaml block and one or two lines.

### Semantics extractor

- `SCENE_CONFIG["semantics"]["extractor"]`: `maskclip` -> `talk2dino`.

## Per-page changes

1. **Preprocessing**: frame extremes as one capped row; prose about half; define blur, clipped
   fraction, parallax and match count.
2. **Reconstruction**: point cloud via `visualize_splat` + `camera_view`, time-colored small
   frustums; drop intrinsics table and crop-box detail (one sentence on the two Ks); quality
   section defines multiview agreement, relative depth residual and parallax-vs-error in plain
   words; comparison figure keeps three panels with labels and legends.
3. **Refinement**: simpler prose; renamed variables (`ff_result`, `ba_result`, ...);
   `track_source="xfeat"` set explicitly; new BA loss curve from `ba.loss_history` (log y);
   before/after cameras via `visualize_splat` (grey feedforward, time-colored BA) with legend;
   trajectory plot with start/end markers and legend; print-only cells merged (about 10 cells).
4. **Train splats**: simpler prose; drop the `sh_degree` rule demo; keep report, weakest views,
   render grid with per-view PSNR in titles.
5. **Mesh**: drop the atlas image; fused/prepared and vertex-color/texture views upright via
   `camera_view`; small sky-mask thumbnail on one frame; texture section to two or three
   sentences.
6. **Feature extraction**: Talk2DINO row first; plain explanation of PCA colors and heatmap.
7. **Segmentation**: pool Talk2DINO features; new sky-mask section (`sky_masks` on the committed
   frame, `overlay_masks`, sky share printed).
8. **Lifting and query**: lift Talk2DINO features; viridis heat, grey unobserved, upright view;
   prompts "tree" and "trash can".
9. **OCR lens**: keyframe figure in the feature-extraction style (image, p("tree") upsampled and
   blended in viridis, best-word labels over the frame with legend; one shared extent); 3D probe
   heat viridis, grey unobserved, upright.
10. **Localization**: `plot_correspondences(..., warp_corners=True)` with inliers only (each
    frame's outline drawn in the other, as in the xfeat demo notebook); full-cloud
    `plot_reprojection` kept; scene view via `visualize_splat` + `camera_view`, keyframes
    time-colored, query red.

## Testing and gates

- Unit tests: `camera_view` (up vector opposite OpenCV y, output keys), frustum `cmap`
  (per-frustum scalars present), `plot_frame_extremes` (one row, width cap),
  `plot_correspondences` (panel sizes on mixed orientations).
- Docstring contract and import-style tests stay green.
- Full sweep on a fresh scene (extractor change reruns semantics), all ten pages rc=0, plus the
  cold-start localization run.
- Docs tests, sphinx build, notebooks under 32 MiB total.
- One commit on `clean/final`; no push.
