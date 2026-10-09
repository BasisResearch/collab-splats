# Tutorial Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ten tutorial pages one consistent, plain-language tutorial with upright 3D views,
time-colored small frustums, viridis heatmaps and Talk2DINO semantics.

**Architecture:** Three small library changes (`camera_view`, a frame-order scalar bar in
`visualize_splat`, a one-row width-capped `plot_frame_extremes`), then each notebook rewritten
against them and executed on the shared tutorial scene.

**Tech Stack:** pyvista, matplotlib, nbconvert, pytest. Spec:
`docs/superpowers/specs/2026-10-09-tutorial-unification-design.md`.

---

## Conventions for every task

- Worktree `W=/workspace/collab-splats/.worktrees/tutorial-rework`, branch `clean/tutorials`.
- Python: `cd $W && PYTHONPATH=$W /opt/venv/reconstruction/bin/python`.
- Execute one page: `bash $SP/run_page.sh <dir> <nb>` (`$SP` = session scratchpad); then view
  its plot images (extract with `$SP/dumpsrc.py` / nbimg extract) before calling the page done.
- No per-task commits: the spec's Landing section asks for one implementation commit (Task 14).
- Notebook edits: rewrite cells with a small Python script over the `.ipynb` JSON (cell list
  replaced wholesale per page), never by hand-editing JSON.
- Prose rules (every page): one sentence per section saying what it shows and why; a one-line
  lead-in above every code cell; every plot followed by one or two lines on what to look for;
  each metric defined in plain words where it first appears; no zarr internals, hashes or flags
  unless the reader needs them.
- 3D views: `visualize_splat(cloud_or_mesh, extrinsics, camera_kwargs={"scale": 0.005 * size,
  "n_poses": 1}, viz_kwargs=camera_view(extrinsics, points))` where `size` is the
  5-95 percentile diagonal of the scene points.
- Heatmaps: viridis; unobserved vertices `nan_color="lightgrey"`.

## File map

- Modify `collab_splats/utils/visualization.py`: add `camera_view`; frame-order scalar bar in
  `visualize_splat`.
- Modify `collab_splats/preproc/viz.py`: `plot_frame_extremes` one row + `max_width`.
- Modify (only if Task 12 shows the thin strip) `collab_splats/localization/viz.py`.
- Tests: `tests/utils/test_visualization.py`, `tests/preproc/test_viz.py`.
- Modify `docs/source/tutorials/tutorial.py`: extractor -> `talk2dino`.
- Modify all ten notebooks under `docs/source/tutorials/`.
- Modify `docs/superpowers/CHANGELOG.md` and `CLAUDE.md` In-Flight list at landing.

---

### Task 1: `camera_view`

**Files:** Modify `collab_splats/utils/visualization.py` (after `apply_view`);
Test `tests/utils/test_visualization.py`.

- [ ] **Step 1: failing tests**

```python
def _walk_poses(n: int = 5) -> np.ndarray:
    """
    w2c poses of a camera stepping along world +x, looking along +z with OpenCV y down.
    """
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = -np.arange(n, dtype=float)
    return poses


def test_camera_view_up_is_opposite_opencv_y():
    view = camera_view(_walk_poses())
    np.testing.assert_allclose(view["view_up"], [0, -1, 0], atol=1e-6)


def test_camera_view_keys_reset_apply_view_defaults():
    view = camera_view(_walk_poses())
    assert {"position", "focal_point", "view_up"} <= set(view)
    assert view["azimuth"] == 0 and view["elevation"] == 0


def test_camera_view_looks_from_the_side():
    # Walk along +x, look along +z: side-on means the eye sits off the x-z plane's forward axis
    view = camera_view(_walk_poses())
    offset = np.asarray(view["position"]) - np.asarray(view["focal_point"])
    assert abs(offset[0]) > abs(offset[2])


def test_camera_view_points_set_focal_point():
    points = np.random.default_rng(0).normal(loc=[3, 0, 10], size=(1000, 3))
    view = camera_view(_walk_poses(), points)
    np.testing.assert_allclose(view["focal_point"], [3, 0, 10], atol=0.3)
```

- [ ] **Step 2:** run `python -m pytest tests/utils/test_visualization.py -k camera_view -q`
  → FAIL (ImportError).
- [ ] **Step 3: implement**

```python
def camera_view(
    extrinsics: np.ndarray,
    points: np.ndarray | None = None,
    *,
    azimuth_deg: float = 90.0,
    elevation_deg: float = 20.0,
    distance: float = 1.5,
) -> dict[str, Any]:
    """
    Upright side-on viz_kwargs derived from OpenCV camera poses.

    - up: mean camera up (-y of each c2w), so OpenCV y-down scenes render upright
    - azimuth 90 looks at the scene from the side of the walking direction, 0 from behind it
    - azimuth/elevation keys are zeroed so apply_view does not rotate the view again

    Args:
        extrinsics: (N, 4, 4) w2c OpenCV poses.
        points: (M, 3) scene points; their 5-95 percentile box sets focal point and size.
        azimuth_deg: angle around up from behind the cameras toward their right side.
        elevation_deg: angle above the horizontal.
        distance: eye distance in units of the scene size.

    Returns:
        A viz_kwargs dict for apply_view / visualize_splat.
    """
    # Mean up and forward of the cameras; forward made orthogonal to up
    c2w = invert_poses(np.asarray(extrinsics, dtype=np.float64))
    centers = c2w[:, :3, 3]
    up = -c2w[:, :3, 1].mean(axis=0)
    up /= np.linalg.norm(up)
    forward = c2w[:, :3, 2].mean(axis=0)
    forward -= up * (forward @ up)

    if np.linalg.norm(forward) < 1e-6:
        forward = np.cross(up, [1.0, 0.0, 0.0])

        if np.linalg.norm(forward) < 1e-6:
            forward = np.cross(up, [0.0, 1.0, 0.0])

    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)

    # Scene center and size: from the points when given, else ahead of the trajectory
    if points is not None:
        lo, hi = np.percentile(np.asarray(points), [5, 95], axis=0)
        focal = (lo + hi) / 2
        size = float(np.linalg.norm(hi - lo))
    else:
        size = float(np.linalg.norm(np.ptp(centers, axis=0))) or 1.0
        focal = centers.mean(axis=0) + 0.5 * size * forward

    # Eye on a sphere around the focal point
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    horizontal = np.cos(az) * -forward + np.sin(az) * right
    direction = np.cos(el) * horizontal + np.sin(el) * up
    position = focal + distance * size * direction

    return {
        "position": tuple(position),
        "focal_point": tuple(focal),
        "view_up": tuple(up),
        "azimuth": 0,
        "elevation": 0,
        "zoom": 1.0,
    }
```

- [ ] **Step 4:** rerun tests → PASS. Fix test helper docstring to one accurate line.

### Task 2: frame-order scalar bar in `visualize_splat`

**Files:** Modify `collab_splats/utils/visualization.py` `visualize_splat` frustum loop;
Test `tests/utils/test_visualization.py`.

- [ ] **Step 1: failing tests**

```python
def test_visualize_splat_time_colored_frustums_get_scalar_bar():
    cloud = pv.PolyData(np.random.default_rng(0).normal(size=(50, 3)))
    plotter = visualize_splat(cloud, _walk_poses(), camera_kwargs={"n_poses": 1})
    assert "frame order" in plotter.scalar_bars
    plotter.close()


def test_visualize_splat_fixed_color_frustums_have_no_scalar_bar():
    cloud = pv.PolyData(np.random.default_rng(0).normal(size=(50, 3)))
    plotter = visualize_splat(cloud, _walk_poses(), camera_kwargs={"color": "red"})
    assert "frame order" not in plotter.scalar_bars
    plotter.close()
```

(`pv.OFF_SCREEN = True` at test-module top if the module does not already set it.)

- [ ] **Step 2:** run → first test FAILS.
- [ ] **Step 3: implement** — replace the per-frustum loop:

```python
    # Work on a copy so the caller's dict is not mutated across calls
    cam_kw = dict(camera_kwargs)

    if aligned_cameras is not None:
        n_poses = cam_kw.pop("n_poses", 3)
        scale = cam_kw.pop("scale", 0.02)
        aspect_ratio = cam_kw.pop("aspect_ratio", 1.33)
        fov = cam_kw.pop("fov", 60)
        frustums = []

        for i in range(0, len(aligned_cameras), n_poses):
            frustum = create_camera_frustum_pyvista(
                aligned_cameras[i], scale=scale, aspect_ratio=aspect_ratio, fov=fov
            )
            frustum.point_data["frame order"] = np.full(frustum.n_points, i)
            frustums.append(frustum)

        # One fixed color when given, else viridis by frame order with a scalar bar
        merged = pv.merge(frustums)

        if "color" in cam_kw:
            plotter.add_mesh(merged, scalars=None, **cam_kw)
        else:
            plotter.add_mesh(
                merged,
                scalars="frame order",
                cmap="viridis",
                scalar_bar_args={"title": "frame order"},
                **cam_kw,
            )
```

  Update the docstring bullet: "frustums are viridis by frame order with a scalar bar unless
  camera_kwargs sets color".
- [ ] **Step 4:** run whole `tests/utils/test_visualization.py` → PASS.

### Task 3: `plot_frame_extremes` one row + `max_width`

**Files:** Modify `collab_splats/preproc/viz.py:326-372`; Test `tests/preproc/test_viz.py`.

- [ ] **Step 1: failing test**

```python
def test_plot_frame_extremes_one_row_width_capped(tmp_path, monkeypatch):
    _stub_decode(monkeypatch)
    report = _fake_video_quality_report()
    path = plot_frame_extremes(report, "fake.mp4", tmp_path, n=3, dpi=100, max_width=600)

    with Image.open(path) as img:
        assert img.width == 600
        assert img.height < img.width
```

- [ ] **Step 2:** run → FAIL (unexpected keyword).
- [ ] **Step 3: implement**

```python
def plot_frame_extremes(
    report: dict,
    video_path: str | Path,
    out_dir: str | Path,
    *,
    column: str = "blur",
    n: int = 6,
    dpi: int = 150,
    max_width: int = 900,
) -> Path:
    """
    The n highest and n lowest frames of one per-frame report column, decoded from the video.

    - one row: the n highest values, then the n lowest
    - each thumbnail is captioned with frame index, wall-clock time and the value, so a reader
      can judge what the number means on this footage
    - one seek per thumbnail (2n decodes) — for notebooks and spot checks

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        video_path: the source video the report was computed from.
        out_dir: directory the PNG is written into; created if absent.
        column: any key of report["frames"] except frame_idx — blur, laplacian, exposure_mean, ...
        n: thumbnails per group.
        dpi: PNG resolution; blur has to stay visible in the thumbnails.
        max_width: PNG width in pixels.

    Returns:
        Path to the written PNG.
    """
    # Unpack columns and rank: highest first, then lowest
    frames, video = report["frames"], report["video"]
    fps = video["fps"]
    idx = np.asarray(frames["frame_idx"])
    values = np.asarray(frames[column], dtype=float)
    order = np.argsort(values)
    picks = [("high", i) for i in order[::-1][:n]] + [("low", i) for i in order[:n]]

    # Decode first: the frame aspect sets the figure height
    info = get_video_info(video_path)
    thumbs = [extract_frame(video_path, int(idx[i]), info=info) for _, i in picks]
    aspect = thumbs[0].shape[0] / thumbs[0].shape[1]

    # One row, width fixed by max_width
    fig_width = max_width / dpi
    panel_width = fig_width / len(picks)
    fig, axes = plt.subplots(
        1, len(picks), figsize=(fig_width, panel_width * aspect + 0.9), squeeze=False
    )

    for ax, (group, i), thumb in zip(axes[0], picks, thumbs):
        ax.imshow(thumb)
        ax.set_title(
            f"{group}: {values[i]:.3g}\n#{idx[i]} t={idx[i] / fps:.1f}s", fontsize=6
        )
        ax.axis("off")

    # Title, save and close
    title = f"{column} extremes ({len(idx)} frames)"
    return _save(fig, out_dir, f"extremes-{column}.png", title=title, dpi=dpi)
```

  Title fontsize: `_save` calls `fig.suptitle(title)`; pass nothing extra — if it overflows at
  900 px, shorten the title (already shortened above).
- [ ] **Step 4:** run `tests/preproc/test_viz.py` → PASS (existing dpi/extremes tests too).

### Task 4: library gates

- [ ] `python -m pytest tests/utils tests/preproc tests/test_docstring_contract.py tests/test_import_style.py -q -p no:randomly`
  → all pass (exit code checked, no `| tail`).
- [ ] `make format` scoped: `ruff check --fix` + `ruff format` on the three touched modules and
  two test files only.

### Task 5: Talk2DINO scene extractor

- [ ] `docs/source/tutorials/tutorial.py`: `"semantics": {"extractor": "talk2dino", "max_epochs": 20}`.
- [ ] Grep pages for `maskclip_lifted`, `maskclip_codes`, `MaskCLIPExtractor` and update in
  Tasks 10-12.

### Task 6: Preprocessing page

Cells (lead-in line above each code cell):
1. Title + one paragraph: video in, measured, keyframes out.
2. Imports + scene (unchanged).
3. `## 1. Capture quality` — what the report holds; define in plain words:
   blur (how soft the frame is: higher = blurrier), clipped fraction (share of pixels pure
   black or white), translation (how far the image content moved between two frames, in px),
   parallax (how much the viewpoint changed, which depth needs), matches (features found in both
   frames; few = fast turn or blank view).
4. Load report + kept frames (merge current cells 3).
5. Photometric plot + "look for" line. 6. Motion plot + line. 7. Correlation plot + line.
8. Extremes: `plot_frame_extremes(report, VIDEO_PATH, plots_dir, column="blur", n=2)` (one
   row, 900 px) + line.
9. `## 2. Choosing keyframes`: quality gate in two bullets (sharpness_k, max_clipped_frac),
   sampler bullets one line each; code unchanged; plot + line.
10. `## 3. The keyframe store` + montage + line. 11. `## 4. Lens distortion` two lines.
12. `## In a pipeline run` yaml unchanged, one line above.

- [ ] Rewrite, run page, check extremes image is one row ≈900 px.

### Task 7: Reconstruction page

1. Title: one paragraph.
2. Imports: drop `pandas`? (kept for report table) — drop `create_camera_frustum_pyvista`,
   `pointcloud_to_polydata` stays; add `camera_view`, `visualize_splat`.
3. `## 1. Feedforward reconstruction` — 3 sentences: model predicts depth + camera per frame;
   already run by the scene; `scene.result` loads it. One sentence on the two intrinsics.
4. Code: `result = scene.result`; print points and frame count only.
5. Code: thin + plot:

```python
# Thin the cloud for display, then view it upright from the side
display_cloud = clean_pointcloud(result, remove_outliers=False, max_points=200_000)
cloud = pointcloud_to_polydata(display_cloud.points, rgb=display_cloud.colors)
lo, hi = np.percentile(display_cloud.points, [5, 95], axis=0)
size = np.linalg.norm(hi - lo)

plotter = visualize_splat(
    cloud,
    display_cloud.extrinsics,
    camera_kwargs={"scale": 0.005 * size, "n_poses": 1, "line_width": 2},
    viz_kwargs=camera_view(display_cloud.extrinsics, display_cloud.points),
)
plotter.show()
```

   + line: cameras colored dark→yellow by time; points should form surfaces in front of them.
6. `## 2. Structure from motion` 3-4 sentences; code unchanged; line on depth scale.
7. `## 3. Judging a reconstruction` — no ground truth, so views check each other. Definitions:
   - multiview agreement: share of a frame's pixels whose depth another view confirms within 5%
     (1 = every pixel confirmed)
   - relative depth residual: (depth seen from view B − depth predicted from view A) / depth;
     0 = agree, 0.1 = 10% off
   - parallax: angle between two views of the same point; small parallax = depth poorly pinned
8. Code: report → worst frames table (columns frame_idx, multiview_agreement,
   median_abs_rel_depth_error).
9. SfM tables via `compute_reconstruction_quality` (one lead-in line).
10. Comparison figure: titles "agreement per frame (higher is better)",
    "relative depth residual (narrow at 0 is better)", "depth error vs parallax"; axis labels
    on every panel; legends.
11. Reading guide: 3 short bullets. 12. `## In a pipeline run` yaml + one line.

- [ ] Rewrite, run, confirm upright side view and small viridis frustums with scalar bar.

### Task 8: Refinement page

Variables: `ff_result` (feedforward, dense), `ba_result`, `lc_result`; `ba`, `lc`.
1. Title: 2 sentences.
2. Imports: add `matplotlib.pyplot as plt`, `apply_view`, `camera_view`, `visualize_splat`
   (keep `create_camera_frustum_pyvista`, `pointcloud_to_polydata`).
3. `## 1. Bundle adjustment` — 3 sentences: tracks points across frames with the xfeat matcher;
   adjusts cameras and focal so tracks, depth and colors agree; result reprojected and cleaned.
4. Code (lead-in "Refine the cameras with xfeat tracks"):

```python
work = work_dir("refinement")
pc = scene.config["pointcloud"]
ff_result = PointcloudResult.load_zarr(scene.pointcloud_zarr, load_images=True)
frame_ids = [frames.frame_idx_from_path(p) for p in ff_result.image_paths]
frame_paths = frames.frame_paths(scene.images_dir, frame_ids)

# The stage's BA settings, with xfeat tracks spelled out
terms = {k: v for k, v in pc["bundle_adjustment"].items() if k != "enabled"}
terms["track_source"] = "xfeat"
ba = BundleAdjustment(BundleAdjustmentConfig(**terms, tracks_cache_dir=work))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    extrinsics, intrinsics = ba.refine(
        ff_result.images, ff_result.confidence, ff_result.world_points,
        ff_result.extrinsics, ff_result.model_intrinsics,
        depth=ff_result.depth, frame_paths=frame_paths,
    )

# New cameras, points reprojected under them, then cleaned
ba_result = dataclasses.replace(
    ff_result, extrinsics=extrinsics, model_intrinsics=intrinsics, intrinsics=None
)
ba_result = ba_result.reproject()
ba_result = clean_pointcloud(
    ba_result, remove_outliers=pc["clean"]["enabled"], max_points=pc["max_points"]
)
```

5. Loss curve (lead-in "How the total loss fell during the solve"):

```python
fig, ax = plt.subplots(figsize=(6, 3))

for solve, history in enumerate(ba.loss_history):
    ax.plot(history, label=f"solve {solve + 1}")

ax.set_yscale("log")
ax.set_xlabel("iteration")
ax.set_ylabel("total loss")
ax.set_title("bundle adjustment loss")
ax.legend()
plt.tight_layout()
```

   + line: should drop fast then flatten; photometric pixels are resampled, so small bumps are
   normal.
6. Summary cell: final loss per term, focal before → after, largest camera shift as % of
   trajectory size (merged print cell).
7. Before/after cameras:

```python
# BA cameras colored by time, feedforward cameras grey
cloud = pointcloud_to_polydata(ba_result.points, rgb=ba_result.colors)
lo, hi = np.percentile(ba_result.points, [5, 95], axis=0)
scale = 0.005 * np.linalg.norm(hi - lo)
plotter = visualize_splat(
    cloud,
    ba_result.extrinsics,
    camera_kwargs={"scale": scale, "n_poses": 1, "line_width": 2, "label": "after BA"},
    viz_kwargs=camera_view(ba_result.extrinsics, ba_result.points),
)

for w2c in ff_result.extrinsics:
    frustum = create_camera_frustum_pyvista(w2c, scale=scale)
    plotter.add_mesh(frustum, color="grey", line_width=1)

plotter.add_legend([("before BA", "grey"), ("after BA", "yellow")], bcolor="white")
plotter.show()
```

   (if `label=` breaks scalar-mapped add_mesh, drop it; the explicit legend list carries it.)
8. `## 2. Loop closure` — 3 sentences: overlapping windows, joined by shared frames, revisits
   add loop edges; BA runs in each window.
9. Code: keep `ff_extrinsics = ff_result.extrinsics` and
   `ff_centers = invert_poses(ff_extrinsics)[:, :3, 3]`, free `ff_result`, `ba_result`, `ba`,
   build creator + `lc` + run → `lc_result`.
10. Summary print (frames, window size, loops, per-window focal).
11. Trajectory plot, pyvista side view with start/end markers and a legend:

```python
# Fit the loop-closure path onto the single-pass path (scale, rotation, shift), then draw both
lc_centers = invert_poses(lc_result.extrinsics)[:, :3, 3]
scale, rotation, translation = umeyama_sim3(lc_centers, ff_centers)
lc_aligned = scale * lc_centers @ rotation.T + translation

plotter = pv.Plotter()
plotter.add_mesh(pv.lines_from_points(ff_centers), color="grey", line_width=3, label="single pass")
plotter.add_mesh(pv.lines_from_points(lc_aligned), color="red", line_width=3, label="loop closure")
plotter.add_points(ff_centers[[0]], color="green", point_size=14, render_points_as_spheres=True, label="start")
plotter.add_points(ff_centers[[-1]], color="black", point_size=14, render_points_as_spheres=True, label="end")
plotter.add_legend(bcolor="white")
apply_view(plotter, camera_view(ff_extrinsics))
plotter.show()
```

   (`ff_extrinsics` kept with `ff_centers`.)
12. `## In a pipeline run` yaml + 2 lines.

- [ ] Rewrite, run, check loss curve, legend, upright views.

### Task 9: Train splats page

- Title 2 sentences; `## 1. Configuration` one sentence + config print; DROP sh_degree cells
  4-5 (one sentence: scaffold decodes color with an MLP, so no SH settings).
- `## 2. Training report`: PSNR (pixel match, dB, higher better, ~25+ is good) and SSIM
  (structure match, 0-1) defined; headline + loss table; weakest views table.
- `## 3. Renders`: code unchanged; grid titles `frame {id}` on top row and
  `render, PSNR {p:.1f} dB` on bottom (PSNR from `per_frame` by image id).
- [ ] Rewrite, run.

### Task 10: Mesh page

- Prose: TSDF section 4 sentences; cleaning 3 sentences; texture 2-3 sentences.
- Sky cell: add small thumbnail of one frame with sky overlaid
  (`overlay_masks(rgbs[0], sky[:1])`, figsize (3, 4)).
- Fused/prepared view: `pl = pv.Plotter(shape=(1, 2))`; per subplot add mesh; then
  `apply_view(pl, camera_view(pointcloud.extrinsics, np.asarray(prepared.vertices)))` once
  with `link_views()`.
- DROP the atlas preview cell and its markdown; keep vertex-vs-texture comparison with
  `camera_view` (zoom via `distance=0.6` to get close).
- [ ] Rewrite, run, verify upright.

### Task 11: Feature extraction + segmentation pages

- Feature extraction: `extractors = {"Talk2DINO": ..., "MaskCLIP": ...}`; explain row panels:
  "PCA colors: similar colors = similar features"; "heatmap: viridis, yellow = matches the word";
  "masked: image kept where the score is high"; pipeline yaml `extractor: talk2dino`.
- Segmentation: `Talk2DinoExtractor` for pooling; new `## 3. Sky masks`:

```python
# Sky probability per pixel from the SkyWater segmenter, thresholded at 0.5
sky_dir = work_dir("segmentation")
frame_store = sky_dir / "frames"
frame_store.mkdir()
Image.fromarray(frame).save(frame_store / "frame_000000.png")
sky = sky_masks(frame_store, cache_dir=sky_dir / "sky")[0]

fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=BARE)
ax.imshow(overlay_masks(frame, sky[None]))
ax.set_title(f"sky: {sky.mean():.0%} of pixels")
```

  (`sky_masks` reads a keyframe dir, so the committed frame is written as `frame_000000.png`.)
- [ ] Rewrite both, run both.

### Task 12: Lifting, OCR lens, localization pages

- Lifting: `Talk2DinoExtractor`; store names `talk2dino_lifted.zarr` / `talk2dino_codes.zarr`;
  feature width printed not hard-coded ("768-D" text removed or taken from shape);
  `cmap="viridis"`, `nan_color="lightgrey"`; keep first-keyframe eye view (upright already)
  only if `camera_view` loses the trunks — default to `camera_view(cloud.extrinsics,
  vertices)`; decide by rendering both once and record which in the page.
- OCR: keyframe figure = 1x3: frame | frame + probe heat upsampled (`cv2.resize` to the frame,
  `cmap="viridis"`, alpha 0.6) | frame + best-word labels (alpha 0.55 over the frame) with a
  legend of colored patches (`matplotlib.patches.Patch`) instead of a colorbar; same extent
  everywhere. 3D: `cmap="viridis"`, `nan_color="lightgrey"`, same view choice as lifting.
- Localization: `plot_correspondences(query, ref_image, query_px[inliers], ref_px[inliers],
  warp_corners=True)`; markdown explains the boxes (cyan = query outline in the keyframe,
  yellow = keyframe outline in the query). Scene view via `visualize_splat` + `camera_view`
  with keyframes time-colored and the query frustum red (`create_camera_frustum_pyvista`,
  scale 2x) + legend. If the portrait/landscape pair renders as a thin strip, fix in
  `plot_correspondences` (shared panel height budget) with a test.
- [ ] Rewrite, run each, check images.

### Task 13: Fresh sweep + gates

- [ ] `bash $SP/sweep_fresh.sh` (rm scene, ten pages, cold-start localization) — all rc=0.
- [ ] View every plot image once more.
- [ ] Docs tests (`tests/docs` if present), sphinx build, `du -cb` of the ten notebooks < 32 MiB.
- [ ] Full `python -m pytest tests/utils tests/preproc tests/localization -q`.

### Task 14: Land

- [ ] CHANGELOG entry; CLAUDE.md Recently Completed line.
- [ ] One commit on `clean/tutorials` with `git commit --only <paths>` (library, tests,
  tutorial.py, ten notebooks, plan, CHANGELOG, CLAUDE.md), trailer.
- [ ] Fast-forward `clean/final` to it (main checkout: preserve the user's `README.md` and
  dead-code spec edits; never stash). No push.
