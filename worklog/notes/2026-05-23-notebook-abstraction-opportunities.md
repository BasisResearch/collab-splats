# Notebook Abstraction Opportunities

Full cross-notebook audit of `/docs/source/tutorials/` and `/stage/`. The three patterns already tracked in `docs/superpowers/specs/2026-05-23-notebook-polish-design.md` (`load_or_run`, `to_device`, `pv_plotter_headless`) are excluded.

---

## Patterns Found

### 1. `clean_and_extract_result(result, name)` — O3D clean + conf stats + print

Creates Open3D PointCloud from `FeedforwardResult`, runs `clean_pointcloud()`, extracts `pts3d`/`colors` as numpy, computes conf mean/std with `nan` guards, prints raw→filtered + conf summary.

**Appearances:** `feedforward_methods.ipynb` cells 7 & 13 (vggt vs mapanything variants); `stage/compare-vggtx-infinitevggt.ipynb` (same pattern)
**Lines saved:** ~10/call

```python
def clean_and_extract_result(result: FeedforwardResult, name: str = "") -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (pts3d, colors_uint8, conf_mean, conf_std)."""
```

---

### 2. `show_pointcloud(mesh, point_size, aligned_cameras)` — polydata + pcd_kwargs + visualize_splat boilerplate

Repeated sequence: `pcd_kwargs = MESH_KWARGS.copy()` → update with point_size/render_points_as_spheres/ambient/diffuse/specular → `visualize_splat(mesh=..., mesh_kwargs=pcd_kwargs, viz_kwargs=VIZ_KWARGS)` → `plotter.show()`.

**Appearances:** `03_splats/visualization.ipynb` cell 12; `stage/FeedforwardMeshing.ipynb` cells 28 & 51; `stage/extract-colmap-infinitevggt.ipynb` cell 27; `stage/extract-colmap-mapanything.ipynb` cell 23; `stage/extract-colmap-vggt.ipynb` cell 10; `stage/plot-colmap-vggt-cameras.ipynb` cells 5 & 20
**Lines saved:** ~10/call (8 notebooks, ~8 cells)

```python
def show_pointcloud(mesh, point_size: int = 2, aligned_cameras=None, camera_kwargs=None) -> pv.Plotter:
    """Visualize a point cloud/mesh with standard collab-splats display settings."""
```

---

### 3. `collect_image_paths(image_dir)` — multi-extension image glob

Two idioms: loop over 6 extensions with `extend`; or `sorted(list(dir.glob("*.JPG")) + list(dir.glob("*.png")))`. Both raise/warn if empty. No consistent implementation across notebooks.

**Appearances:** `01_preprocessing/keyframe_extraction.ipynb` cell 18; `02_pointcloud/feedforward_methods.ipynb` cell 3; `04_semantics/feature_extraction.ipynb` cell 3; `04_semantics/maskclip_vs_talk2dino.ipynb` cell 4; `04_semantics/segmentation.ipynb` cell 3; `stage/compare-vggtx-infinitevggt.ipynb` cell 4; `stage/extract-colmap-infinitevggt.ipynb` cell 4
**Lines saved:** 5–8/call (7 notebooks)

```python
def collect_image_paths(image_dir: Path, extensions=("jpg","jpeg","png","JPG","JPEG","PNG")) -> list[Path]:
    """Return sorted image paths; raises ValueError if empty."""
```

---

### 4. `check_nerfstudio_env()` — conda environment guard

Checks `'nerfstudio' not in sys.executable`, prints warning with `conda activate nerfstudio` instructions if wrong env. Inlined differently each time.

**Appearances:** `stage/compare-vggtx-infinitevggt.ipynb` cell 2; `stage/extract-colmap-infinitevggt.ipynb` cell 2; `stage/extract-colmap-mapanything.ipynb` cell 1; `stage/extract-colmap-vggt.ipynb` cells 1 & 12; `stage/plot-colmap-vggt-cameras.ipynb` cell 1
**Lines saved:** ~5/call (5 stage notebooks)

```python
def check_nerfstudio_env(raise_on_missing: bool = False) -> None:
    """Warn (or raise) if not running in the nerfstudio conda environment."""
```

---

### 5. `print_inference_summary(label, n_images, inference_time, n_points, peak_memory_gb)` — performance summary block

Multi-print block with `"="*70` separators reporting dataset name, image count, total time, per-frame time, FPS, peak GPU memory, point count.

**Appearances:** `stage/compare-vggtx-infinitevggt.ipynb` (~7 cells); `stage/extract-colmap-infinitevggt.ipynb` (~8 cells); `stage/extract-colmap-mapanything.ipynb` (~8 cells); `stage/FeedforwardMeshing.ipynb` cell 3
**Lines saved:** 8–12/call (~25 cells total — highest ROI pattern)

```python
def print_inference_summary(label: str, n_images: int, inference_time: float, n_points: int = 0, peak_memory_gb: float | None = None) -> None:
    """Print a standardised ='*70-delimited performance summary block."""
```

---

### 6. `feature_viz_row(axes, frame, features, sim_map, title_prefix)` — PCA+heatmap+masked image triple plot

Fills 3 matplotlib axes with `pca_to_rgb(features[0], frame)`, `compute_heatmap(frame, sim_map)`, `compute_masked_image(frame, sim_map)` with titles and `ax.axis("off")`.

**Appearances:** `04_semantics/feature_extraction.ipynb` cells 7, 11, 13; `04_semantics/maskclip_vs_talk2dino.ipynb` cells 8 & 12
**Lines saved:** 7–8/call (5 cells across 2 notebooks)

```python
def feature_viz_row(axes, frame: np.ndarray, features: torch.Tensor, sim_map: np.ndarray, title_prefix: str = "") -> None:
    """Render PCA→RGB, heatmap, masked-image into a row of 3 matplotlib axes."""
```

---

### 7. `run_creator_pipeline(creator, image_dir, device)` — 4-step feedforward pipeline

Repeated 4-step sequence: `creator.load_model(device=device)` → `creator.setup_inference(image_dir)` → `creator.run_inference()` → `creator.postprocess()` → `result = creator.outputs`.

**Appearances:** `02_pointcloud/bundle_adjustment.ipynb` cell 10; `02_pointcloud/feedforward_methods.ipynb` cells 5 & 11; `02_pointcloud/slam_loop_closure.ipynb` cell 7
**Lines saved:** ~5/call (4 cells across 3 notebooks)

```python
def run_creator_pipeline(creator: BaseFeedforwardCreator, image_dir: Path, device: str) -> FeedforwardResult:
    """Run the standard 4-step feedforward pipeline and return outputs."""
```

---

### 8. `extrinsics_to_c2w(extrinsics)` — list-comprehension inversion

`[np.linalg.inv(ext) for ext in result.extrinsics]` — always appears alongside `pointcloud_to_polydata` + `visualize_splat` calls or camera frustum loops.

**Appearances:** `feedforward_methods.ipynb` cells 9 & 15; `07_localization/localization.ipynb` cell 13
**Lines saved:** 1–4/call (minor, but easy win)

```python
def extrinsics_to_c2w(extrinsics: np.ndarray) -> list[np.ndarray]:
    """Invert world-to-cam extrinsic matrices to camera-to-world."""
```

---

### 9. `add_camera_frustums(plotter, extrinsics, color, scale, line_width)` — frustum loop

`for ext in result.extrinsics: frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05); plotter.add_mesh(frustum, color=..., line_width=2)`. Appears twice in same cell for different models.

**Appearances:** `feedforward_methods.ipynb` cell 19 (twice — vggt + mapanything); `07_localization/localization.ipynb` cell 13
**Lines saved:** 3–4/call

```python
def add_camera_frustums(plotter: pv.Plotter, extrinsics: np.ndarray, color: str = "cornflowerblue", scale: float = 0.05, line_width: int = 2) -> None:
    """Add camera frustum meshes for all extrinsic matrices to a PyVista plotter."""
```

---

## Summary

| # | Helper | Notebooks | Cells | Lines/use | Priority |
|---|--------|-----------|-------|-----------|----------|
| 1 | `clean_and_extract_result` | 2 | 4 | ~10 | High |
| 2 | `show_pointcloud` | 6 | 8 | ~10 | High |
| 3 | `collect_image_paths` | 7 | 7 | 5–8 | High |
| 4 | `check_nerfstudio_env` | 5 | 6 | ~5 | Medium |
| 5 | `print_inference_summary` | 4 | ~25 | 8–12 | **Highest ROI** |
| 6 | `feature_viz_row` | 2 | 5 | 7–8 | Medium |
| 7 | `run_creator_pipeline` | 3 | 4 | ~5 | Medium |
| 8 | `extrinsics_to_c2w` | 2 | 3 | 1–4 | Low |
| 9 | `add_camera_frustums` | 2 | 2 | 3–4 | Low |

**Also see:** `docs/superpowers/specs/2026-05-23-notebook-polish-design.md` §"Repeated patterns worth extracting" for the 3 previously-identified patterns (`load_or_run`, `to_device`, `pv_plotter_headless`).

**Suggested home for helpers:** `collab_splats/notebooks/helpers.py` (or `collab_splats/utils/notebook.py`)
