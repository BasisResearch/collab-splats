# Pointcloud Feedforward Exploration Notebook — Design Spec

## Problem

The pointcloud module exposes two feedforward creators (`VGGTXCreator`, `MapAnythingCreator`) but has no interactive exploration tooling. Developers have no quick way to run inference, inspect results, clean the pointcloud, and visualize it without writing a full script.

## Goal

A Jupyter notebook at `docs/pointcloud/feedforward_exploration.ipynb` that mirrors the structure of `docs/semantics/feature_extraction.ipynb`: one config cell selects the dataset, then each method gets its own section (inference → clean → visualize).

## Dataset Support

| Dataset | Source | Notes |
|---------|--------|-------|
| `bicycle` | `/workspace/bicycle/images_4` | Static images, ready to use |
| `c0043` | `/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4` | Video; extract frames at 1 FPS via `sample_frames_fps` |

`DATASET = "bicycle"` or `"c0043"` in a single config cell. Both paths resolved to `image_dir` before any method section runs.

## Inference Pipeline

**Do not call `reconstruct()`** — it runs the full COLMAP pipeline and writes to disk. Use the step-by-step API instead:

```python
creator.load_model()
creator.setup_inference(image_dir)
creator.run_inference()
creator.postprocess()
result = creator.outputs  # FeedforwardResult
```

`FeedforwardResult` fields used:
- `pts3d: (P, 3) float32` — world-space XYZ
- `colors: (P, 3) uint8` — RGB [0, 255]
- `extrinsics: (N, 3, 4) float32` — world-to-camera [R|t]
- `intrinsics: (N, 3, 3) float32` — camera K matrices

## Pointcloud Cleaning

`clean_pointcloud` takes an **Open3D PointCloud**. Conversion pattern:

```python
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(result.pts3d)
pcd.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64) / 255.0)
cleaned_pcd, _ = clean_pointcloud(pcd)
pts3d = np.asarray(cleaned_pcd.points, dtype=np.float32)
colors = (np.asarray(cleaned_pcd.colors) * 255).astype(np.uint8)
```

## Visualization

A **notebook-local** `visualize_feedforward_result(pts3d, colors, extrinsics, intrinsics, ...)` helper renders:
1. Point cloud — `pv.PolyData` with RGB scalars
2. Camera frustums — one per image, via `create_camera_frustum_pyvista` (imported from `collab_splats.utils.visualization`)

`create_camera_frustum_pyvista(pose, scale, aspect_ratio, fov)` expects a **4×4 cam-to-world** matrix. Extrinsics are world-to-cam (N,3,4), so invert per camera:

```python
R, t = ext[:3, :3], ext[:3, 3]
c2w = np.eye(4)
c2w[:3, :3] = R.T
c2w[:3, 3] = -R.T @ t
```

PyVista backend: `pv.set_jupyter_backend("trame")` in setup.

## Notebook Structure

| Section | Cells |
|---------|-------|
| §0 Setup | autoreload, imports, device, `visualize_feedforward_result` helper |
| §1 Data Configuration | `DATASET` toggle, path resolution, frame extraction (if c0043) |
| §2 VGGTXCreator | load → infer → postprocess → inspect → clean → visualize |
| §3 MapAnythingCreator | load → infer → postprocess → inspect → clean → visualize |

## Files

| Path | Action |
|------|--------|
| `docs/pointcloud/feedforward_exploration.ipynb` | Create |
| `worklog/history/specs/2026-04-22-pointcloud-feedforward-notebook-design.md` | Create (this file) |
| `collab_splats/utils/visualization.py` | No change — `create_camera_frustum_pyvista` reused as-is |
