# VGGT-Omega Notebook Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add VGGT-Omega as §9–§11 in `feedforward_methods.ipynb`, extend §7→§12 and §8→§13 for 3-way comparison.

**Architecture:** Direct notebook JSON manipulation using a Python helper script that loads the `.ipynb`, splices in new cells at the correct index, and overwrites the file. No library code changes. All edits isolated to one notebook file.

**Tech Stack:** Python (json stdlib), Jupyter notebook format v4

---

## File Map

| Action | File |
|--------|------|
| Modify | `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` |

---

### Task 1: Update imports and intro cell

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

The notebook uses direct string-replacement edits via the Edit tool. Both the intro markdown (cell-0) and the imports code cell (cell-2) need updating before new sections are added.

- [ ] **Step 1: Add VGGTOmegaCreator to imports cell**

In `feedforward_methods.ipynb`, find the imports cell (cell-2). Replace:

```
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
```

With:

```
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
```

- [ ] **Step 2: Update intro markdown (cell-0)**

Replace:

```
This tutorial runs **VGGT-X** and **MapAnything** on keyframes extracted from a real video, then compares the resulting pointclouds side by side. Both models run feedforward inference (no optimisation loop) and write results to zarr caches that downstream notebooks — `semantic_lifting`, `localization`, and `bundle_adjustment` — depend on. **Run this notebook before any of those.**
```

With:

```
This tutorial runs **VGGT-X**, **MapAnything**, and **VGGT-Omega** on keyframes extracted from a real video, then compares the resulting pointclouds side by side. All three models run feedforward inference (no optimisation loop) and write results to zarr caches that downstream notebooks — `semantic_lifting`, `localization`, and `bundle_adjustment` — depend on. **Run this notebook before any of those.**
```

- [ ] **Step 3: Verify JSON is valid**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
print('cells:', len(nb['cells']))
# confirm import cell contains VGGTOmegaCreator
src = ''.join(nb['cells'][2]['source'])
assert 'VGGTOmegaCreator' in src, 'import missing'
print('import OK')
"
```

Expected output:
```
cells: 20
import OK
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): add VGGTOmegaCreator import + update intro"
```

---

### Task 2: Insert §9–§11 VGGT-Omega cells

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

Six cells are inserted after the MapAnything viewer cell (currently index 15 / id `69212a84`) and before the comparison section (currently index 16). Use a Python script to splice them in.

- [ ] **Step 1: Run insertion script**

```bash
/opt/conda/envs/reconstruction/bin/python - << 'EOF'
import json, pathlib

NB = pathlib.Path("docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb")
nb = json.loads(NB.read_text())

new_cells = [
    {
        "cell_type": "markdown",
        "id": "cell-md-s9",
        "metadata": {},
        "source": [
            "## §9 — VGGT-Omega Reconstruction\n",
            "\n",
            "Loads a cached VGGT-Omega reconstruction from zarr if available, skipping GPU inference (~3–5 min on GPU). "
            "VGGT-Omega is a third feedforward method that uses a unified vision transformer for depth and pose estimation. "
            "Runs with default settings (`VGGTOmegaCreator()`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-omega-run",
        "metadata": {},
        "outputs": [],
        "source": [
            "_omega_cache = CACHE_DIR / \"omega\" / \"reconstruction.zarr\"\n",
            "_omega_cache.parent.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "if _omega_cache.exists():\n",
            "    result_omega = FeedforwardResult.load_zarr(_omega_cache)\n",
            "    print(f\"Loaded VGGT-Omega result from cache  ({result_omega.points.shape[0]:,} pts)\")\n",
            "else:\n",
            "    result_omega = VGGTOmegaCreator().run(IMAGES, device)\n",
            "    result_omega.save_zarr(_omega_cache)\n",
            "    print(f\"VGGT-Omega done  →  saved to {_omega_cache}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "cell-md-s10",
        "metadata": {},
        "source": [
            "## §10 — VGGT-Omega Post-processing and Visualisation\n",
            "\n",
            "Applies the same outlier removal and voxel downsampling pipeline as §2/§5. "
            "Confidence metrics are collected for the three-way comparison table."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-omega-postproc",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clean via Open3D statistical outlier removal + voxel downsampling\n",
            "_pcd = o3d.geometry.PointCloud()\n",
            "_pcd.points = o3d.utility.Vector3dVector(result_omega.points)\n",
            "_pcd.colors = o3d.utility.Vector3dVector(result_omega.colors.astype(np.float64) / 255.0)\n",
            "_cleaned, _ = clean_pointcloud(_pcd)\n",
            "\n",
            "pts3d_omega = np.asarray(_cleaned.points, dtype=np.float32)\n",
            "colors_omega = (np.asarray(_cleaned.colors) * 255).astype(np.uint8)\n",
            "\n",
            "conf_omega_mean = result_omega.confidence.cpu().float().mean().item() if result_omega.confidence is not None else float(\"nan\")\n",
            "conf_omega_std  = result_omega.confidence.cpu().float().std().item()  if result_omega.confidence is not None else float(\"nan\")\n",
            "\n",
            "print(f\"Points: {len(result_omega.points):,} raw → {len(pts3d_omega):,} filtered\")\n",
            "print(f\"Conf:   mean={conf_omega_mean:.3f}  std={conf_omega_std:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "cell-md-s11",
        "metadata": {},
        "source": [
            "## §11 — VGGT-Omega Pointcloud Viewer\n",
            "\n",
            "Renders the filtered VGGT-Omega pointcloud with camera frustums. "
            "Green frustums show predicted camera poses; compare pose spread vs. VGGT-X (§3) and MapAnything (§6) above."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-omega-viewer",
        "metadata": {},
        "outputs": [],
        "source": [
            "cloud_omega = pointcloud_to_polydata(pts3d_omega, RGB=colors_omega)\n",
            "c2w_omega = [np.linalg.inv(ext) for ext in result_omega.extrinsics]\n",
            "pl = visualize_splat(cloud_omega, aligned_cameras=c2w_omega)\n",
            "pl.show()"
        ]
    },
]

# Insert after MapAnything viewer cell (id 69212a84)
insert_after_id = "69212a84"
idx = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == insert_after_id)
nb["cells"][idx + 1:idx + 1] = new_cells

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Done. cells: {len(nb['cells'])}")
EOF
```

Expected output:
```
Done. cells: 26
```

- [ ] **Step 2: Verify cell order**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))[:60].replace('\n',' ')
    print(f'{i:>2} [{c[\"cell_type\"][:4]}] {c.get(\"id\",\"?\")} | {src}')
"
```

Expected: cells 16–21 should be the 6 new Omega cells (`cell-md-s9`, `cell-omega-run`, `cell-md-s10`, `cell-omega-postproc`, `cell-md-s11`, `cell-omega-viewer`), followed by `cell-md-s7` (comparison) at index 22.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): add VGGT-Omega §9-§11 reconstruction/post-proc/viewer cells"
```

---

### Task 3: Update §7→§12 comparison and §8→§13 camera overlay

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb`

Two existing cells need content updates: the comparison markdown+code (renumber to §12) and the camera overlay markdown+code (renumber to §13).

- [ ] **Step 1: Update §7 markdown → §12**

In `feedforward_methods.ipynb`, find cell id `cell-md-s7`. Replace its source:

Old:
```
## §7 — Side-by-side Comparison

Tabulates raw vs. filtered point counts and confidence statistics for both models. Higher confidence mean with lower std indicates more reliable depth predictions.
```

New:
```
## §12 — Side-by-side Comparison

Tabulates raw vs. filtered point counts and confidence statistics for all three models. Higher confidence mean with lower std indicates more reliable depth predictions.
```

- [ ] **Step 2: Update §7 code cell → §12 (add Omega row)**

Find cell id `cell-11`. Replace its source:

Old:
```python
col = 14
print(f"{'Model':<{col}} {'Pts raw':>10} {'Pts filt':>10} {'Conf mean':>10} {'Conf std':>10}")
print("-" * (col + 42))
print(f"{'VGGT-X':<{col}} {len(result_vggt.points):>10,} {len(pts3d_vggt):>10,} {conf_vggt_mean:>10.3f} {conf_vggt_std:>10.3f}")
print(f"{'MapAnything':<{col}} {len(result_ma.points):>10,} {len(pts3d_ma):>10,} {conf_ma_mean:>10.3f} {conf_ma_std:>10.3f}")
```

New:
```python
col = 14
print(f"{'Model':<{col}} {'Pts raw':>10} {'Pts filt':>10} {'Conf mean':>10} {'Conf std':>10}")
print("-" * (col + 42))
print(f"{'VGGT-X':<{col}} {len(result_vggt.points):>10,} {len(pts3d_vggt):>10,} {conf_vggt_mean:>10.3f} {conf_vggt_std:>10.3f}")
print(f"{'MapAnything':<{col}} {len(result_ma.points):>10,} {len(pts3d_ma):>10,} {conf_ma_mean:>10.3f} {conf_ma_std:>10.3f}")
print(f"{'VGGT-Omega':<{col}} {len(result_omega.points):>10,} {len(pts3d_omega):>10,} {conf_omega_mean:>10.3f} {conf_omega_std:>10.3f}")
```

- [ ] **Step 3: Update §8 markdown → §13**

Find cell id `cell-md-s8`. Replace its source:

Old:
```
## §8 — Camera Pose Overlay

Overlays camera frustums from both models in a single scene — blue for VGGT-X, orange for MapAnything. Alignment between the two sets indicates consistent global pose estimation.
```

New:
```
## §13 — Camera Pose Overlay

Overlays camera frustums from all three models in a single scene — blue for VGGT-X, orange for MapAnything, green for VGGT-Omega. Alignment across the three sets indicates consistent global pose estimation.
```

- [ ] **Step 4: Update §8 code cell → §13 (add Omega frustums)**

Find cell id `cell-12`. Replace its source:

Old:
```python
pl = pv.Plotter()
for ext in result_vggt.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="cornflowerblue", line_width=2)
for ext in result_ma.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="darkorange", line_width=2)
pl.add_axes()
pl.show()
```

New:
```python
pl = pv.Plotter()
for ext in result_vggt.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="cornflowerblue", line_width=2)
for ext in result_ma.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="darkorange", line_width=2)
for ext in result_omega.extrinsics:
    frustum = create_camera_frustum_pyvista(np.linalg.inv(ext), scale=0.05)
    pl.add_mesh(frustum, color="mediumseagreen", line_width=2)
pl.add_axes()
pl.show()
```

- [ ] **Step 5: Verify final cell count and section headings**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json, re
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
print('total cells:', len(nb['cells']))
for c in nb['cells']:
    if c['cell_type'] == 'markdown':
        src = ''.join(c['source'])
        m = re.search(r'^## (§\d+)', src)
        if m:
            print(m.group(1), '|', src[:60].replace('\n',' '))
"
```

Expected:
```
total cells: 26
§1  | ## §1 — VGGT-X Reconstruction
§2  | ## §2 — VGGT-X Post-processing and Visualisation
§3  | ## §3 — VGGT-X Pointcloud Viewer
§4  | ## §4 — MapAnything Reconstruction
§5  | ## §5 — MapAnything Post-processing and Visualisation
§6  | ## §6 — MapAnything Pointcloud Viewer
§9  | ## §9 — VGGT-Omega Reconstruction
§10 | ## §10 — VGGT-Omega Post-processing and Visualisation
§11 | ## §11 — VGGT-Omega Pointcloud Viewer
§12 | ## §12 — Side-by-side Comparison
§13 | ## §13 — Camera Pose Overlay
```

- [ ] **Step 6: Verify JSON is valid**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb'))
print('JSON valid, cells:', len(nb['cells']))
# spot-check Omega row in comparison cell
src = ''.join(next(c for c in nb['cells'] if c.get('id') == 'cell-11')['source'])
assert 'VGGT-Omega' in src, 'Omega row missing from comparison'
# spot-check green frustum in overlay cell
src = ''.join(next(c for c in nb['cells'] if c.get('id') == 'cell-12')['source'])
assert 'mediumseagreen' in src, 'green frustum missing from overlay'
print('content checks OK')
"
```

Expected:
```
JSON valid, cells: 26
content checks OK
```

- [ ] **Step 7: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb
git commit -m "docs(feedforward): extend comparison §12 and overlay §13 for VGGT-Omega"
```
