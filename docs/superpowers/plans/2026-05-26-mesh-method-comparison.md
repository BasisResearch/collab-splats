# Mesh Method Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comparison notebook that loads the cached C0043 VGGT-Omega result and tests TSDF (tuned params), image-grid (world_points), and image-grid (pixel_indices) meshing side-by-side with PyVista visualization, to empirically pick the best outdoor meshing approach.

**Architecture:** Add `o3d_mesh_to_polydata` bridge to `visualization.py`, then build a stage notebook with inline mesh helpers (not promoted to library yet) that loads from zarr, runs each method, and calls `visualize_splat` per result.

**Tech Stack:** open3d, numpy, pyvista, PIL, scipy, `collab_splats.mesh.utils.find_depth_edges`, `collab_splats.utils.visualization.visualize_splat`, `/opt/conda/envs/nerfstudio/bin/python`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/utils/visualization.py` | Modify | Add `o3d_mesh_to_polydata` function |
| `tests/utils/test_o3d_mesh_to_polydata.py` | Create | Unit tests for the bridge function |
| `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb` | Create | Comparison notebook |

---

### Task 1: Add `o3d_mesh_to_polydata` to visualization.py (TDD)

**Files:**
- Modify: `collab_splats/utils/visualization.py` (append after `pointcloud_to_polydata`)
- Create: `tests/utils/test_o3d_mesh_to_polydata.py`

- [ ] **Step 1: Create test file**

```python
# tests/utils/test_o3d_mesh_to_polydata.py
from __future__ import annotations

import numpy as np
import open3d as o3d
import pyvista as pv

from collab_splats.utils.visualization import o3d_mesh_to_polydata


def _make_o3d_mesh(with_colors: bool = True) -> o3d.geometry.TriangleMesh:
    """Minimal 2-triangle mesh (4 vertices, 2 faces)."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if with_colors:
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def test_returns_polydata():
    mesh = _make_o3d_mesh()
    pd = o3d_mesh_to_polydata(mesh)
    assert isinstance(pd, pv.PolyData)


def test_vertex_count():
    mesh = _make_o3d_mesh()
    pd = o3d_mesh_to_polydata(mesh)
    assert pd.n_points == 4


def test_face_count():
    mesh = _make_o3d_mesh()
    pd = o3d_mesh_to_polydata(mesh)
    assert pd.n_cells == 2


def test_rgb_scalar_present_when_colors():
    mesh = _make_o3d_mesh(with_colors=True)
    pd = o3d_mesh_to_polydata(mesh)
    assert "RGB" in pd.point_data


def test_rgb_dtype_uint8():
    mesh = _make_o3d_mesh(with_colors=True)
    pd = o3d_mesh_to_polydata(mesh)
    assert pd.point_data["RGB"].dtype == np.uint8


def test_no_rgb_when_no_colors():
    mesh = _make_o3d_mesh(with_colors=False)
    pd = o3d_mesh_to_polydata(mesh)
    assert "RGB" not in pd.point_data


def test_vertex_positions_preserved():
    mesh = _make_o3d_mesh()
    pd = o3d_mesh_to_polydata(mesh)
    expected = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_allclose(pd.points, expected, atol=1e-6)
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_o3d_mesh_to_polydata.py -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'o3d_mesh_to_polydata'`

- [ ] **Step 3: Implement `o3d_mesh_to_polydata` in visualization.py**

Append after the `pointcloud_to_polydata` function (after line ~420):

```python
def o3d_mesh_to_polydata(mesh: "o3d.geometry.TriangleMesh") -> pv.PolyData:
    """Convert Open3D TriangleMesh to PyVista PolyData with RGB vertex scalars.

    Compatible with visualize_splat when mesh has vertex colors (MESH_KWARGS applies).
    """
    import open3d as o3d  # noqa: F401 — type-check guard
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    # PyVista face format: [3, v0, v1, v2,  3, v0, v1, v2, ...]
    pv_faces = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int32), faces]
    ).ravel()
    pd = pv.PolyData(verts, pv_faces)
    if mesh.has_vertex_colors():
        pd["RGB"] = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    return pd
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_o3d_mesh_to_polydata.py -v 2>&1 | tail -15
```

Expected: 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/visualization.py tests/utils/test_o3d_mesh_to_polydata.py
git commit -m "feat(viz): add o3d_mesh_to_polydata — Open3D TriangleMesh → PyVista PolyData bridge"
```

---

### Task 2: Write comparison notebook §0–§2 (setup, load, inspect)

**Files:**
- Create: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`

- [ ] **Step 1: Create notebook with §0 Setup cell**

Write the file using Python nbformat. Run:

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'EOF'
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# §0 Setup
cells.append(nbf.v4.new_markdown_cell("## §0 — Setup"))
cells.append(nbf.v4.new_code_cell("""\
from __future__ import annotations
from pathlib import Path
import numpy as np
import open3d as o3d
import pyvista as pv
from PIL import Image as PILImage

from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.mesh.tsdf import Open3DTSDFFusion
from collab_splats.mesh.utils import find_depth_edges
from collab_splats.utils.visualization import (
    visualize_splat, o3d_mesh_to_polydata,
    MESH_KWARGS, VIZ_KWARGS, CAMERA_KWARGS,
)

CACHE_DIR  = Path("../../.cache/birds_c0043")
OMEGA_ZARR = CACHE_DIR / "omega" / "reconstruction.zarr"
OUTPUT_DIR = Path("/tmp/mesh_compare")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Omega zarr exists: {OMEGA_ZARR.exists()}")
"""))

nb.cells = cells
out = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"Written {out}")
EOF
```

- [ ] **Step 2: Add §1 Load cell and §2 Inspect cell directly via nbformat**

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'PYEOF'
import nbformat as nbf
from pathlib import Path

path = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
nb = nbf.read(str(path), as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("## §1 — Load Omega Result"))
nb.cells.append(nbf.v4.new_code_cell("""\
result = FeedforwardResult.load_zarr(OMEGA_ZARR)

print(f"points:        {result.points.shape}")
print(f"colors:        {result.colors.shape}")
print(f"extrinsics:    {result.extrinsics.shape}")
print(f"intrinsics:    {result.intrinsics.shape}")
print(f"model_size:    {result.model_height} x {result.model_width}")
print(f"world_points:  {result.world_points.shape if result.world_points is not None else None}")
print(f"depth:         {result.depth.shape if result.depth is not None else None}")
print(f"confidence:    {result.confidence.shape if result.confidence is not None else None}")
print(f"pixel_indices: {result.pixel_indices.shape if result.pixel_indices is not None else None}")
print(f"image_paths[0]: {result.image_paths[0]}")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("## §2 — Inspect Depth + Confidence"))
nb.cells.append(nbf.v4.new_code_cell("""\
import matplotlib.pyplot as plt

depth = result.depth  # (N, H, W)
conf  = result.confidence.numpy() if hasattr(result.confidence, 'numpy') else result.confidence

# World-space bounding box
pts = result.points
bbox_min, bbox_max = pts.min(0), pts.max(0)
extent = bbox_max - bbox_min
print(f"Scene bbox:  {bbox_min} → {bbox_max}")
print(f"Extent (m):  {extent}  (max={extent.max():.1f}m)")
print()
print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] m")
print(f"Depth 95th pct: {np.percentile(depth, 95):.2f} m")
print(f"Conf  range: [{conf.min():.4f}, {conf.max():.4f}]")
print(f"Conf  median: {np.median(conf):.4f}")
print()
print("Suggested outdoor TSDF params based on extent:")
voxel = float(extent.max()) / 200.0
sdf_trunc = voxel * 5
print(f"  voxel_size={voxel:.3f}  sdf_trunc={sdf_trunc:.3f}  depth_trunc={depth.max()*1.1:.1f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(depth[0], cmap='plasma'); axes[0].set_title("Depth frame 0")
axes[1].imshow(conf[0], cmap='viridis'); axes[1].set_title("Confidence frame 0")
axes[2].hist(depth.ravel(), bins=100, log=True); axes[2].set_title("Depth distribution")
plt.tight_layout(); plt.show()
"""))

nbf.write(nb, str(path))
print("§1 and §2 added")
PYEOF
```

- [ ] **Step 3: Verify notebook is valid JSON**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json; json.load(open('docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb'))
print('valid JSON')
"
```

Expected: `valid JSON`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git commit -m "docs(mesh): start mesh_method_comparison notebook — §0 setup + §1 load + §2 inspect"
```

---

### Task 3: Add §3 Denoising comparison

**Files:**
- Modify: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`

- [ ] **Step 1: Append §3 cell**

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'PYEOF'
import nbformat as nbf
from pathlib import Path

path = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
nb = nbf.read(str(path), as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""\
## §3 — Denoising Variants

Compare three denoising approaches on the raw result:
- **A (raw):** result.points unmodified
- **B (tier-2):** conf + depth-edge + depth-trunc mask on world_points — what VGGT-Omega's own viz uses
- **C (tier-3):** statistical outlier removal on result.points with aligned pixel_indices
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# ── Variant A: raw ────────────────────────────────────────────────────────────
pts_A   = result.points.copy()
colors_A = result.colors.copy()
print(f"A (raw):           {len(pts_A):,} points")

# ── Variant B: conf + depth-edge + depth-trunc mask ──────────────────────────
CONF_PERCENTILE_B = 30.0   # keep top 70% of confident pixels
DEPTH_TRUNC_B     = float(np.percentile(result.depth, 99)) * 1.1  # clip sky/far

depth_b = result.depth  # (N, H, W)
conf_b  = result.confidence.numpy().copy()  # (N, H, W)

for i in range(len(result.image_paths)):
    conf_b[i][find_depth_edges(depth_b[i])] = 0.0
conf_b[depth_b > DEPTH_TRUNC_B] = 0.0
valid_conf = conf_b[conf_b > 0]
thresh_b = np.percentile(valid_conf, CONF_PERCENTILE_B) if len(valid_conf) else 0.0
mask_b = conf_b > thresh_b  # (N, H, W) bool

pts_B   = result.world_points[mask_b].astype(np.float32)
# Load RGB for masked pixels
imgs_b = []
N, H, W = result.world_points.shape[:3]
for i, p in enumerate(result.image_paths):
    img = PILImage.open(p).convert("RGB").resize((W, H), PILImage.BILINEAR)
    imgs_b.append(np.asarray(img, dtype=np.uint8))
imgs_b = np.stack(imgs_b)  # (N, H, W, 3)
colors_B = imgs_b[mask_b]

print(f"B (conf+edge+trunc): {len(pts_B):,} points  (depth_trunc={DEPTH_TRUNC_B:.1f}m, conf_pct={CONF_PERCENTILE_B})")

# ── Variant C: statistical outlier removal ────────────────────────────────────
pcd_c = o3d.geometry.PointCloud()
pcd_c.points = o3d.utility.Vector3dVector(result.points)
pcd_c.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64)/255.0)
_, ind_c = pcd_c.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
ind_c = np.asarray(ind_c, dtype=np.int64)

pts_C    = result.points[ind_c]
colors_C = result.colors[ind_c]
# Keep pixel_indices aligned for §6 pixel_indices mesh
pix_C    = result.pixel_indices[ind_c] if result.pixel_indices is not None else None

print(f"C (stat outlier):    {len(pts_C):,} points  ({len(result.points)-len(pts_C):,} removed)")
"""))

nbf.write(nb, str(path))
print("§3 added")
PYEOF
```

- [ ] **Step 2: Verify JSON valid**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json; json.load(open('docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb'))
print('valid')
"
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git commit -m "docs(mesh): §3 denoising variants — raw / conf+edge+trunc / stat-outlier"
```

---

### Task 4: Add §4 TSDF tuned

**Files:**
- Modify: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`

- [ ] **Step 1: Append §4 cell**

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'PYEOF'
import nbformat as nbf
from pathlib import Path

path = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
nb = nbf.read(str(path), as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""\
## §4 — TSDF (outdoor-tuned params)

Open3DTSDFFusion with parameters scaled to the actual scene extent.
Default params (`voxel=0.02m, sdf_trunc=0.08m, depth_trunc=10m`) are indoor-calibrated and
clip most outdoor geometry. We scale to scene extent from §2.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs  # internal helper

# Compute scene-adaptive params (derived from §2 printout)
_extent_max = float((result.points.max(0) - result.points.min(0)).max())
_voxel_size  = max(0.05, _extent_max / 200.0)   # ~200 voxels across longest axis
_sdf_trunc   = _voxel_size * 5.0                 # 5-voxel band
_depth_trunc = float(np.percentile(result.depth, 99)) * 1.2

print(f"Outdoor TSDF params:")
print(f"  voxel_size  = {_voxel_size:.3f} m")
print(f"  sdf_trunc   = {_sdf_trunc:.3f} m")
print(f"  depth_trunc = {_depth_trunc:.1f} m")

_tsdf_dir = OUTPUT_DIR / "tsdf_tuned"
mesher_tsdf = Open3DTSDFFusion(
    output_dir=_tsdf_dir,
    voxel_size=_voxel_size,
    sdf_trunc=_sdf_trunc,
    depth_trunc=_depth_trunc,
    clean_repair=False,   # raw output first — see what TSDF produces before cleanup
)

from collab_splats.mesh.utils import pointcloud_to_mesh
mesh_result_tsdf = pointcloud_to_mesh(result, _tsdf_dir,
                                       method="open3d_tsdf",
                                       voxel_size=_voxel_size,
                                       sdf_trunc=_sdf_trunc,
                                       depth_trunc=_depth_trunc,
                                       clean_repair=False)

mesh_tsdf = o3d.io.read_triangle_mesh(str(mesh_result_tsdf.mesh_path))
print(f"TSDF mesh: {len(mesh_tsdf.vertices):,} verts, {len(mesh_tsdf.triangles):,} triangles")
"""))

nbf.write(nb, str(path))
print("§4 added")
PYEOF
```

- [ ] **Step 2: Verify JSON valid**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json; json.load(open('docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb'))
print('valid')
"
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git commit -m "docs(mesh): §4 TSDF outdoor-tuned params (voxel/sdf_trunc/depth_trunc scaled to scene)"
```

---

### Task 5: Add §5 Image-grid from world_points

**Files:**
- Modify: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`

- [ ] **Step 1: Append §5 cells**

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'PYEOF'
import nbformat as nbf
from pathlib import Path

path = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
nb = nbf.read(str(path), as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""\
## §5 — Image-grid Mesh (world_points, full grid)

Per-frame quads from the full N×H×W world_points grid.
Quad validity = all 4 corners pass the tier-2 mask (conf + depth-edge + depth-trunc).
Vertices are NOT exactly result.points — this uses the full dense grid.
Comparable to MapAnything's predictions_to_glb approach.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
def _image_grid_from_world_points(result, conf_percentile=30.0,
                                   depth_trunc_m=None, edge_threshold=0.01):
    \"\"\"Image-grid mesh from full world_points grid with conf+edge+trunc masking.\"\"\"
    N, H, W, _ = result.world_points.shape
    depth = result.depth  # (N, H, W)
    conf  = result.confidence.numpy().copy()  # (N, H, W)

    if depth_trunc_m is None:
        depth_trunc_m = float(np.percentile(depth, 99)) * 1.1

    for i in range(N):
        conf[i][find_depth_edges(depth[i], threshold=edge_threshold)] = 0.0
    conf[depth > depth_trunc_m] = 0.0
    valid_conf = conf[conf > 0]
    threshold = np.percentile(valid_conf, conf_percentile) if len(valid_conf) else 0.0
    mask = conf > threshold  # (N, H, W) bool

    all_verts, all_colors, all_faces = [], [], []
    offset = 0
    for i in range(N):
        m = mask[i]  # (H, W)
        idx = np.full((H, W), -1, dtype=np.int64)
        valid_pixels = np.argwhere(m)  # (P, 2) [row, col]
        if len(valid_pixels) == 0:
            continue
        idx[valid_pixels[:, 0], valid_pixels[:, 1]] = np.arange(len(valid_pixels)) + offset

        all_verts.append(result.world_points[i][m].astype(np.float32))
        img = PILImage.open(result.image_paths[i]).convert("RGB").resize((W, H), PILImage.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)
        all_colors.append(rgb[m])

        quad_mask = (
            (idx[:-1, :-1] >= 0) & (idx[1:, :-1] >= 0) &
            (idx[1:,  1:] >= 0) & (idx[:-1, 1:] >= 0)
        )
        v00 = idx[:-1, :-1][quad_mask]
        v10 = idx[1:,  :-1][quad_mask]
        v11 = idx[1:,   1:][quad_mask]
        v01 = idx[:-1,  1:][quad_mask]
        tris = np.concatenate([
            np.stack([v00, v10, v11], 1),
            np.stack([v00, v11, v01], 1),
        ])
        all_faces.append(tris)
        offset += len(valid_pixels)

    verts  = np.concatenate(all_verts)
    colors = np.concatenate(all_colors)
    faces  = np.concatenate(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles     = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return mesh

mesh_wpts = _image_grid_from_world_points(result, conf_percentile=30.0)
print(f"image-grid (world_points): {len(mesh_wpts.vertices):,} verts, {len(mesh_wpts.triangles):,} triangles")
"""))

nbf.write(nb, str(path))
print("§5 added")
PYEOF
```

- [ ] **Step 2: Verify JSON valid**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json; json.load(open('docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb'))
print('valid')
"
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git commit -m "docs(mesh): §5 image-grid from world_points — full grid, conf+edge+trunc quad mask"
```

---

### Task 6: Add §6 Image-grid from pixel_indices

**Files:**
- Modify: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`

- [ ] **Step 1: Append §6 cells**

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'PYEOF'
import nbformat as nbf
from pathlib import Path

path = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
nb = nbf.read(str(path), as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""\
## §6 — Image-grid Mesh (pixel_indices, pcd-aligned)

Vertices === result.points. Topology from pixel_indices lookup.
Quad validity = all 4 corners survived the creator's conf filter AND no depth edge.
This preserves exact pcd↔mesh correspondence — downstream feature lifting works on both.

Note: if pixel_indices is None in this zarr, cell prints a warning and skips.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
def _image_grid_from_pixel_indices(result, edge_threshold=0.01):
    \"\"\"Image-grid mesh where vertices == result.points. Topology from pixel_indices.\"\"\"
    if result.pixel_indices is None:
        print("WARNING: pixel_indices not in zarr — skipping §6. Re-run creator to populate.")
        return None

    N = len(result.image_paths)
    H, W = result.model_height, result.model_width
    depth = result.depth  # (N, H, W)

    # pixel → point-index lookup; -1 = filtered out by creator
    pixel_to_point = np.full((N, H, W), -1, dtype=np.int64)
    fi = result.pixel_indices[:, 0]
    ri = result.pixel_indices[:, 1]
    ci = result.pixel_indices[:, 2]
    pixel_to_point[fi, ri, ci] = np.arange(len(result.points))

    all_faces = []
    for i in range(N):
        idx = pixel_to_point[i]  # (H, W)
        valid = idx >= 0

        # Mask quads spanning depth discontinuities
        if depth is not None:
            edge_mask = find_depth_edges(depth[i], threshold=edge_threshold)
            no_edge = ~(
                edge_mask[:-1, :-1] | edge_mask[1:, :-1] |
                edge_mask[1:,  1:] | edge_mask[:-1, 1:]
            )
        else:
            no_edge = np.ones((H-1, W-1), dtype=bool)

        quad_mask = (
            valid[:-1, :-1] & valid[1:, :-1] &
            valid[1:,  1:] & valid[:-1, 1:] & no_edge
        )
        v00 = idx[:-1, :-1][quad_mask]
        v10 = idx[1:,  :-1][quad_mask]
        v11 = idx[1:,   1:][quad_mask]
        v01 = idx[:-1,  1:][quad_mask]
        tris = np.concatenate([
            np.stack([v00, v10, v11], 1),
            np.stack([v00, v11, v01], 1),
        ])
        all_faces.append(tris)

    faces = np.concatenate(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(result.points.astype(np.float64))
    mesh.triangles     = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        result.colors.astype(np.float64) / 255.0)
    return mesh

mesh_pidx = _image_grid_from_pixel_indices(result)
if mesh_pidx is not None:
    print(f"image-grid (pixel_indices): {len(mesh_pidx.vertices):,} verts, {len(mesh_pidx.triangles):,} triangles")
"""))

nbf.write(nb, str(path))
print("§6 added")
PYEOF
```

- [ ] **Step 2: Verify JSON valid**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json; json.load(open('docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb'))
print('valid')
"
```

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git commit -m "docs(mesh): §6 image-grid from pixel_indices — vertices == result.points, exact pcd correspondence"
```

---

### Task 7: Add §7 Visualization + §8 Summary

**Files:**
- Modify: `docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb`

- [ ] **Step 1: Append §7 and §8 cells**

```bash
/opt/conda/envs/nerfstudio/bin/python - << 'PYEOF'
import nbformat as nbf
from pathlib import Path

path = Path("docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb")
nb = nbf.read(str(path), as_version=4)

nb.cells.append(nbf.v4.new_markdown_cell("""\
## §7 — Visualize Each Method

Each method visualized via `visualize_splat` with camera frustums.
Use `camera_kwargs` scale to match scene extent (larger scale for outdoor scenes).
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
from collab_splats.utils.visualization import o3d_mesh_to_polydata

# Scale camera frustums to scene — default scale=0.02 is for indoor
_scene_scale = float((result.points.max(0) - result.points.min(0)).max())
_cam_kwargs = {**CAMERA_KWARGS, "scale": _scene_scale * 0.02, "n_poses": 5}

def _viz(name, mesh_o3d):
    if mesh_o3d is None:
        print(f"{name}: skipped (None)")
        return
    pd = o3d_mesh_to_polydata(mesh_o3d)
    print(f"\\n{'='*60}\\n{name}\\n{'='*60}")
    print(f"  verts={len(mesh_o3d.vertices):,}  tris={len(mesh_o3d.triangles):,}")
    visualize_splat(
        pd,
        aligned_cameras=list(result.extrinsics),
        mesh_kwargs=MESH_KWARGS,
        camera_kwargs=_cam_kwargs,
        viz_kwargs=VIZ_KWARGS,
    )

_viz("§4 TSDF tuned", mesh_tsdf)
_viz("§5 image-grid (world_points)", mesh_wpts)
_viz("§6 image-grid (pixel_indices)", mesh_pidx)
"""))

nb.cells.append(nbf.v4.new_markdown_cell("## §8 — Summary"))
nb.cells.append(nbf.v4.new_code_cell("""\
def _stats(name, mesh):
    if mesh is None:
        print(f"{name:<35} SKIPPED")
        return
    print(f"{name:<35} verts={len(mesh.vertices):>8,}  tris={len(mesh.triangles):>8,}")

print(f"{'Method':<35} {'Vertices':>8}  {'Triangles':>8}")
print("-" * 60)
_stats("TSDF tuned", mesh_tsdf)
_stats("image-grid (world_points)", mesh_wpts)
_stats("image-grid (pixel_indices)", mesh_pidx)

print(\"\"\"
Visual quality notes (fill in after inspection):
  TSDF tuned         : ___
  image-grid wpts    : ___
  image-grid pidx    : ___

Winner to promote to ImageGridMesher: ___
\"\"\")
"""))

nbf.write(nb, str(path))
print("§7 + §8 added")
PYEOF
```

- [ ] **Step 2: Verify notebook valid + cell count**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import json
nb = json.load(open('docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb'))
print(f'valid JSON, {len(nb[\"cells\"])} cells')
"
```

Expected: `valid JSON, 18 cells` (or similar)

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/mesh_method_comparison.ipynb
git commit -m "docs(mesh): §7 visualize all methods + §8 summary table — comparison notebook complete"
```

---

## Self-Review

**Spec coverage:**
- ✅ `o3d_mesh_to_polydata` in visualization.py (Task 1)
- ✅ §0–§2 setup/load/inspect (Task 2)
- ✅ §3 denoising variants A/B/C (Task 3)
- ✅ §4 TSDF tuned outdoor params (Task 4)
- ✅ §5 image-grid from world_points (Task 5)
- ✅ §6 image-grid from pixel_indices with graceful None fallback (Task 6)
- ✅ §7 visualize_splat per method with camera frustums + scaled camera_kwargs (Task 7)
- ✅ §8 summary table (Task 7)

**Placeholder scan:** No TBDs, no "similar to above", all code complete ✓

**Type consistency:**
- `find_depth_edges(depth_im, threshold, dilation_itr)` — called as `find_depth_edges(depth[i], threshold=edge_threshold)` ✓
- `visualize_splat(pd, aligned_cameras=list(result.extrinsics), ...)` — expects `List[np.ndarray]`, `list()` on `(N,4,4)` array gives list of `(4,4)` arrays ✓
- `o3d_mesh_to_polydata` defined in Task 1, imported in §7 ✓
- `_feedforward_to_tsdf_inputs` used in §4 — check: it's actually `pointcloud_to_mesh` from `collab_splats.mesh.utils` which calls it internally ✓
- `result.confidence.numpy()` — confidence stored as torch.Tensor per FeedforwardResult docstring; `.numpy()` valid ✓
