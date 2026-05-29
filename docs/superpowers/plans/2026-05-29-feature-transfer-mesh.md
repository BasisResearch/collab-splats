# Feature Transfer: Pointcloud → Mesh Vertices — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `transfer_features_to_mesh` to `mesh/utils.py` and wire it into the feedforward mesh notebook so semantic similarity queries work on both the pointcloud and the mesh.

**Architecture:** `lift_features` already lifts 2D feature maps to per-point `(P, D)` features stored in `FeedforwardResult.features`. A new `transfer_features_to_mesh` function delegates to the existing `features2vertex` KNN aggregation to map those to `(M, D)` per-vertex features. The notebook runs the extractor, lifts, meshes, transfers, then demonstrates cosine similarity on both representations.

**Tech Stack:** Python 3.11, numpy, open3d, torch, zarr 3.x, `collab_splats.semantics.features.BaseFeatureExtractor`, `collab_splats.pointcloud.utils.lift_features`, `collab_splats.mesh.utils.features2vertex`

---

### Task 1: Add `transfer_features_to_mesh` to `mesh/utils.py`

**Files:**
- Modify: `collab_splats/mesh/utils.py` (after `features2vertex` at line ~172)

- [ ] **Step 1: Open `collab_splats/mesh/utils.py` and locate insertion point**

Find the line after `features2vertex` ends (the `return features_kNN` line before the `########` divider for "Mesh cleaning / repair"). Insert the new function there.

- [ ] **Step 2: Add `transfer_features_to_mesh` after `features2vertex`**

Insert this block immediately before the `########` Mesh cleaning / repair divider:

```python
def transfer_features_to_mesh(
    result: FeedforwardResult,
    mesh: o3d.geometry.TriangleMesh,
    *,
    k: int = 5,
    sdf_trunc: float = 0.03,
) -> np.ndarray:
    """Transfer per-point features from a FeedforwardResult to mesh vertices via KNN.

    Args:
        result:    FeedforwardResult with features (P, D) and points (P, 3) populated.
        mesh:      Open3D TriangleMesh whose vertices receive the features.
        k:         Neighbors for Gaussian-weighted aggregation (passed to features2vertex).
        sdf_trunc: Truncation distance — pointcloud points farther than this from their
                   nearest vertex are excluded from aggregation.

    Returns:
        (M, D) float32 ndarray of per-vertex features, index-aligned with mesh.vertices.
    """
    assert result.features is not None, (
        "result.features is None — call lift_features() and assign result.features before transferring"
    )
    return features2vertex(
        np.asarray(mesh.vertices),
        result.points,
        result.features,
        k=k,
        sdf_trunc=sdf_trunc,
    )
```

No new imports needed — `o3d`, `np`, `FeedforwardResult`, and `features2vertex` are all already in scope at module level.

- [ ] **Step 3: Verify module imports cleanly**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.mesh.utils import transfer_features_to_mesh; print('OK')"
```

Expected: `OK`

---

### Task 2: Write tests for `transfer_features_to_mesh`

**Files:**
- Create: `tests/mesh/test_feature_transfer.py`

Note: check whether `tests/mesh/__init__.py` exists; create it if not.

- [ ] **Step 1: Create test directory init if missing**

```bash
ls /workspace/collab-splats/tests/mesh/ 2>/dev/null || mkdir -p /workspace/collab-splats/tests/mesh && touch /workspace/collab-splats/tests/mesh/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/mesh/test_feature_transfer.py`:

```python
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.utils import transfer_features_to_mesh


def _make_mesh(n_vertices=20):
    """Toy mesh: random vertices, no real geometry needed."""
    rng = np.random.default_rng(42)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(rng.standard_normal((n_vertices, 3)).astype(np.float64))
    return mesh


def _make_result(n_points=30, dim=8, features=None):
    """Minimal duck-type result — transfer_features_to_mesh only reads .points and .features."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n_points, 3)).astype(np.float32)
    feats = features if features is not None else rng.standard_normal((n_points, dim)).astype(np.float32)
    return SimpleNamespace(points=pts, features=feats)


def test_transfer_features_to_mesh_basic():
    """Output shape is (M, D) and contains non-zero values when points are near vertices."""
    mesh = _make_mesh(n_vertices=20)
    # Place pointcloud points close to mesh vertices so KNN finds matches within sdf_trunc
    mesh_verts = np.asarray(mesh.vertices).astype(np.float32)
    # Jitter mesh vertices slightly to make pointcloud near them
    rng = np.random.default_rng(1)
    pts = mesh_verts + rng.standard_normal(mesh_verts.shape).astype(np.float32) * 0.001
    feats = rng.standard_normal((len(pts), 8)).astype(np.float32)
    result = SimpleNamespace(points=pts, features=feats)

    vertex_features = transfer_features_to_mesh(result, mesh, sdf_trunc=0.1)

    assert vertex_features.shape == (20, 8)
    assert vertex_features.dtype == np.float32
    # At least some vertices should have non-zero features
    assert np.any(vertex_features != 0.0)


def test_transfer_features_to_mesh_none_raises():
    """AssertionError when result.features is None."""
    mesh = _make_mesh()
    result = SimpleNamespace(points=np.zeros((10, 3), dtype=np.float32), features=None)

    with pytest.raises(AssertionError, match="result.features is None"):
        transfer_features_to_mesh(result, mesh)


def test_transfer_features_to_mesh_zero_points_in_range():
    """Returns all-zero array when all points are outside sdf_trunc."""
    mesh = _make_mesh(n_vertices=10)
    # Place pointcloud 100 units away from mesh vertices (which are ~unit-scale)
    pts = np.zeros((5, 3), dtype=np.float32) + 100.0
    feats = np.ones((5, 8), dtype=np.float32)
    result = SimpleNamespace(points=pts, features=feats)

    vertex_features = transfer_features_to_mesh(result, mesh, sdf_trunc=0.03)

    assert vertex_features.shape == (10, 8)
    assert np.all(vertex_features == 0.0)
```

- [ ] **Step 3: Run tests — verify they fail (function not yet imported by test)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/mesh/test_feature_transfer.py -v
```

Expected: `FAILED` with `ImportError` or actual assertion failures depending on whether Task 1 is done. If Task 1 is complete, all three should pass immediately — that's fine.

- [ ] **Step 4: Run tests — verify all pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/mesh/test_feature_transfer.py -v
```

Expected:
```
PASSED tests/mesh/test_feature_transfer.py::test_transfer_features_to_mesh_basic
PASSED tests/mesh/test_feature_transfer.py::test_transfer_features_to_mesh_none_raises
PASSED tests/mesh/test_feature_transfer.py::test_transfer_features_to_mesh_zero_points_in_range
3 passed
```

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/mesh/utils.py tests/mesh/test_feature_transfer.py tests/mesh/__init__.py
git commit -m "feat(mesh): add transfer_features_to_mesh via KNN from pointcloud features"
```

---

### Task 3: Add feature lifting + mesh transfer + similarity demo to notebook

**Files:**
- Modify: `docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb`

The notebook currently has 9 cells (§1 load + §2 TSDF meshing). We add three new sections: §3 feature lifting, §4 transfer to mesh vertices, §5 similarity demo.

**Context:** The notebook uses `result` (a `FeedforwardResult` loaded from zarr) and `mesh_result` (a `MeshResult` from `pointcloud_to_mesh`). `MESH_DIR` is set to `Path("/tmp/feedforward_mesh_demo/mesh")` by the config cell.

- [ ] **Step 1: Add imports cell (insert after existing import cell, cell index 2)**

Add a new markdown + code cell pair after the existing imports. The new code cell adds:

```python
import numpy as np
import torch
import zarr

from collab_splats.mesh.utils import transfer_features_to_mesh
from collab_splats.pointcloud.utils import lift_features
from collab_splats.semantics.features import BaseFeatureExtractor
```

Note: `numpy`, `torch`, `zarr` may already be transitively available — adding explicit imports is correct style here.

- [ ] **Step 2: Add §3 — Feature lifting section**

Append two cells to the notebook:

**Markdown cell:**
```
## §3 — Feature lifting

Extract DINOv2 features for each frame, then lift to per-point 3D features using multi-view
confidence-weighted projection. `extract_and_cache` skips extraction on re-run if the cache
is valid.
```

**Code cell:**
```python
# Extract per-frame DINOv2 features — cached under MESH_DIR
FEATURE_CACHE = MESH_DIR / "features_cache" / "dinov2.zarr"
extractor = BaseFeatureExtractor.get("dinov2")()
zarr_path = extractor.extract_and_cache(result.image_paths, FEATURE_CACHE)

# Load feature maps: list of (D, H_p, W_p) tensors
store = zarr.open(str(zarr_path), mode="r")
feature_maps = [
    torch.from_numpy(np.array(store["features"][i]))
    for i in range(store["features"].shape[0])
]

# Lift 2D features → (P, D) per-point features
lifted = lift_features(feature_maps, result)
result.features = lifted.numpy()

print(f"Lifted features shape: {result.features.shape}")  # (P, D)
```

- [ ] **Step 3: Add §4 — Transfer to mesh vertices section**

Append two cells:

**Markdown cell:**
```
## §4 — Transfer features to mesh vertices

Map per-point features to mesh vertices via KNN Gaussian-weighted aggregation.
Points farther than `sdf_trunc` from any vertex are excluded.
```

**Code cell:**
```python
# Load the mesh written by §2
mesh = o3d.io.read_triangle_mesh(str(mesh_result.mesh_path))
mesh.compute_vertex_normals()

# Transfer: (P, D) pointcloud features → (M, D) vertex features
vertex_features = transfer_features_to_mesh(result, mesh)

print(f"Mesh vertices:   {np.asarray(mesh.vertices).shape[0]}")
print(f"Vertex features: {vertex_features.shape}")   # (M, D)
print(f"Non-zero verts:  {(vertex_features.any(axis=1)).sum()}")
```

- [ ] **Step 4: Add §5 — Similarity demo section**

Append two cells:

**Markdown cell:**
```
## §5 — Similarity query: pointcloud vs mesh

Same text query runs against both `result.features` (pointcloud) and `vertex_features` (mesh).
Cosine similarity scores are normalized to [0, 1] for visualization.
```

**Code cell:**
```python
import torch.nn.functional as F
from collab_splats.semantics.features import BaseFeatureExtractor

# Encode a text query using the same DINOv2 extractor
# (DINOv2 is vision-only; use Talk2DINO for text queries if available,
#  otherwise substitute any compatible text encoder)
QUERY = "floor"

# --- Pointcloud similarity ---
pc_feats = torch.from_numpy(result.features)           # (P, D)
pc_feats_norm = F.normalize(pc_feats, dim=-1)

# --- Mesh vertex similarity ---
vx_feats = torch.from_numpy(vertex_features)           # (M, D)
vx_feats_norm = F.normalize(vx_feats, dim=-1)

# Compute mean feature as a proxy query vector (replace with real text encoding if available)
# To use Talk2DINO text encoding:
#   text_extractor = BaseFeatureExtractor.get("talk2dino")()
#   query_vec = text_extractor.encode_text([QUERY])[0]  # (D,)
query_vec = pc_feats_norm.mean(dim=0)                  # placeholder: mean of all point features

query_norm = F.normalize(query_vec.unsqueeze(0), dim=-1)  # (1, D)

pc_sim = (pc_feats_norm @ query_norm.T).squeeze(-1).numpy()    # (P,)
vx_sim = (vx_feats_norm @ query_norm.T).squeeze(-1).numpy()   # (M,)

# Normalize to [0, 1]
def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)

pc_sim_n = norm01(pc_sim)
vx_sim_n = norm01(vx_sim)

print(f"Pointcloud similarity — mean: {pc_sim_n.mean():.3f}  max: {pc_sim_n.max():.3f}")
print(f"Mesh vertex similarity — mean: {vx_sim_n.mean():.3f}  max: {vx_sim_n.max():.3f}")

# Visualize mesh colored by similarity
import pyvista as pv
pv.set_jupyter_backend("html")

mesh_pv = pv.read(str(mesh_result.mesh_path))
mesh_pv["similarity"] = vx_sim_n

pl = pv.Plotter()
pl.add_mesh(mesh_pv, scalars="similarity", cmap="hot", clim=[0, 1])
pl.show()
```

- [ ] **Step 5: Verify notebook runs without errors**

In a terminal (not in the notebook — avoids OOM from duplicate model loads):

```bash
cd /workspace/collab-splats
/opt/conda/envs/reconstruction/bin/jupyter nbconvert \
  --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb \
  --output /tmp/feedforward_mesh_executed.ipynb 2>&1 | tail -20
```

Expected: `Wrote ... /tmp/feedforward_mesh_executed.ipynb` with no `ERROR` lines.

Note: requires the zarr cache from `feedforward_methods.ipynb` to exist at the path set in `tutorial_config.py`. If missing, run that notebook first.

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb
git commit -m "docs(mesh): add feature lifting and mesh transfer cells to feedforward_mesh notebook"
```

---

## Summary of Changes

| File | Type | What |
|------|------|------|
| `collab_splats/mesh/utils.py` | Modify | Add `transfer_features_to_mesh` after `features2vertex` |
| `tests/mesh/__init__.py` | Create | Empty init (if missing) |
| `tests/mesh/test_feature_transfer.py` | Create | 3 tests for `transfer_features_to_mesh` |
| `docs/source/tutorials/06_mesh/stage/feedforward_mesh.ipynb` | Modify | §3 feature lifting, §4 mesh transfer, §5 similarity demo |
