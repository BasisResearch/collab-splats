# Mesh Feature Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transfer per-point semantic features onto TSDF mesh vertices via a GPU-accelerated `features2vertex`, persist them, and make mesh mode in the dashboard queryable with the same `score_queries` path used for points.

**Architecture:** Keep the CPU KDTree spatial query (asymptotically right; add `workers=-1`) but move the Gaussian-weighted scatter aggregation to a torch float32 GPU kernel using `index_add_`, mirroring `lift_features`. Wire the existing `transfer_features_to_mesh` into the dashboard pipeline after the feature lift, persisting `mesh/vertex_features.npy`. The viewer loads and L2-normalizes those vertex features and scores them exactly like point features when `mode == "mesh"`.

**Tech Stack:** Python 3.11, PyTorch (CUDA), scipy `cKDTree`, Open3D, numpy, pytest. Env: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-04-mesh-feature-transfer-design.md`

---

## File Structure

- `collab_splats/mesh/utils.py` — rewrite `features2vertex` (GPU hybrid); add `persist_mesh_vertex_features` orchestration helper.
- `collab_splats/mesh/base.py` — add `MeshResult.vertex_features` field.
- `collab_splats/dashboard/pipeline.py` — call `persist_mesh_vertex_features` after `_lift_and_compress`.
- `collab_splats/dashboard/viewer.py` — load/normalize mesh vertex features; score them in `mode == "mesh"`.
- `tests/mesh/test_utils.py` — GPU/CPU parity, truncation, shape, query-parity, persist helper.
- `tests/dashboard/test_viewer.py` — mesh-mode scoring smoke.

---

## Task 1: GPU-accelerate `features2vertex`

**Files:**
- Modify: `collab_splats/mesh/utils.py:100-164` (`features2vertex`)
- Test: `tests/mesh/test_utils.py`

Imports already present at top of `mesh/utils.py`: `numpy as np`, `torch`, `cKDTree`, `trange`. No new imports needed.

- [ ] **Step 1: Write the failing parity test**

Add to `tests/mesh/test_utils.py`:

```python
import numpy as np


def _features2vertex_numpy_reference(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """Frozen copy of the original CPU implementation — parity oracle for the GPU rewrite."""
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh_vertices)
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k)
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((len(vertices), features.shape[1]), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    features = features[valid_mask]
    sigma = np.mean(distances)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((len(vertices), features.shape[1]), dtype=features.dtype)
    wsum = np.zeros((len(vertices), 1), dtype=features.dtype)
    for i in range(k):
        np.add.at(out, indices[:, i], features * weights[:, i : i + 1])
        np.add.at(wsum, indices[:, i], weights[:, i : i + 1])
    nz = wsum.squeeze() > 0
    out[nz] /= wsum[nz]
    return out


def test_features2vertex_matches_numpy_reference():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(0)
    # Vertices and points share a bounded cube so most points fall within sdf_trunc.
    vertices = rng.random((200, 3)).astype(np.float64)
    points = rng.random((1000, 3)).astype(np.float64)
    features = rng.random((1000, 8)).astype(np.float32)

    got = features2vertex(vertices, points, features, k=5, sdf_trunc=0.1)
    want = _features2vertex_numpy_reference(vertices, points, features, k=5, sdf_trunc=0.1)

    assert got.shape == (200, 8)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_features2vertex_matches_numpy_reference -v`
Expected: FAIL — current implementation uses float64 `np.add.at`; the test passes only once the rewrite preserves numerics. (If it already passes by luck, the rewrite in Step 3 must still keep it green.)

- [ ] **Step 3: Rewrite `features2vertex` with the GPU hybrid kernel**

Replace the body of `features2vertex` (`mesh/utils.py:100-164`) with:

```python
def features2vertex(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """
    Map point cloud features to mesh vertices using KNN over a KDTree.

    Spatial query runs on the CPU KDTree (multicore, O(N log M)); the Gaussian-weighted
    aggregation runs on the GPU in float32 via index_add_ (mirrors lift_features). Falls
    back to the CPU torch device when CUDA is unavailable.

    Returns np.ndarray (M, D), dtype matching input features.

    Args:
        mesh_vertices: (M, 3) array of mesh vertex positions
        points:        (N, 3) array of input point cloud
        features:      (N, D) array of per-point features
        k:             number of nearest neighbors used for weighting
        sdf_trunc:     truncation distance — points whose nearest vertex is farther are dropped
    """
    vertices = np.asarray(mesh_vertices)
    M = len(vertices)
    D = features.shape[1]

    # Nearest-vertex query for every point; workers=-1 uses all cores.
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k, workers=-1)

    # k=1 collapses the neighbour axis; restore it so the kernel below is uniform.
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Drop points whose closest vertex is beyond the truncation band.
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    feats = features[valid_mask]

    # Move the aggregation to the GPU (float32); one .cpu() at the end.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.as_tensor(np.ascontiguousarray(distances), dtype=torch.float32, device=device)
    idx = torch.as_tensor(np.ascontiguousarray(indices), dtype=torch.long, device=device)
    f = torch.as_tensor(np.ascontiguousarray(feats), dtype=torch.float32, device=device)

    # Gaussian kernel over neighbour distances; normalize weights per point (over k).
    sigma = d.mean()
    w = torch.exp(-(d**2) / (2 * sigma**2))
    w = w / w.sum(dim=1, keepdim=True)

    # Scatter weighted features to vertices; accumulate weights for normalization.
    acc = torch.zeros((M, D), dtype=torch.float32, device=device)
    wsum = torch.zeros((M, 1), dtype=torch.float32, device=device)
    for j in range(k):
        acc.index_add_(0, idx[:, j], f * w[:, j : j + 1])
        wsum.index_add_(0, idx[:, j], w[:, j : j + 1])

    # Normalize aggregated features by summed weights (skip untouched vertices -> stay zero).
    nz = wsum.squeeze(1) > 0
    acc[nz] /= wsum[nz]

    return acc.cpu().numpy().astype(features.dtype)
```

Note: `trange` is no longer used by this function. Leave the import — `clean_repair_mesh` and other helpers in the file may still use `tqdm`/`trange`; only remove it if a lint check flags it unused across the whole module.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_features2vertex_matches_numpy_reference -v`
Expected: PASS

- [ ] **Step 5: Add truncation + shape edge tests**

Add to `tests/mesh/test_utils.py`:

```python
def test_features2vertex_all_far_returns_zeros():
    from collab_splats.mesh.utils import features2vertex

    vertices = np.zeros((10, 3), dtype=np.float64)
    points = np.full((20, 3), 100.0, dtype=np.float64)  # all far beyond sdf_trunc
    features = np.ones((20, 4), dtype=np.float32)

    out = features2vertex(vertices, points, features, k=3, sdf_trunc=0.03)
    assert out.shape == (10, 4)
    assert np.all(out == 0.0)


def test_features2vertex_dtype_preserved():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(1)
    vertices = rng.random((50, 3))
    points = rng.random((100, 3))
    features = rng.random((100, 6)).astype(np.float32)

    out = features2vertex(vertices, points, features, k=4, sdf_trunc=0.2)
    assert out.dtype == np.float32
    assert out.shape == (50, 6)
```

- [ ] **Step 6: Run the full mesh utils test file**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -v`
Expected: PASS (including the pre-existing `test_features2vertex_output_shape`)

- [ ] **Step 7: Commit**

```bash
git add collab_splats/mesh/utils.py tests/mesh/test_utils.py
git commit -m "perf(mesh): GPU-accelerate features2vertex (KDTree workers=-1 + index_add_)"
```

---

## Task 2: Add `MeshResult.vertex_features` field

**Files:**
- Modify: `collab_splats/mesh/base.py:9-12`
- Test: `tests/mesh/test_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/mesh/test_utils.py`:

```python
def test_meshresult_has_vertex_features_field():
    from pathlib import Path

    from collab_splats.mesh.base import MeshResult

    r = MeshResult(mesh_path=Path("/tmp/m.ply"))
    assert r.vertex_features is None  # default

    r2 = MeshResult(mesh_path=Path("/tmp/m.ply"), vertex_features=np.zeros((3, 2)))
    assert r2.vertex_features.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_meshresult_has_vertex_features_field -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'vertex_features'`

- [ ] **Step 3: Add the field**

In `collab_splats/mesh/base.py`, change the `MeshResult` dataclass to:

```python
@dataclass
class MeshResult:
    mesh_path: Path
    pcd_path: Path | None = None
    vertex_features: np.ndarray | None = None
```

`numpy as np` is already imported at the top of `base.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_meshresult_has_vertex_features_field -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/mesh/base.py tests/mesh/test_utils.py
git commit -m "feat(mesh): add MeshResult.vertex_features field"
```

---

## Task 3: `persist_mesh_vertex_features` orchestration helper

**Files:**
- Modify: `collab_splats/mesh/utils.py` (add helper near `transfer_features_to_mesh`, after line 195)
- Test: `tests/mesh/test_utils.py`

This helper reads a written mesh PLY, transfers point features to its vertices, writes
`vertex_features.npy` next to the mesh, and returns the `(M, D)` array. It takes point
features explicitly (the dashboard lifts/normalizes them separately) rather than reading
`result.features`.

- [ ] **Step 1: Write the failing test**

Add to `tests/mesh/test_utils.py`:

```python
def test_persist_mesh_vertex_features(tmp_path):
    import open3d as o3d

    from collab_splats.mesh.utils import persist_mesh_vertex_features

    # Minimal 3-vertex mesh written to disk.
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh_path = tmp_path / "mesh_tsdf.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    # Points sitting on the vertices, each with a distinct 2D feature.
    points = verts.copy()
    feats = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

    out = persist_mesh_vertex_features(mesh_path, points, feats, k=1, sdf_trunc=0.5)

    assert out.shape == (3, 2)
    saved = np.load(mesh_path.parent / "vertex_features.npy")
    np.testing.assert_allclose(saved, out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_persist_mesh_vertex_features -v`
Expected: FAIL with `ImportError: cannot import name 'persist_mesh_vertex_features'`

- [ ] **Step 3: Implement the helper**

Add to `collab_splats/mesh/utils.py` immediately after `transfer_features_to_mesh` (after line 195):

```python
def persist_mesh_vertex_features(
    mesh_path: Path,
    points: np.ndarray,
    point_features: np.ndarray,
    *,
    k: int = 5,
    sdf_trunc: float = 0.03,
) -> np.ndarray:
    """Transfer point features to a written mesh's vertices and cache them as vertex_features.npy.

    Reads the mesh PLY at mesh_path, runs features2vertex against the supplied point
    features (already lifted/normalized by the caller), writes vertex_features.npy beside
    the mesh, and returns the (M, D) per-vertex array (index-aligned with mesh.vertices).
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    vertex_features = features2vertex(vertices, points, point_features, k=k, sdf_trunc=sdf_trunc)
    out_path = Path(mesh_path).parent / "vertex_features.npy"
    np.save(out_path, vertex_features)
    logger.info("saved mesh vertex features → %s  shape=%s", out_path, vertex_features.shape)
    return vertex_features
```

`Path`, `np`, `o3d`, and `logger` are already imported at the top of `mesh/utils.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py::test_persist_mesh_vertex_features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/mesh/utils.py tests/mesh/test_utils.py
git commit -m "feat(mesh): persist_mesh_vertex_features helper (mesh PLY -> vertex_features.npy)"
```

---

## Task 4: Wire the transfer into the dashboard pipeline

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py` (import + call after `_lift_and_compress`, around line 265)
- Test: `tests/dashboard/test_pipeline.py`

`_lift_and_compress` writes `semantics_dir/lifted_normed.npy` = `(P, D)` L2-normed point
features aligned with `result.points`. After it runs, transfer those to the mesh built at
`out_dir/mesh/mesh_tsdf.ply`.

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_pipeline.py`:

```python
def test_transfer_mesh_features_writes_npy(tmp_path, monkeypatch):
    import numpy as np
    import open3d as o3d

    from collab_splats.dashboard import pipeline

    # Build a tiny mesh on disk under out_dir/mesh.
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris)
    )
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh_tsdf.ply"), mesh)

    # Cached point features under semantics/.
    sem_dir = tmp_path / "semantics"
    sem_dir.mkdir()
    np.save(sem_dir / "lifted_normed.npy", np.eye(3, 2, dtype=np.float32))

    class _Result:
        points = verts.copy()

    pipeline._transfer_mesh_features(_Result(), tmp_path, k=1, sdf_trunc=0.5)

    assert (mesh_dir / "vertex_features.npy").exists()
    out = np.load(mesh_dir / "vertex_features.npy")
    assert out.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py::test_transfer_mesh_features_writes_npy -v`
Expected: FAIL with `AttributeError: module 'collab_splats.dashboard.pipeline' has no attribute '_transfer_mesh_features'`

- [ ] **Step 3: Add the `_transfer_mesh_features` helper and call it**

In `collab_splats/dashboard/pipeline.py`, add the import near the other `collab_splats` imports (top of file, alongside `from collab_splats.pointcloud.utils import lift_features`):

```python
from collab_splats.mesh.utils import persist_mesh_vertex_features
```

Add this helper (place it just after `_lift_and_compress`, before `_sample`):

```python
def _transfer_mesh_features(result, out_dir: Path, *, k: int = 5, sdf_trunc: float = 0.03) -> None:
    """Transfer cached point features onto the TSDF mesh vertices and persist vertex_features.npy.

    No-op (logged) if the mesh or the lifted point features are missing — neither is fatal
    to the run.
    """
    mesh_path = Path(out_dir) / "mesh" / "mesh_tsdf.ply"
    lifted_path = Path(out_dir) / "semantics" / "lifted_normed.npy"
    if not mesh_path.exists() or not lifted_path.exists():
        logger.warning("mesh feature transfer skipped: mesh=%s lifted=%s", mesh_path.exists(), lifted_path.exists())
        return
    point_features = np.load(lifted_path)
    persist_mesh_vertex_features(mesh_path, result.points, point_features, k=k, sdf_trunc=sdf_trunc)
```

Then, in `run_pipeline`, immediately after the `_lift_and_compress(result, out_dir / "semantics", op_log)` call (line 265), add:

```python
            # Transfer lifted point features onto the mesh vertices (same feature space -> mesh is queryable).
            op_log.update_progress(94, "mesh: transferring features to vertices")
            _transfer_mesh_features(result, out_dir)
```

`Path`, `np`, and `logger` are already imported at the top of `pipeline.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py::test_transfer_mesh_features_writes_npy -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/pipeline.py tests/dashboard/test_pipeline.py
git commit -m "feat(dashboard): transfer lifted features to mesh vertices in pipeline"
```

---

## Task 5: Query parity — score mesh vertex features in the viewer

**Files:**
- Modify: `collab_splats/dashboard/viewer.py` (add loader + branch in `score_query`)
- Test: `tests/dashboard/test_viewer.py`

`score_query` (viewer.py:229-270) currently scores `self._lifted_normed` (points). In mesh
mode it must score per-vertex features, loaded from `mesh/vertex_features.npy` and
L2-normalized the same way `load_lifted_normed` normalizes point features.

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_viewer.py`:

```python
def test_load_mesh_vertex_features_normalizes(tmp_path):
    import numpy as np

    from collab_splats.dashboard.viewer import load_mesh_vertex_features

    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    # Unnormalized vertex features.
    feats = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    np.save(mesh_dir / "vertex_features.npy", feats)

    out = load_mesh_vertex_features(mesh_dir)
    # Rows L2-normalized to unit length.
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)


def test_load_mesh_vertex_features_missing_returns_none(tmp_path):
    from collab_splats.dashboard.viewer import load_mesh_vertex_features

    assert load_mesh_vertex_features(tmp_path / "nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py::test_load_mesh_vertex_features_normalizes -v`
Expected: FAIL with `ImportError: cannot import name 'load_mesh_vertex_features'`

- [ ] **Step 3: Add the loader**

In `collab_splats/dashboard/viewer.py`, add next to `load_lifted_normed` (after line 51):

```python
def load_mesh_vertex_features(mesh_dir) -> "np.ndarray | None":
    """Load cached mesh vertex features and L2-normalise -> (M, D) float32, or None if absent.

    Matches load_lifted_normed's normalization so mesh features share the point feature
    space and can be scored by the same extractor.score_queries call.
    """
    from pathlib import Path

    path = Path(mesh_dir) / "vertex_features.npy"
    if not path.exists():
        return None
    feats = np.load(path).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-8)
```

`numpy as np` is already imported at the top of `viewer.py`.

- [ ] **Step 4: Run loader tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -k load_mesh_vertex_features -v`
Expected: PASS (both tests)

- [ ] **Step 5: Write the failing mesh-mode scoring test**

Add to `tests/dashboard/test_viewer.py`. This drives `score_query` to use mesh features
when `mode == "mesh"`. It stubs the extractor so no heavy model loads.

```python
def test_score_query_uses_mesh_features_in_mesh_mode(tmp_path, monkeypatch):
    import numpy as np

    from collab_splats.dashboard import viewer as viewer_mod
    from collab_splats.dashboard.viewer import SceneViewer

    captured = {}

    class _StubExtractor:
        def score_queries(self, features, positive, negative=None):
            captured["n"] = features.shape[0]
            import torch

            return torch.ones(features.shape[0])

    v = SceneViewer.__new__(SceneViewer)  # bypass __init__ (no GUI in tests)
    v.mode = "mesh"
    v._result = type("R", (), {"colors": np.zeros((5, 3), dtype=np.uint8)})()
    v._lifted_normed = np.ones((5, 4), dtype=np.float32)          # 5 points
    v._mesh_vertex_features = np.ones((3, 4), dtype=np.float32)   # 3 vertices
    v._extractor_cache = {"talk2dino": _StubExtractor()}
    monkeypatch.setattr(v, "ensure_lifted", lambda op_log=None: None)

    colors = v.score_query(positive=["chair"], op_log=None)

    # Scored the 3 mesh vertices, not the 5 points.
    assert captured["n"] == 3
    assert colors.shape[0] == 3
```

- [ ] **Step 6: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py::test_score_query_uses_mesh_features_in_mesh_mode -v`
Expected: FAIL — `score_query` scores `self._lifted_normed` (5 points) so `captured["n"] == 5`, or `AttributeError` on `_mesh_vertex_features`.

- [ ] **Step 7: Branch `score_query` on mesh mode**

In `viewer.py`, locate the scoring block in `score_query` (viewer.py:261-268):

```python
        _stage(f"query: encoding {len(positive)} positive / {len(negative or [])} negative")
        extractor = self._get_extractor(extractor_name)
        features = torch.from_numpy(self._lifted_normed)  # (P, D)

        _stage(f"query: scoring {features.shape[0]} points")
        scores = extractor.score_queries(features, positive=positive, negative=negative or None)
        sims = scores.detach().cpu().numpy()
        colors = apply_viridis(sims)
        _stage("query: scored")
        return colors
```

Replace the `features = torch.from_numpy(self._lifted_normed)` line and the scoring label
with a mesh-aware selection:

```python
        _stage(f"query: encoding {len(positive)} positive / {len(negative or [])} negative")
        extractor = self._get_extractor(extractor_name)

        # In mesh mode score per-vertex features (same feature space); else score points.
        feature_array = getattr(self, "_mesh_vertex_features", None) if self.mode == "mesh" else None
        if feature_array is None:
            feature_array = self._lifted_normed
        features = torch.from_numpy(feature_array)  # (N, D)

        _stage(f"query: scoring {features.shape[0]} elements")
        scores = extractor.score_queries(features, positive=positive, negative=negative or None)
        sims = scores.detach().cpu().numpy()
        colors = apply_viridis(sims)
        _stage("query: scored")
        return colors
```

- [ ] **Step 8: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py::test_score_query_uses_mesh_features_in_mesh_mode -v`
Expected: PASS

- [ ] **Step 9: Populate `_mesh_vertex_features` on load**

In `viewer.py`, the loader (`SceneViewer.load`, viewer.py:99-118) stashes `_mesh_path` and
`_semantics_dir`. Add mesh-feature loading there so the attribute exists for queries. After
the `self._mesh_path = Path(mesh_path) if mesh_path else None` line (viewer.py:112), add:

```python
        # Per-vertex mesh features (if the pipeline persisted them) — same space as point features.
        self._mesh_vertex_features = (
            load_mesh_vertex_features(self._mesh_path.parent) if self._mesh_path else None
        )
```

Also add a safe default in case code paths construct the viewer without `load`: in the
block where other attributes get defaults (near `self.mode = "pointcloud"`, viewer.py:81),
add:

```python
        self._mesh_vertex_features = None
```

- [ ] **Step 10: Run the viewer test file**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: PASS (existing tests + the three new ones)

- [ ] **Step 11: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "feat(dashboard): score mesh vertex features in mesh mode (query parity)"
```

---

## Task 6: Full suite + format

**Files:** none (verification)

- [ ] **Step 1: Format**

Run: `cd /workspace/collab-splats && black collab_splats/mesh collab_splats/dashboard tests && isort collab_splats/mesh collab_splats/dashboard tests`
Expected: files reformatted/unchanged, no errors.

- [ ] **Step 2: Run mesh + dashboard suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh tests/dashboard -v`
Expected: PASS (no new failures vs baseline; pre-existing known failures in `docs/known-test-failures.md` excepted).

- [ ] **Step 3: Commit any formatting changes**

```bash
git add -A
git commit -m "style: black + isort for mesh feature transfer" || echo "nothing to format"
```

---

## Self-Review Notes

- **Spec coverage:** GPU `features2vertex` (Task 1) ✓; `MeshResult.vertex_features` (Task 2) ✓; persist `.npy` (Task 3) ✓; pipeline wiring (Task 4) ✓; viewer query parity + same renormalization (Task 5) ✓; tests for parity/truncation/shape/query (Tasks 1,3,5) ✓.
- **Out of scope (per spec):** AE-as-color fusion, native volume fusion, post-transfer Laplacian smoothing. Not in any task — intentional.
- **Type consistency:** `features2vertex(mesh_vertices, points, features, k, sdf_trunc) -> (M,D) ndarray`; `persist_mesh_vertex_features(mesh_path, points, point_features, *, k, sdf_trunc) -> ndarray`; `_transfer_mesh_features(result, out_dir, *, k, sdf_trunc) -> None`; `load_mesh_vertex_features(mesh_dir) -> ndarray | None`; viewer attribute `_mesh_vertex_features`. Names consistent across tasks.
- **Risk:** mesh built (pipeline line 246) before features lifted (line 265); transfer correctly runs after the lift in Task 4. Vertex order stability across PLY write/read holds (Open3D preserves order); persisted `.npy` stays aligned.
```
