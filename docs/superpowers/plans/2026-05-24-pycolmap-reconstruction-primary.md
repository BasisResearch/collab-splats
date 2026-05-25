# pycolmap.Reconstruction Primary Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pycolmap.Reconstruction` the primary camera store in `PointcloudResult`, unify field terminology across `FeedforwardResult` and `PointcloudResult`, and delete ~100 lines of numpy-unpack conversion boilerplate.

**Architecture:** `FeedforwardResult` stays pure numpy/zarr (dense domain). `PointcloudResult` drops stored `camera_poses`/`camera_intrinsics`/`colmap_reconstruction: Any` fields and replaces them with a typed `reconstruction: pycolmap.Reconstruction` field plus four `@property` views (`points`, `colors`, `extrinsics`, `intrinsics`). Terminology across both types is unified (`pts3d→points`, `conf→confidence`). `colmap_reconstruction_to_result` and `_colmap_recon_to_result` are deleted; `build_colmap()` constructs `PointcloudResult` directly.

**Tech Stack:** Python 3.11, pycolmap 4.0.4, numpy, pytest (`/opt/conda/envs/nerfstudio/bin/python`)

**Spec:** `docs/superpowers/specs/2026-05-24-pycolmap-reconstruction-primary-design.md`

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/base.py` | Rewrite `PointcloudResult`; delete `_colmap_recon_to_result` |
| `collab_splats/pointcloud/utils.py` | Delete `colmap_reconstruction_to_result`; update `lift_features` field refs |
| `collab_splats/pointcloud/feedforward/base.py` | Rename `pts3d→points`, `conf→confidence` (field + all serialization); update `build_colmap()` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Rename `pts3d→points`, `conf→confidence` in `_postprocess` |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Same renames in `_postprocess` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Same renames in `_postprocess` |
| `collab_splats/pointcloud/bundle_adjustment.py` | `result.conf → result.confidence` in `refine()` |
| `collab_splats/pointcloud/sfm.py` | Replace `colmap_reconstruction_to_result(recon)` with direct `PointcloudResult(...)` |
| `collab_splats/pointcloud/__init__.py` | Remove `_colmap_recon_to_result` export |
| `tests/integration/test_pipeline_cu121.py` | Update `FeedforwardResult(pts3d=…)` → `points=…`; replace `colmap_reconstruction_to_result` usage with direct `PointcloudResult` |
| `tests/pointcloud/test_vggtx_creator.py` | Update `frame` assertion: `NERFSTUDIO` → `COLMAP` |
| `tests/pointcloud/test_mapanything_creator.py` | Same |
| `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` | `camera_poses.shape[0]` → `len(result.reconstruction.images)` |

---

## Task 1: Rename `FeedforwardResult.pts3d → points`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py`
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Modify: `collab_splats/pointcloud/utils.py`
- Modify: `tests/integration/test_pipeline_cu121.py`

- [ ] **Step 1: Rename the field in `FeedforwardResult` dataclass**

In `collab_splats/pointcloud/feedforward/base.py`, line ~44, change:
```python
    pts3d: np.ndarray            # (P, 3) float32 — world-space XYZ points
```
to:
```python
    points: np.ndarray           # (P, 3) float32 — world-space XYZ points
```

- [ ] **Step 2: Update `save()` and `load()` in `FeedforwardResult`**

In `save()` (~line 68), change `pts3d=self.pts3d` → `points=self.points` in the arrays dict:
```python
        arrays: dict = dict(
            points=self.points,
            colors=self.colors,
            ...
        )
```

In `load()` (~line 86), change `pts3d=d["pts3d"]` → `points=d["points"]`:
```python
        return cls(
            points=d["points"],
            colors=d["colors"],
            ...
        )
```

- [ ] **Step 3: Update `save_zarr()` and `load_zarr()` in `FeedforwardResult`**

In `save_zarr()`, change the zarr key and field reference:
```python
        for name, arr in (
            ("points", self.points),   # was ("pts3d", self.pts3d)
            ("colors", self.colors),
            ...
        ):
```

In `load_zarr()`, change:
```python
        points = store["points"][:]   # was pts3d = store["pts3d"][:]
```
and update the constructor call: `pts3d=pts3d` → `points=points`.

- [ ] **Step 4: Update `_postprocess` return in `vggtx.py`**

In `collab_splats/pointcloud/feedforward/vggtx.py`, the `_postprocess` method returns `FeedforwardResult(pts3d=pts3d, ...)`. Change to:
```python
        return FeedforwardResult(
            points=pts3d,
            colors=colors,
            ...
        )
```

- [ ] **Step 5: Update `_postprocess` return in `vggt_omega.py`**

In `collab_splats/pointcloud/feedforward/vggt_omega.py`, same pattern:
```python
        return FeedforwardResult(
            points=pts3d,
            colors=colors,
            ...
        )
```

- [ ] **Step 6: Update `_postprocess` return in `mapanything.py`**

In `collab_splats/pointcloud/feedforward/mapanything.py`, same pattern.

- [ ] **Step 7: Update `build_colmap()` in feedforward/base.py**

The `build_colmap` method references `o.pts3d`:
```python
        recon = build_pycolmap_reconstruction(
            o.pts3d, o.colors, o.extrinsics, o.intrinsics,
```
Change to:
```python
        recon = build_pycolmap_reconstruction(
            o.points, o.colors, o.extrinsics, o.intrinsics,
```

Also update the log line at ~line 607:
```python
        n_pts = len(self.outputs.points)   # was self.outputs.pts3d
```

- [ ] **Step 8: Update `lift_features()` in `utils.py`**

`lift_features` checks for required field names and accesses `result.pts3d`:

Find (~line 832):
```python
    for name in ("pts3d", "pixel_indices", "depth", "conf", "extrinsics", "intrinsics"):
```
Change to:
```python
    for name in ("points", "pixel_indices", "depth", "confidence", "extrinsics", "intrinsics"):
```

Find (~line 843 and 851):
```python
    P = result.pts3d.shape[0]
    pts_h = np.concatenate([result.pts3d.astype(np.float64), ...
```
Change both to `result.points`.

- [ ] **Step 9: Update `test_pipeline_cu121.py` construction + assertions**

Find `FeedforwardResult(pts3d=...)` constructor call (~line 105) and change to `points=`:
```python
    original = FeedforwardResult(
        points=rng.standard_normal((P, 3)).astype(np.float32),
        colors=rng.integers(0, 255, (P, 3), dtype=np.uint8),
        ...
    )
```

Find `original.pts3d` in assertions (~line 118) and change to `original.points`:
```python
    np.testing.assert_array_equal(original.points, loaded.points)
```

- [ ] **Step 10: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -30
```

Expected: All tests pass. If any test references `pts3d`, grep for missed callsites:
```bash
grep -rn '\.pts3d\b' /workspace/collab-splats/collab_splats/ /workspace/collab-splats/tests/ --include='*.py' | grep -v __pycache__
```

- [ ] **Step 11: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py \
        collab_splats/pointcloud/feedforward/vggtx.py \
        collab_splats/pointcloud/feedforward/vggt_omega.py \
        collab_splats/pointcloud/feedforward/mapanything.py \
        collab_splats/pointcloud/utils.py \
        tests/integration/test_pipeline_cu121.py
git commit -m "refactor(feedforward): rename FeedforwardResult.pts3d → points"
```

---

## Task 2: Rename `FeedforwardResult.conf → confidence`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py`
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`

- [ ] **Step 1: Rename field in `FeedforwardResult` dataclass**

In `collab_splats/pointcloud/feedforward/base.py`, line ~58, change:
```python
    conf: "torch.Tensor | None" = None           # (N, H, W) confidence scores
```
to:
```python
    confidence: "torch.Tensor | None" = None     # (N, H, W) confidence scores
```

Update the docstring at line ~41 ("Optional fields `images`, `conf`") → "Optional fields `images`, `confidence`".

- [ ] **Step 2: Update `save_zarr()` zarr key**

In `save_zarr()`, find (~line 135):
```python
        if self.conf is not None:
            conf_np = self.conf
```
Change to:
```python
        if self.confidence is not None:
            conf_np = self.confidence
```
And change the zarr key from `"conf"` to `"confidence"`:
```python
            store.create_array("confidence", data=conf_np, chunks=chunks, compressors=lz4)
```

- [ ] **Step 3: Update `load_zarr()` zarr key**

In `load_zarr()`, find (~line 200):
```python
        conf = torch.from_numpy(store["conf"][:]) if "conf" in store else None
```
Change to (checking both keys for backward-compat with old zarr stores):
```python
        _conf_key = "confidence" if "confidence" in store else ("conf" if "conf" in store else None)
        confidence = torch.from_numpy(store[_conf_key][:]) if _conf_key else None
```

Update the constructor call: `conf=conf` → `confidence=confidence`.

- [ ] **Step 4: Update `_postprocess` in `vggtx.py`**

Find the `FeedforwardResult(...)` return statement. Change `conf=conf` → `confidence=conf`:
```python
        return FeedforwardResult(
            points=pts3d,
            colors=colors,
            ...
            confidence=conf,
            world_points=world_points,
            ...
        )
```

- [ ] **Step 5: Update `_postprocess` in `vggt_omega.py`**

Same pattern: `conf=conf` → `confidence=conf`.

- [ ] **Step 6: Update `_postprocess` in `mapanything.py`**

Same pattern.

- [ ] **Step 7: Update `bundle_adjustment.py` `refine()` method**

In `BundleAdjustment.refine()`, find (~line 81):
```python
        tracks, vis_scores, pts3d_tracks = _extract_tracks_vggsfm(
            result.images, result.conf, result.world_points,
```
Change to:
```python
        tracks, vis_scores, pts3d_tracks = _extract_tracks_vggsfm(
            result.images, result.confidence, result.world_points,
```

- [ ] **Step 8: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -30
```

Grep for missed callsites if failures:
```bash
grep -rn '\bconf\b' /workspace/collab-splats/collab_splats/pointcloud/ --include='*.py' | grep -v __pycache__ | grep -v 'depth_conf\|conf_threshold\|conf_mask\|conf_np\|conf_tensor\|_conf\|conf_\|#'
```

- [ ] **Step 9: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py \
        collab_splats/pointcloud/feedforward/vggtx.py \
        collab_splats/pointcloud/feedforward/vggt_omega.py \
        collab_splats/pointcloud/feedforward/mapanything.py \
        collab_splats/pointcloud/bundle_adjustment.py
git commit -m "refactor(feedforward): rename FeedforwardResult.conf → confidence"
```

---

## Task 3: Rewrite `PointcloudResult` in `base.py`

**Files:**
- Modify: `collab_splats/pointcloud/base.py`
- Test: `tests/integration/test_pipeline_cu121.py`

- [ ] **Step 1: Write a failing test for the new `PointcloudResult` API**

Add to `tests/integration/test_pipeline_cu121.py`:

```python
def test_pointcloudresult_new_api():
    """PointcloudResult takes reconstruction as primary; exposes points/colors/extrinsics/intrinsics as properties."""
    from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction
    from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame

    P = 10
    rng = np.random.default_rng(7)
    pts3d = rng.standard_normal((P, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (P, 3), dtype=np.uint8)
    image_names = [f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]
    image_paths = [Path(name) for name in image_names]

    recon = build_pycolmap_reconstruction(
        pts3d=pts3d,
        colors=colors,
        extrinsics=_synthetic_extrinsics(N_FRAMES),
        intrinsics=_synthetic_intrinsics(N_FRAMES),
        image_width=W,
        image_height=H,
        image_names=image_names,
    )

    result = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=image_paths,
    )

    # Properties return correct shapes
    assert result.points.shape == (P, 3)
    assert result.colors.shape == (P, 3)
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)
    assert result.intrinsics.shape == (N_FRAMES, 3, 3)

    # Old stored fields no longer exist
    assert not hasattr(result, "camera_poses")
    assert not hasattr(result, "camera_intrinsics")
    assert not hasattr(result, "colmap_reconstruction")

    # reconstruction is always set
    assert result.reconstruction is recon
    assert result.frame == CoordinateFrame.COLMAP
    assert len(result.image_paths) == N_FRAMES
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/integration/test_pipeline_cu121.py::test_pointcloudresult_new_api -v 2>&1 | tail -20
```

Expected: FAIL — `PointcloudResult` does not accept `reconstruction=` kwarg yet.

- [ ] **Step 3: Rewrite `PointcloudResult` in `base.py`**

Replace the entire `PointcloudResult` dataclass and `_colmap_recon_to_result` function (lines 19–114) with:

```python
@dataclass
class PointcloudResult:
    """Sparse reconstruction output: pycolmap.Reconstruction + scene metadata.

    reconstruction is the primary store for cameras, images, and 3D points.
    frame declares the coordinate system of the world origin in reconstruction.
    image_paths defines the canonical frame ordering for extrinsics/intrinsics.
    """

    reconstruction: pycolmap.Reconstruction          # primary — always set
    frame: CoordinateFrame                           # coord system of world origin
    image_paths: list[Path]                          # canonical frame ordering (N entries)
    confidence: np.ndarray | None = None             # (P,) float32 — feedforward per-point
    world_transform: np.ndarray | None = None        # (3, 4) applied COLMAP→nerfstudio axis swap

    @property
    def points(self) -> np.ndarray:
        """(P, 3) float32 world XYZ of the tracked sparse point set, ordered by point3D_id.

        P is the filtered sparse set — smaller than FeedforwardResult.points which
        contains all feedforward model output including untracked points.
        Recomputes on each access from reconstruction.points3D — always reflects
        current reconstruction state.
        """
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([p.xyz for p in pts3d.values()], dtype=np.float32)

    @property
    def colors(self) -> np.ndarray:
        """(P, 3) uint8 RGB, same order as points."""
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array([p.color for p in pts3d.values()], dtype=np.uint8)

    @property
    def extrinsics(self) -> np.ndarray:
        """(N, 4, 4) float32 w2c transforms, ordered by image_paths.

        Convention: x_cam = E @ x_world (homogeneous). OpenCV camera axes
        (X right, Y down, Z into scene). Frame is declared by self.frame.
        """
        name_to_image = {img.name: img for img in self.reconstruction.images.values()}
        result = []
        for path in self.image_paths:
            img = name_to_image[path.name]
            R = img.cam_from_world.rotation.matrix()
            t = img.cam_from_world.translation
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3] = t
            result.append(E)
        return np.stack(result) if result else np.zeros((0, 4, 4), dtype=np.float32)

    @property
    def intrinsics(self) -> np.ndarray:
        """(N, 3, 3) float32 K matrices (PINHOLE/linear part only), ordered by image_paths.

        K[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        Distortion params are NOT captured here. Access reconstruction.cameras[id]
        directly for the full pycolmap.Camera when distortion-correct projection is needed.
        """
        name_to_image = {img.name: img for img in self.reconstruction.images.values()}
        cameras = self.reconstruction.cameras
        result = []
        for path in self.image_paths:
            img = name_to_image[path.name]
            cam = cameras[img.camera_id]
            params = cam.params   # [fx, fy, cx, cy, ...] — first 4 always fx/fy/cx/cy
            K = np.array([[params[0], 0, params[2]],
                          [0, params[1], params[3]],
                          [0,         0,          1]], dtype=np.float32)
            result.append(K)
        return np.stack(result) if result else np.zeros((0, 3, 3), dtype=np.float32)
```

Also remove the `_WORLD_TRANSFORM` constant and `_colmap_recon_to_result` function — they will be deleted in Task 5.

Keep `CoordinateFrame`, `BasePointcloudCreator`, and all imports. Remove the `Any` import if it becomes unused.

- [ ] **Step 4: Run the new test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/integration/test_pipeline_cu121.py::test_pointcloudresult_new_api -v 2>&1 | tail -20
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/base.py tests/integration/test_pipeline_cu121.py
git commit -m "feat(pointcloud): rewrite PointcloudResult — pycolmap.Reconstruction primary"
```

---

## Task 4: Update `build_colmap()` in `feedforward/base.py`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`

- [ ] **Step 1: Replace `colmap_reconstruction_to_result(recon)` with direct construction**

In `BaseFeedforwardCreator.build_colmap()` (around line 624), the last line is:
```python
        return colmap_reconstruction_to_result(recon)
```

Replace with:
```python
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=o.image_paths,
        )
```

Add the `PointcloudResult` and `CoordinateFrame` imports at the top of the file (they come from `..base`):
```python
from ..base import BasePointcloudCreator, PointcloudResult, CoordinateFrame
```

Remove the `colmap_reconstruction_to_result` import from `..utils` (or just remove it from the import line if other imports remain).

- [ ] **Step 2: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q -k "not test_build_pycolmap_reconstruction_roundtrip" 2>&1 | tail -30
```

(The roundtrip test references `colmap_reconstruction_to_result` — it will be updated in Task 6.)

- [ ] **Step 3: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py
git commit -m "refactor(feedforward): build_colmap() constructs PointcloudResult directly"
```

---

## Task 5: Update `sfm.py` (ColmapCreator + HlocCreator)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py`

- [ ] **Step 1: Replace `colmap_reconstruction_to_result` in `ColmapCreator.reconstruct()`**

In `sfm.py`, `ColmapCreator.reconstruct()` ends with:
```python
        return colmap_reconstruction_to_result(recon)
```

Replace with:
```python
        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=image_paths,
        )
```

- [ ] **Step 2: Replace in `HlocCreator.reconstruct()`**

Same pattern — `HlocCreator.reconstruct()` ends with:
```python
        return colmap_reconstruction_to_result(recon)
```

Replace with:
```python
        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=image_paths,
        )
```

- [ ] **Step 3: Update imports in `sfm.py`**

Remove `from .utils import colmap_reconstruction_to_result`. Add `CoordinateFrame` to the import from `.base`:
```python
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -30
```

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/sfm.py
git commit -m "refactor(sfm): ColmapCreator + HlocCreator construct PointcloudResult directly"
```

---

## Task 6: Delete conversion boilerplate + update imports

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Modify: `collab_splats/pointcloud/base.py`
- Modify: `collab_splats/pointcloud/__init__.py`

- [ ] **Step 1: Delete `colmap_reconstruction_to_result` from `utils.py`**

Remove the entire function (lines 46–76):
```python
def colmap_reconstruction_to_result(
    recon: pycolmap.Reconstruction,
    confidence: np.ndarray | None = None,
) -> PointcloudResult:
    ...
    return _colmap_recon_to_result(recon, confidence)
```

Remove from the import at line 21:
```python
from .base import PointcloudResult, _colmap_recon_to_result
```
Change to:
```python
from .base import PointcloudResult
```

Remove the `COLMAP → result conversion` section header comments.

- [ ] **Step 2: Delete `_colmap_recon_to_result` and `_WORLD_TRANSFORM` from `base.py`**

These should already be gone if done in Task 3. Verify:
```bash
grep -n '_colmap_recon_to_result\|_WORLD_TRANSFORM' /workspace/collab-splats/collab_splats/pointcloud/base.py
```

Expected: no output.

- [ ] **Step 3: Update `pointcloud/__init__.py`**

Line 2 currently exports `_colmap_recon_to_result`:
```python
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult, _colmap_recon_to_result
```

Change to:
```python
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -30
```

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py collab_splats/pointcloud/base.py collab_splats/pointcloud/__init__.py
git commit -m "refactor(pointcloud): delete colmap_reconstruction_to_result boilerplate"
```

---

## Task 7: Update tests

**Files:**
- Modify: `tests/integration/test_pipeline_cu121.py`
- Modify: `tests/pointcloud/test_vggtx_creator.py`
- Modify: `tests/pointcloud/test_mapanything_creator.py`

- [ ] **Step 1: Rewrite `test_build_pycolmap_reconstruction_roundtrip` in `test_pipeline_cu121.py`**

The test currently calls `colmap_reconstruction_to_result` (deleted) and checks `camera_poses`/`camera_intrinsics` (removed). Replace:

```python
def test_build_pycolmap_reconstruction_roundtrip():
    """build_pycolmap_reconstruction → PointcloudResult with reconstruction primary."""
    from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction
    from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame

    P = 30
    rng = np.random.default_rng(42)
    pts3d = rng.standard_normal((P, 3)).astype(np.float32)
    colors = rng.integers(0, 255, (P, 3), dtype=np.uint8)
    image_names = [f"frame_{i:04d}.jpg" for i in range(N_FRAMES)]
    image_paths = [Path(name) for name in image_names]

    recon = build_pycolmap_reconstruction(
        pts3d=pts3d,
        colors=colors,
        extrinsics=_synthetic_extrinsics(N_FRAMES),
        intrinsics=_synthetic_intrinsics(N_FRAMES),
        image_width=W,
        image_height=H,
        image_names=image_names,
    )
    assert len(recon.images) == N_FRAMES
    assert len(recon.cameras) == N_FRAMES
    assert len(recon.points3D) == P

    result = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=image_paths,
    )
    assert result.points.shape == (P, 3)
    assert result.extrinsics.shape == (N_FRAMES, 4, 4)
    assert result.intrinsics.shape == (N_FRAMES, 3, 3)
    assert result.frame == CoordinateFrame.COLMAP
```

- [ ] **Step 2: Update `test_feedforward_result_save_load` in `test_pipeline_cu121.py`**

Construction uses `pts3d=` → `points=`:
```python
    original = FeedforwardResult(
        points=rng.standard_normal((P, 3)).astype(np.float32),
        ...
    )
```

Assertion uses `original.pts3d` → `original.points`:
```python
    np.testing.assert_array_equal(original.points, loaded.points)
```

- [ ] **Step 3: Update `frame` assertions in `test_vggtx_creator.py`**

Lines ~101-102 assert `CoordinateFrame.NERFSTUDIO`. After the refactor, `build_colmap()` sets `frame=CoordinateFrame.COLMAP`. Change:
```python
    assert result.frame == CoordinateFrame.COLMAP    # was NERFSTUDIO
```

Also update any `result.camera_poses` or `result.camera_intrinsics` references to use the new API:
```python
    assert len(result.image_paths) == N               # frame count via image_paths
    assert result.extrinsics.shape == (N, 4, 4)
    assert result.intrinsics.shape == (N, 3, 3)
```

- [ ] **Step 4: Update `frame` assertions in `test_mapanything_creator.py`**

Same as step 3 — `CoordinateFrame.NERFSTUDIO` → `CoordinateFrame.COLMAP` and update field references.

- [ ] **Step 5: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -q 2>&1 | tail -30
```

Expected: all pass. If any test still references old field names, grep:
```bash
grep -rn 'camera_poses\|camera_intrinsics\|colmap_reconstruction\|\.pts3d\b\|\.conf\b' \
    /workspace/collab-splats/tests/ --include='*.py' | grep -v __pycache__
```

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_pipeline_cu121.py \
        tests/pointcloud/test_vggtx_creator.py \
        tests/pointcloud/test_mapanything_creator.py
git commit -m "test(pointcloud): update assertions for new PointcloudResult API"
```

---

## Task 8: Update notebooks

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`

- [ ] **Step 1: Update the notebook cell that reads `camera_poses`**

In `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`, find the cell containing:
```python
print(f"COLMAP: {len(colmap_result.points):,} points, {colmap_result.camera_poses.shape[0]} cameras")
```

Change to:
```python
print(f"COLMAP: {len(colmap_result.points):,} points, {len(colmap_result.image_paths)} cameras")
```

- [ ] **Step 2: Run full test suite one final time**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -q 2>&1 | tail -20
```

Expected: all pass, no warnings about old field names.

- [ ] **Step 3: Final grep for stale references**

```bash
grep -rn 'camera_poses\|camera_intrinsics\|colmap_reconstruction\b\|_colmap_recon_to_result\|colmap_reconstruction_to_result\|\.pts3d\b\|result\.conf\b' \
    /workspace/collab-splats/collab_splats/ /workspace/collab-splats/tests/ --include='*.py' | \
    grep -v __pycache__ | grep -v third_party
```

Expected: no output (or only legitimate uses like `colmap_reconstruction_to_result` in test docstrings).

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
git commit -m "docs(notebook): update bundle_adjustment notebook for new PointcloudResult API"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `pycolmap.Reconstruction` primary with typed field — Task 3
- ✅ `image_paths: list[Path]` for frame ordering — Task 3
- ✅ Four `@property` views (`points`, `colors`, `extrinsics`, `intrinsics`) — Task 3
- ✅ `pts3d→points`, `conf→confidence` renames — Tasks 1, 2
- ✅ Delete `_colmap_recon_to_result` + `colmap_reconstruction_to_result` — Tasks 5, 6
- ✅ `build_colmap()` direct construction — Task 4
- ✅ `sfm.py` ColmapCreator + HlocCreator updated — Task 5
- ✅ `lift_features` field refs updated — Task 1 Step 8
- ✅ `bundle_adjustment.py` `result.conf` updated — Task 2 Step 7
- ✅ `__init__.py` export cleaned — Task 6 Step 3
- ✅ Tests updated — Task 7
- ✅ Notebooks updated — Task 8

**Notes:**
- `confidence` zarr key: load_zarr checks both `"confidence"` (new) and `"conf"` (old) for backward compat with existing zarr stores
- `frame` in `build_colmap()` is set to `CoordinateFrame.COLMAP` — tests checking `NERFSTUDIO` must be updated (Task 7 Steps 3-4)
- `sfm.py` image ordering uses `sorted(..., key=lambda p: p.name)` since COLMAP SfM doesn't guarantee insertion order
- `world_transform` is `None` in feedforward path (no axis transform applied); `sfm.py` also sets `None` since transform is not applied — callers using nerfstudio integration should check `result.frame` and apply `_WORLD_TRANSFORM` if needed (out of scope for this PR)
