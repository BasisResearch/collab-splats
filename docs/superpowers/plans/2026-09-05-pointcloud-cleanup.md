# Pointcloud Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `collab_splats/pointcloud/` to the one SfM path that actually ships (InstantSfM + Video-Depth-Anything metric depth + per-frame scale alignment), splitting the 1289-line `sfm.py` into a `sfm/` package plus `vda.py` and `depth_align.py`, and deleting every dead branch, config key, and helper that path does not use.

**Architecture:** `PointcloudResult` shrinks to `(reconstruction, image_paths)` and grows two methods (`from_colmap`, `write_ply`) that absorb `export.py` and `Reconstructor._load_pointcloud_from_disk`. `sfm.py` splits by responsibility: `vda.py` (depth generation), `depth_align.py` (COLMAP-scale depth → `FeedforwardResult`), and `sfm/{colmap,hloc,instantsfm}.py` (one creator per file). `Reconstructor._run_sfm` becomes a thin ~25-line orchestrator over those modules, and every pointcloud import in `reconstructor.py` moves to the module top.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pycolmap 4.0.4, open3d, numpy, zarr, pytest, black + isort (never repo-wide), Video-Depth-Anything (`third_party/Video-Depth-Anything`), InstantSfM (optional, not installed in this venv).

**Source spec:** `docs/superpowers/specs/2026-09-05-pointcloud-cleanup-design.md` (rev 4).

---

## Ground Rules

- Python is **always** `/opt/venv/reconstruction/bin/python`. Never bare `python`.
- Formatting is **never** repo-wide. `black <files> && isort <files>` on the files that task touched, nothing else.
- flake8's config is inert in this repo: E501 fires at 79 columns, and `isort` (profile=black) wraps at 88. Write long import lists parenthesized.
- InstantSfM is **not installed** in this venv. Any test that touches `InstantSfMCreator.reconstruct` must stub the `instantsfm.*` imports; tests only ever exercise config building and the helpers around it.
- Every task ends green: `- [ ] Step N: Commit` only after the task's test command passes.
- Commit style: conventional commits with scope, e.g. `refactor(pointcloud): ...`.

---

## File Structure

**After this plan:**

```
collab_splats/pointcloud/
  __init__.py        # registry: get_creator, make_creator(name, **kwargs); re-exports the 4 creators
  base.py            # PointcloudResult(reconstruction, image_paths) + from_colmap/write_ply; BasePointcloudCreator
  utils.py           # clean_pointcloud + the feature-lifting / plane-fitting helpers that survive
  vda.py             # Video-Depth-Anything metric depth: _load_vda_model, generate_vda_depth
  depth_align.py     # result_from_reconstruction: VDA depth rescaled to COLMAP -> FeedforwardResult
  sfm/
    __init__.py      # re-exports ColmapCreator, HlocCreator, InstantSfMCreator
    colmap.py        # ColmapCreator (pycolmap incremental)
    hloc.py          # HlocCreator (netvlad + superpoint + superglue)
    instantsfm.py    # InstantSfMCreator, SIFT database, upstream patches, stem rename
  feedforward/       # untouched except two lines in feedforward/base.py

tests/pointcloud/
  test_base.py           # from_colmap / write_ply / abstract creator
  test_pointcloud_utils.py
  test_vda.py            # NEW — generate_vda_depth
  test_depth_align.py    # result_from_reconstruction + its privates
  sfm/
    __init__.py          # NEW (empty)
    test_colmap.py       # NEW
    test_hloc.py         # NEW
    test_instantsfm.py   # NEW
```

**Deleted files:**

| File | Why |
|---|---|
| `collab_splats/pointcloud/export.py` | folded into `PointcloudResult.write_ply` |
| `collab_splats/pointcloud/sfm.py` | split into `vda.py`, `depth_align.py`, `sfm/` |
| `tests/pointcloud/test_export.py` | tests a deleted module |
| `tests/pointcloud/test_export_wiring.py` | tests a deleted module |
| `tests/pointcloud/test_instantsfm.py` | split into `test_vda.py` + `test_depth_align.py` + `sfm/test_instantsfm.py` |
| `tests/pointcloud/test_sfm_creator.py` | split into `tests/pointcloud/sfm/test_{colmap,hloc,instantsfm}.py` |
| `tests/wrapper/test_sfm_result.py` | result-assembly cases move to `test_depth_align.py`; the rename case moves to `sfm/test_instantsfm.py` with the helper it covers |
| `tests/wrapper/test_vda_context.py` | the VDA context stream is deleted |
| `tests/wrapper/test_transforms_json.py` | `transforms.json` is no longer written |

---

## Task 1: `PointcloudResult.write_ply` and `PointcloudResult.from_colmap`

Additive only — nothing is deleted here, so the tree stays green. Task 2 switches the callers over.

**Files:**
- Modify: `collab_splats/pointcloud/base.py:18-31` (docstring + imports) and append two methods
- Test: `tests/pointcloud/test_base.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_base.py`:

```python
def _recon_with_images(names=("frame_000000",)):
    """
    PINHOLE camera + one image per name (translation z = index) + one coloured point3D.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=8, height=6, params=[4.0, 4.0, 4.0, 3.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
    recon.add_point3D(
        xyz=np.array([1.0, 2.0, 3.0]),
        track=pycolmap.Track(),
        color=np.array([10, 20, 30], dtype=np.uint8),
    )
    return recon


def _write_model(tmp_path, recon):
    """
    Write recon to <tmp_path>/colmap/sparse/0 and return the <tmp_path>/colmap dir.
    """
    sparse = tmp_path / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True)
    recon.write_binary(str(sparse))
    return tmp_path / "colmap"


def test_write_ply_roundtrips_through_open3d(tmp_path):
    """
    write_ply exports xyz+rgb from the reconstruction and creates missing parents.
    """
    result = PointcloudResult(
        reconstruction=_recon_with_images(),
        frame=CoordinateFrame.COLMAP,
        image_paths=[Path("frame_000000")],
    )
    out = tmp_path / "nested" / "sparse_pc.ply"

    result.write_ply(out)

    pcd = o3d.io.read_point_cloud(str(out))
    np.testing.assert_allclose(np.asarray(pcd.points), [[1.0, 2.0, 3.0]], atol=1e-5)
    np.testing.assert_allclose(np.asarray(pcd.colors), [[10 / 255, 20 / 255, 30 / 255]], atol=2e-3)


def test_from_colmap_reads_the_model_and_keeps_caller_order(tmp_path):
    """
    from_colmap loads <colmap_dir>/sparse/0 and orders extrinsics by the caller's image_paths.
    """
    colmap_dir = _write_model(tmp_path, _recon_with_images(["frame_000000", "frame_000001"]))

    r = PointcloudResult.from_colmap(colmap_dir, [Path("frame_000001"), Path("frame_000000")])

    assert r.image_paths == [Path("frame_000001"), Path("frame_000000")]
    assert r.extrinsics.shape == (2, 4, 4)
    # extrinsics are w2c: image frame_000001 was placed at translation z = 1.0
    np.testing.assert_allclose(r.extrinsics[0][2, 3], 1.0, atol=1e-6)
    np.testing.assert_allclose(r.extrinsics[1][2, 3], 0.0, atol=1e-6)


def test_from_colmap_rejects_names_missing_from_the_model(tmp_path):
    """
    A requested image the model never registered is a hard error, not a silent KeyError later.
    """
    colmap_dir = _write_model(tmp_path, _recon_with_images(["frame_000000"]))

    with pytest.raises(ValueError, match="frame_000001"):
        PointcloudResult.from_colmap(colmap_dir, [Path("frame_000000"), Path("frame_000001")])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_base.py -x -q -k "write_ply or from_colmap"
```

Expected: FAIL — `AttributeError: 'PointcloudResult' object has no attribute 'write_ply'` and `AttributeError: type object 'PointcloudResult' has no attribute 'from_colmap'`.

- [ ] **Step 3: Implement both methods**

In `collab_splats/pointcloud/base.py`, insert these two methods into `PointcloudResult` immediately after the dataclass fields and before the `points` property:

```python
    @classmethod
    def from_colmap(cls, colmap_dir: Path, image_paths: list[Path]) -> "PointcloudResult":
        """
        Load a written COLMAP model from ``<colmap_dir>/sparse/0`` in the caller's frame order.

        - ``image_paths`` is the canonical ordering; every entry must be registered in the model.
        - Names are matched on ``Path.name``, the same key ``extrinsics``/``intrinsics`` use.
        """
        sparse_dir = Path(colmap_dir) / "sparse" / "0"
        recon = pycolmap.Reconstruction()
        recon.read(str(sparse_dir))

        # The frame-store naming is a contract nothing enforces at write time. Check it here:
        # unchecked, a mismatch surfaces as a bare KeyError from .extrinsics several frames into
        # a downstream stage, saying nothing about which two artifacts disagree.
        registered = {img.name for img in recon.images.values()}
        missing = [p.name for p in image_paths if p.name not in registered]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(image_paths)} requested frames are not registered in "
                f"{sparse_dir} (first: {missing[0]}); the frame store and the reconstruction "
                f"describe different runs."
            )

        return cls(reconstruction=recon, frame=CoordinateFrame.COLMAP, image_paths=list(image_paths))

    def write_ply(self, path: Path) -> None:
        """
        Write the binary little-endian sparse_pc.ply for this reconstruction's point set.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.reconstruction.export_PLY(str(path))
```

Note: `frame` is still a required field in this task (Task 3 deletes it), so `from_colmap`
passes `frame=CoordinateFrame.COLMAP` — the value every existing construction site already
passes. `CoordinateFrame` is defined in this same module (`base.py:13`), so no import moves.
The dataclass itself is untouched here; Task 3 drops both the field and this argument.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_base.py -x -q
```

Expected: PASS (all tests in the file, including the pre-existing ones).

- [ ] **Step 5: Verify `export_PLY` matches the old writer byte-for-byte**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_export.py tests/pointcloud/test_export_wiring.py -q
```

Expected: PASS — `export.py` is still in place and untouched; this confirms Task 1 broke nothing.

- [ ] **Step 6: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/base.py tests/pointcloud/test_base.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/base.py tests/pointcloud/test_base.py
git add collab_splats/pointcloud/base.py tests/pointcloud/test_base.py
git commit -m "feat(pointcloud): PointcloudResult.from_colmap and .write_ply"
```

---

## Task 2: Delete `export.py`; every PLY write goes through `write_ply`

**Files:**
- Delete: `collab_splats/pointcloud/export.py`, `tests/pointcloud/test_export.py`, `tests/pointcloud/test_export_wiring.py`
- Modify: `collab_splats/pointcloud/base.py` (drop `BasePointcloudCreator._write_ply`)
- Modify: `collab_splats/pointcloud/feedforward/base.py:1157-1163`
- Modify: `collab_splats/wrapper/reconstructor.py:25` (import), `:1008-1012`, `:1019-1050`, `:1088-1096`
- Modify: `configs/base.yaml:93` (`pointcloud.export_max_points`)
- Test: `tests/wrapper/test_reconstructor_export.py`

- [ ] **Step 1: Rewrite the export test to describe the new behaviour**

Replace the whole contents of `tests/wrapper/test_reconstructor_export.py` with:

```python
"""The post-clean sparse_pc.ply written by build_pointcloud."""

from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

from collab_splats.pointcloud.base import CoordinateFrame, PointcloudResult


def test_build_pointcloud_ply_is_readable_by_open3d(tmp_path):
    """
    The PLY written from a PointcloudResult round-trips through open3d with xyz+rgb intact.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=8, height=6, params=[4.0, 4.0, 4.0, 3.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    recon.add_image_with_trivial_frame(
        pycolmap.Image(name="frame_000000", camera_id=1, image_id=1), pycolmap.Rigid3d()
    )
    for i in range(3):
        recon.add_point3D(
            xyz=np.array([float(i), 0.0, 1.0]),
            track=pycolmap.Track(),
            color=np.array([i, 2 * i, 3 * i], dtype=np.uint8),
        )

    out = tmp_path / "backend" / "sparse_pc.ply"
    result = PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=[Path("frame_000000")],
    )
    result.write_ply(out)

    pcd = o3d.io.read_point_cloud(str(out))
    assert np.asarray(pcd.points).shape == (3, 3)
    assert out.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_export.py -x -q
```

Expected: PASS. There is no red phase here: Task 1 already added `write_ply`, and Step 1 replaced
the whole file, so the three old tests that exercised `export_max_points` are already gone. This step
is the baseline proving the new test drives `PointcloudResult.write_ply` and not the doomed
`export.py` path. `frame` is still a required field until Task 3, which is why the test passes it.

- [ ] **Step 3: Delete `export.py` and its tests**

```bash
git rm collab_splats/pointcloud/export.py tests/pointcloud/test_export.py tests/pointcloud/test_export_wiring.py
```

- [ ] **Step 4: Delete `BasePointcloudCreator._write_ply`**

In `collab_splats/pointcloud/base.py`, delete the entire `_write_ply` method (the last method in the file, including its inline-import comment). Also fix the abstract `reconstruct` docstring, replacing:

```
            {output_dir}/sparse_pc.ply   (binary little-endian from the feedforward path; see pointcloud/export.py)

        transforms.json is written by Reconstructor._write_transforms_json from the returned result.
```

with:

```
            {output_dir}/sparse_pc.ply   (binary little-endian; PointcloudResult.write_ply)
```

- [ ] **Step 5: Switch the feedforward writer over**

In `collab_splats/pointcloud/feedforward/base.py`, replace lines 1157-1163:

```python
        # Write the binary sparse_pc.ply
        result = PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=o.image_paths,
        )
        self._write_ply(result, Path(output_dir))
```

with:

```python
        # Write the binary sparse_pc.ply
        result = PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=o.image_paths,
        )
        result.write_ply(Path(output_dir) / "sparse_pc.ply")
```

Only the last line changes — `frame` is still a required field until Task 3 drops it.

- [ ] **Step 6: Switch the wrapper over**

In `collab_splats/wrapper/reconstructor.py`:

1. Delete the import on line 25: `from collab_splats.pointcloud.export import write_pointcloud_ply`.
2. Delete the whole `_export_pointcloud_ply` method (lines 1088-1096).
3. Replace the call block at lines 1008-1012:

```python
        # Re-export the PLY from the FINAL result — clean may have dropped points since
        # the creator wrote its copy. Density is opt-in via pointcloud.export_max_points.
        self._export_pointcloud_ply(result)
```

with:

```python
        # Re-export the PLY from the FINAL result — clean may have dropped points since
        # the creator wrote its own copy.
        result.write_ply(self.backend_dir / "sparse_pc.ply")
```

4. Replace the body of `_load_pointcloud_from_disk` (lines 1019-1050) with a thin delegation — tests call this method by name, so the method itself stays:

```python
    def _load_pointcloud_from_disk(self) -> "PointcloudResult":
        """
        Load the written COLMAP model into a PointcloudResult in frames.zarr order.
        """
        # Rebuild image_paths from frames.zarr in store order, so it lines up with the per-frame
        # arrays the downstream stages index. The creators register COLMAP images as
        # frame_{source_idx:06d} with NO extension — the frame_*.jpg spelling elsewhere is the
        # zarr/localization id namespace, not this one.
        frame_indices = FrameStore.open(self.frames_zarr).frame_indices()
        image_paths = [Path(f"frame_{int(fi):06d}") for fi in frame_indices]
        return PointcloudResult.from_colmap(self.backend_dir / "colmap", image_paths)
```

`PointcloudResult` is already imported inside `_run_sfm` at line 1148; leave that alone for now (Task 11 hoists it) but add a module-top import so this method resolves it:

```python
from collab_splats.pointcloud.base import PointcloudResult
```

placed with the other `collab_splats.pointcloud` imports near line 25.

- [ ] **Step 7: Drop the `export_max_points` config key**

In `configs/base.yaml`, delete line 93 and any comment line directly above it:

```yaml
  export_max_points: null      # cap points written to sparse_pc.ply (null = all)
```

- [ ] **Step 8: Run the affected tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper -x -q
```

Expected: PASS.

- [ ] **Step 9: Confirm nothing still references the deleted module**

```bash
rtk proxy grep -rn "pointcloud.export\|write_pointcloud_ply\|export_max_points\|_write_ply\|_export_pointcloud_ply" collab_splats tests configs evals docs/source
```

Expected: no output.

- [ ] **Step 10: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/base.py collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor_export.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/base.py collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor_export.py
git add -A collab_splats/pointcloud collab_splats/wrapper/reconstructor.py tests configs/base.yaml
git commit -m "refactor(pointcloud): fold export.py into PointcloudResult.write_ply"
```

---

## Task 3: Delete `CoordinateFrame`, `.frame`, `.world_transform`, `.confidence`

Every producer already sets `frame=CoordinateFrame.COLMAP` and nothing reads it; nothing anywhere
constructs `PointcloudResult` with `world_transform=` or `confidence=`. The `.confidence` reads in
`geometry/metrics.py`, `mesh/utils.py`, `pointcloud/utils.py` and `feedforward/base.py` are all on
`FeedforwardResult` (per-pixel `(N, H, W)`), not on `PointcloudResult` (per-point `(P,)`) — leave
them alone.

**Files:**
- Modify: `collab_splats/pointcloud/base.py:6,13-15,18-31,60-64`
- Modify: `collab_splats/pointcloud/__init__.py:2,79`
- Modify: `collab_splats/pointcloud/sfm.py:21,95,215`
- Modify: `collab_splats/pointcloud/feedforward/base.py:42`
- Modify: `collab_splats/wrapper/reconstructor.py:1148,1210`
- Modify: `collab_splats/pointcloud/utils.py:6` (comment), `collab_splats/geometry/transforms.py:5` (comment)
- Test: `tests/pointcloud/test_base.py`, `tests/wrapper/test_reconstructor_export.py`, `tests/pointcloud/test_sfm_creator.py:7,70`, `tests/pointcloud/test_vggtx_creator.py:8,39`, `tests/pointcloud/feedforward/test_mapanything_creator.py`, `tests/integration/test_pipeline_cu121.py:70,95,101,255,276,293`

- [ ] **Step 1: Delete the frame-related tests and the `frame=` kwargs from the surviving ones**

In `tests/pointcloud/test_base.py`:
- Delete `test_result_colmap_frame`, `test_result_nerfstudio_frame`, and `test_coordinate_frame_values` entirely.
- Delete `CoordinateFrame,` from the `collab_splats.pointcloud.base` import block.
- Delete the `frame=CoordinateFrame.COLMAP,` line from `test_result_fields` and from
  `test_write_ply_roundtrips_through_open3d` (added in Task 1).

Sweep every remaining site:

```bash
rtk proxy grep -rn "CoordinateFrame\|world_transform" collab_splats tests evals docs/source configs
```

In each hit: delete `CoordinateFrame` from the import list, delete the `frame=CoordinateFrame.COLMAP,`
kwarg, and delete any `assert result.frame == ...` / `assert r.world_transform is None` line.
The known hits are:

| File | Action |
|---|---|
| `collab_splats/pointcloud/__init__.py:2,79` | drop from import and from `__all__` |
| `collab_splats/pointcloud/sfm.py:21,95,215` | drop from import and both `PointcloudResult(...)` calls |
| `collab_splats/pointcloud/feedforward/base.py:42` | drop from the `..base` import |
| `collab_splats/pointcloud/feedforward/base.py` (the `PointcloudResult(...)` in the PLY writer) | drop the `frame=` kwarg |
| `tests/wrapper/test_reconstructor_export.py` | drop from the import and from the `PointcloudResult(...)` call (both added in Task 2) |
| `collab_splats/wrapper/reconstructor.py:1148,1210` | drop from the function-local import and the `PointcloudResult(...)` in `_run_sfm` |
| `tests/pointcloud/test_sfm_creator.py:7,70` | drop import + `assert result.frame == CoordinateFrame.COLMAP` |
| `tests/pointcloud/test_vggtx_creator.py:8,39` | same |
| `tests/pointcloud/feedforward/test_mapanything_creator.py` | same |
| `tests/integration/test_pipeline_cu121.py:70,95,101,255,276,293` | drop both local imports, both `frame=` kwargs, both `assert result.frame` lines |

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud -x -q
```

Expected: FAIL — `ImportError: cannot import name 'CoordinateFrame'` is *not* it yet (the enum still
exists); the failure is `TypeError: PointcloudResult.__init__() got an unexpected keyword argument`
only after Step 3. At this step the suite should still PASS; that is the baseline confirming the test
edits are self-consistent.

- [ ] **Step 3: Delete the enum and the three fields**

In `collab_splats/pointcloud/base.py`:
- Delete `from enum import Enum` (line 6).
- Delete the whole `class CoordinateFrame(str, Enum):` block (lines 13-15).
- Replace the dataclass docstring and field block with:

```python
@dataclass
class PointcloudResult:
    """
    Sparse reconstruction output: a pycolmap.Reconstruction plus its canonical frame order.

    - ``reconstruction`` is the primary store for cameras, images, and 3D points.
    - ``image_paths`` defines the ordering of ``extrinsics``/``intrinsics``.
    """

    reconstruction: pycolmap.Reconstruction  # primary — always set
    image_paths: list[Path]  # canonical frame ordering (N entries)
```

- In `from_colmap` (Task 1), drop the now-dead argument so the return reads
  `return cls(reconstruction=recon, image_paths=list(image_paths))`.
- In the `extrinsics` docstring, delete the trailing sentence `Frame is declared by self.frame.`
  so it reads:

```python
        """(N, 4, 4) float32 w2c transforms, ordered by image_paths.

        Convention: x_cam = E @ x_world (homogeneous). OpenCV camera axes
        (X right, Y down, Z into scene).
        """
```

- [ ] **Step 4: Fix the two stale comments that name the enum**

`collab_splats/geometry/transforms.py:5`:

```python
  OpenGL camera axes:  X right, Y up,    Z backward  (nerfstudio / OpenGL convention)
```

`collab_splats/pointcloud/utils.py:6`:

```python
  - All public functions that produce poses output the nerfstudio world frame (c2w, OpenGL axes, +Z up):
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/integration/test_pipeline_cu121.py -x -q
```

Expected: PASS.

- [ ] **Step 6: Confirm the symbol is gone**

```bash
rtk proxy grep -rn "CoordinateFrame\|world_transform\|\.frame\b" collab_splats tests evals | grep -v "cam_from_world\|world_from_cam"
```

Expected: no `CoordinateFrame` or `world_transform` hits.

- [ ] **Step 7: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/base.py collab_splats/pointcloud/__init__.py collab_splats/pointcloud/sfm.py collab_splats/pointcloud/utils.py collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py collab_splats/geometry/transforms.py tests/pointcloud/test_base.py tests/pointcloud/test_sfm_creator.py tests/pointcloud/test_vggtx_creator.py tests/pointcloud/feedforward/test_mapanything_creator.py tests/integration/test_pipeline_cu121.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/base.py collab_splats/pointcloud/__init__.py collab_splats/pointcloud/sfm.py collab_splats/pointcloud/utils.py collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py collab_splats/geometry/transforms.py tests/pointcloud/test_base.py tests/pointcloud/test_sfm_creator.py tests/pointcloud/test_vggtx_creator.py tests/pointcloud/feedforward/test_mapanything_creator.py tests/integration/test_pipeline_cu121.py
git add -A collab_splats tests
git commit -m "refactor(pointcloud): drop CoordinateFrame, frame, world_transform, confidence from PointcloudResult"
```

---

## Task 4: Delete `transforms.json`

Nothing in the pipeline reads it — the splats, mesh and localization stages all read
`pointcloud.zarr` and the COLMAP model. `FrameStore.frame_idx_from_path` survives: it has five
other live callers (`geometry/metrics.py:681`, `dashboard/localize.py:686`,
`wrapper/reconstructor.py:1776,1809`).

**Files:**
- Delete: `tests/wrapper/test_transforms_json.py`
- Modify: `collab_splats/wrapper/reconstructor.py:1013-1014` (call site), `:1098-1136` (method)
- Modify: `configs/README.md:238,503,555,616,624`

- [ ] **Step 1: Delete the test file**

```bash
git rm tests/wrapper/test_transforms_json.py
```

- [ ] **Step 2: Delete the call site**

In `collab_splats/wrapper/reconstructor.py`, delete these two lines from `build_pointcloud`:

```python
        # Write the pose+intrinsics transforms.json alongside the COLMAP model
        self._write_transforms_json(result)
```

- [ ] **Step 3: Delete the method**

Delete the entire `_write_transforms_json` method (from `def _write_transforms_json` through the
`logger.info("transforms.json written to %s", out)` line).

- [ ] **Step 4: Check whether `json` is still used**

```bash
rtk proxy grep -n "json\." collab_splats/wrapper/reconstructor.py | head
```

If there are no remaining `json.` uses, delete `import json` from the module top. (There are other
uses at time of writing — expect the import to stay.)

- [ ] **Step 5: Drop `transforms.json` from the docs**

In `configs/README.md`, remove `transforms.json` from the output listings at lines 238, 503, 555,
616 and 624 — in each case the file is one entry in a bulleted or tabular list of stage outputs;
delete just that entry, leaving the surrounding list intact.

- [ ] **Step 6: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper -x -q
```

Expected: PASS.

- [ ] **Step 7: Confirm no references remain**

```bash
rtk proxy grep -rn "transforms.json\|_write_transforms_json" collab_splats tests configs evals docs/source
```

Expected: no output.

- [ ] **Step 8: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/wrapper/reconstructor.py
/opt/venv/reconstruction/bin/isort collab_splats/wrapper/reconstructor.py
git add -A collab_splats/wrapper/reconstructor.py configs/README.md tests
git commit -m "refactor(wrapper): stop writing transforms.json"
```

---

## Task 5: Prune `utils.py` to the survivor set; `clean_pointcloud` returns a keep-mask

`utils.py` is 980 lines of which the shipping pipeline uses eight functions. The three-step
`clean_pointcloud(pcd, downsample_kwargs, outlier_kwargs, distance_kwargs)` becomes one statistical
outlier keep-mask over a `(P, 3)` array, and `Reconstructor._clean_pointcloud` disappears into six
lines in `build_pointcloud`. (`voxel_size` in the old wrapper method was already a no-op: the
`voxel_down_sample` result was assigned to a local and discarded.)

**Survivors:** `confidence_mask`, `subsample_points`, `fit_dominant_plane`, `_grid_sample_at_pixels`,
`_sample_at_source_pixels`, `lift_features`, `reproject_pixels`, `cross_frame_attention_ratio`,
plus the rewritten `clean_pointcloud`.

**Files:**
- Modify: `collab_splats/pointcloud/utils.py` (delete lines 23, 35-40, 48-175, 244-298, 344-499, 535-695; rewrite 176-243; hoist imports)
- Modify: `collab_splats/pointcloud/__init__.py:23,85,86`
- Modify: `collab_splats/wrapper/reconstructor.py:1005-1007,1052-1086`
- Modify: `configs/base.yaml:103-107`
- Modify: `docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb:97,184`
- Modify: `docs/source/tutorials/03_loop_closure/slam_loop_closure.ipynb:47,1172,1175`
- Test: `tests/pointcloud/test_pointcloud_utils.py`, `tests/pointcloud/test_base.py:11`, `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test for the new `clean_pointcloud`**

In `tests/pointcloud/test_pointcloud_utils.py`, replace the import block at lines 1-15 with:

```python
import numpy as np
import open3d as o3d
import pytest

from collab_splats.pointcloud.utils import (
    clean_pointcloud,
    cross_frame_attention_ratio,
    fit_dominant_plane,
)
```

Delete these test blocks entirely: `filter_distance` (lines 26-118), `voxel_downsample`
(119-150), every `clean_pointcloud` kwargs/logging case (151-201 and 256-352), the OBB cases
(353-384), and `get_points_in_mask` (385-415). Keep `cross_frame_attention_ratio` (202-255) and
`fit_dominant_plane` (416-end).

Add:

```python
########################################################
########## clean_pointcloud ############################
########################################################


def test_clean_pointcloud_masks_the_far_outlier():
    """
    A tight cluster plus one distant point: the mask keeps the cluster, drops the outlier.
    """
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(200, 3))
    points = np.vstack([cluster, [[50.0, 50.0, 50.0]]])

    keep = clean_pointcloud(points)

    assert keep.dtype == bool
    assert keep.shape == (201,)
    assert keep[:200].all()
    assert not keep[200]


def test_clean_pointcloud_keeps_everything_when_too_few_points():
    """
    Below the neighbourhood size open3d cannot form a statistic — keep all rather than throw.
    """
    keep = clean_pointcloud(np.zeros((5, 3), dtype=np.float32))

    assert keep.shape == (5,)
    assert keep.all()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py -x -q -k clean_pointcloud
```

Expected: FAIL — `TypeError: clean_pointcloud() got an unexpected keyword argument` / the old
signature returns a tuple, so `keep.dtype` raises `AttributeError: 'tuple' object has no attribute 'dtype'`.

- [ ] **Step 3: Rewrite `clean_pointcloud` and delete the dead helpers**

In `collab_splats/pointcloud/utils.py`:

1. Hoist the imports. Replace lines 15-23 with:

```python
import numpy as np
import open3d as o3d
import pycolmap
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    invert_poses,
    rotation_align_vectors,
)
```

(`from .base import PointcloudResult` on line 23 is unused — deleting it also removes the
`base ↔ utils` import cycle that forced `_write_ply`'s function-local import.)

2. Delete lines 35-40 (`_DEFAULT_DOWNSAMPLE_KWARGS`, `_DEFAULT_OUTLIER_KWARGS`,
   `_DEFAULT_DISTANCE_KWARGS`, `_UNSET` and its comment).

3. Delete these functions entirely: `_radial_mask`, `_bbox_mask`, `filter_distance`,
   `voxel_downsample`, `clean_pcd`, `remove_far_points`, `density_filter`,
   `compute_obb_from_points`, `get_points_in_mask`, `voxel_downsample_point_cloud`.

4. Delete the now-redundant `import open3d as o3d` lines inside `fit_dominant_plane` (line 512)
   and the `from collab_splats.geometry.transforms import rotation_align_vectors` inside it
   (line 514) — both are top-level now.

5. Replace the whole `clean_pointcloud` function with:

```python
def clean_pointcloud(
    points: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Statistical-outlier keep-mask over a (P, 3) world-point array.

    - Returns a (P,) bool array: True for the points open3d keeps.
    - All-True when there are not enough points to form the neighbourhood statistic.
    """
    pts = np.asarray(points, dtype=np.float64)

    # remove_statistical_outlier needs more points than neighbours or it throws; a cloud that
    # small has no outlier structure to find anyway.
    if len(pts) <= nb_neighbors:
        return np.ones(len(pts), dtype=bool)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    keep = np.zeros(len(pts), dtype=bool)
    keep[np.asarray(keep_idx, dtype=int)] = True
    return keep
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py -x -q
```

Expected: PASS.

- [ ] **Step 5: Drop the deleted names from the package `__init__`**

In `collab_splats/pointcloud/__init__.py`, delete line 23
(`from .utils import compute_obb_from_points, get_points_in_mask`) and the two `__all__` entries
`"compute_obb_from_points",` and `"get_points_in_mask",` (lines 85-86).

- [ ] **Step 6: Delete the stale import in `tests/pointcloud/test_base.py`**

Delete line 11:

```python
from collab_splats.pointcloud.utils import clean_pcd, remove_far_points, density_filter
```

and delete the `test_utils_clean_pcd_returns_tuple` test that uses it.

- [ ] **Step 7: Inline the cleaning step in the wrapper**

In `collab_splats/wrapper/reconstructor.py`, delete the whole `_clean_pointcloud` method
(lines 1052-1086) and replace the call block at lines 1005-1007:

```python
        # Apply cleaning step if enabled
        clean_cfg = pc_cfg["clean"]
        if clean_cfg["enabled"]:
            result = self._clean_pointcloud(result, clean_cfg)
```

with:

```python
        # Statistical outlier removal on the final sparse set. Rejected point3D IDs are deleted
        # from the reconstruction in place, so the PLY and the zarr agree with the model.
        if pc_cfg["clean"]["enabled"] and result.reconstruction.points3D:
            point3d_ids = list(result.reconstruction.points3D.keys())
            keep = clean_pointcloud(result.points)
            for pid, keep_this in zip(point3d_ids, keep):
                if not keep_this:
                    result.reconstruction.delete_point3D(pid)
            logger.info("Pointcloud after cleaning: %d points", result.reconstruction.num_points3D())
```

Add the module-top import next to the other `collab_splats.pointcloud` imports:

```python
from collab_splats.pointcloud.utils import clean_pointcloud
```

- [ ] **Step 8: Reduce the `clean` config block**

In `configs/base.yaml`, replace lines 103-107:

```yaml
  clean:
    enabled: true
    outlier_removal: true
    voxel_size: null
    confidence_threshold: null
```

with:

```yaml
  clean:
    enabled: true              # statistical outlier removal on the final sparse point set
```

Then fix the `clean` cases in `tests/wrapper/test_reconstructor.py` — search for
`"clean"` in that file and reduce every literal clean dict to `{"enabled": True}` or
`{"enabled": False}`:

```bash
rtk proxy grep -n "outlier_removal\|voxel_size\|confidence_threshold" tests/wrapper/test_reconstructor.py tests/wrapper/test_sfm_config.py
```

- [ ] **Step 9: Fix the two notebooks**

`docs/source/tutorials/02_pointcloud/feedforward_methods.ipynb` — cell source line 97, change the
import to:

```python
from collab_splats.pointcloud.utils import clean_pointcloud
```

and line 184, change:

```python
cleaned, _ = clean_pointcloud(pcd)
```

to:

```python
keep = clean_pointcloud(np.asarray(pcd.points))
cleaned = pcd.select_by_index(np.flatnonzero(keep))
```

`docs/source/tutorials/03_loop_closure/slam_loop_closure.ipynb` — line 47, change the import of
`voxel_downsample_point_cloud` to `subsample_points`, and at lines 1172 and 1175 replace each

```python
pts, cols = voxel_downsample_point_cloud(pts, cols, voxel_size=...)
```

with

```python
pts, cols = subsample_points(pts, cols, max_points=200_000)
```

- [ ] **Step 10: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper -x -q
```

Expected: PASS.

- [ ] **Step 11: Confirm the deleted helpers are gone**

```bash
rtk proxy grep -rn "clean_pcd\|remove_far_points\|density_filter\|voxel_downsample\|filter_distance\|compute_obb_from_points\|get_points_in_mask\|_UNSET" collab_splats tests evals docs/source
```

Expected: no output (the `test_pipeline_cu121.py:171` hit is a prose comment — reword it to name
`clean_pointcloud` instead).

- [ ] **Step 12: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/utils.py collab_splats/pointcloud/__init__.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_pointcloud_utils.py tests/pointcloud/test_base.py tests/wrapper/test_reconstructor.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/utils.py collab_splats/pointcloud/__init__.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_pointcloud_utils.py tests/pointcloud/test_base.py tests/wrapper/test_reconstructor.py
git add -A collab_splats tests configs/base.yaml docs/source/tutorials
git commit -m "refactor(pointcloud): prune utils to the shipping set; clean_pointcloud returns a keep-mask"
```

---

## Task 6: Create `pointcloud/vda.py`

`generate_vda_depth` loses `fps`, `encoder`, `input_size` and `keep_rows`, returns the resized
depth stack instead of a directory, and gets its checkpoint from the Hugging Face hub instead of a
`setup.sh` `wget`. The public `vda_depth_complete` and the one-entry `_VDA_MODEL_CONFIGS` table
both go; the idempotent skip survives as a private check inside `generate_vda_depth` that loads
the existing maps and returns them.

**Files:**
- Create: `collab_splats/pointcloud/vda.py`
- Create: `tests/pointcloud/test_vda.py`
- Modify: `collab_splats/pointcloud/sfm.py` (delete lines 218-385, the whole VDA block)
- Modify: `evals/scripts/eval.py:70-72,358-360`
- Modify: `setup.sh` (delete the `VDA_CKPT` + `wget` block, lines ~88-95; fix the comment at ~74)
- Modify: `tests/pointcloud/test_instantsfm.py` (delete the six VDA tests, lines 41-94)

- [ ] **Step 1: Write the failing tests**

Create `tests/pointcloud/test_vda.py`:

```python
"""Video-Depth-Anything metric depth generation."""

import numpy as np
import pytest

from collab_splats.pointcloud import vda

_NAMES = ["frame_000000.jpg", "frame_000001.jpg"]


def test_missing_clone_raises_actionable_import_error(tmp_path, monkeypatch):
    """
    No third_party clone -> ImportError naming setup.sh, before any weight download.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(ImportError, match="setup.sh"):
        vda.generate_vda_depth(frames, tmp_path, _NAMES)


def test_existing_depth_set_is_loaded_not_recomputed(tmp_path, monkeypatch):
    """
    An exact per-stem npy set short-circuits inference and comes back as the stacked array.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")  # inference would raise
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for i, name in enumerate(_NAMES):
        np.save(npy_dir / f"{name[:-4]}.npy", np.full((4, 6), float(i + 1), dtype=np.float32))

    depths = vda.generate_vda_depth(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)

    assert depths.shape == (2, 4, 6)
    assert depths.dtype == np.float32
    np.testing.assert_allclose(depths[0], 1.0)
    np.testing.assert_allclose(depths[1], 2.0)


def test_partial_depth_set_does_not_short_circuit(tmp_path, monkeypatch):
    """
    One map for two names is treated as missing -> the inference path -> missing clone raises.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ImportError, match="setup.sh"):
        vda.generate_vda_depth(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)


def test_wrong_named_depth_set_does_not_short_circuit(tmp_path, monkeypatch):
    """
    Right count, wrong stems (a stale selection): the gate compares stem sets, not counts.
    """
    monkeypatch.setattr(vda, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for stem in ("frame_000000", "frame_000007"):
        np.save(npy_dir / f"{stem}.npy", np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ImportError, match="setup.sh"):
        vda.generate_vda_depth(np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)


def test_names_and_frames_must_align(tmp_path):
    """
    Names are consumed positionally against frames — a length mismatch is a hard error.
    """
    with pytest.raises(ValueError, match="one-to-one"):
        vda.generate_vda_depth(np.zeros((3, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES)


def test_inference_result_is_resized_and_written(tmp_path, monkeypatch):
    """
    A stubbed model's (N, H, W) output is nearest-resized to depth_width and written per stem.
    """
    class _StubModel:
        def infer_video_depth(self, frames, fps, input_size, device, fp32):
            n, h, w = frames.shape[:3]
            return np.tile(np.arange(n, dtype=np.float32)[:, None, None], (1, h, w)), fps

    monkeypatch.setattr(vda, "_load_vda_model", lambda device: _StubModel())
    frames = np.zeros((2, 40, 80, 3), dtype=np.uint8)

    depths = vda.generate_vda_depth(frames, tmp_path, _NAMES, depth_width=20)

    assert depths.shape == (2, 10, 20)  # 80x40 -> 20x10 keeps the aspect ratio
    np.testing.assert_allclose(depths[1], 1.0)
    written = sorted(p.name for p in (tmp_path / "depth_vda" / "images" / "npy").glob("*.npy"))
    assert written == ["frame_000000.npy", "frame_000001.npy"]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_vda.py -x -q
```

Expected: FAIL — `ImportError: cannot import name 'vda' from 'collab_splats.pointcloud'`.

- [ ] **Step 3: Create `collab_splats/pointcloud/vda.py`**

```python
# collab_splats/pointcloud/vda.py
"""
Video-Depth-Anything metric depth for the SfM path.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Repo root -> third_party clone (setup.sh owns creation); module-level so tests can monkeypatch
VDA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Video-Depth-Anything"

def _load_vda_model(device: str):
    """
    Construct the VDA metric vitl model on `device` from the hub checkpoint.

    - Split out from generate_vda_depth so the write path is testable without a GPU or the clone.
    """
    # Lazy heavy import — VDA lives in a third_party clone (its root on sys.path), not
    # site-packages. Upstream HEAD (4f5ae23) has no metric_depth/ subdir: `video_depth_anything/`
    # sits at the clone root and `video_depth.py:27` imports a TOP-LEVEL `utils` namespace package
    # (`utils/util.py`) from the same root. Probed 2026-08-23: no foreign top-level `utils` in the
    # venv — a regular `utils` package anywhere on sys.path would shadow VDA's namespace one
    # regardless of insert order, so re-probe if a dependency ever ships one.
    if not (VDA_ROOT / "video_depth_anything").is_dir():
        raise ImportError(
            f"Video-Depth-Anything clone not found at {VDA_ROOT} — run setup.sh "
            "(clones the repo at 4f5ae23)"
        )
    if str(VDA_ROOT) not in sys.path:
        sys.path.insert(0, str(VDA_ROOT))
    from video_depth_anything.video_depth import VideoDepthAnything

    # Upstream weights live on the hub, not in the clone — DepthAnything/Video-Depth-Anything @ 4f5ae23
    ckpt = hf_hub_download(
        repo_id="depth-anything/Metric-Video-Depth-Anything-Large",
        filename="metric_video_depth_anything_vitl.pth",
    )

    # metric=True loads the metric head AND disables infer_video_depth's cross-window
    # scale-and-shift chaining (video_depth.py:135), so consecutive windows are stitched on the
    # head's own absolute output rather than fitted to each other. Measured 2026-08-26: this is
    # why a full-video pass does not improve metric contiguity.
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()


def generate_vda_depth(
    frames: np.ndarray,
    out_dir: Path,
    names: list[str],
    *,
    depth_width: int = 518,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run VDA metric depth over the keyframes; write InstantSfM's depth layout and return the stack.

    - frames: (N, H, W, 3) uint8 RGB in frames.zarr order; names: one staged filename per frame.
    - Writes out_dir/depth_vda/images/npy/<stem>.npy — the layout instantsfm's
      ReadDepthsIntoFeatures single-camera branch consumes (data_reader.py:404-407).
    - depth_width: VDA returns depth at input resolution (300 x 1080p = 2.5 GB), too heavy for
      pointcloud.zarr; each map is nearest-resized to this width (no blending across depth
      discontinuities). Any depth resolution is valid for SfM — instantsfm's
      sample_depth_at_pixel normalises keypoints by camera w/h.
    - Returns (N, h, depth_width) float32, the same maps that were written.
    - Idempotent: an exact per-stem npy set is loaded and returned without running inference.

    Attribution: inference pattern follows
    https://github.com/DepthAnything/Video-Depth-Anything @ 4f5ae23 run.py:45-57.
    """
    if len(names) != len(frames):
        raise ValueError(f"names ({len(names)}) and frames ({len(frames)}) must align one-to-one")

    npy_dir = Path(out_dir) / "depth_vda" / "images" / "npy"
    stems = [Path(n).stem for n in names]

    # Idempotent skip: the exact per-frame stem set is authoritative (a leftover extra map means
    # a stale selection, so the set must match exactly, not merely cover `names`)
    if npy_dir.is_dir() and {p.stem for p in npy_dir.glob("*.npy")} == set(stems):
        logger.info("VDA depth exists at %s (%d maps) — loading", npy_dir, len(stems))
        return np.stack([np.load(npy_dir / f"{s}.npy") for s in stems]).astype(np.float32)

    model = _load_vda_model(device)

    # Metric inference over the whole sequence, at input resolution. `fps` reaches nothing:
    # infer_video_depth takes it as target_fps and never reads it (4f5ae23 video_depth.py:70
    # signature, :162 return), so it resamples nothing.
    logger.info("VDA metric inference: %d frames (writing %d maps)", len(frames), len(names))
    depths, _fps = model.infer_video_depth(frames, 1.0, input_size=518, device=device, fp32=False)
    depths = np.asarray(depths, dtype=np.float32)

    # Free the GPU before the caller's InstantSfM CUDA step — resize/write below is CPU-only
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Nearest-resize to depth_width and write one map per frame, keyed by image stem
    h, w = depths.shape[1:3]
    depth_hw = (int(round(depth_width * h / w)), depth_width)
    npy_dir.mkdir(parents=True, exist_ok=True)
    out = np.empty((len(names), depth_hw[0], depth_hw[1]), dtype=np.float32)
    for i, (stem, depth) in enumerate(zip(stems, depths, strict=True)):
        small = cv2.resize(depth, (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        np.save(npy_dir / f"{stem}.npy", small)
        out[i] = small
    logger.info("VDA depths written: %s (%d maps @ %dx%d)", npy_dir, len(names), depth_hw[1], depth_hw[0])
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_vda.py -x -q
```

Expected: PASS (6 tests).

- [ ] **Step 5: Delete the old VDA block from `sfm.py`**

Delete lines 218-385 of `collab_splats/pointcloud/sfm.py` — the whole
`# Video Depth Anything` section: `VDA_ROOT`, `VDA_CHECKPOINT`, `_VDA_MODEL_CONFIGS`,
`vda_depth_complete`, `_load_vda_model`, `generate_vda_depth`. Then delete the six VDA tests in
`tests/pointcloud/test_instantsfm.py` (lines 41-94: `test_vda_missing_clone_raises_actionable_import_error`,
`test_vda_skips_when_depths_exist`, `test_vda_incomplete_npy_set_does_not_skip`,
`test_vda_depth_complete_is_an_exact_stem_set_check`, `test_vda_wrong_named_npy_set_does_not_skip`).

`reconstructor.py` still imports `generate_vda_depth` and `vda_depth_complete` from
`collab_splats.pointcloud.sfm` — repoint that import now so the module still loads:

```python
from collab_splats.pointcloud.vda import generate_vda_depth
```

and delete `vda_depth_complete` from the import list. Its only use is inside `_ensure_vda_depth`,
which Task 7 deletes; until then, replace the guard body at `reconstructor.py` in
`_ensure_vda_depth` with a direct call (the new function is itself idempotent):

```python
        generate_vda_depth(frames, self.backend_dir, names)
```

- [ ] **Step 6: Update `evals/scripts/eval.py`**

Replace the import at lines 70-72 with:

```python
from collab_splats.pointcloud.sfm import InstantSfMCreator
from collab_splats.pointcloud.vda import generate_vda_depth
```

and replace lines 358-360:

```python
    if use_depths and not vda_depth_complete(output_dir, names):
        generate_vda_depth(frames, fps=1.0, out_dir=output_dir, names=names)
```

with:

```python
    if use_depths:
        generate_vda_depth(frames, output_dir, names)
```

- [ ] **Step 7: Update `setup.sh`**

Delete the checkpoint download block (the `VDA_CKPT=` variable and the
`wget ... .part && mv ...` lines, roughly lines 88-95) — `huggingface_hub` fetches the weight on
first use now. Keep the clone + `git checkout 4f5ae23` pin. Update the comment near line 74 to
name the new module:

```bash
# Video-Depth-Anything: metric depth for the SfM path (collab_splats/pointcloud/vda.py::generate_vda_depth).
# Weights are pulled from the HF hub on first use; this only needs the source clone.
```

- [ ] **Step 8: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/evals -x -q
```

Expected: PASS.

- [ ] **Step 9: Confirm the deleted names are gone**

```bash
rtk proxy grep -rn "vda_depth_complete\|VDA_CHECKPOINT\|_VDA_MODEL_CONFIGS" collab_splats tests evals setup.sh
```

Expected: no output.

- [ ] **Step 10: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/vda.py collab_splats/pointcloud/sfm.py collab_splats/wrapper/reconstructor.py evals/scripts/eval.py tests/pointcloud/test_vda.py tests/pointcloud/test_instantsfm.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/vda.py collab_splats/pointcloud/sfm.py collab_splats/wrapper/reconstructor.py evals/scripts/eval.py tests/pointcloud/test_vda.py tests/pointcloud/test_instantsfm.py
git add -A collab_splats tests evals setup.sh
git commit -m "refactor(pointcloud): extract vda.py; VDA weights from the HF hub"
```

---

## Task 7: Delete the VDA context stream

The context stream ran VDA over a contiguous constant-rate grid and kept only the keyframe rows.
Measured 2026-08-26 it does not improve metric contiguity (`metric=True` disables
`compute_scale_and_shift` at `video_depth.py:135`), so the whole apparatus goes: the config knob,
the grid plumbing through `extract_frames`, the `depth_vda/inputs.json` sidecar, and
`decode_context`.

`context_indices` **stays** — `preproc/sampling.py:416` uses it for `sample_fps`'s own targets,
independent of this feature. The `candidates=` parameter on `_sample_by_quality`/`sample_fps`/
`sample_uniform` also stays (unused after this task); removing it belongs to the in-flight preproc
work, not here.

**Files:**
- Delete: `tests/wrapper/test_vda_context.py`
- Modify: `collab_splats/wrapper/reconstructor.py:49,133-182,184-193,220-227,243-249,270-278,285-303,318-325,938,1170-1171,1214-1324`
- Modify: `collab_splats/preproc/video.py:261-345` (delete `decode_context`)
- Modify: `configs/base.yaml:47` (+ its comment block)
- Modify: `configs/README.md:353`
- Test: `tests/preproc/test_video.py`, `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Delete the context test file and the `decode_context` tests**

```bash
git rm tests/wrapper/test_vda_context.py
rtk proxy grep -n "decode_context\|vda_context_fps" tests/preproc/test_video.py tests/wrapper/test_reconstructor.py tests/preproc/test_sampling.py
```

Delete every test function that the grep names, and delete `decode_context` from the import block
at the top of `tests/preproc/test_video.py`.

- [ ] **Step 2: Run the suite to confirm the baseline**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc tests/wrapper -x -q
```

Expected: PASS — the production code is untouched so far.

- [ ] **Step 3: Strip `vda_context_fps` from `extract_frames`**

In `collab_splats/wrapper/reconstructor.py`:

1. Delete the two module-level helpers `_context_keep_rows` (lines 133-161) and `_video_unchanged`
   (lines 164-182) in full.
2. Delete the `vda_context_fps: float | None = None,` parameter from the `extract_frames` signature.
3. Delete this docstring bullet:

```
    - vda_context_fps restricts every selected frame (target and blur substitute) to the
      constant-rate grid the sfm stage runs VDA over, so the keyframes are grid members by
      construction and their depth rows map back by position. Video input only.
```

4. In the image-directory branch, delete the `"vda_context_fps": None,` provenance entry and the
   whole `if vda_context_fps:` warning block above `if undistort:`.
5. Delete the `optical_flow` guard:

```python
    # Config error, not a path error: check the combination before the probe so a bad
    # frame_selection does not surface as an ffprobe failure
    if vda_context_fps and frame_selection == "optical_flow":
        raise ValueError(
            "preproc.vda_context_fps requires frame_selection 'fps' or 'uniform' — "
            "optical_flow picks frames by motion and cannot be restricted to a grid."
        )
```

6. Delete the candidate-grid block:

```python
    # Context grid: when the VDA context stream is enabled the keyframes must be grid
    # members, so the depth rows map back to them by position (see _context_keep_rows).
    # context_indices is range(0, total, step), so the grid always spans the whole video.
    candidates = None
    if vda_context_fps:
        candidates = context_indices(str(input_path), target_fps=vda_context_fps)
        logger.info(
            "VDA context grid: %d frames at %.2f fps; keyframes will be drawn from it",
            len(candidates), vda_context_fps,
        )
```

and delete the `candidates=candidates,` kwarg from both the `sample_fps(...)` and
`sample_uniform(...)` calls.

7. Delete the `"vda_context_fps": vda_context_fps,` entry from the video-branch `prov` dict.
8. Change line 49 from

```python
from collab_splats.preproc.video import context_indices, decode_context
```

to whatever remains of it — after this task neither name is used in `reconstructor.py`, so delete
the import line entirely. Verify with:

```bash
rtk proxy grep -n "context_indices\|decode_context\|DistortionProfile" collab_splats/wrapper/reconstructor.py
```

(`DistortionProfile` must still appear — `_apply_undistortion` uses it.)

9. In `preprocess()` (line ~938), delete the `vda_context_fps=pre_cfg["vda_context_fps"],` argument
   from the `extract_frames(...)` call.

- [ ] **Step 4: Delete `_ensure_vda_depth`**

Delete the whole `_ensure_vda_depth` method from `reconstructor.py` (from `def _ensure_vda_depth`
through the closing `depth_sidecar.write_text(...)` line), and replace its call site in `_run_sfm`:

```python
        # VDA metric depth for every keyframe — cached across runs, stamped with what made it
        self._ensure_vda_depth(backend_dir, store, names)
```

with:

```python
        # VDA metric depth for every keyframe; re-uses a complete map set, re-runs otherwise
        generate_vda_depth(np.ascontiguousarray(store.images()), backend_dir, names)
```

- [ ] **Step 5: Delete `decode_context`**

Delete the whole `decode_context` function from `collab_splats/preproc/video.py` (lines 261-345).
Keep `context_indices` (line 189).

- [ ] **Step 6: Delete the config key**

In `configs/base.yaml`, delete `vda_context_fps: null` at line 47 together with the ~8 comment
lines directly above it that describe the context grid. In `configs/README.md`, delete the
`preproc.vda_context_fps` table row at line 353.

- [ ] **Step 7: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc tests/wrapper -x -q
```

Expected: PASS.

- [ ] **Step 8: Confirm the feature is gone**

```bash
rtk proxy grep -rn "vda_context_fps\|decode_context\|_context_keep_rows\|_video_unchanged\|_ensure_vda_depth\|inputs.json" collab_splats tests configs evals
```

Expected: no output.

- [ ] **Step 9: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/wrapper/reconstructor.py collab_splats/preproc/video.py tests/preproc/test_video.py tests/wrapper/test_reconstructor.py
/opt/venv/reconstruction/bin/isort collab_splats/wrapper/reconstructor.py collab_splats/preproc/video.py tests/preproc/test_video.py tests/wrapper/test_reconstructor.py
git add -A collab_splats tests configs
git commit -m "refactor(preproc): delete the VDA context stream"
```

---

## Task 8: Create `pointcloud/depth_align.py`

`Reconstructor._sfm_result_from_reconstruction` (140 lines in the wrapper) and
`apply_depth_alignment` (in `sfm.py`) become one module-level function,
`result_from_reconstruction`, that builds the `FeedforwardResult` **already at COLMAP scale**.
The affine model goes entirely — it never shipped, and `pointcloud.instantsfm.depth_align` had
exactly one valid production value.

New public surface: `result_from_reconstruction(reconstruction, depths, images, names, *, min_obs=20)
-> tuple[FeedforwardResult, dict]`. Privates: `_tracked_point3d_ids`,
`_pixel_indices_from_reconstruction`, `_depth_correspondences`, `_fit_depth_scales`.

`_fit_depth_scales` returns `(scales, stats)` — the caller needs `global_scale`, the before/after
ratio percentiles and `fallback_frames` for the log line and the zarr attrs.

**Deleted:** `align_depth_affine`, `_solve_disparity`, `_fit_affine_disparity`,
`_apply_affine_depth`, `MIN_AFFINE_OBS`, `AFFINE_REJECT_ROUNDS`, `AFFINE_MIN_FAR_DISPARITY_FRAC`,
`DepthAlignModel`, `DEPTH_ALIGN_MODELS`, `apply_depth_alignment`,
`align_depth_to_reconstruction`, `Reconstructor._sfm_result_from_reconstruction`.

**Zarr attrs kept:** `depth_scale: "colmap"`, `depth_scales`, `depth_scale_fallback_frames`.
**Dropped:** `depth_align_model`, and every affine attr.

**Files:**
- Create: `collab_splats/pointcloud/depth_align.py`
- Modify: `collab_splats/pointcloud/sfm.py` (delete lines 218-861's remaining depth-alignment block)
- Modify: `collab_splats/wrapper/reconstructor.py` (imports, `validate_config`, `_run_sfm`, delete `_sfm_result_from_reconstruction`, splats log line)
- Modify: `configs/base.yaml:116` (`depth_align`)
- Test: `tests/pointcloud/test_depth_align.py` (rewritten), `tests/wrapper/test_sfm_result.py` (cases moved in), `tests/wrapper/test_sfm_config.py`

- [ ] **Step 1: Rewrite the depth-alignment tests**

In `tests/pointcloud/test_depth_align.py`:

1. Change the import at line 11 to:

```python
from collab_splats.pointcloud import depth_align
```

and replace every `sfm.` prefix in the kept tests with `depth_align.`, with these renames:
`sfm._depth_correspondences` → `depth_align._depth_correspondences`,
`sfm.align_depth_to_reconstruction` → `depth_align._fit_depth_scales`.
`_fit_depth_scales` takes `min_obs` as a keyword-only argument instead of reading
`MIN_ALIGN_OBS`, so any test that referenced `sfm.MIN_ALIGN_OBS` passes `min_obs=20` explicitly.

2. Keep exactly these tests: `_fake_reconstruction` (line 17),
`test_correspondences_pair_track_depth_with_sampled_vda_depth` (76),
`test_correspondences_drop_zero_out_of_bounds_and_behind_camera_samples` (89),
`test_correspondences_raise_on_unregistered_name` (109),
`test_scale_alignment_recovers_a_constant_ratio` (116),
`test_alignment_rejects_a_names_to_depth_row_mismatch` (227).

3. Delete every other test and helper in the file — the whole affine block
(`_affine_observations`, `_affine_scene`, `_two_depth_scene`, lines 131-408) and every
`test_apply_depth_alignment_*` (410-500). Also delete `import json` if nothing else uses it.

4. Move the pixel-provenance tests out of `tests/pointcloud/test_instantsfm.py` and into this
file — the helper `_recon_with_keypoint` (line 7), `_NAMES` (38),
`test_tracked_point3d_ids_drops_observationless_points` (95),
`test_pixel_indices_from_reconstruction_scales_to_depth_res` (109) and
`test_pixel_indices_clamped_to_grid` (124) — `git rm`-ing nothing, just cut/paste, with every
`sfm.` prefix in them changed to `depth_align.`. Delete `_NAMES` if the surviving
`test_instantsfm.py` no longer uses it; delete `from collab_splats.pointcloud import sfm` there
if nothing else in that file uses it.

5. Append the new builder tests, ported from `tests/wrapper/test_sfm_result.py`:

```python
########################################################
########## result_from_reconstruction ##################
########################################################

ORIG_W, ORIG_H = 64, 48  # frames.zarr resolution
DEPTH_W, DEPTH_H = 16, 12  # VDA depth grid (4x downscale)
K_PARAMS = [50.0, 50.0, 32.0, 24.0]  # fx, fy, cx, cy at ORIG res


def _pycolmap_scene(names, cam_w=ORIG_W, cam_h=ORIG_H):
    """
    One PINHOLE camera at cam_w x cam_h, one image per name, one point3D seen in every image.
    """
    import pycolmap

    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=cam_w, height=cam_h, params=K_PARAMS, camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


def _scene_inputs(n=2):
    """
    (recon, depths, images, names) for an n-frame scene at constant VDA depth 2.0.
    """
    names = [f"frame_{i:06d}.jpg" for i in range(n)]
    recon = _pycolmap_scene([Path(x).stem for x in names])
    depths = np.full((n, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.stack([np.full((ORIG_H, ORIG_W, 3), 40 * (i + 1), dtype=np.uint8) for i in range(n)])
    return recon, depths, images, names


def test_result_from_reconstruction_shapes_and_k_rescaling():
    """
    Rows follow `names`; K is rescaled from camera res to the depth grid; images land at depth res.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    out, attrs = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    assert out.depth.shape == (2, DEPTH_H, DEPTH_W)
    assert out.images.shape == (2, 3, DEPTH_H, DEPTH_W)
    assert out.world_points.shape == (2, DEPTH_H, DEPTH_W, 3)
    assert out.model_width == DEPTH_W and out.model_height == DEPTH_H
    assert out.image_paths == [Path("frame_000000"), Path("frame_000001")]
    assert out.confidence is None
    # fx was 50 at width 64; the depth grid is 16 wide -> 50 * 16/64
    np.testing.assert_allclose(out.intrinsics[0][0, 0], 50.0 * DEPTH_W / ORIG_W, rtol=1e-5)
    np.testing.assert_allclose(out.intrinsics[0][1, 1], 50.0 * DEPTH_H / ORIG_H, rtol=1e-5)
    np.testing.assert_allclose(out.original_coords[0], [0, 0, ORIG_W, ORIG_H, ORIG_W, ORIG_H])


def test_result_from_reconstruction_rescales_depth_to_the_colmap_world():
    """
    VDA depth 2.0 against a COLMAP depth of 5.0 -> scale 2.5, stamped in the attrs.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, attrs = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    np.testing.assert_allclose(out.depth, 5.0, rtol=1e-4)
    assert attrs["depth_scale"] == "colmap"
    np.testing.assert_allclose(attrs["depth_scales"], [2.5], rtol=1e-4)
    assert attrs["depth_scale_fallback_frames"] == []
    assert "depth_align_model" not in attrs


def test_result_from_reconstruction_reunprojects_world_points_from_aligned_depth():
    """
    world_points are re-derived from the ALIGNED depth, not scaled from the VDA-metric ones.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, _ = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    # identity pose for frame 0: the world z of every unprojected pixel is the aligned depth
    np.testing.assert_allclose(out.world_points[0, :, :, 2], 5.0, rtol=1e-4)


def test_result_from_reconstruction_refuses_camera_resolution_mismatch():
    """
    COLMAP cameras at a different resolution than the frames mean a stale staged set / SIFT db.
    """
    names = ["frame_000000.jpg"]
    recon = _pycolmap_scene(["frame_000000"], cam_w=128, cam_h=96)
    depths = np.full((1, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.zeros((1, ORIG_H, ORIG_W, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="camera resolution"):
        depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)


def test_result_from_reconstruction_refuses_partial_registration():
    """
    Fewer registered images than frames leaves rows without poses — refuse rather than pad.
    """
    recon, depths, images, names = _scene_inputs(n=2)
    names.append("frame_000002.jpg")
    depths = np.concatenate([depths, depths[:1]])
    images = np.concatenate([images, images[:1]])

    with pytest.raises(RuntimeError, match="partial"):
        depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)


def test_result_from_reconstruction_refuses_name_mismatch():
    """
    Registered names that are not the requested stems mean two different runs.
    """
    recon, depths, images, _ = _scene_inputs(n=2)

    with pytest.raises(ValueError, match="do not match"):
        depth_align.result_from_reconstruction(
            recon, depths, images, ["frame_000000.jpg", "frame_000007.jpg"], min_obs=1
        )
```

Add `from pathlib import Path` and `import numpy as np` / `import pytest` at the top if the trimmed
file no longer has them.

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py -x -q
```

Expected: FAIL — `ImportError: cannot import name 'depth_align' from 'collab_splats.pointcloud'`.

- [ ] **Step 3: Create `collab_splats/pointcloud/depth_align.py`**

```python
# collab_splats/pointcloud/depth_align.py
"""
Build a FeedforwardResult from an InstantSfM COLMAP model + VDA depth, at the COLMAP world scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from vggt.utils.geometry import unproject_depth_map_to_point_map

from .feedforward.base import FeedforwardResult

logger = logging.getLogger(__name__)


########################################################
########## Track-observation correspondences ###########
########################################################


def _tracked_point3d_ids(recon: pycolmap.Reconstruction) -> list[int]:
    """
    Sorted point3D ids that carry at least one observation.

    - InstantSfM exports sub-min-track-length points with EMPTY tracks (the writer consistency
      patch drops their unverifiable observations); no observation means no pixel provenance,
      so the result tail excludes them.
    """
    return sorted(pid for pid, p in recon.points3D.items() if len(p.track.elements) > 0)


def _pixel_indices_from_reconstruction(
    recon: pycolmap.Reconstruction,
    point3d_ids: list[int],
    name_to_row: dict[str, int],
    scale_x: float,
    scale_y: float,
    depth_hw: tuple[int, int],
) -> np.ndarray:
    """
    Synthesize (P, 3) int32 [frame_row, row, col] pixel indices from COLMAP tracks.

    - First track observation per point3D; keypoint xy is original-res, scaled to the depth
      grid and clamped in-bounds.
    - lift_features requires pixel_indices; SfM results have no dense source pixel, so the
      observing keypoint is the honest substitute.
    """
    h, w = depth_hw
    out = np.zeros((len(point3d_ids), 3), dtype=np.int32)

    # One observation per point: the first track element's keypoint, scaled + clamped
    for i, pid in enumerate(point3d_ids):
        elem = recon.points3D[pid].track.elements[0]
        image = recon.images[elem.image_id]
        xy = image.points2D[elem.point2D_idx].xy
        col = min(max(int(xy[0] * scale_x), 0), w - 1)
        row = min(max(int(xy[1] * scale_y), 0), h - 1)
        out[i] = (name_to_row[image.name], row, col)

    return out


def _depth_correspondences(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Per-frame (d_colmap, d_vda) pairs from track observations, both positive and in bounds.

    - Each points2D carrying a point3D gives an exact pixel plus that point's z in the camera
      frame; the pixel is rescaled from native camera resolution to the depth grid and
      nearest-sampled into VDA depth.
    - Returns one (d_colmap, d_vda) tuple per row of `depth`, in `image_names` order; a frame
      with no usable observation gets a pair of empty arrays.
    """
    # Row order is the caller's: one name per depth row, and every name registered
    if len(image_names) != depth.shape[0]:
        raise ValueError(f"{len(image_names)} image names for {depth.shape[0]} depth maps — rows would misalign")
    name_to_image = {image.name: image for image in reconstruction.images.values()}
    missing = [name for name in image_names if name not in name_to_image]
    if missing:
        raise ValueError(f"{len(missing)} image names not in reconstruction (first: {missing[0]})")

    _n_frames, grid_h, grid_w = depth.shape
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for row, name in enumerate(image_names):
        image = name_to_image[name]
        camera = reconstruction.cameras[image.camera_id]

        # Track observations: exact 2D pixel + the observed point's depth in this view
        observations = [p for p in image.points2D if p.has_point3D()]
        if not observations:
            pairs.append((np.zeros(0), np.zeros(0)))
            continue
        xyz = np.stack([reconstruction.points3D[p.point3D_id].xyz for p in observations])
        cam_from_world = image.cam_from_world().matrix()
        d_colmap = (xyz @ cam_from_world[:3, :3].T + cam_from_world[:3, 3])[:, 2]

        # Rescale native pixels to the depth grid (the localization ref_px bug class —
        # native-res keypoints indexed into a model-res grid), then nearest-sample
        xy = np.stack([p.xy for p in observations])
        u = np.rint(xy[:, 0] * (grid_w / camera.width)).astype(np.int64)
        v = np.rint(xy[:, 1] * (grid_h / camera.height)).astype(np.int64)
        in_bounds = (u >= 0) & (u < grid_w) & (v >= 0) & (v < grid_h)
        d_vda = np.zeros(len(observations))
        d_vda[in_bounds] = depth[row, v[in_bounds], u[in_bounds]]

        # Keep pairs with positive depth on both sides
        valid = in_bounds & (d_vda > 0) & (d_colmap > 0)
        pairs.append((d_colmap[valid], d_vda[valid]))

    return pairs


def _fit_depth_scales(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],
    depth: np.ndarray,
    *,
    min_obs: int,
) -> tuple[np.ndarray, dict]:
    """
    Per-frame scale factors aligning VDA depth to the reconstruction's world scale.

    - s_i = median(d_colmap / d_vda) per frame; frames with fewer than `min_obs` valid pairs
      inherit the global median of the fitted scales; zero fitted frames raises.
    - Returns (scales, stats): (N,) float64 depth multipliers, and a stats dict with the global
      scale, fallback frames, and the pooled ratio spread before/after alignment (the
      after-spread is the unit-level success check).
    """
    n_frames = depth.shape[0]
    scales = np.full(n_frames, np.nan)
    pooled_ratios: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []

    # One robust scale per frame from its track observations; below the obs floor, don't fit
    for row, (d_colmap, d_vda) in enumerate(_depth_correspondences(reconstruction, image_names, depth)):
        if len(d_colmap) == 0:
            continue
        ratios = d_colmap / d_vda
        pooled_ratios.append(ratios)
        pooled_rows.append(np.full(len(ratios), row))
        if len(ratios) >= min_obs:
            scales[row] = np.median(ratios)

    fitted = ~np.isnan(scales)
    if not fitted.any():
        raise ValueError(
            f"depth alignment: no frame has >= {min_obs} valid track observations — "
            "the reconstruction is too sparse to align VDA depth to the COLMAP world."
        )

    # Thin frames inherit the scene answer (below the obs floor: don't fit, inherit)
    global_scale = float(np.median(scales[fitted]))
    fallback_frames = [image_names[i] for i in np.flatnonzero(~fitted)]
    if fallback_frames:
        logger.warning(
            "depth alignment: %d frames under %d obs (first: %s) — using global scale",
            len(fallback_frames), min_obs, fallback_frames[0],
        )
    scales[~fitted] = global_scale

    # Pooled spread: before = one global scale for all frames, after = per-frame scales
    ratios_all = np.concatenate(pooled_ratios)
    rows_all = np.concatenate(pooled_rows).astype(np.int64)
    stats = {
        "global_scale": global_scale,
        "n_fallback": len(fallback_frames),
        "fallback_frames": fallback_frames,
        "ratio_p10_p50_p90_before": [float(x) for x in np.percentile(ratios_all / global_scale, [10, 50, 90])],
        "ratio_p10_p50_p90_after": [float(x) for x in np.percentile(ratios_all / scales[rows_all], [10, 50, 90])],
    }
    return scales, stats


########################################################
########## COLMAP model -> FeedforwardResult ###########
########################################################


def result_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    depths: np.ndarray,
    images: np.ndarray,
    names: list[str],
    *,
    min_obs: int = 20,
) -> tuple[FeedforwardResult, dict]:
    """
    Build a COLMAP-scale FeedforwardResult from an InstantSfM model + VDA depth maps.

    - names: staged filenames (frame_NNNNNN.jpg); the model registers their stems, in this order.
    - depths: (N, h, w) VDA metric depth; images: (N, H, W, 3) uint8 RGB at frames.zarr resolution.
    - The depth grid is the result's model resolution: K, images and pixel_indices are scaled to
      it (pairing original-res K with model-res depth is the 2026-08-11 mesh-regression class).
    - Depth is rescaled to the COLMAP world before anything is derived from it, so the zarr and
      the model share one scale (splat depth targets, mesh fusion, localization lookup).
    - confidence / mv_* stay absent — SfM has no learned per-pixel confidence.
    - Returns (result, attrs) where attrs are the alignment provenance for save_zarr.
    """
    stems = [Path(n).stem for n in names]

    # Every requested frame must be registered — a partial model leaves rows without poses
    if len(reconstruction.images) != len(stems):
        raise RuntimeError(
            f"InstantSfM registered {len(reconstruction.images)}/{len(stems)} frames — partial "
            "registration is not supported; re-run with more overlap"
        )
    # Rows of depths/images follow `names`; everything derived from the model follows sorted
    # image name. Requiring the two to be the same order is what keeps them aligned — the
    # pipeline's frame_NNNNNN naming already guarantees it, so a mismatch is a real bug.
    images_sorted = sorted(reconstruction.images.values(), key=lambda im: im.name)
    registered = [im.name for im in images_sorted]
    if registered != stems:
        raise ValueError(
            f"registered image names do not match the requested frames in order (first "
            f"registered: {registered[0]}, first expected: {stems[0]}); the frame store and "
            "the reconstruction describe different runs."
        )
    name_to_row = {name: row for row, name in enumerate(registered)}

    depths = np.asarray(depths, dtype=np.float32)
    n, h, w = depths.shape

    # Poses: cam_from_world (w2c) as homogeneous 4x4
    extrinsics = np.stack(
        [np.vstack([im.cam_from_world().matrix(), [0.0, 0.0, 0.0, 1.0]]) for im in images_sorted]
    ).astype(np.float32)

    # COLMAP K is at staged-jpg (original) resolution; rescale it to the depth grid. The COLMAP
    # cameras must be at the frames' resolution, else the staged set / SIFT DB came from
    # a different store.
    orig_h, orig_w = images.shape[1:3]
    cam_dims = {
        (reconstruction.cameras[im.camera_id].width, reconstruction.cameras[im.camera_id].height)
        for im in images_sorted
    }
    if cam_dims != {(orig_w, orig_h)}:
        raise ValueError(
            f"COLMAP camera resolution {sorted(cam_dims)} does not match the frames "
            f"({orig_w}x{orig_h}); the staged images / SIFT database came from a different store."
        )
    sx, sy = w / orig_w, h / orig_h
    intrinsics = np.stack([reconstruction.cameras[im.camera_id].calibration_matrix() for im in images_sorted])
    intrinsics = intrinsics.astype(np.float32)
    intrinsics[:, 0, :] *= sx
    intrinsics[:, 1, :] *= sy

    # Align VDA depth to the COLMAP world FIRST — world_points below must come from the aligned
    # depth (t is not scale-invariant, so scaled world points would be wrong).
    scales, stats = _fit_depth_scales(reconstruction, registered, depths, min_obs=min_obs)
    logger.info(
        "depth alignment: global scale %.4f, ratio p10/p50/p90 %s -> %s, %d fallback frames",
        stats["global_scale"],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_before"]],
        [round(x, 4) for x in stats["ratio_p10_p50_p90_after"]],
        stats["n_fallback"],
    )
    depths = (depths * scales[:, None, None]).astype(np.float32)

    # Sparse points in point3D-id order; pixel_indices from each point's first observation.
    # Observation-less points (InstantSfM's sub-min-track-length exports) are dropped.
    point3d_ids = _tracked_point3d_ids(reconstruction)
    points = np.array([reconstruction.points3D[pid].xyz for pid in point3d_ids], dtype=np.float32).reshape(-1, 3)
    colors = np.array([reconstruction.points3D[pid].color for pid in point3d_ids], dtype=np.uint8).reshape(-1, 3)
    pixel_indices = _pixel_indices_from_reconstruction(
        reconstruction, point3d_ids, name_to_row, scale_x=sx, scale_y=sy, depth_hw=(h, w)
    )

    # RGB at depth res as (N, 3, H, W) float32 in [0, 1] — the feedforward images convention
    images_arr = np.stack([cv2.resize(images[i], (w, h), interpolation=cv2.INTER_AREA) for i in range(n)])
    images_arr = images_arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    # Dense world points by unprojecting the ALIGNED depth through the rescaled K and w2c poses
    world_points = unproject_depth_map_to_point_map(
        depths[..., None], extrinsics[:, :3, :], intrinsics
    ).astype(np.float32)

    # No crop: the depth grid is a full-frame resize, so the crop box is the whole original frame
    # in ORIGINAL pixels — [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], the loger convention.
    original_coords = np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]] * n, dtype=np.float32)

    result = FeedforwardResult(
        points=points,
        colors=colors,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[Path(im.name) for im in images_sorted],
        original_coords=original_coords,
        model_width=w,
        model_height=h,
        images=images_arr,  # numpy float32 on purpose — save_zarr accepts it; no torch tensor needed
        world_points=world_points,
        depth=depths,
        pixel_indices=pixel_indices,
    )
    attrs = {
        "depth_scale": "colmap",
        "depth_scales": [float(s) for s in scales],
        "depth_scale_fallback_frames": stats["fallback_frames"],
    }
    return result, attrs
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_depth_align.py -x -q
```

Expected: PASS.

- [ ] **Step 5: Delete the old alignment code from `sfm.py`**

Delete from `collab_splats/pointcloud/sfm.py` (what is left of the depth-alignment section after
Task 6 removed the VDA block): `DepthAlignModel`, `DEPTH_ALIGN_MODELS`, `MIN_ALIGN_OBS`,
`MIN_AFFINE_OBS`, `AFFINE_REJECT_ROUNDS`, `AFFINE_MIN_FAR_DISPARITY_FRAC`,
`_depth_correspondences`, `align_depth_to_reconstruction`, `_solve_disparity`,
`_fit_affine_disparity`, `_apply_affine_depth`, `align_depth_affine`, `apply_depth_alignment`,
`_tracked_point3d_ids`, `_pixel_indices_from_reconstruction`, and the two section-divider comment
blocks around them. Then drop the imports that only they used (`Literal`, `get_args`,
`unproject_depth_map_to_point_map`, `FeedforwardResult`) — check with:

```bash
/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/pointcloud/sfm.py
```

- [ ] **Step 6: Wire `_run_sfm` to the new builder**

In `collab_splats/wrapper/reconstructor.py`:

1. Replace the pointcloud imports (lines 25-34) with:

```python
from collab_splats.pointcloud.depth_align import result_from_reconstruction
from collab_splats.pointcloud.sfm import InstantSfMCreator
from collab_splats.pointcloud.vda import generate_vda_depth
```

2. Delete the whole `_sfm_result_from_reconstruction` method.
3. In `_run_sfm`, replace the depth/build/align block:

```python
        # VDA metric depth for every keyframe; re-uses a complete map set, re-runs otherwise
        generate_vda_depth(np.ascontiguousarray(store.images()), backend_dir, names)
```
```python
        # Unified pointcloud.zarr at VDA depth res, with provenance from the installed package
        outputs = self._sfm_result_from_reconstruction(recon, backend_dir, store)

        # Align VDA depth to the COLMAP world before anything persists — the zarr and the
        # model must share one scale (splat depth targets, mesh fusion, localization lookup).
        # Raises rather than writing a VDA-metric zarr; depth_scale attrs mark aligned scenes.
        align_attrs = apply_depth_alignment(outputs, recon, model=pc_cfg["instantsfm"]["depth_align"])
```

with:

```python
        # VDA metric depth for every keyframe; re-uses a complete map set, re-runs otherwise
        frames = np.ascontiguousarray(store.images())
        depths = generate_vda_depth(frames, backend_dir, names)
```
```python
        # Unified pointcloud.zarr at VDA depth res, already rescaled to the COLMAP world —
        # the zarr and the model must share one scale (splat depth targets, mesh fusion,
        # localization lookup). Raises rather than writing a VDA-metric zarr.
        outputs, align_attrs = result_from_reconstruction(recon, depths, frames, names)
```

(keeping the `generate_vda_depth` call where it already sits, before the creator runs).

4. In `validate_config`, delete the `depth_align in DEPTH_ALIGN_MODELS` check from the sfm block
   (~lines 878-900). Keep the `random_seed` range check.
5. In the splats log line (~line 1830-1838), delete the `depth_align=%s` clause and its
   `instantsfm_cfg.get("depth_align") if instantsfm_cfg else None` argument, and the now-unused
   `instantsfm_cfg = ...` lookup above it:

```python
                zero_fraction = 100.0 * float((depth_targets <= 0).mean())
                logger.info(
                    "splats depth targets: mesh.conf_percentile=%s not applied (no confidence "
                    "channel); %.2f%% of target pixels are zero",
                    conf_percentile,
                    zero_fraction,
                )
```

- [ ] **Step 7: Drop the `depth_align` config key**

In `configs/base.yaml`, delete `depth_align: scale` (line 116) and its comment lines. In
`configs/README.md`, delete the `instantsfm.depth_align` row (line 358). In
`tests/wrapper/test_sfm_config.py`, delete the `depth_align` validation test.

- [ ] **Step 8: Move the wrapper's builder tests out**

```bash
git rm tests/wrapper/test_sfm_result.py
```

(its scaling / resolution / partial-registration / name-mismatch cases were ported into
`tests/pointcloud/test_depth_align.py` in Step 1; its
`test_rename_images_to_stems_round_trips_through_write_binary` case is ported verbatim into
`tests/pointcloud/sfm/test_instantsfm.py` in Task 9 Step 3, where the helper it covers lands.
Nothing this file tested is lost. `_rename_images_to_stems` is uncovered for the span between
this task and Task 9 Step 3 — it is untouched code in that window, and Task 9 re-covers it
before changing it.)

- [ ] **Step 9: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper -x -q
```

Expected: PASS.

- [ ] **Step 10: Confirm the affine path is gone**

```bash
rtk proxy grep -rn "affine\|DEPTH_ALIGN_MODELS\|DepthAlignModel\|apply_depth_alignment\|align_depth_to_reconstruction\|depth_align_model\|_sfm_result_from_reconstruction" collab_splats tests configs evals
```

Expected: no output.

- [ ] **Step 11: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/depth_align.py collab_splats/pointcloud/sfm.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_depth_align.py tests/wrapper/test_sfm_config.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/depth_align.py collab_splats/pointcloud/sfm.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_depth_align.py tests/wrapper/test_sfm_config.py
git add -A collab_splats tests configs
git commit -m "refactor(pointcloud): extract depth_align.py; delete the affine depth model"
```

---

## Task 9: `sfm.py` → the `sfm/` package

`sfm.py` is down to three creators plus the InstantSfM support code after Tasks 6 and 8 emptied
it of VDA and depth alignment. Split it into one file per backend. The move is mechanical — the
only behaviour changes are on `InstantSfMCreator`: `features` and `single_camera` go, the stem
rename moves in from the wrapper, `_sift_database_valid` is rewritten on `pycolmap.Database`,
and `_SIFT_NUM_THREADS` becomes a keyword default.

Tests move with the code, so the tree stays green inside this one task.

**Files:**
- Create: `collab_splats/pointcloud/sfm/__init__.py`, `sfm/colmap.py`, `sfm/hloc.py`, `sfm/instantsfm.py`
- Delete: `collab_splats/pointcloud/sfm.py`
- Modify: `collab_splats/wrapper/reconstructor.py:78,506-517,882-883,1174-1177`
- Test: `tests/pointcloud/sfm/{__init__,test_colmap,test_hloc,test_instantsfm}.py` (created), `tests/pointcloud/test_sfm_creator.py` + `tests/pointcloud/test_instantsfm.py` (deleted), `tests/test_cu121_migration.py:99`

- [ ] **Step 1: Create the package skeleton and move the two classical backends**

```bash
mkdir -p collab_splats/pointcloud/sfm
git mv collab_splats/pointcloud/sfm.py collab_splats/pointcloud/sfm/instantsfm.py
```

Create `collab_splats/pointcloud/sfm/__init__.py`:

```python
# collab_splats/pointcloud/sfm/__init__.py
"""
SfM backends: classical COLMAP, learned-feature hloc, and global-solver InstantSfM.
"""

from .colmap import ColmapCreator
from .hloc import HlocCreator
from .instantsfm import InstantSfMCreator

__all__ = ["ColmapCreator", "HlocCreator", "InstantSfMCreator"]
```

Create `collab_splats/pointcloud/sfm/colmap.py` by cutting `ColmapCreator` out of
`instantsfm.py` (it is the first class in the file) and trimming its docstring to the contract:

```python
# collab_splats/pointcloud/sfm/colmap.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from ..base import BasePointcloudCreator, PointcloudResult

logger = logging.getLogger(__name__)


@dataclass
class ColmapCreator(BasePointcloudCreator):
    """
    Pointcloud via pycolmap: SIFT extraction -> exhaustive matching -> incremental mapping.

    - camera_model: COLMAP camera model string (SIMPLE_PINHOLE, SIMPLE_RADIAL, OPENCV, ...).
    - single_camera: one shared camera for every image (video from one device) vs one per image.
    - Writes the binary model to output_dir/colmap/sparse/0; the SIFT DB to colmap/database.db.
    """

    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = False

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / "colmap" / "database.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # pycolmap >=4.0: camera_model lives in ImageReaderOptions, not as a
        # top-level kwarg of extract_features.
        camera_mode = pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
        reader_opts = pycolmap.ImageReaderOptions(camera_model=self.camera_model)
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=camera_mode,
            reader_options=reader_opts,
        )
        pycolmap.match_exhaustive(str(db_path))
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir.parent),  # colmap/sparse/ -> creates 0/ inside
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")

        # Cluster 0 is the model; image_paths follow filename order, the pipeline's row order
        recon = reconstructions[0]
        recon.write_binary(str(sparse_dir))
        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(reconstruction=recon, image_paths=image_paths)
```

Create `collab_splats/pointcloud/sfm/hloc.py` by cutting `HlocCreator` out of `instantsfm.py`,
replacing its 60-line option catalogue with one line per field, and making the missing-hloc
`ImportError` actionable:

```python
# collab_splats/pointcloud/sfm/hloc.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..base import BasePointcloudCreator, PointcloudResult

logger = logging.getLogger(__name__)


@dataclass
class HlocCreator(BasePointcloudCreator):
    """
    Pointcloud via hloc: retrieval -> learned features -> learned matching -> COLMAP mapper.

    - retrieval_conf: hloc retrieval config key; see `hloc.extract_features.confs`.
    - feature_conf: hloc local-feature config key; see `hloc.extract_features.confs`.
    - matcher_conf: hloc matcher config key; see `hloc.match_features.confs`.
    - Writes the binary model to output_dir/colmap/sparse/0, hloc intermediates to colmap/hloc/.
    """

    retrieval_conf: str = "netvlad"
    feature_conf: str = "superpoint_aachen"
    matcher_conf: str = "superglue"

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        # hloc is a third_party clone, not a locked dependency — import inside the one method
        # that needs it so the module (and the registry) import without it installed
        try:
            from hloc import (
                extract_features,
                match_features,
                pairs_from_retrieval,
                reconstruction,
            )
        except ImportError as err:
            raise ImportError(
                "hloc is not installed — run `bash setup/hloc.sh` to clone and install it"
            ) from err

        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        hloc_dir = output_dir / "colmap" / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        # Retrieval first: O(N) candidate pairs instead of the O(N^2) exhaustive set
        retrieval_path = extract_features.main(extract_features.confs[self.retrieval_conf], image_dir, hloc_dir)
        pairs_path = hloc_dir / "pairs.txt"
        pairs_from_retrieval.main(retrieval_path, pairs_path)

        # Learned features + matcher over those pairs, then the COLMAP incremental mapper
        feature_path = extract_features.main(extract_features.confs[self.feature_conf], image_dir, hloc_dir)
        match_path = match_features.main(
            match_features.confs[self.matcher_conf],
            pairs_path,
            features=feature_path,
            matches=hloc_dir / "matches.h5",
        )
        recon = reconstruction.main(
            sfm_dir=sparse_dir,
            image_dir=image_dir,
            pairs=pairs_path,
            features=feature_path,
            matches=match_path,
        )
        if recon is None:
            raise RuntimeError("reconstruction failed — hloc returned None")

        image_paths = sorted(
            [image_dir / img.name for img in recon.images.values()],
            key=lambda p: p.name,
        )
        return PointcloudResult(reconstruction=recon, image_paths=image_paths)
```

Delete both classes from `sfm/instantsfm.py` and reduce its module header to:

```python
# collab_splats/pointcloud/sfm/instantsfm.py
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import torch

logger = logging.getLogger(__name__)
```

(`cv2`, `sqlite3`, `Sequence`, `Literal`, `get_args`, `unproject_depth_map_to_point_map` and the
`..base` import are all dead here once the two creators leave and Task 8 removed the alignment
code. `sys` stays — the `_patch_*` functions use it.)

- [ ] **Step 2: Confirm the split imports**

```bash
/opt/venv/reconstruction/bin/python -c "from collab_splats.pointcloud.sfm import ColmapCreator, HlocCreator, InstantSfMCreator; print('ok')"
/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/pointcloud/sfm/
```

Expected: `ok`, and no pyflakes output.

- [ ] **Step 3: Write the failing tests for the InstantSfM changes**

Create `tests/pointcloud/sfm/__init__.py` (empty) and
`tests/pointcloud/sfm/test_instantsfm.py`. Move into it, unchanged except for the import line,
every surviving test from `tests/pointcloud/test_instantsfm.py`:
`test_creator_config_copy_prevents_module_dict_leak` (137),
`test_creator_retriangulation_flag_flips_skip_retriangulation` (150),
`test_track_id_patch_renumbers_packed_64bit_ids` (161),
`test_pypose_robustmodel_target_patch_defaults_none` (195),
`test_bae_pcg_patch_keeps_column_shape` (214),
`test_colmap_write_patch_produces_pycolmap_readable_model` (242),
`test_nudge_edge_keypoints_pulls_exact_edge_inward_only` (302).
The import becomes:

```python
from collab_splats.pointcloud.sfm import instantsfm
```

and every `sfm.` prefix in those tests becomes `instantsfm.`.

Then append the new cases:

```python
########################################################
########## SIFT database validity ######################
########################################################


def test_sift_database_valid_is_false_for_a_missing_file(tmp_path):
    """
    No DB at all is the first-run case, not a corruption.
    """
    assert instantsfm._sift_database_valid(tmp_path / "nope.db") is False


def test_sift_database_valid_is_false_for_a_non_database_file(tmp_path):
    """
    A crashed colmap can leave a truncated/garbage file; pycolmap raises rather than returning.
    """
    db = tmp_path / "garbage.db"
    db.write_bytes(b"not a sqlite file")

    assert instantsfm._sift_database_valid(db) is False


def test_sift_database_valid_is_false_for_an_empty_database(tmp_path):
    """
    An OOM-killed extractor leaves a well-formed but empty DB — an existence check would
    cache-hit on it and feed ReadColmapDatabase zero tracks.
    """
    db = tmp_path / "empty.db"
    pycolmap.Database.open(str(db)).close()

    assert instantsfm._sift_database_valid(db) is False


########################################################
########## Stem rename #################################
########################################################


def _recon(names):
    """
    One PINHOLE camera, one image per name, one point3D observed in every image.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


def test_rename_images_to_stems_round_trips_through_write_binary(tmp_path):
    # InstantSfM names (frame_000000.jpg) -> contract stems, persisted in the rewritten model
    recon = _recon(["frame_000000.jpg", "frame_000003.jpg"])
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    instantsfm._rename_images_to_stems(recon, sparse_dir)
    assert sorted(im.name for im in recon.images.values()) == ["frame_000000", "frame_000003"]
    reread = pycolmap.Reconstruction(str(sparse_dir))
    assert sorted(im.name for im in reread.images.values()) == ["frame_000000", "frame_000003"]
    assert reread.num_points3D() == 1
```

This case is `tests/wrapper/test_sfm_result.py:115-125` moved verbatim — only the call is
retargeted (`_rename_images_to_stems` -> `instantsfm._rename_images_to_stems`), because Step 5
moves the helper itself out of the wrapper and into the creator. Its `_recon` helper
(`test_sfm_result.py:17`) comes with it, with `ORIG_W`/`ORIG_H`/`K_PARAMS` inlined since nothing
else in this file uses them.

The file needs `import pycolmap` at the top alongside `import numpy as np` — the surviving
tests only used pycolmap inside `test_colmap_write_patch_produces_pycolmap_readable_model`
(a function-local import), and two users make it a module import.

- [ ] **Step 4: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_instantsfm.py -x -q
```

Expected: one failure — `AttributeError: module 'collab_splats.pointcloud.sfm.instantsfm'
has no attribute '_rename_images_to_stems'`.

The three `_sift_database_valid` cases pass against the old `sqlite3` implementation on
purpose: they pin the observable contract (missing / unreadable / empty DB all -> False) across
Step 5's swap to `pycolmap.Database.open`. A characterisation test that went red here would
mean the rewrite changed behaviour, not that it worked.

- [ ] **Step 5: Rewrite the InstantSfM pieces**

In `collab_splats/pointcloud/sfm/instantsfm.py`:

1. Move `_rename_images_to_stems` here from `collab_splats/wrapper/reconstructor.py:506-517`,
   unchanged apart from the docstring's cross-reference — the rename is the creator's output
   contract, not the Reconstructor's:

```python
def _rename_images_to_stems(recon: pycolmap.Reconstruction, sparse_dir: Path) -> None:
    """
    Rename COLMAP images to their filename stems and rewrite the binary model in place.

    - InstantSfM registers images under their filenames (frame_000000.jpg); the pipeline
      contract is frame_{source_idx:06d} with NO extension (see PointcloudResult.from_colmap).
    - pycolmap.Image.name is settable by reference, so the rename lands on the model itself.
    """
    for im in recon.images.values():
        im.name = Path(im.name).stem
    recon.write_binary(str(sparse_dir))
```

The upstream `instantsfm.*` imports stay function-local in `_build_config` and `reconstruct`.
CLAUDE.md's "imports at top" rule exempts optional heavy deps, instantsfm is not installed in
this venv, and hoisting them behind a `try/except` would buy nothing here — no test in this
file calls `reconstruct`, so nothing needs those names monkeypatchable at module scope.

2. Replace `_sift_database_valid` — `pycolmap.Database` reads the same file colmap wrote, so
   the hand-rolled SQL and the `sqlite3` import both go:

```python
def _sift_database_valid(database_path: Path) -> bool:
    """
    True when the SIFT DB holds both extraction and matching output.

    - A crashed colmap subprocess (e.g. OOM-killed under the cgroup cap) leaves a partial or
      unreadable DB behind; an existence-only check would cache-hit on it and feed
      ReadColmapDatabase zero tracks.
    - Database.open CREATES the file when absent, so the exists() pre-check must stay; it
      raises RuntimeError (not sqlite3.Error) on a file no registered factory can open.
    """
    if not database_path.exists():
        return False
    try:
        db = pycolmap.Database.open(str(database_path))
    except RuntimeError:
        return False
    try:
        return db.num_keypoints() > 0 and db.num_verified_image_pairs() > 0
    finally:
        db.close()
```

3. Turn `_SIFT_NUM_THREADS` into a keyword default and drop `single_camera` — the
   `depth_vda/images/npy` layout is upstream's single-camera branch, so `False` was never a
   working path:

```python
def _generate_sift_database(image_path: Path, database_path: Path, *, num_threads: int = 8) -> None:
    """
    Build the COLMAP SIFT feature database: extraction + exhaustive matching.

    - num_threads: CPU SIFT thread cap. colmap's default (-1) spawns one thread per HOST core
      — 96 here — and per-thread RAM blows past the 46.6 GB container cgroup cap (measured:
      OOM-kill at default, clean 1.5 min run at 8 threads on 100 frames of 1920x1080).
    - Drives the colmap CLI, not pycolmap: the system binary is the CUDA build (GPU SIFT,
      measured 100x1920x1080 extraction 7 s vs 90 s, matching 55 s vs ~816 s) while the wheel
      is CPU-only. Reimplements upstream GenerateDatabase (cre185/InstantSfM
      instantsfm/controllers/feature_handler.py:18-57 @ d3e599e), which forces CPU with no
      thread cap and swallows CalledProcessError — a colmap crash there surfaces only as an
      empty-tracks IndexError much later.
    - On failure the partial DB is unlinked so a re-run rebuilds from scratch.
    """
    env = os.environ.copy()
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    extractor_cmd = [
        "colmap",
        "feature_extractor",
        "--image_path",
        str(image_path),
        "--database_path",
        str(database_path),
        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        "1" if use_gpu else "0",
    ]
    matcher_cmd = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--SiftMatching.use_gpu",
        "1" if use_gpu else "0",
    ]
    if not use_gpu:
        extractor_cmd += ["--SiftExtraction.num_threads", str(num_threads)]
        matcher_cmd += ["--SiftMatching.num_threads", str(num_threads)]

    try:
        for cmd in (extractor_cmd, matcher_cmd):
            logger.info("InstantSfM: running %s %s (%s)", cmd[0], cmd[1], "gpu" if use_gpu else "cpu")
            subprocess.run(cmd, check=True, env=env)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        database_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"COLMAP SIFT database build failed ({err}) — is the `colmap` binary installed "
            "(CUDA-built for the GPU path) and is there enough memory?"
        ) from err
```

Delete the `_SIFT_NUM_THREADS` constant and its comment block.

4. Trim the dataclass and take the stem rename in:

```python
@dataclass
class InstantSfMCreator:
    """
    Global SfM via InstantSfM's python API on a staged scene directory.

    - use_depths: feed depth_vda/ maps into the solve as depth priors.
    - retriangulation: GLOMAP-style retriangulate + re-BA after the global solve.
    - random_seed: seeds InstantSfM's RUNTIME_OPTIONS; None = unseeded (upstream draws initial
      camera translations and track xyzs from an unseeded uniform, so runs differ).
    - reconstruct(data_dir): data_dir/images/ (+ depth_vda/) -> pycolmap.Reconstruction whose
      image names are filename stems (frame_NNNNNN); model written to data_dir/colmap/sparse/0,
      SIFT DB at data_dir/colmap/instantsfm.db.
    - Not a BasePointcloudCreator on purpose: a staged scene dir in, a Reconstruction out; the
      Reconstructor wraps it.
    - License: CC-BY-NC-4.0 (non-commercial) — cleared for this repo's research use; revisit
      before any commercial deployment.
    """

    use_depths: bool = True
    retriangulation: bool = False
    random_seed: int | None = None
```

`_build_config` becomes `config = Config("colmap")` (upstream 0.3.0 supports no other feature
backend), the `_generate_sift_database` call loses its third positional argument, and
`reconstruct` ends with the rename instead of returning straight after the read-back:

```python
        # Read back the written model, then rename its images to the pipeline's stem contract
        recon = pycolmap.Reconstruction(str(sparse_dst))
        _rename_images_to_stems(recon, sparse_dst)
        logger.info("InstantSfM: %d registered images, %d points3D", recon.num_reg_images(), recon.num_points3D())
        return recon
```

- [ ] **Step 6: Run them to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_instantsfm.py -x -q
```

Expected: PASS.

- [ ] **Step 7: Split the creator tests**

Create `tests/pointcloud/sfm/test_colmap.py` from `tests/pointcloud/test_sfm_creator.py` lines
1-75 (the `tiny_image_dir` fixture and the five `test_colmap_creator_*` tests), with the imports
rewritten to:

```python
from collab_splats.pointcloud.sfm import ColmapCreator
from collab_splats.pointcloud.base import PointcloudResult
```

and every patch target retargeted from `collab_splats.pointcloud.sfm.pycolmap.*` to
`collab_splats.pointcloud.sfm.colmap.pycolmap.*` (lines 38-40 and 49-51). Drop the
`CoordinateFrame` import and any `result.frame` assertion (Task 3 removed the field).

Create `tests/pointcloud/sfm/test_hloc.py` from lines 77-200 (the three `test_hloc_creator_*`
tests), copying the `tiny_image_dir` fixture in verbatim — the two files are independent —
importing `from collab_splats.pointcloud.sfm import HlocCreator`, and retargeting every
`collab_splats.pointcloud.sfm.<name>` patch to `collab_splats.pointcloud.sfm.hloc.<name>`.

Move lines 202-221 (`test_instantsfm_random_seed_defaults_to_none`,
`test_instantsfm_random_seed_reaches_runtime_options`) into
`tests/pointcloud/sfm/test_instantsfm.py`.

```bash
git rm tests/pointcloud/test_sfm_creator.py tests/pointcloud/test_instantsfm.py
```

- [ ] **Step 8: Drop `features` / `single_camera` and the rename helper from the wrapper**

In `collab_splats/wrapper/reconstructor.py`:

1. Delete `_INSTANTSFM_FEATURES = {"colmap"}` (line 78).
2. Delete the `features` allowlist check in `validate_config` (lines 881-883) and the
   `features = ...` lookup feeding it.
3. Delete `_rename_images_to_stems` (lines 506-517) — Step 5 moved it into
   `sfm/instantsfm.py`.
4. In `_run_sfm`, the creator construction and the rename call become:

```python
        creator = InstantSfMCreator(
            retriangulation=pc_cfg["instantsfm"]["retriangulation"],
            random_seed=pc_cfg["instantsfm"]["random_seed"],
        )
        recon = creator.reconstruct(backend_dir)
```

(the `_rename_images_to_stems(recon, ...)` line goes — the creator does it now).

5. In `configs/base.yaml`, delete `features: colmap` from the `instantsfm` block; in
   `configs/README.md`, delete the `pointcloud.instantsfm.features` row; in
   `tests/wrapper/test_sfm_config.py`, delete the `features` allowlist test.

- [ ] **Step 9: Update the import-coverage list**

In `tests/test_cu121_migration.py`, replace `"collab_splats.pointcloud.sfm",` (line 99) with:

```python
        "collab_splats.pointcloud.sfm",
        "collab_splats.pointcloud.sfm.colmap",
        "collab_splats.pointcloud.sfm.hloc",
        "collab_splats.pointcloud.sfm.instantsfm",
        "collab_splats.pointcloud.vda",
        "collab_splats.pointcloud.depth_align",
```

- [ ] **Step 10: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/test_cu121_migration.py -x -q
```

Expected: PASS.

- [ ] **Step 11: Confirm the old module and the dropped fields are gone**

```bash
rtk proxy grep -rn "pointcloud\.sfm\b\|pointcloud/sfm\.py\|_INSTANTSFM_FEATURES\|single_camera=\|_SIFT_NUM_THREADS" collab_splats tests configs evals docs/source
rtk proxy grep -rn "_rename_images_to_stems" collab_splats tests
```

Expected: from the first, only `from collab_splats.pointcloud.sfm import ...` package imports
and `ColmapCreator.single_camera` (which keeps the field). From the second, only
`collab_splats/pointcloud/sfm/instantsfm.py` (definition + call) and
`tests/pointcloud/sfm/test_instantsfm.py` — nothing under `collab_splats/wrapper/`.

- [ ] **Step 12: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/sfm collab_splats/wrapper/reconstructor.py tests/pointcloud/sfm tests/wrapper/test_sfm_config.py tests/test_cu121_migration.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/sfm collab_splats/wrapper/reconstructor.py tests/pointcloud/sfm tests/wrapper/test_sfm_config.py tests/test_cu121_migration.py
git add -A collab_splats tests configs
git commit -m "refactor(pointcloud): split sfm.py into the sfm/ backend package"
```

---

## Task 10: `make_creator` loses `use_lc` / `lc_config`

`pointcloud/__init__.py` reaches into `collab_splats.geometry` through a deferred import whose
only reason to exist is the cycle it dodges: `geometry.loop_closure.wrapper` imports
`pointcloud.feedforward`, so `geometry` cannot be imported at `pointcloud` load time. One test
is the only caller. Delete the wrapping and the cycle goes with it; the Reconstructor and the
tutorials already build `LoopClosure(base, config=...)` themselves.

Tasks 3 and 5 already trimmed `CoordinateFrame`, `compute_obb_from_points` and
`get_points_in_mask` out of this file; this task finishes it.

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`
- Test: `tests/geometry/loop_closure/test_wrapper.py:154-161`

- [ ] **Step 1: Rewrite the failing test**

In `tests/geometry/loop_closure/test_wrapper.py`, replace `test_make_creator_with_lc`
(lines 154-161) with:

```python
def test_loop_closure_wraps_a_registry_creator():
    """
    make_creator builds the base creator; LoopClosure wrapping is the caller's job.
    """
    from collab_splats.geometry.loop_closure.wrapper import LoopClosure
    from collab_splats.pointcloud import make_creator
    from collab_splats.pointcloud.feedforward import VGGTXCreator

    creator = LoopClosure(make_creator("vggtx"))

    assert isinstance(creator, LoopClosure)
    assert isinstance(creator.base, VGGTXCreator)
```

- [ ] **Step 2: Run it to verify it passes on the old code too**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/loop_closure/test_wrapper.py -x -q
```

Expected: PASS — the new test exercises the surviving API, so it is green before and after.
This step is the safety net for the deletion in Step 3, not a red-then-green cycle: there is no
new behaviour to drive out, only a parameter to remove.

- [ ] **Step 3: Delete the parameters**

In `collab_splats/pointcloud/__init__.py`, replace `make_creator` with:

```python
def make_creator(name: str, **kwargs) -> BasePointcloudCreator:
    """
    Construct a registered pointcloud creator.

    - name: registry key; see get_creator.
    - kwargs: forwarded to the creator's constructor.
    - Returns the creator instance. Wrap it in LoopClosure yourself if you want loop closure.
    """
    return get_creator(name)(**kwargs)
```

and tighten `get_creator` to the docstring contract:

```python
def get_creator(name: str) -> type[BasePointcloudCreator]:
    """
    Look up a pointcloud creator class by registry name.

    - name: one of _REGISTRY's keys (colmap, hloc, mapanything, vggtx, plus vggt_omega /
      vggt_spark / loger when their optional deps are installed).
    - Returns the class; raises KeyError with the available names when unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
```

- [ ] **Step 4: Confirm the cycle is gone**

Both import orders must work with no deferred-import trick anywhere:

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.loop_closure; import collab_splats.pointcloud; print('ok')"
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud; import collab_splats.geometry.loop_closure; print('ok')"
rtk proxy grep -n "use_lc\|lc_config\|from collab_splats.geometry" collab_splats/pointcloud/__init__.py
```

Expected: `ok` twice, and no grep output.

- [ ] **Step 5: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/geometry -x -q
```

Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/pointcloud/__init__.py tests/geometry/loop_closure/test_wrapper.py
/opt/venv/reconstruction/bin/isort collab_splats/pointcloud/__init__.py tests/geometry/loop_closure/test_wrapper.py
git add collab_splats/pointcloud/__init__.py tests/geometry/loop_closure/test_wrapper.py
git commit -m "refactor(pointcloud): drop use_lc from make_creator; break the geometry import cycle"
```

---

## Task 11: `_run_sfm` becomes orchestration only; hoist the pointcloud imports

Tasks 2 and 5-9 replaced every inline pointcloud routine in the Reconstructor with a module
call. What is left is bookkeeping: the method's docstring still describes work it no longer
does, and four pointcloud symbols are still imported inside function bodies against the repo's
"imports at top" rule — the import cycle that justified them died with `export.py` (Task 2) and
`make_creator(use_lc=)` (Task 10).

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:23,25-34,562,1021-1023,1148-1160,1431,1769`
- Test (regression net, unchanged): `tests/wrapper/test_reconstructor.py`,
  `tests/wrapper/test_sfm_wiring.py`

- [ ] **Step 1: Hoist the pointcloud imports**

Pure refactor, no new test: Step 2 catches the one failure a hoist can actually cause (a
resurrected import cycle), and the wrapper suite in Step 4 covers the behaviour.

In `collab_splats/wrapper/reconstructor.py`, make the pointcloud import block read:

```python
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.depth_align import result_from_reconstruction
from collab_splats.pointcloud.sfm import InstantSfMCreator
from collab_splats.pointcloud.utils import (
    clean_pointcloud,
    confidence_mask,
    lift_features,
)
from collab_splats.pointcloud.vda import generate_vda_depth
```

then delete these function-local lines:
- `from collab_splats.pointcloud.utils import lift_features` in `_lift_and_save` (line 562)
- `from collab_splats.pointcloud.base import PointcloudResult` in `_load_pointcloud_from_disk`
  (line 1023) and its `import pycolmap` (1021) if Task 2's rewrite left either behind
- `from collab_splats.pointcloud.base import PointcloudResult` in `_run_sfm` (line 1148)
- `from collab_splats.pointcloud.utils import confidence_mask` in the splats stage (line 1769)
- the duplicate `from vggt.utils.geometry import unproject_depth_map_to_point_map` in
  `refine_poses` (line 1431) — the module-level one at line 23 stays, `refine_poses` is now its
  only user.

`PointcloudResult` is imported for real now, so drop it from the `TYPE_CHECKING` block
(line 59) and change every `-> "PointcloudResult"` annotation in the file to
`-> PointcloudResult`.

- [ ] **Step 2: Verify the module imports both ways**

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.wrapper.reconstructor; print('ok')"
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud, collab_splats.wrapper.reconstructor; print('ok')"
```

Expected: `ok` twice. A resurrected `pointcloud -> geometry.loop_closure -> pointcloud` cycle
surfaces here as an ImportError at import time, not in any test.

- [ ] **Step 3: Bring `_run_sfm` to its final shape**

Replace the whole method with:

```python
    def _run_sfm(self) -> PointcloudResult:
        """
        SfM pointcloud path: staged keyframes -> VDA metric depth -> InstantSfM global mapping.

        - Stages frames.zarr keyframes to backend_dir/images/ (InstantSfM reads a directory).
        - Returns the PointcloudResult for the shared tail; writes pointcloud.zarr with the
          alignment and version provenance on the way.
        """
        pc_cfg = self.config["pointcloud"]
        backend = pc_cfg["backend"]
        if backend != "instantsfm":
            raise NotImplementedError(f"sfm backend {backend!r} is not implemented — only 'instantsfm' is")
        backend_dir = self.backend_dir
        backend_dir.mkdir(parents=True, exist_ok=True)
        store = FrameStore.open(self.frames_zarr)
        names = [f"frame_{int(fi):06d}.jpg" for fi in store.frame_indices()]

        # Stage keyframes as jpgs — exactly what FrameStore.export writes, so a complete staged set
        # is reused as-is. Any other set (partial, or from a different selection) is re-staged,
        # and the SIFT database keyed on it is dropped so InstantSfM cannot reuse stale features.
        image_dir = backend_dir / "images"
        staged = sorted(p.name for p in image_dir.iterdir()) if image_dir.is_dir() else []
        if staged != names:
            shutil.rmtree(image_dir, ignore_errors=True)
            (backend_dir / "colmap" / "instantsfm.db").unlink(missing_ok=True)
            store.export(image_dir, ext="jpg")
            logger.info("Staged %d keyframes to %s", len(names), image_dir)

        # VDA metric depth for every keyframe, cached across runs by stem
        frames = np.ascontiguousarray(store.images())
        depths = generate_vda_depth(frames, backend_dir, names)

        # Global SfM via the upstream python API; writes colmap/instantsfm.db + colmap/sparse/0
        # and returns a model whose image names are the frame_NNNNNN stems
        creator = InstantSfMCreator(
            retriangulation=pc_cfg["instantsfm"]["retriangulation"],
            random_seed=pc_cfg["instantsfm"]["random_seed"],
        )
        recon = creator.reconstruct(backend_dir)
        del creator
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Dense result at VDA depth resolution, already rescaled into the COLMAP world — the zarr
        # and the model must share one scale (splat depth targets, mesh fusion, localization)
        outputs, align_attrs = result_from_reconstruction(recon, depths, frames, names)

        zarr_path = backend_dir / "pointcloud.zarr"
        outputs.save_zarr(
            zarr_path,
            extra_attrs={
                "method": "sfm",
                "backend": "instantsfm",
                "instantsfm_version": importlib.metadata.version("instantsfm"),
                **align_attrs,
            },
        )
        logger.info("pointcloud.zarr saved: %s  (%s pts)", zarr_path, f"{len(outputs.points):,}")

        return PointcloudResult(reconstruction=recon, image_paths=outputs.image_paths)
```

- [ ] **Step 4: Run the wrapper suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper -x -q
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/black collab_splats/wrapper/reconstructor.py
/opt/venv/reconstruction/bin/isort collab_splats/wrapper/reconstructor.py
git add collab_splats/wrapper/reconstructor.py
git commit -m "refactor(wrapper): _run_sfm orchestrates only; hoist pointcloud imports to module scope"
```

---

## Task 12: Documentation and stale path references

Earlier tasks deleted each config key beside the code that read it. What is left is prose and
comments that still name `pointcloud/sfm.py`, `export.py` or the deleted knobs. These are the
lines a future reader would trust and be wrong.

**Files:**
- Modify: `configs/README.md:355,491`
- Modify: `docs/source/api/pointcloud.rst:10-12`
- Modify: `CLAUDE.md:57-59`
- Modify: `collab_splats/preproc/undistort.py:83`
- Modify: `collab_splats/remote/sources.py:71`
- Modify: `setup.sh:74`
- Modify: `docs/superpowers/CHANGELOG.md`

- [ ] **Step 1: Fix the `configs/README.md` backend row and the sfm prose**

Line 355 — replace `pointcloud/sfm.py` with the package path:

```markdown
| `pointcloud.backend` | str | `vggt_omega` | feedforward: `vggt_omega`, `vggtx`, `mapanything`, or `loger`; sfm: `instantsfm` (`colmap`/`hloc` validate — `ColmapCreator`/`HlocCreator` exist in `pointcloud/sfm/` — but are not wired into `Reconstructor._run_sfm`, which raises `NotImplementedError`) |
```

Line 491 — the `instantsfm.features` clause described a check Task 9 deleted:

```markdown
**Unsupported with sfm (all `ValueError` at config validation):** `bundle_adjustment: true`
(InstantSfM runs its own global BA; `refine_poses` / `--stages refine` also refuse) and
`loop_closure` (global mapper, not a sequential submap pipeline).
```

- [ ] **Step 2: Point the API docs at the new modules**

In `docs/source/api/pointcloud.rst`, replace the single `collab_splats.pointcloud.sfm`
automodule block (lines 10-12) with:

```rst
.. automodule:: collab_splats.pointcloud.sfm.colmap
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.sfm.hloc
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.sfm.instantsfm
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.vda
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.depth_align
   :members:
   :show-inheritance:
```

- [ ] **Step 3: Update the architecture tree in `CLAUDE.md`**

Replace lines 57-59 with:

```markdown
    sfm/                   # ColmapCreator, HlocCreator, InstantSfMCreator (global SfM via upstream python API)
    vda.py                 # generate_vda_depth: Video-Depth-Anything metric depth per keyframe
    depth_align.py         # result_from_reconstruction: COLMAP model + VDA depth -> FeedforwardResult at COLMAP scale
    utils.py               # lift_features, reproject_pixels, clean_pointcloud, confidence_mask, subsample_points
```

(the `export.py` line goes — the writer is `PointcloudResult.write_ply` now.) Also update the
one-line pipeline summary above the tree: `sfm: InstantSfM + VDA depth` is still accurate, so it
stays.

- [ ] **Step 4: Fix the three stale code comments**

`collab_splats/preproc/undistort.py:83`:

```python
# pointcloud/sfm/instantsfm.py::_generate_sift_database.
```

`collab_splats/remote/sources.py:71`:

```python
    # InstantSfM's SIFT database (pointcloud/sfm/instantsfm.py) — same class of artifact, its own
```

`setup.sh:74`:

```bash
# Imported from the clone root via sys.path (collab_splats/pointcloud/vda.py:generate_vda_depth).
```

- [ ] **Step 5: Verify no stale path survives**

```bash
rtk proxy grep -rn "pointcloud/sfm\.py\|pointcloud/export\.py\|write_pointcloud_ply\|transforms\.json\|vda_context_fps\|export_max_points\|instantsfm\.features\|instantsfm\.depth_align" collab_splats tests configs docs evals setup.sh CLAUDE.md
```

Expected: only hits inside `docs/superpowers/specs/` and `docs/superpowers/plans/` (the design
spec and this plan, which describe the old state on purpose).

- [ ] **Step 6: Append the CHANGELOG entry**

Add to `docs/superpowers/CHANGELOG.md`, newest first, matching the existing entry format:

```markdown
## 2026-09-05 — pointcloud cleanup

Reduced `collab_splats/pointcloud/` to the one SfM path that ships.
[spec](specs/2026-09-05-pointcloud-cleanup-design.md) · [plan](plans/2026-09-05-pointcloud-cleanup.md)

- `export.py` deleted; PLY writing is `PointcloudResult.write_ply` (pycolmap `export_PLY`,
  byte-identical output). `PointcloudResult.from_colmap` replaces the wrapper's disk loader.
- `CoordinateFrame`, `PointcloudResult.frame` / `.world_transform` / `.confidence` deleted —
  every producer wrote COLMAP/identity/None and no consumer branched on them.
- `transforms.json` deleted: a nerfstudio-format artifact with no reader in the repo.
- `sfm.py` split into `sfm/{colmap,hloc,instantsfm}.py`, plus `vda.py`
  (`generate_vda_depth` returns the depth stack) and `depth_align.py`
  (`result_from_reconstruction` builds the `FeedforwardResult` already at COLMAP scale).
- The affine depth-alignment model and `pointcloud.instantsfm.depth_align` are gone; scale
  alignment is the only path. `instantsfm.features` and `InstantSfMCreator.single_camera` are
  gone (one valid value each, never dispatched on).
- The VDA context stream (`preproc.vda_context_fps`, `preproc/video.py::decode_context`) is
  gone — measured 2026-08-26 and refuted for the metric path.
- `utils.py` pruned to what has callers; `clean_pointcloud` returns a keep mask and the
  Reconstructor's hand-rolled open3d block (whose `voxel_size` was a no-op) is one call.
- `Reconstructor._run_sfm` is orchestration only; pointcloud imports are at module scope, and
  the `pointcloud -> geometry.loop_closure` cycle is gone (`make_creator` lost `use_lc`).
```

Nothing leaves `CLAUDE.md`'s `## In-Flight Work` list: this work was specced and planned in
one pass, so it never had an entry there.

- [ ] **Step 7: Commit**

```bash
git add -A configs docs CLAUDE.md setup.sh collab_splats
git commit -m "docs(pointcloud): retarget paths at the sfm package; changelog entry"
```

---

## Task 13: Full verification

Nothing new is written here. This is the gate that says the refactor is done.

**Files:** none modified unless a check fails.

- [ ] **Step 1: Import order both ways**

The cycle this refactor removed is invisible to the test suite (pytest imports in one order).
Check both:

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.loop_closure; import collab_splats.pointcloud; print('ok')"
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud; import collab_splats.geometry.loop_closure; print('ok')"
```

Expected: `ok` twice.

- [ ] **Step 2: Full affected suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/geometry tests/evals tests/preproc tests/examples tests/test_cu121_migration.py -q
```

Expected: all pass. Cross-check any failure against `docs/known-test-failures.md` before
treating it as a regression.

- [ ] **Step 3: Whole suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
```

Expected: pass, modulo the documented known failures.

- [ ] **Step 4: Dead-symbol sweep**

Every name this refactor deleted, in one pass:

```bash
rtk proxy grep -rn "CoordinateFrame\|world_transform\|write_pointcloud_ply\|_write_ply\|_export_pointcloud_ply\|_write_transforms_json\|_load_pointcloud_from_disk\|_clean_pointcloud\|_ensure_vda_depth\|_context_keep_rows\|_video_unchanged\|decode_context\|vda_depth_complete\|VDA_CHECKPOINT\|apply_depth_alignment\|align_depth_affine\|align_depth_to_reconstruction\|DEPTH_ALIGN_MODELS\|DepthAlignModel\|_sfm_result_from_reconstruction\|_INSTANTSFM_FEATURES\|_SIFT_NUM_THREADS\|clean_pcd\|remove_far_points\|density_filter\|voxel_downsample_point_cloud\|compute_obb_from_points\|get_points_in_mask\|filter_distance" collab_splats tests evals configs docs/source setup.sh CLAUDE.md
```

Expected: no output. Hits inside `docs/superpowers/` are expected and excluded above.

- [ ] **Step 5: Lint the touched files**

```bash
/opt/venv/reconstruction/bin/python -m pyflakes collab_splats/pointcloud collab_splats/wrapper/reconstructor.py collab_splats/preproc/video.py
```

Expected: no output. (Do NOT run `black .` / `isort .` — the venv's black is newer than the
version the repo was formatted with, and a repo-wide run rewrites unrelated files.)

- [ ] **Step 6: Refresh the knowledge graph**

```bash
graphify update .
```

- [ ] **Step 7: Commit anything the checks moved**

```bash
git status --short
git add -A collab_splats tests graphify-out
git commit -m "chore(pointcloud): post-refactor verification pass"
```

(Skip the commit if `git status --short` is clean.)

---
