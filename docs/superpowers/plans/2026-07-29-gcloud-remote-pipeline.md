# GCloud Remote Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `run_pipeline` pull a scene video from the `environments-curated` GCS bucket, reconstruct it, push consumer-ready outputs (binary `sparse_pc.ply`, `mesh/mesh.ply`, compressed semantics, intermediates) to `environments-processed`, verify the push, and delete the local copy.

**Architecture:** The dashboard's rclone wrapper (`collab_splats/dashboard/sources.py::SessionSource`) is retargeted at the new flat one-video-per-directory buckets, renamed `SceneSource`, and relocated to a new `collab_splats/remote/` package so both the dashboard and CLI share it. The per-scene driver logic currently inlined in `docs/examples/run_pipeline.py` moves into `collab_splats/wrapper/batch.py`, leaving two thin drivers: `run_pipeline.py` (local paths) and `run_pipeline_remote.py` (scene ids or `--all`). Along the way the pointcloud gains an explicit binary PLY writer, the mesh output name unifies on `mesh/mesh.ply`, and semantics converges on one compressed artifact set.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), rclone (remote `collab-data`), zarr v3, numpy, torch, open3d, pycolmap, nerfstudio, Panel (dashboard), pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-gcloud-remote-pipeline-design.md`

---

## File Structure

**New files**

| path | responsibility |
|---|---|
| `collab_splats/pointcloud/export.py` | `write_pointcloud_ply(points, colors, path, max_points=None)` — binary-little-endian PLY writer. Only file that knows the PLY byte layout. |
| `collab_splats/remote/__init__.py` | Re-export `SceneSource`, `PULL_EXCLUDES`, `PUSH_EXCLUDES`, `parse_rclone_percent`. |
| `collab_splats/remote/sources.py` | `SceneSource` — list/fetch `environments-curated`, push/pull/verify `environments-processed`. Moved from `collab_splats/dashboard/sources.py`, retargeted to flat scene ids. |
| `collab_splats/wrapper/batch.py` | `build_scene_config`, `run_scene`, `run_all`, `scene_output_dir`, `collect_videos` — per-scene driver core, importable by both CLI drivers and by tests. Moved out of `docs/examples/run_pipeline.py`. |
| `docs/examples/run_pipeline_remote.py` | Remote driver: fetch → `run_scene` → push → verify → delete. |
| `tests/pointcloud/test_export.py` | PLY writer tests. |
| `tests/remote/__init__.py` | (empty) package marker so `tests/remote` collects. |
| `tests/remote/test_sources.py` | `SceneSource` tests (moved + retargeted from `tests/dashboard/test_sources*.py`). |
| `tests/wrapper/test_batch.py` | `batch.py` tests. |
| `tests/examples/test_run_pipeline_remote.py` | Remote-driver tests (fake `SceneSource`). |

**Modified files**

| path | change |
|---|---|
| `collab_splats/pointcloud/base.py` | `_write_ply` helper on `BasePointcloudCreator`; docstring says binary. |
| `collab_splats/pointcloud/feedforward/base.py` | Build `PointcloudResult` before returning, write binary PLY over nerfstudio's ASCII one. |
| `collab_splats/wrapper/reconstructor.py` | Read `pointcloud.max_points`; re-export PLY post-clean; `_write_transforms_json` merges instead of clobbering; semantics writes `features.zarr` + `autoencoder.pt`. |
| `configs/base.yaml` | `pointcloud.max_points`, `pointcloud.export_max_points`, `semantics.target_cosine`, `semantics.max_epochs`. |
| `collab_splats/semantics/compression.py` | `fit(..., target_cosine=None, epochs=...)` early stop + `recon_cosine`/`recon_mse`/`epochs_run` in the checkpoint. |
| `collab_splats/mesh/tsdf.py` | `mesh_tsdf.ply` → `mesh.ply`. |
| `collab_splats/wrapper/splatter.py` | mesh candidates → `mesh.ply`. |
| `collab_splats/dashboard/{app,localize,pipeline,shell}.py` | `SceneSource` import; `(session, stem)` → `scene`; `mesh.ply`; `features.zarr`. |
| `collab_splats/mesh/utils.py` | vertex features at latent dim (no 768-D assumption). |
| `docs/examples/run_pipeline.py` | Thin: argparse + `batch.run_all`. |
| existing tests listed per task | renamed paths / scene ids. |

**Deleted**

| path | why |
|---|---|
| `collab_splats/dashboard/sources.py` | moved to `collab_splats/remote/sources.py` (spec decision 3, no shim). |
| `tests/dashboard/test_sources.py`, `tests/dashboard/test_sources_field.py` | moved to `tests/remote/test_sources.py`. |
| `collab_splats/dashboard/pipeline.py::_write_frames_jpegs` | jpg path retired (spec cleanup 1). |
| `SessionSource.list_field_sessions` / `list_rgb_cameras` / `list_camera_videos` / `fetch_field_video` | multi-camera `fieldwork_curated` layout does not exist under `environments-curated` (spec decision 2). |

---

## Conventions for every task

- Python: `/opt/venv/reconstruction/bin/python` — never bare `python`.
- **Do not run `black` at all.** Write code that is already compliant with `line-length = 120`
  (`pyproject [tool.black]`) by hand. The repo is not black-clean, and file-scoping is *not*
  sufficient: measured on Task 5, `black collab_splats/wrapper/splatter.py` rewrites 37 lines
  of untouched pre-existing code and `tests/wrapper/test_splatter_query.py` 23, burying a
  two-line change. `isort <paths>` on touched files is safe. (Every `black . && isort .` in the
  task steps below is superseded by this rule.)
- The working tree carries pre-existing unrelated changes (`configs/loop_closure.yaml`,
  `data/tutorial/README.md`, untracked `docs/source/tutorials/07_localization/ref_image.jpg`).
  Never stage them. Prefer `git add <explicit paths>` over `git add -A`.
- Any task touching `collab_splats/dashboard/**` must end with `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke` printing `SMOKE PASS`.
- `docs/superpowers/**` is gitignored: stage it with `git add -f`.
- Commit style: conventional with scope, e.g. `feat(pointcloud):`.

---

### Task 1: Binary PLY writer

**Files:**
- Create: `collab_splats/pointcloud/export.py`
- Test: `tests/pointcloud/test_export.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pointcloud/test_export.py`:

```python
"""Binary PLY export: header shape, byte layout, open3d round-trip, density cap."""

import numpy as np
import open3d as o3d
import pytest

from collab_splats.pointcloud.export import write_pointcloud_ply


def _cloud(n=100):
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
    return points, colors


def test_writes_binary_little_endian_header(tmp_path):
    points, colors = _cloud(10)
    out = write_pointcloud_ply(points, colors, tmp_path / "sparse_pc.ply")
    head = out.read_bytes().split(b"end_header\n")[0]
    assert b"format binary_little_endian 1.0" in head
    assert b"element vertex 10" in head
    assert b"property float x" in head
    assert b"property uchar red" in head


def test_payload_is_15_bytes_per_vertex(tmp_path):
    points, colors = _cloud(50)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply")
    raw = out.read_bytes()
    payload = raw.split(b"end_header\n", 1)[1]
    assert len(payload) == 50 * 15


def test_open3d_reads_back_exact_values(tmp_path):
    points, colors = _cloud(200)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply")
    pcd = o3d.io.read_point_cloud(str(out))
    got_pts = np.asarray(pcd.points, dtype=np.float32)
    got_col = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    assert got_pts.shape == points.shape
    np.testing.assert_array_equal(got_pts, points)
    np.testing.assert_array_equal(got_col, colors)


def test_colors_default_to_mid_grey_when_absent(tmp_path):
    points, _ = _cloud(20)
    out = write_pointcloud_ply(points, None, tmp_path / "p.ply")
    pcd = o3d.io.read_point_cloud(str(out))
    col = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    assert col.shape == (20, 3)
    assert np.all(col == 128)


def test_max_points_caps_density(tmp_path):
    points, colors = _cloud(1000)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply", max_points=100)
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) == 100


def test_max_points_none_keeps_every_point(tmp_path):
    points, colors = _cloud(1000)
    out = write_pointcloud_ply(points, colors, tmp_path / "p.ply", max_points=None)
    pcd = o3d.io.read_point_cloud(str(out))
    assert len(pcd.points) == 1000


def test_rejects_mismatched_colors(tmp_path):
    points, colors = _cloud(30)
    with pytest.raises(ValueError, match="colors length"):
        write_pointcloud_ply(points, colors[:5], tmp_path / "p.ply")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_export.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'collab_splats.pointcloud.export'`

- [ ] **Step 3: Write the implementation**

Create `collab_splats/pointcloud/export.py`:

```python
"""Point-cloud export to binary PLY.

Single place that knows the on-disk PLY byte layout. Written as
``format binary_little_endian 1.0`` with float32 xyz + uchar rgb = 15 bytes per
vertex, ~1.5x smaller than the ASCII file nerfstudio's ``create_ply_from_colmap``
emits and lossless in the coordinates (ASCII truncates to 6 significant digits).
Read back byte-exact by open3d, which is what every nerfstudio dataparser uses.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from collab_splats.pointcloud.utils import subsample_points

logger = logging.getLogger(__name__)

########
# Constants
########

# Packed (no alignment padding) => itemsize 15, matching the header below.
_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)

_HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)

# Colour for clouds with no RGB (SfM tracks without image colour, synthetic tests).
_DEFAULT_GREY = 128


########
# Export
########


def write_pointcloud_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: Path,
    max_points: Optional[int] = None,
) -> Path:
    """Write a (P, 3) cloud + optional (P, 3) uint8 colors to path as a binary PLY.

    Args:
        points: (P, 3) float coordinates.
        colors: (P, 3) uint8 RGB, or None for uniform mid-grey.
        path: destination file; parents are created.
        max_points: optional density cap. None (default) writes every point —
            thinning is opt-in so the exported cloud matches the reconstruction.

    Returns:
        The path written.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (P, 3), got {points.shape}")

    # Validate colours before any thinning so the error names the caller's own arrays.
    # uint8 is required, not coerced: PointcloudResult.colors (base.py:49-54) is always
    # uint8, so scale-sniffing a float array would be dead code that silently inverts
    # dark 0-255 floats and crashes on empty ones.
    if colors is not None:
        colors = np.ascontiguousarray(colors)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"colors must be (P, 3), got {colors.shape}")
        if len(colors) != len(points):
            raise ValueError(f"colors length {len(colors)} != points length {len(points)}")
        if colors.dtype != np.uint8:
            raise ValueError(f"colors must be uint8 0-255, got {colors.dtype}")

    # Optional density cap — reuses the pipeline's exact-budget sampler (seeded, reproducible)
    if max_points is not None and len(points) > max_points:
        points, colors = subsample_points(points, colors, max_points=max_points)
        points = np.ascontiguousarray(points, dtype=np.float32)

    # Fill the packed vertex record; colours are already validated uint8 (P, 3)
    verts = np.empty(len(points), dtype=_VERTEX_DTYPE)
    verts["x"] = points[:, 0]
    verts["y"] = points[:, 1]
    verts["z"] = points[:, 2]
    if colors is None:
        verts["red"] = verts["green"] = verts["blue"] = _DEFAULT_GREY
    else:
        verts["red"] = colors[:, 0]
        verts["green"] = colors[:, 1]
        verts["blue"] = colors[:, 2]

    # Header is ASCII, payload is raw little-endian records
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_HEADER.format(n=len(points)).encode("ascii"))
        f.write(verts.tobytes())

    logger.info("wrote %s (%d points)", path, len(points))
    return path
```

> **Landed as `e3e6421` + `f29d9a9`.** Beyond the code above, the shipped tests also cover:
> RGBA / 1-D / non-uint8 colour rejection, nested-parent creation, an empty `(0, 3)` cloud,
> and point↔colour index alignment across the `max_points` cap. 12 tests in the file.
>
> **Downstream caveat for Tasks 2-3:** open3d refuses to parse a zero-vertex PLY — it logs
> `Read PLY failed: number of vertex <= 0` and returns an empty cloud. A genuinely empty
> reconstruction therefore round-trips as "no points" rather than as an error.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_export.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/pointcloud/export.py tests/pointcloud/test_export.py
git commit -m "feat(pointcloud): binary PLY writer with optional density cap"
```

---

### Task 2: Write the binary PLY from the pointcloud stage

nerfstudio's `colmap_to_json` unconditionally writes an ASCII `sparse_pc.ply` (`nerfstudio/process_data/colmap_utils.py:92-99` — no opt-out) and records `ply_file_path` in `transforms.json`. We keep the filename and the `transforms.json` entry, and overwrite the file with our binary version straight from `result.points`.

**Files:**
- Modify: `collab_splats/pointcloud/base.py:100-120`
- Modify: `collab_splats/pointcloud/feedforward/base.py:897-907`
- Test: `tests/pointcloud/test_export_wiring.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/pointcloud/test_export_wiring.py`:

```python
"""BasePointcloudCreator._write_ply turns a result into a binary sparse_pc.ply."""

import numpy as np
import open3d as o3d

from collab_splats.pointcloud.base import BasePointcloudCreator


class _StubResult:
    """Minimal stand-in for PointcloudResult — only points/colors are read."""

    def __init__(self, points, colors):
        self.points = points
        self.colors = colors


class _Creator(BasePointcloudCreator):
    def reconstruct(self, image_dir, output_dir):  # pragma: no cover - not exercised
        raise NotImplementedError


def test_write_ply_emits_binary_sparse_pc(tmp_path):
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, size=(40, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(40, 3)).astype(np.uint8)

    out = _Creator()._write_ply(_StubResult(points, colors), tmp_path)

    assert out == tmp_path / "sparse_pc.ply"
    assert b"format binary_little_endian 1.0" in out.read_bytes()[:80]
    assert len(o3d.io.read_point_cloud(str(out)).points) == 40


def test_write_ply_honours_max_points(tmp_path):
    points = np.zeros((500, 3), dtype=np.float32)
    out = _Creator()._write_ply(_StubResult(points, None), tmp_path, max_points=50)
    assert len(o3d.io.read_point_cloud(str(out)).points) == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_export_wiring.py -v`
Expected: FAIL — `AttributeError: '_Creator' object has no attribute '_write_ply'`

- [ ] **Step 3: Add `_write_ply` to `BasePointcloudCreator`**

In `collab_splats/pointcloud/base.py`, replace the `reconstruct` docstring block and `_write_transforms` (currently lines 100-120) with:

```python
class BasePointcloudCreator(ABC):
    @abstractmethod
    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        """images in image_dir → sparse pointcloud + camera poses written to output_dir.

        Produces:
            {output_dir}/colmap/sparse/0/{cameras,images,points3D}.bin
            {output_dir}/transforms.json
            {output_dir}/sparse_pc.ply   (binary little-endian; see pointcloud/export.py)

        Raises:
            RuntimeError: if reconstruction fails
            FileNotFoundError: if image_dir does not exist
        """
        ...

    def _write_transforms(self, sparse_dir: Path, output_dir: Path) -> None:
        from nerfstudio.process_data.colmap_utils import colmap_to_json

        colmap_to_json(recon_dir=sparse_dir, output_dir=output_dir)

    def _write_ply(self, result, output_dir: Path, max_points: int | None = None) -> Path:
        """Overwrite nerfstudio's ASCII sparse_pc.ply with our binary one.

        colmap_to_json always emits an ASCII sparse_pc.ply and points transforms.json
        at that filename, so we keep the name and replace the bytes — smaller file,
        no coordinate truncation, same consumers.
        """
        from collab_splats.pointcloud.export import write_pointcloud_ply

        return write_pointcloud_ply(result.points, result.colors, Path(output_dir) / "sparse_pc.ply", max_points)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_export_wiring.py -v`
Expected: 2 passed

- [ ] **Step 5: Call it from the feedforward assembly path**

In `collab_splats/pointcloud/feedforward/base.py`, replace the tail of `_assemble_result` (currently lines 897-907, the block starting `# Write binary COLMAP reconstruction to disk`) with:

```python
        # Write binary COLMAP reconstruction to disk and export transforms.json
        sparse_dir = Path(output_dir) / "colmap" / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        recon.write_binary(str(sparse_dir))
        self._write_transforms(sparse_dir, Path(output_dir))

        # Replace the ASCII sparse_pc.ply colmap_to_json just wrote with a binary one
        result = PointcloudResult(
            reconstruction=recon,
            frame=CoordinateFrame.COLMAP,
            image_paths=o.image_paths,
        )
        self._write_ply(result, Path(output_dir))
        console.log(f"  done in {time.perf_counter() - t0:.1f}s")
        return result
```

- [ ] **Step 6: Run the pointcloud suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ -q`
Expected: all pass, no new failures vs `docs/known-test-failures.md`

- [ ] **Step 7: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/pointcloud/base.py collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_export_wiring.py
git commit -m "feat(pointcloud): emit binary sparse_pc.ply from the feedforward path"
```

> **Note for the implementer:** `collab_splats/pointcloud/sfm.py:197` also calls `_write_transforms`. `pointcloud.method: sfm` raises before reaching it (`configs/base.yaml` marks it NOT YET IMPLEMENTED, `reconstructor.py:519` warns it is untested), so it is deliberately left alone — do not add a `_write_ply` call there.

---

### Task 3: Expose the point cap in config, re-export post-clean

`max_points` (`feedforward/base.py:794`) caps the *confidence mask* during inference so a full-resolution cloud is never materialised — removing it produced a 5.1 GB `points3D.bin` and a 50 GB SIGKILL. It stays as a memory guard and becomes configurable. Export density is a separate, off-by-default knob.

The clean step (`reconstructor.py:537-540`) can drop points *after* the creator wrote the PLY, so the Reconstructor re-exports post-clean.

**Files:**
- Modify: `configs/base.yaml:26-40`
- Modify: `collab_splats/wrapper/reconstructor.py:131-138` (`_run_feedforward` signature), `:175-181` (creator construction), `:526-546` (call site + post-clean export)
- Test: `tests/wrapper/test_reconstructor_export.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/wrapper/test_reconstructor_export.py`:

```python
"""pointcloud.max_points / export_max_points reach the creator and the PLY."""

import numpy as np
import open3d as o3d
import pytest
import yaml

from collab_splats.wrapper.reconstructor import Reconstructor


def _config(tmp_path, **pc):
    return {
        "input_path": str(tmp_path / "in.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega", **pc},
    }


def test_defaults_expose_max_points(tmp_path):
    r = Reconstructor(_config(tmp_path))
    assert r.config["pointcloud"]["max_points"] == 500_000
    assert r.config["pointcloud"]["export_max_points"] is None


def test_max_points_override_survives_merge(tmp_path):
    r = Reconstructor(_config(tmp_path, max_points=120_000, export_max_points=50_000))
    assert r.config["pointcloud"]["max_points"] == 120_000
    assert r.config["pointcloud"]["export_max_points"] == 50_000


def test_export_pointcloud_ply_rewrites_after_clean(tmp_path):
    """_export_pointcloud_ply writes backend_dir/sparse_pc.ply from the final result."""

    class _Result:
        points = np.zeros((300, 3), dtype=np.float32)
        colors = None

    r = Reconstructor(_config(tmp_path, export_max_points=100))
    r.backend_dir.mkdir(parents=True, exist_ok=True)
    out = r._export_pointcloud_ply(_Result())
    assert out == r.backend_dir / "sparse_pc.ply"
    assert len(o3d.io.read_point_cloud(str(out)).points) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_export.py -v`
Expected: FAIL — `KeyError: 'max_points'` on the first two, `AttributeError: ... '_export_pointcloud_ply'` on the third

- [ ] **Step 3: Add the config keys**

In `configs/base.yaml`, in the `pointcloud:` block, insert after the `loop_closure: false` line:

```yaml
  # Inference-time memory guard: caps the confidence mask so a full-resolution cloud is
  # never materialised (feedforward/base.py:794). Lower it on small-GPU/RAM machines.
  max_points: 500000
  # Export density for sparse_pc.ply. null = write every reconstructed point (default);
  # set an int to thin the exported file only — inference is unaffected.
  export_max_points: null
```

- [ ] **Step 4: Thread `max_points` into the creator**

In `collab_splats/wrapper/reconstructor.py`, change the `_run_feedforward` signature (line 131) to add a keyword after `viz_port`:

```python
def _run_feedforward(
    backend: str,
    frames_zarr: Path,
    output_dir: Path,
    loop_closure: bool | dict,
    viz_enabled: bool,
    viz_port: int,
    max_points: int,
) -> tuple["PointcloudResult", "Viewer | None"]:
```

and change creator construction (line 181) from `creator = creator_map[backend]()` to:

```python
    # max_points caps the confidence mask during inference — a memory guard, not a preference
    creator = creator_map[backend](max_points=max_points)
```

- [ ] **Step 5: Pass it at the call site and re-export post-clean**

In `collab_splats/wrapper/reconstructor.py`, add `max_points=pc_cfg["max_points"],` to the `_run_feedforward(...)` call (after `viz_port=...`, line 533), and insert the export right after the clean step, before `self._write_transforms_json(result)`:

```python
        # Apply cleaning step if enabled
        clean_cfg = pc_cfg["clean"]
        if clean_cfg["enabled"]:
            result = self._clean_pointcloud(result, clean_cfg)

        # Re-export the PLY from the FINAL result — clean may have dropped points since
        # the creator wrote its copy. Density is opt-in via pointcloud.export_max_points.
        self._export_pointcloud_ply(result)

        # Write nerfstudio-compatible transforms.json
        self._write_transforms_json(result)
```

Add the method next to `_write_transforms_json` (near line 604):

```python
    def _export_pointcloud_ply(self, result) -> Path:
        """Write backend_dir/sparse_pc.ply (binary) from the post-clean result."""
        from collab_splats.pointcloud.export import write_pointcloud_ply

        self.backend_dir.mkdir(parents=True, exist_ok=True)
        return write_pointcloud_ply(
            result.points,
            result.colors,
            self.backend_dir / "sparse_pc.ply",
            self.config["pointcloud"]["export_max_points"],
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_export.py tests/wrapper/ -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add configs/base.yaml collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor_export.py
git commit -m "feat(pointcloud): config-driven max_points guard + opt-in export density"
```

---

### Task 4: Stop `_write_transforms_json` clobbering `ply_file_path`

`_run_feedforward` passes `output_dir=self.backend_dir` (`reconstructor.py:530`), so nerfstudio's `colmap_to_json` writes `backend_dir/transforms.json` with `ply_file_path` and `applied_transform`. Then `_write_transforms_json` (`:604-637`) writes the *same* path with only `{"camera_model", "frames"}`, deleting both keys — splatfacto loses its seed cloud. Fix: merge into whatever is already there.

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:604-640`
- Test: `tests/wrapper/test_transforms_merge.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/wrapper/test_transforms_merge.py`:

```python
"""_write_transforms_json must preserve keys colmap_to_json already wrote."""

import json

import numpy as np

from collab_splats.wrapper.reconstructor import Reconstructor


class _Result:
    """Two-frame stand-in: only extrinsics/intrinsics/image_paths are read."""

    points = np.zeros((3, 3), dtype=np.float32)
    colors = None
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 2)
    intrinsics = np.stack([np.eye(3, dtype=np.float32)] * 2)
    image_paths = ["frame_000000.jpg", "frame_000001.jpg"]


def _reconstructor(tmp_path):
    return Reconstructor(
        {
            "input_path": str(tmp_path / "in.mp4"),
            "output_path": str(tmp_path / "out"),
            "pointcloud": {"method": "feedforward", "backend": "vggt_omega"},
        }
    )


def test_preserves_ply_file_path_and_applied_transform(tmp_path):
    r = _reconstructor(tmp_path)
    r.backend_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        "camera_model": "PINHOLE",
        "ply_file_path": "sparse_pc.ply",
        "applied_transform": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]],
        "frames": [{"file_path": "stale.jpg"}],
    }
    (r.backend_dir / "transforms.json").write_text(json.dumps(existing))

    r._write_transforms_json(_Result())

    out = json.loads((r.backend_dir / "transforms.json").read_text())
    assert out["ply_file_path"] == "sparse_pc.ply"
    assert out["applied_transform"] == existing["applied_transform"]
    # frames are ours, not the stale ones
    assert len(out["frames"]) == 2
    assert out["frames"][0]["file_path"] != "stale.jpg"


def test_works_with_no_existing_file(tmp_path):
    r = _reconstructor(tmp_path)
    r.backend_dir.mkdir(parents=True, exist_ok=True)
    r._write_transforms_json(_Result())
    out = json.loads((r.backend_dir / "transforms.json").read_text())
    assert out["camera_model"] == "PINHOLE"
    assert len(out["frames"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_transforms_merge.py -v`
Expected: `test_preserves_ply_file_path_and_applied_transform` FAILS with `KeyError: 'ply_file_path'`; the second test passes

- [ ] **Step 3: Make the writer merge**

In `collab_splats/wrapper/reconstructor.py::_write_transforms_json`, replace the final write block (the `out = self.backend_dir / "transforms.json"` lines through the `json.dump` call, around lines 636-640) with:

```python
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        out = self.backend_dir / "transforms.json"

        # Merge over whatever colmap_to_json already wrote: it owns ply_file_path and
        # applied_transform (splatfacto needs both), we own camera_model + frames.
        payload: dict = {}
        if out.exists():
            with open(out) as f:
                payload = json.load(f)
        payload["camera_model"] = "PINHOLE"
        payload["frames"] = frames

        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
```

Confirm `import json` is at the top of the module; add it if absent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_transforms_merge.py tests/wrapper/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_transforms_merge.py
git commit -m "fix(wrapper): merge transforms.json instead of dropping ply_file_path"
```

---

### Task 5: Unify the mesh filename on `mesh/mesh.ply`

`Reconstructor.mesh()` skip-checks `mesh/mesh.ply` (`reconstructor.py:742`) while TSDF writes `mesh_tsdf.ply` (`mesh/tsdf.py:95`) — so `overwrite=False` never skips and `webapp/routers/visualize.py:51` always reports "no mesh". Renaming the TSDF output fixes both with no new logic.

**Files:**
- Modify: `collab_splats/mesh/tsdf.py:95`
- Modify: `collab_splats/wrapper/splatter.py:480`
- Modify: `collab_splats/dashboard/localize.py:661`, `collab_splats/dashboard/pipeline.py:151,491`, `collab_splats/dashboard/app.py:649-650`
- Modify: `tests/mesh/test_utils.py:133`, `tests/dashboard/test_pipeline.py:133,209`, `tests/dashboard/test_viewer.py` (7 sites), `tests/wrapper/test_splatter_query.py` (5 sites), `tests/wrapper/test_splatter_mesh.py` (3 sites)

- [ ] **Step 1: Write the failing test**

Append to `tests/mesh/test_utils.py`:

```python
def test_tsdf_writes_mesh_ply(tmp_path, monkeypatch):
    """Open3DTSDFFusion output filename must match Reconstructor.mesh()'s skip-check."""
    import open3d as o3d

    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    fusion = Open3DTSDFFusion(output_dir=tmp_path)
    mesh = o3d.geometry.TriangleMesh.create_box()
    monkeypatch.setattr(
        o3d.pipelines.integration.ScalableTSDFVolume,
        "extract_triangle_mesh",
        lambda self: mesh,
    )
    assert fusion.output_dir / "mesh.ply" == tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(tmp_path / "mesh.ply"), mesh)
    assert not (tmp_path / "mesh_tsdf.ply").exists()
```

- [ ] **Step 2: Rename in `mesh/tsdf.py`**

Replace line 95:

```python
        # Filename matches Reconstructor.mesh()'s skip-check and the webapp's mesh lookup
        raw_path = self.output_dir / "mesh.ply"
```

- [ ] **Step 3: Update every reader**

```bash
grep -rl 'mesh_tsdf' collab_splats/ tests/ \
  | xargs sed -i 's/mesh_tsdf_clean\.ply/mesh_clean.ply/g; s/mesh_tsdf\.ply/mesh.ply/g'
```

Then fix the now-stale comment at `collab_splats/dashboard/app.py:649` — replace the two lines:

```python
            # TSDF writes mesh_tsdf.ply (see mesh/tsdf.py), not mesh.ply.
            mesh_path = out / "mesh" / "mesh_tsdf.ply"
```

with:

```python
            mesh_path = out / "mesh" / "mesh.ply"
```

and in `collab_splats/wrapper/splatter.py:480` confirm the candidate list now reads:

```python
            candidates = [mesh_dir / "mesh_clean.ply", mesh_dir / "mesh.ply"]
```

- [ ] **Step 4: Verify no references remain**

Run: `grep -rn 'mesh_tsdf' collab_splats/ tests/`
Expected: no output

- [ ] **Step 5: Run the affected suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/ tests/wrapper/ tests/dashboard/ -q`
Expected: all pass

- [ ] **Step 6: Run the dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`

- [ ] **Step 7: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats tests
git commit -m "fix(mesh): unify TSDF output on mesh/mesh.ply so skip-checks and readers agree"
```

- [ ] **Step 8: Update the tutorial notebooks that name the old file**

Three notebooks reference `mesh_tsdf.ply`: `docs/source/tutorials/03_splats/derive_splats.ipynb`, `docs/source/tutorials/06_mesh/create_mesh.ipynb`, `docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb`. Update the string in each (source cells only; leave `docs/_build/` alone — it is regenerated output):

```bash
for nb in docs/source/tutorials/03_splats/derive_splats.ipynb \
          docs/source/tutorials/06_mesh/create_mesh.ipynb \
          docs/source/tutorials/02_pointcloud/feedforward_mesh.ipynb; do
  sed -i 's/mesh_tsdf_clean\.ply/mesh_clean.ply/g; s/mesh_tsdf\.ply/mesh.ply/g' "$nb"
done
git add docs/source/tutorials
git commit -m "docs(tutorials): rename mesh_tsdf.ply references to mesh.ply"
```

---

### Task 6: Autoencoder stops on reconstruction cosine, records its quality

Cross-scene semantic comparison happens in *decoded* 768-D space, so what matters is reconstruction fidelity, not epoch count. `fit` already computes cosine per batch — turn it into a stop signal and persist the achieved value so a consumer can tell whether a scene's codes are trustworthy.

**Files:**
- Modify: `collab_splats/semantics/compression.py:131-236`
- Modify: `configs/base.yaml` (`semantics:` block)
- Test: `tests/semantics/test_compression_target.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/semantics/test_compression_target.py`:

```python
"""target_cosine early-stop and persisted reconstruction metrics."""

import torch

from collab_splats.semantics.compression import FeatureAutoencoder


def _features(n=512, d=32):
    torch.manual_seed(0)
    return torch.randn(n, d)


def test_fit_records_metrics_on_the_model():
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.fit(_features(), epochs=2)
    assert 0.0 <= ae.recon_cosine <= 1.0
    assert ae.recon_mse >= 0.0
    assert ae.epochs_run == 2


def test_target_cosine_stops_early():
    """A trivially-reachable target must stop before the epoch ceiling."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=32)
    ae.fit(_features(), epochs=50, target_cosine=-1.0)
    assert ae.epochs_run == 1


def test_target_cosine_unreachable_runs_to_ceiling():
    ae = FeatureAutoencoder(input_dim=32, latent_dim=2)
    ae.fit(_features(), epochs=3, target_cosine=1.5)
    assert ae.epochs_run == 3


def test_metrics_survive_save_load(tmp_path):
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.fit(_features(), epochs=2)
    ae.save(tmp_path)
    loaded = FeatureAutoencoder.load(tmp_path)
    assert loaded.recon_cosine == ae.recon_cosine
    assert loaded.recon_mse == ae.recon_mse
    assert loaded.epochs_run == ae.epochs_run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_compression_target.py -v`
Expected: FAIL — `AttributeError: 'FeatureAutoencoder' object has no attribute 'recon_cosine'`

- [ ] **Step 3: Track and persist the metrics**

In `collab_splats/semantics/compression.py`, at the end of `__init__` (after the existing attribute assignments, before the method definitions at line ~100), add:

```python
        # Reconstruction quality from the last fit(); 0.0 until trained. Persisted in the
        # checkpoint so a consumer can judge whether decoded 768-D codes are trustworthy.
        self.recon_cosine: float = 0.0
        self.recon_mse: float = 0.0
        self.epochs_run: int = 0
```

Change the `fit` signature (line 131) to add `target_cosine`:

```python
    def fit(
        self,
        features: Tensor,
        reg_target: Optional[Tensor] = None,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        on_epoch: Optional[Callable[[int, int, float], None]] = None,
        target_cosine: Optional[float] = None,
    ) -> None:
```

Extend the docstring Args with:

```
            target_cosine: stop once mean reconstruction cosine reaches this value, using
                ``epochs`` as a ceiling. None (default) always runs the full ``epochs``.
                Decoded 768-D features are what cross-scene comparison uses, so
                reconstruction fidelity is the right stop signal, not a fixed epoch count.
```

Inside the mini-batch loop, accumulate the two components separately. Replace the block from `# Main reconstruction: MSE + cosine loss` through `n_batches += 1` with:

```python
                # Main reconstruction: MSE + cosine loss (tracked separately for metrics)
                mse = F.mse_loss(recon, x)
                cos = F.cosine_similarity(recon, x).mean()
                loss = mse + (1 - cos)

                # Regularization head loss (weighted cosine)
                if reg_target is not None:
                    reg_batch = reg_target[idx].to(features.device)
                    loss = loss + self._reg_weight * (1 - F.cosine_similarity(self.reg_head(h), reg_batch).mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cos += cos.item()
                epoch_mse += mse.item()
                n_batches += 1
```

Initialise the two new accumulators next to `epoch_loss = 0.0` (line ~169):

```python
            epoch_loss = 0.0
            epoch_cos = 0.0
            epoch_mse = 0.0
            n_batches = 0
```

Replace the epoch-tail block (from `avg_loss = epoch_loss / max(n_batches, 1)` through the `lr_scheduler.step()` lines, ~196-204) with:

```python
            denom = max(n_batches, 1)
            avg_loss = epoch_loss / denom
            self.recon_cosine = epoch_cos / denom
            self.recon_mse = epoch_mse / denom
            self.epochs_run = epoch + 1

            pbar.set_postfix(loss=f"{avg_loss:.6f}", cos=f"{self.recon_cosine:.4f}")
            logger.debug(
                "epoch %d/%d  loss=%.6f  cos=%.4f", epoch + 1, epochs, avg_loss, self.recon_cosine
            )
            if on_epoch is not None:
                on_epoch(epoch + 1, epochs, avg_loss)

            # Advance LR schedule once per epoch if provided
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Early stop once reconstruction is good enough; epochs is the ceiling
            if target_cosine is not None and self.recon_cosine >= target_cosine:
                logger.info(
                    "target cosine %.4f reached at epoch %d/%d (cos=%.4f) — stopping",
                    target_cosine,
                    epoch + 1,
                    epochs,
                    self.recon_cosine,
                )
                break
```

In `save`, add the three metrics to the payload (line ~225):

```python
        payload = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "regularization_kwargs": self.regularization_kwargs,
            "state_dict": self.state_dict(),
            # Fit quality — lets a consumer decide whether to trust decoded 768-D codes
            "recon_cosine": self.recon_cosine,
            "recon_mse": self.recon_mse,
            "epochs_run": self.epochs_run,
        }
```

In `load`, restore them after `ae.load_state_dict(...)` (line ~251):

```python
        ae.load_state_dict(payload["state_dict"])
        # Older checkpoints predate the metrics — default to 0.0 rather than failing
        ae.recon_cosine = payload.get("recon_cosine", 0.0)
        ae.recon_mse = payload.get("recon_mse", 0.0)
        ae.epochs_run = payload.get("epochs_run", 0)
        ae.eval()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/ -q`
Expected: all pass

- [ ] **Step 5: Add the config knobs**

In `configs/base.yaml`, replace the `semantics:` block with:

```yaml
semantics:
  enabled: true
  extractor: talk2dino        # talk2dino | dinov2 | maskclip
  n_components: 64            # autoencoder latent dim; null = no compression
  # Stop training once mean reconstruction cosine hits this, with max_epochs as the
  # ceiling. Cross-scene comparison happens in decoded 768-D space, so fidelity is the
  # right stop signal — more epochs cannot align two scenes' 64-D bases.
  target_cosine: 0.95
  max_epochs: 100
```

- [ ] **Step 6: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/semantics/compression.py configs/base.yaml tests/semantics/test_compression_target.py
git commit -m "feat(semantics): cosine-target early stop + persisted reconstruction metrics"
```

---

### Task 7: One semantics artifact set; retire `lifted_normed.npy` and the JPG path

Two writers disagree today. `Reconstructor._lift_and_save` writes 64-D `features.zarr` and calls `ae.save(output_dir / "compressor.pt")` — and since `FeatureAutoencoder.save` appends `autoencoder.pt` to its argument, that produces a *directory* named `compressor.pt` containing `autoencoder.pt`. The dashboard writes 768-D `lifted_normed.npy` plus `semantics/autoencoder.pt`. Converge on `semantics/features.zarr` (64-D) + `semantics/autoencoder.pt`.

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:247-297` (`_lift_and_save`)
- Modify: `collab_splats/dashboard/pipeline.py:59-70` (delete `_write_frames_jpegs`), `:86-90` (`_extract_semantics`), `:93-142` (`_lift_and_compress`), `:145-157` (`_transfer_mesh_features`), plus the `_write_frames_jpegs` / `_extract_semantics` call sites
- Modify: `collab_splats/dashboard/{viewer,app,localize,sources}.py` (`lifted_normed` readers)
- Test: `tests/semantics/test_artifact_layout.py` (create); update `tests/dashboard/test_pipeline.py`, `test_app.py`, `test_viewer.py`, `test_viewer_lift.py`

Measured `lifted_normed` reference counts (2026-07-29) — 65 total, so budget accordingly:
`viewer.py` 17, `app.py` 7, `pipeline.py` 2, `localize.py` 1, `sources.py` 1; `test_viewer.py` 14,
`test_viewer_lift.py` 12, `test_app.py` 10, `test_pipeline.py` 1. `sources.py` was missing from an
earlier draft of this list. Only one real `compressor.pt` site exists (`reconstructor.py:294`), so
Step 4's sweep is nearly a no-op — do not expect it to find much.

- [ ] **Step 1: Write the failing test**

Create `tests/semantics/test_artifact_layout.py`:

```python
"""Both lifting paths must land on semantics/features.zarr + semantics/autoencoder.pt."""

import numpy as np
import torch
import zarr

from collab_splats.semantics.compression import FeatureAutoencoder


def test_load_features_and_decode_round_trip(tmp_path):
    """features.zarr holds latent codes; autoencoder.pt decodes them back to input_dim."""
    torch.manual_seed(0)
    latent = torch.randn(64, 8)

    store = zarr.open(str(tmp_path / "features.zarr"), mode="w")
    store["features"] = latent.numpy()

    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.save(tmp_path)

    assert (tmp_path / "autoencoder.pt").is_file()

    codes = np.asarray(zarr.open(str(tmp_path / "features.zarr"), mode="r")["features"])
    assert codes.shape == (64, 8)

    decoded = FeatureAutoencoder.load(tmp_path).per_point_decode(torch.from_numpy(codes))
    assert decoded.shape == (64, 32)
```

- [ ] **Step 2: Run the test — it should PASS, not fail**

This one is a characterization test, not a red-green test: it pins the target layout that the
later steps must satisfy, using only `FeatureAutoencoder` (already correct after Task 6).

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/test_artifact_layout.py -v`
Expected: PASS.

If it fails, `FeatureAutoencoder.save/load` regressed in Task 6 — fix that before continuing.

An earlier draft of Step 1 also asserted `not (tmp_path / "compressor.pt").exists()`. That was
vacuous — nothing in a test that only calls `FeatureAutoencoder` could ever have created it — so it
has been dropped. The assertion that actually earns its keep belongs in Step 6's test, which drives
`_lift_and_save` and can therefore catch the old `ae.save(output_dir / "compressor.pt")` behaviour
for real. Put it there, not here.

- [ ] **Step 3: Fix the Reconstructor's checkpoint path and pass the config knobs**

In `collab_splats/wrapper/reconstructor.py::_lift_and_save`, change the signature to accept the new knobs and fix the `ae.save` argument. Replace lines 247-297 with:

```python
def _lift_and_save(
    zarr_path: Path,
    feedforward_zarr: Path,
    output_dir: Path,
    n_components: int | None,
    target_cosine: float | None = None,
    max_epochs: int = 10,
) -> Path:
    """Load 2D feature cache + FeedforwardResult, lift to 3D, compress, save.

    Writes output_dir/features.zarr (latent codes) and, when compressing,
    output_dir/autoencoder.pt (weights + fit metrics) — the pair a consumer needs
    to recover full-dimensionality features.
    """
    # Heavy optional stack — see the import note below before touching these two
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.pointcloud.utils import lift_features

    # Validate feedforward zarr exists before attempting load
    if not feedforward_zarr.exists():
        raise FileNotFoundError(
            f"feedforward.zarr not found at {feedforward_zarr}. "
            "Run build_pointcloud() with a feedforward backend first."
        )

    # Load feature maps from zarr cache: (N, D, H_p, W_p)
    store = zarr.open(str(zarr_path), mode="r")
    features_arr = store["features"]
    feature_maps = [torch.from_numpy(np.array(features_arr[i])) for i in range(features_arr.shape[0])]

    # Load FeedforwardResult with depth/pixel data for lifting
    ff_result = FeedforwardResult.load_zarr(feedforward_zarr)

    # Lift 2D features to 3D: (P, D)
    lifted = lift_features(feature_maps, ff_result)

    # Optional autoencoder compression → latent codes persisted alongside the weights
    if n_components is not None:
        # Move lifted to GPU for autoencoder training; lift_features returns CPU tensor
        if torch.cuda.is_available():
            lifted = lifted.cuda()
        ae = FeatureAutoencoder(input_dim=lifted.shape[-1], latent_dim=n_components)
        ae.fit(lifted, epochs=max_epochs, target_cosine=target_cosine)
        lifted = ae.per_point_encode(lifted)
        # save() appends autoencoder.pt to the dir it is given — pass the dir, not a filename
        ae.save(output_dir)

    # Save lifted features as zarr
    output_dir.mkdir(parents=True, exist_ok=True)
    out_store = zarr.open(str(output_dir / "features.zarr"), mode="w")
    out_store["features"] = lifted.detach().cpu().numpy()
    return output_dir
```

**Imports.** The current body puts all six imports inside the function, against the project's
imports-at-top rule. Sort them into the two cases the rule actually allows:

- **Hoist to module top:** `torch`, `zarr` (the snippet above assumes the module-level name is
  `zarr`, so drop the `as zarr_lib` alias and use `zarr.open`), and `FeatureAutoencoder`. All three
  are hard dependencies. `numpy as np` is *already* at `reconstructor.py:15` — do not re-add it.
- **Keep inline (the rule's stated exception):** `FeedforwardResult` and `lift_features`.
  `collab_splats.pointcloud.utils` pulls the optional feedforward stack (vggt / mapanything);
  `dashboard/viewer.py:41` keeps them inline for exactly this reason and says so.

Before hoisting, **verify** that importing `collab_splats.wrapper.reconstructor` still works in an
env without the `feedforward` extra — that is the property the inline imports were protecting. If
hoisting `torch`/`zarr`/`FeatureAutoencoder` breaks any import or introduces a cycle, stop and
report rather than working around it.

Find the `_lift_and_save(` call site in `Reconstructor.semantics` (`grep -n '_lift_and_save' collab_splats/wrapper/reconstructor.py`) and add the two knobs:

```python
            target_cosine=sem_cfg["target_cosine"],
            max_epochs=sem_cfg["max_epochs"],
```

Note the naming: `max_epochs` is the config key and `_lift_and_save`'s parameter name; `fit`'s
parameter is `epochs`. The translation happens once, inside `_lift_and_save`
(`ae.fit(lifted, epochs=max_epochs, ...)` above). Do not rename `fit`'s parameter — that would
churn its other call sites for no gain.

- [ ] **Step 4: Sweep any remaining `compressor.pt` readers**

The FastAPI `collab_splats/webapp/` prototype used to hold two of these; it was deleted on
2026-07-29 (dead code, superseded by `collab_splats/dashboard/`). Sweep anyway in case another
reader exists:

```bash
grep -rln --include='*.py' 'compressor\.pt' collab_splats/ \
    | xargs -r sed -i 's/compressor\.pt/autoencoder.pt/g'
grep -rn --include='*.py' 'compressor.pt' collab_splats/
```
Expected from the grep: no output

`--include='*.py'` is load-bearing: without it `grep -rln` also matches compiled
`__pycache__/*.pyc`, `sed -i` then rewrites those binaries, and the verification grep keeps
printing matches from them — which reads as "the sweep didn't work". As of 2026-07-29 there is
exactly **one** real site: `collab_splats/wrapper/reconstructor.py:294`. If the sweep reports more
than one `.py` file, stop and re-read — something changed since the plan was written.

⚠️ **That one site needs a rewrite, not a rename.** It reads
`ae.save(output_dir / "compressor.pt")`, but `FeatureAutoencoder.save` (`compression.py:265-272`)
does `path.mkdir(parents=True, exist_ok=True)` and then writes `path/autoencoder.pt`. Its argument
is a **directory**, so today that line creates a directory literally named `compressor.pt/`
containing `autoencoder.pt` — a pre-existing bug. Running the sed over it yields
`autoencoder.pt/autoencoder.pt`, which is still wrong and now also collides with the name the
loader wants. Fix it by hand instead:

```python
        ae.save(output_dir)
```

`output_dir` is already the per-extractor lifted dir (`backend_dir / "semantics" / extractor_name`)
that the same function writes `features.zarr` into, so this lands the artifact pair side by side —
exactly the layout `load_point_features` and `FeatureAutoencoder.load` expect. Then re-run the
verification grep. Any on-disk scene carrying the old `compressor.pt/` directory has to re-run
`extract_semantics(overwrite=True)`; note it alongside the Task 5 mesh re-run in the docs.

- [ ] **Step 5: Converge the dashboard on `features.zarr`**

In `collab_splats/dashboard/pipeline.py::_lift_and_compress`, replace the caching tail (lines 139-142) with:

```python
    op_log.update_progress(94, "semantics: caching lifted features")
    # Persist LATENT codes + weights (not decoded 768-D): same artifact pair the
    # Reconstructor path writes, ~12x smaller, and decodable on read.
    out = zarr.open(str(Path(semantics_dir) / "features.zarr"), mode="w")
    out["features"] = compressed_pts.detach().cpu().numpy()
    ae.save(Path(semantics_dir))
    op_log.append_line(f"semantics: lift + encode + cache in {time.perf_counter() - t:.1f}s")
```

The `decoded` / `normed` locals become unused — delete lines 136-138 (`with torch.no_grad(): decoded = ...; normed = ...`).

Add a shared reader next to `_load_feature_maps` (after line 102):

```python
def load_point_features(semantics_dir: Path, *, decode: bool = True) -> np.ndarray:
    """Read semantics/features.zarr; decode latent codes back to full dim by default.

    Consumers that compare features across scenes must decode — the 64-D bases of two
    independently-trained autoencoders are not aligned, the decoded space is.
    """
    codes = np.asarray(zarr.open(str(Path(semantics_dir) / "features.zarr"), mode="r")["features"])
    if not decode:
        return codes
    ae = FeatureAutoencoder.load(Path(semantics_dir))
    with torch.no_grad():
        decoded = ae.per_point_decode(torch.from_numpy(codes))
        return torch.nn.functional.normalize(decoded, dim=1).cpu().numpy()
```

- [ ] **Step 5b: Fix the wildcard-glob collision this step creates (MUST NOT SKIP)**

⚠️ Writing `features.zarr` into `semantics_dir` breaks the existing 2D-cache reader. The dashboard
layout is **flat** — `semantics_dir` holds *both* the extractor's 2D patch cache
(`{extractor_name}.zarr`, written by `extract_and_cache*` as `cache_dir / f"{self.name}.zarr"`) and,
after Step 5, the lifted per-point `features.zarr`. Two helpers pick the 2D cache with a wildcard:

- `collab_splats/dashboard/pipeline.py:100`
- `collab_splats/dashboard/viewer.py:37`

both `store_path = next(Path(semantics_dir).glob("*.zarr"))`. `Path.glob` yields in `os.scandir`
order, not sorted, so post-Step-5 this returns `features.zarr` or `talk2dino.zarr`
**nondeterministically**. Picking `features.zarr` feeds `(P, latent)` per-point codes into
`lift_features` where it expects `(N, D, H_p, W_p)` per-frame maps — the exact
same-dtype-wrong-meaning failure flagged in Step 8. Fix both call sites:

```python
    # semantics_dir also holds the lifted per-point features.zarr; the 2D cache is the
    # extractor-named store. Wildcard-glob without this filter picks either one at random.
    store_path = next(p for p in Path(semantics_dir).glob("*.zarr") if p.name != "features.zarr")
```

Regression test (`tests/dashboard/test_semantics_store_selection.py`, new). It must FAIL before the
filter and PASS after — assert on the *name*, and seed the dir so the wrong answer is reachable:

```python
def test_load_feature_maps_ignores_lifted_features_store(tmp_path):
    """features.zarr must never be mistaken for the extractor's 2D patch cache."""
    # Both stores in one flat dir, as the dashboard writes them
    cache = zarr.open(str(tmp_path / "talk2dino.zarr"), mode="w")
    cache["features"] = np.zeros((2, 4, 3, 3), dtype=np.float32)
    lifted = zarr.open(str(tmp_path / "features.zarr"), mode="w")
    lifted["features"] = np.zeros((7, 4), dtype=np.float32)

    maps = pipeline._load_feature_maps(tmp_path)
    # 2 frames of (D=4, H_p=3, W_p=3) — not 7 points of (4,)
    assert len(maps) == 2
    assert tuple(maps[0].shape) == (4, 3, 3)
```

Add the identical test against `viewer._load_feature_maps` — the two helpers are byte-identical
copies, so a fix applied to one and not the other is the likely failure mode.

Do **not** merge the two copies into one shared helper in this task, tempting as it is:
`tests/dashboard/test_viewer_lift.py:16,60` patch `collab_splats.dashboard.viewer._load_feature_maps`
by name, and Task 7 is already large. Note it for Task 12 instead.

- [ ] **Step 6: Update every `lifted_normed` reader**

`grep -rn --include='*.py' 'lifted_normed' collab_splats/` lists the sites. Measured 2026-07-29 —
**28 references across 5 files**, not 4: `viewer.py` **17**, `app.py` **7**, `pipeline.py` 2,
`localize.py` 1, `sources.py` 1. `viewer.py` carries most of the load, so do NOT treat this as a
one-line-per-file swap; read each of its 17 sites.

Replace each `np.load(<dir>/"semantics"/"lifted_normed.npy")` with
`load_point_features(<dir> / "semantics")`, importing it from `collab_splats.dashboard.pipeline`.
Sites that are *filename strings* rather than loads (`sources.py`'s `PULL_EXCLUDES`-style lists,
existence checks, log messages) need the name changed to `features.zarr` — and note
`features.zarr` is a **directory**, so `.exists()` still works but any `.is_file()` or file-size
check does not. In `_transfer_mesh_features` (`pipeline.py:145-157`) that is:

```python
    mesh_path = Path(out_dir) / "mesh" / "mesh.ply"
    features_zarr = Path(out_dir) / "semantics" / "features.zarr"
    if not mesh_path.exists() or not features_zarr.exists():
        logger.warning(
            "mesh feature transfer skipped: mesh=%s features=%s", mesh_path.exists(), features_zarr.exists()
        )
        return
    # Transfer the LATENT codes — vertex_features.npy is latent-dim, decode on read
    point_features = load_point_features(Path(out_dir) / "semantics", decode=False)
    persist_mesh_vertex_features(mesh_path, result.points, point_features, k=k, sdf_trunc=sdf_trunc)
```

Verify: `grep -rn 'lifted_normed' collab_splats/` → no output.

- [ ] **Step 7: Delete the JPG export path**

Delete `_write_frames_jpegs` (`collab_splats/dashboard/pipeline.py:59-70`) and its call site, and make `_extract_semantics` read the canonical store instead of a JPG dir. Replace `_extract_semantics` (lines 86-90) with:

```python
def _extract_semantics(extractor_name: str, frames_zarr: Path, out_dir: Path) -> Path:
    """Extract + cache patch features straight from frames.zarr — no JPG export."""
    extractor = BaseFeatureExtractor.get(extractor_name)()
    return extractor.extract_and_cache_from_zarr(frames_zarr, out_dir)
```

Find its caller (`grep -n '_extract_semantics\|_write_frames_jpegs' collab_splats/dashboard/pipeline.py`) and pass the `frames.zarr` path in place of the JPG dir; delete the `_write_frames_jpegs(...)` call and any now-unused `frames_dir` local. Remove the `from PIL import Image` import if nothing else uses it (`grep -n 'Image' collab_splats/dashboard/pipeline.py`).

- [ ] **Step 8: Update the tests that assert the old artifacts**

```bash
grep -rn 'lifted_normed\|_write_frames_jpegs' tests/
```

For each hit, replace `lifted_normed.npy` fixtures with a `features.zarr` + `autoencoder.pt` pair. Pattern to use in `tests/dashboard/test_viewer.py`, `test_viewer_lift.py`, `test_app.py`, `test_pipeline.py`:

```python
def _write_semantics(semantics_dir, n_points=32, latent=8, input_dim=32):
    """Write the semantics artifact pair a scene dir is expected to carry."""
    semantics_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(semantics_dir / "features.zarr"), mode="w")
    store["features"] = np.zeros((n_points, latent), dtype=np.float32)
    FeatureAutoencoder(input_dim=input_dim, latent_dim=latent).save(semantics_dir)
```

`numpy`, `zarr` and `FeatureAutoencoder` go at the top of each test file, not inside the helper.

Delete assertions on `_write_frames_jpegs` outright — the function is gone.

⚠️ **This step is where the real risk lives.** The artifact does not just change filename — it
changes *content*: `lifted_normed.npy` held L2-normalized **decoded** full-dim features, while
`features.zarr` holds **raw latent codes**. Every one of the 65 call sites has to be classified as
"needs decoding" (comparison, querying, anything semantic) or "latent is fine" (storage, transfer,
shape-only). Getting one wrong yields plausible-looking garbage rather than an error, because both
are float arrays of the same length. Go site by site; do not blanket-replace.

- [ ] **Step 9: Run the suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/semantics/ tests/dashboard/ tests/wrapper/ tests/mesh/ -q`
Expected: all pass

- [ ] **Step 10: Run the dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`

- [ ] **Step 11: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats tests
git commit -m "refactor(semantics): converge on features.zarr + autoencoder.pt, drop lifted_normed and the JPG path"
```

---

### Task 8: `SceneSource` in `collab_splats/remote/`

Retarget the rclone wrapper at `environments-curated` / `environments-processed`, drop the `reconstruction/` path prefix and the `(session, stem)` pair in favour of one flat `scene` id, add `verify_push`, and delete the multi-camera field-session methods (that layout does not exist in the new buckets).

**Files:**
- Create: `collab_splats/remote/__init__.py`, `collab_splats/remote/sources.py`
- Create: `tests/remote/__init__.py`, `tests/remote/test_sources.py`
- Delete: `collab_splats/dashboard/sources.py`, `tests/dashboard/test_sources.py`, `tests/dashboard/test_sources_field.py`

- [ ] **Step 1: Move the module with history, then retarget it**

```bash
mkdir -p collab_splats/remote tests/remote
git mv collab_splats/dashboard/sources.py collab_splats/remote/sources.py
git mv tests/dashboard/test_sources.py tests/remote/test_sources.py
git rm tests/dashboard/test_sources_field.py
touch tests/remote/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Replace the whole of `tests/remote/test_sources.py` with:

```python
"""SceneSource: flat scene ids, new buckets, verified push, caching."""

import pytest

import collab_splats.remote.sources as sources
from collab_splats.remote.sources import (
    CURATED_BUCKET,
    PROCESSED_BUCKET,
    PULL_EXCLUDES,
    PUSH_EXCLUDES,
    SceneSource,
    parse_rclone_percent,
)


class _FakeClient:
    """Returns canned rclone listings keyed by (bucket, path)."""

    remote_name = "collab-data"

    def __init__(self, listings=None):
        self._listings = listings or {}
        self.cmds = []

    def list_directory(self, bucket, path):
        key = (bucket, path)
        if key not in self._listings:
            raise RuntimeError(f"no such path: {key}")
        return self._listings[key]

    def _cmd(self, *args):
        self.cmds.append(list(args))
        return ["true"]


def _dirs(*names):
    return [{"Name": n, "IsDir": True} for n in names]


def _files(*names):
    return [{"Name": n, "IsDir": False} for n in names]


def _fake_popen(monkeypatch, lines=(), returncode=0, capture=None):
    """Replace subprocess.Popen so streaming helpers run without rclone."""

    class _P:
        def __init__(self, cmd, *a, **k):
            if capture is not None:
                capture.append(cmd)
            self.stdout = iter(lines)
            self.returncode = returncode

        def wait(self):
            return self.returncode

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(sources.subprocess, "Popen", _P)


########
# Buckets + listing
########


def test_buckets_are_the_environments_pair():
    assert CURATED_BUCKET == "environments-curated"
    assert PROCESSED_BUCKET == "environments-processed"


def test_list_scenes_returns_dirs_at_bucket_root():
    client = _FakeClient(
        {(CURATED_BUCKET, ""): _dirs("2026_07_20-birds-C0043", "2026_07_21-rats-C0100") + _files("notes.txt")}
    )
    assert SceneSource(client).list_scenes() == ["2026_07_20-birds-C0043", "2026_07_21-rats-C0100"]


def test_scene_video_picks_the_single_video():
    client = _FakeClient({(CURATED_BUCKET, "2026_07_20-birds-C0043"): _files("C0043.MP4", "readme.md")})
    assert SceneSource(client).scene_video("2026_07_20-birds-C0043") == "C0043.MP4"


def test_scene_video_raises_when_no_video():
    client = _FakeClient({(CURATED_BUCKET, "s"): _files("readme.md")})
    with pytest.raises(FileNotFoundError, match="no video"):
        SceneSource(client).scene_video("s")


def test_list_processed_scenes_reads_processed_bucket():
    client = _FakeClient({(PROCESSED_BUCKET, ""): _dirs("2026_07_20-birds-C0043")})
    assert SceneSource(client).list_processed_scenes() == ["2026_07_20-birds-C0043"]


def test_has_processed_true_when_listing_nonempty():
    client = _FakeClient({(PROCESSED_BUCKET, "s"): _files("transforms.json")})
    assert SceneSource(client).has_processed("s") is True


def test_has_processed_false_on_error():
    assert SceneSource(_FakeClient()).has_processed("nope") is False


########
# Transfers — one scene id, no reconstruction/ prefix
########


def test_fetch_video_copies_to_dest(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient({(CURATED_BUCKET, "s"): _files("C0043.MP4")})
    local = SceneSource(client).fetch_video("s", dest_dir=tmp_path)
    assert local == tmp_path / "C0043.MP4"
    assert client.cmds[0][:3] == ["copyto", f"collab-data:{CURATED_BUCKET}/s/C0043.MP4", str(local)]


def test_push_outputs_targets_scene_dir_with_excludes(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).push_outputs(tmp_path, "s")
    cmd = client.cmds[0]
    assert cmd[0] == "copy"
    assert cmd[2] == f"collab-data:{PROCESSED_BUCKET}/s"
    for pattern in PUSH_EXCLUDES:
        assert "--exclude" in cmd and pattern in cmd


def test_push_excludes_features_and_frames():
    assert "features/**" in PUSH_EXCLUDES
    assert "frames.zarr/**" in PUSH_EXCLUDES


def test_pull_processed_passes_exclude_flags(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).pull_processed("s", tmp_path, excludes=("frames.zarr/**",))
    cmd = client.cmds[0]
    assert cmd[1] == f"collab-data:{PROCESSED_BUCKET}/s"
    assert "--exclude" in cmd and "frames.zarr/**" in cmd


def test_pull_zarr_members_pulls_each_member(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).pull_zarr_members("s", tmp_path, members=("feedforward.zarr/depth",))
    assert any("feedforward.zarr/depth" in " ".join(c) for c in client.cmds)


########
# Verify
########


def test_verify_push_true_on_clean_check(monkeypatch, tmp_path):
    _fake_popen(monkeypatch, lines=["0 differences found\n"], returncode=0)
    client = _FakeClient()
    assert SceneSource(client).verify_push(tmp_path, "s") is True
    cmd = client.cmds[0]
    assert cmd[0] == "check"
    assert "--one-way" in cmd


def test_verify_push_false_on_nonzero_exit(monkeypatch, tmp_path):
    _fake_popen(monkeypatch, lines=["1 differences found\n"], returncode=1)
    assert SceneSource(_FakeClient()).verify_push(tmp_path, "s") is False


def test_verify_push_uses_the_same_excludes_as_push(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient()
    SceneSource(client).verify_push(tmp_path, "s")
    cmd = client.cmds[0]
    for pattern in PUSH_EXCLUDES:
        assert pattern in cmd


########
# Cache + misc
########


def test_listing_cache_hits_once():
    client = _FakeClient({(CURATED_BUCKET, ""): _dirs("s")})
    calls = []
    orig = client.list_directory
    client.list_directory = lambda b, p: (calls.append((b, p)), orig(b, p))[1]
    src = SceneSource(client)
    src.list_scenes()
    src.list_scenes()
    assert len(calls) == 1


def test_invalidate_drops_one_key():
    client = _FakeClient({(CURATED_BUCKET, ""): _dirs("s"), (PROCESSED_BUCKET, ""): _dirs("s")})
    calls = []
    orig = client.list_directory
    client.list_directory = lambda b, p: (calls.append((b, p)), orig(b, p))[1]
    src = SceneSource(client)
    src.list_scenes()
    src.list_processed_scenes()
    assert len(calls) == 2
    src.invalidate(("list_scenes",))
    src.list_processed_scenes()  # still cached
    assert len(calls) == 2
    src.list_scenes()  # dropped, refetches
    assert len(calls) == 3


def test_push_outputs_invalidates_processed_listings(monkeypatch, tmp_path):
    _fake_popen(monkeypatch)
    client = _FakeClient({(PROCESSED_BUCKET, "s"): _files("transforms.json")})
    src = SceneSource(client)
    src.has_processed("s")
    src.push_outputs(tmp_path, "s")
    assert ("has_processed", "s") not in src._listing_cache


def test_parse_rclone_percent_extracts_progress():
    assert parse_rclone_percent("Transferred: 1.2M / 3.4M, 42%, 1M/s") == 42
    assert parse_rclone_percent("no percent here") is None


def test_field_session_methods_are_gone():
    """The multi-camera fieldwork layout does not exist under environments-curated."""
    for gone in ("list_field_sessions", "list_rgb_cameras", "list_camera_videos", "fetch_field_video"):
        assert not hasattr(SceneSource, gone)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/remote/test_sources.py -q`
Expected: collection error — `ImportError: cannot import name 'CURATED_BUCKET'` / `'SceneSource'`

- [ ] **Step 4: Retarget `collab_splats/remote/sources.py`**

Apply these changes to the moved file:

1. Module docstring → `"""rclone-backed access to the environments-curated / environments-processed GCS buckets."""`
2. Constants block: replace `CURATED_BUCKET`, `PROCESSED_BUCKET`, `ROOT`, `_FIELD_SESSION_RE` with:

```python
CURATED_BUCKET = "environments-curated"
PROCESSED_BUCKET = "environments-processed"

_VIDEO_EXTS = (".mp4", ".mov")

# Curated dir names look like YYYY_MM_DD-PARENTFOLDER-VIDEONAME and ARE the scene id:
# flat, one video each, no reconstruction/ prefix.
_SCENE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-.+$")

# Dense per-frame arrays a viewer does not need — pulled on demand instead.
PULL_EXCLUDES = (
    "feedforward.zarr/depth/**",
    "feedforward.zarr/world_points/**",
    "feedforward.zarr/confidence/**",
    "feedforward.zarr/conf/**",
    "feedforward.zarr/features/**",
    "feedforward.zarr/pixel_indices/**",
    "feedforward.zarr/images/**",
)

# Never pushed: raw 2D feature maps (regenerable from frames + extractor) and the
# keyframe store (regenerable from the curated video, which stays in the bucket).
PUSH_EXCLUDES = (
    "features/**",
    "frames.zarr/**",
)
```

3. Rename the class `SessionSource` → `SceneSource`.
4. Replace `list_sessions` / `list_videos` with the flat pair:

```python
    def list_scenes(self) -> list[str]:
        """Curated scene ids — dir names directly under environments-curated."""

        def _produce():
            entries = self._require_client().list_directory(CURATED_BUCKET, "")
            return sorted(e["Name"] for e in entries if e.get("IsDir") and _SCENE_RE.match(e["Name"]))

        return self._cached(("list_scenes",), _produce)

    def scene_video(self, scene: str) -> str:
        """The single video filename inside a curated scene dir."""

        def _produce():
            entries = self._require_client().list_directory(CURATED_BUCKET, scene)
            return sorted(
                e["Name"] for e in entries if not e.get("IsDir") and e["Name"].lower().endswith(_VIDEO_EXTS)
            )

        videos = self._cached(("scene_video", scene), _produce)
        if not videos:
            raise FileNotFoundError(f"no video in {CURATED_BUCKET}/{scene}")
        if len(videos) > 1:
            logger.warning("%s holds %d videos; using %s", scene, len(videos), videos[0])
        return videos[0]
```

5. Replace `fetch_video(session, name, dest_dir, on_line)` with:

```python
    def fetch_video(self, scene: str, dest_dir: Path, on_line=None) -> Path:
        """Copy the scene's video to dest_dir; returns the local path."""
        name = self.scene_video(scene)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        cmd = self._require_client()._cmd(
            "copyto",
            f"{self._remote}:{CURATED_BUCKET}/{scene}/{name}",
            str(local),
            "--stats",
            "2s",
            "--stats-one-line",
        )
        self._run_streaming(cmd, "fetch", on_line)
        return local
```

6. Delete `list_field_sessions`, `list_rgb_cameras`, `list_camera_videos`, `fetch_field_video` entirely.
7. Collapse every remaining `(session, stem)` parameter pair to a single `scene`, and every remote path from `f"{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"` to `f"{PROCESSED_BUCKET}/{scene}"`. That covers `list_localization_dbs`, `has_processed`, `pull_processed`, `pull_zarr_members`, `push_outputs`. Rename `list_processed_stems(session)` → `list_processed_scenes()` reading `(PROCESSED_BUCKET, "")`. Cache keys become 2-tuples, e.g. `("has_processed", scene)`.
8. `push_outputs` gains the exclude flags and the new invalidation keys:

```python
    def push_outputs(self, local_dir: Path, scene: str, on_line=None) -> None:
        """Upload a processed scene dir to environments-processed/<scene>/."""
        args = [
            "copy",
            str(local_dir),
            f"{self._remote}:{PROCESSED_BUCKET}/{scene}",
            "--gcs-bucket-policy-only",
            "--transfers",
            "8",
            "--retries",
            "3",
            "--timeout",
            "300s",
            "--contimeout",
            "60s",
            "--stats",
            "2s",
            "--stats-one-line",
        ]
        # Regenerable artifacts stay local — see PUSH_EXCLUDES
        for pattern in PUSH_EXCLUDES:
            args += ["--exclude", pattern]
        self._run_streaming(self._require_client()._cmd(*args), "push", on_line)

        # Anything cached about this scene's processed state is now stale
        self.invalidate(("has_processed", scene))
        self.invalidate(("list_localization_dbs", scene))
        self.invalidate(("list_processed_scenes",))
```

9. Add `verify_push` right after it:

```python
    def verify_push(self, local_dir: Path, scene: str, on_line=None) -> bool:
        """True when every local file (minus PUSH_EXCLUDES) exists remotely and matches.

        --one-way so remote-only leftovers from an earlier run are not a failure. This is
        the gate the remote driver requires before deleting anything local.
        """
        args = [
            "check",
            str(local_dir),
            f"{self._remote}:{PROCESSED_BUCKET}/{scene}",
            "--one-way",
            "--gcs-bucket-policy-only",
        ]
        # Excluded files were never uploaded — checking them would always fail
        for pattern in PUSH_EXCLUDES:
            args += ["--exclude", pattern]
        try:
            self._run_streaming(self._require_client()._cmd(*args), "check", on_line)
        except RuntimeError as exc:
            logger.error("verify failed for %s: %s", scene, exc)
            return False
        return True
```

10. Add a `self._remote` property if the file used `self._client.remote_name` inline; keep whichever the moved code already does, but make it one accessor so the strings above resolve.

- [ ] **Step 5: Write the package init**

Create `collab_splats/remote/__init__.py`:

```python
"""Remote (GCS via rclone) access to curated inputs and processed outputs."""

from collab_splats.remote.sources import (
    CURATED_BUCKET,
    PROCESSED_BUCKET,
    PULL_EXCLUDES,
    PUSH_EXCLUDES,
    SceneSource,
    parse_rclone_percent,
)

__all__ = [
    "CURATED_BUCKET",
    "PROCESSED_BUCKET",
    "PULL_EXCLUDES",
    "PUSH_EXCLUDES",
    "SceneSource",
    "parse_rclone_percent",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/remote/test_sources.py -v`
Expected: all pass (dashboard tests will fail until Task 9 — that is expected here)

- [ ] **Step 7: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/remote tests/remote tests/dashboard
git commit -m "refactor(remote): SceneSource over environments-curated/processed with verified push"
```

⚠️ Do NOT re-add `collab_splats/dashboard/sources.py` here. Step 1's `git mv` already staged that
deletion, so naming a path that no longer exists makes `git add` fail with
`fatal: pathspec ... did not match any files`. Same reason `-A` is dropped: the convention block
forbids it, and the move is already staged.

---

### Task 9: Retarget the dashboard at `SceneSource`

Mechanical: one import path, one class name, `(session, stem)` → `scene`, `list_sessions` → `list_scenes`, `list_processed_stems` → `list_processed_scenes`, local layout `<base>/<session>/<stem>/` → `<base>/<scene>/`.

**Files:**
- Modify: `collab_splats/dashboard/app.py` (lines 26, 98, 106, 313, 372, 378, 457, 492, 540, 592, 630-634, 767, 784, 797)
- Modify: `collab_splats/dashboard/localize.py` (20, 134, 353, 359, 383, 403, 499, 517, 631, 661)
- Modify: `collab_splats/dashboard/pipeline.py` (19, 193, 200, 216, **349**, 474, 483, 503, 517)
- Modify: `collab_splats/dashboard/shell.py` (15, 26, 31)
- Modify: `collab_splats/dashboard/operation_log.py` (12) — imports `parse_rclone_percent` from the deleted module; the Step 2 sed fixes it, but it is the 5th importer, so include it when you verify
- Modify: `tests/dashboard/{test_app,test_localize_page,test_pipeline,test_run_localization,test_shell}.py`

The complete set of files importing the deleted module is exactly: `app.py`, `pipeline.py`, `localize.py`, `operation_log.py`, `shell.py`, `tests/dashboard/test_app.py` (`PULL_EXCLUDES` only), `tests/dashboard/test_shell.py`. Expect 8 collection errors before you start.

- [ ] **Step 1: Run the dashboard suite to see the failures**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q`
Expected: many `ImportError: cannot import name 'SessionSource' from 'collab_splats.dashboard.sources'`

- [ ] **Step 2: Swap imports and the class name**

⚠️ `--include='*.py'` is required. Without it `grep -rl` also matches the compiled
`__pycache__/*.pyc` blobs and `sed -i` rewrites them in place; a mangled `.pyc` whose
header still validates gets imported and fails in a way that looks nothing like this change.

```bash
grep -rl --include='*.py' 'dashboard.sources\|SessionSource' collab_splats/ tests/ \
  | xargs sed -i \
      -e 's/from collab_splats\.dashboard\.sources import/from collab_splats.remote import/' \
      -e 's/import collab_splats\.dashboard\.sources as sources/import collab_splats.remote.sources as sources/' \
      -e 's/SessionSource/SceneSource/g'
```

- [ ] **Step 3: Collapse `(session, stem)` to `scene`**

Work file by file. In `collab_splats/dashboard/app.py`, `localize.py`, `pipeline.py`:

- Every `SceneSource` call that passed `(session, stem)` now passes one `scene`. Sites: `app.py:378` (`list_processed_stems(session)` → `list_processed_scenes()`), `:457`, `:540`, `:592`, `:630-634`, `:784`; `localize.py:403`; `pipeline.py:200`, `:483`, `:517`.
- Every local scene dir built as `base_dir / session / stem` becomes `base_dir / scene`. Sites: `localize.py:661`, and the `out_dir` construction in `pipeline.py` and `app.py` (`grep -n 'base_dir /\|_base_dir /' collab_splats/dashboard/*.py`).
- The scene selector that previously drove two widgets (session dropdown + video dropdown) becomes one. In `app.py:313` use `list_scenes()`; delete the `list_videos(session)` fetch at `:372` and the dependent video-select widget, since a curated scene holds exactly one video — `fetch_video(scene, dest_dir)` needs no name.
- `app.py:492`: `fetch_video(session, name, local.parent, ...)` → `fetch_video(scene, local.parent, on_line=on_line)`.

**`localize.py` needs more than a rename — 5 dropdowns become 2.** The page has two independent multi-level selectors, and both collapse. This is the largest single piece of Task 9; do it as a deliberate rewrite of the selection layer, not a sed.

*Reconstruction side (2 → 1).* `self.scene_session` + `self.scene_video` (`:159-160`) become one `self.scene = pn.widgets.Select(name="Scene", options=[])` populated from `list_processed_scenes()`. Drop the `_on_scene_session` → `_on_scene_video` cascade; a single `_on_scene` watcher calls `list_localization_dbs(scene)` (was `(session, stem)` at `:403`) to populate `self.method`. At `:648-649`, `scene_key = (scene_session, stem)` → `scene_key = scene`, and `mesh_path = self._base_dir / scene_session / stem / "mesh" / "mesh.ply"` → `self._base_dir / scene / "mesh" / "mesh.ply"`.

*Query side (3 → 1).* `self.field_session` + `self.camera` + `self.query_video` (`:161-163`) become one `self.query_scene = pn.widgets.Select(name="Query scene", options=[])` populated from `list_scenes()` (curated). **A curated scene dir now holds exactly one video, so a query video IS just another curated scene** — that is the whole reason the field/camera levels disappear. Delete `_on_field_session` (`:494`) and `_on_camera` (`:508`) outright; rename `_on_query_video` → `_on_query_scene`, reading only `event.new` instead of `self.field_session.value, self.camera.value`.

Also update, or the page will not import or will silently misbehave:
- `_refresh_listings` (`:342`): one listing, not two. Drop the `list_field_sessions` try-block and the `self.field_session.options` setter; the remaining call becomes `list_processed_scenes()`.
- Widget layout rows at `:188-190` and `:211-213`: remove the two deleted widgets from both.
- `_on_run` guard at `:647`: `if not (scene_session and stem and fs and cam and name)` → `if not (scene and query_scene)`, and its message → `"select a scene and a query scene first"`.
- Module docstring `:1` ("localize an rgb_X field-camera frame") and the widget-group docstring `:158` ("Scene (reconstruction) + query (field camera)") both describe the deleted layout.

- [ ] **Step 3b: `_ensure_local_query_video` — signature AND fast path**

`:622` is `_ensure_local_query_video(self, field_session, camera, name)` and builds `local = self._base_dir / "queries" / field_session / camera / name`, using `local.exists()` as the fast path under `self._fetch_lock`.

`fetch_video(scene, dest_dir, on_line=None)` resolves the filename itself (Task 8 requirement 6), so the caller no longer receives a `name` to build the local path from. Recover it from `scene_video(scene)`, which is **memoized on `("scene_video", scene)`** — so after the first call the fast path costs nothing and the original `local.exists()` check survives intact:

```python
    def _ensure_local_query_video(self, scene: str) -> Path:
        # Lock: the on-select prefetch thread and a Load-video click can request the same
        # file concurrently — two racing rclone writers let a frame decode from a partial
        # file. The second caller blocks, then hits the exists() fast path.
        with self._fetch_lock:
            # scene_video is a memoized listing, so naming the file to probe is free.
            local = self._base_dir / "queries" / scene / self._source.scene_video(scene)
            if local.exists():
                return local
            on_line = self._op_log.rclone_progress("⬇ fetching query video")
            return self._source.fetch_video(scene, local.parent, on_line=on_line)
```

Do **not** reach for `sources._VIDEO_EXTS` — it is private (`sources.py:23`) and deliberately not in `collab_splats.remote.__all__`. Do not glob the directory either; `Path.glob` yields in `os.scandir` order, so a stray sibling file makes the pick nondeterministic. `scene_video` raising `FileNotFoundError` for an empty scene dir is correct here — the old code would have failed at fetch time anyway, just later and less clearly.

- [ ] **Step 3c: `pipeline.py:349` — the last `reconstruction/` string in the codebase**

```python
                video_ref=f"reconstruction/{session}/{stem}/{Path(video_path).name}",
```

This embeds **both** the deleted bucket prefix and the `(session, stem)` pair, and it is persisted — it flows into `run_config.yaml` and from there onto the extractor zarr group via `_stamp_db_provenance` (`:398`, `:411`). Replace with the flat scene:

```python
                video_ref=f"{scene}/{Path(video_path).name}",
```

Then confirm nothing anywhere still writes the old prefix:
`grep -rn --include='*.py' 'reconstruction/' collab_splats/` → expect zero hits outside comments.

Two tests assert the old string shape. Neither *fails* (both round-trip whatever string they are handed rather than asserting production output), so they are staleness, not breakage — update them so the fixtures stop documenting a layout that no longer exists:
- `tests/dashboard/test_config.py:25,31` — `video_ref="reconstruction/2026_05_07/clip_03.mp4"` → `"2026_05_07-birds-clip_03/clip_03.mp4"`. **Note this file is not in the Files list above and is not reached by Step 5's `tests/dashboard/` run passing** — it passes either way; fix it for coherence.
- `tests/localization/test_provenance.py:72,83` — builds its own field-shaped dict (`"2024_02_06-session_0001/rgb_1/cam.mp4"`) and asserts `provenance[0]["camera"] == "rgb_1"`. It tests the localizer's *opaque* provenance storage, so dropping `camera` in Step 3c does not break it. Leave the assertion working but retarget the fixture to a flat scene id and a `scene` key, so no test still models the deleted camera hierarchy.

- [ ] **Step 3d: the provenance dict loses two keys**

`:653-657` writes provenance into the localization DB:

```python
        provenance = {"video_ref": f"{fs}/{cam}/{name}", "session": fs, "camera": cam, "frame_idx": int(frame_idx)}
```

`session` and `camera` have no meaning in the flat layout. Replace with:

```python
        provenance = {"video_ref": query_scene, "scene": query_scene, "frame_idx": int(frame_idx)}
```

**This changes an on-disk metadata schema**, so grep for readers before assuming it is free: `grep -rn '"camera"\|\[.camera.\]\|provenance' collab_splats/ tests/`. If any consumer indexes `provenance["camera"]` or `["session"]`, fix that consumer in this task — a `KeyError` here surfaces only at localization run time, which no test in `tests/dashboard/` reaches.

After each file, run `/opt/venv/reconstruction/bin/python -c "import collab_splats.dashboard.app"` (and the same for `localize`, `pipeline`, `shell`) and fix any `NameError`/`TypeError` before moving on.

- [ ] **Step 4: Update the dashboard tests**

Apply the same collapse in `tests/dashboard/`:

```bash
sed -i 's/list_processed_stems/list_processed_scenes/g; s/list_sessions/list_scenes/g' tests/dashboard/*.py
```

Then fix the arity by hand at the sites `grep -n 'pull_processed\|push_outputs\|has_processed\|pull_zarr_members\|fetch_video' tests/dashboard/*.py` reports. Example — `tests/dashboard/test_pipeline.py:76`:

```python
    source.push_outputs.assert_called_with(out, "2026_05_07-birds-clip_03", on_line=ANY)
```

and `tests/dashboard/test_run_localization.py:27-32`:

```python
    def pull_processed(self, scene, dest, excludes=()):
        self.excludes = excludes
        return dest

    def push_outputs(self, out_dir, scene, on_line=None):
        self.pushed = (out_dir, scene)
```

Delete the field-session tests in `tests/dashboard/test_localize_page.py:24` (`source.list_field_sessions.return_value = []`) and any test asserting the camera widgets.

- [ ] **Step 5: Run the dashboard suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -q`
Expected: all pass

- [ ] **Step 6: Run the dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`

- [ ] **Step 7: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/dashboard tests/dashboard
git commit -m "refactor(dashboard): flat scene ids via SceneSource, drop field-session browsing"
```

---

### Task 10: Move the per-scene driver core into `collab_splats/wrapper/batch.py`

`docs/examples/run_pipeline.py` currently holds `collect_videos`, `scene_output_dir`, `build_scene_config`, `run_scene`, `run_all`. Both drivers need those, and a `docs/examples/` script is not importable — move them into the package.

**Files:**
- Create: `collab_splats/wrapper/batch.py`
- Create: `tests/wrapper/test_batch.py`
- Modify: `docs/examples/run_pipeline.py` (becomes argparse + `run_all`)

- [ ] **Step 1: Write the failing tests**

Create `tests/wrapper/test_batch.py`:

```python
"""batch.py: output-dir derivation, config build, failure isolation."""

from pathlib import Path

import pytest
import yaml

from collab_splats.wrapper import batch


def test_collect_videos_expands_directories(tmp_path):
    d = tmp_path / "clips"
    d.mkdir()
    (d / "a.mp4").touch()
    (d / "b.MOV").touch()
    (d / "notes.txt").touch()
    assert batch.collect_videos([d]) == [d / "a.mp4", d / "b.MOV"]


def test_collect_videos_passes_files_through(tmp_path):
    f = tmp_path / "x.mp4"
    f.touch()
    assert batch.collect_videos([f]) == [f]


def test_scene_output_dir_uses_date_parent(tmp_path):
    video = tmp_path / "2026-07-20" / "C0043.MP4"
    assert batch.scene_output_dir(video, tmp_path / "out") == tmp_path / "out" / "2026_07_20" / "C0043"


def test_scene_output_dir_falls_back_to_stem(tmp_path):
    assert batch.scene_output_dir(tmp_path / "C0043.MP4", tmp_path / "out") == tmp_path / "out" / "C0043"


def test_scene_output_dir_accepts_explicit_name(tmp_path):
    """Remote scenes name their own output dir — the curated dir name IS the scene id."""
    out = batch.scene_output_dir(tmp_path / "C0043.MP4", tmp_path / "out", name="2026_07_20-birds-C0043")
    assert out == tmp_path / "out" / "2026_07_20-birds-C0043"


def test_build_scene_config_sets_paths(tmp_path):
    cfg = batch.build_scene_config(tmp_path / "v.mp4", tmp_path / "out", override_config={"semantics": {"enabled": False}})
    assert cfg["input_path"] == str(tmp_path / "v.mp4")
    assert cfg["output_path"] == str(tmp_path / "out" / "v")
    assert cfg["semantics"] == {"enabled": False}


def test_run_scene_writes_run_config(tmp_path, monkeypatch):
    calls = []

    class _FakeReconstructor:
        def __init__(self, config, config_dir=None):
            self.config = dict(config)

        def run_pipeline(self, stages=None, overwrite=False):
            calls.append((stages, overwrite))

    monkeypatch.setattr(batch, "Reconstructor", _FakeReconstructor)
    out, _ = batch.run_scene(tmp_path / "v.mp4", tmp_path / "out", None, None, None, False)
    assert (out / "run_config.yaml").exists()
    assert yaml.safe_load((out / "run_config.yaml").read_text())["input_path"] == str(tmp_path / "v.mp4")
    assert calls == [(None, False)]


def test_run_all_isolates_one_failure(tmp_path, monkeypatch):
    def _run_scene(video, *a, **k):
        if "bad" in Path(video).name:
            raise RuntimeError("boom")
        return Path(tmp_path / "out"), None

    monkeypatch.setattr(batch, "run_scene", _run_scene)
    code = batch.run_all([tmp_path / "good.mp4", tmp_path / "bad.mp4"], tmp_path / "out", None, None, None, False)
    assert code == 1


def test_run_all_returns_zero_when_all_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "run_scene", lambda video, *a, **k: (Path(tmp_path / "out"), None))
    assert batch.run_all([tmp_path / "a.mp4"], tmp_path / "out", None, None, None, False) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_batch.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'collab_splats.wrapper.batch'`

- [ ] **Step 3: Create `collab_splats/wrapper/batch.py`**

```python
"""Per-scene pipeline driver core, shared by the local and remote CLI drivers.

The drivers own argument parsing and (for remote) transfer; everything about turning
one video into one output dir lives here so both stay thin and this stays testable.
"""

import logging
import re
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.wrapper.reconstructor import Reconstructor

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs"
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


########
# Input discovery
########


def collect_videos(paths) -> list[Path]:
    """Expand each path (a video file or a directory of videos) into a flat video list."""
    videos: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # Non-recursive: only videos directly inside the directory
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
            if not found:
                logger.warning("No videos (%s) in directory: %s", sorted(VIDEO_EXTS), p)
            videos.extend(found)
        else:
            videos.append(p)
    return videos


def scene_output_dir(video, output_root, name: str | None = None) -> Path:
    """Derive the output dir for one video.

    name wins when given — remote scenes are named by their curated dir, which already
    encodes date + parent + video. Otherwise: <output-root>/<session-date>/<stem>, the
    live outputs/ layout, falling back to <output-root>/<stem> with no date dir present.
    """
    output_root = Path(output_root)
    if name is not None:
        return output_root / name
    stem = Path(video).stem
    for parent in Path(video).parents:
        if _DATE_RE.fullmatch(parent.name):
            return output_root / parent.name.replace("-", "_") / stem
    return output_root / stem


########
# Single scene
########


def build_scene_config(video, output_root, override_config=None, name: str | None = None) -> dict:
    """Build a per-video override dict. Reconstructor merges base.yaml defaults itself."""
    # Only carry the shared --config overrides plus per-video paths; defaults come from base.yaml
    config = merge({}, override_config) if override_config else {}
    config["input_path"] = str(video)
    config["output_path"] = str(scene_output_dir(video, output_root, name=name))
    return config


def run_scene(
    video,
    output_root,
    config_dir,
    override_config,
    stages,
    overwrite,
    name: str | None = None,
):
    """Run the full pipeline for a single video. Returns (output_path, Reconstructor)."""
    config = build_scene_config(video, output_root, override_config, name=name)
    r = Reconstructor(config, config_dir=config_dir or DEFAULT_CONFIG_DIR)

    # Persist run_config.yaml for reproducibility before running any stage
    output_path = Path(r.config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    run_cfg = output_path / "run_config.yaml"
    if not run_cfg.exists() or overwrite:
        with open(run_cfg, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)

    r.run_pipeline(stages=stages, overwrite=overwrite)
    return output_path, r


########
# Batch
########


def run_all(videos, output_root, config_dir, override_config, stages, overwrite, keep_viewer=False) -> int:
    """Run every video; continue past failures. Returns the process exit code.

    When keep_viewer, blocks after the batch on the last successfully-reconstructed
    scene's viser Viewer (if pointcloud.viz.enabled produced one), so the final scene
    stays browsable.
    """
    results = []
    last_reconstructor = None
    for video in videos:
        video = Path(video)
        logger.info("=== Video: %s ===", video.name)
        try:
            out, r = run_scene(video, output_root, config_dir, override_config, stages, overwrite)
            results.append((video.name, "OK", str(out)))
            last_reconstructor = r
        except Exception as exc:  # isolate one video's failure from the batch
            logger.exception("Video failed: %s", video.name)
            results.append((video.name, "FAIL", str(exc)))

    # Summary
    logger.info("==== Summary ====")
    for name, status, info in results:
        logger.info("%s: %s (%s)", status, name, info)
    failed = [n for n, s, _ in results if s == "FAIL"]

    # Optionally keep the last scene's viser viewer alive for browser inspection
    if keep_viewer:
        viewer = getattr(last_reconstructor, "viewer", None)
        if viewer is not None:
            logger.info("--keep-viewer: viser server staying up — inspect the scene in a browser (Ctrl-C to exit).")
            viewer.serve_forever()
        else:
            logger.info(
                "--keep-viewer set but no viewer was created (pointcloud.viz.enabled is false, "
                "or no video succeeded); nothing to keep alive."
            )

    return 1 if failed else 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_batch.py -v`
Expected: 9 passed

- [ ] **Step 5: Slim `docs/examples/run_pipeline.py`**

Delete `collect_videos`, `scene_output_dir`, `build_scene_config`, `run_scene`, `run_all` from the file (lines 62-158) and the now-unused `re` / `merge` / `Reconstructor` imports. Replace the import block (lines 39-59) with:

```python
import argparse
import logging
import sys
from pathlib import Path

import yaml

from collab_splats.wrapper.batch import DEFAULT_CONFIG_DIR, collect_videos, run_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
```

`main()` stays exactly as it is — it already calls `collect_videos` and `run_all` with these signatures.

- [ ] **Step 6: Verify the driver still works end to end on args**

Run: `/opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py --help`
Expected: usage text, exit 0

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/examples/ tests/wrapper/ -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add collab_splats/wrapper/batch.py tests/wrapper/test_batch.py docs/examples/run_pipeline.py
git commit -m "refactor(wrapper): move per-scene driver core into batch.py"
```

---

### Task 11: The remote driver

Fetch → reconstruct → push → verify → delete. Deletion is gated on `verify_push`; the curated video stays in the bucket, so the worst case after a delete is re-fetch and reprocess.

**Files:**
- Create: `docs/examples/run_pipeline_remote.py`
- Create: `tests/examples/test_run_pipeline_remote.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/examples/test_run_pipeline_remote.py`:

```python
"""Remote driver: order of operations, verify gate, cleanup, failure isolation."""

import importlib.util
from pathlib import Path

import pytest

_DRIVER = Path(__file__).parent.parent.parent / "docs" / "examples" / "run_pipeline_remote.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_pipeline_remote", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeSource:
    def __init__(self, scenes=("s1",), verify=True):
        self._scenes = list(scenes)
        self._verify = verify
        self.calls = []

    def list_scenes(self):
        self.calls.append(("list_scenes",))
        return list(self._scenes)

    def fetch_video(self, scene, dest_dir, on_line=None):
        self.calls.append(("fetch", scene))
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        video = dest_dir / f"{scene}.mp4"
        video.write_bytes(b"video")
        return video

    def push_outputs(self, local_dir, scene, on_line=None):
        self.calls.append(("push", scene))

    def verify_push(self, local_dir, scene, on_line=None):
        self.calls.append(("verify", scene))
        return self._verify


@pytest.fixture()
def driver():
    return _load_driver()


def _patch_run_scene(driver, monkeypatch, fail_on=()):
    def _run_scene(video, output_root, config_dir, override_config, stages, overwrite, name=None):
        if name in fail_on:
            raise RuntimeError("reconstruction failed")
        out = Path(output_root) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "sparse_pc.ply").write_bytes(b"ply")
        return out, None

    monkeypatch.setattr(driver.batch, "run_scene", _run_scene)


def test_happy_path_order_is_fetch_run_push_verify_delete(driver, monkeypatch, tmp_path):
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)
    code = driver.run_remote(src, ["s1"], tmp_path, None, None, None, False, keep_local=False)
    assert code == 0
    assert [c[0] for c in src.calls] == ["fetch", "push", "verify"]
    assert not (tmp_path / "s1").exists()


def test_all_processes_every_listed_scene(driver, monkeypatch, tmp_path):
    src = _FakeSource(scenes=("s1", "s2"))
    _patch_run_scene(driver, monkeypatch)
    code = driver.run_remote(src, None, tmp_path, None, None, None, False, keep_local=False)
    assert code == 0
    assert [c for c in src.calls if c[0] == "push"] == [("push", "s1"), ("push", "s2")]


def test_failed_verify_keeps_local_data(driver, monkeypatch, tmp_path):
    src = _FakeSource(verify=False)
    _patch_run_scene(driver, monkeypatch)
    code = driver.run_remote(src, ["s1"], tmp_path, None, None, None, False, keep_local=False)
    assert code == 1
    assert (tmp_path / "s1" / "sparse_pc.ply").exists()


def test_keep_local_skips_deletion(driver, monkeypatch, tmp_path):
    src = _FakeSource()
    _patch_run_scene(driver, monkeypatch)
    code = driver.run_remote(src, ["s1"], tmp_path, None, None, None, False, keep_local=True)
    assert code == 0
    assert (tmp_path / "s1" / "sparse_pc.ply").exists()


def test_reconstruction_failure_does_not_push_or_delete(driver, monkeypatch, tmp_path):
    src = _FakeSource(scenes=("s1", "s2"))
    _patch_run_scene(driver, monkeypatch, fail_on=("s1",))
    code = driver.run_remote(src, None, tmp_path, None, None, None, False, keep_local=False)
    assert code == 1
    assert ("push", "s1") not in src.calls
    assert ("push", "s2") in src.calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/examples/test_run_pipeline_remote.py -q`
Expected: FAIL — `FileNotFoundError` on `run_pipeline_remote.py`

- [ ] **Step 3: Write the driver**

Create `docs/examples/run_pipeline_remote.py`:

```python
#!/usr/bin/env python3
"""Reconstruct scenes straight out of the environments-curated GCS bucket.

For each scene: pull its video, run the same pipeline as run_pipeline.py, push the
outputs to environments-processed/<scene>/, verify the push with `rclone check
--one-way`, then delete the local copy. The curated video stays in the bucket, so a
deleted scene is always re-fetchable.

Scene ids are the curated directory names: YYYY_MM_DD-PARENTFOLDER-VIDEONAME.

Usage:
    # Named scenes
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \\
        2026_07_20-birds-C0043 2026_07_21-rats-C0100

    # Everything in the bucket
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all

    # Keep the local copy for inspection (skips the delete, not the push)
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all --keep-local

Credentials come from the existing rclone remote (`collab-data`) — nothing is read
from the environment or passed on the command line. One scene's failure does not abort
the batch; the process exits non-zero if any scene failed.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

from collab_splats.remote import PUSH_EXCLUDES, SceneSource
from collab_splats.wrapper import batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_remote(
    source,
    scenes,
    output_root,
    config_dir,
    override_config,
    stages,
    overwrite,
    keep_local: bool,
) -> int:
    """Fetch → reconstruct → push → verify → delete for each scene. Returns exit code."""
    output_root = Path(output_root)
    scene_ids = list(scenes) if scenes else source.list_scenes()
    if not scene_ids:
        logger.error("No scenes to process")
        return 2

    results = []
    for scene in scene_ids:
        logger.info("=== Scene: %s ===", scene)
        scene_dir = output_root / scene
        try:
            # 1. Pull the video into the scene's own output dir so cleanup is one rmtree
            video = source.fetch_video(scene, scene_dir, on_line=logger.info)

            # 2. Same pipeline as the local driver; name= pins the output dir to the scene id
            out, _ = batch.run_scene(
                video, output_root, config_dir, override_config, stages, overwrite, name=scene
            )

            # 3. Push, excluding regenerable artifacts (see PUSH_EXCLUDES)
            source.push_outputs(out, scene, on_line=logger.info)

            # 4. Verify before touching anything local — a failed check leaves the data put
            if not source.verify_push(out, scene, on_line=logger.info):
                results.append((scene, "FAIL", "push verification failed; local data kept"))
                continue

            # 5. Reclaim the disk (video included — it lives in the curated bucket)
            if keep_local:
                logger.info("--keep-local: leaving %s in place", out)
            else:
                shutil.rmtree(scene_dir, ignore_errors=True)
                logger.info("removed local scene dir %s", scene_dir)

            results.append((scene, "OK", str(out)))
        except Exception as exc:  # isolate one scene's failure from the batch
            logger.exception("Scene failed: %s", scene)
            results.append((scene, "FAIL", str(exc)))

    logger.info("==== Summary ====")
    for scene, status, info in results:
        logger.info("%s: %s (%s)", status, scene, info)
    logger.info("push excluded: %s", ", ".join(PUSH_EXCLUDES))

    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


def main():
    """Parse args and process the requested curated scenes."""
    parser = argparse.ArgumentParser(
        description="Reconstruct scenes from environments-curated and push to environments-processed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("scenes", nargs="*", metavar="SCENE", help="Curated scene ids. Omit with --all.")
    parser.add_argument("--all", action="store_true", help="Process every scene in environments-curated.")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        dest="output_root",
        help="Working dir; each scene lands in <output-root>/<scene>/ and is removed after a verified push.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional shared override YAML merged over base.yaml."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=batch.DEFAULT_CONFIG_DIR,
        dest="config_dir",
        help=f"Directory holding base.yaml. Default: {batch.DEFAULT_CONFIG_DIR}",
    )
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help="Steps to run: preproc,pointcloud,semantics,mesh,localize. Default: config-enabled steps.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run steps even if outputs already exist.")
    parser.add_argument(
        "--keep-local",
        action="store_true",
        dest="keep_local",
        help="Skip the post-push delete and leave the scene dir on disk.",
    )
    args = parser.parse_args()

    if not args.scenes and not args.all:
        parser.error("give one or more SCENE ids, or --all")

    override_config = None
    if args.config:
        with open(args.config) as f:
            override_config = yaml.safe_load(f)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    code = run_remote(
        source=SceneSource(),
        scenes=args.scenes or None,
        output_root=args.output_root,
        config_dir=args.config_dir,
        override_config=override_config,
        stages=stages,
        overwrite=args.overwrite,
        keep_local=args.keep_local,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/examples/test_run_pipeline_remote.py -v`
Expected: 5 passed

- [ ] **Step 5: Check the CLI surface**

Run: `/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py --help`
Expected: usage text, exit 0

Run: `/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py --output-root /tmp/x`
Expected: exit 2 with `error: give one or more SCENE ids, or --all`

- [ ] **Step 6: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add docs/examples/run_pipeline_remote.py tests/examples/test_run_pipeline_remote.py
git commit -m "feat(examples): remote pipeline driver with verified push and local cleanup"
```

---

### Task 12: Documentation, full suite, final gates

**Carried-forward items found during Tasks 1-5 — resolve here:**

1. **nerfstudio cannot load a published scene, by decision.** Both stock dataparser routes need
   real image files: `nerfstudio_dataparser.py:127,134` do `Path(frame["file_path"])`
   unconditionally, and `colmap_dataparser.py:93,176` resolve `data/images/{im_data.name}`. Our
   `transforms.json` frames key on `frame_idx` against `frames.zarr`, and no `images/` dir exists
   (removed in the frame-store migration). The user chose **not** to export one — downstream gets
   `sparse_pc.ply` + mesh + features + raw COLMAP binaries, and reads poses via `pycolmap`.
   Say this explicitly in the output-contract table so no consumer expects `ns-train --data`
   to work.
2. **No `mesh_tsdf.ply` back-compat.** Task 5 renamed the TSDF output; scenes reconstructed
   before it still hold the old name, so `splatter.py:480` raises `FileNotFoundError` and
   `dashboard/app.py:649` shows no mesh. Deliberate — a legacy fallback would restore the
   two-name ambiguity the rename existed to remove. Document that such scenes need
   `mesh(overwrite=True)` re-run once.
3. **`docs/source/tutorials/06_mesh/create_mesh.ipynb` cell 8 recorded output is fabricated.**
   It lists a `mesh_clean.ply` (was `mesh_tsdf_clean.ply`) that current code cannot produce —
   `mesh/tsdf.py:100-104` hard-raises when `clean_repair=True`. Pre-existing; Task 5's rename
   only relabelled it. Fix by re-executing the notebook, not by hand-editing output.
4. **`mesh_clean.ply` is a dead probe.** `splatter.py:480` looks for it first but nothing writes
   it (same `clean_repair` raise). Either remove the probe or implement cleaning; do not leave
   dead code looking live.
5. **Two byte-identical `_load_feature_maps` copies** — `dashboard/pipeline.py:98` and
   `dashboard/viewer.py:34`. Task 7 Step 5b has to fix the same glob bug in both, which is exactly
   the duplication smell. Merge to one shared helper here; `tests/dashboard/test_viewer_lift.py:16,60`
   patch `viewer._load_feature_maps` by name and will need re-pointing.
6. **`target_cosine` is a self-scored gate.** Task 6's `recon_cosine` is measured on the training
   set; measured gap vs held-out was +0.205 at 300 samples (1.0000 vs 0.7949). Docstrings and
   `configs/base.yaml` now say "training set", but wording is mitigation, not a fix: a `0.95` gate
   cannot fail for the reason a consumer cares about, and small tutorial/demo scenes are exactly
   the ones that publish an inflated 1.0. A held-out split at gate time is cheap (one `randperm`
   slice, no architecture change) — decide here whether published scenes need it.
7. **`_AE_EPOCHS = 10` is STILL LIVE — Task 7 did not remove it.** Now at
   `dashboard/pipeline.py:80`, used ungated at `:127`. Task 7 wired `target_cosine`/`max_epochs`
   through the *Reconstructor* path only, so **three AE policies now coexist**: `configs/base.yaml`
   (0.95 / 100), `viewer.UPGRADE_TARGET_COSINE`/`UPGRADE_MAX_EPOCHS` (0.95 / 100), and the
   dashboard's fresh path (10 epochs, no gate). The consequence is inverted from the obvious one:
   a legacy scene *self-upgraded* on the query path is held to a **stricter** fidelity bar than a
   scene the dashboard reconstructs from scratch. Unify here — one policy, config-sourced.
8. **`wrapper/reconstructor.py::_lift_and_save` is all inline imports** (`numpy`, `torch`,
   `zarr`, `FeedforwardResult`, `lift_features`, `FeatureAutoencoder` at `:257-261,286-289`),
   against the project's imports-at-top rule. Task 7 rewrites this function's body; hoist the
   imports while there rather than preserving the violation. — **RESOLVED in Task 7** (`777925e`):
   `torch`/`zarr`/`FeatureAutoencoder` hoisted; `FeedforwardResult`/`lift_features` deliberately
   left in-function because `pointcloud.utils` pulls the optional feedforward extra.

**Carried forward from the Task 7 review — resolve here unless a later task claims them:**

9. **The two semantics writers converged on filenames but NOT on location.**
   `wrapper/reconstructor.py:747` writes `{backend}/semantics/{extractor_name}/features.zarr`;
   the dashboard writes flat `{scene}/semantics/features.zarr` (`dashboard/app.py:773`). So
   `load_point_features(out / "semantics")` **cannot read a Reconstructor scene**. This is a
   pre-existing divergence (pre-Task-7 the two disagreed on name *and* location), not a Task 7
   regression — but it is exactly the thing that breaks the published output contract, since
   Tasks 10-11 drive the *Reconstructor* path while the dashboard is only a browser. Pick one
   canonical location and state it in the output-contract table. Note the dashboard is
   single-extractor-per-scene (`RunConfig.semantic_extractor`) whereas Reconstructor carries the
   extractor dimension in the path, so a flat layout loses information the general path has.
10. **`min(latent_dim, feats.shape[1])` in `viewer._save_point_features:91`** means any extractor
    with `input_dim <= 64` gets an identity-width bottleneck: no compression *and* a lossy
    encode/decode round-trip — strictly worse than storing the features directly. Decide whether
    to skip the autoencoder entirely below some width.
11. **Unbatched decode in `load_point_features`.** `ae.per_point_decode(torch.from_numpy(codes))`
    materialises the full `(P, 768)` result in one call on CPU — ~2.3 GB for a 500k-point scene,
    on every legacy scene's first query. Batch it, or reuse `utils/torch_utils.batch_iterator`.
12. **`latent_dim` is dead-ish API** — `viewer._save_point_features`'s `latent_dim: int = 64`
    parameter is never passed by its only caller. Per the project's "if a param is always default,
    ask whether it should exist" rule, drop it or wire it to config.
13. **`_extract_semantics`'s `Path` return is ignored** by its caller
    (`dashboard/pipeline.py:72-75`). Either consume it or make the function `-> None`.
14. **`frames.zarr` in `PUSH_EXCLUDES` contradicts the reason it is NOT in `PULL_EXCLUDES`.**
    Task 8's `PUSH_EXCLUDES` skips `frames.zarr/**` as "regenerable from the curated video", but
    the existing `PULL_EXCLUDES` comment (`dashboard/sources.py:30-31`) states the opposite intent:
    "frames.zarr is NOT excluded: it is now the sole persistent frame source, so pulled scenes need
    it (e.g. localization pixel reads)." Never pushing it makes pulling it impossible, so a consumer
    who pulls a processed scene cannot localize without re-running preproc — and must reproduce the
    *same* keyframes to stay consistent with the reconstruction. Uniform sampling is single-pass
    deterministic, so this is reproducible in principle, but it is a hidden precondition, and the
    user's stated second consumer class ("internal further-processing needs the intermediates")
    arguably includes the keyframe store. Decide explicitly: push it (bandwidth/storage cost) or
    document the re-run requirement in the output contract. `feedforward.zarr` IS pushed, so the
    other intermediates are covered either way.
15. **`SceneSource`'s processed-side probes conflate "absent" with "rclone unreachable."**
    `remote/sources.py:194,209,222` (`list_localization_dbs`, `list_processed_scenes`, `has_processed`)
    each wrap their listing in a bare `except Exception` and memoize the `[]`/`False` fallback. That is
    correct for a genuinely absent directory — an unprocessed scene has no remote dir — but a transient
    rclone/network outage produces the identical answer, so **every scene reports unprocessed and a batch
    driver re-runs the entire bucket.** With the 60s listing TTL, one blip poisons a whole sweep. This is
    inherited behavior (the pre-move dashboard code did the same, where a human would notice a suddenly
    empty dropdown); it becomes expensive only now that an unattended driver acts on the answer.
    Options: distinguish not-found from transport errors and re-raise the latter; or keep the fallback but
    don't memoize it. Deliberately NOT fixed inside Task 8 — it is a production behavior change, not the
    test-strength issue the review was scoped to, and it needs a decision on whether a driver should abort
    or skip on an unreachable bucket. Note `has_processed` is what gates the skip-already-done logic, so
    the failure mode is wasted recompute, not data loss.
16. **The two entry points disagree on which video extensions exist.** The local driver's
    `_VIDEO_EXTS` (`docs/examples/run_pipeline.py:58`, moving to `collab_splats/wrapper/batch.py`
    in Task 10) is `{".mp4", ".mov", ".avi"}`; `SceneSource._VIDEO_EXTS`
    (`collab_splats/remote/sources.py:23`) is `(".mp4", ".mov")`. So an `.avi` in a curated scene dir
    is accepted by a local run over a pulled directory but makes `scene_video(scene)` raise
    `FileNotFoundError`, and the remote driver skips that scene as video-less. Failure mode is a
    silent skip, not corruption, and the two constants serve different domains (local filesystem
    scan vs. curated-bucket listing) — so this is a consistency question, not a bug. Decide whether
    to unify on one tuple (and where it should live) or to document that the curated bucket accepts
    mp4/mov only. Deliberately NOT folded into Task 10: `_VIDEO_EXTS` moves verbatim there, and
    changing its value is a behavior change outside that task's scope.
17. **`configs/minimal.yaml` is referenced in two places but has never existed.** `configs/README.md:25`
    documents `python docs/examples/run_pipeline.py --output-root ... --config configs/minimal.yaml ...`
    and the same invocation appears in the script's own module docstring (`docs/examples/run_pipeline.py:27`).
    The repo only ever contained `configs/base.yaml` and `configs/loop_closure.yaml` — `minimal.yaml` is
    absent from the entire git history, so this documented command has always failed with
    `FileNotFoundError`. Found while spec-reviewing Task 10; it is pre-existing staleness, identical
    before and after that commit, not a regression. Fix both sites in Task 12 (which already owns
    `configs/README.md`): either point them at `configs/base.yaml` or drop the `--config` example.

18. **`SceneSource.has_processed()` is never called by the remote driver.** The method exists and
    works, but `run_remote` reconstructs every scene it is handed, so a re-issued `--all` re-runs
    every already-published scene from scratch. Deliberately NOT fixed in Task 11: the approved
    spec fixed the CLI surface at `--all` / named scenes / `--keep-local` (spec decision 10 — "no
    `--list`, `--fetch-only`, `--pull-processed`"), and a `--skip-processed` flag is beyond it.
    Follow-up, not a Task 12 item; it needs a spec amendment first, and item 15's fix is a
    prerequisite (a `has_processed` that silently returns `False` on a transport error would make
    `--skip-processed` skip nothing and look like it worked).
19. **`_run_streaming` merges stderr into stdout**, so rclone's own diagnostics — 403, expired
    credentials, bad remote name — are emitted through `on_line` at INFO rather than ERROR. An
    operator scanning for `ERROR` in an unattended run sees nothing. Pre-existing, not introduced
    by Task 8. Task 12 (item 15) touches this code, so at minimum ensure the exception raised on
    non-zero exit carries the rclone output; separating the streams is a larger change.
20. **`_FakeSource` fidelity to `SceneSource` is unenforced.** `tests/examples/test_run_pipeline_remote.py`
    fakes the source by duck-typing. If a `SceneSource` method signature changes, the fake keeps
    the old shape and 26 tests stay green against an interface that no longer exists. An
    `inspect.signature` guard comparing fake against real for the four methods the driver uses
    (`list_curated_scenes`, `scene_video`, `fetch_scene`, `push_outputs`, `verify_push`) would
    close it. Test-infrastructure hardening, not a defect — optional in Task 12.

21. **Item 15's fix landed on the processed-side probes only; the curated side still lies.**
    `132baba` routed `list_localization_dbs` / `list_processed_scenes` / `has_processed` through
    the new `_lsjson` (exit 3/4 → absent, everything else → raise), but `list_scenes`
    (`remote/sources.py:189-196`) and `scene_video` (`:198-213`) still call
    `RcloneClient.list_directory`, which collapses **every** failure into `[]`. Consequence, worse
    than the one item 15 described: `docs/examples/run_pipeline_remote.py:59` gets its whole work
    list from `list_scenes()`, so a broken rclone yields `[]` and `:61` logs
    `"No scenes to process"` and exits 2 — the unattended driver reports an empty bucket and stops
    quietly, which is exactly what "if the rclone isn't working abort loudly" forbids. `scene_video`
    compounds it by raising `FileNotFoundError(f"no video in {CURATED_BUCKET}/{scene}")` when the
    real cause is an unreachable bucket. **This is a defect in my brief, not the implementer's
    work** — item 15 named only the three processed-side probes because it was written about the
    dashboard, before the remote driver existed. Fix: route both through `_lsjson`. Safe for the
    dashboard: both `list_scenes` call sites already catch `Exception` and degrade to `[]` with a
    user-visible line (`dashboard/app.py:307-313`, `dashboard/localize.py:351-356`), so raising
    changes a silently-empty dropdown into a reported failure.

22. **`mesh_clean.ply` survives in two tutorials, and in this plan's own self-review.** Sweeping
    for every symbol Task 12 deleted turned up only two live references, both in notebooks:
    `docs/source/tutorials/03_splats/derive_splats.ipynb:276` builds
    `rade-features/mesh/mesh_clean.ply` as a real path (only the filename is stale —
    `rade-features` is the current splat `train_method`, `configs/base.yaml:73`), and
    `06_mesh/create_mesh.ipynb:219,244` documents it in an
    output table. No mesher in this repo writes that name: the `clean_repair` path that would
    have produced it raises (`mesh/tsdf.py`), which is why item 4 deleted the candidate list from
    `wrapper/splatter.py`. The **Type consistency** section at the bottom of this plan asserts the
    opposite — "Mesh filename is `mesh.ply` (and `mesh_clean.ply` for the cleaned variant)" — so
    the plan's own self-review propagated the wrong contract, and item 4 caught it. Notebook fix
    folds into carry-forward #3, which needs these re-executed against real data rather than
    hand-edited; 03_splats additionally has no committed splat data to run against.
    Everything else swept clean: `_load_feature_maps`, `UPGRADE_TARGET_COSINE`,
    `UPGRADE_MAX_EPOCHS` and `_AE_EPOCHS` appear only in `tests/dashboard/test_pipeline.py:252-259`
    and `test_viewer_lift.py:34`, as `assert not hasattr` guards against the duplicate policies
    coming back, plus stale dated `graphify-out/*/GRAPH_REPORT.md` snapshots and `docs/_build/`
    artifacts, both regenerated rather than edited.

23. **Items 15 and 21 fully resolved in `5237a50`, and the fix round found a bigger bug than the
    one it was sent for.** `_lsjson` now returns `list[dict]` and raises on **any** non-zero exit;
    `_RCLONE_ABSENT_CODES` is deleted; real absence is an empty listing, which is where all five
    probes now log. `list_scenes` and `scene_video` route through it, so `run_pipeline_remote.py:59`
    dies loudly on a broken rclone instead of reporting an empty bucket. Two of my own briefs were
    wrong and were corrected in the process:
    - **`--stats` has been emitting nothing, on every streamed command.** rclone logs stats at INFO
      while its default `--log-level` is NOTICE, so every interval line was filtered before leaving
      the process — measured: a 1.5 GB `copy` printed zero progress until `--stats-log-level NOTICE`
      was added. `on_line` / `parse_rclone_percent` were therefore **dead on all four** of
      `fetch_video`, `pull_processed`, `pull_zarr_members`, `push_outputs`, not just on verify. Now
      centralised in `_stats_args()`. My F3 finding named `--fast-list` and a heartbeat; the silence
      had a deeper cause than I stated.
    - **`rclone check` exit codes cannot separate a mismatch from a check that never ran** — exit 1
      covers content difference, missing destination and unconfigured remote alike (verified,
      v1.53.3-DEV). `verify_push` classifies on rclone's own `"differences found"` summary text
      instead (`_CHECK_MISMATCH_MARKER`). Both branches still return False, so the destructive
      delete stays blocked either way; only the operator-facing cause differs. `_run_streaming`
      keeps a 5-line tail, which always retains the marker because rclone repeats it in its final
      `Failed to check with N errors: last error was: ...` line.
    `--fast-list` + `--checkers 16` speed the verify (frames.zarr is thousands of small chunks)
    with **no** `--size-only`/`--checksum` downgrade — it gates a destructive local delete, so it
    stays a full content comparison. 6/6 mutants killed. Counts: `tests/remote/` 109,
    `tests/examples/` 30, `tests/dashboard/` 227, SMOKE PASS.

24. **Owed: nobody has watched the now-live progress callbacks against the real remote.** Raising
    rclone's stats to NOTICE turns four previously-silent `on_line` callbacks live at 2s intervals,
    which is a real behaviour change beyond the five findings — dashboard progress bars and the
    driver log both get chattier, and the parse path (`parse_rclone_percent` on
    `"100.141M / 1.490 GBytes, 7%, ..."`) has only been exercised against a synthetic copy. The
    `--smoke` gate passes and `check`'s own stats lines carry no `%`, so they parse to `None` rather
    than to bogus progress. Still wants one live `--all` run watched by a human.

25. **F6, `602b9e9`: making transport faults raise exposed the driver's error taxonomy as too
    coarse.** `run_pipeline_remote.py`'s `except Exception` treated "this video is bad" and "rclone
    has stopped working" identically — mark FAIL, continue — so one expired credential turned into
    a batch where every remaining scene failed and the summary blamed all of them. Before item 21's
    fix this was unreachable, because `list_scenes` swallowed transport faults into `[]`. Fixed with
    `SceneSource.check_available()`: one **uncached** `lsjson` against `CURATED_BUCKET`, re-probed
    after any scene failure. Healthy remote → that really was one bad scene, continue. Dead remote →
    abort, record the rest `SKIPPED` (never `FAIL` — they never ran, so they are safe to re-run),
    and return a new `EXIT_REMOTE_UNAVAILABLE = 3` so cron can tell infrastructure death from a data
    problem. Exit codes are now named constants documented in the module docstring: 0 ok,
    1 scene failed, 2 nothing to do, 3 aborted.
    Two design points worth keeping:
    - Bypassing `_cached` is load-bearing, not incidental. The 60s TTL would hand back a success
      recorded *before* the credentials expired, which defeats the entire probe. Mutating the probe
      to route through `_cached` is one of the two mutants that must stay red.
    - `check_available` catches broad `Exception` deliberately: a probe asked "is rclone working?"
      must return an answer, never raise the fault it was sent to detect. Exit-code classification
      was ruled out for the reason item 23 records — a dead remote surfaces as a bare `RuntimeError`
      out of `_run_streaming`, indistinguishable from a reconstruction error.
    **My F6 rationale overstated the cost and the implementer corrected it:** I claimed a 20-scene
    batch wasted 19 GPU reconstructions. It did not — `fetch_video` runs *before* `batch.run_scene`,
    so each subsequent scene died at the fetch with no GPU work. The real waste was ~19 futile
    fetches with rclone's default retries, plus a summary blaming 20 scenes for one infrastructure
    fault. The fix stands on the operator requirement and the misleading summary alone.
    Residual edge, accepted: if the **last** scene's `verify_push` returns False because the remote
    died, the batch exits 1 rather than 3, since that path `continue`s without raising. Local data
    is kept either way, which is the property that matters; for any non-final scene the next
    `fetch_video` raises and triggers the abort.
    Counts: `tests/remote/` + `tests/examples/` 158 passed (+19), `tests/dashboard/` 226 + the
    documented `test_view_transform_scales_to_target_radius` flake, SMOKE PASS. 2/2 mutants killed.
    Also fixed en route: the `no video … skipping` log moved inside `scene_video`'s producer, so it
    fires once per TTL instead of on every cached hit.

**User decisions, 2026-07-30 — these resolve items 6, 9, 14, 15 and 16:**

- **Item 14 — push `frames.zarr`.** Drop `frames.zarr/**` from `PUSH_EXCLUDES`. A pulled scene
  must be localizable without re-running preproc, which is what `PULL_EXCLUDES` already assumed.
  Note the same constant governs the dashboard's push (`dashboard/pipeline.py:248`), so the
  change lands in two places; only the remote driver's push is verify-gated.
- **Item 15 — split the two failure modes.** A path genuinely absent from the bucket → skip that
  scene and log it. rclone itself failing (missing binary, bad config, transport error) → abort
  loudly rather than reporting every scene as unprocessed. Applies to
  `list_localization_dbs`, `list_processed_scenes`, `has_processed` (`remote/sources.py:194,209,222`).
- **Item 9 — the existing backend-keyed layout is canonical**;
  `{backend}/semantics/{extractor}/` stays, and the **dashboard** changes to match rather than
  the Reconstructor. A per-point latent code is indexed by the backend's point set, so it belongs
  under the backend; a flat layout would let four backends over one scene overwrite each other.
  The genuinely backend-agnostic artifact is the 2D feature map, which already lives outside the
  backend tree at `features/**`. Do not hoist `autoencoder.pt` away from the codes it was fit on.
- **Item 6 — keep the training-set `recon_cosine`.** No held-out split. Documentation-only:
  the metric is already labelled "training set" in docstrings and `configs/base.yaml`.
- **Item 16 — unify on three extensions.** Add `.avi` to `SceneSource._VIDEO_EXTS`
  (`remote/sources.py:23`) so a curated `.avi` stops being silently skipped. The two constants
  stay separate (different domains: local filesystem scan vs curated-bucket listing).

**Resolutions landed, 2026-07-30:**

- `132baba` — items **14** and **16**, plus item **15** on the processed-side probes only (see
  item 21 for the curated-side gap that review found, and for the exit-code inversion: on GCS an
  absent prefix is exit 0, and exit 3 means a missing *bucket*, so `_RCLONE_ABSENT_CODES` mapped
  a config fault to "absent" and left the required log unreachable).
- `99dd44e` + `57b9b4a` — items **1**, **2**, **6** and **17**: output-contract table, the
  nerfstudio and `mesh_tsdf.ply` limitations, the training-set caveat on `recon_cosine`, and the
  never-existent `configs/minimal.yaml` in both `configs/README.md` and the driver docstring.
  `docs/README.md` also pointed at the retired `examples/run_all_datasets.sh`; fixed.
- `02a135f` — item **9**, read path only. New `resolve_semantics_dir(scene_dir) -> Path | None`
  (`dashboard/pipeline.py:118`) prefers the flat layout, else `sorted(glob("*/semantics/*/features.zarr"))`,
  warning when several backends match. **The plan understated this item:** the divergence is the
  whole dashboard layout, not just semantics (`pipeline.py` writes flat `feedforward.zarr` and
  `mesh/` too), so writing backend-keyed semantics alone would have invented a third layout and
  migrating everything would orphan every dashboard scene on disk. The dashboard is a browser;
  the Reconstructor tree is the contract.
- `11a7ee0` — items **4**, **5**, **7**, **10**, **11**, **12**, **13**. One config-sourced
  `AutoencoderPolicy` via `semantics_ae_policy()` retires `_AE_EPOCHS` and viewer's `UPGRADE_*`
  (three policies → one); `_load_feature_maps` deduped to a single public
  `pipeline.load_feature_maps`, with `test_viewer_lift.py:34` asserting the viewer copy stays
  gone; decode batched through `batch_iterator`; `latent_dim` param dropped in favour of
  `resolve_latent_dim`, which warns when the clamp binds. Item 10's skip-the-autoencoder branch
  was **not** needed — every registered extractor (`dinov2`, `maskclip`, `talk2dino`) is
  ViT-width, so the clamp only binds on fixtures and on `n_components: null`.
- Gates at this point: `tests/dashboard/` + `tests/wrapper/` = 334 passed; `--smoke` = SMOKE PASS.
- Item **8** was already resolved in Task 7 (`777925e`). Item **3** still needs
  `docs/source/tutorials/06_mesh/create_mesh.ipynb` re-executed against real data.
- Doc defect found en route: `CLAUDE.md`'s architecture tree listed `semantics/features.py`, but
  that is a package (`semantics/features/`), and the SAM extractors it named are segmentation
  backends, not feature extractors. Fixed.

**Files:**
- Modify: `CLAUDE.md` (In-Flight Work), `docs/README.md` or `docs/source/index.rst` (whichever indexes the drivers), `configs/README.md`
- Modify: `docs/superpowers/plans/2026-07-29-gcloud-remote-pipeline.md` (tick the boxes as you go)

- [ ] **Step 1: Document the output contract**

Add to `configs/README.md`, under the `pointcloud:` section:

```markdown
### Processed scene layout

A processed scene (`environments-processed/<scene>/`) carries:

| path | consumer |
|---|---|
| `<backend>/sparse_pc.ply` | any pipeline — binary little-endian, float32 xyz + uchar rgb |
| `<backend>/mesh/mesh.ply` | any pipeline |
| `<backend>/transforms.json` | camera poses, `ply_file_path`, `applied_transform` (splatfacto) |
| `<backend>/semantics/<extractor>/features.zarr` | per-point latent codes (`semantics.n_components`-D) |
| `<backend>/semantics/<extractor>/autoencoder.pt` | decoder to full 768-D + `recon_cosine` / `recon_mse` |
| `<backend>/colmap/sparse/0/*.bin` | further processing inside this repo |
| `<backend>/feedforward.zarr` | further processing inside this repo (depth, poses, confidence) |
| `frames.zarr` | the keyframes the reconstruction was built from; required to localize |
| `run_config.yaml` | exact settings used |

Every geometry-derived artifact sits under `<backend>/`, including `sparse_pc.ply`
(`reconstructor.py:621` writes `backend_dir / "sparse_pc.ply"`) and the per-point
semantics. One scene may be reconstructed by several backends — the LC parity work runs
four — so a per-point latent code is only meaningful next to the point set it indexes.

Not pushed (`PUSH_EXCLUDES` in `collab_splats/remote/sources.py`): `features/**` (raw 2D
feature maps, regenerable from frames + extractor) and the source video, which the remote
driver fetches into the very scene dir it later pushes and which already lives in
`environments-curated`. Video patterns use bracket character classes (`*.[Mm][Pp]4`)
because rclone globs are case sensitive and camera files are commonly uppercase.

Comparing semantics across scenes: decode to 768-D first. Two independently-trained
autoencoders do not share a 64-D basis, so raw latent codes are not comparable; the
decoded space is (limited by each scene's `recon_cosine`, targeted at 0.95).
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the **In-Flight Work** list, add:

```markdown
Recently completed (2026-07-29): **gcloud-remote-pipeline** — `environments-curated` → reconstruct → `environments-processed` with verified push + local cleanup; binary `sparse_pc.ply`; mesh unified on `mesh/mesh.ply`; semantics converged on `features.zarr` + `autoencoder.pt`; `SessionSource` → `collab_splats/remote/SceneSource` ([spec](docs/superpowers/specs/2026-07-29-gcloud-remote-pipeline-design.md) · [plan](docs/superpowers/plans/2026-07-29-gcloud-remote-pipeline.md)). Remote driver: `docs/examples/run_pipeline_remote.py`.
```

- [x] **Step 3: Run the full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: no failures beyond those already listed in `docs/known-test-failures.md`. If a new failure appears, fix it before committing — do not add it to the known-failures file.

**Result:** `1455 passed, 2 skipped, 3 xfailed` — clean, zero failures. Add `-p no:randomly`: pytest-randomly is installed, so an unqualified run shuffles order and any "green" claim is order-specific.

Two new failures appeared on the first gate run and were fixed, not filed:
- `tests/wrapper/test_reconstructor.py` ×2 — the review round made `_lift_and_save`'s `target_cosine`/`max_epochs` required, and the test helper drove it positionally with only `max_epochs`. Fixed in the helper. The sole production caller (`reconstructor.py:771`) already threads both from `sem_cfg`.
- `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius` — the known flake at `docs/known-test-failures.md:73` (unseeded `np.random.rand(500, 3)`), passes in isolation and passed on the confirming run.

- [x] **Step 4: Run the dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`

**Result:** `SMOKE PASS: page (8358 B) + bokeh.min.js (1264808 B) served; bind took 3s`

- [x] **Step 5: Refresh the knowledge graph**

Run: `graphify update .`
Expected: completes without error

**Result:** rebuilt to 4928 nodes / 9743 edges (was 4600 / 9006). An earlier attempt tripped the overwrite guard (`new graph has 30796 nodes but existing graph.json has 41044`); `--force` was declined at the time rather than bake in a possibly-truncated graph. The clean re-run needed no force, and the guard's numbers turned out to be a different counting unit — actual node count *grew*, and the new symbols (`write_pointcloud_ply`, `SceneSource`, `verify_push`, `export.py`, `batch.py`, `run_pipeline_remote`) are all indexed while the deleted `webapp/` is absent.

- [x] **Step 6: Commit**

```bash
# NO `black .` (see Conventions). isort only the paths on the git add line below.
git add configs/README.md CLAUDE.md graphify-out
git add -f docs/superpowers/plans/2026-07-29-gcloud-remote-pipeline.md
git commit -m "docs(remote): document the processed scene layout and remote driver"
```

**Deviation:** `graphify-out` is gitignored (`.gitignore:1`) and cannot be staged — dropped from the `git add`. `configs/README.md` and `CLAUDE.md` landed in `f1f4ed0` as part of the review-round commits below.

- [ ] **Step 7: End-to-end smoke against one real scene (manual, needs the bucket)**

```bash
/opt/venv/reconstruction/bin/python docs/examples/run_pipeline_remote.py \
  --output-root /workspace/outputs/remote-smoke \
  --keep-local \
  $( /opt/venv/reconstruction/bin/python -c "from collab_splats.remote import SceneSource; print(SceneSource().list_scenes()[0])" )
```

Verify:
- `outputs/remote-smoke/<scene>/<backend>/sparse_pc.ply` starts with `format binary_little_endian 1.0`
- `outputs/remote-smoke/<scene>/<backend>/mesh/mesh.ply` exists (with `mesh.enabled: true`)
- `transforms.json` contains `ply_file_path`
- `semantics/features.zarr` second dim == `semantics.n_components`
- rclone shows the scene under `environments-processed/<scene>/` with no `features/` (the 2D patch cache). `frames.zarr/` **is** expected — the user chose "Push it", so it left `PUSH_EXCLUDES`; this line said otherwise while the exclude list was still being settled.
- Re-run without `--keep-local` and confirm the local scene dir is gone and the remote copy is intact

Record the result in `docs/superpowers/plans/2026-07-29-gcloud-remote-pipeline.md` under this step. Heavy run — use tmux, not a notebook (container cgroup cap is 46.6 GB).

**Still owed.** Not run: needs the real bucket and a human watching. Four progress callbacks are live, so this is the run that confirms they report sensibly.

---

## Review round (post-Task-12)

The final two-stage review over the whole implementation raised 15 items; all are fixed and committed. Each behavioural fix is pinned by a test.

| Commit | Covers |
|---|---|
| `c26cb04` | `check_available` re-attempts `RcloneClient()` (a construction-time verdict lasted the whole process, so a transient fault was permanent); `PUSH_EXCLUDES` feature pattern anchored; `_run_streaming` tail keeps errors over progress lines; `verify_push` handler de-duplicated; `_SCENE_RE` → public `SCENE_ID_RE` |
| `6042f64` | Every scene-failure path probes the remote and aborts the batch, not just the exception path (a dead remote mid-batch previously recorded N quiet FAILs — the opposite of "abort loudly"); `list_scenes()` failure exits 3 not 1; cleanup deletes the dir `verify_push` verified; `--scenes` ids validated against `SCENE_ID_RE` |
| `1f20878` | Unreachable backend-keyed branch removed from `resolve_semantics_dir` |
| `91fe680` | `_lift_and_save` training args required instead of defaulted |
| `f1f4ed0` | Docs: dashboard read path, exit-code precedence, architecture tree |
| `6a1861d` | Stale mock in `test_feedforward_logging` (a `MagicMock` reconstruction is truthy, so `PointcloudResult.points` skipped its empty-guard and produced a `(0,)` array once Task 2 wired the PLY writer in) |

**Two review claims were rejected after checking the code:**
- The 12a review said `SceneSource()` construction at `main():212` can raise like `list_scenes()`. It cannot — `__init__:112-116` catches `Exception` and degrades to `self._client = None`. Wrapping it would have been dead code.
- Important finding #1 asked for the dashboard and the driver to share one semantics layout. Declined per the recorded decision at lines 2911-2915 ("The dashboard is a browser; the Reconstructor tree is the contract") — unifying would orphan every dashboard scene already on disk. Documented as a limitation instead, which is what makes the removed glob branch dead code rather than groundwork.

**The `PUSH_EXCLUDES` anchoring was sent back for measurement** rather than accepted from rclone's documented filter rules, since the same class of assumption had already produced wrong briefs for `--stats` and for `check`'s exit codes. Measured on v1.53.3-DEV: an unanchored `features/**` also excludes `sub/feedforward.zarr/features/x`, which would have dropped a real `feedforward.zarr` member. Latent only because every creator currently passes `features=None`.

---

## Self-Review

**Spec coverage**

| spec decision | task |
|---|---|
| 1 — buckets replace `fieldwork_*` everywhere incl. dashboard | 8 (constants), 9 (dashboard) |
| 2 — scene id = flat curated dir name, no `reconstruction/` prefix | 8 (`_SCENE_RE`, path building), 10 (`scene_output_dir(name=)`) |
| 3 — `SessionSource` → `SceneSource` in `collab_splats/remote/` | 8 |
| 4 — per-scene core into the package, two thin drivers | 10, 11 |
| 5 — rclone throughout via existing `RcloneClient`, creds in remote config | 8 (unchanged client), 11 (`SceneSource()` no-arg) |
| 6 — we write the pointcloud PLY, binary, from `result.points` | 1, 2, 3 |
| 6a — name stays `sparse_pc.ply` | 2 (overwrite in place, `transforms.json` untouched) |
| 6b — mesh unifies on `mesh/mesh.ply` | 5 |
| 6c — cap splits: `pointcloud.max_points` guard + separate export density | 3 |
| 7 — push excludes `features/**` and `frames/**`, compressed semantics ships | 8 (`PUSH_EXCLUDES`), 12 (documented) |
| 8 — semantics converges on `features.zarr` + weights, `lifted_normed` retired | 7 |
| 8a — `target_cosine=0.95`, `max_epochs=100`, metrics in the checkpoint | 6 |
| 9 — delete local after verified push, `--keep-local` skips | 11 |
| 10 — no `--list`, `--fetch-only`, `--pull-processed` | 11 (CLI has none) |
| cleanup 1 — delete the jpg path | 7 step 7 |
| cleanup 5 — resolve the two `transforms.json` writers | 4 |

**Placeholder scan:** no "TBD", no "similar to Task N", no "add appropriate error handling". Every code step carries the code. Task 9 and Task 7 steps 6-8 are mechanical sweeps over enumerated line numbers with a worked example per pattern rather than every diff — the grep that finds the sites and the exact replacement shape are both given.

**Type consistency**

- `write_pointcloud_ply(points, colors, path, max_points=None) -> Path` — defined Task 1, used Task 2 (`_write_ply`) and Task 3 (`_export_pointcloud_ply`) with that order.
- `BasePointcloudCreator._write_ply(result, output_dir, max_points=None) -> Path` — Task 2, tested Task 2.
- `Reconstructor._export_pointcloud_ply(result) -> Path` — Task 3, called Task 3.
- `FeatureAutoencoder.fit(..., target_cosine=None)`, attributes `recon_cosine` / `recon_mse` / `epochs_run` — Task 6, consumed by `_lift_and_save(target_cosine=, max_epochs=)` in Task 7 and documented in Task 12.
- `SceneSource` methods take one `scene: str`: `list_scenes()`, `scene_video(scene)`, `fetch_video(scene, dest_dir, on_line=None)`, `list_processed_scenes()`, `has_processed(scene)`, `list_localization_dbs(scene)`, `pull_processed(scene, dest_dir, excludes=(), on_line=None)`, `pull_zarr_members(scene, dest_dir, members, on_line=None)`, `push_outputs(local_dir, scene, on_line=None)`, `verify_push(local_dir, scene, on_line=None) -> bool`. Task 8 defines, Tasks 9 and 11 use exactly these.
- `batch.run_scene(video, output_root, config_dir, override_config, stages, overwrite, name=None)` — Task 10 defines; Task 11 calls positionally then `name=scene`; `run_pipeline.py`'s existing `main()` matches `run_all`'s unchanged signature.
- `load_point_features(semantics_dir, *, decode=True) -> np.ndarray` — Task 7 step 5 defines, steps 6 use.
- Mesh filename is `mesh.ply` uniformly from Task 5 onward; Tasks 7 and 12 use the new name.
  (This line originally added "and `mesh_clean.ply` for the cleaned variant" — wrong, no mesher
  writes that name. See item 22.)
