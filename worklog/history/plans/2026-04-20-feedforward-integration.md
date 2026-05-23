# Feedforward Pointcloud Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate MapAnything and VGGT-X as feedforward pointcloud backends, fix coordinate convention bugs, standardize output paths, and clean up stage/ files.

**Architecture:** Four-creator registry (colmap/hloc/mapanything/vggtx) under `BasePointcloudCreator`. Feedforward creators share `BaseFeedforwardCreator` template: `_run_inference()` writes COLMAP binary to `{output_dir}/colmap/sparse/0/`, base class calls `_write_transforms()` once. All creators produce identical disk output contract.

**Tech Stack:** pycolmap, hloc, stage/mapanything_utils.py, stage/vggt_utils.py, nerfstudio (colmap_to_json only), pytest

**Spec:** `worklog/history/specs/2026-04-20-pointcloud-feedforward-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `collab_splats/pointcloud/base.py` | Modify | CoordinateFrame, PointcloudResult fields, _colmap_recon_to_result fix, _write_transforms, rename create→reconstruct |
| `collab_splats/pointcloud/sfm.py` | Rewrite | ColmapCreator + HlocCreator (split from NerfstudioSfmCreator) |
| `collab_splats/pointcloud/feedforward.py` | Rewrite | BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator |
| `collab_splats/pointcloud/__init__.py` | Modify | 4-key registry: colmap/hloc/mapanything/vggtx |
| `stage/mapanything_utils.py` | Add (copy) | MapAnything pipeline steps 1–5 |
| `stage/preproc_utils.py` | Add (copy) | Video rotation + frame loading utils |
| `stage/vggt_utils.py` | Add (copy) | VGGT-X inference → COLMAP |
| `stage/feedforward.py` | Delete | Replaced by BaseFeedforwardCreator |
| `stage/optical_flow.py` | Delete | Already absorbed into frame_sampling.py |
| `tests/pointcloud/test_base.py` | Modify | Add CoordinateFrame, frame/world_transform, transform B assertion |
| `tests/pointcloud/test_sfm_creator.py` | Rewrite | ColmapCreator + HlocCreator tests |
| `tests/pointcloud/test_mapanything_creator.py` | Modify | Update defaults, disk output contract |
| `tests/pointcloud/test_registry.py` | Rewrite | 4 new keys |
| `tests/pointcloud/test_vggtx_creator.py` | Create | VGGTXCreator defaults + param tests |

---

## Task 1: Copy stage/ source files

No TDD — these are existing, tested files being moved into this branch.

**Files:**
- Create: `stage/mapanything_utils.py`
- Create: `stage/preproc_utils.py`
- Create: `stage/vggt_utils.py`

- [ ] **Step 1: Copy mapanything_utils.py and preproc_utils.py from tlb-improve-mesh**

```bash
mkdir -p stage
git show tlb-improve-mesh:stage/mapanything_utils.py > stage/mapanything_utils.py
git show tlb-improve-mesh:stage/preproc_utils.py > stage/preproc_utils.py
```

- [ ] **Step 2: Copy vggt_utils.py from nerfstudio fork (keep original in fork)**

```bash
cp /workspace/nerfstudio/nerfstudio/process_data/vggt_utils.py stage/vggt_utils.py
```

- [ ] **Step 3: Verify files copied correctly**

```bash
head -5 stage/mapanything_utils.py stage/preproc_utils.py stage/vggt_utils.py
```

Expected: each file shows its module docstring, no errors.

- [ ] **Step 4: Commit**

```bash
git add stage/mapanything_utils.py stage/preproc_utils.py stage/vggt_utils.py
git commit -m "feat(stage): add mapanything_utils, preproc_utils, vggt_utils from source branches"
```

---

## Task 2: base.py — CoordinateFrame enum + PointcloudResult fields + rename

**Files:**
- Modify: `collab_splats/pointcloud/base.py`
- Modify: `tests/pointcloud/test_base.py`

- [ ] **Step 1: Write failing tests for CoordinateFrame and new PointcloudResult fields**

In `tests/pointcloud/test_base.py`, add after the existing imports:

```python
from collab_splats.pointcloud.base import CoordinateFrame
```

Add these tests:

```python
def test_coordinate_frame_values():
    assert CoordinateFrame.COLMAP == "colmap"
    assert CoordinateFrame.NERFSTUDIO == "nerfstudio"
    assert isinstance(CoordinateFrame.COLMAP, str)


def test_result_has_frame_and_world_transform():
    r = PointcloudResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        confidence=None,
        camera_poses=None,
        camera_intrinsics=None,
        colmap_reconstruction=None,
    )
    assert r.frame == CoordinateFrame.NERFSTUDIO
    assert r.world_transform is None


def test_result_explicit_frame():
    r = PointcloudResult(
        points=np.zeros((5, 3), dtype=np.float32),
        colors=np.zeros((5, 3), dtype=np.uint8),
        confidence=None,
        camera_poses=None,
        camera_intrinsics=None,
        colmap_reconstruction=None,
        frame=CoordinateFrame.COLMAP,
    )
    assert r.frame == CoordinateFrame.COLMAP


def test_base_creator_abstract_method_is_reconstruct():
    import inspect
    abstract_methods = BasePointcloudCreator.__abstractmethods__
    assert "reconstruct" in abstract_methods
    assert "create" not in abstract_methods
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_base.py::test_coordinate_frame_values tests/pointcloud/test_base.py::test_result_has_frame_and_world_transform tests/pointcloud/test_base.py::test_base_creator_abstract_method_is_reconstruct -v
```

Expected: ImportError on CoordinateFrame, AttributeError on frame field, and "reconstruct" assertion failure.

- [ ] **Step 3: Add CoordinateFrame + update PointcloudResult + rename create→reconstruct in base.py**

Replace the top of `collab_splats/pointcloud/base.py` with:

```python
# collab_splats/pointcloud/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


class CoordinateFrame(str, Enum):
    COLMAP = "colmap"         # w2c, OpenCV axes, world -Y up
    NERFSTUDIO = "nerfstudio" # c2w, OpenGL axes, world +Z up


@dataclass
class PointcloudResult:
    points: np.ndarray                    # (N, 3) float32, world XYZ
    colors: np.ndarray                    # (N, 3) uint8, RGB
    confidence: np.ndarray | None         # (N,) float32 — feedforward only
    camera_poses: np.ndarray | None       # (M, 4, 4) float32
    camera_intrinsics: np.ndarray | None  # (M, 3, 3) float32, K per image
    colmap_reconstruction: Any | None     # pycolmap.Reconstruction — BA + pycolmap API
    frame: CoordinateFrame = CoordinateFrame.NERFSTUDIO
    world_transform: np.ndarray | None = None
    # (3, 4) applied_transform: COLMAP world → nerfstudio world.
    # Matches transforms.json["applied_transform"].
    # None when keep_original_world_coordinate=True.


class BasePointcloudCreator(ABC):
    @abstractmethod
    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        """images in image_dir → sparse pointcloud + camera poses written to output_dir.

        Produces:
            {output_dir}/colmap/sparse/0/{cameras,images,points3D}.bin
            {output_dir}/transforms.json
            {output_dir}/sparse_pc.ply

        Raises:
            RuntimeError: if reconstruction fails
            FileNotFoundError: if image_dir does not exist
        """
        ...

    def _write_transforms(self, sparse_dir: Path, output_dir: Path) -> None:
        from nerfstudio.process_data.colmap_utils import colmap_to_json
        colmap_to_json(recon_dir=sparse_dir, output_dir=output_dir)
```

- [ ] **Step 4: Run new tests to verify they pass**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_base.py::test_coordinate_frame_values tests/pointcloud/test_base.py::test_result_has_frame_and_world_transform tests/pointcloud/test_base.py::test_base_creator_abstract_method_is_reconstruct tests/pointcloud/test_base.py::test_result_explicit_frame -v
```

Expected: 4 PASS.

- [ ] **Step 5: Run full test_base.py to check for regressions**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_base.py -v
```

Expected: `test_result_fields` still passes (fields unchanged); `test_base_creator_is_abstract` still passes.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/base.py tests/pointcloud/test_base.py
git commit -m "feat(pointcloud/base): CoordinateFrame enum, frame/world_transform fields, rename create→reconstruct"
```

---

## Task 3: base.py — fix _colmap_recon_to_result (transform B + world_transform)

**Files:**
- Modify: `collab_splats/pointcloud/base.py`
- Modify: `tests/pointcloud/test_base.py`

**Background:** The current `_colmap_recon_to_result` applies only Transform A (OpenCV→OpenGL camera axis flip). It is missing Transform B (COLMAP world -Y → nerfstudio world +Z). The existing test `test_colmap_recon_to_result_convention` uses a real `pycolmap.Reconstruction` with identity w2c and checks for Transform A output. We need to update it to verify Transform B and the new `frame` + `world_transform` fields.

With identity w2c (`pycolmap.Rigid3d()`), the expected final c2w after both transforms:
- After Transform A: `[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]`
- After Transform B (row swap [0,2,1,3] + negate row 2): `[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]`

The `world_transform` stored as (3,4):
```
[[1, 0,  0, 0],
 [0, 0,  1, 0],
 [0,-1,  0, 0]]
```

- [ ] **Step 1: Update test_colmap_recon_to_result_convention to check Transform B + new fields**

Replace the existing `test_colmap_recon_to_result_convention` in `tests/pointcloud/test_base.py`:

```python
def test_colmap_recon_to_result_convention():
    """_colmap_recon_to_result applies both OpenCV→OpenGL (A) and COLMAP→nerfstudio world (B).

    Identity w2c → expected c2w after both transforms:
        [[1,  0,  0, 0],
         [0,  0, -1, 0],
         [0,  1,  0, 0],
         [0,  0,  0, 1]]
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=64, height=64, params=[50.0, 32.0, 32.0])
    cam.camera_id = 1
    recon.add_camera(cam)

    img = pycolmap.Image(name="frame_0000.jpg", camera_id=1)
    img.image_id = 1
    img.cam_from_world = pycolmap.Rigid3d()  # identity w2c
    recon.add_image(img)
    recon.register_image(img.image_id)

    recon.add_point3D(
        xyz=np.array([0.0, 0.0, 1.0]),
        track=pycolmap.Track(),
        color=np.array([128, 128, 128], dtype=np.uint8),
    )

    result = _colmap_recon_to_result(recon)

    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
    assert result.world_transform.shape == (3, 4)

    pose = result.camera_poses[0]
    expected = np.array([
        [1,  0,  0, 0],
        [0,  0, -1, 0],
        [0,  1,  0, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)
    np.testing.assert_allclose(pose, expected, atol=1e-5)

    expected_world_transform = np.array([
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0, -1, 0, 0],
    ], dtype=np.float32)
    np.testing.assert_allclose(result.world_transform, expected_world_transform, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails (Transform B not yet applied)**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_base.py::test_colmap_recon_to_result_convention -v
```

Expected: FAIL — pose assertion fails (current code only applies Transform A, so pose = `[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]` not the Transform B result), and `world_transform` is None.

- [ ] **Step 3: Update _colmap_recon_to_result in base.py to apply Transform B and populate world_transform**

Find the `_colmap_recon_to_result` function in `collab_splats/pointcloud/base.py` and replace it entirely with:

```python
# Standard COLMAP world (-Y up) → nerfstudio world (+Z up) rotation, stored as (3, 4).
# Matches transforms.json["applied_transform"] for standard COLMAP data.
_WORLD_TRANSFORM = np.array([
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0, -1, 0, 0],
], dtype=np.float32)


def _colmap_recon_to_result(
    recon: pycolmap.Reconstruction,
    confidence: np.ndarray | None = None,
) -> PointcloudResult:
    """Convert pycolmap.Reconstruction to PointcloudResult in nerfstudio world frame.

    Applies two transforms to each camera pose:
      A — OpenCV → OpenGL camera axes: c2w[:3, 1:3] *= -1
      B — COLMAP world (-Y) → nerfstudio world (+Z): row-swap + negate
    """
    pts3d = recon.points3D
    if pts3d:
        points = np.array([p.xyz for p in pts3d.values()], dtype=np.float32)
        colors = np.array([p.color for p in pts3d.values()], dtype=np.uint8)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)

    poses = []
    intrinsics = []
    for image in recon.images.values():
        if not image.registered:
            continue
        w2c_34 = image.cam_from_world.matrix()          # (3, 4)
        w2c = np.vstack([w2c_34, [0.0, 0.0, 0.0, 1.0]])  # (4, 4)
        c2w = np.linalg.inv(w2c)

        # Transform A: OpenCV → OpenGL camera axes
        c2w[:3, 1:3] *= -1

        # Transform B: COLMAP world (-Y up) → nerfstudio world (+Z up)
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        poses.append(c2w.astype(np.float32))

        cam = recon.cameras[image.camera_id]
        fy = getattr(cam, "focal_length_y", None) or cam.focal_length
        fx = cam.focal_length
        cx, cy = cam.principal_point_x, cam.principal_point_y
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        intrinsics.append(K)

    return PointcloudResult(
        points=points,
        colors=colors,
        confidence=confidence,
        camera_poses=np.stack(poses) if poses else None,
        camera_intrinsics=np.stack(intrinsics) if intrinsics else None,
        colmap_reconstruction=recon,
        frame=CoordinateFrame.NERFSTUDIO,
        world_transform=_WORLD_TRANSFORM.copy(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_base.py -v
```

Expected: ALL PASS including the updated `test_colmap_recon_to_result_convention`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/base.py tests/pointcloud/test_base.py
git commit -m "fix(pointcloud/base): apply transform B in _colmap_recon_to_result, populate world_transform and frame"
```

---

## Task 4: sfm.py — ColmapCreator (fix output path + split)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py`
- Modify: `tests/pointcloud/test_sfm_creator.py`

**Bug:** `NerfstudioSfmCreator._run_pycolmap` writes to `output_dir/sparse/0/`. Required path: `output_dir/colmap/sparse/0/`.

- [ ] **Step 1: Write failing tests for ColmapCreator**

Replace `tests/pointcloud/test_sfm_creator.py` entirely:

```python
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
from collab_splats.pointcloud.sfm import ColmapCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


@pytest.fixture
def tiny_image_dir(tmp_path):
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"frame_{i:04d}.jpg")
    return tmp_path


def test_colmap_creator_defaults():
    c = ColmapCreator()
    assert c.camera_model == "SIMPLE_RADIAL"
    assert c.single_camera is False


def test_colmap_creator_single_camera():
    c = ColmapCreator(single_camera=True)
    assert c.single_camera is True


def test_colmap_creator_output_path(tiny_image_dir, tmp_path):
    """ColmapCreator must write binary files to output_dir/colmap/sparse/0/."""
    out = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("pycolmap.extract_features"), \
         patch("pycolmap.match_exhaustive"), \
         patch("pycolmap.incremental_mapping", return_value={0: mock_recon}), \
         patch.object(ColmapCreator, "_write_transforms") as mock_wt:
        creator = ColmapCreator()
        creator.reconstruct(tiny_image_dir, out)
        sparse_dir = out / "colmap" / "sparse" / "0"
        mock_wt.assert_called_once_with(sparse_dir, out)


def test_colmap_creator_no_reconstruction_raises(tiny_image_dir, tmp_path):
    out = tmp_path / "out"
    with patch("pycolmap.extract_features"), \
         patch("pycolmap.match_exhaustive"), \
         patch("pycolmap.incremental_mapping", return_value={}):
        creator = ColmapCreator()
        with pytest.raises(RuntimeError, match="reconstruction failed"):
            creator.reconstruct(tiny_image_dir, out)


def test_colmap_creator_missing_image_dir_raises(tmp_path):
    creator = ColmapCreator()
    with pytest.raises(FileNotFoundError):
        creator.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


@pytest.mark.gpu
def test_colmap_creator_smoke(tiny_image_dir, tmp_path):
    out = tmp_path / "out"
    creator = ColmapCreator(single_camera=True)
    try:
        result = creator.reconstruct(tiny_image_dir, out)
        assert isinstance(result, PointcloudResult)
        assert result.frame == CoordinateFrame.NERFSTUDIO
        assert result.world_transform is not None
        assert result.points.shape[1] == 3
        if result.camera_poses is not None:
            assert result.camera_poses.shape[1:] == (4, 4)
    except RuntimeError as e:
        assert "reconstruction failed" in str(e).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_sfm_creator.py::test_colmap_creator_defaults -v
```

Expected: ImportError — `ColmapCreator` does not exist yet.

- [ ] **Step 3: Rewrite sfm.py with ColmapCreator**

Replace `collab_splats/pointcloud/sfm.py` entirely:

```python
# collab_splats/pointcloud/sfm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class ColmapCreator(BasePointcloudCreator):
    """Pointcloud via pycolmap SIFT feature extraction + exhaustive matching."""

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

        camera_mode = (
            pycolmap.CameraMode.SINGLE if self.single_camera else pycolmap.CameraMode.AUTO
        )
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=camera_mode,
            camera_model=self.camera_model,
        )
        pycolmap.match_exhaustive(str(db_path))
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(image_dir),
            output_path=str(sparse_dir.parent),  # colmap/sparse/ → creates 0/ inside
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")

        recon = reconstructions[0]
        recon.write_binary(str(sparse_dir))
        self._write_transforms(sparse_dir, output_dir)
        return _colmap_recon_to_result(recon)
```

Note: `HlocCreator` is added in Task 5. Do not add it yet.

- [ ] **Step 4: Run ColmapCreator tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_sfm_creator.py -v -k "not smoke"
```

Expected: PASS for defaults, single_camera, output_path, no_reconstruction_raises, missing_image_dir_raises.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/sfm.py tests/pointcloud/test_sfm_creator.py
git commit -m "feat(pointcloud/sfm): ColmapCreator with correct colmap/sparse/0/ output path"
```

---

## Task 5: sfm.py — HlocCreator (direct hloc, no nerfstudio dep)

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py`
- Modify: `tests/pointcloud/test_sfm_creator.py`

- [ ] **Step 1: Write failing tests for HlocCreator**

Add to `tests/pointcloud/test_sfm_creator.py`:

```python
from collab_splats.pointcloud.sfm import HlocCreator


def test_hloc_creator_defaults():
    c = HlocCreator()
    assert c.retrieval_conf == "netvlad"
    assert c.feature_conf == "superpoint_aachen"
    assert c.matcher_conf == "superglue"


def test_hloc_creator_output_path(tiny_image_dir, tmp_path):
    """HlocCreator must write binary files to output_dir/colmap/sparse/0/."""
    out = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("hloc.extract_features.main", return_value=tmp_path / "feats.h5"), \
         patch("hloc.pairs_from_retrieval.main"), \
         patch("hloc.match_features.main", return_value=tmp_path / "matches.h5"), \
         patch("hloc.reconstruction.main", return_value=mock_recon), \
         patch.object(HlocCreator, "_write_transforms") as mock_wt:
        creator = HlocCreator()
        creator.reconstruct(tiny_image_dir, out)
        sparse_dir = out / "colmap" / "sparse" / "0"
        mock_wt.assert_called_once_with(sparse_dir, out)


def test_hloc_creator_no_reconstruction_raises(tiny_image_dir, tmp_path):
    out = tmp_path / "out"
    with patch("hloc.extract_features.main", return_value=tmp_path / "feats.h5"), \
         patch("hloc.pairs_from_retrieval.main"), \
         patch("hloc.match_features.main", return_value=tmp_path / "matches.h5"), \
         patch("hloc.reconstruction.main", return_value=None):
        creator = HlocCreator()
        with pytest.raises(RuntimeError, match="reconstruction failed"):
            creator.reconstruct(tiny_image_dir, out)


def test_hloc_creator_missing_image_dir_raises(tmp_path):
    creator = HlocCreator()
    with pytest.raises(FileNotFoundError):
        creator.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_hloc_no_nerfstudio_dep():
    """HlocCreator must not import nerfstudio at module level."""
    import collab_splats.pointcloud.sfm as sfm_module
    import inspect
    src = inspect.getsource(sfm_module)
    assert "nerfstudio" not in src.split("class HlocCreator")[1].split("class ")[0], \
        "HlocCreator must not import nerfstudio"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_sfm_creator.py::test_hloc_creator_defaults -v
```

Expected: ImportError — `HlocCreator` does not exist yet.

- [ ] **Step 3: Add HlocCreator to sfm.py**

Append to `collab_splats/pointcloud/sfm.py`:

```python

@dataclass
class HlocCreator(BasePointcloudCreator):
    """Pointcloud via hloc (SuperPoint+SuperGlue feature matching). No nerfstudio dependency."""

    retrieval_conf: str = "netvlad"
    feature_conf: str = "superpoint_aachen"
    matcher_conf: str = "superglue"

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        hloc_dir = output_dir / "colmap" / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)

        retrieval_path = extract_features.main(
            extract_features.confs[self.retrieval_conf], image_dir, hloc_dir
        )
        pairs_path = hloc_dir / "pairs.txt"
        pairs_from_retrieval.main(retrieval_path, pairs_path)

        feature_path = extract_features.main(
            extract_features.confs[self.feature_conf], image_dir, hloc_dir
        )
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

        self._write_transforms(sparse_dir, output_dir)
        return _colmap_recon_to_result(recon)
```

- [ ] **Step 4: Run all sfm tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_sfm_creator.py -v -k "not smoke"
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/sfm.py tests/pointcloud/test_sfm_creator.py
git commit -m "feat(pointcloud/sfm): HlocCreator with direct hloc calls, no nerfstudio dep"
```

---

## Task 6: feedforward.py — BaseFeedforwardCreator + MapAnythingCreator

**Files:**
- Rewrite: `collab_splats/pointcloud/feedforward.py`
- Modify: `tests/pointcloud/test_mapanything_creator.py`

**Key facts from mapanything_utils.py:**
- `export_to_colmap(outputs, views, image_names, output_dir, model)` → writes to `output_dir/colmap/sparse/0/`, returns `Path`
- `rescale_to_original_dimensions(colmap_sparse_dir, image_paths, model_width, model_height, output_dir)` → returns `Path` to rescaled sparse dir (inside `output_dir`)
- `load_and_preprocess_images(image_dir)` → returns `(views, image_paths)`
- `run_mapanything_inference(model, views, ..., confidence_percentile=35.0, ...)` → `List[Dict]`

- [ ] **Step 1: Write failing tests for MapAnythingCreator**

Replace `tests/pointcloud/test_mapanything_creator.py` entirely:

```python
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from collab_splats.pointcloud.feedforward import MapAnythingCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


def test_mapanything_defaults():
    c = MapAnythingCreator()
    assert c.model_name == "facebook/map-anything"
    assert c.confidence_percentile == 35.0
    assert c.use_multiview_confidence is True
    assert c.minibatch_size == 1


def test_mapanything_is_feedforward_creator():
    assert issubclass(MapAnythingCreator, BaseFeedforwardCreator)


def test_mapanything_missing_image_dir_raises(tmp_path):
    c = MapAnythingCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_mapanything_run_inference_writes_to_standard_path(tmp_path):
    """_run_inference must ensure binary files land at output_dir/colmap/sparse/0/."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    mock_model = MagicMock()
    mock_views = [{"img": np.zeros((3, 518, 336))}]
    mock_image_paths = [image_dir / "img.jpg"]
    mock_outputs = [{}]
    sparse_path = output_dir / "colmap" / "sparse" / "0"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("stage.mapanything_utils.load_mapanything_model", return_value=mock_model), \
         patch("stage.mapanything_utils.load_and_preprocess_images",
               return_value=(mock_views, mock_image_paths)), \
         patch("stage.mapanything_utils.run_mapanything_inference", return_value=mock_outputs), \
         patch("stage.mapanything_utils.export_to_colmap", return_value=sparse_path) as mock_export, \
         patch("stage.mapanything_utils.rescale_to_original_dimensions", return_value=sparse_path), \
         patch("pycolmap.Reconstruction", return_value=mock_recon), \
         patch.object(MapAnythingCreator, "_write_transforms"):
        creator = MapAnythingCreator()
        creator._run_inference(image_dir, output_dir)
        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert kwargs.get("output_dir") == output_dir or mock_export.call_args[0][3] == output_dir


def test_mapanything_run_inference_passes_confidence_percentile(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    sparse_path = output_dir / "colmap" / "sparse" / "0"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("stage.mapanything_utils.load_mapanything_model", return_value=MagicMock()), \
         patch("stage.mapanything_utils.load_and_preprocess_images",
               return_value=([{"img": np.zeros((3, 518, 336))}], [image_dir / "img.jpg"])), \
         patch("stage.mapanything_utils.run_mapanything_inference",
               return_value=[{}]) as mock_infer, \
         patch("stage.mapanything_utils.export_to_colmap", return_value=sparse_path), \
         patch("stage.mapanything_utils.rescale_to_original_dimensions", return_value=sparse_path), \
         patch("pycolmap.Reconstruction", return_value=mock_recon), \
         patch.object(MapAnythingCreator, "_write_transforms"):
        creator = MapAnythingCreator(confidence_percentile=50.0)
        creator._run_inference(image_dir, output_dir)
        _, kwargs = mock_infer.call_args
        assert kwargs.get("confidence_percentile") == 50.0


@pytest.mark.gpu
def test_mapanything_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = MapAnythingCreator()
    result = c.reconstruct(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
    assert result.points.shape[1] == 3
    assert (tmp_path / "out" / "transforms.json").exists()
    assert (tmp_path / "out" / "colmap" / "sparse" / "0" / "cameras.bin").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_mapanything_creator.py::test_mapanything_defaults -v
```

Expected: ImportError — `BaseFeedforwardCreator` not found, and defaults check fails (old code has `conf_threshold`, not `confidence_percentile`).

- [ ] **Step 3: Rewrite feedforward.py with BaseFeedforwardCreator + MapAnythingCreator**

Replace `collab_splats/pointcloud/feedforward.py` entirely:

```python
# collab_splats/pointcloud/feedforward.py
from __future__ import annotations

import sys
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class BaseFeedforwardCreator(BasePointcloudCreator):
    """Template for feedforward (depth-estimation) pointcloud creators.

    Subclasses implement _run_inference() which writes binary COLMAP files to
    output_dir/colmap/sparse/0/ and returns the pycolmap.Reconstruction.
    Base class calls _write_transforms() once after _run_inference() completes.
    """

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        sparse_dir = output_dir / "colmap" / "sparse" / "0"
        recon = self._run_inference(image_dir, output_dir)
        self._write_transforms(sparse_dir, output_dir)
        return _colmap_recon_to_result(recon)

    @abstractmethod
    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        """Run model inference, write binary to output_dir/colmap/sparse/0/, return Reconstruction."""
        ...


def _add_stage_to_path() -> None:
    repo_root = Path(__file__).parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    """Pointcloud via MapAnything feedforward depth + pose estimation.

    No feature matching — depth and pose estimated directly by the network.
    Populates PointcloudResult.confidence via inference filtering.
    """

    model_name: str = "facebook/map-anything"
    confidence_percentile: float = 35.0   # keep points above this percentile (top 65%)
    use_multiview_confidence: bool = True  # multi-view depth consistency filter
    minibatch_size: int = 1               # frames processed at once (1 = most memory-efficient)

    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        _add_stage_to_path()
        from stage.mapanything_utils import (
            load_mapanything_model,
            load_and_preprocess_images,
            run_mapanything_inference,
            export_to_colmap,
            rescale_to_original_dimensions,
        )

        model = load_mapanything_model(model_name=self.model_name)
        views, image_paths = load_and_preprocess_images(image_dir)
        image_names = [p.name for p in image_paths]

        model_width = views[0]["img"].shape[-1]
        model_height = views[0]["img"].shape[-2]

        outputs = run_mapanything_inference(
            model,
            views,
            memory_efficient_inference=True,
            minibatch_size=self.minibatch_size,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            use_multiview_confidence=self.use_multiview_confidence,
            confidence_percentile=self.confidence_percentile,
        )

        # Steps 4–5: export at model resolution, then rescale intrinsics to original dims
        sparse_dir = export_to_colmap(
            outputs, views, image_names, output_dir=output_dir, model=model
        )
        rescaled_sparse_dir = rescale_to_original_dimensions(
            sparse_dir, image_paths, model_width, model_height, output_dir=output_dir
        )

        return pycolmap.Reconstruction(str(rescaled_sparse_dir))
```

- [ ] **Step 4: Run MapAnythingCreator tests (non-GPU)**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_mapanything_creator.py -v -k "not smoke"
```

Expected: ALL PASS (mocks bypass actual model calls).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_mapanything_creator.py
git commit -m "feat(pointcloud/feedforward): BaseFeedforwardCreator + MapAnythingCreator with correct inference params"
```

---

## Task 7: feedforward.py — VGGTXCreator

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`
- Create: `tests/pointcloud/test_vggtx_creator.py`

**Key fact:** `run_vggt(image_dir, colmap_dir, use_global_alignment=False)` writes to `colmap_dir/sparse/0/`. Pass `colmap_dir = output_dir / "colmap"`.

- [ ] **Step 1: Write failing tests for VGGTXCreator**

Create `tests/pointcloud/test_vggtx_creator.py`:

```python
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from collab_splats.pointcloud.feedforward import VGGTXCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame


def test_vggtx_defaults():
    c = VGGTXCreator()
    assert c.use_global_alignment is False


def test_vggtx_is_feedforward_creator():
    assert issubclass(VGGTXCreator, BaseFeedforwardCreator)


def test_vggtx_missing_image_dir_raises(tmp_path):
    c = VGGTXCreator()
    with pytest.raises(FileNotFoundError):
        c.reconstruct(tmp_path / "nonexistent", tmp_path / "out")


def test_vggtx_calls_run_vggt_with_correct_colmap_dir(tmp_path):
    """VGGTXCreator must call run_vggt with colmap_dir=output_dir/colmap."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    sparse_dir = output_dir / "colmap" / "sparse" / "0"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("stage.vggt_utils.run_vggt") as mock_run_vggt, \
         patch("pycolmap.Reconstruction", return_value=mock_recon), \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator()
        creator._run_inference(image_dir, output_dir)

        mock_run_vggt.assert_called_once()
        _, kwargs = mock_run_vggt.call_args
        expected_colmap_dir = output_dir / "colmap"
        assert kwargs.get("colmap_dir") == str(expected_colmap_dir) or \
               mock_run_vggt.call_args[0][1] == str(expected_colmap_dir)


def test_vggtx_global_alignment_passed_through(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("stage.vggt_utils.run_vggt") as mock_run_vggt, \
         patch("pycolmap.Reconstruction", return_value=mock_recon), \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator(use_global_alignment=True)
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run_vggt.call_args
        assert kwargs.get("use_global_alignment") is True


@pytest.mark.gpu
def test_vggtx_reconstruct_smoke(tmp_path):
    from PIL import Image as PILImage
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        PILImage.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")

    c = VGGTXCreator()
    result = c.reconstruct(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.frame == CoordinateFrame.NERFSTUDIO
    assert result.world_transform is not None
    assert (tmp_path / "out" / "transforms.json").exists()
    assert (tmp_path / "out" / "colmap" / "sparse" / "0" / "cameras.bin").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_vggtx_creator.py::test_vggtx_defaults -v
```

Expected: ImportError — `VGGTXCreator` not found.

- [ ] **Step 3: Add VGGTXCreator to feedforward.py**

Append to `collab_splats/pointcloud/feedforward.py`:

```python

@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-X feedforward pose + depth estimation.

    use_global_alignment=False (default): per-frame depth only.
    use_global_alignment=True: cross-camera alignment via _run_global_alignment().
    Validate alignment before enabling — see vggt_utils.py notes.
    """

    use_global_alignment: bool = False

    def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
        _add_stage_to_path()
        from stage.vggt_utils import run_vggt

        colmap_dir = output_dir / "colmap"
        run_vggt(
            image_dir=str(image_dir),
            colmap_dir=str(colmap_dir),
            use_global_alignment=self.use_global_alignment,
        )

        sparse_dir = colmap_dir / "sparse" / "0"
        return pycolmap.Reconstruction(str(sparse_dir))
```

- [ ] **Step 4: Run VGGTXCreator tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_vggtx_creator.py -v -k "not smoke"
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_vggtx_creator.py
git commit -m "feat(pointcloud/feedforward): VGGTXCreator with use_global_alignment=False default"
```

---

## Task 8: Registry + __init__.py + cleanup

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `tests/pointcloud/test_registry.py`
- Delete: `stage/feedforward.py`
- Delete: `stage/optical_flow.py`

- [ ] **Step 1: Write failing registry tests**

Replace `tests/pointcloud/test_registry.py` entirely:

```python
import pytest
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.sfm import ColmapCreator, HlocCreator
from collab_splats.pointcloud.feedforward import MapAnythingCreator, VGGTXCreator
from collab_splats.pointcloud.base import BasePointcloudCreator


def test_get_creator_colmap():
    assert get_creator("colmap") is ColmapCreator


def test_get_creator_hloc():
    assert get_creator("hloc") is HlocCreator


def test_get_creator_mapanything():
    assert get_creator("mapanything") is MapAnythingCreator


def test_get_creator_vggtx():
    assert get_creator("vggtx") is VGGTXCreator


def test_get_creator_unknown_raises():
    with pytest.raises(KeyError, match="unknown pointcloud backend"):
        get_creator("nonexistent")


def test_all_creators_are_instantiable():
    for name in ("colmap", "hloc", "mapanything", "vggtx"):
        cls = get_creator(name)
        instance = cls()
        assert isinstance(instance, BasePointcloudCreator)


def test_old_keys_removed():
    with pytest.raises(KeyError):
        get_creator("sfm")
    with pytest.raises(KeyError):
        get_creator("feedforward")
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_registry.py::test_get_creator_colmap -v
```

Expected: KeyError — `"colmap"` not in old registry.

- [ ] **Step 3: Update __init__.py with 4-key registry**

Replace `collab_splats/pointcloud/__init__.py` entirely:

```python
# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult, _colmap_recon_to_result
from .sfm import ColmapCreator, HlocCreator
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}


def get_creator(name: str) -> type[BasePointcloudCreator]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "CoordinateFrame",
    "ColmapCreator",
    "HlocCreator",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "get_creator",
]
```

- [ ] **Step 4: Run registry tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/test_registry.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Delete dead stage files**

```bash
git rm stage/feedforward.py stage/optical_flow.py
```

- [ ] **Step 6: Run full pointcloud test suite**

```bash
cd /workspace/collab-splats && python -m pytest tests/pointcloud/ -v -k "not smoke and not gpu"
```

Expected: ALL PASS. Note any failures and fix before committing.

- [ ] **Step 7: Run full project test suite**

```bash
cd /workspace/collab-splats && python -m pytest tests/ -v -k "not smoke and not gpu"
```

Expected: same pass/fail ratio as before this task (25 pass, 1 skip, pre-existing nerfstudio failures unchanged).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/__init__.py tests/pointcloud/test_registry.py
git commit -m "feat(pointcloud): 4-creator registry (colmap/hloc/mapanything/vggtx); delete stage/feedforward.py + optical_flow.py"
```

---

## Self-Review Checklist

After implementation, verify:

- [ ] `PointcloudResult.frame` is always `CoordinateFrame.NERFSTUDIO` from all four creators
- [ ] `PointcloudResult.world_transform` shape is `(3, 4)` and non-None
- [ ] `output_dir/colmap/sparse/0/cameras.bin` exists after any `reconstruct()` call
- [ ] `output_dir/transforms.json` exists after any `reconstruct()` call
- [ ] `get_creator("sfm")` raises `KeyError` (old key removed)
- [ ] `HlocCreator` source contains no top-level `nerfstudio` import
- [ ] `stage/feedforward.py` and `stage/optical_flow.py` are deleted from the repo
