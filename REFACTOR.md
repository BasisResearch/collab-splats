# Pointcloud Submodule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify pointcloud creation behind `BasePointcloudCreator` with `NerfstudioSfmCreator` and `MapAnythingCreator` backends; `PointcloudResult` carries cam2world OpenGL camera poses.

**Architecture:** `base.py` owns `PointcloudResult`, `BasePointcloudCreator` ABC, and `_colmap_recon_to_result()` helper. Both backends call this helper to normalize COLMAP output to cam2world OpenGL — the convention NerfStudio Splatfacto expects. `Splatter` dispatches via string registry.

**Tech Stack:** pycolmap==3.10, nerfstudio (hloc_utils), MapAnything (stage/), Open3D, numpy

**Spec:** `docs/superpowers/specs/2026-04-16-pointcloud-submodule-design.md`

**Prereq:** Agent 1 semantics refactor complete. Verify: `python -c "from collab_splats.semantics.extractors import BaseExtractor; print('ok')"`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `collab_splats/pointcloud/__init__.py` | Create | Registry: `get_creator(name)` |
| `collab_splats/pointcloud/base.py` | Create | `PointcloudResult`, `BasePointcloudCreator`, `_colmap_recon_to_result()` |
| `collab_splats/pointcloud/utils.py` | Create | `clean_pcd`, `remove_far_points`, `density_filter` |
| `collab_splats/pointcloud/sfm.py` | Create | `NerfstudioSfmCreator` |
| `collab_splats/pointcloud/feedforward.py` | Create | `MapAnythingCreator` |
| `collab_splats/utils/pointcloud.py` | Modify | Re-export shim (no API break) |
| `stage/feedforward.py` | Modify | Add `extractor` param to `feedforward_to_pointcloud()` |
| `collab_splats/wrapper/splatter.py` | Modify | Add `pointcloud_method` config key |
| `tests/pointcloud/` | Create | All tests |

---

## Task 1: Skeleton, base.py, utils.py

**Files:**
- Create: `collab_splats/pointcloud/__init__.py`
- Create: `collab_splats/pointcloud/base.py`
- Create: `collab_splats/pointcloud/utils.py`
- Create: `tests/pointcloud/__init__.py`
- Create: `tests/pointcloud/test_base.py`
- Modify: `collab_splats/utils/pointcloud.py`

- [ ] **Read the existing utils file**

```bash
cat collab_splats/utils/pointcloud.py
```

Note the exact signatures of `clean_pcd()`, `remove_far_points()`, `density_filter()`. You will copy them verbatim into `utils.py`.

- [ ] **Write the failing tests**

```python
# tests/pointcloud/test_base.py
import numpy as np
import pytest
import pycolmap
import open3d as o3d
from collab_splats.pointcloud.base import (
    PointcloudResult,
    BasePointcloudCreator,
    _colmap_recon_to_result,
)
from collab_splats.pointcloud.utils import clean_pcd, remove_far_points, density_filter


def test_result_fields():
    r = PointcloudResult(
        points=np.zeros((10, 3), dtype=np.float32),
        colors=np.zeros((10, 3), dtype=np.uint8),
        confidence=None,
        camera_poses=np.eye(4, dtype=np.float32)[None],
        camera_intrinsics=np.eye(3, dtype=np.float32)[None],
        colmap_reconstruction=None,
    )
    assert r.points.shape == (10, 3)
    assert r.camera_poses.shape == (1, 4, 4)


def test_base_creator_is_abstract():
    with pytest.raises(TypeError):
        BasePointcloudCreator()


def test_colmap_recon_to_result_convention():
    """_colmap_recon_to_result must output cam2world OpenGL poses.
    Build a minimal Reconstruction with identity w2c, verify flipped Y/Z output.
    """
    recon = pycolmap.Reconstruction()

    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=64,
        height=64,
        params=[50.0, 32.0, 32.0],  # f, cx, cy
    )
    cam.camera_id = 1
    recon.add_camera(cam)

    img = pycolmap.Image(name="frame_0000.jpg", camera_id=1)
    img.image_id = 1
    img.cam_from_world = pycolmap.Rigid3d()  # identity w2c
    recon.add_image(img)

    p = pycolmap.Point3D(xyz=np.array([0.0, 0.0, 1.0]))
    p.point3D_id = 1
    p.color = np.array([128, 128, 128], dtype=np.uint8)
    recon.add_point3D(p)

    result = _colmap_recon_to_result(recon)

    assert result.camera_poses is not None
    c2w = result.camera_poses[0]
    assert c2w.shape == (4, 4)
    np.testing.assert_array_almost_equal(c2w[3], [0.0, 0.0, 0.0, 1.0])
    # identity w2c → c2w = I (OpenCV) → flip Y,Z cols → cols 1,2 negated
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 1:3] *= -1
    np.testing.assert_array_almost_equal(c2w, expected, decimal=5)


def test_utils_clean_pcd_returns_tuple():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    result_pcd, indices = clean_pcd(pcd)
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert isinstance(indices, np.ndarray)
```

- [ ] **Run to confirm FAIL**

```bash
pytest tests/pointcloud/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'collab_splats.pointcloud'`

- [ ] **Create `collab_splats/pointcloud/__init__.py`** (empty for now)

```python
# collab_splats/pointcloud/__init__.py
```

- [ ] **Create `collab_splats/pointcloud/base.py`**

```python
# collab_splats/pointcloud/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


@dataclass
class PointcloudResult:
    points: np.ndarray                     # (N, 3) float32, world XYZ
    colors: np.ndarray                     # (N, 3) uint8, RGB
    confidence: np.ndarray | None          # (N,) float32 — MapAnything only
    camera_poses: np.ndarray | None        # (M, 4, 4) float32, cam2world OpenGL
    camera_intrinsics: np.ndarray | None   # (M, 3, 3) float32, K per image
    colmap_reconstruction: Any | None      # pycolmap.Reconstruction — NS interop only


class BasePointcloudCreator(ABC):
    @abstractmethod
    def create(self, image_dir: Path, output_dir: Path, **kwargs) -> PointcloudResult:
        """images in image_dir → sparse pointcloud + camera poses.

        Raises:
            RuntimeError: if reconstruction fails
        """
        ...


def _colmap_recon_to_result(
    recon: pycolmap.Reconstruction,
    confidence: np.ndarray | None = None,
) -> PointcloudResult:
    """Convert pycolmap.Reconstruction to PointcloudResult.

    Camera poses normalized to cam2world OpenGL:
        COLMAP w2c (OpenCV) → inv → c2w (OpenCV) → c2w[:,1:3]*=-1 → c2w (OpenGL)
    Mirrors nerfstudio/data/dataparsers/colmap_dataparser.py:167-169.
    """
    pts3d = recon.points3D
    if not pts3d:
        raise RuntimeError("reconstruction produced 0 points")

    points = np.array([p.xyz for p in pts3d.values()], dtype=np.float32)
    colors = np.array([p.color for p in pts3d.values()], dtype=np.uint8)

    images = sorted(recon.images.values(), key=lambda i: i.image_id)
    cameras = recon.cameras

    poses, intrinsics = [], []
    for img in images:
        R = img.rotation_matrix()
        t = img.tvec
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c).astype(np.float32)
        c2w[:3, 1:3] *= -1  # OpenCV → OpenGL: flip Y and Z
        poses.append(c2w)

        cam = cameras[img.camera_id]
        fx = getattr(cam, "focal_length_x", None) or cam.focal_length
        fy = getattr(cam, "focal_length_y", None) or cam.focal_length
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
    )
```

- [ ] **Create `collab_splats/pointcloud/utils.py`**

Copy the three functions verbatim from `collab_splats/utils/pointcloud.py`:

```bash
cat collab_splats/utils/pointcloud.py
```

Create `collab_splats/pointcloud/utils.py` with those exact function bodies plus this header:

```python
# collab_splats/pointcloud/utils.py
"""Geometric pointcloud utilities. Moved from collab_splats/utils/pointcloud.py."""
```

- [ ] **Replace `collab_splats/utils/pointcloud.py` with re-export shim**

```python
# collab_splats/utils/pointcloud.py
# Re-export shim — do not delete until all callers migrated
from collab_splats.pointcloud.utils import (  # noqa: F401
    clean_pcd,
    remove_far_points,
    density_filter,
)
```

- [ ] **Create `tests/pointcloud/__init__.py`** (empty)

- [ ] **Run tests to confirm PASS**

```bash
pytest tests/pointcloud/test_base.py -v
```

Expected: 4 tests PASS.

- [ ] **Verify shim**

```bash
python -c "from collab_splats.utils.pointcloud import clean_pcd, remove_far_points, density_filter; print('shim OK')"
```

- [ ] **Commit**

```bash
git add collab_splats/pointcloud/ collab_splats/utils/pointcloud.py tests/pointcloud/
git commit -m "refactor: create pointcloud submodule, move geometric utils"
```

---

## Task 2: NerfstudioSfmCreator

**Files:**
- Create: `collab_splats/pointcloud/sfm.py`
- Create: `tests/pointcloud/test_sfm_creator.py`

- [ ] **Write the failing tests**

```python
# tests/pointcloud/test_sfm_creator.py
import numpy as np
import pytest
from PIL import Image
from collab_splats.pointcloud.sfm import NerfstudioSfmCreator
from collab_splats.pointcloud.base import PointcloudResult


@pytest.fixture
def tiny_image_dir(tmp_path):
    for i in range(3):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"frame_{i:04d}.jpg")
    return tmp_path


def test_sfm_defaults():
    c = NerfstudioSfmCreator()
    assert c.use_hloc is True
    assert c.feature_type == "superpoint_aachen"
    assert c.matcher_type == "superglue"
    assert c.num_matched == 50
    assert c.camera_model == "SIMPLE_RADIAL"
    assert c.single_camera is False


def test_sfm_pycolmap_mode():
    c = NerfstudioSfmCreator(use_hloc=False, single_camera=True)
    assert c.use_hloc is False
    assert c.single_camera is True


def test_sfm_create_result_or_graceful_error(tiny_image_dir, tmp_path):
    """Synthetic images likely fail SfM — accept either valid result or RuntimeError."""
    creator = NerfstudioSfmCreator(use_hloc=False, single_camera=True)
    try:
        result = creator.create(tiny_image_dir, tmp_path / "out")
        assert isinstance(result, PointcloudResult)
        assert result.points.shape[1] == 3
        assert result.colors.dtype == np.uint8
        if result.camera_poses is not None:
            assert result.camera_poses.shape[1:] == (4, 4)
    except RuntimeError as e:
        assert "reconstruction" in str(e).lower() or "colmap" in str(e).lower()
```

- [ ] **Run to confirm FAIL**

```bash
pytest tests/pointcloud/test_sfm_creator.py -v
```

- [ ] **Create `collab_splats/pointcloud/sfm.py`**

```python
# collab_splats/pointcloud/sfm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pycolmap

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result


@dataclass
class NerfstudioSfmCreator(BasePointcloudCreator):
    """Pointcloud via NerfStudio SfM (hloc or pycolmap SIFT).

    use_hloc=True: SuperPoint+SuperGlue via nerfstudio hloc_utils.
    use_hloc=False: classical SIFT via pycolmap directly.
    """

    use_hloc: bool = True
    feature_type: str = "superpoint_aachen"
    matcher_type: str = "superglue"
    num_matched: int = 50
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = False

    def create(self, image_dir: Path, output_dir: Path, **kwargs) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recon = self._run_hloc(image_dir, output_dir) if self.use_hloc else self._run_pycolmap(image_dir, output_dir)
        return _colmap_recon_to_result(recon)

    def _run_hloc(self, image_dir: Path, output_dir: Path):
        from nerfstudio.process_data.hloc_utils import run_hloc
        recon = run_hloc(
            image_dir=image_dir,
            output_dir=output_dir,
            sfm_tool="colmap",
            feature_type=self.feature_type,
            matcher_type=self.matcher_type,
            num_matched=self.num_matched,
            verbose=False,
        )
        if recon is None:
            raise RuntimeError("reconstruction failed — hloc returned None")
        return recon

    def _run_pycolmap(self, image_dir: Path, output_dir: Path):
        db_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        pycolmap.extract_features(
            database_path=db_path, image_path=image_dir,
            camera_model=self.camera_model, single_camera=self.single_camera,
        )
        pycolmap.match_exhaustive(db_path)
        reconstructions = pycolmap.incremental_mapping(
            database_path=db_path, image_path=image_dir, output_path=sparse_dir,
        )
        if not reconstructions:
            raise RuntimeError("reconstruction failed — pycolmap incremental_mapping returned no results")
        return reconstructions[0]
```

- [ ] **Run to confirm PASS**

```bash
pytest tests/pointcloud/test_sfm_creator.py -v
```

- [ ] **Commit**

```bash
git add collab_splats/pointcloud/sfm.py tests/pointcloud/test_sfm_creator.py
git commit -m "feat: NerfstudioSfmCreator (hloc + pycolmap backends)"
```

---

## Task 3: MapAnythingCreator

**Files:**
- Create: `collab_splats/pointcloud/feedforward.py`
- Create: `tests/pointcloud/test_mapanything_creator.py`
- Modify: `stage/feedforward.py`

- [ ] **Read `stage/feedforward.py` to find the exact signature**

```bash
grep -n "def feedforward_to_pointcloud\|def prepare_outputs_for_export" stage/feedforward.py
```

- [ ] **Write the failing tests**

```python
# tests/pointcloud/test_mapanything_creator.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from collab_splats.pointcloud.feedforward import MapAnythingCreator
from collab_splats.pointcloud.base import PointcloudResult


def test_mapanything_defaults():
    c = MapAnythingCreator()
    assert c.conf_threshold == 1.5
    assert c.subsample_factor == 1
    assert c.extractor is None


def test_mapanything_accepts_extractor():
    from collab_splats.semantics.extractors import BaseExtractor
    mock = MagicMock(spec=BaseExtractor)
    c = MapAnythingCreator(extractor=mock)
    assert c.extractor is mock


def test_mapanything_missing_dir_raises(tmp_path):
    c = MapAnythingCreator()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        c.create(tmp_path / "nonexistent", tmp_path / "out")


@pytest.mark.gpu
def test_mapanything_create_smoke(tmp_path):
    from PIL import Image
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_dir / f"frame_{i:04d}.jpg")
    c = MapAnythingCreator(subsample_factor=4)
    result = c.create(image_dir, tmp_path / "out")
    assert isinstance(result, PointcloudResult)
    assert result.points.shape[1] == 3
    assert result.confidence is not None
    assert result.camera_poses is not None
    assert result.camera_poses.shape[1:] == (4, 4)
    assert result.camera_intrinsics.shape[1:] == (3, 3)
```

- [ ] **Run non-GPU tests to confirm FAIL**

```bash
pytest tests/pointcloud/test_mapanything_creator.py -v -k "not gpu"
```

- [ ] **Update `stage/feedforward.py`** — add `extractor` param to `feedforward_to_pointcloud()`

Adapt to the actual signature you found. Pattern:

```python
def feedforward_to_pointcloud(
    self,
    conf_threshold: float = 1.5,
    subsample_factor: int = 1,
    feature_extractor_name: str | None = None,
    pca_components: int | None = None,
    extractor=None,  # BaseExtractor | None
):
    if extractor is not None:
        feature_extractor_name = None  # object takes precedence over name
    # rest of body unchanged; pass extractor to underlying feature call
```

- [ ] **Create `collab_splats/pointcloud/feedforward.py`**

```python
# collab_splats/pointcloud/feedforward.py
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result

if TYPE_CHECKING:
    from collab_splats.semantics.extractors import BaseExtractor


@dataclass
class MapAnythingCreator(BasePointcloudCreator):
    """Pointcloud via MapAnything feedforward reconstruction.

    No feature matching — depth + poses estimated directly by the network.
    Unique: populates PointcloudResult.confidence.
    extractor: None = no semantic features, pointcloud only.
    """

    conf_threshold: float = 1.5
    subsample_factor: int = 1
    extractor: BaseExtractor | None = field(default=None, repr=False)

    def create(self, image_dir: Path, output_dir: Path, **kwargs) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from stage.feedforward import Reconstructor
        from stage.mapanything_utils import build_colmap_reconstruction

        reconstructor = Reconstructor(image_dir=image_dir, output_dir=output_dir)
        pcd_dict = reconstructor.feedforward_to_pointcloud(
            conf_threshold=self.conf_threshold,
            subsample_factor=self.subsample_factor,
            extractor=self.extractor,
        )

        raw_outputs = reconstructor.prepare_outputs_for_export()
        recon = build_colmap_reconstruction(raw_outputs)
        pose_result = _colmap_recon_to_result(recon)

        colors = pcd_dict["colors"]
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

        return PointcloudResult(
            points=pcd_dict["points"].astype(np.float32),
            colors=colors,
            confidence=pcd_dict.get("confidences"),
            camera_poses=pose_result.camera_poses,
            camera_intrinsics=pose_result.camera_intrinsics,
            colmap_reconstruction=recon,
        )
```

**Note:** If `build_colmap_reconstruction()` takes different args, read `stage/mapanything_utils.py` and adapt — goal is `raw_outputs → pycolmap.Reconstruction → _colmap_recon_to_result()`.

- [ ] **Run non-GPU tests to confirm PASS**

```bash
pytest tests/pointcloud/test_mapanything_creator.py -v -k "not gpu"
```

- [ ] **Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_mapanything_creator.py stage/feedforward.py
git commit -m "feat: MapAnythingCreator with BaseExtractor injection"
```

---

## Task 4: Registry + Splatter integration

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py`
- Modify: `collab_splats/wrapper/splatter.py`
- Create: `tests/pointcloud/test_registry.py`

- [ ] **Read `collab_splats/wrapper/splatter.py`** — find where pointcloud creation currently happens

- [ ] **Write the failing tests**

```python
# tests/pointcloud/test_registry.py
import pytest
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.sfm import NerfstudioSfmCreator
from collab_splats.pointcloud.feedforward import MapAnythingCreator
from collab_splats.pointcloud.base import BasePointcloudCreator


def test_get_creator_sfm():
    assert get_creator("sfm") is NerfstudioSfmCreator


def test_get_creator_feedforward():
    assert get_creator("feedforward") is MapAnythingCreator


def test_get_creator_unknown_raises():
    with pytest.raises(KeyError, match="unknown pointcloud backend"):
        get_creator("nonexistent")


def test_creator_instantiable():
    assert isinstance(get_creator("sfm")(), BasePointcloudCreator)
```

- [ ] **Run to confirm FAIL**

```bash
pytest tests/pointcloud/test_registry.py -v
```

- [ ] **Implement `collab_splats/pointcloud/__init__.py`**

```python
# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result
from .sfm import NerfstudioSfmCreator
from .feedforward import MapAnythingCreator

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "sfm": NerfstudioSfmCreator,
    "feedforward": MapAnythingCreator,
}


def get_creator(name: str) -> type[BasePointcloudCreator]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = [
    "BasePointcloudCreator", "PointcloudResult", "_colmap_recon_to_result",
    "NerfstudioSfmCreator", "MapAnythingCreator", "get_creator",
]
```

- [ ] **Update `collab_splats/wrapper/splatter.py`**

Add `pointcloud_method: str = "sfm"` to config. Replace hardcoded MapAnything call:

```python
from collab_splats.pointcloud import get_creator

# where pointcloud creation happened:
creator = get_creator(self.config.pointcloud_method)()   # adapt to actual config access
result = creator.create(image_dir=image_dir, output_dir=output_dir)
# result.camera_poses replaces transforms.json reads downstream
```

- [ ] **Run registry tests to confirm PASS**

```bash
pytest tests/pointcloud/test_registry.py -v
```

- [ ] **Run full suite**

```bash
make test
```

- [ ] **Type check**

```bash
mypy collab_splats/pointcloud/
```

- [ ] **Commit**

```bash
git add collab_splats/pointcloud/__init__.py collab_splats/wrapper/splatter.py tests/pointcloud/test_registry.py
git commit -m "feat: pointcloud registry + Splatter.pointcloud_method"
```

---

## Verification

```bash
python -c "
from collab_splats.pointcloud import get_creator
print(get_creator('sfm')())
print(get_creator('feedforward')())
print('registry OK')
"
python -c "from collab_splats.utils.pointcloud import clean_pcd, remove_far_points, density_filter; print('shim OK')"
python -c "from collab_splats.wrapper.splatter import Splatter; print('Splatter OK')"
mypy collab_splats/pointcloud/
make test
```

---

## Adding VGGT-X later (reference only — out of scope)

```python
# collab_splats/pointcloud/vggt.py
@dataclass
class VGGTCreator(BasePointcloudCreator):
    use_global_alignment: bool = True
    scale_factor: float = 2.5

    def create(self, image_dir, output_dir, **kwargs):
        from nerfstudio.process_data.vggt_utils import run_vggt
        run_vggt(image_dir=image_dir, colmap_dir=output_dir / "sparse",
                 use_global_alignment=self.use_global_alignment, scale_factor=self.scale_factor)
        recon = pycolmap.Reconstruction(output_dir / "sparse")
        return _colmap_recon_to_result(recon)
```

Register as `"vggt"` in `__init__.py`. Same `_colmap_recon_to_result()` — no coordinate conversion to write.
