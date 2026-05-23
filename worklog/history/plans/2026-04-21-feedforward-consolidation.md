# Feedforward Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `stage/mapanything_utils.py` and `stage/vggt_utils.py` logic into `collab_splats/pointcloud/`, eliminating the `_add_stage_to_path()` sys.path hack and replacing scattered imports with clean public entry points `run_mapanything()` and `run_vggt()`.

**Architecture:** Each model module (`_mapanything.py`, `_vggt.py`) exposes one public function returning the same 8-tuple; `feedforward.py` holds two shared COLMAP functions used by both `_run_inference()` methods, which become ~15 lines of pure orchestration.

**Tech Stack:** pycolmap, numpy, mapanything (Python package), VGGT-X (Python package)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `collab_splats/pointcloud/utils.py` | Modify | Add `filter_points_by_spatial_extent`, `voxel_downsample_point_cloud` |
| `collab_splats/pointcloud/feedforward.py` | Modify | Add shared build/rescale fns; rewrite both `_run_inference()` methods; remove `_add_stage_to_path` |
| `collab_splats/pointcloud/_mapanything.py` | Create | `run_mapanything()` + internal helpers |
| `collab_splats/pointcloud/_vggt.py` | Create | `run_vggt()` + internal helpers |
| `tests/pointcloud/test_feedforward_shared.py` | Create | Tests for `build_pycolmap_reconstruction` |
| `tests/pointcloud/test_mapanything_module.py` | Create | Tests for `run_mapanything()` contract |
| `tests/pointcloud/test_vggt_module.py` | Create | Tests for `run_vggt()` contract |
| `tests/pointcloud/test_mapanything_creator.py` | Modify | Replace sys.modules hack with clean patches |
| `tests/pointcloud/test_vggtx_creator.py` | Modify | Replace sys.modules hack with clean patches |

---

### Task 1: Add point filter utils to `pointcloud/utils.py`

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Create: `tests/pointcloud/test_pointcloud_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pointcloud/test_pointcloud_utils.py
import numpy as np
import pytest
from collab_splats.pointcloud.utils import filter_points_by_spatial_extent, voxel_downsample_point_cloud


def test_filter_removes_outliers():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1000, 3)).astype(np.float32)
    colors = np.zeros((1000, 3), dtype=np.uint8)
    pts[:10] = 1000.0  # obvious outliers
    out_pts, out_colors = filter_points_by_spatial_extent(pts, colors, percentile_range=(1.0, 99.0))
    assert len(out_pts) < 1000
    assert out_pts.max() < 100.0
    assert out_colors.shape[0] == out_pts.shape[0]


def test_filter_empty():
    pts = np.zeros((0, 3), dtype=np.float32)
    colors = np.zeros((0, 3), dtype=np.uint8)
    out_pts, out_colors = filter_points_by_spatial_extent(pts, colors)
    assert out_pts.shape == (0, 3)


def test_voxel_downsample_reduces_points():
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((10000, 3)).astype(np.float32)
    colors = np.zeros((10000, 3), dtype=np.uint8)
    down_pts, down_colors = voxel_downsample_point_cloud(pts, colors, voxel_fraction=0.05)
    assert len(down_pts) < 10000
    assert down_colors.shape[0] == down_pts.shape[0]
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/pointcloud/test_pointcloud_utils.py -v
```

Expected: `ImportError` — functions don't exist yet.

- [ ] **Step 3: Copy functions into `utils.py`**

Copy verbatim from `stage/mapanything_utils.py`:
- `filter_points_by_spatial_extent`: L313–369
- `voxel_downsample_point_cloud`: L370–471

Add any missing imports at the top of `collab_splats/pointcloud/utils.py`. Replace any `CONSOLE.print(...)` calls with `print(...)` — the stage file uses nerfstudio's `CONSOLE` which isn't available in this module.

- [ ] **Step 4: Run tests to confirm PASS**

```bash
python -m pytest tests/pointcloud/test_pointcloud_utils.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py tests/pointcloud/test_pointcloud_utils.py
git commit -m "feat(pointcloud): add filter_points_by_spatial_extent, voxel_downsample_point_cloud to utils"
```

---

### Task 2: Add shared COLMAP functions to `feedforward.py`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`
- Create: `tests/pointcloud/test_feedforward_shared.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pointcloud/test_feedforward_shared.py
import numpy as np
import pytest
import pycolmap
from collab_splats.pointcloud.feedforward import build_pycolmap_reconstruction


def _make_inputs(n=3, p=50):
    pts3d = np.random.randn(p, 3).astype(np.float32)
    colors = np.random.randint(0, 255, (p, 3), dtype=np.uint8)
    extrinsics = np.tile(np.eye(4), (n, 1, 1)).astype(np.float32)[:, :3, :]  # (n, 3, 4)
    intrinsics = np.tile(
        np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    names = [f"frame_{i:04d}.jpg" for i in range(n)]
    return pts3d, colors, extrinsics, intrinsics, names


def test_camera_image_point_counts():
    pts3d, colors, extrinsics, intrinsics, names = _make_inputs(n=3, p=50)
    recon = build_pycolmap_reconstruction(pts3d, colors, extrinsics, intrinsics, 512, 512, names)
    assert len(recon.cameras) == 3
    assert len(recon.images) == 3
    assert len(recon.points3D) == 50


def test_accepts_4x4_extrinsics():
    pts3d, colors, _, intrinsics, names = _make_inputs(n=2, p=10)
    extrinsics_4x4 = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)  # (2, 4, 4)
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics_4x4, intrinsics, 512, 512, names[:2]
    )
    assert len(recon.cameras) == 2


def test_simple_pinhole_model():
    pts3d, colors, extrinsics, intrinsics, names = _make_inputs(n=2, p=5)
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, 518, 518, names[:2],
        camera_model="SIMPLE_PINHOLE",
    )
    assert len(recon.cameras) == 2
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/pointcloud/test_feedforward_shared.py -v
```

Expected: `ImportError` — `build_pycolmap_reconstruction` doesn't exist yet.

- [ ] **Step 3: Add `build_pycolmap_reconstruction` to `feedforward.py`**

Add `import numpy as np` to the imports at the top. Add this function after the `BaseFeedforwardCreator` class, before `_add_stage_to_path`:

```python
def build_pycolmap_reconstruction(
    pts3d: np.ndarray,
    colors: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: list[str],
    camera_model: str = "PINHOLE",
) -> pycolmap.Reconstruction:
    """Build pycolmap Reconstruction from pointcloud + camera data. No Point2D tracks."""
    recon = pycolmap.Reconstruction()
    exts = extrinsics[:, :3, :] if extrinsics.shape[1] == 4 else extrinsics
    colors_u8 = (
        colors if colors.dtype == np.uint8
        else (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    )

    for xyz, rgb in zip(pts3d, colors_u8):
        recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb)

    for i, name in enumerate(image_names):
        K = intrinsics[i]
        if camera_model == "PINHOLE":
            params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        else:  # SIMPLE_PINHOLE
            params = [(K[0, 0] + K[1, 1]) / 2.0, K[0, 2], K[1, 2]]
        camera = pycolmap.Camera(
            model=camera_model,
            width=image_width,
            height=image_height,
            params=params,
            camera_id=i + 1,
        )
        recon.add_camera(camera)
        R, t = exts[i, :3, :3], exts[i, :3, 3]
        image = pycolmap.Image(
            id=i + 1,
            name=name,
            camera_id=i + 1,
            cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t),
        )
        image.registered = True
        recon.add_image(image)

    return recon
```

- [ ] **Step 4: Add `_rescale_reconstruction_to_original_dimensions` to `feedforward.py`**

Copy verbatim from `stage/mapanything_utils.py:L1006–L1126`. The function signature is:

```python
def _rescale_reconstruction_to_original_dimensions(
    reconstruction: pycolmap.Reconstruction,
    image_paths: list[Path],
    original_image_sizes: np.ndarray,   # (N, 6): [top_x, top_y, crop_r, crop_b, orig_w, orig_h]
    image_size: tuple[int, int],         # (model_w, model_h)
    shared_camera: bool = True,
    shift_point2d_to_original_res: bool = False,
    verbose: bool = False,
) -> pycolmap.Reconstruction:
```

Replace any `CONSOLE.print(...)` in the body with `print(...)`. Add any missing imports (e.g. `copy`, `List`, `Tuple`) at the top of `feedforward.py`.

- [ ] **Step 5: Run tests to confirm PASS**

```bash
python -m pytest tests/pointcloud/test_feedforward_shared.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py tests/pointcloud/test_feedforward_shared.py
git commit -m "feat(feedforward): add build_pycolmap_reconstruction and _rescale_reconstruction_to_original_dimensions"
```

---

### Task 3: Create `_mapanything.py`

**Files:**
- Create: `collab_splats/pointcloud/_mapanything.py`
- Create: `tests/pointcloud/test_mapanything_module.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pointcloud/test_mapanything_module.py
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_run_mapanything_returns_8tuple(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 2, 20

    mock_views = [{"img": np.zeros((3, 64, 64))} for _ in range(n)]
    mock_image_paths = [image_dir / f"frame_{i:04d}.jpg" for i in range(n)]
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)
    mock_extrinsics = np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32)
    mock_intrinsics = np.eye(3)[None].repeat(n, axis=0).astype(np.float32)
    mock_original_coords = np.zeros((n, 6), dtype=np.float32)

    with patch("collab_splats.pointcloud._mapanything._load_mapanything_model", return_value=MagicMock()), \
         patch("collab_splats.pointcloud._mapanything._load_and_preprocess_images",
               return_value=(mock_views, mock_image_paths, mock_original_coords)), \
         patch("collab_splats.pointcloud._mapanything._run_mapanything_inference",
               return_value=[{}] * n), \
         patch("collab_splats.pointcloud._mapanything._collect_pts3d_from_outputs",
               return_value=(mock_pts3d, mock_colors, mock_extrinsics, mock_intrinsics)), \
         patch("collab_splats.pointcloud._mapanything.voxel_downsample_point_cloud",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._mapanything import run_mapanything
        result = run_mapanything(image_dir, "facebook/map-anything")

    assert len(result) == 8
    pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = result
    assert pts3d.shape[1] == 3
    assert colors.shape[1] == 3
    assert extrinsics.shape == (n, 3, 4)
    assert intrinsics.shape == (n, 3, 3)
    assert len(image_paths) == n
    assert original_coords.shape == (n, 6)
    assert isinstance(model_w, int) and isinstance(model_h, int)


def test_run_mapanything_passes_inference_kwargs(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 2, 5

    mock_views = [{"img": np.zeros((3, 64, 64))} for _ in range(n)]
    mock_image_paths = [image_dir / f"frame_{i:04d}.jpg" for i in range(n)]
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)
    mock_extrinsics = np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32)
    mock_intrinsics = np.eye(3)[None].repeat(n, axis=0).astype(np.float32)
    mock_original_coords = np.zeros((n, 6), dtype=np.float32)

    with patch("collab_splats.pointcloud._mapanything._load_mapanything_model", return_value=MagicMock()), \
         patch("collab_splats.pointcloud._mapanything._load_and_preprocess_images",
               return_value=(mock_views, mock_image_paths, mock_original_coords)), \
         patch("collab_splats.pointcloud._mapanything._run_mapanything_inference",
               return_value=[{}] * n) as mock_infer, \
         patch("collab_splats.pointcloud._mapanything._collect_pts3d_from_outputs",
               return_value=(mock_pts3d, mock_colors, mock_extrinsics, mock_intrinsics)), \
         patch("collab_splats.pointcloud._mapanything.voxel_downsample_point_cloud",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._mapanything import run_mapanything
        run_mapanything(image_dir, "facebook/map-anything",
                        confidence_percentile=60.0, minibatch_size=4)

        _, kwargs = mock_infer.call_args
        assert kwargs["confidence_percentile"] == 60.0
        assert kwargs["minibatch_size"] == 4
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/pointcloud/test_mapanything_module.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `collab_splats/pointcloud/_mapanything.py`**

```python
# collab_splats/pointcloud/_mapanything.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import voxel_downsample_point_cloud


def run_mapanything(
    image_dir: Path,
    model_name: str,
    *,
    confidence_percentile: float = 35.0,
    use_multiview_confidence: bool = True,
    minibatch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], np.ndarray, int, int]:
    """Load + preprocess + inference + collect pts3d + voxel downsample.

    Returns:
        (pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h)
    """
    model = _load_mapanything_model(model_name)
    views, image_paths, original_coords = _load_and_preprocess_images(image_dir)
    model_w: int = views[0]["img"].shape[-1]
    model_h: int = views[0]["img"].shape[-2]
    outputs = _run_mapanything_inference(
        model, views,
        confidence_percentile=confidence_percentile,
        use_multiview_confidence=use_multiview_confidence,
        minibatch_size=minibatch_size,
    )
    pts3d, colors, extrinsics, intrinsics = _collect_pts3d_from_outputs(outputs)
    pts3d, colors = voxel_downsample_point_cloud(pts3d, colors)
    return pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h


def _load_mapanything_model(model_name: str) -> Any:
    """Adapt from stage/mapanything_utils.py:L57–123 (load_mapanything_model)."""
    try:
        pass  # mapanything import happens inside
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e
    # Implement by adapting stage/mapanything_utils.py:L57–123.
    # That function loads the model checkpoint from HuggingFace and returns the model object.
    raise NotImplementedError("adapt from stage/mapanything_utils.py:L57–123")


def _load_and_preprocess_images(
    image_dir: Path,
) -> tuple[list[dict], list[Path], np.ndarray]:
    """Load images and compute original_coords array for later rescaling.

    Returns:
        views: preprocessed view dicts (MapAnything format)
        image_paths: sorted list of Path objects
        original_coords: (N, 6) float32 — [0, 0, model_w, model_h, orig_w, orig_h]
    """
    from PIL import Image as PILImage
    try:
        from mapanything.utils.image import load_images
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix in exts)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    views = load_images([str(p) for p in image_paths])
    model_h: int = views[0]["img"].shape[-2]
    model_w: int = views[0]["img"].shape[-1]

    original_coords = np.array(
        [
            [0, 0, model_w, model_h, PILImage.open(p).width, PILImage.open(p).height]
            for p in image_paths
        ],
        dtype=np.float32,
    )
    return views, image_paths, original_coords


def _run_mapanything_inference(
    model: Any,
    views: list[dict],
    *,
    confidence_percentile: float,
    use_multiview_confidence: bool,
    minibatch_size: int,
) -> list[dict]:
    """Run model.infer(). Adapt from stage/mapanything_utils.py:L208–312."""
    import torch
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=minibatch_size,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            use_multiview_confidence=use_multiview_confidence,
            confidence_percentile=confidence_percentile,
        )
    return outputs


def _collect_pts3d_from_outputs(
    outputs: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pts3d, colors, extrinsics, intrinsics from MapAnything output dicts.

    Adapt from stage/mapanything_utils.py:export_predictions_to_colmap_internal L761–830.
    That function collects all_points, all_colors, intrinsics_list, extrinsics_list
    in a per-frame loop over outputs. Stop before the build/write steps.

    Returns:
        pts3d: (P, 3) float32 world-space points (all frames concatenated)
        colors: (P, 3) uint8 RGB
        extrinsics: (N, 3, 4) float32 world2cam [R|t]
        intrinsics: (N, 3, 3) float32 K matrices
    """
    raise NotImplementedError(
        "adapt data collection loop from stage/mapanything_utils.py:L761–830"
    )
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
python -m pytest tests/pointcloud/test_mapanything_module.py -v
```

Expected: PASS — all helpers are mocked so `NotImplementedError` bodies are not reached.

- [ ] **Step 5: Implement the `NotImplementedError` bodies**

- `_load_mapanything_model`: adapt L57–123 of `stage/mapanything_utils.py`
- `_collect_pts3d_from_outputs`: adapt L761–830 of `stage/mapanything_utils.py` (the loop inside `export_predictions_to_colmap_internal` that collects `all_points`, `all_colors`, `intrinsics_list`, `extrinsics_list`; stop before the `build_colmap_reconstruction` call)

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/_mapanything.py tests/pointcloud/test_mapanything_module.py
git commit -m "feat(pointcloud): add _mapanything.py with run_mapanything() entry point"
```

---

### Task 4: Create `_vggt.py`

**Files:**
- Create: `collab_splats/pointcloud/_vggt.py`
- Create: `tests/pointcloud/test_vggt_module.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pointcloud/test_vggt_module.py
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch


def _mock_vggt_data(image_dir, n=3):
    return {
        "extrinsic": np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32),
        "instrinsics_downsampled": np.eye(3)[None].repeat(n, axis=0).astype(np.float32),
        "depth": np.zeros((n, 64, 64), dtype=np.float32),
        "depth_conf": np.ones((n, 64, 64), dtype=np.float32),
        "images": np.zeros((n, 3, 64, 64), dtype=np.float32),
        "image_paths": [image_dir / f"frame_{i:04d}.jpg" for i in range(n)],
        "original_coords": np.zeros((n, 6), dtype=np.float32),
    }


def test_run_vggt_returns_8tuple(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 3, 30
    data = _mock_vggt_data(image_dir, n)
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)
    extrinsics = data["extrinsic"]
    intrinsics = data["instrinsics_downsampled"]

    with patch("collab_splats.pointcloud._vggt._is_vggt_available", return_value=True), \
         patch("collab_splats.pointcloud._vggt._run_vggt_inference", return_value=data), \
         patch("collab_splats.pointcloud._vggt._maybe_run_global_alignment",
               return_value=(extrinsics, intrinsics)), \
         patch("collab_splats.pointcloud._vggt._unproject_and_filter_points",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._vggt import run_vggt
        result = run_vggt(image_dir, tmp_path / "colmap", "facebook/vggt")

    assert len(result) == 8
    pts3d, colors, ext, intr, image_paths, orig, model_w, model_h = result
    assert pts3d.shape[1] == 3
    assert len(image_paths) == n
    assert isinstance(model_w, int) and isinstance(model_h, int)


def test_run_vggt_raises_if_not_installed(tmp_path):
    with patch("collab_splats.pointcloud._vggt._is_vggt_available", return_value=False):
        from collab_splats.pointcloud._vggt import run_vggt
        with pytest.raises(RuntimeError, match="VGGT-X not installed"):
            run_vggt(tmp_path / "imgs", tmp_path / "colmap", "facebook/vggt")


def test_global_alignment_flag_passed(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    n, p = 2, 5
    data = _mock_vggt_data(image_dir, n)
    mock_pts3d = np.zeros((p, 3), dtype=np.float32)
    mock_colors = np.zeros((p, 3), dtype=np.uint8)

    with patch("collab_splats.pointcloud._vggt._is_vggt_available", return_value=True), \
         patch("collab_splats.pointcloud._vggt._run_vggt_inference", return_value=data), \
         patch("collab_splats.pointcloud._vggt._maybe_run_global_alignment",
               return_value=(data["extrinsic"], data["instrinsics_downsampled"])) as mock_align, \
         patch("collab_splats.pointcloud._vggt._unproject_and_filter_points",
               return_value=(mock_pts3d, mock_colors)):

        from collab_splats.pointcloud._vggt import run_vggt
        run_vggt(image_dir, tmp_path / "colmap", "facebook/vggt", use_global_alignment=True)

        _, kwargs = mock_align.call_args
        assert kwargs["use_global_alignment"] is True
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
python -m pytest tests/pointcloud/test_vggt_module.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `collab_splats/pointcloud/_vggt.py`**

```python
# collab_splats/pointcloud/_vggt.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def run_vggt(
    image_dir: Path,
    colmap_dir: Path,
    model_name: str,
    *,
    use_global_alignment: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path], np.ndarray, int, int]:
    """Load + preprocess + inference + unproject + optional global alignment + filter.

    Returns:
        (pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h)
    """
    if not _is_vggt_available():
        raise RuntimeError(
            "VGGT-X not installed. pip install git+https://github.com/Linketic/VGGT-X.git"
        )

    vggt_data = _run_vggt_inference(image_dir, colmap_dir, model_name)

    extrinsic = vggt_data["extrinsic"]
    intrinsic = vggt_data["instrinsics_downsampled"]  # note: typo preserved from stage

    extrinsic, intrinsic = _maybe_run_global_alignment(
        vggt_data, extrinsic, intrinsic, colmap_dir,
        use_global_alignment=use_global_alignment,
    )

    pts3d, colors = _unproject_and_filter_points(
        depth=vggt_data["depth"],
        depth_conf=vggt_data["depth_conf"],
        images=vggt_data["images"],
        extrinsic=extrinsic,
        intrinsic=intrinsic,
    )

    model_h: int = int(vggt_data["depth"].shape[1])
    model_w: int = int(vggt_data["depth"].shape[2])
    return (
        pts3d, colors, extrinsic, intrinsic,
        vggt_data["image_paths"], vggt_data["original_coords"],
        model_w, model_h,
    )


def _is_vggt_available() -> bool:
    """Adapt from stage/vggt_utils.py:L104–111."""
    try:
        import vggt  # noqa: F401
        return True
    except ImportError:
        try:
            import streamvggt  # noqa: F401
            return True
        except ImportError:
            return False


def _run_vggt_inference(image_dir: Path, colmap_dir: Path, model_name: str) -> dict[str, Any]:
    """Load model + preprocess images + run forward pass.

    Adapt from stage/vggt_utils.py:_run_vggt_inference (L974–1119).
    Returns dict with keys: images, extrinsic, instrinsics_downsampled (typo preserved),
    depth, depth_conf, image_paths, original_coords.
    """
    raise NotImplementedError("adapt from stage/vggt_utils.py:_run_vggt_inference L974–1119")


def _maybe_run_global_alignment(
    vggt_data: dict,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    colmap_dir: Path,
    *,
    use_global_alignment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run global alignment if requested, else return cameras unchanged."""
    if not use_global_alignment:
        return extrinsic, intrinsic
    refined_ext, refined_int, _ = _run_global_alignment(
        images=vggt_data["images"],
        image_paths=vggt_data["image_paths"],
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_conf=vggt_data["depth_conf"],
        colmap_dir=colmap_dir,
        depth_map=vggt_data["depth"],
    )
    return refined_ext, refined_int


def _unproject_and_filter_points(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    images: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    conf_threshold: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject depth to world pts3d, filter by confidence threshold.

    Adapt from stage/vggt_utils.py:
    - unproject_depth_map_to_point_map call at run_vggt L170–177
    - _filter_and_prepare_points_for_pycolmap L1888–1942

    Returns (pts3d, colors): (P, 3) float32 and (P, 3) uint8.
    """
    raise NotImplementedError(
        "adapt unproject + filter from stage/vggt_utils.py:L170–177 and L1888–1942"
    )


def _run_global_alignment(
    images: Any,
    image_paths: list[Path],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_conf: np.ndarray,
    colmap_dir: Path,
    depth_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Adapt from stage/vggt_utils.py:_run_global_alignment L1943–2035."""
    raise NotImplementedError("adapt from stage/vggt_utils.py:_run_global_alignment L1943–2035")
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
python -m pytest tests/pointcloud/test_vggt_module.py -v
```

Expected: PASS — internals mocked.

- [ ] **Step 5: Implement the `NotImplementedError` bodies**

- `_run_vggt_inference`: adapt `stage/vggt_utils.py:_run_vggt_inference` L974–1119. Note the typo `instrinsics_downsampled` (double r) — preserve it to match the dict key used in `run_vggt`.
- `_unproject_and_filter_points`: adapt the unproject call (L170–177 of `run_vggt` in stage) + `_filter_and_prepare_points_for_pycolmap` (L1888–1942). The filter function returns `(points3d, points_xyf, points_rgb)` — discard `points_xyf`, return `(points3d, points_rgb)`.
- `_run_global_alignment`: adapt `stage/vggt_utils.py:_run_global_alignment` L1943–2035.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/_vggt.py tests/pointcloud/test_vggt_module.py
git commit -m "feat(pointcloud): add _vggt.py with run_vggt() entry point"
```

---

### Task 5: Rewrite `_run_inference()` methods + update tests

**Files:**
- Modify: `collab_splats/pointcloud/feedforward.py`
- Modify: `tests/pointcloud/test_mapanything_creator.py`
- Modify: `tests/pointcloud/test_vggtx_creator.py`

- [ ] **Step 1: Add `model_name` field to `VGGTXCreator`**

In `feedforward.py`, update the `VGGTXCreator` dataclass:

```python
@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """Pointcloud via VGGT-X feedforward pose + depth estimation."""

    use_global_alignment: bool = False
    model_name: str = "facebook/vggt"
```

- [ ] **Step 2: Rewrite `MapAnythingCreator._run_inference()`**

Replace the existing body entirely:

```python
def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
    from ._mapanything import run_mapanything

    pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = run_mapanything(
        image_dir, self.model_name,
        confidence_percentile=self.confidence_percentile,
        use_multiview_confidence=self.use_multiview_confidence,
        minibatch_size=self.minibatch_size,
    )
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, model_w, model_h,
        [p.name for p in image_paths],
    )
    recon = _rescale_reconstruction_to_original_dimensions(
        recon, image_paths, original_coords, (model_w, model_h)
    )
    sparse_dir = output_dir / "colmap" / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    recon.write_binary(str(sparse_dir))
    return pycolmap.Reconstruction(str(sparse_dir))
```

- [ ] **Step 3: Rewrite `VGGTXCreator._run_inference()`**

Replace the existing body entirely:

```python
def _run_inference(self, image_dir: Path, output_dir: Path) -> pycolmap.Reconstruction:
    from ._vggt import run_vggt

    colmap_dir = output_dir / "colmap"
    pts3d, colors, extrinsics, intrinsics, image_paths, original_coords, model_w, model_h = run_vggt(
        image_dir, colmap_dir, self.model_name,
        use_global_alignment=self.use_global_alignment,
    )
    recon = build_pycolmap_reconstruction(
        pts3d, colors, extrinsics, intrinsics, model_w, model_h,
        [p.name for p in image_paths],
        camera_model="SIMPLE_PINHOLE",
    )
    recon = _rescale_reconstruction_to_original_dimensions(
        recon, image_paths, original_coords, (model_w, model_h)
    )
    sparse_dir = colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    recon.write_binary(str(sparse_dir))
    return pycolmap.Reconstruction(str(sparse_dir))
```

- [ ] **Step 4: Remove `_add_stage_to_path` and `import sys` from `feedforward.py`**

Delete `import sys` (line 4) and the entire `_add_stage_to_path()` function (4 lines).

- [ ] **Step 5: Update `test_mapanything_creator.py`**

Replace `test_mapanything_run_inference_passes_inference_params` (the test using `sys.modules`, L27–71) with:

```python
def test_mapanything_run_inference_passes_inference_params(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    n, p = 2, 10

    mock_8tuple = (
        np.zeros((p, 3), dtype=np.float32),
        np.zeros((p, 3), dtype=np.uint8),
        np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32),
        np.eye(3)[None].repeat(n, axis=0).astype(np.float32),
        [image_dir / f"frame_{i:04d}.jpg" for i in range(n)],
        np.zeros((n, 6), dtype=np.float32),
        518, 336,
    )
    mock_recon = MagicMock()
    mock_recon.images = {}
    mock_recon.cameras = {}
    mock_recon.points3D = {}

    with patch("collab_splats.pointcloud._mapanything.run_mapanything",
               return_value=mock_8tuple) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=mock_recon), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=mock_recon), \
         patch("pycolmap.Reconstruction", return_value=mock_recon), \
         patch.object(MapAnythingCreator, "_write_transforms"):
        creator = MapAnythingCreator(confidence_percentile=50.0, minibatch_size=2)
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run.call_args
        assert kwargs["confidence_percentile"] == 50.0
        assert kwargs["minibatch_size"] == 2
        assert kwargs["use_multiview_confidence"] is True
```

Remove `import sys` from the test file imports.

- [ ] **Step 6: Update `test_vggtx_creator.py`**

Replace `test_vggtx_calls_run_vggt_with_correct_colmap_dir` (L24–58) and `test_vggtx_global_alignment_passed_through` (L61–89) with:

```python
def _run_vggt_mock_8tuple(image_dir, n=2, p=10):
    return (
        np.zeros((p, 3), dtype=np.float32),
        np.zeros((p, 3), dtype=np.uint8),
        np.eye(4)[None, :3, :].repeat(n, axis=0).astype(np.float32),
        np.eye(3)[None].repeat(n, axis=0).astype(np.float32),
        [image_dir / f"frame_{i:04d}.jpg" for i in range(n)],
        np.zeros((n, 6), dtype=np.float32),
        518, 518,
    )


def _mock_recon():
    m = MagicMock()
    m.images = {}
    m.cameras = {}
    m.points3D = {}
    return m


def test_vggtx_calls_run_vggt_with_correct_colmap_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("collab_splats.pointcloud._vggt.run_vggt",
               return_value=_run_vggt_mock_8tuple(image_dir)) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=_mock_recon()), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=_mock_recon()), \
         patch("pycolmap.Reconstruction", return_value=_mock_recon()), \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator()
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run.call_args
        assert kwargs["colmap_dir"] == output_dir / "colmap"
        assert kwargs["use_global_alignment"] is False


def test_vggtx_global_alignment_passed_through(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("collab_splats.pointcloud._vggt.run_vggt",
               return_value=_run_vggt_mock_8tuple(image_dir)) as mock_run, \
         patch("collab_splats.pointcloud.feedforward.build_pycolmap_reconstruction",
               return_value=_mock_recon()), \
         patch("collab_splats.pointcloud.feedforward._rescale_reconstruction_to_original_dimensions",
               return_value=_mock_recon()), \
         patch("pycolmap.Reconstruction", return_value=_mock_recon()), \
         patch.object(VGGTXCreator, "_write_transforms"):
        creator = VGGTXCreator(use_global_alignment=True)
        creator._run_inference(image_dir, output_dir)

        _, kwargs = mock_run.call_args
        assert kwargs["use_global_alignment"] is True
```

Also update `test_vggtx_defaults` to assert the new field:

```python
def test_vggtx_defaults():
    c = VGGTXCreator()
    assert c.use_global_alignment is False
    assert c.model_name == "facebook/vggt"
```

Remove `import sys` from the test file imports.

- [ ] **Step 7: Run all non-GPU tests**

```bash
python -m pytest tests/pointcloud/ -v -k "not smoke and not gpu"
```

Expected: all PASS. Fix any failures before proceeding.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward.py \
        tests/pointcloud/test_mapanything_creator.py \
        tests/pointcloud/test_vggtx_creator.py
git commit -m "refactor(feedforward): rewrite _run_inference() to use run_mapanything/run_vggt; remove _add_stage_to_path"
```

---

### Task 6: Verify

- [ ] **Step 1: Import check**

```bash
python -c "
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator, VGGTXCreator, build_pycolmap_reconstruction
)
from collab_splats.pointcloud._mapanything import run_mapanything
from collab_splats.pointcloud._vggt import run_vggt
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 2: Full non-GPU test suite**

```bash
python -m pytest tests/pointcloud/ -v -k "not smoke and not gpu"
```

Expected: all PASS.

- [ ] **Step 3: Confirm no stage imports remain in package**

```bash
grep -r "_add_stage_to_path\|stage\.mapanything_utils\|stage\.vggt_utils" collab_splats/ tests/
```

Expected: 0 hits.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify feedforward consolidation complete — no stage imports in package"
```
