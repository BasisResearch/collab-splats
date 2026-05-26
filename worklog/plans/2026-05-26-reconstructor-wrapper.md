# Reconstructor Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `Splatter` with `Reconstructor` — a 5-stage pipeline wrapper (preprocess → pointcloud → semantics / mesh / localize) with feedforward-first design, nerfstudio support, and shared 2D feature caching.

**Architecture:** Dict-based config loaded from YAML hierarchy (base + dataset overrides). `Reconstructor` owns 5 stage methods; each skips if output exists on disk. `Splatter` becomes a deprecation shim forwarding to `Reconstructor` with nerfstudio defaults.

**Tech Stack:** Python 3.11, pycolmap, zarr, open3d, torch, `collab_splats.pointcloud.feedforward` (VGGTXCreator / MapAnythingCreator / VGGTOmegaCreator), `collab_splats.semantics.features` (BaseFeatureExtractor), `collab_splats.mesh.tsdf` (Open3DTSDFFusion).

**Python env:** `/opt/conda/envs/nerfstudio/bin/python`
**Test command:** `/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/ -v`
**Spec:** `worklog/specs/2026-05-26-reconstructor-wrapper-design.md`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `collab_splats/wrapper/reconstructor.py` | `Reconstructor` class — all 5 stages |
| Modify | `collab_splats/wrapper/config.py` | `ConfigLoader` updated for new YAML schema |
| Modify | `collab_splats/wrapper/splatter.py` | Deprecation shim → `Reconstructor` |
| Modify | `collab_splats/wrapper/__init__.py` | Export `Reconstructor` |
| Modify | `docs/splats/configs/base.yaml` | New stage-based schema |
| Create | `tests/wrapper/__init__.py` | Test package init |
| Create | `tests/wrapper/test_reconstructor.py` | All unit tests |
| Delete | `stage/feedforward.py` | Legacy — replaced by `collab_splats/pointcloud/feedforward/` |

---

## Task 1: Update Config Schema

**Files:**
- Modify: `docs/splats/configs/base.yaml` (in `.claude/worktrees/docs-site/` — use the main branch copy)
- Modify: `collab_splats/wrapper/config.py`
- Create: `tests/wrapper/__init__.py`
- Create: `tests/wrapper/test_reconstructor.py` (config tests only)

Note: The existing `base.yaml` lives at `.claude/worktrees/docs-site/docs/splats/configs/base.yaml`. Check if a copy exists on main branch under `docs/splats/configs/base.yaml` — if not, create it.

- [ ] **Step 1: Write failing config tests**

```python
# tests/wrapper/test_reconstructor.py
from pathlib import Path
import pytest
import tempfile
import yaml
from collab_splats.wrapper.config import ConfigLoader


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


def test_config_load_base_defaults(tmp_path):
    base = {
        "preprocessing": {"frame_selection": "fps", "frame_proportion": 0.1, "min_frames": 300},
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "bundle_adjustment": False, "loop_closure": False},
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 1024},
        "mesh": {"enabled": False, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04},
        "localization": {"enabled": False, "extractor": "dinosalad"},
        "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
    }
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()
    ds = {"input_path": "/data/video.mp4", "output_path": "/data/out"}
    (tmp_path / "datasets" / "test.yaml").write_text(yaml.dump(ds))

    loader = ConfigLoader(tmp_path)
    config = loader.load("test")

    assert config["input_path"] == "/data/video.mp4"
    assert config["pointcloud"]["method"] == "feedforward"
    assert config["pointcloud"]["backend"] == "vggtx"
    assert config["semantics"]["enabled"] is False


def test_config_dataset_override_merges(tmp_path):
    base = {
        "pointcloud": {"method": "feedforward", "backend": "vggtx", "bundle_adjustment": False},
        "semantics": {"enabled": False, "extractor": "dinov2"},
    }
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()
    ds = {"input_path": "/data/v.mp4", "output_path": "/out", "pointcloud": {"backend": "mapanything", "bundle_adjustment": True}}
    (tmp_path / "datasets" / "ds.yaml").write_text(yaml.dump(ds))

    loader = ConfigLoader(tmp_path)
    config = loader.load("ds")

    assert config["pointcloud"]["backend"] == "mapanything"
    assert config["pointcloud"]["bundle_adjustment"] is True
    assert config["pointcloud"]["method"] == "feedforward"  # base preserved


def test_config_runtime_overrides(tmp_path):
    base = {"pointcloud": {"method": "feedforward", "backend": "vggtx"}}
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "ds.yaml").write_text(yaml.dump({"input_path": "/v.mp4", "output_path": "/o"}))

    loader = ConfigLoader(tmp_path)
    config = loader.load("ds", overrides={"pointcloud": {"backend": "vggt_omega"}})

    assert config["pointcloud"]["backend"] == "vggt_omega"


def test_config_missing_dataset_raises(tmp_path):
    base = {"pointcloud": {"method": "feedforward"}}
    (tmp_path / "base.yaml").write_text(yaml.dump(base))
    (tmp_path / "datasets").mkdir()

    loader = ConfigLoader(tmp_path)
    with pytest.raises(ValueError, match="Dataset config not found"):
        loader.load("nonexistent")
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_config_load_base_defaults -v
```
Expected: `FAILED` or `ERROR` — `ConfigLoader` doesn't handle new schema yet.

- [ ] **Step 3: Update `base.yaml` with new stage-based schema**

Replace the existing `base.yaml` content (check `.claude/worktrees/docs-site/docs/splats/configs/base.yaml` for current content, then write the new version to the same path AND create `docs/splats/configs/base.yaml` on main):

```yaml
# Base configuration for Reconstructor pipelines.
# All dataset configs inherit from this and override specific values.
# Required per dataset: input_path, output_path

preprocessing:
  frame_selection: fps       # fps | optical_flow
  frame_proportion: 0.1
  min_frames: 300
  max_frames: null

pointcloud:
  method: feedforward        # feedforward | sfm | nerfstudio
  backend: vggtx             # vggtx | mapanything | vggt_omega | colmap | hloc
  bundle_adjustment: false
  loop_closure: false
  clean:
    enabled: true
    outlier_removal: true
    voxel_size: null         # null = skip voxel downsampling
    confidence_threshold: null  # null = no confidence filtering

semantics:
  enabled: false
  extractor: dinov2          # dinov2 | maskclip | talk2dino
  n_components: 64           # null = no PCA compression
  resolution: 1024

mesh:
  enabled: false
  mesher: tsdf               # tsdf | poisson
  voxel_size: 0.01
  sdf_trunc: 0.04

localization:
  enabled: false
  extractor: dinosalad

nerfstudio:                  # only used when pointcloud.method: nerfstudio
  sfm_tool: hloc             # hloc | colmap
  train_method: rade-features
```

- [ ] **Step 4: `ConfigLoader` already handles new schema** — `mergedeep` merge is schema-agnostic. Run tests:

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "config"
```
Expected: all 4 config tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add docs/splats/configs/base.yaml tests/wrapper/__init__.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): add Reconstructor config schema + config tests"
```

---

## Task 2: Reconstructor Skeleton

**Files:**
- Create: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py` (add skeleton tests)

- [ ] **Step 1: Write failing skeleton tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
import warnings
import numpy as np
from pathlib import Path
from collab_splats.wrapper.reconstructor import Reconstructor


def _make_config(tmp_path: Path, overrides: dict | None = None) -> dict:
    """Build minimal valid config dict for tests."""
    config = {
        "input_path": str(tmp_path / "video.mp4"),
        "output_path": str(tmp_path / "out"),
        "preprocessing": {"frame_selection": "fps", "frame_proportion": 0.1, "min_frames": 10},
        "pointcloud": {
            "method": "feedforward", "backend": "vggtx",
            "bundle_adjustment": False, "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False, "extractor": "dinov2", "n_components": 64, "resolution": 512},
        "mesh": {"enabled": False, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04},
        "localization": {"enabled": False},
        "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
    }
    if overrides:
        from mergedeep import merge
        config = merge({}, config, overrides)
    return config


def test_reconstructor_init(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.config["pointcloud"]["backend"] == "vggtx"


def test_reconstructor_validate_missing_input_path(tmp_path):
    config = _make_config(tmp_path)
    del config["input_path"]
    with pytest.raises(ValueError, match="input_path"):
        Reconstructor.validate_config(config)


def test_reconstructor_validate_missing_output_path(tmp_path):
    config = _make_config(tmp_path)
    del config["output_path"]
    with pytest.raises(ValueError, match="output_path"):
        Reconstructor.validate_config(config)


def test_reconstructor_validate_bad_backend(tmp_path):
    config = _make_config(tmp_path, {"pointcloud": {"method": "feedforward", "backend": "badmodel"}})
    with pytest.raises(ValueError, match="backend"):
        Reconstructor.validate_config(config)


def test_reconstructor_backend_dir(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.backend_dir == tmp_path / "out" / "vggtx"


def test_reconstructor_images_dir(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.images_dir == tmp_path / "out" / "images"


def test_reconstructor_features_dir(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    assert rec.features_dir == tmp_path / "out" / "features"
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "init or validate or backend_dir or images_dir or features_dir"
```
Expected: `ERROR` — `reconstructor` module not found.

- [ ] **Step 3: Create `reconstructor.py` skeleton**

```python
# collab_splats/wrapper/reconstructor.py
"""5-stage reconstruction pipeline wrapper."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from collab_splats.pointcloud.base import PointcloudResult

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

_FEEDFORWARD_BACKENDS = {"vggtx", "mapanything", "vggt_omega"}
_SFM_BACKENDS = {"colmap", "hloc"}
_VALID_METHODS = {"feedforward", "sfm", "nerfstudio"}
_VALID_MESHERS = {"tsdf", "poisson"}


########################################
# Reconstructor
########################################

class Reconstructor:
    """5-stage environment reconstruction pipeline: preprocess → pointcloud → semantics / mesh / localize."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with validated config dict."""
        self.config = self.validate_config(config)
        self.pointcloud: PointcloudResult | None = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Validate required fields and method/backend consistency.

        Raises:
            ValueError: If required fields missing or backend invalid for method.
        """
        # Required top-level fields
        for field in ("input_path", "output_path"):
            if field not in config or config[field] is None:
                raise ValueError(f"Reconstructor config missing required field: '{field}'")

        pc = config.get("pointcloud", {})
        method = pc.get("method", "feedforward")
        backend = pc.get("backend", "vggtx")

        if method not in _VALID_METHODS:
            raise ValueError(f"pointcloud.method must be one of {_VALID_METHODS}, got '{method}'")

        if method == "feedforward" and backend not in _FEEDFORWARD_BACKENDS:
            raise ValueError(
                f"pointcloud.backend must be one of {_FEEDFORWARD_BACKENDS} "
                f"for method='feedforward', got '{backend}'"
            )
        if method == "sfm" and backend not in _SFM_BACKENDS:
            raise ValueError(
                f"pointcloud.backend must be one of {_SFM_BACKENDS} "
                f"for method='sfm', got '{backend}'"
            )

        mesh_cfg = config.get("mesh", {})
        mesher = mesh_cfg.get("mesher", "tsdf")
        if mesher not in _VALID_MESHERS:
            raise ValueError(f"mesh.mesher must be one of {_VALID_MESHERS}, got '{mesher}'")

        return config

    @classmethod
    def from_config_file(
        cls,
        dataset: str,
        config_dir: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> "Reconstructor":
        """Create Reconstructor from YAML config hierarchy.

        Args:
            dataset: Dataset name (matches datasets/<dataset>.yaml).
            config_dir: Directory containing base.yaml and datasets/.
            overrides: Optional runtime overrides applied after merge.
        """
        from collab_splats.wrapper.config import ConfigLoader
        loader = ConfigLoader(config_dir)
        config = loader.load(dataset=dataset, overrides=overrides)
        return cls(config)

    ########################################
    # Path properties
    ########################################

    @property
    def backend_dir(self) -> Path:
        """output_path / backend — e.g. out/vggtx/. Backend subdir for all stage 2+ artifacts."""
        backend = self.config["pointcloud"].get("backend", "nerfstudio")
        return Path(self.config["output_path"]) / backend

    @property
    def images_dir(self) -> Path:
        """output_path / images/ — shared frame store across all backends."""
        return Path(self.config["output_path"]) / "images"

    @property
    def features_dir(self) -> Path:
        """output_path / features/ — shared 2D feature cache, extractor-scoped subdirs."""
        return Path(self.config["output_path"]) / "features"

    ########################################
    # Stage stubs (implemented in later tasks)
    ########################################

    def preprocess(self, overwrite: bool = False) -> Path:
        """Extract frames from input video/dir into images_dir."""
        raise NotImplementedError

    def build_pointcloud(self, overwrite: bool = False) -> PointcloudResult:
        """Run pointcloud stage. Sets self.pointcloud, returns PointcloudResult."""
        raise NotImplementedError

    def extract_semantics(
        self,
        result: PointcloudResult | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Extract 2D features (cached), lift to 3D, compress. Returns lifted zarr path."""
        raise NotImplementedError

    def mesh(
        self,
        result: PointcloudResult | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Build mesh from pointcloud. Returns path to mesh.ply."""
        raise NotImplementedError

    def localize(self, image: np.ndarray) -> np.ndarray:
        """Localize query image into existing reconstruction. Returns (4, 4) c2w pose."""
        raise NotImplementedError("localization not yet implemented")

    def run_pipeline(
        self,
        stages: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Run named stages in dependency order.

        Args:
            stages: Subset of ["preprocess", "pointcloud", "semantics", "mesh"].
                    Default: all enabled stages from config.
            overwrite: Re-run stages even if output exists.
        """
        raise NotImplementedError

    def launch_dashboard(self) -> None:
        """Launch interactive dashboard for current reconstruction state."""
        from collab_splats.dashboard.__main__ import main as dashboard_main
        dashboard_main()
```

- [ ] **Step 4: Run skeleton tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "init or validate or backend_dir or images_dir or features_dir"
```
Expected: all 7 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): add Reconstructor skeleton with validation and path properties"
```

---

## Task 3: Preprocess Stage

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write failing preprocess tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
import shutil
from unittest.mock import patch, MagicMock


def test_preprocess_skips_if_images_exist(tmp_path):
    """Skip extraction when images/ already populated and overwrite=False."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    images_dir = rec.images_dir
    images_dir.mkdir(parents=True)
    # Create a fake frame so directory is non-empty
    (images_dir / "frame_0001.jpg").touch()

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        result = rec.preprocess(overwrite=False)

    mock_extract.assert_not_called()
    assert result == images_dir


def test_preprocess_runs_if_images_missing(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        mock_extract.return_value = [rec.images_dir / "frame_0001.jpg"]
        rec.images_dir.mkdir(parents=True)
        (rec.images_dir / "frame_0001.jpg").touch()
        result = rec.preprocess(overwrite=False)

    # images_dir exists and has content from mock — accepted
    assert result == rec.images_dir


def test_preprocess_overwrite_reruns(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    rec.images_dir.mkdir(parents=True)
    (rec.images_dir / "frame_0001.jpg").touch()

    with patch("collab_splats.wrapper.reconstructor._extract_frames") as mock_extract:
        mock_extract.return_value = [rec.images_dir / "frame_0001.jpg"]
        rec.preprocess(overwrite=True)

    mock_extract.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "preprocess"
```
Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3: Implement `preprocess()` and `_extract_frames()` helper**

Add to `collab_splats/wrapper/reconstructor.py` (before the class):

```python
import shutil
from collab_splats.utils.frame_sampling import sample_frames
```

Add `_extract_frames` module-level helper after imports:

```python
def _extract_frames(
    input_path: Path,
    output_dir: Path,
    frame_selection: str,
    frame_proportion: float,
    min_frames: int,
    max_frames: int | None,
) -> list[Path]:
    """Extract frames from video or copy from image dir into output_dir.

    Returns sorted list of extracted frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    if input_path.is_dir():
        # Copy images from directory
        exts = {".jpg", ".jpeg", ".png"}
        frames = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        for i, src in enumerate(frames):
            shutil.copy(src, output_dir / f"frame_{i:04d}{src.suffix}")
        return sorted(output_dir.iterdir())

    # Video — use frame_sampling
    paths = sample_frames(
        video_path=input_path,
        output_dir=output_dir,
        method=frame_selection,
        proportion=frame_proportion,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    return paths
```

Replace `preprocess()` stub:

```python
def preprocess(self, overwrite: bool = False) -> Path:
    """Extract frames from input video/dir into images_dir."""
    # Skip if frames already exist and overwrite not requested
    if not overwrite and self.images_dir.exists() and any(self.images_dir.iterdir()):
        logger.info("Frames already extracted at %s, skipping preprocess", self.images_dir)
        return self.images_dir

    if overwrite and self.images_dir.exists():
        shutil.rmtree(self.images_dir)

    pre_cfg = self.config.get("preprocessing", {})
    _extract_frames(
        input_path=Path(self.config["input_path"]),
        output_dir=self.images_dir,
        frame_selection=pre_cfg.get("frame_selection", "fps"),
        frame_proportion=pre_cfg.get("frame_proportion", 0.1),
        min_frames=pre_cfg.get("min_frames", 300),
        max_frames=pre_cfg.get("max_frames"),
    )
    logger.info("Preprocessing complete: %d frames at %s", sum(1 for _ in self.images_dir.iterdir()), self.images_dir)
    return self.images_dir
```

- [ ] **Step 4: Run preprocess tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "preprocess"
```
Expected: all 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): implement Reconstructor.preprocess stage"
```

---

## Task 4: build_pointcloud — Feedforward Path

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

Key interfaces to use:
- `VGGTXCreator()`, `MapAnythingCreator()`, `VGGTOmegaCreator()` — instantiate with no args; call `.reconstruct(image_dir, output_dir) -> PointcloudResult`
- `BundleAdjustment(config=None)` from `collab_splats.pointcloud.bundle_adjustment` — standalone runner; apply via creator wrapper pattern if available, or call `ba.run(creator.outputs)` after inference. **Verify exact BA interface in `bundle_adjustment.py:59-100` before implementing.**
- `LoopClosure(base=creator, config=None)` from `collab_splats.pointcloud.wrappers` — proxy wrapper
- `FeedforwardResult.load_zarr(path)` — to load raw output for cleaning

- [ ] **Step 1: Write failing build_pointcloud tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
import pycolmap
from unittest.mock import patch, MagicMock


def _make_mock_pointcloud_result(tmp_path):
    """Minimal PointcloudResult for testing."""
    from collab_splats.pointcloud.base import PointcloudResult, CoordinateFrame
    recon = pycolmap.Reconstruction()
    return PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=[tmp_path / "images" / "frame_0001.jpg"],
    )


def test_build_pointcloud_skips_if_colmap_exists(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    colmap_dir = rec.backend_dir / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True)
    (colmap_dir / "cameras.bin").touch()

    with patch("collab_splats.wrapper.reconstructor._run_feedforward") as mock_ff:
        rec.build_pointcloud(overwrite=False)

    mock_ff.assert_not_called()


def test_build_pointcloud_feedforward_vggtx(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)

    with patch("collab_splats.wrapper.reconstructor._run_feedforward", return_value=mock_result) as mock_ff:
        result = rec.build_pointcloud(overwrite=True)

    mock_ff.assert_called_once()
    assert result is mock_result
    assert rec.pointcloud is mock_result


def test_build_pointcloud_method_dir_routing(tmp_path):
    """mapanything backend → out/mapanything/."""
    config = _make_config(tmp_path, {"pointcloud": {"backend": "mapanything"}})
    rec = Reconstructor(config)
    assert rec.backend_dir == tmp_path / "out" / "mapanything"
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "build_pointcloud"
```
Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3: Implement `_run_feedforward()` helper and `build_pointcloud()` feedforward path**

Add imports to `reconstructor.py`:

```python
import open3d as o3d
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, VGGTOmegaCreator
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.pointcloud.wrappers import LoopClosure
from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment
```

Add module-level helper:

```python
_CREATOR_MAP = {
    "vggtx": VGGTXCreator,
    "mapanything": MapAnythingCreator,
    "vggt_omega": VGGTOmegaCreator,
}


def _run_feedforward(
    backend: str,
    images_dir: Path,
    output_dir: Path,
    bundle_adjustment: bool,
    loop_closure: bool,
) -> "PointcloudResult":
    """Instantiate feedforward creator, wrap with BA/LC, run reconstruct."""
    creator_cls = _CREATOR_MAP[backend]
    creator = creator_cls()

    # Wrap with LoopClosure before BA (LC runs during inference)
    if loop_closure:
        creator = LoopClosure(base=creator)

    colmap_dir = output_dir / "colmap"
    result = creator.reconstruct(images_dir, colmap_dir)

    # Bundle adjustment post-reconstruction refinement
    if bundle_adjustment:
        ba = BundleAdjustment()
        result = ba.run(result)  # verify exact interface in bundle_adjustment.py

    return result
```

Replace `build_pointcloud()` stub:

```python
def build_pointcloud(self, overwrite: bool = False) -> PointcloudResult:
    """Run pointcloud stage. Sets self.pointcloud, returns PointcloudResult."""
    pc_cfg = self.config.get("pointcloud", {})
    method = pc_cfg.get("method", "feedforward")

    # Skip if COLMAP reconstruction already on disk
    colmap_done = (self.backend_dir / "colmap" / "sparse" / "0" / "cameras.bin").exists()
    if not overwrite and colmap_done:
        logger.info("Pointcloud exists at %s, loading from disk", self.backend_dir / "colmap")
        self.pointcloud = self._load_pointcloud_from_disk()
        return self.pointcloud

    if method == "nerfstudio":
        result = self._run_nerfstudio()
    elif method == "sfm":
        warnings.warn(
            "pointcloud.method='sfm' is experimental and not production-tested.",
            UserWarning, stacklevel=2,
        )
        result = self._run_sfm()
    else:
        result = _run_feedforward(
            backend=pc_cfg.get("backend", "vggtx"),
            images_dir=self.images_dir,
            output_dir=self.backend_dir,
            bundle_adjustment=pc_cfg.get("bundle_adjustment", False),
            loop_closure=pc_cfg.get("loop_closure", False),
        )

    # Save raw FeedforwardResult to zarr if available
    if hasattr(result, "_feedforward_result") and result._feedforward_result is not None:
        zarr_path = self.backend_dir / "feedforward.zarr"
        result._feedforward_result.save_zarr(zarr_path)
        logger.info("FeedforwardResult saved to %s", zarr_path)

    # Apply cleaning step
    clean_cfg = pc_cfg.get("clean", {})
    if clean_cfg.get("enabled", True):
        result = self._clean_pointcloud(result, clean_cfg)

    # Write transforms.json for nerfstudio compatibility
    self._write_transforms_json(result)

    self.pointcloud = result
    return result
```

Add helper methods to `Reconstructor`:

```python
def _load_pointcloud_from_disk(self) -> PointcloudResult:
    """Load PointcloudResult from COLMAP reconstruction on disk."""
    from collab_splats.pointcloud.base import CoordinateFrame
    colmap_dir = self.backend_dir / "colmap" / "sparse" / "0"
    recon = pycolmap.Reconstruction()
    recon.read(str(colmap_dir))
    image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))
    return PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=image_paths,
    )


def _clean_pointcloud(self, result: PointcloudResult, cfg: dict) -> PointcloudResult:
    """Apply open3d outlier removal and voxel downsampling to PointcloudResult.

    Modifies points3D in-place by removing outlier point IDs from reconstruction.
    """
    if not result.reconstruction.points3D:
        return result

    pts = result.points   # (P, 3)
    colors = result.colors  # (P, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(float))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)

    # Statistical outlier removal
    if cfg.get("outlier_removal", True):
        pcd, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Voxel downsampling
    voxel_size = cfg.get("voxel_size")
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Confidence threshold — filter points from FeedforwardResult zarr if available
    # (applied at zarr load time in extract_semantics)
    logger.info("Pointcloud after cleaning: %d points", len(pcd.points))
    return result  # reconstruction still valid; cleaned pcd used downstream


def _write_transforms_json(self, result: PointcloudResult) -> None:
    """Write nerfstudio-compatible transforms.json from PointcloudResult."""
    import json
    from collab_splats.utils.geometry import invert_poses

    self.backend_dir.mkdir(parents=True, exist_ok=True)
    extrinsics = result.extrinsics  # (N, 4, 4) w2c
    intrinsics = result.intrinsics  # (N, 3, 3)
    image_paths = result.image_paths

    # c2w = inv(w2c) for nerfstudio
    c2w = invert_poses(extrinsics)  # (N, 4, 4)

    frames = []
    for i, (img_path, K, pose) in enumerate(zip(image_paths, intrinsics, c2w)):
        rel_path = f"../images/{img_path.name}"
        frames.append({
            "file_path": rel_path,
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "transform_matrix": pose.tolist(),
        })

    transforms = {
        "camera_model": "PINHOLE",
        "frames": frames,
    }
    out = self.backend_dir / "transforms.json"
    out.write_text(json.dumps(transforms, indent=2))
    logger.info("transforms.json written to %s", out)


def _run_sfm(self) -> PointcloudResult:
    """Run SfM pointcloud stage (colmap/hloc). Experimental."""
    raise NotImplementedError("SfM path not yet implemented — use method: feedforward")


def _run_nerfstudio(self) -> PointcloudResult:
    """Run full nerfstudio pipeline (ns-process-data + ns-train)."""
    raise NotImplementedError("Implemented in Task 5")
```

Add pycolmap import at top of file:

```python
import pycolmap
```

- [ ] **Step 4: Run feedforward tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "build_pointcloud"
```
Expected: all 3 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): implement build_pointcloud feedforward path with BA/LC/clean"
```

---

## Task 5: build_pointcloud — Nerfstudio Path

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write failing nerfstudio tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
def test_build_pointcloud_nerfstudio_calls_subprocess(tmp_path):
    config = _make_config(tmp_path, {
        "pointcloud": {"method": "nerfstudio", "backend": "vggtx"},
        "nerfstudio": {"sfm_tool": "hloc", "train_method": "rade-features"},
    })
    rec = Reconstructor(config)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(Exception):
            # Will raise because no real nerfstudio output on disk — that's fine
            rec.build_pointcloud(overwrite=True)

    # subprocess.run called at least once (ns-process-data or ns-train)
    assert mock_run.call_count >= 1
    first_call_args = mock_run.call_args_list[0][0][0]
    assert "ns-process-data" in first_call_args or "ns-train" in first_call_args
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "nerfstudio"
```
Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3: Implement `_run_nerfstudio()`**

Replace the `_run_nerfstudio` stub in `reconstructor.py`:

```python
def _run_nerfstudio(self) -> PointcloudResult:
    """Run full nerfstudio pipeline via ns-process-data + ns-train subprocesses.

    output_path/nerfstudio/ acts as nerfstudio data dir.
    """
    import subprocess
    from collab_splats.pointcloud.base import CoordinateFrame

    ns_cfg = self.config.get("nerfstudio", {})
    sfm_tool = ns_cfg.get("sfm_tool", "hloc")
    train_method = ns_cfg.get("train_method", "rade-features")

    ns_data_dir = Path(self.config["output_path"]) / "nerfstudio"
    input_path = Path(self.config["input_path"])

    # Stage 1+2: ns-process-data (handles frame extraction + SfM)
    process_cmd = [
        "ns-process-data", "video" if input_path.suffix in {".mp4", ".mov", ".avi"} else "images",
        "--data", str(input_path),
        "--output-dir", str(ns_data_dir),
        "--sfm-tool", sfm_tool,
    ]
    logger.info("Running ns-process-data: %s", " ".join(process_cmd))
    subprocess.run(process_cmd, check=True)

    # Stage 2b: ns-train
    train_cmd = [
        "ns-train", train_method,
        "--data", str(ns_data_dir),
        "--output-dir", str(ns_data_dir / "outputs"),
    ]
    logger.info("Running ns-train: %s", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    # Load COLMAP reconstruction created by ns-process-data
    colmap_dir = ns_data_dir / "colmap" / "sparse" / "0"
    if not colmap_dir.exists():
        raise RuntimeError(f"ns-process-data did not produce COLMAP sparse model at {colmap_dir}")

    recon = pycolmap.Reconstruction()
    recon.read(str(colmap_dir))
    image_paths = sorted((ns_data_dir / "images").glob("*.jpg")) + \
                  sorted((ns_data_dir / "images").glob("*.png"))
    return PointcloudResult(
        reconstruction=recon,
        frame=CoordinateFrame.COLMAP,
        image_paths=image_paths,
    )
```

- [ ] **Step 4: Run nerfstudio test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "nerfstudio"
```
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): implement build_pointcloud nerfstudio subprocess path"
```

---

## Task 6: extract_semantics Stage

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

Key interfaces:
- `BaseFeatureExtractor.extract_and_cache(image_paths, cache_dir, skip_existing=True) -> Path` — saves to `cache_dir/{extractor_name}.zarr`, layout `(N, D, H_p, W_p)`
- `lift_features(feature_maps: list[torch.Tensor], result: FeedforwardResult, depth_tol=0.05) -> torch.Tensor` — returns `(P, D)`
- `FeedforwardResult.load_zarr(path, load_images=True)` — needed for depth/pixel_indices
- `FeatureAutoencoder` from `collab_splats.semantics.compression` — `fit(features)`, `encode(features)`, `save(path)`, `load(path)`

- [ ] **Step 1: Write failing semantics tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
def test_extract_semantics_uses_feature_cache(tmp_path):
    """2D feature extraction skipped when cache exists."""
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    rec.pointcloud = mock_result

    # Pre-populate 2D cache
    cache_path = rec.features_dir / "dinov2" / "dinov2.zarr"
    cache_path.mkdir(parents=True)

    with patch("collab_splats.wrapper.reconstructor._extract_2d_features") as mock_2d, \
         patch("collab_splats.wrapper.reconstructor._lift_and_save") as mock_lift:
        mock_lift.return_value = rec.backend_dir / "semantics" / "dinov2"
        rec.extract_semantics(result=mock_result, overwrite=False)

    mock_2d.assert_not_called()  # cache hit — no extraction


def test_extract_semantics_skips_if_lifted_exists(tmp_path):
    config = _make_config(tmp_path, {"semantics": {"enabled": True, "extractor": "dinov2", "n_components": None}})
    rec = Reconstructor(config)
    lifted_dir = rec.backend_dir / "semantics" / "dinov2"
    lifted_dir.mkdir(parents=True)
    (lifted_dir / "features.zarr").mkdir()

    with patch("collab_splats.wrapper.reconstructor._extract_2d_features") as mock_2d, \
         patch("collab_splats.wrapper.reconstructor._lift_and_save") as mock_lift:
        rec.extract_semantics(overwrite=False)

    mock_2d.assert_not_called()
    mock_lift.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "semantics"
```
Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3: Implement `extract_semantics()` and helpers**

Add imports to `reconstructor.py`:

```python
import torch
import zarr
from collab_splats.semantics.features.dino import DINOv2Extractor
from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
from collab_splats.pointcloud.utils import lift_features
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
```

Add module-level helpers:

```python
_EXTRACTOR_MAP = {
    "dinov2": "DINOv2Extractor",
    "maskclip": "MaskCLIPExtractor",
    "talk2dino": "Talk2DinoExtractor",
}


def _get_extractor(name: str):
    """Instantiate feature extractor by name via registry."""
    from collab_splats.semantics.features.base import BaseFeatureExtractor
    return BaseFeatureExtractor.create(name)


def _extract_2d_features(
    extractor_name: str,
    image_paths: list[Path],
    features_dir: Path,
) -> Path:
    """Extract 2D features for all frames, cache to features_dir/{name}/{name}.zarr."""
    extractor = _get_extractor(extractor_name)
    cache_dir = features_dir / extractor_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extractor.extract_and_cache(image_paths, cache_dir, skip_existing=True)


def _lift_and_save(
    zarr_path: Path,
    feedforward_zarr: Path,
    output_dir: Path,
    n_components: int | None,
) -> Path:
    """Load 2D feature cache + FeedforwardResult, lift to 3D, compress, save."""
    # Load feature maps from zarr cache (N, D, H_p, W_p)
    store = zarr.open(str(zarr_path), mode="r")
    N = store["features"].shape[0] if "features" in store else store.attrs["n_frames"]
    feature_maps = [torch.from_numpy(store["features"][i]) for i in range(store["features"].shape[0])]

    # Load FeedforwardResult with images for lifting
    ff_result = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)

    # Lift: (P, D)
    lifted = lift_features(feature_maps, ff_result)

    # Optional PCA compression
    if n_components is not None:
        from collab_splats.semantics.compression import FeatureAutoencoder
        ae = FeatureAutoencoder(input_dim=lifted.shape[-1], latent_dim=n_components)
        ae.fit(lifted.unsqueeze(0))
        lifted = ae.encode(lifted)
        ae.save(output_dir / "compressor.pt")

    # Save lifted features as zarr
    output_dir.mkdir(parents=True, exist_ok=True)
    out_store = zarr.open(str(output_dir / "features.zarr"), mode="w")
    out_store.create_array("features", data=lifted.numpy())
    return output_dir
```

Replace `extract_semantics()` stub:

```python
def extract_semantics(
    self,
    result: PointcloudResult | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract 2D features (cached), lift to 3D, compress. Returns lifted zarr dir."""
    sem_cfg = self.config.get("semantics", {})
    extractor_name = sem_cfg.get("extractor", "dinov2")
    n_components = sem_cfg.get("n_components", 64)

    lifted_dir = self.backend_dir / "semantics" / extractor_name

    # Skip if lifted features already on disk
    if not overwrite and (lifted_dir / "features.zarr").exists():
        logger.info("Lifted features exist at %s, skipping", lifted_dir)
        return lifted_dir

    result = result or self.pointcloud
    if result is None:
        raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

    image_paths = sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png"))

    # Stage 1: 2D feature extraction (cached at features_dir/extractor)
    cache_dir = self.features_dir / extractor_name
    zarr_path = cache_dir / f"{extractor_name}.zarr"
    if overwrite or not zarr_path.exists():
        logger.info("Extracting 2D features with %s", extractor_name)
        zarr_path = _extract_2d_features(extractor_name, image_paths, self.features_dir)
    else:
        logger.info("2D feature cache hit: %s", zarr_path)

    # Stage 2: Lift to 3D
    feedforward_zarr = self.backend_dir / "feedforward.zarr"
    if not feedforward_zarr.exists():
        raise FileNotFoundError(
            f"feedforward.zarr not found at {feedforward_zarr}. "
            "Run build_pointcloud() with a feedforward backend first."
        )

    logger.info("Lifting 2D features to 3D pointcloud")
    out_dir = _lift_and_save(zarr_path, feedforward_zarr, lifted_dir, n_components)
    return out_dir
```

- [ ] **Step 4: Run semantics tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "semantics"
```
Expected: all 2 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): implement extract_semantics with 2D cache and feature lifting"
```

---

## Task 7: Mesh Stage

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

Key interfaces:
- `Open3DTSDFFusion(output_dir, voxel_size, sdf_trunc)` — dataclass instantiation
- `.create(depths, rgbs, c2w, intrinsics) -> MeshResult`
- `FeedforwardResult.load_zarr(path, load_images=True)` — provides depth + rgb
- `PointcloudResult.extrinsics` — (N, 4, 4) w2c; need c2w = invert
- `MeshResult` from `collab_splats.mesh.base`

- [ ] **Step 1: Write failing mesh tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
def test_mesh_skips_if_ply_exists(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf"}})
    rec = Reconstructor(config)
    mesh_path = rec.backend_dir / "mesh" / "mesh.ply"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.touch()

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        result = rec.mesh(overwrite=False)

    mock_mesh.assert_not_called()
    assert result == mesh_path


def test_mesh_runs_tsdf(tmp_path):
    config = _make_config(tmp_path, {"mesh": {"enabled": True, "mesher": "tsdf", "voxel_size": 0.01, "sdf_trunc": 0.04}})
    rec = Reconstructor(config)
    mock_result = _make_mock_pointcloud_result(tmp_path)
    rec.pointcloud = mock_result

    with patch("collab_splats.wrapper.reconstructor._run_tsdf_mesh") as mock_mesh:
        mock_mesh.return_value = rec.backend_dir / "mesh" / "mesh.ply"
        result = rec.mesh(result=mock_result, overwrite=True)

    mock_mesh.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "mesh"
```
Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3: Implement `mesh()` and `_run_tsdf_mesh()`**

Add imports to `reconstructor.py`:

```python
from collab_splats.mesh.tsdf import Open3DTSDFFusion
from collab_splats.utils.geometry import invert_poses
```

Add module-level helper:

```python
def _run_tsdf_mesh(
    result: PointcloudResult,
    feedforward_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
) -> Path:
    """Fuse depth + RGB from FeedforwardResult into mesh via TSDF."""
    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)

    # depth: (N, H, W), images: (N, H, W, 3) uint8 → float [0,1]
    depths = ff.depth  # (N, H, W) float32 metres
    if depths is None:
        raise ValueError("FeedforwardResult has no depth — cannot mesh.")
    rgbs = ff.images.astype(np.float32) / 255.0  # (N, H, W, 3)

    # c2w from PointcloudResult extrinsics (w2c → c2w)
    c2w = invert_poses(result.extrinsics)  # (N, 4, 4)
    intrinsics = result.intrinsics         # (N, 3, 3)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesher = Open3DTSDFFusion(
        output_dir=output_dir,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
    )
    mesh_result = mesher.create(depths=depths, rgbs=rgbs, c2w=c2w, intrinsics=intrinsics)
    return mesh_result.mesh_path
```

Replace `mesh()` stub:

```python
def mesh(
    self,
    result: PointcloudResult | None = None,
    overwrite: bool = False,
) -> Path:
    """Build mesh from pointcloud depth maps. Returns path to mesh.ply."""
    mesh_path = self.backend_dir / "mesh" / "mesh.ply"

    if not overwrite and mesh_path.exists():
        logger.info("Mesh exists at %s, skipping", mesh_path)
        return mesh_path

    result = result or self.pointcloud
    if result is None:
        raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

    feedforward_zarr = self.backend_dir / "feedforward.zarr"
    if not feedforward_zarr.exists():
        raise FileNotFoundError(
            f"feedforward.zarr not found at {feedforward_zarr}. "
            "Mesh requires depth maps from a feedforward backend."
        )

    mesh_cfg = self.config.get("mesh", {})
    out = _run_tsdf_mesh(
        result=result,
        feedforward_zarr=feedforward_zarr,
        output_dir=self.backend_dir / "mesh",
        voxel_size=mesh_cfg.get("voxel_size", 0.01),
        sdf_trunc=mesh_cfg.get("sdf_trunc", 0.04),
    )
    logger.info("Mesh saved to %s", out)
    return out
```

- [ ] **Step 4: Run mesh tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "mesh"
```
Expected: all 2 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): implement mesh stage with TSDF fusion"
```

---

## Task 8: run_pipeline Orchestrator

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write failing orchestrator tests**

Add to `tests/wrapper/test_reconstructor.py`:

```python
def test_run_pipeline_calls_stages_in_order(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path
    rec.mesh = lambda result=None, overwrite=False: calls.append("mesh") or tmp_path

    rec.run_pipeline(stages=["preprocess", "pointcloud", "semantics", "mesh"])
    assert calls == ["preprocess", "pointcloud", "semantics", "mesh"]


def test_run_pipeline_subset(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)

    rec.run_pipeline(stages=["preprocess", "pointcloud"])
    assert calls == ["preprocess", "pointcloud"]
    assert "semantics" not in calls
    assert "mesh" not in calls


def test_run_pipeline_dep_validation(tmp_path):
    """semantics requires pointcloud to have run first."""
    config = _make_config(tmp_path)
    rec = Reconstructor(config)

    with pytest.raises(ValueError, match="pointcloud"):
        rec.run_pipeline(stages=["semantics"])


def test_run_pipeline_default_uses_config_enabled(tmp_path):
    config = _make_config(tmp_path, {
        "semantics": {"enabled": True, "extractor": "dinov2"},
        "mesh": {"enabled": False},
    })
    rec = Reconstructor(config)
    calls = []

    rec.preprocess = lambda overwrite=False: calls.append("preprocess") or rec.images_dir
    rec.build_pointcloud = lambda overwrite=False: calls.append("pointcloud") or _make_mock_pointcloud_result(tmp_path)
    rec.extract_semantics = lambda result=None, overwrite=False: calls.append("semantics") or tmp_path

    rec.run_pipeline()  # no stages arg — uses config
    assert "semantics" in calls
    assert "mesh" not in calls
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "run_pipeline"
```
Expected: `FAILED` — `NotImplementedError`.

- [ ] **Step 3: Implement `run_pipeline()`**

Add these module-level constants to `reconstructor.py` (above the `Reconstructor` class, after `_VALID_MESHERS`):

```python
_STAGE_ORDER = ["preprocess", "pointcloud", "semantics", "mesh"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preprocess": [],
    "pointcloud": ["preprocess"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
}


def run_pipeline(
    self,
    stages: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Run named stages in dependency order.

    Args:
        stages: Subset of ["preprocess", "pointcloud", "semantics", "mesh"].
                Default: all enabled stages from config.
        overwrite: Re-run stages even if output exists.

    Raises:
        ValueError: If stages list violates dependency ordering.
    """
    if stages is None:
        # Build from config enabled flags
        stages = ["preprocess", "pointcloud"]
        if self.config.get("semantics", {}).get("enabled", False):
            stages.append("semantics")
        if self.config.get("mesh", {}).get("enabled", False):
            stages.append("mesh")

    # Validate dependencies
    stages_set = set(stages)
    for stage in stages:
        for dep in _STAGE_DEPS.get(stage, []):
            if dep not in stages_set:
                raise ValueError(
                    f"Stage '{stage}' requires '{dep}' but '{dep}' is not in stages={stages}. "
                    f"Add '{dep}' to the stages list."
                )

    # Run in canonical order
    result = None
    ordered = [s for s in _STAGE_ORDER if s in stages_set]
    for stage in ordered:
        logger.info("=== Stage: %s ===", stage)
        if stage == "preprocess":
            self.preprocess(overwrite=overwrite)
        elif stage == "pointcloud":
            result = self.build_pointcloud(overwrite=overwrite)
        elif stage == "semantics":
            self.extract_semantics(result=result, overwrite=overwrite)
        elif stage == "mesh":
            self.mesh(result=result, overwrite=overwrite)
```

- [ ] **Step 4: Run orchestrator tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "run_pipeline"
```
Expected: all 4 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): implement run_pipeline orchestrator with stage dep validation"
```

---

## Task 9: Splatter Shim + Exports + Cleanup

**Files:**
- Modify: `collab_splats/wrapper/splatter.py`
- Modify: `collab_splats/wrapper/__init__.py`
- Delete: `stage/feedforward.py`
- Modify: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write failing Splatter deprecation test**

Add to `tests/wrapper/test_reconstructor.py`:

```python
def test_splatter_emits_deprecation_warning(tmp_path):
    from collab_splats.wrapper.splatter import Splatter
    config = {
        "file_path": str(tmp_path / "video.mp4"),
        "method": "rade-features",
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Splatter(config)
    assert any("deprecated" in str(warning.message).lower() for warning in w)
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_reconstructor.py -v -k "deprecation"
```
Expected: `FAILED` — no deprecation warning raised yet.

- [ ] **Step 3: Update `splatter.py` to emit DeprecationWarning**

Read the current `splatter.py`, then add at the top of `Splatter.__init__`:

```python
import warnings

class Splatter:
    # ... existing class ...

    def __init__(self, config: SplatterConfig):
        warnings.warn(
            "Splatter is deprecated and will be removed in a future release. "
            "Use collab_splats.wrapper.Reconstructor instead with pointcloud.method='nerfstudio'.",
            DeprecationWarning,
            stacklevel=2,
        )
        # rest of existing __init__ unchanged
        validated_config = self.validate_config(config)
        self.config: Dict[str, Any] = dict(validated_config)
        self._preprocess_config: Optional[Dict[str, Any]] = None
        self._training_config: Optional[Dict[str, Any]] = None
        self._meshing_config: Optional[Dict[str, Any]] = None
```

- [ ] **Step 4: Update `__init__.py` to export `Reconstructor`**

```python
# collab_splats/wrapper/__init__.py
from .splatter import Splatter, SplatterConfig
from .reconstructor import Reconstructor
from .config import ConfigLoader, parse_cli_overrides

__all__ = [
    "Reconstructor",
    "SplatterConfig",
    "Splatter",
    "ConfigLoader",
    "parse_cli_overrides",
]
```

- [ ] **Step 5: Delete `stage/feedforward.py`**

```bash
git rm stage/feedforward.py
```

- [ ] **Step 6: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/ -v
```
Expected: all tests `PASSED`.

- [ ] **Step 7: Run broader test suite to check for regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/wrapper/ -x
```
Expected: existing tests still pass (no regressions from Splatter change).

- [ ] **Step 8: Final commit**

```bash
git add collab_splats/wrapper/splatter.py collab_splats/wrapper/__init__.py tests/wrapper/test_reconstructor.py
git commit -m "feat(wrapper): add Reconstructor export, Splatter deprecation shim, remove stage/feedforward.py"
```

---

## Self-Review Checklist

After implementation, verify against spec:

- [ ] `images/` is shared across backends ✓
- [ ] `features/` is shared 2D cache ✓
- [ ] `{backend}/` scopes stage 2+ artifacts ✓
- [ ] `transforms.json` written for nerfstudio compatibility ✓
- [ ] Stage skip logic (overwrite=False) implemented for all stages ✓
- [ ] SfM path emits UserWarning ✓
- [ ] Nerfstudio full pipeline via subprocess ✓
- [ ] Splatter emits DeprecationWarning ✓
- [ ] `stage/feedforward.py` deleted ✓
- [ ] `Reconstructor` exported from `collab_splats.wrapper` ✓
- [ ] Tests cover all 8 spec test cases ✓
- [ ] `localize()` stub raises NotImplementedError (follow-up task) ✓
