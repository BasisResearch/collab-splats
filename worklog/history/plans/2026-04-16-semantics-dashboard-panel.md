# Semantics Dashboard — Panel Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gradio semantics dashboard with a Panel app that adds base-dir video browsing, per-video YAML config editing, and a Splatter training launcher, while fixing the NumPy import crash.

**Architecture:** `SemanticsDashboard(param.Parameterized)` mirrors the `DataDashboard` pattern from `collab-data`. `MaterialTemplate` provides a sidebar (base-dir → species → date → video → config) and a main area (Training tab + Explore tab). State flows via `param.watch` callbacks; long-running training uses `subprocess.Popen` + a periodic callback to stream stdout to the UI.

**Tech Stack:** Panel, param, PyYAML (already present), existing `ConfigLoader` from `collab_splats.wrapper.config`, existing semantics extractors/segmentation.

**Spec:** `worklog/history/specs/2026-04-16-semantics-dashboard-redesign.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `collab_splats/__init__.py` | Remove eager model imports (NumPy fix) |
| Create | `collab_splats/dashboard/video_discovery.py` | `discover_videos()` — filesystem scan |
| Create | `collab_splats/dashboard/config_panel.py` | `ConfigPanel(param.Parameterized)` — YAML R/W |
| Replace | `collab_splats/dashboard/semantics.py` | `SemanticsDashboard` + Panel `build_app()` |
| Modify | `collab_splats/dashboard/__main__.py` | Wire `--base-dir` arg, call `pn.serve()` |
| Modify | `pyproject.toml` | Add `panel`, `param` deps; add `tests/dashboard` ignore rule |
| Create | `tests/dashboard/__init__.py` | Empty — makes pytest find the package |
| Create | `tests/dashboard/test_video_discovery.py` | Tests for `discover_videos()` |
| Create | `tests/dashboard/test_config_panel.py` | Tests for `ConfigPanel` load/save/round-trip |
| Create | `tests/dashboard/test_semantics_smoke.py` | Smoke tests — instantiate, layout, watch chain |

---

## Task 1: Fix NumPy Crash

**Files:**
- Modify: `collab_splats/__init__.py:5-9`
- Create: `tests/dashboard/__init__.py`
- Create: `tests/dashboard/test_numpy_fix.py`

- [ ] **Step 1: Create tests/dashboard/__init__.py**

```python
```
(empty file)

- [ ] **Step 2: Write the failing test**

Create `tests/dashboard/test_numpy_fix.py`:

```python
"""Verify the dashboard can be imported without triggering NumPy ABI warnings."""
import subprocess
import sys


def test_dashboard_import_no_numpy_warning():
    """Importing the dashboard must not produce a UserWarning about NumPy."""
    result = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c",
         "from collab_splats.dashboard import semantics"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Dashboard import raised UserWarning (likely NumPy ABI):\n{result.stderr}"
    )


def test_collab_splats_init_no_torch():
    """collab_splats top-level __init__ must not import torch at module load time."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import collab_splats; import sys; assert 'torch' not in sys.modules, "
         "'torch was imported at collab_splats import time'"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 3: Run to confirm it fails**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_numpy_fix.py -v
```

Expected: FAIL — NumPy UserWarning is raised.

- [ ] **Step 4: Fix `collab_splats/__init__.py`**

Remove the eager model imports. The file should become:

```python
"""collab-splats: Extension tools for nerfstudio"""

__version__ = "0.0.1"

from collab_splats.wrapper.splatter import Splatter, SplatterConfig
from collab_splats.utils.camera_utils import ColmapCamera
from collab_splats.utils.trainer_config import _TrainerConfig, _ExperimentConfig

__all__ = [
    "SplatterConfig",
    "Splatter",
    "ColmapCamera",
    "_TrainerConfig",
    "_ExperimentConfig",
]
```

- [ ] **Step 5: Run test — confirm pass**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_numpy_fix.py -v
```

Expected: PASS

- [ ] **Step 6: Confirm existing tests still pass**

```bash
conda activate nerfstudio && make test
```

Expected: all existing tests PASS. If any test imports `RadegsModel` from `collab_splats` (top-level), update that import to `from collab_splats.models.rade_gs_model import RadegsModel`.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/__init__.py tests/dashboard/__init__.py tests/dashboard/test_numpy_fix.py
git commit -m "fix: remove eager torch imports from __init__ to fix NumPy ABI crash"
```

---

## Task 2: Add `panel` and `param` Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add panel and param to optional dependencies**

In `pyproject.toml`, find the `[project.optional-dependencies]` section. Add `panel` and `param` to the `dev` or a new `dashboard` extra. Also add `panel` to the dashboard section if it exists, else add alongside gradio:

```toml
# Find the existing dependencies list that contains "gradio" and add after it:
"panel>=1.3.0",
"param>=2.0.0",
```

- [ ] **Step 2: Install**

```bash
conda activate nerfstudio && pip install "panel>=1.3.0" "param>=2.0.0"
```

Expected: installed without conflicts.

- [ ] **Step 3: Verify import**

```bash
conda activate nerfstudio && python -c "import panel as pn; import param; print(pn.__version__, param.__version__)"
```

Expected: prints version numbers without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add panel and param for dashboard redesign"
```

---

## Task 3: `video_discovery.py` — Filesystem Scanner

**Files:**
- Create: `collab_splats/dashboard/video_discovery.py`
- Create: `tests/dashboard/test_video_discovery.py`

The data layout is `{base_dir}/{species}/{date}/SplatsSD/*.MP4` where `date` is `YYYY-MM-DD`.
YAML filenames use `MMDDYYYY` format: `{species}_date-{MMDDYYYY}_video-{stem}.yaml`.
The `yaml_name_for_video()` helper bridges this conversion.

- [ ] **Step 1: Write failing tests**

Create `tests/dashboard/test_video_discovery.py`:

```python
"""Tests for video filesystem discovery."""
from pathlib import Path
import pytest
from collab_splats.dashboard.video_discovery import discover_videos, yaml_name_for_video


# ---------------------------------------------------------------------------
# discover_videos
# ---------------------------------------------------------------------------

def test_discover_videos_single_video(tmp_path):
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()

    result = discover_videos(tmp_path)

    assert list(result.keys()) == ["birds"]
    assert list(result["birds"].keys()) == ["2024-02-06"]
    assert result["birds"]["2024-02-06"] == [video]


def test_discover_videos_multiple_species(tmp_path):
    for species, date, vid in [
        ("birds", "2024-02-06", "C0043.MP4"),
        ("ants", "2025-11-16", "GH010210.MP4"),
    ]:
        p = tmp_path / species / date / "SplatsSD" / vid
        p.parent.mkdir(parents=True)
        p.touch()

    result = discover_videos(tmp_path)

    assert set(result.keys()) == {"birds", "ants"}


def test_discover_videos_multiple_dates(tmp_path):
    for date, vid in [("2024-02-06", "C0043.MP4"), ("2024-05-27", "GH010097.MP4")]:
        p = tmp_path / "birds" / date / "SplatsSD" / vid
        p.parent.mkdir(parents=True)
        p.touch()

    result = discover_videos(tmp_path)

    assert set(result["birds"].keys()) == {"2024-02-06", "2024-05-27"}


def test_discover_videos_empty_base(tmp_path):
    assert discover_videos(tmp_path) == {}


def test_discover_videos_missing_base():
    assert discover_videos(Path("/nonexistent/path/xyz")) == {}


def test_discover_videos_ignores_dir_without_splatssd(tmp_path):
    # date dir exists but no SplatsSD/ inside
    (tmp_path / "birds" / "2024-02-06").mkdir(parents=True)
    assert discover_videos(tmp_path) == {}


def test_discover_videos_ignores_empty_splatssd(tmp_path):
    (tmp_path / "birds" / "2024-02-06" / "SplatsSD").mkdir(parents=True)
    assert discover_videos(tmp_path) == {}


def test_discover_videos_lowercase_extension(tmp_path):
    video = tmp_path / "rats" / "2024-07-11" / "SplatsSD" / "C0119.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    result = discover_videos(tmp_path)

    assert result["rats"]["2024-07-11"] == [video]


# ---------------------------------------------------------------------------
# yaml_name_for_video
# ---------------------------------------------------------------------------

def test_yaml_name_for_video_basic():
    name = yaml_name_for_video("birds", "2024-02-06", "C0043")
    assert name == "birds_date-02062024_video-C0043"


def test_yaml_name_for_video_ants():
    name = yaml_name_for_video("ants", "2025-11-16", "GH010210")
    assert name == "ants_date-11162025_video-GH010210"


def test_yaml_name_for_video_rats():
    name = yaml_name_for_video("rats", "2024-07-11", "C0119")
    assert name == "rats_date-07112024_video-C0119"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_video_discovery.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.dashboard.video_discovery'`

- [ ] **Step 3: Implement `video_discovery.py`**

Create `collab_splats/dashboard/video_discovery.py`:

```python
"""Filesystem scanner for fieldwork video data.

Expected layout:
    {base_dir}/{species}/{date}/SplatsSD/*.MP4

where date is YYYY-MM-DD.

YAML dataset config names use MMDDYYYY format:
    {species}_date-{MMDDYYYY}_video-{stem}
"""

from __future__ import annotations

from pathlib import Path


def discover_videos(base_dir: Path | str) -> dict[str, dict[str, list[Path]]]:
    """Scan base_dir for videos matching the fieldwork layout.

    Args:
        base_dir: Root directory (e.g. /workspace/fieldwork-data).

    Returns:
        Nested dict: {species: {date: [video_paths]}}.
        Empty dict if base_dir doesn't exist or contains no videos.
    """
    base = Path(base_dir)
    if not base.exists():
        return {}

    result: dict[str, dict[str, list[Path]]] = {}

    for species_dir in sorted(base.iterdir()):
        if not species_dir.is_dir():
            continue
        for date_dir in sorted(species_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            splats_dir = date_dir / "SplatsSD"
            if not splats_dir.exists():
                continue
            videos = sorted(
                list(splats_dir.glob("*.MP4")) + list(splats_dir.glob("*.mp4"))
            )
            if not videos:
                continue
            result.setdefault(species_dir.name, {})[date_dir.name] = videos

    return result


def yaml_name_for_video(species: str, date_dir: str, video_stem: str) -> str:
    """Build the dataset YAML name for a video.

    Converts YYYY-MM-DD directory date to MMDDYYYY as used in YAML filenames.

    Args:
        species: e.g. "birds"
        date_dir: Directory name, e.g. "2024-02-06"
        video_stem: Video filename without extension, e.g. "C0043"

    Returns:
        e.g. "birds_date-02062024_video-C0043"
    """
    parts = date_dir.split("-")
    if len(parts) == 3:
        yyyy, mm, dd = parts
        date_str = f"{mm}{dd}{yyyy}"
    else:
        date_str = date_dir.replace("-", "")
    return f"{species}_date-{date_str}_video-{video_stem}"
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_video_discovery.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/video_discovery.py tests/dashboard/test_video_discovery.py
git commit -m "feat: add video_discovery module for fieldwork filesystem scanning"
```

---

## Task 4: `config_panel.py` — YAML Config Widget

**Files:**
- Create: `collab_splats/dashboard/config_panel.py`
- Create: `tests/dashboard/test_config_panel.py`

`ConfigLoader(config_dir)` takes the **directory** containing `base.yaml` and `datasets/`.
`ConfigLoader.load(dataset_name)` takes the dataset stem (e.g. `"birds_date-02062024_video-C0043"`).

- [ ] **Step 1: Write failing tests**

Create `tests/dashboard/test_config_panel.py`:

```python
"""Tests for ConfigPanel YAML load/save/round-trip."""
from pathlib import Path
import yaml
import pytest
from collab_splats.dashboard.config_panel import ConfigPanel

CONFIGS_DIR = Path(__file__).parents[2] / "docs" / "splats" / "configs"


def test_config_panel_loads_base_defaults():
    """Loading with yaml_path=None gives base.yaml defaults."""
    panel = ConfigPanel(configs_dir=CONFIGS_DIR)
    panel.load_from_yaml(None)

    assert panel.method == "rade-features"
    assert panel.sfm_tool == "hloc"
    assert abs(panel.frame_proportion - 0.25) < 1e-9
    assert panel.min_frames == 100


def test_config_panel_loads_existing_yaml(tmp_path):
    """Loading a dataset YAML overrides base defaults."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    # Copy base.yaml into tmp configs dir
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")

    dataset_yaml = datasets_dir / "birds_date-02062024_video-C0043.yaml"
    dataset_yaml.write_text(
        "method: rade-gs\nframe_proportion: 0.10\nfile_path: /workspace/test.MP4\n"
    )

    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(dataset_yaml)

    assert panel.method == "rade-gs"
    assert abs(panel.frame_proportion - 0.10) < 1e-9
    assert panel.sfm_tool == "hloc"  # not overridden — stays at base default


def test_config_panel_save_writes_overrides_only(tmp_path):
    """save_to_yaml writes only fields that differ from base."""
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    (tmp_path / "datasets").mkdir()

    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(None)
    panel.method = "rade-gs"  # change from base default "rade-features"
    # leave frame_proportion at 0.25 (same as base)

    out = tmp_path / "datasets" / "test_dataset.yaml"
    panel.save_to_yaml(out)

    written = yaml.safe_load(out.read_text())
    assert written.get("method") == "rade-gs"
    assert "frame_proportion" not in written  # unchanged from base — omitted


def test_config_panel_round_trip(tmp_path):
    """Load → mutate → save → reload → same values."""
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    (tmp_path / "datasets").mkdir()

    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(None)
    panel.method = "splatfacto"
    panel.sfm_tool = "colmap"
    panel.frame_proportion = 0.5
    panel.min_frames = 200

    out = tmp_path / "datasets" / "roundtrip.yaml"
    panel.save_to_yaml(out)

    panel2 = ConfigPanel(configs_dir=tmp_path)
    panel2.load_from_yaml(out)

    assert panel2.method == "splatfacto"
    assert panel2.sfm_tool == "colmap"
    assert abs(panel2.frame_proportion - 0.5) < 1e-9
    assert panel2.min_frames == 200


def test_config_panel_to_splatter_config(tmp_path):
    """to_splatter_config returns dict with correct keys."""
    import shutil
    shutil.copy(CONFIGS_DIR / "base.yaml", tmp_path / "base.yaml")
    (tmp_path / "datasets").mkdir()

    panel = ConfigPanel(configs_dir=tmp_path)
    panel.load_from_yaml(None)

    config = panel.to_splatter_config(
        file_path=tmp_path / "video.MP4",
        output_path=tmp_path / "output",
    )

    assert config["file_path"] == str(tmp_path / "video.MP4")
    assert config["method"] == "rade-features"
    assert config["input_type"] == "video"
    assert config["output_path"] == str(tmp_path / "output")
    assert abs(config["frame_proportion"] - 0.25) < 1e-9
    assert config["min_frames"] == 100
```

- [ ] **Step 2: Run to confirm they fail**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_config_panel.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `config_panel.py`**

Create `collab_splats/dashboard/config_panel.py`:

```python
"""ConfigPanel — param-based widget for per-video Splatter config.

Loads and saves per-video YAML overrides on top of base.yaml.
Uses ConfigLoader for hierarchical merging (base ← dataset).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import param
import yaml

from collab_splats.wrapper.config import ConfigLoader
from collab_splats.wrapper.splatter import SplatterConfig

_METHODS = ["rade-features", "rade-gs", "splatfacto", "feature-splatting"]
_SFM_TOOLS = ["hloc", "colmap"]


class ConfigPanel(param.Parameterized):
    """Reactive config widget backed by YAML files.

    Args:
        configs_dir: Directory containing base.yaml and datasets/.
                     Defaults to docs/splats/configs/ relative to repo root.
    """

    method = param.Selector(default="rade-features", objects=_METHODS)
    sfm_tool = param.Selector(default="hloc", objects=_SFM_TOOLS)
    frame_proportion = param.Number(default=0.25, bounds=(0.01, 1.0), step=0.01)
    min_frames = param.Integer(default=100, bounds=(10, 1000))

    def __init__(self, configs_dir: Path | str | None = None, **params: Any):
        super().__init__(**params)
        if configs_dir is None:
            configs_dir = Path(__file__).parents[2] / "docs" / "splats" / "configs"
        self._configs_dir = Path(configs_dir)
        self._loader = ConfigLoader(self._configs_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_from_yaml(self, yaml_path: Path | None) -> None:
        """Populate params from yaml_path (merged with base.yaml).

        Args:
            yaml_path: Full path to dataset YAML, or None to load base defaults.
        """
        if yaml_path is not None and yaml_path.exists():
            dataset_name = yaml_path.stem
            try:
                config = self._loader.load(dataset_name)
            except ValueError:
                # Dataset not in configs_dir — fall back to raw YAML + base merge
                raw = self._load_raw_yaml(yaml_path)
                base = self._loader.base_config
                config = {**base, **raw}
                if "preprocess" in raw:
                    config["preprocess"] = {**base.get("preprocess", {}), **raw["preprocess"]}
        else:
            config = dict(self._loader.base_config)

        self.method = config.get("method", self.param.method.default)
        self.sfm_tool = config.get("preprocess", {}).get(
            "sfm_tool", self.param.sfm_tool.default
        )
        self.frame_proportion = float(
            config.get("frame_proportion", self.param.frame_proportion.default)
        )
        self.min_frames = int(
            config.get("min_frames", self.param.min_frames.default)
        )

    @staticmethod
    def _load_raw_yaml(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_to_yaml(self, yaml_path: Path) -> None:
        """Write only the fields that differ from base.yaml.

        Args:
            yaml_path: Destination path (created if it doesn't exist).
        """
        base = self._loader.base_config
        override: dict = {}

        if self.method != base.get("method"):
            override["method"] = self.method
        if abs(self.frame_proportion - float(base.get("frame_proportion", 0.25))) > 1e-9:
            override["frame_proportion"] = self.frame_proportion
        if self.min_frames != int(base.get("min_frames", 100)):
            override["min_frames"] = self.min_frames
        base_sfm = base.get("preprocess", {}).get("sfm_tool", "hloc")
        if self.sfm_tool != base_sfm:
            override.setdefault("preprocess", {})["sfm_tool"] = self.sfm_tool

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(override, f, default_flow_style=False)

    # ------------------------------------------------------------------
    # Splatter integration
    # ------------------------------------------------------------------

    def to_splatter_config(
        self, file_path: Path | str, output_path: Path | str
    ) -> SplatterConfig:
        """Build a SplatterConfig dict from current param values."""
        return SplatterConfig(
            file_path=str(file_path),
            method=self.method,
            input_type="video",
            output_path=str(output_path),
            frame_proportion=self.frame_proportion,
            min_frames=self.min_frames,
        )

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self) -> "pn.Column":  # type: ignore[name-defined]  # noqa: F821
        """Return a Panel Column of widgets bound to this ConfigPanel's params."""
        import panel as pn

        return pn.Column(
            pn.widgets.Select.from_param(self.param.method, name="Method"),
            pn.widgets.Select.from_param(self.param.sfm_tool, name="SfM Tool"),
            pn.widgets.FloatSlider.from_param(
                self.param.frame_proportion, name="Frame proportion"
            ),
            pn.widgets.IntSlider.from_param(self.param.min_frames, name="Min frames"),
        )
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_config_panel.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/config_panel.py tests/dashboard/test_config_panel.py
git commit -m "feat: add ConfigPanel for per-video YAML config editing"
```

---

## Task 5: `SemanticsDashboard` — Skeleton + Smoke Tests

**Files:**
- Replace: `collab_splats/dashboard/semantics.py`
- Create: `tests/dashboard/test_semantics_smoke.py`

This task builds the full class skeleton with sidebar, watch chain, and both tab containers. Helper functions from the original Gradio `semantics.py` (`_load_frames_from_video`, `_pca_to_rgb`, `_overlay_pca`, `_colorize_masks`) are kept as module-level functions — their logic is unchanged.

- [ ] **Step 1: Write smoke tests first**

Create `tests/dashboard/test_semantics_smoke.py`:

```python
"""Smoke tests for SemanticsDashboard — no browser, no CUDA required."""
import panel as pn
import pytest
from pathlib import Path
from collab_splats.dashboard.semantics import SemanticsDashboard


def test_instantiation_no_crash(tmp_path):
    """SemanticsDashboard instantiates without error."""
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    assert dashboard is not None


def test_create_layout_returns_material_template(tmp_path):
    """create_layout() returns a Panel MaterialTemplate."""
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    layout = dashboard.create_layout()
    assert isinstance(layout, pn.template.MaterialTemplate)


def test_refresh_species_empty_dir(tmp_path):
    """_refresh_species on empty dir clears species/date/video selects."""
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    assert dashboard.species_select.options == []


def test_refresh_species_populates_with_videos(tmp_path):
    """_refresh_species finds species when SplatsSD videos exist."""
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()

    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()

    assert "birds" in dashboard.species_select.options


def test_watch_chain_populates_dates(tmp_path):
    """Selecting a species populates the date select."""
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()

    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    dashboard.species_select.value = "birds"

    assert "2024-02-06" in dashboard.date_select.options


def test_watch_chain_populates_videos(tmp_path):
    """Selecting a date populates the video select."""
    video = tmp_path / "birds" / "2024-02-06" / "SplatsSD" / "C0043.MP4"
    video.parent.mkdir(parents=True)
    video.touch()

    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    dashboard._refresh_species()
    dashboard.species_select.value = "birds"
    dashboard.date_select.value = "2024-02-06"

    assert "C0043" in dashboard.video_select.options
```

- [ ] **Step 2: Run to confirm they fail**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_semantics_smoke.py -v
```

Expected: FAIL — module not found (old Gradio semantics.py has no `SemanticsDashboard`).

- [ ] **Step 3: Replace `semantics.py` with Panel implementation**

Replace `collab_splats/dashboard/semantics.py` entirely:

```python
"""
Semantics + training pipeline dashboard.

Launch with:
    collab-dashboard semantics --base-dir /workspace/fieldwork-data
    panel serve collab_splats/dashboard/__main__.py --args --base-dir /workspace/fieldwork-data

Four components:
    Sidebar  — base dir → species → date → video → YAML config editor
    Training tab — launch/stop Splatter, stream logs
    Explore tab  — PCA features, MobileSAM segmentation, Talk2DINO heatmaps
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import panel as pn
import param
from PIL import Image

from collab_splats.dashboard.config_panel import ConfigPanel
from collab_splats.dashboard.video_discovery import discover_videos, yaml_name_for_video

CONFIGS_DIR = Path(__file__).parents[2] / "docs" / "splats" / "configs"
DATASETS_DIR = CONFIGS_DIR / "datasets"


# ---------------------------------------------------------------------------
# Image helpers (unchanged from original Gradio version)
# ---------------------------------------------------------------------------


def _load_frames_from_video(video_path: str, fps: float) -> list[np.ndarray]:
    """Extract frames from a video at the given FPS."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required. pip install opencv-python"
        ) from e
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(native_fps / fps))
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def _pca_to_rgb(features_chw: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    C, H, W = features_chw.shape
    flat = features_chw.reshape(C, -1).T
    rgb_flat = PCA(n_components=3).fit_transform(flat)
    lo, hi = rgb_flat.min(0), rgb_flat.max(0)
    rgb_flat = (rgb_flat - lo) / (hi - lo + 1e-8)
    return (rgb_flat.reshape(H, W, 3) * 255).astype(np.uint8)


def _overlay_pca(frame_rgb: np.ndarray, pca_rgb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    frame_pil = Image.fromarray(frame_rgb).resize((pca_rgb.shape[1], pca_rgb.shape[0]))
    blended = (np.array(frame_pil).astype(float) * (1 - alpha) + pca_rgb.astype(float) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def _colorize_masks(frame_rgb: np.ndarray, masks_tensor: Any) -> np.ndarray:
    COLORS = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (80, 255, 255), (255, 80, 255),
        (255, 160, 80), (160, 80, 255),
    ]
    vis = frame_rgb.copy().astype(np.float32)
    for i, mask in enumerate(masks_tensor):
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        m = mask.numpy().astype(bool)
        vis[m] = vis[m] * 0.45 + color * 0.55
    return vis.clip(0, 255).astype(np.uint8)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------


class SemanticsDashboard(param.Parameterized):
    """Panel dashboard for semantic exploration and Splatter training."""

    def __init__(self, base_dir: str = "/workspace/fieldwork-data", **params: Any):
        super().__init__(**params)

        # Internal state
        self._base_dir = base_dir
        self._video_tree: dict = {}
        self._frames: list[np.ndarray] = []
        self._current_frame: np.ndarray | None = None
        self._training_process: subprocess.Popen | None = None
        self._poll_cb: Any = None
        self._config_panel = ConfigPanel(configs_dir=CONFIGS_DIR)

        # Load extractor names (lazy import to avoid NumPy crash)
        try:
            from collab_splats.semantics.features import BaseFeatureExtractor
            extractor_names = list(BaseFeatureExtractor._registry.keys()) or ["(none)"]
        except Exception:
            extractor_names = ["(none)"]

        # ---- Sidebar widgets ----
        self.base_dir_input = pn.widgets.TextInput(
            name="Base directory", value=base_dir, width=350
        )
        self.species_select = pn.widgets.Select(
            name="Species", options=[], width=350
        )
        self.date_select = pn.widgets.Select(
            name="Date", options=[], width=350
        )
        self.video_select = pn.widgets.Select(
            name="Video", options=[], width=350
        )
        self.save_yaml_btn = pn.widgets.Button(
            name="Save YAML", button_type="success", width=160
        )
        self.reset_yaml_btn = pn.widgets.Button(
            name="Reset to base", button_type="light", width=160
        )

        # ---- Status / loading ----
        self.status_pane = pn.pane.HTML("<p>Ready</p>", width=800, height=30)
        self.loading_modal = pn.pane.HTML("", visible=False, sizing_mode="stretch_both")

        # ---- Training tab widgets ----
        self.output_path_input = pn.widgets.TextInput(
            name="Output path", value="/workspace/outputs", width=700
        )
        self.launch_btn = pn.widgets.Button(
            name="Launch Splatter", button_type="primary", width=160
        )
        self.stop_btn = pn.widgets.Button(
            name="Stop", button_type="danger", width=100, disabled=True
        )
        self.training_log = pn.widgets.TextAreaInput(
            name="Training log", value="", rows=20, width=700, disabled=True
        )

        # ---- Explore tab widgets ----
        self.fps_slider = pn.widgets.IntSlider(
            name="FPS to extract", value=5, start=1, end=30, width=350
        )
        self.extract_frames_btn = pn.widgets.Button(
            name="Extract Frames", button_type="primary", width=160
        )
        self.frame_count_txt = pn.pane.HTML("")
        self.frame_slider = pn.widgets.IntSlider(
            name="Frame index", value=0, start=0, end=0, width=700
        )
        self.current_frame_pane = pn.pane.PNG(None, width=700)
        self.extractor_dd = pn.widgets.Select(
            name="Extractor", options=extractor_names, width=200
        )
        self.device_dd = pn.widgets.Select(
            name="Device", options=["cpu", "cuda"], width=100
        )
        self.extract_features_btn = pn.widgets.Button(
            name="Extract Features", button_type="primary", width=160
        )
        self.feature_overlay_pane = pn.pane.PNG(None, width=700)
        self.seg_strategy_dd = pn.widgets.Select(
            name="Strategy", options=["object", "auto"], width=200
        )
        self.seg_device_dd = pn.widgets.Select(
            name="Device", options=["cpu", "cuda"], width=100
        )
        self.seg_btn = pn.widgets.Button(
            name="Segment", button_type="primary", width=100
        )
        self.seg_output_pane = pn.pane.PNG(None, width=700)
        self.seg_count_txt = pn.pane.HTML("")
        self.hf_model_dd = pn.widgets.Select(
            name="Talk2DINO model",
            options=["lorebianchi98/Talk2DINOv3-ViTB", "lorebianchi98/Talk2DINO-ViTB"],
            width=300,
        )
        self.query_device_dd = pn.widgets.Select(
            name="Device", options=["cpu", "cuda"], width=100
        )
        self.text_pairs_input = pn.widgets.TextAreaInput(
            name="Text pairs (JSON)",
            value='{"object": [["object", "thing"], ["background", "empty"]]}',
            rows=5,
            width=700,
        )
        self.method_dd = pn.widgets.Select(
            name="Method", options=["standard", "pairwise"], width=150
        )
        self.temp_slider = pn.widgets.FloatSlider(
            name="Softmax temperature", value=0.05, start=0.001, end=0.1, step=0.001, width=350
        )
        self.query_btn = pn.widgets.Button(
            name="Generate Heatmaps", button_type="primary", width=160
        )
        self.query_gallery = pn.GridBox(ncols=3)

        # ---- Wire callbacks ----
        self.base_dir_input.param.watch(self._on_base_dir_change, "value")
        self.species_select.param.watch(self._on_species_change, "value")
        self.date_select.param.watch(self._on_date_change, "value")
        self.video_select.param.watch(self._on_video_change, "value")
        self.save_yaml_btn.on_click(self._save_yaml)
        self.reset_yaml_btn.on_click(self._reset_yaml)
        self.launch_btn.on_click(self._launch_training)
        self.stop_btn.on_click(self._stop_training)
        self.extract_frames_btn.on_click(self._extract_frames)
        self.frame_slider.param.watch(self._on_frame_slider_change, "value")
        self.extract_features_btn.on_click(self._extract_features)
        self.seg_btn.on_click(self._run_segmentation)
        self.query_btn.on_click(self._run_semantic_query)

        # Initial scan
        self._refresh_species()

    # ------------------------------------------------------------------
    # Watch chain: base_dir → species → date → video
    # ------------------------------------------------------------------

    def _on_base_dir_change(self, event: Any) -> None:
        self._base_dir = self.base_dir_input.value
        self._refresh_species()

    def _refresh_species(self) -> None:
        try:
            self._video_tree = discover_videos(self._base_dir)
            species = sorted(self._video_tree.keys())
            self.species_select.options = species
            self.date_select.options = []
            self.video_select.options = []
            if species:
                self.species_select.value = species[0]
            self._update_status(f"Found {len(species)} species in {self._base_dir}")
        except Exception as e:
            self._update_status(f"Error scanning {self._base_dir}: {e}", error=True)

    def _on_species_change(self, event: Any) -> None:
        species = self.species_select.value
        if not species:
            self.date_select.options = []
            self.video_select.options = []
            return
        dates = sorted(self._video_tree.get(species, {}).keys())
        self.date_select.options = dates
        self.video_select.options = []
        if dates:
            self.date_select.value = dates[0]

    def _on_date_change(self, event: Any) -> None:
        species = self.species_select.value
        date = self.date_select.value
        if not species or not date:
            self.video_select.options = []
            return
        videos = self._video_tree.get(species, {}).get(date, [])
        stems = [v.stem for v in videos]
        self.video_select.options = stems
        if stems:
            self.video_select.value = stems[0]

    def _on_video_change(self, event: Any) -> None:
        try:
            self._load_video()
        except Exception as e:
            self._update_status(f"Error loading video: {e}", error=True)

    def _load_video(self) -> None:
        species = self.species_select.value
        date = self.date_select.value
        video_stem = self.video_select.value
        if not all([species, date, video_stem]):
            return

        yaml_name = yaml_name_for_video(species, date, video_stem)
        yaml_path = DATASETS_DIR / f"{yaml_name}.yaml"
        self._config_panel.load_from_yaml(yaml_path if yaml_path.exists() else None)
        self._update_status(
            f"Loaded {'existing' if yaml_path.exists() else 'base'} config for {video_stem}"
        )

    # ------------------------------------------------------------------
    # YAML save / reset
    # ------------------------------------------------------------------

    def _save_yaml(self, event: Any) -> None:
        try:
            species = self.species_select.value
            date = self.date_select.value
            video_stem = self.video_select.value
            if not all([species, date, video_stem]):
                self._update_status("Select a video before saving.", error=True)
                return
            yaml_name = yaml_name_for_video(species, date, video_stem)
            yaml_path = DATASETS_DIR / f"{yaml_name}.yaml"
            self._config_panel.save_to_yaml(yaml_path)
            self._update_status(f"Saved config → {yaml_path.name}")
        except Exception as e:
            self._update_status(f"Save failed: {e}", error=True)

    def _reset_yaml(self, event: Any) -> None:
        self._config_panel.load_from_yaml(None)
        self._update_status("Reset to base defaults")

    # ------------------------------------------------------------------
    # Training tab
    # ------------------------------------------------------------------

    def _resolve_video_path(self) -> Path | None:
        species = self.species_select.value
        date = self.date_select.value
        video_stem = self.video_select.value
        if not all([species, date, video_stem]):
            return None
        videos = self._video_tree.get(species, {}).get(date, [])
        for v in videos:
            if v.stem == video_stem:
                return v
        return None

    def _launch_training(self, event: Any) -> None:
        video_path = self._resolve_video_path()
        if video_path is None:
            self._update_status("Select a video before launching.", error=True)
            return
        if self._training_process and self._training_process.poll() is None:
            self._update_status("Training already running.", error=True)
            return

        output_path = Path(self.output_path_input.value)
        config = self._config_panel.to_splatter_config(video_path, output_path)

        # Write a temp config file for the subprocess
        import tempfile, yaml as _yaml
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            _yaml.dump(dict(config), f)
            tmp_cfg = f.name

        self.training_log.value = f"Launching Splatter for {video_path.name}...\n"
        self.launch_btn.disabled = True
        self.stop_btn.disabled = False

        self._training_process = subprocess.Popen(
            [
                sys.executable, "-c",
                f"from collab_splats.wrapper.splatter import Splatter; "
                f"import yaml; cfg = yaml.safe_load(open('{tmp_cfg}')); "
                f"Splatter(cfg).run()",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._poll_cb = pn.state.add_periodic_callback(
            self._poll_training_log, period=500
        )
        self._update_status(f"Training started (PID {self._training_process.pid})")

    def _poll_training_log(self) -> None:
        if self._training_process is None:
            return
        line = self._training_process.stdout.readline()
        if line:
            self.training_log.value += line
        if self._training_process.poll() is not None:
            rc = self._training_process.returncode
            if rc == 0:
                self._update_status("Training complete ✓")
            else:
                self._update_status(f"Training failed (exit {rc})", error=True)
            self._poll_cb.stop()
            self._poll_cb = None
            self.launch_btn.disabled = False
            self.stop_btn.disabled = True

    def _stop_training(self, event: Any) -> None:
        if self._training_process and self._training_process.poll() is None:
            self._training_process.terminate()
            self._update_status("Training stopped by user")
        if self._poll_cb:
            self._poll_cb.stop()
            self._poll_cb = None
        self.launch_btn.disabled = False
        self.stop_btn.disabled = True

    # ------------------------------------------------------------------
    # Explore tab
    # ------------------------------------------------------------------

    def _extract_frames(self, event: Any) -> None:
        video_path = self._resolve_video_path()
        if video_path is None:
            self._update_status("Select a video first.", error=True)
            return
        try:
            self._show_loading(f"Extracting frames from {video_path.name}...")
            self._frames = _load_frames_from_video(str(video_path), self.fps_slider.value)
            n = len(self._frames)
            self.frame_slider.end = max(0, n - 1)
            self.frame_slider.value = 0
            self.frame_count_txt.object = f"<p>{n} frames extracted</p>"
            if self._frames:
                self._current_frame = self._frames[0]
                self.current_frame_pane.object = _to_png_bytes(self._frames[0])
            self._update_status(f"Extracted {n} frames")
        except Exception as e:
            self._update_status(f"Frame extraction failed: {e}", error=True)
        finally:
            self._hide_loading()

    def _on_frame_slider_change(self, event: Any) -> None:
        idx = int(self.frame_slider.value)
        if self._frames and idx < len(self._frames):
            self._current_frame = self._frames[idx]
            self.current_frame_pane.object = _to_png_bytes(self._frames[idx])

    def _extract_features(self, event: Any) -> None:
        if self._current_frame is None:
            self._update_status("Extract frames first.", error=True)
            return
        try:
            import torch
            from collab_splats.semantics.features import BaseFeatureExtractor

            extractor_name = self.extractor_dd.value
            device = self.device_dd.value
            cls = BaseFeatureExtractor.get(extractor_name)
            extractor = cls(device=device)
            pil = Image.fromarray(self._current_frame)

            if extractor_name == "talk2dino":
                pil_pre = extractor.preprocess(pil)
                feats = extractor.forward(pil_pre)
                n = feats.shape[0]
                g = int(n ** 0.5)
                feat_np = feats.cpu().float().numpy().reshape(g, g, -1).transpose(2, 0, 1)
            elif extractor_name == "dinov2":
                tensor, H, W = extractor.preprocess(pil)
                feats = extractor.forward(tensor)
                feat_np = extractor.reshape(feats, H, W).numpy()
            else:
                tensor = extractor.preprocess(pil).unsqueeze(0)
                feat_np = extractor.forward(tensor)[0].cpu().float().numpy()

            pca_rgb = _pca_to_rgb(feat_np)
            overlay = _overlay_pca(self._current_frame, pca_rgb)
            self.feature_overlay_pane.object = _to_png_bytes(overlay)
            self._update_status("Feature extraction complete")
        except Exception as e:
            self._update_status(f"Feature extraction failed: {e}", error=True)

    def _run_segmentation(self, event: Any) -> None:
        if self._current_frame is None:
            self._update_status("Extract frames first.", error=True)
            return
        try:
            from collab_splats.semantics.segmentation import Segmentation

            seg = Segmentation(
                backend="mobilesamv2",
                strategy=self.seg_strategy_dd.value,
                device=self.seg_device_dd.value,
            )
            result = seg.segment(self._current_frame)
            if result is None:
                self.seg_output_pane.object = _to_png_bytes(self._current_frame)
                self.seg_count_txt.object = "<p>0 masks found</p>"
            else:
                masks, _ = result
                vis = _colorize_masks(self._current_frame, masks)
                self.seg_output_pane.object = _to_png_bytes(vis)
                self.seg_count_txt.object = f"<p>{len(masks)} masks found</p>"
            self._update_status("Segmentation complete")
        except Exception as e:
            self._update_status(f"Segmentation failed: {e}", error=True)

    def _run_semantic_query(self, event: Any) -> None:
        if self._current_frame is None:
            self._update_status("Extract frames first.", error=True)
            return
        try:
            text_pairs_raw = json.loads(self.text_pairs_input.value)
            text_pairs = {k: (list(v[0]), list(v[1])) for k, v in text_pairs_raw.items()}
        except json.JSONDecodeError as e:
            self._update_status(f"Invalid JSON: {e}", error=True)
            return
        try:
            from collab_splats.semantics.features import Talk2DinoExtractor

            extractor = Talk2DinoExtractor(
                hf_model_id=self.hf_model_dd.value, device=self.query_device_dd.value
            )
            heatmaps = extractor.compute_semantic_heatmap(
                Image.fromarray(self._current_frame),
                text_pairs,
                self.temp_slider.value,
                self.method_dd.value,
            )
            items = []
            for label, masked_img in heatmaps.items():
                img_u8 = (masked_img * 255).clip(0, 255).astype(np.uint8)
                items.append(
                    pn.Column(
                        pn.pane.PNG(_to_png_bytes(img_u8), width=220),
                        pn.pane.HTML(f"<p><b>{label}</b></p>"),
                    )
                )
            self.query_gallery.objects = items
            self._update_status(f"Generated {len(items)} heatmaps")
        except Exception as e:
            self._update_status(f"Semantic query failed: {e}", error=True)

    # ------------------------------------------------------------------
    # Loading overlay (matches collab-data pattern)
    # ------------------------------------------------------------------

    def _show_loading(self, message: str = "Loading...") -> None:
        self.loading_modal.object = f"""
        <div style='position:fixed;top:0;left:0;width:100%;height:100%;
                    background:rgba(0,0,0,0.5);z-index:10000;
                    display:flex;align-items:center;justify-content:center;'>
          <div style='background:white;padding:40px;border-radius:16px;
                      box-shadow:0 12px 40px rgba(0,0,0,0.15);text-align:center;'>
            <p style='color:#555;font-size:16px;'>{message}</p>
          </div>
        </div>"""
        self.loading_modal.visible = True
        self._update_status(f"🔄 {message}")

    def _hide_loading(self) -> None:
        self.loading_modal.object = ""
        self.loading_modal.visible = False

    def _update_status(self, msg: str, error: bool = False) -> None:
        color = "red" if error else "inherit"
        self.status_pane.object = f"<p style='color:{color}'>{msg}</p>"

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def create_layout(self) -> pn.template.MaterialTemplate:
        sidebar = pn.Column(
            "## Video Browser",
            self.base_dir_input,
            self.species_select,
            self.date_select,
            self.video_select,
            pn.layout.Divider(),
            "## Config",
            self._config_panel.panel(),
            pn.Row(self.save_yaml_btn, self.reset_yaml_btn),
        )

        training_tab = pn.Column(
            self.output_path_input,
            pn.Row(self.launch_btn, self.stop_btn),
            self.training_log,
        )

        explore_tab = pn.Column(
            "### Extract Frames",
            pn.Row(self.fps_slider, self.extract_frames_btn),
            self.frame_count_txt,
            self.frame_slider,
            self.current_frame_pane,
            pn.layout.Divider(),
            "### Feature Extraction (PCA)",
            pn.Row(self.extractor_dd, self.device_dd, self.extract_features_btn),
            self.feature_overlay_pane,
            pn.layout.Divider(),
            "### Segmentation (MobileSAM)",
            pn.Row(self.seg_strategy_dd, self.seg_device_dd, self.seg_btn),
            self.seg_output_pane,
            self.seg_count_txt,
            pn.layout.Divider(),
            "### Semantic Query (Talk2DINO)",
            pn.Row(self.hf_model_dd, self.query_device_dd),
            self.text_pairs_input,
            pn.Row(self.method_dd, self.temp_slider),
            self.query_btn,
            self.query_gallery,
        )

        return pn.template.MaterialTemplate(
            title="collab-splats Semantic Explorer",
            sidebar=[sidebar],
            main=[
                self.loading_modal,
                self.status_pane,
                pn.Tabs(
                    ("Training", training_tab),
                    ("Explore", explore_tab),
                ),
            ],
            header_background="#2596be",
            sidebar_width=400,
        )


# ---------------------------------------------------------------------------
# Entry point (kept for backward-compatibility with __main__.py)
# ---------------------------------------------------------------------------


def build_app(base_dir: str = "/workspace/fieldwork-data") -> pn.template.MaterialTemplate:
    pn.extension()
    dashboard = SemanticsDashboard(base_dir=base_dir)
    return dashboard.create_layout()


def run_app(host: str = "0.0.0.0", port: int = 7860, base_dir: str = "/workspace/fieldwork-data") -> None:
    app = build_app(base_dir=base_dir)
    pn.serve(app, address=host, port=port, show=False)
```

- [ ] **Step 4: Run smoke tests — confirm pass**

```bash
conda activate nerfstudio && pytest tests/dashboard/test_semantics_smoke.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite — no regressions**

```bash
conda activate nerfstudio && make test
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "feat: replace Gradio semantics dashboard with Panel SemanticsDashboard"
```

---

## Task 6: Update `__main__.py` CLI

**Files:**
- Modify: `collab_splats/dashboard/__main__.py`

- [ ] **Step 1: Replace `__main__.py`**

```python
"""
collab_splats dashboard launcher.

Usage:
    collab-dashboard semantics [--base-dir DIR] [--host HOST] [--port PORT]
    python -m collab_splats.dashboard semantics
"""

import argparse


DASHBOARDS = {
    "semantics": "collab_splats.dashboard.semantics:run_app",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="collab-dashboard",
        description="Launch a collab-splats interactive dashboard.",
    )
    parser.add_argument(
        "mode",
        choices=list(DASHBOARDS.keys()),
        help="Dashboard to launch.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port (default: 7860)"
    )
    parser.add_argument(
        "--base-dir",
        default="/workspace/fieldwork-data",
        help="Root directory for video discovery (default: /workspace/fieldwork-data)",
    )
    args = parser.parse_args()

    module_path, func_name = DASHBOARDS[args.mode].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    run_fn = getattr(mod, func_name)
    run_fn(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI**

```bash
conda activate nerfstudio && python -m collab_splats.dashboard semantics --help
```

Expected: prints help with `--base-dir`, `--host`, `--port` options.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/dashboard/__main__.py
git commit -m "feat: update dashboard CLI to support --base-dir flag"
```

---

## Task 7: Full Verification

- [ ] **Step 1: Run all tests**

```bash
conda activate nerfstudio && make test
```

Expected: all PASS including the new `tests/dashboard/` tests.

- [ ] **Step 2: Mypy**

```bash
conda activate nerfstudio && mypy collab_splats/dashboard/
```

Expected: no errors.

- [ ] **Step 3: Ruff + black**

```bash
conda activate nerfstudio && ruff check collab_splats/dashboard/ && black collab_splats/dashboard/
```

Expected: clean.

- [ ] **Step 4: Launch the app and verify UI manually**

```bash
conda activate nerfstudio && collab-dashboard semantics --base-dir /workspace/fieldwork-data
```

Open browser at `http://localhost:7860` and verify:
- Sidebar shows species dropdown populated from `/workspace/fieldwork-data/`
- Selecting species → date → video populates each select in turn
- Selecting a video with existing YAML (e.g. `birds/2024-02-06`) shows correct method/frame_proportion in config
- Selecting a video without YAML shows base defaults
- [Save YAML] writes `docs/splats/configs/datasets/{name}.yaml`
- [Reset to base] restores sliders
- Explore tab: [Extract Frames] → frame gallery appears, slider updates current frame
- PCA overlay renders when extractor is run
- No NumPy UserWarning in terminal output

- [ ] **Step 5: Final commit**

```bash
git add -p  # stage any remaining formatting changes
git commit -m "chore: linting and formatting for dashboard redesign"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Fix NumPy crash → Task 1
- [x] Base directory browser → Task 5 (`_refresh_species`, sidebar)
- [x] Video discovery from fieldwork-data layout → Task 3 (`video_discovery.py`)
- [x] YAML per-video read/write → Task 4 (`config_panel.py`) + Task 5 (`_save_yaml`, `_load_video`)
- [x] Video metadata (species, date visible in selects) → Task 5 sidebar
- [x] Splatter training config → Task 4 (`ConfigPanel` params)
- [x] Training launcher with live logs → Task 5 (`_launch_training`, `_poll_training_log`)
- [x] Semantic exploration tabs (PCA, seg, Talk2DINO) → Task 5 Explore tab
- [x] Dashboard tests → Tasks 1, 3, 4, 5 (`tests/dashboard/`)
- [x] Panel dependency added → Task 2
- [x] `--base-dir` CLI arg → Task 6
- [x] `loading_modal` pattern matches collab-data → Task 5 (`_show_loading`)

**Type consistency:**
- `discover_videos()` returns `dict[str, dict[str, list[Path]]]` — used consistently in `_refresh_species`, `_on_species_change`, `_on_date_change`, `_resolve_video_path`
- `yaml_name_for_video(species, date_dir, video_stem)` — called in `_load_video` and `_save_yaml` with same arg order
- `ConfigPanel.load_from_yaml(yaml_path: Path | None)` — called with `None` for base defaults and `yaml_path` for existing YAMLs
- `ConfigPanel.to_splatter_config(file_path, output_path)` — called in `_launch_training`
- `run_app(host, port, base_dir)` — signature matches `__main__.py` call
