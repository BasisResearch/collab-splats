# Splats Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework `collab_splats/dashboard/` into one streamlined Panel page: select a session/video (rclone via collab-data), configure in the sidebar, Run the primitives pipeline (sample → pointcloud → mesh → semantics), and view pointcloud/mesh beside a text-query similarity map.

**Architecture:** Four new focused modules — `sources.py` (GCS browse/transfer), `config.py` (typed run config), `pipeline.py` (background orchestrator), `viewer.py` (side-by-side PyVista) — assembled by a rewritten slim `app.py`. Heavy backend (frame sampling, creators, mesh, semantics) is reused unchanged. collab-data is a one-way pip dependency for its pure-Python `RcloneClient`.

**Tech Stack:** Panel + param, PyVista/VTK, zarr, rclone (via collab-data), pytest.

---

## File structure

| File | Responsibility |
|---|---|
| `collab_splats/dashboard/sources.py` | **Create.** `SessionSource`: list sessions/videos, fetch mp4 local, check/pull/push processed outputs via `RcloneClient`. |
| `collab_splats/dashboard/config.py` | **Create.** `RunConfig` dataclass: sampling method+params, env model+conf, extractor, mesh params, query. `to_yaml`/`from_yaml`. |
| `collab_splats/dashboard/pipeline.py` | **Create.** `run_pipeline(video_path, session, stem, config, op_log, source, base_dir)`: sample→pointcloud→mesh→semantics, write `run_config.yaml`, auto-push. Plus `_write_frames_zarr`. |
| `collab_splats/dashboard/viz_utils.py` | **Create.** Move `pointcloud_to_polydata` + `_apply_viridis` here (shared by viewer; currently in `panes/visualize.py`). |
| `collab_splats/dashboard/viewer.py` | **Create.** `SplitViewer`: two linked PyVista plotters — RGB (pcd\|mesh) left, similarity right; `load(result, mesh_path)`, `set_mode`, `query(text, extractor)`. |
| `collab_splats/dashboard/app.py` | **Rewrite.** Single-page assembly: sidebar (source + collapsible config + query + Run/Force) + `SplitViewer` + progress strip; caching wiring. |
| `collab_splats/dashboard/__main__.py` | **Modify.** Single `collab-dashboard` command → new app. |
| `collab_splats/dashboard/panes/` | **Delete** after helpers extracted: `preprocess.py`, `reconstruct.py`, `semantics.py`, `visualize.py`, `localize.py`, `_placeholder.py`. |
| `pyproject.toml` | **Modify.** Add `collab_data` dependency. |
| `tests/dashboard/` | **Modify/Create.** New tests for sources/config/pipeline/viewer/app; delete obsolete pane tests. |

**Reused unchanged:** `state.py` (AppState), `operation_log.py` (OperationLog), `video_server.py`, and all of `pointcloud/`, `mesh/`, `semantics/`, `utils/frame_sampling.py`.

---

## Task 1: Add collab-data dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, under `[project] dependencies`, add the collab-data package (editable sibling install). Add this line to the dependencies array:

```toml
  "collab_data @ file:///workspace/collab-data",
```

- [ ] **Step 2: Install and verify the import**

Run: `/opt/venv/reconstruction/bin/python -m pip install -e /workspace/collab-data --no-deps`
Then: `/opt/venv/reconstruction/bin/python -c "from collab_data.data_dashboard.rclone_client import RcloneClient; print('ok')"`
Expected: prints `ok` (use `--no-deps` to avoid pulling collab-data's heavy extras; we only need the pure-Python rclone module).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build(dashboard): depend on collab_data for rclone client"
```

---

## Task 2: `config.py` — typed RunConfig

**Files:**
- Create: `collab_splats/dashboard/config.py`
- Test: `tests/dashboard/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_config.py
from pathlib import Path

from collab_splats.dashboard.config import RunConfig


def test_runconfig_defaults():
    cfg = RunConfig()
    assert cfg.sampling_method == "balanced"
    assert cfg.max_frames == 50
    assert cfg.env_model == "vggt_omega"
    assert cfg.conf_threshold == 50.0
    assert cfg.semantic_extractor == "talk2dino"
    assert cfg.query == ""
    assert cfg.mesh_voxel_size == 0.01


def test_runconfig_yaml_roundtrip(tmp_path: Path):
    cfg = RunConfig(env_model="mapanything", conf_threshold=35.0, query="chair")
    cfg.frame_indices = [0, 5, 10]
    path = tmp_path / "run_config.yaml"
    cfg.to_yaml(path, video_ref="reconstruction/2026_05_07/clip_03.mp4")
    loaded = RunConfig.from_yaml(path)
    assert loaded.env_model == "mapanything"
    assert loaded.conf_threshold == 35.0
    assert loaded.query == "chair"
    assert loaded.frame_indices == [0, 5, 10]
    assert loaded.video_ref == "reconstruction/2026_05_07/clip_03.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.dashboard.config`.

- [ ] **Step 3: Write the implementation**

```python
# collab_splats/dashboard/config.py
"""Typed run configuration for the splats dashboard pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

########
# Config
########


@dataclass
class RunConfig:
    """All knobs for one pipeline run; serialised to run_config.yaml for provenance."""

    # Frame sampling
    sampling_method: str = "balanced"      # "balanced" | "optical_flow"
    max_frames: int = 50
    min_disparity: float = 50.0            # optical_flow only

    # Environment (pointcloud) model
    env_model: str = "vggt_omega"          # "vggt_omega" | "vggtx" | "mapanything"
    conf_threshold: float = 50.0

    # Semantics
    semantic_extractor: str = "talk2dino"
    query: str = ""

    # Mesh (TSDF) params
    mesh_voxel_size: float = 0.01
    mesh_sdf_trunc: float = 0.04
    mesh_depth_trunc: float = 10.0
    mesh_clean_repair: bool = True

    # Provenance — filled by the pipeline, not the UI
    frame_indices: list[int] = field(default_factory=list)
    video_ref: str = ""

    def to_yaml(self, path: Path, video_ref: str = "") -> None:
        """Write config (incl. provenance) to a YAML file."""
        data = asdict(self)
        if video_ref:
            data["video_ref"] = video_ref
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        """Load config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/config.py tests/dashboard/test_config.py
git commit -m "feat(dashboard): typed RunConfig with yaml provenance"
```

---

## Task 3: `sources.py` — GCS session browsing + transfer

`RcloneClient` API (from collab-data): `RcloneClient(remote_name="collab-data")`; `list_directory(bucket, path="") -> List[Dict]` (items have keys `Name`, `Size`, `IsDir`); `copy_local_to_remote(local_path, bucket, remote_path) -> bool`; `_cmd(*args) -> List[str]` (builds the `rclone` argv). The remote name is on `client.remote_name`.

**Files:**
- Create: `collab_splats/dashboard/sources.py`
- Test: `tests/dashboard/test_sources.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_sources.py
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from collab_splats.dashboard.sources import SessionSource


def _client():
    c = MagicMock()
    c.remote_name = "collab-data"
    c._cmd = lambda *a: ["rclone", *a]
    return c


def test_list_sessions_returns_dir_names_sorted():
    client = _client()
    client.list_directory.return_value = [
        {"Name": "2026_05_08", "IsDir": True},
        {"Name": "2026_05_07", "IsDir": True},
        {"Name": "notes.txt", "IsDir": False},
    ]
    src = SessionSource(client)
    assert src.list_sessions() == ["2026_05_07", "2026_05_08"]
    client.list_directory.assert_called_with("fieldwork_curated", "reconstruction")


def test_list_videos_filters_mp4():
    client = _client()
    client.list_directory.return_value = [
        {"Name": "clip_01.mp4", "IsDir": False},
        {"Name": "clip_02.MP4", "IsDir": False},
        {"Name": "meta.json", "IsDir": False},
        {"Name": "sub", "IsDir": True},
    ]
    src = SessionSource(client)
    assert src.list_videos("2026_05_07") == ["clip_01.mp4", "clip_02.MP4"]
    client.list_directory.assert_called_with(
        "fieldwork_curated", "reconstruction/2026_05_07"
    )


def test_fetch_video_invokes_rclone_copyto(monkeypatch, tmp_path):
    client = _client()
    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("collab_splats.dashboard.sources.subprocess.run", fake_run)
    src = SessionSource(client)
    local = src.fetch_video("2026_05_07", "clip_03.mp4", tmp_path)
    assert local == tmp_path / "clip_03.mp4"
    assert calls["cmd"] == [
        "rclone", "copyto",
        "collab-data:fieldwork_curated/reconstruction/2026_05_07/clip_03.mp4",
        str(tmp_path / "clip_03.mp4"),
    ]


def test_has_processed_true_when_listing_nonempty():
    client = _client()
    client.list_directory.return_value = [{"Name": "feedforward.zarr", "IsDir": True}]
    src = SessionSource(client)
    assert src.has_processed("2026_05_07", "clip_03") is True
    client.list_directory.assert_called_with(
        "fieldwork_processed", "reconstruction/2026_05_07/clip_03"
    )


def test_has_processed_false_on_error():
    client = _client()
    client.list_directory.side_effect = RuntimeError("not found")
    src = SessionSource(client)
    assert src.has_processed("2026_05_07", "clip_03") is False


def test_push_outputs_calls_copy_local_to_remote(tmp_path):
    client = _client()
    client.copy_local_to_remote.return_value = True
    src = SessionSource(client)
    src.push_outputs(tmp_path, "2026_05_07", "clip_03")
    client.copy_local_to_remote.assert_called_with(
        str(tmp_path), "fieldwork_processed", "reconstruction/2026_05_07/clip_03"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.dashboard.sources`.

- [ ] **Step 3: Write the implementation**

```python
# collab_splats/dashboard/sources.py
"""Browse and transfer reconstruction sessions/videos to and from GCS via rclone."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from collab_data.data_dashboard.rclone_client import RcloneClient

logger = logging.getLogger(__name__)

########
# Constants
########

CURATED_BUCKET = "fieldwork_curated"
PROCESSED_BUCKET = "fieldwork_processed"
ROOT = "reconstruction"
_VIDEO_EXTS = (".mp4", ".mov")

########
# Source
########


class SessionSource:
    """Lists sessions/videos under fieldwork_curated/reconstruction and moves outputs."""

    def __init__(self, client: RcloneClient | None = None) -> None:
        self._client = client if client is not None else RcloneClient()

    def list_sessions(self) -> list[str]:
        """Return sorted YYYY_MM_DD session directory names."""
        items = self._client.list_directory(CURATED_BUCKET, ROOT)
        return sorted(i["Name"] for i in items if i.get("IsDir"))

    def list_videos(self, session: str) -> list[str]:
        """Return video filenames under a session."""
        items = self._client.list_directory(CURATED_BUCKET, f"{ROOT}/{session}")
        return [
            i["Name"]
            for i in items
            if not i.get("IsDir") and i["Name"].lower().endswith(_VIDEO_EXTS)
        ]

    def fetch_video(self, session: str, name: str, dest_dir: Path) -> Path:
        """rclone-copy a remote video to dest_dir; return the local path."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{self._client.remote_name}:{CURATED_BUCKET}/{ROOT}/{session}/{name}"
        subprocess.run(
            self._client._cmd("copyto", remote, str(local)), check=True
        )
        return local

    def has_processed(self, session: str, stem: str) -> bool:
        """True if processed outputs already exist for this video."""
        try:
            items = self._client.list_directory(
                PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}"
            )
        except Exception:  # rclone errors when the path does not exist
            return False
        return bool(items)

    def pull_processed(self, session: str, stem: str, dest_dir: Path) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{self._client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        subprocess.run(
            self._client._cmd("copy", remote, str(dest_dir)), check=True
        )
        return dest_dir

    def push_outputs(self, local_dir: Path, session: str, stem: str) -> None:
        """rclone-copy the full local output tree to fieldwork_processed."""
        self._client.copy_local_to_remote(
            str(local_dir), PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/sources.py tests/dashboard/test_sources.py
git commit -m "feat(dashboard): SessionSource for rclone session browse + transfer"
```

---

## Task 4: Extract shared viz helpers into `viz_utils.py`

`panes/visualize.py` defines `pointcloud_to_polydata(points, RGB=...)` and `_apply_viridis(sims)`. The new viewer needs them after the panes are deleted. Move them to a standalone module.

**Files:**
- Create: `collab_splats/dashboard/viz_utils.py`
- Test: `tests/dashboard/test_viz_utils.py`

- [ ] **Step 1: Find the current definitions**

Run: `grep -n "def pointcloud_to_polydata\|def _apply_viridis" collab_splats/dashboard/panes/visualize.py`
Expected: two line numbers. Open those functions and copy their bodies verbatim into the new module below (replace the `...` placeholders with the real bodies you read).

- [ ] **Step 2: Write the failing test**

```python
# tests/dashboard/test_viz_utils.py
import numpy as np

from collab_splats.dashboard.viz_utils import pointcloud_to_polydata, apply_viridis


def test_polydata_has_points():
    pts = np.random.rand(10, 3).astype(np.float32)
    rgb = (np.random.rand(10, 3) * 255).astype(np.uint8)
    poly = pointcloud_to_polydata(pts, RGB=rgb)
    assert poly.n_points == 10


def test_apply_viridis_shape_and_dtype():
    sims = np.linspace(-1.0, 1.0, 12).astype(np.float32)
    rgb = apply_viridis(sims)
    assert rgb.shape == (12, 3)
    assert rgb.dtype == np.uint8
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viz_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.dashboard.viz_utils`.

- [ ] **Step 4: Write the module**

```python
# collab_splats/dashboard/viz_utils.py
"""Shared PyVista/colour helpers for the dashboard viewer."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from matplotlib import cm


def pointcloud_to_polydata(points: np.ndarray, RGB: np.ndarray | None = None) -> "pv.PolyData":
    """Build a PyVista point cloud; attach RGB scalars if given.

    (Paste the verbatim body from panes/visualize.py here.)
    """
    ...


def apply_viridis(sims: np.ndarray) -> np.ndarray:
    """Map a (P,) similarity array to (P, 3) uint8 viridis colours.

    (Paste the verbatim body of _apply_viridis from panes/visualize.py here,
    renamed to apply_viridis — drop the leading underscore.)
    """
    ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viz_utils.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/viz_utils.py tests/dashboard/test_viz_utils.py
git commit -m "refactor(dashboard): extract pointcloud_to_polydata + apply_viridis to viz_utils"
```

---

## Task 5: `pipeline.py` — background run orchestrator

Backend APIs (verbatim): `sample_frames_fps(video_path, fps, on_progress=None, max_frames=None, verbose=True) -> (frames, indices)`; `sample_frames_optical_flow(video_path, min_disparity=50.0, max_frames=200, ...) -> (frames, scores)`; `get_video_info(video_path) -> {"total_frames","fps","duration_s","width","height"}`. Creators: `VGGTOmegaCreator(conf_threshold=...)`, `VGGTXCreator(conf_threshold=...)`, `MapAnythingCreator(confidence_percentile=...)`; `creator.reconstruct(image_dir: Path, output_dir: Path)`; `creator.outputs` is a `FeedforwardResult`; `FeedforwardResult.save_zarr(path)`. Mesh: `pointcloud_to_mesh(result, output_dir, method="open3d_tsdf", voxel_size=, sdf_trunc=, depth_trunc=, clean_repair=)`. Semantics: `BaseFeatureExtractor.get(name)() ` then `extract_and_cache(image_paths, cache_dir)`. `OperationLog`: `start_op(name)`, `update_progress(pct, message="")`, `finish_op()`, `error_op(error_msg)`.

The creators consume an **image directory**, not frames in memory. So the pipeline writes sampled frames as JPEGs into `frames/` (and also `frames.zarr` for the viewer's RGB/lift path).

**Files:**
- Create: `collab_splats/dashboard/pipeline.py`
- Test: `tests/dashboard/test_pipeline.py`

- [ ] **Step 1: Write the failing test (mock all heavy work)**

```python
# tests/dashboard/test_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard import pipeline as pl


def _fake_frames(n=3):
    return [np.zeros((4, 4, 3), np.uint8) for _ in range(n)], [0, 1, 2]


def test_run_pipeline_orders_steps_and_pushes(tmp_path):
    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig(env_model="vggt_omega", semantic_extractor="talk2dino")
    video = tmp_path / "clip_03.mp4"
    video.write_bytes(b"x")

    fake_result = MagicMock()
    creator = MagicMock()
    creator.outputs = fake_result

    with patch.object(pl, "sample_frames_fps", return_value=_fake_frames()), \
         patch.object(pl, "get_video_info", return_value={"duration_s": 3.0, "fps": 30}), \
         patch.object(pl, "_write_frames_zarr") as wz, \
         patch.object(pl, "_write_frames_jpegs", return_value=tmp_path / "frames"), \
         patch.object(pl, "_build_creator", return_value=creator), \
         patch.object(pl, "pointcloud_to_mesh") as mesh, \
         patch.object(pl, "_extract_semantics") as sem:
        out = pl.run_pipeline(
            video_path=video, session="2026_05_07", stem="clip_03",
            config=cfg, op_log=op_log, source=source, base_dir=tmp_path / "outputs",
        )

    # outputs land in base_dir/session/stem
    assert out == tmp_path / "outputs" / "2026_05_07" / "clip_03"
    # each stage ran
    wz.assert_called_once()
    creator.reconstruct.assert_called_once()
    fake_result.save_zarr.assert_called_once()
    mesh.assert_called_once()
    sem.assert_called_once()
    # run_config.yaml written with frame indices
    cfg_path = out / "run_config.yaml"
    assert cfg_path.exists()
    loaded = RunConfig.from_yaml(cfg_path)
    assert loaded.frame_indices == [0, 1, 2]
    # auto-push on success
    source.push_outputs.assert_called_with(out, "2026_05_07", "clip_03")
    op_log.finish_op.assert_called_once()


def test_run_pipeline_does_not_push_on_failure(tmp_path):
    op_log = MagicMock()
    source = MagicMock()
    cfg = RunConfig()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    with patch.object(pl, "sample_frames_fps", return_value=_fake_frames()), \
         patch.object(pl, "get_video_info", return_value={"duration_s": 3.0, "fps": 30}), \
         patch.object(pl, "_write_frames_zarr"), \
         patch.object(pl, "_write_frames_jpegs", return_value=tmp_path / "frames"), \
         patch.object(pl, "_build_creator", side_effect=RuntimeError("boom")):
        with __import__("pytest").raises(RuntimeError):
            pl.run_pipeline(
                video_path=video, session="s", stem="clip",
                config=cfg, op_log=op_log, source=source, base_dir=tmp_path / "o",
            )

    source.push_outputs.assert_not_called()
    op_log.error_op.assert_called_once()


def test_build_creator_maps_models():
    with patch.object(pl, "VGGTOmegaCreator") as omega:
        pl._build_creator("vggt_omega", 50.0)
        omega.assert_called_with(conf_threshold=50.0)
    with patch.object(pl, "MapAnythingCreator") as ma:
        pl._build_creator("mapanything", 35.0)
        ma.assert_called_with(confidence_percentile=35.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.dashboard.pipeline`.

- [ ] **Step 3: Write the implementation**

```python
# collab_splats/dashboard/pipeline.py
"""Run the primitives pipeline for one video: sample -> pointcloud -> mesh -> semantics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource
from collab_splats.mesh.utils import pointcloud_to_mesh
from collab_splats.pointcloud.feedforward import (
    MapAnythingCreator,
    VGGTOmegaCreator,
    VGGTXCreator,
)
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.utils.frame_sampling import (
    get_video_info,
    sample_frames_fps,
    sample_frames_optical_flow,
)

logger = logging.getLogger(__name__)

########
# Helpers
########


def _write_frames_zarr(frames: list[np.ndarray], path: Path) -> None:
    """Write RGB frames (N, H, W, 3) uint8 to a zarr group at path/frames."""
    arr = np.stack(frames).astype(np.uint8)
    root = zarr.open_group(str(path), mode="w")
    root.create_array("frames", shape=arr.shape, dtype=arr.dtype, data=arr)


def _write_frames_jpegs(frames: list[np.ndarray], frames_dir: Path) -> Path:
    """Write frames as zero-padded JPEGs for creators that consume an image dir."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(frames_dir / f"{i:05d}.jpg")
    return frames_dir


def _build_creator(env_model: str, conf: float):
    """Instantiate the selected feedforward creator with its confidence arg."""
    if env_model == "vggt_omega":
        return VGGTOmegaCreator(conf_threshold=conf)
    if env_model == "vggtx":
        return VGGTXCreator(conf_threshold=conf)
    if env_model == "mapanything":
        return MapAnythingCreator(confidence_percentile=conf)
    raise ValueError(f"unknown env_model: {env_model}")


def _extract_semantics(extractor_name: str, image_dir: Path, out_dir: Path) -> None:
    """Extract + cache patch features for the sampled frames."""
    extractor = BaseFeatureExtractor.get(extractor_name)()
    image_paths = sorted(image_dir.glob("*.jpg"))
    extractor.extract_and_cache(image_paths, out_dir)


def _sample(video_path: Path, config: RunConfig, op_log: OperationLog):
    """Sample frames per the configured method; return (frames, indices)."""
    info = get_video_info(str(video_path))

    def on_progress(done: int, total: int) -> None:
        op_log.update_progress(int(5 + 15 * done / max(total, 1)), "sampling frames")

    if config.sampling_method == "optical_flow":
        frames, _ = sample_frames_optical_flow(
            str(video_path),
            min_disparity=config.min_disparity,
            max_frames=config.max_frames,
            on_progress=on_progress,
            verbose=False,
        )
        indices = list(range(len(frames)))
    else:
        duration_s = info.get("duration_s") or (
            info["total_frames"] / (info.get("fps") or 30.0)
        )
        target_fps = config.max_frames / max(duration_s, 1.0)
        frames, indices = sample_frames_fps(
            str(video_path),
            fps=target_fps,
            max_frames=config.max_frames,
            on_progress=on_progress,
            verbose=False,
        )
    return frames, indices


########
# Orchestrator
########


def run_pipeline(
    *,
    video_path: Path,
    session: str,
    stem: str,
    config: RunConfig,
    op_log: OperationLog,
    source: SessionSource,
    base_dir: Path,
) -> Path:
    """Execute the full pipeline; write outputs under base_dir/session/stem; push on success."""
    out_dir = Path(base_dir) / session / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    op_log.start_op(f"{session}/{stem}")
    try:
        # 1. sample frames
        frames, indices = _sample(Path(video_path), config, op_log)
        _write_frames_zarr(frames, out_dir / "frames.zarr")
        image_dir = _write_frames_jpegs(frames, out_dir / "frames")
        config.frame_indices = list(indices)

        # 2. pointcloud (environment model)
        op_log.update_progress(25, f"pointcloud: {config.env_model}")
        creator = _build_creator(config.env_model, config.conf_threshold)
        creator.reconstruct(image_dir, out_dir)
        result = creator.outputs
        result.save_zarr(out_dir / "feedforward.zarr")

        # 3. TSDF mesh
        op_log.update_progress(60, "mesh (TSDF)")
        pointcloud_to_mesh(
            result,
            out_dir / "mesh",
            method="open3d_tsdf",
            voxel_size=config.mesh_voxel_size,
            sdf_trunc=config.mesh_sdf_trunc,
            depth_trunc=config.mesh_depth_trunc,
            clean_repair=config.mesh_clean_repair,
        )

        # 4. semantic features
        op_log.update_progress(80, f"semantics: {config.semantic_extractor}")
        _extract_semantics(
            config.semantic_extractor, image_dir, out_dir / "semantics"
        )

        # 5. provenance + push
        config.to_yaml(
            out_dir / "run_config.yaml",
            video_ref=f"reconstruction/{session}/{stem}/{Path(video_path).name}",
        )
        op_log.update_progress(95, "pushing to fieldwork_processed")
        source.push_outputs(out_dir, session, stem)
        op_log.finish_op()
        return out_dir
    except Exception as exc:  # surface, do not push partial outputs
        logger.exception("pipeline failed")
        op_log.error_op(str(exc))
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Verify `extract_and_cache` argument names**

Run: `grep -n "def extract_and_cache" collab_splats/semantics/features/base.py`
Expected: confirm signature is `extract_and_cache(self, image_paths, cache_dir, batch_size=1, skip_existing=True)`. If the positional names differ, adjust `_extract_semantics` accordingly, then re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/pipeline.py tests/dashboard/test_pipeline.py
git commit -m "feat(dashboard): background pipeline orchestrator with provenance + auto-push"
```

---

## Task 6: `viewer.py` — side-by-side split viewer

Two PyVista plotters in a Panel `Row`, cameras linked. Left renders RGB pointcloud or mesh; right renders the same pointcloud recoloured by text-query cosine similarity. Lift per-frame features to points with `lift_features(feature_maps, result, depth_tol=0.05) -> (P, D)`; text encode with a `BaseQueryableExtractor` via `encode_text([query]) -> (1, D)`.

**Files:**
- Create: `collab_splats/dashboard/viewer.py`
- Test: `tests/dashboard/test_viewer.py`

- [ ] **Step 1: Write the failing test (off-screen, synthetic data)**

```python
# tests/dashboard/test_viewer.py
import numpy as np
import pytest

from collab_splats.dashboard.viewer import SplitViewer


class _FakeResult:
    def __init__(self, p=20):
        self.points = np.random.rand(p, 3).astype(np.float32)
        self.colors = (np.random.rand(p, 3) * 255).astype(np.uint8)


def test_viewer_loads_pointcloud_offscreen():
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    assert v.left_actor is not None


def test_set_mode_pcd_to_mesh_toggles(tmp_path):
    v = SplitViewer(off_screen=True)
    v.load(_FakeResult(), mesh_path=None)
    v.set_mode("mesh")          # no mesh.ply -> status set, no crash
    assert v.mode == "mesh"
    v.set_mode("pointcloud")
    assert v.mode == "pointcloud"


def test_recolor_by_similarity_updates_right(monkeypatch):
    v = SplitViewer(off_screen=True)
    res = _FakeResult(p=20)
    v.load(res, mesh_path=None)
    # Inject lifted features directly (normalised) and a fake extractor
    v._lifted_normed = np.random.rand(20, 8).astype(np.float32)

    class _Ext:
        def encode_text(self, texts):
            import torch
            return torch.ones(1, 8)

    monkeypatch.setattr(v, "_get_extractor", lambda name: _Ext())
    colors = v.query("chair", extractor_name="talk2dino")
    assert colors.shape == (20, 3)
    assert colors.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.dashboard.viewer`.

- [ ] **Step 3: Write the implementation**

```python
# collab_splats/dashboard/viewer.py
"""Side-by-side PyVista viewer: RGB pointcloud/mesh (left) + similarity heatmap (right)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import panel as pn
import pyvista as pv

from collab_splats.dashboard.viz_utils import apply_viridis, pointcloud_to_polydata
from collab_splats.semantics.features.base import BaseQueryableExtractor

logger = logging.getLogger(__name__)


class SplitViewer:
    """Two linked plotters; left = RGB pcd/mesh, right = query similarity."""

    def __init__(self, off_screen: bool = False) -> None:
        self._left = pv.Plotter(off_screen=off_screen)
        self._right = pv.Plotter(off_screen=off_screen)
        self._left_pane = pn.pane.VTK(self._left.ren_win, sizing_mode="stretch_both", min_height=500)
        self._right_pane = pn.pane.VTK(self._right.ren_win, sizing_mode="stretch_both", min_height=500)
        self.layout = pn.Row(self._left_pane, self._right_pane, sizing_mode="stretch_both")

        self.mode = "pointcloud"
        self.left_actor = None
        self._right_actor = None
        self._result = None
        self._mesh_path: Path | None = None
        self._lifted_normed: np.ndarray | None = None
        self._extractor_cache: dict = {}
        self._status = ""

    # ---- loading -------------------------------------------------------

    def load(self, result, mesh_path: Path | None, lifted_normed: np.ndarray | None = None) -> None:
        """Load a FeedforwardResult (+ optional mesh + lifted features) into both panes."""
        self._result = result
        self._mesh_path = Path(mesh_path) if mesh_path else None
        self._lifted_normed = lifted_normed
        self._render_left()
        self._render_right(None)

    def _render_left(self) -> None:
        self._left.clear()
        if self.mode == "mesh" and self._mesh_path and self._mesh_path.exists():
            self.left_actor = self._left.add_mesh(pv.read(str(self._mesh_path)), rgb=True)
        else:
            if self.mode == "mesh":
                self._status = "mesh.ply not found."
            cloud = pointcloud_to_polydata(self._result.points, RGB=self._result.colors)
            self.left_actor = self._left.add_mesh(
                cloud, scalars="RGB", rgb=True, point_size=2
            )
        self._left_pane.synchronize()

    def _render_right(self, colors: np.ndarray | None) -> None:
        self._right.clear()
        rgb = colors if colors is not None else self._result.colors
        cloud = pointcloud_to_polydata(self._result.points, RGB=rgb)
        self._right_actor = self._right.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)
        # Link the right camera to the left so the two views stay in sync.
        self._right.camera = self._left.camera
        self._right_pane.synchronize()

    # ---- interactions --------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch the left pane between 'pointcloud' and 'mesh'."""
        self.mode = mode
        if self._result is not None:
            self._render_left()

    def _get_extractor(self, name: str):
        """Construct (and cache) a queryable extractor by registry name."""
        if name not in self._extractor_cache:
            self._extractor_cache[name] = BaseQueryableExtractor.get(name)()
        return self._extractor_cache[name]

    def query(self, text: str, extractor_name: str) -> np.ndarray:
        """Recolour the right pane by cosine similarity to the text query; return colours."""
        if not text or self._lifted_normed is None:
            self._render_right(None)
            return self._result.colors
        extractor = self._get_extractor(extractor_name)
        text_emb = extractor.encode_text([text])           # (1, D) torch
        vec = text_emb.detach().cpu().numpy()[0]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = self._lifted_normed @ vec                   # (P,)
        colors = apply_viridis(sims)
        self._render_right(colors)
        return colors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer.py -v`
Expected: PASS (3 passed). If VTK requires a display, set `PYVISTA_OFF_SCREEN=true` and `export DISPLAY=` is unset — off_screen plotters use OSMesa; the existing `tests/dashboard/test_visualize.py` already runs off-screen, so the harness supports it.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer.py
git commit -m "feat(dashboard): SplitViewer with synced RGB + similarity panes"
```

---

## Task 7: Helper to load lifted features for the viewer

The viewer needs per-point normalised features to query. Build a small loader that reads cached patch features + the FeedforwardResult and calls `lift_features`.

**Files:**
- Modify: `collab_splats/dashboard/viewer.py` (add module-level function)
- Test: `tests/dashboard/test_viewer_lift.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dashboard/test_viewer_lift.py
from unittest.mock import patch

import numpy as np
import torch

from collab_splats.dashboard.viewer import load_lifted_normed


def test_load_lifted_normed_normalises():
    fake_lifted = torch.tensor([[3.0, 4.0], [0.0, 2.0]])  # norms 5, 2
    with patch("collab_splats.dashboard.viewer.lift_features", return_value=fake_lifted), \
         patch("collab_splats.dashboard.viewer._load_feature_maps", return_value=["fm"]):
        out = load_lifted_normed(result=object(), semantics_dir=object())
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer_lift.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_lifted_normed'`.

- [ ] **Step 3: Confirm the cached-feature layout, then implement**

Run: `grep -n "def extract_and_cache\b" collab_splats/semantics/features/base.py` and read the docstring for the zarr layout (features `(N, D, H_p, W_p)`, one chunk per frame). Then add to `viewer.py`:

```python
# Add near the top of collab_splats/dashboard/viewer.py
import zarr
import torch
from collab_splats.pointcloud.utils import lift_features


def _load_feature_maps(semantics_dir) -> list:
    """Load per-frame dense feature maps (D, H_p, W_p) from the cached zarr."""
    # The extractor writes cache_dir/{name}.zarr with array 'features' (N, D, H_p, W_p).
    store = next(Path(semantics_dir).glob("*.zarr"))
    arr = zarr.open_array(str(store / "features"), mode="r")
    return [torch.from_numpy(np.asarray(arr[i])) for i in range(arr.shape[0])]


def load_lifted_normed(result, semantics_dir) -> np.ndarray:
    """Lift cached features to points and L2-normalise → (P, D) float32."""
    feature_maps = _load_feature_maps(semantics_dir)
    lifted = lift_features(feature_maps, result)          # (P, D) torch
    lifted = lifted.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(lifted, axis=1, keepdims=True)
    return lifted / (norms + 1e-8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_viewer_lift.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/viewer.py tests/dashboard/test_viewer_lift.py
git commit -m "feat(dashboard): load + normalise lifted features for similarity query"
```

---

## Task 8: Rewrite `app.py` — single-page assembly

Build the sidebar, the `SplitViewer`, and the progress strip. Wire: session select → video select → (cached? load : enable Run) → Run/Force (background thread → `run_pipeline`) → on done load viewer; query box → `viewer.query`. Reuse `AppState` and `OperationLog`.

**Files:**
- Rewrite: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py` (replace existing)

- [ ] **Step 1: Replace the app test with construction + wiring tests**

```python
# tests/dashboard/test_app.py
from unittest.mock import MagicMock, patch

from collab_splats.dashboard.app import SplatsApp


def _app(tmp_path):
    source = MagicMock()
    source.list_sessions.return_value = ["2026_05_07"]
    source.list_videos.return_value = ["clip_03.mp4"]
    source.has_processed.return_value = False
    with patch("collab_splats.dashboard.app.SplitViewer"):
        return SplatsApp(base_dir=tmp_path, source=source), source


def test_app_populates_sessions(tmp_path):
    app, source = _app(tmp_path)
    assert app.session_select.options == ["2026_05_07"]


def test_selecting_session_lists_videos(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    assert "clip_03.mp4" in app.video_select.options


def test_run_button_spawns_pipeline(tmp_path):
    app, source = _app(tmp_path)
    app.session_select.value = "2026_05_07"
    app.video_select.value = "clip_03.mp4"
    with patch("collab_splats.dashboard.app.threading.Thread") as thread, \
         patch.object(app, "_ensure_local_video", return_value=tmp_path / "clip_03.mp4"):
        app._on_run(event=None, force=True)
    thread.assert_called_once()


def test_view(tmp_path):
    app, _ = _app(tmp_path)
    assert app.view() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: FAIL — `ImportError: cannot import name 'SplatsApp'`.

- [ ] **Step 3: Write the new app**

```python
# collab_splats/dashboard/app.py
"""Streamlined single-page splats dashboard: select -> configure -> run -> visualize."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import panel as pn
import param

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.pipeline import run_pipeline
from collab_splats.dashboard.sources import SessionSource
from collab_splats.dashboard.viewer import SplitViewer, load_lifted_normed
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

logger = logging.getLogger(__name__)

pn.extension("vtk")

_ENV_MODELS = ["vggt_omega", "vggtx", "mapanything"]
_EXTRACTORS = ["talk2dino", "maskclip", "dinov2"]
_SAMPLERS = ["balanced", "optical_flow"]

########
# App
########


class SplatsApp(param.Parameterized):
    """Single-page dashboard wiring source, pipeline, and the split viewer."""

    def __init__(self, base_dir: Path = Path("/workspace/outputs"),
                 source: SessionSource | None = None, **params) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SessionSource()
        self._op_log = OperationLog()
        self._viewer = SplitViewer()
        self._build_sidebar()
        self._refresh_sessions()

    # ---- sidebar -------------------------------------------------------

    def _build_sidebar(self) -> None:
        self.session_select = pn.widgets.Select(name="Session", options=[])
        self.video_select = pn.widgets.Select(name="Video", options=[])
        self.sampling = pn.widgets.Select(name="Frame sampling", options=_SAMPLERS, value="balanced")
        self.max_frames = pn.widgets.IntSlider(name="Max frames", start=10, end=200, value=50)
        self.env_model = pn.widgets.Select(name="Environment model", options=_ENV_MODELS, value="vggt_omega")
        self.conf = pn.widgets.FloatSlider(name="Confidence", start=0, end=100, value=50.0)
        self.extractor = pn.widgets.Select(name="Semantic model", options=_EXTRACTORS, value="talk2dino")
        self.query = pn.widgets.TextInput(name="Query", placeholder="e.g. chair")
        self.mesh_voxel = pn.widgets.FloatInput(name="voxel_size", value=0.01)
        self.mesh_sdf = pn.widgets.FloatInput(name="sdf_trunc", value=0.04)
        self.mesh_depth = pn.widgets.FloatInput(name="depth_trunc", value=10.0)
        self.view_mode = pn.widgets.RadioButtonGroup(options=["pointcloud", "mesh"], value="pointcloud")
        self.run_btn = pn.widgets.Button(name="Run", button_type="primary")
        self.force_btn = pn.widgets.Button(name="Force re-run", button_type="warning")

        self.session_select.param.watch(self._on_session, "value")
        self.video_select.param.watch(self._on_video, "value")
        self.query.param.watch(self._on_query, "value")
        self.view_mode.param.watch(lambda e: self._viewer.set_mode(e.new), "value")
        self.run_btn.on_click(lambda e: self._on_run(e, force=False))
        self.force_btn.on_click(lambda e: self._on_run(e, force=True))

        self._sidebar = pn.Column(
            "## Source", self.session_select, self.video_select,
            pn.Card(self.sampling, self.max_frames, title="Frame sampling", collapsed=True),
            pn.Card(self.env_model, self.conf, title="Environment model", collapsed=True),
            pn.Card(self.extractor, self.query, title="Semantics", collapsed=False),
            pn.Card(self.mesh_voxel, self.mesh_sdf, self.mesh_depth, title="Mesh params", collapsed=True),
            "### View", self.view_mode,
            pn.Row(self.run_btn, self.force_btn),
        )

    # ---- data wiring ---------------------------------------------------

    def _refresh_sessions(self) -> None:
        try:
            self.session_select.options = self._source.list_sessions()
        except Exception as exc:  # rclone unavailable -> degrade
            logger.warning("session listing failed: %s", exc)
            self.session_select.options = []

    def _on_session(self, event) -> None:
        if not event.new:
            return
        self.video_select.options = self._source.list_videos(event.new)

    def _on_video(self, event) -> None:
        if not event.new:
            return
        session, stem = self.session_select.value, Path(event.new).stem
        out = self._base_dir / session / stem
        if (out / "feedforward.zarr").exists() or self._source.has_processed(session, stem):
            self._load_outputs(session, stem)

    def _current_config(self) -> RunConfig:
        return RunConfig(
            sampling_method=self.sampling.value,
            max_frames=self.max_frames.value,
            env_model=self.env_model.value,
            conf_threshold=self.conf.value,
            semantic_extractor=self.extractor.value,
            query=self.query.value,
            mesh_voxel_size=self.mesh_voxel.value,
            mesh_sdf_trunc=self.mesh_sdf.value,
            mesh_depth_trunc=self.mesh_depth.value,
        )

    def _ensure_local_video(self, session: str, name: str) -> Path:
        cache = self._base_dir / session / Path(name).stem / "frames"
        local = self._base_dir / session / Path(name).stem / name
        if not local.exists():
            local = self._source.fetch_video(session, name, local.parent)
        return local

    def _on_run(self, event, force: bool) -> None:
        session, name = self.session_select.value, self.video_select.value
        if not session or not name:
            return
        stem = Path(name).stem
        config = self._current_config()

        def worker() -> None:
            video = self._ensure_local_video(session, name)
            run_pipeline(
                video_path=video, session=session, stem=stem, config=config,
                op_log=self._op_log, source=self._source, base_dir=self._base_dir,
            )
            self._load_outputs(session, stem)

        threading.Thread(target=worker, daemon=True).start()

    def _load_outputs(self, session: str, stem: str) -> None:
        out = self._base_dir / session / stem
        if not (out / "feedforward.zarr").exists():
            self._source.pull_processed(session, stem, out)
        result = FeedforwardResult.load_zarr(out / "feedforward.zarr")
        try:
            lifted = load_lifted_normed(result, out / "semantics")
        except Exception:  # semantics optional for viewing
            lifted = None
        mesh_path = out / "mesh" / "mesh.ply"
        self._viewer.load(result, mesh_path=mesh_path if mesh_path.exists() else None,
                           lifted_normed=lifted)

    def _on_query(self, event) -> None:
        self._viewer.query(event.new, extractor_name=self.extractor.value)

    # ---- layout --------------------------------------------------------

    def view(self) -> pn.template.MaterialTemplate:
        """Assemble the full page."""
        progress = pn.Column(
            pn.indicators.Progress(value=self._op_log.param.progress, width=400),
            pn.pane.Str(self._op_log.param.current_op),
        )
        main = pn.Column(self._viewer.layout, progress, sizing_mode="stretch_both")
        return pn.template.MaterialTemplate(
            title="splats", sidebar=[self._sidebar], main=[main],
            header_background="#2596be", sidebar_width=340,
        )


def run_app(host: str = "0.0.0.0", port: int = 7860,
            base_dir: str = "/workspace/outputs") -> None:
    """Serve the splats dashboard."""
    def factory() -> pn.template.MaterialTemplate:
        return SplatsApp(base_dir=Path(base_dir)).view()

    pn.serve(factory, address=host, port=port, show=False, title="splats")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): rewrite app as single-page splats dashboard"
```

---

## Task 9: Update CLI entry point

**Files:**
- Modify: `collab_splats/dashboard/__main__.py`

- [ ] **Step 1: Replace the mode-based CLI with a single command**

```python
# collab_splats/dashboard/__main__.py
"""CLI entry point for the splats dashboard."""

import argparse

from collab_splats.dashboard.app import run_app


def main() -> None:
    """Parse args and serve the dashboard."""
    parser = argparse.ArgumentParser(
        prog="collab-dashboard", description="Launch the collab-splats dashboard."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-dir", default="/workspace/outputs")
    args = parser.parse_args()
    run_app(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports and shows help**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --help`
Expected: argparse help with `--host`, `--port`, `--base-dir` (no `mode` argument).

- [ ] **Step 3: Commit**

```bash
git add collab_splats/dashboard/__main__.py
git commit -m "feat(dashboard): single collab-dashboard CLI command"
```

---

## Task 10: Delete obsolete panes and their tests

The single-page app no longer uses the tabbed panes. Their reusable helpers were moved in Tasks 4–7.

**Files:**
- Delete: `collab_splats/dashboard/panes/` (whole directory) and `__init__.py` references
- Delete: `tests/dashboard/test_preprocess.py`, `test_reconstruct_pane.py`, `test_visualize.py`, `test_localize.py`, `test_localize_pane.py`, `test_smoke.py`

- [ ] **Step 1: Check for stray imports of the panes**

Run: `grep -rn "dashboard.panes\|from .panes\|LocalizePane\|SemanticsPane\|ScenePanel\|PreprocessPane\|ReconstructPane" collab_splats/ tests/ | grep -v "\.pyc"`
Expected: only references inside `panes/` itself and `dashboard/__init__.py`. Note any others — fix them in Step 3.

- [ ] **Step 2: Read and clean `dashboard/__init__.py`**

Open `collab_splats/dashboard/__init__.py`. Remove exports of `SemanticsPane`, `LocalizePane`, and any pane imports. Keep `App`/`run_app` updated to the new names (`SplatsApp`, `run_app`). Set:

```python
# collab_splats/dashboard/__init__.py
"""Splats dashboard package."""

from collab_splats.dashboard.app import SplatsApp, run_app
from collab_splats.dashboard.operation_log import OperationLog

__all__ = ["SplatsApp", "run_app", "OperationLog"]
```

- [ ] **Step 3: Delete the panes and obsolete tests**

```bash
git rm -r collab_splats/dashboard/panes
git rm tests/dashboard/test_preprocess.py tests/dashboard/test_reconstruct_pane.py \
       tests/dashboard/test_visualize.py tests/dashboard/test_localize.py \
       tests/dashboard/test_localize_pane.py tests/dashboard/test_smoke.py
```

If Step 1 found references outside `panes/`, fix them now.

- [ ] **Step 4: Run the full dashboard test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -v`
Expected: PASS — only the new tests (config, sources, viz_utils, pipeline, viewer, viewer_lift, app) collected; no import errors.

- [ ] **Step 5: Commit**

```bash
git add -A collab_splats/dashboard tests/dashboard
git commit -m "refactor(dashboard): remove tabbed panes superseded by single-page app"
```

---

## Task 11: Full-suite regression + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: no new failures vs the baseline in `docs/known-test-failures.md`. Investigate any dashboard-related failure.

- [ ] **Step 2: Format**

Run: `black collab_splats/dashboard tests/dashboard && isort collab_splats/dashboard tests/dashboard`
Expected: files reformatted/clean.

- [ ] **Step 3: Manual smoke (requires rclone creds + GPU)**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860`
Then in a browser: select a session → a video → open the Semantics card, type a query → Run → confirm progress advances and the split viewer shows pointcloud (left) + similarity (right). Toggle pointcloud/mesh. Note: run in tmux, not a notebook (memory cap).

- [ ] **Step 4: Commit any formatting**

```bash
git add -A
git commit -m "style(dashboard): black + isort"
```

---

## Self-review notes

- **Spec coverage:** session browse (Task 3) · sample→pointcloud→mesh→semantics (Task 5) · side-by-side viewer + text query (Tasks 6–7) · collapsible sidebar config + defaults vggt_omega/talk2dino (Task 8) · caching load-if-exists + Force re-run (Task 8) · provenance frame_indices + full-tree auto-push (Task 5) · processed path mirror (Task 3) · standalone CLI (Task 9) · drop panes/localize/BA-LC (Task 10).
- **Known risk to validate during impl:** `extract_and_cache` arg names (Task 5 Step 5) and the cached-feature zarr layout (Task 7 Step 3) are confirmed by reading before coding. Mesh runs inline (not the old subprocess isolation) — acceptable for v1; if OOM appears, wrap `pointcloud_to_mesh` in a `multiprocessing.Process` as a follow-up.
- **Deferred (spec non-goals):** BA/LC, localize, collab-data nav entry, webapp/ removal.
```
