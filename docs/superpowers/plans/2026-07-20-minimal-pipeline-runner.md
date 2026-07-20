# Minimal Pipeline Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-video scene launcher and a localization-database pipeline stage, reusing the existing `Reconstructor`.

**Architecture:** Wire a new `localize` stage into `Reconstructor` (builds the per-frame local-feature cache inside `feedforward.zarr`), then add a thin `run_scenes.py` that maps each input video to a `Reconstructor` run under a common `--output-root`. Fix the `localization` block in `base.yaml`. Refresh the two existing example scripts.

**Tech Stack:** Python 3.11, argparse, PyYAML, mergedeep, zarr, pytest. Interpreter: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-07-20-minimal-pipeline-runner-design.md`

---

## Task 1: Fix localization config in base.yaml

**Files:**
- Modify: `configs/base.yaml:41-43`

- [ ] **Step 1: Edit the localization block**

Replace the current block:
```yaml
localization:
  enabled: false
  extractor: dinosalad
```
with:
```yaml
localization:
  enabled: false
  extractor: loma          # local matcher registry key (dinosalad was invalid — it is a RETRIEVAL extractor)
  radius: 8.0              # CameraLocalizer search radius
```

- [ ] **Step 2: Sanity-check YAML parses**

Run: `/opt/venv/reconstruction/bin/python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['localization'])"`
Expected: `{'enabled': False, 'extractor': 'loma', 'radius': 8.0}`

- [ ] **Step 3: Commit**

```bash
git add configs/base.yaml
git commit -m "fix(config): localization.extractor loma + radius (dinosalad was a retrieval extractor)"
```

---

## Task 2: Add localize stage to Reconstructor

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (constants `_STAGE_ORDER`/`_STAGE_DEPS` ~L27-33; add helpers after `_run_tsdf_mesh` ~L263; add method + remove dead `localize` stub ~L642-644; `run_pipeline` ~L646-690)
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/wrapper/test_reconstructor.py`:
```python
########################################
# Localization stage
########################################

def test_localize_in_stage_order_and_deps():
    from collab_splats.wrapper import reconstructor as R
    assert "localize" in R._STAGE_ORDER
    assert R._STAGE_DEPS["localize"] == ["pointcloud"]


def test_run_pipeline_auto_includes_localize_when_enabled(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
    rec = Reconstructor(config)
    called = []
    with patch.object(rec, "preprocess"), \
         patch.object(rec, "build_pointcloud", return_value=None), \
         patch.object(rec, "build_localization_db", side_effect=lambda **k: called.append("localize")):
        rec.run_pipeline()
    assert called == ["localize"]


def test_run_pipeline_omits_localize_when_disabled(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": False}})
    rec = Reconstructor(config)
    called = []
    with patch.object(rec, "preprocess"), \
         patch.object(rec, "build_pointcloud", return_value=None), \
         patch.object(rec, "build_localization_db", side_effect=lambda **k: called.append("localize")):
        rec.run_pipeline()
    assert called == []


def test_localize_without_pointcloud_raises(tmp_path):
    config = _make_config(tmp_path)
    rec = Reconstructor(config)
    with pytest.raises(ValueError, match="requires 'pointcloud'"):
        rec.run_pipeline(stages=["localize"])


def test_build_localization_db_missing_zarr_raises(tmp_path):
    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
    rec = Reconstructor(config)
    with pytest.raises(FileNotFoundError, match="feedforward.zarr"):
        rec.build_localization_db()


def test_build_localization_db_skips_when_exists(tmp_path):
    from collab_splats.wrapper import reconstructor as R
    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma"}})
    rec = Reconstructor(config)
    ff = rec.backend_dir / "feedforward.zarr"
    ff.mkdir(parents=True)
    with patch.object(R, "_localization_db_exists", return_value=True), \
         patch.object(R, "_build_localization_db") as build:
        out = rec.build_localization_db(overwrite=False)
    build.assert_not_called()
    assert out == ff


def test_build_localization_db_runs_when_missing(tmp_path):
    from collab_splats.wrapper import reconstructor as R
    config = _make_config(tmp_path, {"localization": {"enabled": True, "extractor": "loma", "radius": 8.0}})
    rec = Reconstructor(config)
    ff = rec.backend_dir / "feedforward.zarr"
    ff.mkdir(parents=True)
    with patch.object(R, "_localization_db_exists", return_value=False), \
         patch.object(R, "_build_localization_db") as build:
        rec.build_localization_db(overwrite=False)
    build.assert_called_once_with(ff, "loma", 8.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -k "localize or localization_db" -v`
Expected: FAIL (`_STAGE_DEPS` has no `localize`; `build_localization_db` / `_localization_db_exists` / `_build_localization_db` not defined).

- [ ] **Step 3: Update stage constants**

In `reconstructor.py` replace the constants block (~L27-33):
```python
_STAGE_ORDER = ["preprocess", "pointcloud", "semantics", "mesh", "localize"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preprocess": [],
    "pointcloud": ["preprocess"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
}
```

- [ ] **Step 4: Add module-level helpers**

Insert after `_run_tsdf_mesh` (before the `# Reconstructor` divider, ~L263):
```python
def _localization_db_exists(feedforward_zarr: Path, extractor_name: str) -> bool:
    """True if the local-feature DB group already exists in feedforward.zarr."""
    import zarr as zarr_lib
    try:
        store = zarr_lib.open_group(str(feedforward_zarr), mode="r")
        return (
            "local_features" in store
            and extractor_name in store["local_features"]
            and "reconstruction" in store["local_features"][extractor_name]
        )
    except Exception:
        return False


def _build_localization_db(feedforward_zarr: Path, extractor_name: str, radius: float) -> Path:
    """Build the per-frame local-feature localization cache into feedforward.zarr.

    Loads the FeedforwardResult, runs the local matcher over every DB frame, and persists
    keypoints/descriptors to group local_features/{extractor_name}/reconstruction.
    """
    # Heavy deps kept inline so the module imports without GPU/model libs
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.localization.localizer import CameraLocalizer
    from collab_splats.localization.extractors import BaseLocalExtractor

    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)
    extractor = BaseLocalExtractor.get(extractor_name)()

    # from_feedforward is cache-first: on miss it runs GPU extraction + save_index
    CameraLocalizer.from_feedforward(
        ff,
        extractor=extractor,
        extractor_name=extractor_name,
        zarr_path=feedforward_zarr,
        radius=radius,
    )
    logger.info("Localization DB built: %s :: local_features/%s", feedforward_zarr, extractor_name)
    return feedforward_zarr
```

- [ ] **Step 5: Replace the dead `localize` stub with the stage method**

Replace (~L642-644):
```python
    def localize(self, image: np.ndarray) -> np.ndarray:
        """Localize query image into existing reconstruction. Returns (4, 4) c2w pose."""
        raise NotImplementedError("localization not yet implemented")
```
with:
```python
    def build_localization_db(self, result: "PointcloudResult | None" = None, overwrite: bool = False) -> Path:
        """Build/refresh the per-frame local-feature localization cache in feedforward.zarr."""
        loc_cfg = self.config.get("localization", {})
        extractor_name = loc_cfg.get("extractor", "loma")
        radius = loc_cfg.get("radius", 8.0)

        feedforward_zarr = self.backend_dir / "feedforward.zarr"
        if not feedforward_zarr.exists():
            raise FileNotFoundError(
                f"feedforward.zarr not found at {feedforward_zarr}. "
                "Localization DB requires a feedforward pointcloud stage first."
            )

        # Skip if the DB group already exists and overwrite not requested
        if not overwrite and _localization_db_exists(feedforward_zarr, extractor_name):
            logger.info(
                "Localization DB exists at %s :: local_features/%s, skipping",
                feedforward_zarr, extractor_name,
            )
            return feedforward_zarr

        return _build_localization_db(feedforward_zarr, extractor_name, radius)
```

- [ ] **Step 6: Wire run_pipeline**

In `run_pipeline`, extend the auto-stages block (~L661-667):
```python
        if stages is None:
            # Build from config enabled flags; preprocess + pointcloud always included
            stages = ["preprocess", "pointcloud"]
            if self.config.get("semantics", {}).get("enabled", False):
                stages.append("semantics")
            if self.config.get("mesh", {}).get("enabled", False):
                stages.append("mesh")
            if self.config.get("localization", {}).get("enabled", False):
                stages.append("localize")
```
and add a dispatch branch in the execution loop (after the `mesh` branch, ~L689-690):
```python
            elif stage == "mesh":
                self.mesh(result=result, overwrite=overwrite)
            elif stage == "localize":
                self.build_localization_db(result=result, overwrite=overwrite)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -k "localize or localization_db" -v`
Expected: PASS (7 tests).

- [ ] **Step 8: Run the full reconstructor suite (no regressions)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -v`
Expected: PASS (all).

- [ ] **Step 9: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit -m "feat(pipeline): add localize stage — build local-feature localization DB in feedforward.zarr"
```

---

## Task 3: Multi-video scene launcher

**Files:**
- Create: `docs/examples/run_scenes.py`
- Create: `tests/examples/__init__.py`
- Create: `tests/examples/test_run_scenes.py`

- [ ] **Step 1: Write failing tests**

Create `tests/examples/__init__.py` (empty file).

Create `tests/examples/test_run_scenes.py`:
```python
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

# Load run_scenes.py by path (docs/examples is not a package)
_MODULE_PATH = Path(__file__).parent.parent.parent / "docs" / "examples" / "run_scenes.py"
_spec = importlib.util.spec_from_file_location("run_scenes", _MODULE_PATH)
run_scenes = importlib.util.module_from_spec(_spec)
sys.modules["run_scenes"] = run_scenes
_spec.loader.exec_module(run_scenes)


def _write_configs(tmp_path):
    """Create a minimal configs/ dir with base.yaml."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    base = {
        "input_path": None, "output_path": None,
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega"},
        "semantics": {"enabled": False},
        "localization": {"enabled": False, "extractor": "loma"},
    }
    (cfg_dir / "base.yaml").write_text(yaml.dump(base))
    return cfg_dir


def test_build_scene_config_maps_paths(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    video = Path("/data/birds/C0043.MP4")
    config = run_scenes.build_scene_config(video, "/out", cfg_dir)
    assert config["input_path"] == "/data/birds/C0043.MP4"
    assert config["output_path"] == "/out/C0043"
    assert config["pointcloud"]["backend"] == "vggt_omega"  # base preserved


def test_build_scene_config_merges_override(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    override = {"localization": {"enabled": True}}
    config = run_scenes.build_scene_config(Path("/data/x.mp4"), "/out", cfg_dir, override)
    assert config["localization"]["enabled"] is True
    assert config["localization"]["extractor"] == "loma"  # base preserved


def test_run_scenes_runs_pipeline_per_video(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    v1.touch(); v2.touch()
    fake = MagicMock()
    fake.config = {"output_path": str(tmp_path / "out" / "a")}
    with patch.object(run_scenes, "Reconstructor", return_value=fake) as R:
        code = run_scenes.run_all(
            [v1, v2], output_root=tmp_path / "out", config_dir=cfg_dir,
            override_config=None, stages=None, overwrite=False,
        )
    assert R.call_count == 2
    assert fake.run_pipeline.call_count == 2
    assert code == 0


def test_run_scenes_continues_on_failure(tmp_path):
    cfg_dir = _write_configs(tmp_path)
    v1, v2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    v1.touch(); v2.touch()
    fake = MagicMock()
    fake.config = {"output_path": str(tmp_path / "out" / "x")}
    fake.run_pipeline.side_effect = [RuntimeError("boom"), None]
    with patch.object(run_scenes, "Reconstructor", return_value=fake):
        code = run_scenes.run_all(
            [v1, v2], output_root=tmp_path / "out", config_dir=cfg_dir,
            override_config=None, stages=None, overwrite=False,
        )
    assert fake.run_pipeline.call_count == 2  # did not abort after first failure
    assert code == 1                          # non-zero because one failed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/examples/test_run_scenes.py -v`
Expected: FAIL (`run_scenes.py` does not exist / functions undefined).

- [ ] **Step 3: Write run_scenes.py**

Create `docs/examples/run_scenes.py`:
```python
#!/usr/bin/env python3
"""Run the minimal reconstruction pipeline over one or more scene videos.

Usage:
    # One scene
    python docs/examples/run_scenes.py --output-root /workspace/outputs scene.MP4

    # Many scenes (shell glob), semantics + localization on via a shared override YAML
    python docs/examples/run_scenes.py \\
        --output-root /workspace/outputs \\
        --config configs/minimal.yaml \\
        /data/birds/*.MP4

    # Only some stages
    python docs/examples/run_scenes.py --output-root /workspace/outputs \\
        --stages preprocess,pointcloud,localize scene.MP4

Each video V maps to output_path = <output-root>/<V-stem>/. Config is base.yaml
(optionally deep-merged with --config). One scene's failure does not abort the batch;
the process exits non-zero if any scene failed.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.wrapper.config import ConfigLoader
from collab_splats.wrapper.reconstructor import Reconstructor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs"


def build_scene_config(video, output_root, config_dir, override_config=None):
    """Build a per-scene config from base.yaml with input/output paths set."""
    loader = ConfigLoader(config_dir)
    config = merge({}, loader.base_config)          # copy of base.yaml
    if override_config:
        config = merge({}, config, override_config)  # shared --config overrides
    config["input_path"] = str(video)
    config["output_path"] = str(Path(output_root) / Path(video).stem)
    return config


def run_scene(video, output_root, config_dir, override_config, stages, overwrite):
    """Run the pipeline for a single video. Returns the scene output path."""
    config = build_scene_config(video, output_root, config_dir, override_config)
    r = Reconstructor(config)

    # Persist run_config.yaml for reproducibility (same behavior as reconstruct.py)
    output_path = Path(r.config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    run_cfg = output_path / "run_config.yaml"
    if not run_cfg.exists() or overwrite:
        with open(run_cfg, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)

    r.run_pipeline(stages=stages, overwrite=overwrite)
    return output_path


def run_all(videos, output_root, config_dir, override_config, stages, overwrite):
    """Run every scene; continue past failures. Returns process exit code."""
    results = []
    for video in videos:
        video = Path(video)
        logger.info("=== Scene: %s ===", video.name)
        try:
            out = run_scene(video, output_root, config_dir, override_config, stages, overwrite)
            results.append((video.name, "OK", str(out)))
        except Exception as exc:  # isolate one scene's failure from the batch
            logger.exception("Scene failed: %s", video.name)
            results.append((video.name, "FAIL", str(exc)))

    # Summary
    logger.info("==== Summary ====")
    for name, status, info in results:
        logger.info("%s: %s (%s)", status, name, info)
    failed = [n for n, s, _ in results if s == "FAIL"]
    return 1 if failed else 0


def main():
    """Parse args and run the pipeline over the given scene videos."""
    parser = argparse.ArgumentParser(
        description="Run the minimal reconstruction pipeline over one or more scene videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("videos", nargs="+", metavar="VIDEO", help="One or more video files.")
    parser.add_argument("--output-root", required=True, type=Path, dest="output_root",
                        help="Parent dir; each scene lands in <output-root>/<video-stem>/.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional shared override YAML merged over base.yaml.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, dest="config_dir",
                        help=f"Directory holding base.yaml. Default: {DEFAULT_CONFIG_DIR}")
    parser.add_argument("--stages", default=None, metavar="STAGE[,STAGE,...]",
                        help="Stages to run: preprocess,pointcloud,semantics,mesh,localize. "
                             "Default: config-enabled stages.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run stages even if outputs already exist.")
    args = parser.parse_args()

    override_config = None
    if args.config:
        with open(args.config) as f:
            override_config = yaml.safe_load(f)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    code = run_all(
        videos=args.videos,
        output_root=args.output_root,
        config_dir=args.config_dir,
        override_config=override_config,
        stages=stages,
        overwrite=args.overwrite,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/examples/test_run_scenes.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add docs/examples/run_scenes.py tests/examples/__init__.py tests/examples/test_run_scenes.py
git commit -m "feat(examples): run_scenes.py — run minimal pipeline over one or more scene videos"
```

---

## Task 4: Refresh reconstruct.py docs/help

**Files:**
- Modify: `docs/examples/reconstruct.py` (docstring ~L4-27; `--stages` help ~L78-81)

- [ ] **Step 1: Add localize to the --stages help**

Replace the `--stages` argument help (~L78-81):
```python
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help="Comma-separated stages to run: preprocess,pointcloud,semantics,mesh,localize. "
        "Default: all enabled stages in config.",
    )
```

- [ ] **Step 2: Add a localization usage example to the docstring**

Insert into the module docstring after the "Run specific stages only" example (~L9-10):
```python
    # Build the localization database (local-feature cache) as part of the run
    python docs/examples/reconstruct.py --dataset birds_c0043 \\
        localization.enabled=true --stages preprocess,pointcloud,localize
```

- [ ] **Step 3: Verify the CLI help renders**

Run: `/opt/venv/reconstruction/bin/python docs/examples/reconstruct.py --help`
Expected: help text shows `preprocess,pointcloud,semantics,mesh,localize` and the new example. No traceback.

- [ ] **Step 4: Commit**

```bash
git add docs/examples/reconstruct.py
git commit -m "docs(examples): reconstruct.py — document localize stage"
```

---

## Task 5: Verify run_all_datasets.sh + reconcile interpreter

**Files:**
- Modify: `docs/examples/run_all_datasets.sh:6`

- [ ] **Step 1: Determine the canonical interpreter**

Run: `ls -d /opt/venv/reconstruction/bin/python /opt/conda/envs/reconstruction/bin/python 2>&1`
Pick the path that exists. CLAUDE.md cites `/opt/venv/reconstruction/bin/python`; the script currently uses `/opt/conda/envs/reconstruction/bin/python`. If only one exists, use it. **If both or neither is ambiguous, stop and ask the user which to standardize on** (this is the flagged open item).

- [ ] **Step 2: Update the PYTHON line if needed**

If the canonical path differs from line 6, edit:
```bash
PYTHON=/opt/venv/reconstruction/bin/python
```
(Otherwise leave unchanged and note it in the commit.)

- [ ] **Step 3: Verify the script is syntactically valid**

Run: `bash -n docs/examples/run_all_datasets.sh`
Expected: no output (valid). The script needs no localize change — it passes no `--stages`, so localization runs only when a dataset config sets `localization.enabled: true`.

- [ ] **Step 4: Commit**

```bash
git add docs/examples/run_all_datasets.sh
git commit -m "chore(examples): reconcile run_all_datasets.sh interpreter path"
```

---

## Task 6: Full suite + formatting

- [ ] **Step 1: Format**

Run: `cd /workspace/collab-splats && black docs/examples/run_scenes.py collab_splats/wrapper/reconstructor.py tests/examples/ tests/wrapper/test_reconstructor.py && isort docs/examples/run_scenes.py collab_splats/wrapper/reconstructor.py tests/examples/ tests/wrapper/test_reconstructor.py`

- [ ] **Step 2: Run the affected test suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/examples/ -v`
Expected: PASS (all).

- [ ] **Step 3: Commit any formatting changes**

```bash
git add -A
git commit -m "style: black + isort on minimal pipeline runner" || echo "nothing to format"
```

---

## Self-Review Notes

- **Spec coverage:** launcher (Task 3), localize stage + wiring (Task 2), base.yaml fix incl. loma default (Task 1), reconstruct.py refresh (Task 4), run_all_datasets.sh verify + interpreter reconcile (Task 5), tests (Tasks 2/3). All spec sections mapped.
- **Type consistency:** `build_localization_db`, `_build_localization_db(ff, name, radius)`, `_localization_db_exists(ff, name)`, `run_all`, `run_scene`, `build_scene_config` used identically across tasks and tests.
- **Open item:** interpreter reconcile is gated on a live-filesystem check in Task 5 Step 1, with an explicit stop-and-ask fallback.
