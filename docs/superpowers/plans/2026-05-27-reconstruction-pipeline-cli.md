# Reconstruction Pipeline CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `docs/examples/reconstruct.py` CLI + `configs/reconstruction/` hierarchy so any dataset can be run end-to-end (pointcloud → semantics) with a single command, and run all 10 fieldwork datasets with `vggt_omega`.

**Architecture:** Thin CLI wrapper around existing `Reconstructor.from_config_file().run_pipeline()`. Config hierarchy: `configs/reconstruction/base.yaml` (defaults) + `datasets/<name>.yaml` (per-dataset overrides). Every run writes `run_config.yaml` to the output dir for auditability. Final task launches all datasets sequentially in tmux and monitors.

**Tech Stack:** Python 3.11, PyYAML, mergedeep, `collab_splats.wrapper.{reconstructor,config}`, pytest, tmux

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `.gitignore` | Add `data/` and `outputs/` |
| Create | `configs/reconstruction/base.yaml` | All Reconstructor defaults (omega + talk2dino) |
| Create | `configs/reconstruction/README.md` | User-facing docs: why, how to run, how to add datasets |
| Create | `configs/reconstruction/datasets/birds_c0043.yaml` | input_path + frame_proportion override |
| Create | `configs/reconstruction/datasets/birds_c0065.yaml` | " |
| Create | `configs/reconstruction/datasets/birds_c0067.yaml` | " |
| Create | `configs/reconstruction/datasets/birds_gh010070.yaml` | " |
| Create | `configs/reconstruction/datasets/birds_gh010097.yaml` | " |
| Create | `configs/reconstruction/datasets/birds_gh010105.yaml` | " |
| Create | `configs/reconstruction/datasets/birds_gh010164.yaml` | " |
| Create | `configs/reconstruction/datasets/birds_pxl_20231105.yaml` | " |
| Create | `configs/reconstruction/datasets/ants_gh010210.yaml` | " |
| Create | `configs/reconstruction/datasets/rats_c0119.yaml` | " |
| Create | `tests/scripts/__init__.py` | Test package marker |
| Create | `tests/scripts/test_reconstruct.py` | Unit tests for the CLI |
| Create | `docs/examples/reconstruct.py` | CLI entry point |

---

### Task 1: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add entries**

Append to `.gitignore`:

```
# Eval benchmark data (downloaded on-demand via evals/download_7scenes.sh)
data/

# Reconstruction outputs (written to /workspace/outputs/ outside repo)
outputs/
```

- [ ] **Step 2: Verify git no longer tracks data/**

```bash
git check-ignore -v data/7scenes/chess/seq-01
```

Expected output contains `data/`.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore data/ and outputs/ directories"
```

---

### Task 2: Create `configs/reconstruction/base.yaml`

**Files:**
- Create: `configs/reconstruction/base.yaml`

- [ ] **Step 1: Create directory and file**

```bash
mkdir -p configs/reconstruction/datasets
```

Write `configs/reconstruction/base.yaml`:

```yaml
# Base configuration for Reconstructor pipelines.
# All dataset configs inherit from this and override specific values.
#
# Required per dataset (no defaults — Reconstructor.validate_config() raises if null):
#   input_path  — absolute path to input video or image directory
#   output_path — absolute path to output root (created if absent)
#
# See configs/reconstruction/README.md for full documentation.

input_path: null
output_path: null

preprocessing:
  frame_selection: fps        # fps | optical_flow
  frame_proportion: 0.1       # fraction of total frames to extract
  min_frames: 300             # extract at least this many frames
  max_frames: null            # null = no upper cap

pointcloud:
  method: feedforward         # feedforward | sfm
  backend: vggt_omega         # vggt_omega | vggtx | mapanything  (feedforward only)
  bundle_adjustment: false
  loop_closure: false
  clean:
    enabled: true
    outlier_removal: true
    voxel_size: null          # null = adaptive per-scene
    confidence_threshold: null

semantics:
  enabled: true
  extractor: talk2dino        # talk2dino | dinov2 | maskclip
  n_components: 64            # PCA compression dim; null = no compression

mesh:
  enabled: false
  mesher: tsdf                # tsdf | poisson
  voxel_size: 0.01
  sdf_trunc: 0.04

localization:
  enabled: false
  extractor: dinosalad
```

- [ ] **Step 2: Sanity-load with ConfigLoader**

```bash
/opt/conda/envs/reconstruction/bin/python - << 'EOF'
from collab_splats.wrapper.config import ConfigLoader
# base.yaml alone should load without error
import yaml
with open("configs/reconstruction/base.yaml") as f:
    cfg = yaml.safe_load(f)
assert cfg["pointcloud"]["backend"] == "vggt_omega"
assert cfg["semantics"]["extractor"] == "talk2dino"
assert cfg["semantics"]["enabled"] is True
assert cfg["input_path"] is None
print("base.yaml OK:", list(cfg.keys()))
EOF
```

Expected:
```
base.yaml OK: ['input_path', 'output_path', 'preprocessing', 'pointcloud', 'semantics', 'mesh', 'localization']
```

- [ ] **Step 3: Commit**

```bash
git add configs/reconstruction/base.yaml
git commit -m "feat(configs): add configs/reconstruction/base.yaml with omega+talk2dino defaults"
```

---

### Task 3: Create `configs/reconstruction/README.md`

**Files:**
- Create: `configs/reconstruction/README.md`

- [ ] **Step 1: Write README**

```markdown
# Reconstruction Pipeline Configs

Configuration templates for `docs/examples/reconstruct.py`. Each run of the pipeline
reads these files to know what data to process, which backend to use, and where
to write outputs.

---

## Why configs live here (not with the data)

This project separates **code + configs** (versioned in git) from **data + outputs**
(too large for git, live outside the repo):

```
/workspace/
  collab-splats/               ← this repo (configs live here)
    configs/reconstruction/
    docs/examples/reconstruct.py
    data/                      ← gitignored — eval benchmarks, downloaded on-demand

  fieldwork-data/              ← input videos (never in repo, ~GB each)
  outputs/                     ← reconstruction outputs (never in repo, ~10-50 GB per scene)
    birds_c0043_omega/
      run_config.yaml          ← exact settings that produced this output
      vggt_omega/
        feedforward.zarr
        semantics/talk2dino/features.zarr
```

**Why not relative paths?** This project runs in a fixed container environment
(`/workspace/` mount). Absolute `/workspace/...` paths are stable. Portability
across environments is handled by the container image + volume mounts.

---

## Running a dataset

```bash
# Standard run (uses base defaults: vggt_omega + talk2dino, semantics on)
python docs/examples/reconstruct.py --dataset birds_c0043

# Specific stages only
python docs/examples/reconstruct.py --dataset birds_c0043 --stages preprocess,pointcloud

# Override any config key at CLI (dotted key=value, any depth)
python docs/examples/reconstruct.py --dataset birds_c0043 \
  semantics.extractor=dinov2 \
  pointcloud.bundle_adjustment=true

# Experiment variant — send output to a separate dir
python docs/examples/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_ba_experiment

# Re-run from a saved config for exact reproducibility
python docs/examples/reconstruct.py \
  --config /workspace/outputs/birds_c0043_omega/run_config.yaml

# Force re-run even if outputs exist
python docs/examples/reconstruct.py --dataset birds_c0043 --overwrite
```

---

## Adding a new dataset

1. Copy `datasets/birds_c0043.yaml` to `datasets/<your_name>.yaml`
2. Set `input_path` to the absolute path of the video or image directory
3. Set `output_path` to where outputs should be written (e.g. `/workspace/outputs/<your_name>`)
4. Tune `preprocessing.frame_proportion` if needed (higher = more frames = slower but better)
5. Run: `python docs/examples/reconstruct.py --dataset <your_name>`

Everything else is inherited from `base.yaml`. You only need to override what differs.

---

## Running experiments (multiple configs, same dataset)

Don't create a new dataset YAML for each experiment. Instead, override `output_path` at CLI:

```bash
# Experiment A: omega + talk2dino (base defaults)
python docs/examples/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_omega_t2d

# Experiment B: vggtx + dinov2
python docs/examples/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_vggtx_dino \
  pointcloud.backend=vggtx \
  semantics.extractor=dinov2
```

Each output dir gets its own `run_config.yaml` recording the exact settings used.

---

## Config key reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_path` | str | **required** | Absolute path to video (.MP4) or image directory |
| `output_path` | str | **required** | Absolute path for outputs (created if absent) |
| `preprocessing.frame_selection` | str | `fps` | Frame sampling method: `fps` or `optical_flow` |
| `preprocessing.frame_proportion` | float | `0.1` | Fraction of total frames to extract |
| `preprocessing.min_frames` | int | `300` | Minimum frames to extract regardless of proportion |
| `preprocessing.max_frames` | int\|null | `null` | Cap on frames extracted; null = no cap |
| `pointcloud.method` | str | `feedforward` | `feedforward` or `sfm` |
| `pointcloud.backend` | str | `vggt_omega` | `vggt_omega`, `vggtx`, or `mapanything` |
| `pointcloud.bundle_adjustment` | bool | `false` | Run LM bundle adjustment after pointcloud |
| `pointcloud.loop_closure` | bool | `false` | Run loop closure after pointcloud |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points |
| `semantics.enabled` | bool | `true` | Extract and lift semantic features |
| `semantics.extractor` | str | `talk2dino` | Feature extractor: `talk2dino`, `dinov2`, `maskclip` |
| `semantics.n_components` | int\|null | `64` | PCA compression dim; null = no compression |
| `mesh.enabled` | bool | `false` | Build TSDF/Poisson mesh |
| `mesh.mesher` | str | `tsdf` | `tsdf` or `poisson` |
| `mesh.voxel_size` | float | `0.01` | TSDF voxel size in metres |
| `mesh.sdf_trunc` | float | `0.04` | TSDF truncation distance in metres |

---

## Where outputs land

```
<output_path>/
  run_config.yaml              ← full merged config (exact settings used)
  images/                      ← extracted frames
  features/                    ← 2D feature cache (extractor name subdir)
  <backend>/                   ← e.g. vggt_omega/
    feedforward.zarr           ← depth maps, poses, points
    semantics/
      <extractor>/
        features.zarr          ← lifted 3D features (N_points × D)
        compressor.pt          ← PCA compressor (if n_components set)
    mesh/
      mesh.ply                 ← (if mesh.enabled=true)

```

---

## Dashboard

Point the dashboard at `output_path` to visualise results. The dashboard reads
`<backend>/feedforward.zarr` for the pointcloud and
`<backend>/semantics/<extractor>/features.zarr` for semantic features.
```

- [ ] **Step 2: Commit**

```bash
git add configs/reconstruction/README.md
git commit -m "docs(configs): add configs/reconstruction/README.md with workspace layout rationale"
```

---

### Task 4: Create 10 dataset configs

**Files:**
- Create: `configs/reconstruction/datasets/*.yaml` (10 files)

- [ ] **Step 1: Write all 10 dataset YAMLs**

`configs/reconstruction/datasets/birds_c0043.yaml`:
```yaml
# Birds - 2024-02-06 - C0043
input_path: /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4
output_path: /workspace/outputs/birds_c0043

preprocessing:
  frame_proportion: 0.25
```

`configs/reconstruction/datasets/birds_c0065.yaml`:
```yaml
# Birds - 2024-05-18 - C0065
input_path: /workspace/fieldwork-data/birds/2024-05-18/SplatsSD/C0065.MP4
output_path: /workspace/outputs/birds_c0065

preprocessing:
  frame_proportion: 0.25
```

`configs/reconstruction/datasets/birds_c0067.yaml`:
```yaml
# Birds - 2024-05-19 - C0067
input_path: /workspace/fieldwork-data/birds/2024-05-19/SplatsSD/C0067.MP4
output_path: /workspace/outputs/birds_c0067

preprocessing:
  frame_proportion: 0.25
```

`configs/reconstruction/datasets/birds_gh010070.yaml`:
```yaml
# Birds - 2024-05-23 - GH010070
input_path: /workspace/fieldwork-data/birds/2024-05-23/SplatsSD/GH010070.MP4
output_path: /workspace/outputs/birds_gh010070

preprocessing:
  frame_proportion: 0.125
```

`configs/reconstruction/datasets/birds_gh010097.yaml`:
```yaml
# Birds - 2024-05-27 - GH010097
input_path: /workspace/fieldwork-data/birds/2024-05-27/SplatsSD/GH010097.MP4
output_path: /workspace/outputs/birds_gh010097

preprocessing:
  frame_proportion: 0.14
```

`configs/reconstruction/datasets/birds_gh010105.yaml`:
```yaml
# Birds - 2024-05-27 - GH010105
input_path: /workspace/fieldwork-data/birds/2024-05-27/SplatsSD/GH010105.MP4
output_path: /workspace/outputs/birds_gh010105

preprocessing:
  frame_proportion: 0.25
```

`configs/reconstruction/datasets/birds_gh010164.yaml`:
```yaml
# Birds - 2024-06-01 - GH010164
input_path: /workspace/fieldwork-data/birds/2024-06-01/SplatsSD/GH010164.MP4
output_path: /workspace/outputs/birds_gh010164

preprocessing:
  frame_proportion: 0.10
```

`configs/reconstruction/datasets/birds_pxl_20231105.yaml`:
```yaml
# Birds - 2023-11-05 - PXL_20231105_154956078
input_path: /workspace/fieldwork-data/birds/2023-11-05/SplatsSD/PXL_20231105_154956078.mp4
output_path: /workspace/outputs/birds_pxl_20231105

preprocessing:
  frame_proportion: 0.25
```

`configs/reconstruction/datasets/ants_gh010210.yaml`:
```yaml
# Ants - 2025-11-16 - GH010210
input_path: /workspace/fieldwork-data/ants/2025-11-16/SplatsSD/GH010210.MP4
output_path: /workspace/outputs/ants_gh010210

preprocessing:
  frame_proportion: 0.08
```

`configs/reconstruction/datasets/rats_c0119.yaml`:
```yaml
# Rats - 2024-07-11 - C0119
input_path: /workspace/fieldwork-data/rats/2024-07-11/SplatsSD/C0119.MP4
output_path: /workspace/outputs/rats_c0119

preprocessing:
  frame_proportion: 0.25
```

- [ ] **Step 2: Verify ConfigLoader loads each config**

```bash
/opt/conda/envs/reconstruction/bin/python - << 'EOF'
from collab_splats.wrapper.config import ConfigLoader

loader = ConfigLoader("configs/reconstruction")
datasets = loader.list_datasets()
print(f"Found {len(datasets)} datasets:", sorted(datasets))

# Check one fully merged config
cfg = loader.load("birds_c0043")
assert cfg["pointcloud"]["backend"] == "vggt_omega", cfg["pointcloud"]["backend"]
assert cfg["semantics"]["extractor"] == "talk2dino"
assert cfg["semantics"]["enabled"] is True
assert cfg["input_path"] == "/workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4"
assert cfg["output_path"] == "/workspace/outputs/birds_c0043"
assert cfg["preprocessing"]["frame_proportion"] == 0.25

# Check ants has low frame_proportion
cfg_ants = loader.load("ants_gh010210")
assert cfg_ants["preprocessing"]["frame_proportion"] == 0.08

print("All dataset configs OK")
EOF
```

Expected:
```
Found 10 datasets: ['ants_gh010210', 'birds_c0043', 'birds_c0065', 'birds_c0067', 'birds_gh010070', 'birds_gh010097', 'birds_gh010105', 'birds_gh010164', 'birds_pxl_20231105', 'rats_c0119']
All dataset configs OK
```

- [ ] **Step 3: Commit**

```bash
git add configs/reconstruction/datasets/
git commit -m "feat(configs): add 10 dataset configs for reconstruction pipeline"
```

---

### Task 5: Write failing tests for `docs/examples/reconstruct.py`

**Files:**
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/test_reconstruct.py`

- [ ] **Step 1: Create test package**

```bash
mkdir -p tests/scripts
touch tests/scripts/__init__.py
```

- [ ] **Step 2: Write test file**

Write `tests/scripts/test_reconstruct.py`:

```python
"""Tests for docs/examples/reconstruct.py CLI."""

import importlib.util
import sys
import yaml
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "reconstruct.py"
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "reconstruction"


def _load_main():
    """Load the reconstruct module and return its main() function."""
    spec = importlib.util.spec_from_file_location("reconstruct", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def _make_mock_reconstructor(tmp_path):
    """Return (mock_cls, mock_instance) with config.output_path set to tmp_path."""
    mock_r = MagicMock()
    mock_r.config = {"output_path": str(tmp_path), "pointcloud": {"backend": "vggt_omega"}}
    mock_r.backend_dir = tmp_path / "vggt_omega"
    mock_cls = MagicMock()
    mock_cls.from_config_file.return_value = mock_r
    mock_cls.return_value = mock_r  # direct Reconstructor(config) path
    return mock_cls, mock_r


def test_dataset_arg_calls_from_config_file(monkeypatch, tmp_path):
    """--dataset NAME calls Reconstructor.from_config_file with correct args."""
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--dataset", "birds_c0043"])
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_cls.from_config_file.assert_called_once_with(
        dataset="birds_c0043",
        config_dir=DEFAULT_CONFIG_DIR,
        overrides=None,
    )


def test_stages_parsed_to_list(monkeypatch, tmp_path):
    """--stages preprocess,pointcloud passes list to run_pipeline."""
    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--dataset", "birds_c0043", "--stages", "preprocess,pointcloud"],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_r.run_pipeline.assert_called_once_with(
        stages=["preprocess", "pointcloud"], overwrite=False
    )


def test_overwrite_flag(monkeypatch, tmp_path):
    """--overwrite passes overwrite=True to run_pipeline."""
    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--dataset", "birds_c0043", "--overwrite"],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_r.run_pipeline.assert_called_once_with(stages=None, overwrite=True)


def test_key_value_overrides_parsed(monkeypatch, tmp_path):
    """KEY=VALUE positional args are parsed and passed as overrides dict."""
    monkeypatch.setattr(
        sys, "argv",
        [
            "reconstruct.py", "--dataset", "birds_c0043",
            "pointcloud.backend=vggtx",
            "semantics.enabled=true",
        ],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    mock_cls.from_config_file.assert_called_once_with(
        dataset="birds_c0043",
        config_dir=DEFAULT_CONFIG_DIR,
        overrides={"pointcloud": {"backend": "vggtx"}, "semantics": {"enabled": True}},
    )


def test_direct_config_path_loads_yaml(monkeypatch, tmp_path):
    """--config /path/to/config.yaml loads YAML directly, skips dataset hierarchy."""
    config_yaml = tmp_path / "myconfig.yaml"
    cfg_data = {
        "input_path": "/data/video.mp4",
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggtx"},
        "semantics": {"enabled": True, "extractor": "dinov2", "n_components": 64},
    }
    config_yaml.write_text(yaml.dump(cfg_data))

    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--config", str(config_yaml)],
    )
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path / "out")
    mock_r.config = {**cfg_data, "output_path": str(tmp_path / "out")}
    mock_cls.return_value = mock_r
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    # Should call Reconstructor(config) directly, not from_config_file
    mock_cls.assert_called_once()
    mock_cls.from_config_file.assert_not_called()


def test_run_config_yaml_written_to_output_dir(monkeypatch, tmp_path):
    """main() writes run_config.yaml into output_path before running pipeline."""
    monkeypatch.setattr(sys, "argv", ["reconstruct.py", "--dataset", "birds_c0043"])
    mock_cls, mock_r = _make_mock_reconstructor(tmp_path)
    mock_r.config = {
        "input_path": "/data/video.mp4",
        "output_path": str(tmp_path),
        "pointcloud": {"backend": "vggt_omega"},
    }
    main = _load_main()

    with patch("collab_splats.wrapper.reconstructor.Reconstructor", mock_cls):
        main()

    run_cfg = tmp_path / "run_config.yaml"
    assert run_cfg.exists(), "run_config.yaml was not written"
    loaded = yaml.safe_load(run_cfg.read_text())
    assert loaded["pointcloud"]["backend"] == "vggt_omega"


def test_dataset_and_config_are_mutually_exclusive(monkeypatch, tmp_path, capsys):
    """--dataset and --config together cause argparse error (SystemExit)."""
    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text("{}")
    monkeypatch.setattr(
        sys, "argv",
        ["reconstruct.py", "--dataset", "birds_c0043", "--config", str(config_yaml)],
    )
    main = _load_main()
    with pytest.raises(SystemExit):
        main()
```

- [ ] **Step 3: Run tests — verify they all FAIL with ImportError or NameError (script doesn't exist yet)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/scripts/test_reconstruct.py -v 2>&1 | head -40
```

Expected: All 7 tests FAIL (script not found / import error).

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/scripts/__init__.py tests/scripts/test_reconstruct.py
git commit -m "test(scripts): add failing tests for reconstruct.py CLI (TDD)"
```

---

### Task 6: Implement `docs/examples/reconstruct.py`

**Files:**
- Create: `docs/examples/reconstruct.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""CLI entry point for the Reconstructor pipeline.

Usage:
    # Run with a named dataset (config from configs/reconstruction/datasets/)
    python docs/examples/reconstruct.py --dataset birds_c0043

    # Run specific stages only
    python docs/examples/reconstruct.py --dataset birds_c0043 --stages preprocess,pointcloud

    # Override any config value (dotted key=value)
    python docs/examples/reconstruct.py --dataset birds_c0043 \\
        pointcloud.backend=vggtx \\
        semantics.extractor=dinov2

    # Experiment variant: send output to a separate dir
    python docs/examples/reconstruct.py --dataset birds_c0043 \\
        output_path=/workspace/outputs/birds_c0043_ba \\
        pointcloud.bundle_adjustment=true

    # Re-run from a saved run_config.yaml (exact reproducibility)
    python docs/examples/reconstruct.py \\
        --config /workspace/outputs/birds_c0043/run_config.yaml

    # Force re-run even if outputs already exist
    python docs/examples/reconstruct.py --dataset birds_c0043 --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "configs" / "reconstruction"


def main() -> None:
    """Parse args and run Reconstructor pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Reconstructor pipeline from config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config source: exactly one of --dataset or --config
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        metavar="NAME",
        help="Dataset name — loads configs/reconstruction/datasets/<NAME>.yaml",
    )
    source.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="Direct path to a YAML config file (e.g. a saved run_config.yaml)",
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        dest="config_dir",
        help=f"Config directory for --dataset lookups. Default: {DEFAULT_CONFIG_DIR}",
    )
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help="Comma-separated stages to run: preprocess,pointcloud,semantics,mesh. "
             "Default: all enabled stages in config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run stages even if output already exists on disk.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dotted-path config overrides, e.g. pointcloud.backend=vggtx semantics.enabled=true",
    )

    args = parser.parse_args()

    from collab_splats.wrapper.config import parse_cli_overrides
    from collab_splats.wrapper.reconstructor import Reconstructor

    # Parse KEY=VALUE overrides
    overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    # Build Reconstructor from either a dataset template or a direct YAML
    if args.config:
        # Direct YAML path (e.g. a saved run_config.yaml for exact reproducibility)
        logger.info("Loading config from file: %s", args.config)
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if overrides:
            from mergedeep import merge
            config = merge({}, config, overrides)
        r = Reconstructor(config)
    else:
        # Named dataset: merge base.yaml + datasets/<name>.yaml + overrides
        logger.info("Loading config: dataset=%s config_dir=%s", args.dataset, args.config_dir)
        r = Reconstructor.from_config_file(
            dataset=args.dataset,
            config_dir=args.config_dir,
            overrides=overrides,
        )

    # Write run_config.yaml to output dir before running (for auditability)
    output_path = Path(r.config["output_path"])
    run_cfg_path = output_path / "run_config.yaml"
    if not run_cfg_path.exists() or args.overwrite:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(run_cfg_path, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)
        logger.info("Config saved: %s", run_cfg_path)
    else:
        logger.info("Config exists (use --overwrite to replace): %s", run_cfg_path)

    # Parse stages
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    logger.info(
        "Running pipeline: stages=%s overwrite=%s", stages or "auto", args.overwrite
    )
    r.run_pipeline(stages=stages, overwrite=args.overwrite)
    logger.info("Pipeline complete. Output: %s", r.backend_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable**

```bash
chmod +x docs/examples/reconstruct.py
```

---

### Task 7: Run tests and commit

**Files:** none new

- [ ] **Step 1: Run all 7 tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/scripts/test_reconstruct.py -v
```

Expected:
```
PASSED tests/scripts/test_reconstruct.py::test_dataset_arg_calls_from_config_file
PASSED tests/scripts/test_reconstruct.py::test_stages_parsed_to_list
PASSED tests/scripts/test_reconstruct.py::test_overwrite_flag
PASSED tests/scripts/test_reconstruct.py::test_key_value_overrides_parsed
PASSED tests/scripts/test_reconstruct.py::test_direct_config_path_loads_yaml
PASSED tests/scripts/test_reconstruct.py::test_run_config_yaml_written_to_output_dir
PASSED tests/scripts/test_reconstruct.py::test_dataset_and_config_are_mutually_exclusive

7 passed
```

If any test fails, fix `docs/examples/reconstruct.py` (not the tests) and re-run.

- [ ] **Step 2: Run full test suite to check for regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q --ignore=tests/integration 2>&1 | tail -10
```

Expected: no new failures.

- [ ] **Step 3: Verify CLI help works**

```bash
/opt/conda/envs/reconstruction/bin/python docs/examples/reconstruct.py --help
```

Expected: help text prints with `--dataset`, `--config`, `--stages`, `--overwrite`, `KEY=VALUE` documented.

- [ ] **Step 4: Commit**

```bash
git add docs/examples/reconstruct.py
git commit -m "feat(scripts): add reconstruct.py CLI — dataset configs + run_config.yaml auditability"
```

---

### Task 8: Launch all datasets with vggt-omega and monitor

**Goal:** Run all 10 fieldwork datasets sequentially in a tmux session. Each run uses default
settings (vggt_omega + talk2dino). Monitor for completion or errors.

> **Memory constraint:** Container cgroup cap is 46.6 GB. Run one dataset at a time.
> Do NOT run parallel processes during inference (OOM risk).

- [ ] **Step 1: Verify fieldwork-data exists for at least one dataset**

```bash
ls /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4 && echo "OK"
```

If missing, fieldwork-data volume is not mounted — stop and report.

- [ ] **Step 2: Create a batch runner script and launch in tmux**

```bash
# Create the batch script
cat > /tmp/run_all_datasets.sh << 'BATCH'
#!/bin/bash
set -e
PYTHON=/opt/conda/envs/reconstruction/bin/python
SCRIPT=/workspace/collab-splats/docs/examples/reconstruct.py
LOG_DIR=/workspace/outputs/logs
mkdir -p "$LOG_DIR"

DATASETS=(
  birds_c0043
  birds_c0065
  birds_c0067
  birds_gh010070
  birds_gh010097
  birds_gh010105
  birds_gh010164
  birds_pxl_20231105
  ants_gh010210
  rats_c0119
)

for dataset in "${DATASETS[@]}"; do
  echo "=================================================="
  echo "Starting: $dataset  $(date)"
  echo "=================================================="
  $PYTHON "$SCRIPT" --dataset "$dataset" \
    2>&1 | tee "$LOG_DIR/${dataset}.log"
  echo "Finished: $dataset  $(date)"
done

echo "ALL DATASETS COMPLETE"
BATCH
chmod +x /tmp/run_all_datasets.sh

# Launch in a new tmux session
tmux new-session -d -s reconstruction_run \
  "cd /workspace/collab-splats && bash /tmp/run_all_datasets.sh 2>&1 | tee /tmp/reconstruction_run.log"
echo "Launched. Session: reconstruction_run"
echo "Monitor: tmux attach -t reconstruction_run"
```

- [ ] **Step 3: Monitor progress**

Poll `/tmp/reconstruction_run.log` every 5 minutes for completion or errors.

Watch for:
- `Pipeline complete. Output:` — dataset finished successfully
- `Error` / `Traceback` / `OOM` — dataset failed; note which one and continue monitoring
- `ALL DATASETS COMPLETE` — all done

Report status after each dataset completes.
