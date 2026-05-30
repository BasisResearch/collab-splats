# Design: Reconstruction Pipeline CLI

**Date:** 2026-05-27  
**Status:** Draft  
**Scope:** `docs/reconstruct.py` + `configs/reconstruction/` hierarchy + workspace layout convention

---

## Goal

A single CLI entry point that takes a dataset config, runs the full reconstruction pipeline
(preprocess → pointcloud → semantics → optional mesh), and writes self-auditing outputs.
Supports flexible stage selection and per-run config overrides so researchers can run any
combination of backends and extractors without editing files.

---

## Workspace Layout (Option C)

Data and outputs live **outside** the repository. Code, configs, and tests live **inside**.

```
/workspace/
  collab-splats/               ← git repo (code + configs)
    configs/reconstruction/    ← pipeline config templates (versioned)
    docs/reconstruct.py     ← CLI entry point
    data/                      ← eval benchmark data (gitignored, downloaded on-demand)
    evals/                     ← eval harness
    ...

  fieldwork-data/              ← raw input videos (never in repo, too large)
    birds/
    ants/
    rats/

  outputs/                     ← reconstruction outputs (never in repo)
    birds_c0043_omega/
      run_config.yaml          ← full merged config written at run start
      vggt_omega/
        feedforward.zarr
        semantics/talk2dino/features.zarr
        mesh/mesh.ply
```

### Why this split

- **Input videos** are 1-10 GB each. Git cannot handle them; they are container-volume data.
- **Reconstruction outputs** (zarr depth maps, lifted feature stores) grow to 10-50 GB per scene.
  They are too large and too mutable to version-control.
- **Configs** are small, text-only, and capture intent — perfect for git.
- **`run_config.yaml`** in each output dir records the exact merged settings that produced that
  output, enabling full reproducibility without relying on git history.
- This project runs in a fixed container environment (`/workspace/` mount), so absolute
  `/workspace/...` paths in configs are stable and correct. Portability across environments is
  handled by the container image + volume mounts, not by relative paths.

### `data/` inside the repo

`data/` holds eval benchmark data (7-Scenes etc.) downloaded on demand by `evals/download_7scenes.sh`.
It is **gitignored** — large, reproducibly downloadable, not fieldwork data.

---

## Config Hierarchy

```
configs/
  reconstruction/
    README.md                  ← explains layout, why, how to add datasets
    base.yaml                  ← all Reconstructor defaults
    datasets/
      birds_c0043.yaml
      birds_c0065.yaml
      birds_c0067.yaml
      birds_gh010070.yaml
      birds_gh010097.yaml
      birds_gh010105.yaml
      birds_gh010164.yaml
      birds_pxl_20231105.yaml
      ants_gh010210.yaml
      rats_c0119.yaml
```

### `base.yaml` schema

Defaults to `vggt_omega` backend + `talk2dino` semantics. Semantics enabled by default;
mesh and localization off (opt-in).

```yaml
# Base configuration for Reconstructor pipelines.
# All dataset configs inherit from this and override specific values.
#
# Required per dataset (no defaults):
#   input_path  — absolute path to input video or image directory
#   output_path — absolute path to output root (created if absent)
#
# See configs/reconstruction/README.md for full documentation.

input_path: null
output_path: null

preprocessing:
  frame_selection: fps        # fps | optical_flow
  frame_proportion: 0.1       # fraction of frames to extract
  min_frames: 300
  max_frames: null            # null = no cap

pointcloud:
  method: feedforward         # feedforward | sfm
  backend: vggt_omega         # vggt_omega | vggtx | mapanything  (feedforward only)
  bundle_adjustment: false
  loop_closure: false
  clean:
    enabled: true
    outlier_removal: true
    voxel_size: null          # null = adaptive
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

### Dataset config example (`birds_c0043.yaml`)

Only overrides what differs from base. `output_path` is **required** — it is `null` in
`base.yaml` as a sentinel. `Reconstructor.validate_config()` raises `ValueError` if it remains
null at run time, forcing each experiment to go to an explicit, named directory.

```yaml
# Birds - 2024-02-06 - C0043
input_path: /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4
output_path: /workspace/outputs/birds_c0043

preprocessing:
  frame_proportion: 0.25
```

### All 10 dataset configs

| File | input_path | frame_proportion |
|---|---|---|
| `birds_c0043.yaml` | `.../birds/2024-02-06/SplatsSD/C0043.MP4` | 0.25 |
| `birds_c0065.yaml` | `.../birds/2024-05-18/SplatsSD/C0065.MP4` | 0.25 |
| `birds_c0067.yaml` | `.../birds/2024-05-19/SplatsSD/C0067.MP4` | 0.25 |
| `birds_gh010070.yaml` | `.../birds/2024-05-23/SplatsSD/GH010070.MP4` | 0.125 |
| `birds_gh010097.yaml` | `.../birds/2024-05-27/SplatsSD/GH010097.MP4` | 0.14 |
| `birds_gh010105.yaml` | `.../birds/2024-05-27/SplatsSD/GH010105.MP4` | 0.25 |
| `birds_gh010164.yaml` | `.../birds/2024-06-01/SplatsSD/GH010164.MP4` | 0.10 |
| `birds_pxl_20231105.yaml` | `.../birds/2023-11-05/SplatsSD/PXL_20231105_154956078.mp4` | 0.25 |
| `ants_gh010210.yaml` | `.../ants/2025-11-16/SplatsSD/GH010210.MP4` | 0.08 |
| `rats_c0119.yaml` | `.../rats/2024-07-11/SplatsSD/C0119.MP4` | 0.25 |

All `output_path` values: `/workspace/outputs/<dataset_name>` (matches filename stem).

---

## CLI: `docs/reconstruct.py`

### Interface

```bash
# Standard run (uses base defaults: vggt_omega + talk2dino)
python docs/reconstruct.py --dataset birds_c0043

# Specific stages only
python docs/reconstruct.py --dataset birds_c0043 --stages preprocess,pointcloud

# Override any config value at CLI (dotted key=value)
python docs/reconstruct.py --dataset birds_c0043 \
  semantics.extractor=dinov2 \
  pointcloud.backend=vggtx \
  pointcloud.bundle_adjustment=true

# Send output to a custom dir for this experiment
python docs/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_ba_experiment

# Re-run from a prior run's saved config (full reproducibility)
python docs/reconstruct.py \
  --config /workspace/outputs/birds_c0043_ba_experiment/run_config.yaml

# Use a different config dir
python docs/reconstruct.py --dataset custom_scene \
  --config-dir /path/to/my/configs

# Force re-run even if outputs exist
python docs/reconstruct.py --dataset birds_c0043 --overwrite
```

### Arguments

| Arg | Default | Description |
|---|---|---|
| `--dataset NAME` | — | Dataset name → `datasets/<name>.yaml`. Mutually exclusive with `--config`. |
| `--config PATH` | — | Direct YAML path (bypasses dataset hierarchy). For re-running from `run_config.yaml`. |
| `--config-dir PATH` | `configs/reconstruction` | Base of config hierarchy. |
| `--stages s1,s2` | from config | Comma-separated subset: `preprocess,pointcloud,semantics,mesh`. |
| `--overwrite` | false | Re-run stages even if output exists. |
| `KEY=VALUE ...` | — | Dotted-path overrides applied last. Supports bool/int/float/str coercion. |

`--dataset` and `--config` are mutually exclusive. Both can accept `KEY=VALUE` overrides.

### `run_config.yaml` written to output dir

At the start of every run, the fully merged config (base + dataset + overrides) is written
to `<output_path>/run_config.yaml`. This enables:

1. **Auditing**: open any output dir, see exactly what settings produced it.
2. **Reproducibility**: `--config path/to/run_config.yaml` re-runs identically.
3. **Dashboard discovery**: dashboard can read `run_config.yaml` to know which backend/extractor to load.

If `run_config.yaml` already exists and `--overwrite` is not set, it is left untouched (prevents
accidental config mutation on partial re-runs).

### Implementation sketch

```python
#!/usr/bin/env python3
"""CLI entry point for Reconstructor pipeline.

Usage:
    python docs/reconstruct.py --dataset birds_c0043
    python docs/reconstruct.py --config /workspace/outputs/birds_c0043/run_config.yaml
    python docs/reconstruct.py --dataset birds_c0043 pointcloud.backend=vggtx
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
    parser = argparse.ArgumentParser(description="Run Reconstructor pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Dataset name (datasets/<name>.yaml)")
    source.add_argument("--config", type=Path, help="Direct path to a YAML config file")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()

    from collab_splats.wrapper.config import ConfigLoader, parse_cli_overrides
    from collab_splats.wrapper.reconstructor import Reconstructor

    overrides = parse_cli_overrides(args.overrides) if args.overrides else {}
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    if args.config:
        # Load directly from a YAML file (e.g. a saved run_config.yaml)
        with open(args.config) as f:
            config = yaml.safe_load(f)
        from mergedeep import merge
        config = merge({}, config, overrides)
        r = Reconstructor(config)
    else:
        r = Reconstructor.from_config_file(
            dataset=args.dataset,
            config_dir=args.config_dir,
            overrides=overrides or None,
        )

    # Write run_config.yaml to output dir for auditability
    output_path = Path(r.config["output_path"])
    run_cfg_path = output_path / "run_config.yaml"
    if not run_cfg_path.exists() or args.overwrite:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(run_cfg_path, "w") as f:
            yaml.dump(r.config, f, default_flow_style=False, sort_keys=False)
        logger.info("Config written: %s", run_cfg_path)

    r.run_pipeline(stages=stages, overwrite=args.overwrite)
    logger.info("Done. Output: %s", r.backend_dir)


if __name__ == "__main__":
    main()
```

---

## `.gitignore` updates

```
# Eval benchmark data (downloaded on-demand via evals/download_7scenes.sh)
data/

# Reconstruction outputs (written to /workspace/outputs/ outside repo)
outputs/
```

---

## `configs/reconstruction/README.md`

New-user-facing README. Must cover:

1. **Why configs live here, not with the data** — the workspace layout rationale (code versioned,
   data/outputs too large for git, container paths are stable).
2. **How to run** — minimal example: `python docs/reconstruct.py --dataset birds_c0043`.
3. **How to add a dataset** — copy `datasets/birds_c0043.yaml`, set `input_path` + `output_path`,
   tune `frame_proportion`. Two-minute job.
4. **How to run experiments** — override `output_path` at CLI to separate experiment dirs;
   re-run from `run_config.yaml` for exact reproducibility.
5. **Config key reference** — table of all base.yaml keys with type, default, and description.
6. **Where outputs land** — describes `run_config.yaml`, `feedforward.zarr`, `semantics/`,
   `mesh/` within `<output_path>/<backend>/`.
7. **Dashboard** — one-liner: point dashboard at `output_path` to visualise results.

---

## Dashboard integration

No code changes required. The dashboard reads from `backend_dir / "semantics" / <extractor> /
"features.zarr"` — which is exactly what `Reconstructor` writes. Point the dashboard at
`output_path` (or `output_path/<backend>/`) to visualise results.

Future enhancement (out of scope): dashboard auto-discovers `run_config.yaml` to pre-populate
backend and extractor dropdowns.

---

## Tests

`tests/scripts/test_reconstruct.py`:
- Mock `Reconstructor.from_config_file` and `run_pipeline`
- Verify `--dataset` + overrides parse and reach `Reconstructor` correctly
- Verify `--stages preprocess,pointcloud` → `["preprocess", "pointcloud"]`
- Verify `--config path.yaml` loads YAML directly (bypass hierarchy)
- Verify `run_config.yaml` is written to output dir

---

## Out of scope

- Dashboard auto-discovery of `run_config.yaml`
- Batch runner (`for dataset in configs/reconstruction/datasets/*.yaml`)
- Localization stage wiring (stage exists but not yet implemented)
- Moving `fieldwork-data/` or `outputs/` — those dirs are pre-existing container volumes
