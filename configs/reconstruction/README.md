# Reconstruction Pipeline Configs

Configuration templates for `scripts/reconstruct.py`. Each run of the pipeline
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
    scripts/reconstruct.py
    data/                      ← gitignored — eval benchmarks, downloaded on-demand

  fieldwork-data/              ← input videos (never in repo, ~GB each)
  outputs/                     ← reconstruction outputs (never in repo, ~10-50 GB per scene)
    birds_c0043_omega/
      run_config.yaml          ← exact settings that produced this output
      vggt_omega/
        feedforward.zarr
        semantics/talk2dino/features.zarr
```

**Why absolute paths?** This project runs in a fixed container environment
(`/workspace/` mount). Absolute `/workspace/...` paths are stable across sessions.
Portability across environments is handled by the container image + volume mounts,
not by relative paths in config files.

**Why not put configs with the data?** Configs are small text files that describe
*intent* — what to run and how. They benefit from version control (git blame, diffs,
review). Output data is large, mutable, and reproducible from the configs — it doesn't
belong in git.

---

## Running a dataset

```bash
# Standard run (uses base defaults: vggt_omega + talk2dino, semantics on)
python scripts/reconstruct.py --dataset birds_c0043

# Specific stages only
python scripts/reconstruct.py --dataset birds_c0043 --stages preprocess,pointcloud

# Override any config key at CLI (dotted key=value, any depth)
python scripts/reconstruct.py --dataset birds_c0043 \
  semantics.extractor=dinov2 \
  pointcloud.bundle_adjustment=true

# Experiment variant — send output to a separate dir
python scripts/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_ba_experiment

# Re-run from a saved config for exact reproducibility
python scripts/reconstruct.py \
  --config /workspace/outputs/birds_c0043_omega/run_config.yaml

# Force re-run even if outputs exist
python scripts/reconstruct.py --dataset birds_c0043 --overwrite
```

---

## Adding a new dataset

1. Copy `datasets/birds_c0043.yaml` to `datasets/<your_name>.yaml`
2. Set `input_path` to the absolute path of the video or image directory
3. Set `output_path` to where outputs should be written (e.g. `/workspace/outputs/<your_name>`)
4. Tune `preprocessing.frame_proportion` if needed (higher = more frames = slower but denser)
5. Run: `python scripts/reconstruct.py --dataset <your_name>`

Everything else inherits from `base.yaml`. Only override what differs.

---

## Running experiments (multiple configs, same dataset)

Don't create a new dataset YAML for each experiment. Override `output_path` at CLI instead:

```bash
# Experiment A: omega + talk2dino (base defaults)
python scripts/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_omega_t2d

# Experiment B: vggtx + dinov2
python scripts/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_vggtx_dino \
  pointcloud.backend=vggtx \
  semantics.extractor=dinov2

# Experiment C: omega + BA enabled
python scripts/reconstruct.py --dataset birds_c0043 \
  output_path=/workspace/outputs/birds_c0043_omega_ba \
  pointcloud.bundle_adjustment=true
```

Each output dir gets its own `run_config.yaml` recording the exact settings used.
Reproduce any experiment: `python scripts/reconstruct.py --config path/to/run_config.yaml`

---

## Config key reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_path` | str | **required** | Absolute path to video (.MP4) or image directory |
| `output_path` | str | **required** | Absolute path for outputs (created if absent) |
| `preprocessing.frame_selection` | str | `fps` | Frame sampling: `fps` or `optical_flow` |
| `preprocessing.frame_proportion` | float | `0.1` | Fraction of total frames to extract |
| `preprocessing.min_frames` | int | `300` | Minimum frames regardless of proportion |
| `preprocessing.max_frames` | int\|null | `null` | Cap on frames; null = no cap |
| `pointcloud.method` | str | `feedforward` | `feedforward` or `sfm` |
| `pointcloud.backend` | str | `vggt_omega` | `vggt_omega`, `vggtx`, or `mapanything` |
| `pointcloud.bundle_adjustment` | bool | `false` | Run LM bundle adjustment after pointcloud |
| `pointcloud.loop_closure` | bool | `false` | Run loop closure after pointcloud |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points |
| `semantics.enabled` | bool | `true` | Extract and lift semantic features |
| `semantics.extractor` | str | `talk2dino` | `talk2dino`, `dinov2`, or `maskclip` |
| `semantics.n_components` | int\|null | `64` | PCA compression dim; null = no compression |
| `mesh.enabled` | bool | `false` | Build TSDF/Poisson mesh (opt-in) |
| `mesh.mesher` | str | `tsdf` | `tsdf` or `poisson` |
| `mesh.voxel_size` | float | `0.01` | TSDF voxel size in metres |
| `mesh.sdf_trunc` | float | `0.04` | TSDF truncation distance in metres |

---

## Where outputs land

```
<output_path>/
  run_config.yaml              ← full merged config (exact settings used — for reproducibility)
  images/                      ← extracted keyframes
  features/                    ← 2D feature cache (one subdir per extractor)
  <backend>/                   ← e.g. vggt_omega/
    feedforward.zarr           ← depth maps, poses, confidence, 3D points
    semantics/
      <extractor>/
        features.zarr          ← lifted 3D features (N_points × n_components)
        compressor.pt          ← PCA compressor weights (if n_components set)
    mesh/
      mesh.ply                 ← (only if mesh.enabled=true)
```

---

## Dashboard

Point the dashboard at `output_path` to visualise results. The dashboard reads
`<backend>/feedforward.zarr` for the pointcloud and
`<backend>/semantics/<extractor>/features.zarr` for semantic features.
