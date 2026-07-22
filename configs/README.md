# Reconstruction Pipeline Configs

`base.yaml` is the single template for the reconstruction pipeline. It defines every
stage's defaults; each run sets `input_path` / `output_path` and overrides only what
differs. There is no per-dataset config file — you point the runner at video paths.

---

## Running videos

Use `docs/examples/run_pipeline.py` — the top-level entry point. Point it at a single
video, several videos, or directories of videos:

```bash
# Single video
python docs/examples/run_pipeline.py --output-root /workspace/outputs scene.MP4

# Several videos + a directory (dirs are globbed for *.mp4/*.mov/*.avi)
python docs/examples/run_pipeline.py --output-root /workspace/outputs \
  /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4 \
  /workspace/fieldwork-data/rats/2024-07-11/SplatsSD/

# Turn on extra stages via a shared override YAML
python docs/examples/run_pipeline.py --output-root /workspace/outputs \
  --config configs/minimal.yaml /workspace/fieldwork-data/birds/*.MP4

# Specific steps only
python docs/examples/run_pipeline.py --output-root /workspace/outputs \
  --stages preproc,pointcloud,localize scene.MP4
```

### Steps run per video

1. **keyframe extraction** — sample sharp, well-exposed frames from the video
2. **vggt_omega pointcloud** — feed-forward 3D reconstruction (poses + depth + points)
3. **talk2dino semantics** — 2D features lifted to 3D, autoencoder-compressed
4. **localization database** — per-frame local-feature cache for camera localization

`preprocess` + `pointcloud` always run. `semantics`, `mesh`, and `localize` run only
when enabled in the config (`semantics.enabled` / `mesh.enabled` /
`localization.enabled`), or when named explicitly via `--stages`.

### Where outputs land

Each video is written to `<output-root>/<session-date>/<video-stem>/` when a date-like
dir (`YYYY-MM-DD`) appears in the video's path, else `<output-root>/<video-stem>/`:

```
<output-root>/2024_02_06/C0043/
  run_config.yaml              ← full merged config (exact settings used — for reproducibility)
  images/                      ← extracted keyframes
  features/                    ← 2D feature cache (one subdir per extractor)
  <backend>/                   ← e.g. vggt_omega/
    feedforward.zarr           ← depth maps, poses, confidence, 3D points
                               ←   (+ local_features/<extractor>/reconstruction if localize ran)
    semantics/
      <extractor>/
        features.zarr          ← lifted 3D features (N_points × latent_dim)
        compressor.pt          ← autoencoder weights (if semantics.n_components set)
    mesh/
      mesh.ply                 ← (only if mesh.enabled=true)
```

---

## Reproducing an exact run

Every output dir gets a `run_config.yaml` recording the exact settings used. Re-run it
with `docs/examples/reconstruct.py` (the `input_path` / `output_path` are baked in):

```bash
python docs/examples/reconstruct.py \
  --config /workspace/outputs/2024_02_06/C0043/run_config.yaml

# Tweak a saved run and send it to a separate dir
python docs/examples/reconstruct.py \
  --config /workspace/outputs/2024_02_06/C0043/run_config.yaml \
  output_path=/workspace/outputs/2024_02_06/C0043_vggtx \
  pointcloud.backend=vggtx
```

---

## Why configs live here (not with the data)

This project separates **code + configs** (versioned in git) from **data + outputs**
(too large for git, live outside the repo):

```
/workspace/
  collab-splats/               ← this repo (configs live here)
    configs/base.yaml
    docs/examples/run_pipeline.py
  fieldwork-data/              ← input videos (never in repo, ~GB each)
  outputs/                     ← reconstruction outputs (never in repo, ~10-50 GB per scene)
```

Absolute `/workspace/...` paths are stable across sessions in the fixed container.
Config is a small text file describing *intent* (what to run, how); output data is
large, mutable, and reproducible from `run_config.yaml` — it doesn't belong in git.

---

## Config key reference (`base.yaml`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_path` | str | **set per run** | Absolute path to video (.MP4) or image directory |
| `output_path` | str | **set per run** | Absolute path for outputs (created if absent) |
| `preprocessing.frame_selection` | str | `fps` | Frame sampling: `fps` or `optical_flow` |
| `preprocessing.frame_proportion` | float | `0.1` | Fraction of total frames to extract |
| `preprocessing.min_frames` | int | `150` | Minimum frames regardless of proportion |
| `preprocessing.max_frames` | int\|null | `200` | Cap on frames (vggt_omega OOMs above ~300) |
| `pointcloud.method` | str | `feedforward` | `feedforward`, `sfm`, or `nerfstudio` |
| `pointcloud.backend` | str | `vggt_omega` | `vggt_omega`, `vggtx`, or `mapanything` |
| `pointcloud.bundle_adjustment` | bool | `false` | Run LM bundle adjustment after pointcloud |
| `pointcloud.loop_closure` | bool | `false` | Run loop closure after pointcloud |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points |
| `semantics.enabled` | bool | `true` | Extract and lift semantic features |
| `semantics.extractor` | str | `talk2dino` | `talk2dino`, `dinov2`, or `maskclip` |
| `semantics.n_components` | int\|null | `64` | Autoencoder latent dim; null = no compression |
| `mesh.enabled` | bool | `false` | Build TSDF/Poisson mesh (opt-in) |
| `mesh.mesher` | str | `tsdf` | `tsdf` or `poisson` |
| `mesh.voxel_size` | float | `0.01` | TSDF voxel size in metres |
| `mesh.sdf_trunc` | float | `0.04` | TSDF truncation distance in metres |
| `localization.enabled` | bool | `false` | Build the localization database (opt-in) |
| `localization.extractor` | str | `loma` | Local matcher: `loma`, `loma-g`, `disk`, `xfeat` |

---

## Where outputs land

```
<output_path>/
  run_config.yaml              ← full merged config (exact settings used — for reproducibility)
  frames.zarr                  ← canonical decode-once keyframe store (chunked images + records + provenance)
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

Point the dashboard at a scene's output dir to visualise results. It reads
`<backend>/feedforward.zarr` for the pointcloud and
`<backend>/semantics/<extractor>/features.zarr` for semantic features.
