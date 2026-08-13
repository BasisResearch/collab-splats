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
  --config my_overrides.yaml /workspace/fieldwork-data/birds/*.MP4   # your own YAML, merged over base.yaml

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
  frames.zarr                  ← decode-once keyframe store (there is no images/ dir)
  semantics/
    <extractor>.zarr           ← 2D patch cache, one per extractor (backend-agnostic)
  <backend>/                   ← e.g. vggt_omega/
    feedforward.zarr           ← depth maps, poses, confidence, 3D points
                               ←   (+ local_features/<extractor>/reconstruction if localize ran)
    sparse_pc.ply
    mesh.ply                   ← (only if mesh.enabled=true)
    semantics/
      <extractor>_lifted.zarr  ← lifted 3D features (N_points × latent_dim)
      <extractor>_ae.pt        ← autoencoder weights (if semantics.n_components set)
```

---

## The two drivers

There are two entry points. They share `collab_splats/wrapper/batch.py`, so the stages,
the config merge, and the `run_config.yaml` they leave behind are identical — they differ
only in where scenes come from and what happens afterwards.

| driver | input | where a scene lands |
|---|---|---|
| `docs/examples/run_pipeline.py` | local videos and/or directories of videos (dirs are globbed for `*.mp4`/`*.mov`/`*.avi`) | `<output-root>/<session-date>/<video-stem>/` when a date-like dir (`YYYY-MM-DD`) appears in the video's path, else `<output-root>/<video-stem>/` |
| `docs/examples/run_pipeline_remote.py` | scenes in the `environments-curated` GCS bucket — named scene ids, or `--all` | `<output-root>/<scene>/`, deleted again after a verified push |

Shared flags: `--output-root` (required), `--config` (override YAML merged over
`base.yaml`), `--config-dir`, `--stages`, `--overwrite`. `run_pipeline.py` adds
`--keep-viewer` (keep the viser viewer alive for browser inspection).
`run_pipeline_remote.py` adds `--all` and `--keep-local`.

### Remote scenes

Scene ids are the curated directory names: `YYYY_MM_DD-PARENTFOLDER-VIDEONAME`.

```bash
# Named scenes
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \
  2026_07_20-birds-C0043 2026_07_21-rats-C0100

# Everything in the bucket
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all

# Keep the local copy for inspection (skips the delete, not the push)
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all --keep-local
```

Per scene: pull the video, reconstruct, push to `environments-processed/<scene>/`, verify
the push with `rclone check --one-way`, then delete the local copy. The curated video
stays in the bucket, so a deleted scene is always re-fetchable.

Nothing local is deleted until the push verifies. A failed verification leaves all local
data in place, and so does a failed reconstruction — the local scene dir is named in a
warning for inspection or retry. `--keep-local` skips the delete only; the push and the
verification still run. One scene's failure does not abort the batch.

A failure that turns out to be rclone itself is treated differently. After any scene
failure the driver re-probes the remote, and if rclone has become unreachable it stops
instead of marching the rest of the list into the same fault. No exception type or exit
code can distinguish the two cases — a dead remote and a reconstruction error both
surface as a bare `RuntimeError` — so the question is asked directly. Scenes that never
ran are reported `SKIPPED`, not `FAIL`, and can be re-run unchanged.

| exit | meaning |
|---|---|
| 0 | every scene succeeded |
| 1 | one or more scenes failed on their own merits; the rest still ran |
| 2 | nothing to do (no scenes named, none curated) |
| 3 | aborted early — rclone became unreachable, so later scenes never ran |

An unattended runner should treat 3 as an infrastructure alert and 1 as a data problem.
**3 outranks 1**: a run where a scene failed and rclone then died exits 3, deliberately — the
transport fault is the actionable cause, and the scenes that never ran are safe to retry as-is.

Credentials come from the existing rclone remote (`collab-data`) — nothing is read from
the environment or passed on the command line.

### Re-running one stage against a processed scene

`--stages` naming only *leaf* stages — `mesh`, `semantics`, `localize` — pulls the scene back
out of `environments-processed` instead of rebuilding it from its curated video:

```bash
# Re-mesh every processed scene with a new voxel size
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \
  --stages mesh --overwrite --config remesh.yaml --all

# Add semantics to one scene reconstructed without it (no --overwrite: nothing to replace)
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \
  --stages semantics 2026_07_20-birds-C0043
```

A leaf stage is one nothing else depends on, so re-running it cannot invalidate anything
downstream. Any `--stages` set that includes `preproc` or `pointcloud` therefore takes the
normal path — full rebuild from the curated video — and a run can never leave a stale
downstream artifact next to a fresh upstream one. The rule is derived from the dependency
graph in `Reconstructor`, not a hardcoded list.

With `--all`, the bucket listed follows the same rule: a leaf re-run enumerates
`environments-processed`, everything else enumerates `environments-curated`.

The whole scene is pulled, with no excludes. `PULL_EXCLUDES` is the *viewer's* default and
drops `depth`/`world_points`/`images` — exactly what meshing reads.

Five things are errors rather than surprises, and each fails only its own scene:

| situation | outcome |
|---|---|
| scene has no processed outputs | `FileNotFoundError` — run the full pipeline first |
| pulled scene has no `run_config.yaml` | `FileNotFoundError` — the backend is unknowable |
| pulled `run_config.yaml` has no `pointcloud.backend` | `ValueError` — the backend is unknowable |
| `--config` backend ≠ pulled backend | `ValueError` naming both — never a silent retarget |
| named leaf stage's output already exists | `ValueError` — pass `--overwrite` to replace it |

That last one applies only to *leaf* stages named on `--stages`. When `--stages` is omitted the
set comes from the config's `enabled` flags, where skipping completed stages is what makes a
re-run resume rather than fail. Naming a leaf stage means asking for it; inheriting it from
config does not. Named *non-leaf* stages also still skip when already done — that is how
`--stages preproc,pointcloud,localize` resumes after a `localize` failure.

Config for a re-run is the pulled `run_config.yaml` minus the sections of the stages being
re-run, with `base.yaml` and `--config` supplying fresh parameters for exactly those. Provenance
for every stage that is *not* re-running is preserved verbatim.

Nothing here deletes a remote object. The push is still `rclone copy`, so a re-run overwrites
the artifacts it produced and leaves everything else in place.

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

## Choosing a frame sampler

Each method has exactly one density knob. `max_frames` is the frame budget — the
target count for `uniform`, a ceiling for the other two.

| `frame_selection` | Density knob | What it holds constant |
|---|---|---|
| `fps` | `preprocessing.fps` | Wall-clock interval between frames — so the baseline between consecutive frames is fixed regardless of how long the video is. Count floats. |
| `uniform` | `preprocessing.max_frames` | Frame count. Spacing floats with video length. |
| `optical_flow` | `min_disparity` (creator-level) | Inter-frame motion. Both count and spacing float. |

Prefer `fps` for reconstruction: registration quality depends on the baseline
between consecutive frames, and a count-based knob leaves that free to vary by an
order of magnitude between a 1-minute and a 20-minute video.

**The band.** `fps` yields a count that grows with video length, so
`[min_frames, max_frames]` bounds it. Outside the band the targets are re-spread
evenly across the **whole** video and the effective fps is logged at WARNING —
never truncated, which would hand the reconstructor a scene that stops halfway.

**`max_frames` still dominates on long video.** At `fps: 1.0` the default
`max_frames: 300` binds past ~5 minutes, and beyond that the spacing is whatever
300 frames over the whole video gives you. The cap is a measured GPU limit, not a
preference — `fps` cannot route around it. That limit is `vggt_omega`'s, though, not
the pipeline's: `loger` is windowed and is expected to run well past 300 frames, but
its own ceiling has **not yet been swept**, so the default stays where VGGT-Omega
needs it.

**Passing a knob that belongs to another method raises `ValueError`** (e.g. `fps=`
with `frame_selection: uniform`). There is no silently-ignored knob.

---

## Config key reference (`base.yaml`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_path` | str | **set per run** | Absolute path to video (.MP4) or image directory |
| `output_path` | str | **set per run** | Absolute path for outputs (created if absent) |
| `preprocessing.frame_selection` | str | `fps` | Frame sampling: `fps`, `uniform`, or `optical_flow` |
| `preprocessing.fps` | float | `1.0` | `fps` method only: samples per second |
| `preprocessing.min_frames` | int\|null | `null` | `fps` method only: floor on the resulting count |
| `preprocessing.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300 — not a LoGeR limit, see below) |
| `pointcloud.method` | str | `feedforward` | `feedforward`, `sfm`, or `nerfstudio` |
| `pointcloud.backend` | str | `vggt_omega` | `vggt_omega`, `vggtx`, `mapanything`, or `loger` |
| `pointcloud.<backend>` | dict | `{}` | Per-backend creator kwargs, e.g. `pointcloud.loger.window_size`. Only the block matching `backend` is read. `max_points` is rejected here. |
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

### The `loger` backend

LoGeR is a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window
inference. Requires `bash setup/loger.sh` once; weights download from HuggingFace on
first use.

**Choose it for:** sequences past the ~300-frame ceiling where VGGT-Omega OOMs, and long
captures where drift accumulates — the TTT memory is designed to carry state across the
sequence.

**Avoid it for:** short sequences — reasoning, not measured: below roughly a couple of
windows the set-based VGGT models see every frame jointly and the windowing buys nothing,
but no crossover has been swept; anything needing loop closure, which
`loger` refuses (thresholds are calibrated per backbone and none exists yet); captures
where intrinsics genuinely vary, e.g. zoom, which the shared-K fit cannot represent; and
unordered image collections — LoGeR's windows are sequential, whereas the VGGT family is
set-based and has no ordering requirement.

**Intrinsics differ from every other backend.** VGGT-family backends and MapAnything
*predict* K. LoGeR does not: K is *solved* from its predicted pointmap by a
confidence-weighted median pinhole fit, shared across all frames. The failure modes are
inverted — a predicted K can be geometrically invalid (a principal point outside the
image), whereas a fitted K is centre-principal by construction but can be
plausibly-but-globally-wrong. There is no fallback focal; a degenerate fit raises.

**`max_frames` is not tuned for LoGeR.** The default 300 is VGGT-Omega's GPU limit and
lives in the preproc stage, which runs first. Raise it to use LoGeR's windowing. The real
ceiling is `FeedforwardResult`, which holds dense per-frame images, world points, depth,
and confidence — 8.13 MB/frame at LoGeR's default `pixel_limit` of 255,000 — against a
46.6 GB container cap. The cap binds every backend; the per-frame figure is LoGeR's own,
since each backend resolves frames differently. LoGeR is merely the first backend able to
feed the buffer enough frames for the cap to matter.

---

## Where outputs land

```
<output_path>/
  run_config.yaml              ← full merged config (exact settings used — for reproducibility)
  frames.zarr                  ← canonical decode-once keyframe store (chunked images + records + provenance)
  semantics/
    <extractor>.zarr           ← 2D patch cache, one per extractor (backend-agnostic)
  <backend>/                   ← e.g. vggt_omega/
    feedforward.zarr           ← depth maps, poses, confidence, 3D points
    sparse_pc.ply
    mesh.ply                   ← (only if mesh.enabled=true)
    semantics/
      <extractor>_lifted.zarr  ← lifted 3D features (N_points × n_components)
      <extractor>_ae.pt        ← autoencoder weights, needed to decode them (if n_components set)
```

The 2D patch cache sits at the scene root because it depends only on the frames; the lift is
what depends on the backend, so `<extractor>_lifted.zarr` sits under `<backend>/`. The
`_lifted` suffix is what separates the two in the dashboard's flat layout, where both live in
one `semantics/` dir. Naming both halves after the extractor also lets two extractors coexist
in the same scene instead of overwriting each other.

### Processed scene layout

A processed scene (`environments-processed/<scene>/`) carries:

| path | consumer |
|---|---|
| `<backend>/sparse_pc.ply` | any pipeline — binary little-endian, float32 xyz + uchar rgb |
| `<backend>/mesh.ply` | any pipeline |
| `<backend>/transforms.json` | camera poses, `ply_file_path`, `applied_transform` (splatfacto) |
| `<backend>/semantics/<extractor>_lifted.zarr` | per-point latent codes (`semantics.n_components`-D) |
| `<backend>/semantics/<extractor>_ae.pt` | decoder to full 768-D + `recon_cosine` / `recon_mse` |
| `<backend>/colmap/sparse/0/*.bin` | further processing inside this repo |
| `<backend>/feedforward.zarr` | further processing inside this repo (depth, poses, confidence) |
| `frames.zarr` | the keyframes the reconstruction was built from; required to localize |
| `run_config.yaml` | exact settings used |

Every geometry-derived artifact sits under `<backend>/`, including `sparse_pc.ply` and the
per-point semantics. One scene may be reconstructed by several backends, so a per-point
latent code is only meaningful next to the point set it indexes. `frames.zarr` sits at the
scene root instead: the keyframes are decoded once from the video and shared by every
backend that reconstructs the scene.

Not pushed (`PUSH_EXCLUDES` in `collab_splats/remote/sources.py`): `/semantics/**` at the
scene root (raw 2D patch maps, regenerable from frames + extractor — note the leading slash,
which is what keeps `<backend>/semantics/**` in the push) and the source video, which the remote
driver fetches into the very scene dir it later pushes and which already lives in
`environments-curated`. `frames.zarr` **is** pushed — it is the sole persistent keyframe
store, so localization or a correspondence plot against a published scene works directly,
with no re-decode of the curated video.

Comparing semantics across scenes: decode to 768-D first. Two independently-trained
autoencoders do not share a 64-D basis, so raw latent codes are not comparable; the
decoded space is. Each scene's `recon_cosine` is measured on the **training set**, not a
held-out split, so treat it as an upper bound on fidelity rather than a generalisation
estimate.

#### `ns-train --data` does not work on a published scene, by design

Both stock nerfstudio dataparser routes need real image files on disk. The nerfstudio
dataparser calls `Path(frame["file_path"])` unconditionally
(`nerfstudio_dataparser.py:127,134`), and the COLMAP dataparser resolves
`data/images/{im_data.name}` (`colmap_dataparser.py:93,176`). Our `transforms.json` frames
key on `frame_idx` against `frames.zarr` instead, and there is no `images/` directory — it
was removed in the frame-store migration. No `images/` export will be added. Downstream
consumers get `sparse_pc.ply` + the mesh + the features + the raw COLMAP binaries, and
read poses via `pycolmap`. Do not expect `ns-train --data` to work against one of these
scenes.

#### The dashboard cannot browse a scene published by the remote driver

The published tree is backend-keyed (`<scene>/<backend>/feedforward.zarr`) and the dashboard reads
flat (`<scene>/feedforward.zarr`, `<scene>/semantics/`, `<scene>/mesh.ply`). Pointing the
dashboard at a published scene does not just fail to load — its existence gate never trips, so the
scene re-pulls from GCS on every select and then errors in the op log. This is deliberate: the
dashboard's flat layout is what every dashboard scene already on disk uses, and unifying the read
path would orphan them all. Use the viewer or a notebook for published scenes.

#### Scenes reconstructed before the layout rename need one re-run

The TSDF output is now `<backend>/mesh.ply` (was `mesh/mesh_tsdf.ply`, then `mesh/mesh.ply`),
so older scenes make `wrapper/splatter.py` raise `FileNotFoundError` and the dashboard show no
mesh. Semantics moved the same way: the 2D cache is `<scene>/semantics/<extractor>.zarr` (was
`features/<extractor>/<extractor>.zarr`) and the lifted pair is
`<backend>/semantics/<extractor>_lifted.zarr` + `_ae.pt` (was
`<backend>/semantics/<extractor>/features.zarr` + `autoencoder.pt`). There are deliberately no
legacy fallbacks — they would restore exactly the two-name ambiguity the rename removed. Such
scenes need `mesh(overwrite=True)` and `extract_semantics(overwrite=True)` run once; the 2D
cache re-extracts, which is the expensive half.

---

## Dashboard

The dashboard browses **dashboard-produced scenes**, whose tree is flat: it reads
`<scene>/feedforward.zarr` for the pointcloud, `<scene>/semantics/<extractor>_lifted.zarr` for
semantic features, and `<scene>/mesh.ply` for the mesh. Point it at a scene the dashboard itself
built. It cannot read the backend-keyed tree the remote driver publishes — see "The dashboard
cannot browse a scene published by the remote driver" under Where outputs land.
