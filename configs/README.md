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
  images/frame_NNNNNN.png      ← decode-once keyframe store (COLMAP-style dir, lossless PNG)
  frames.json                  ← selection records + provenance for those frames
  video_quality_report.json    ← per-frame photometry + per-pair motion of the source video
  photometric.png              ← the report rendered: blur / laplacian / exposure / clipped fractions
  motion.png                   ←   per-pair translation / parallax (failed pairs = red | at 0) / matches
  semantics/
    <extractor>.zarr           ← 2D patch cache, one per extractor (backend-agnostic)
  <backend>/                   ← e.g. vggt_omega/ (or instantsfm/ for method: sfm)
    pointcloud.zarr            ← depth maps, poses, 3D points (+ confidence when the method produces it)
                               ←   (+ local_features/<extractor>/reconstruction if localize ran)
    sparse_pc.ply
    mesh.ply                   ← (only if mesh.enabled=true)
    texture/                   ← mesh.ply (UV-carrying) + albedo.png (only if mesh.texture=true)
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

Scene ids are the curated directory names. `YYYY_MM_DD-PARENTFOLDER-VIDEONAME` is the common
convention, but any flat path-safe name is a valid scene (e.g. the
`audiomoth_only_deployments-...` deployments); dirs that fail the safety filter are named in
the driver's log.

```bash
# Named scenes
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \
  2026_07_20-birds-C0043 2026_07_21-rats-C0100

# Everything in the bucket
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all

# Keep the local copy for inspection (skips the delete, not the push)
python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all --keep-local
```

Scenes that already exist in `environments-processed` are skipped (a `SKIPPED` row in the
summary) unless `--overwrite` is passed, so an `--all` run only processes what is new and a
batch with nothing to rebuild exits 0. The check is directory presence, so a partially-pushed
scene counts as processed — `--overwrite` (with the scene named) is the way to redo it.
Leaf-stage re-runs are exempt: their work list comes from the processed bucket by definition,
and the per-stage refusal below governs overwrite there.

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

`--stages` naming only *leaf* stages — `refine`, `mesh`, `semantics`, `splats`, `localize`, `verify`,
`reconstruction_quality_report` — pulls the scene back
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

`verify` is a leaf stage: `--stages verify` re-runs geometric verification against a
processed scene (needs `colmap/` + `pointcloud.zarr` locally). Outputs under
`<backend>/colmap/`: `verified/` (COLMAP model whose points carry real feature tracks;
poses/cameras identical to `sparse/0`), `verification.json` (per-pair epipolar +
relative-pose stats, per-frame track survival and reprojection error), and `database.db`
(local build artifact, excluded from pushes). `sparse/0` is never modified.

- `<backend>/reconstruction_quality_report.json` — reference-free scene error
  report. One per-pair table (keyed on frame index, so epipolar and depth columns
  join), a per-frame table, per-frame percentile ranks, running-error curves along
  the trajectory, and rank correlations for error-vs-depth, error-vs-separation and
  confidence-vs-error. Written by the always-on `reconstruction_quality_report` leaf
  stage; re-runnable with `--stages reconstruction_quality_report --overwrite`.

  Named for what it scores. The sibling artefact `video_quality_report.json` scores
  the capture — blur, exposure, parallax — before any reconstruction exists; this one
  scores the reconstruction built from it.

  The stage runs no model and no matcher. It loads `colmap/verification.json`
  when it exists; in a full pipeline run verify is ordered ahead of it, so the
  report reads verify's output rather than triggering it. With the shipping
  default (`geometric_verification: false`) no such file is produced, the
  epipolar block records `{"available": false, "reason": ...}`, and the depth and
  photometric channels still emit — the report never reaches around an explicit
  opt-out to charge a default run for verify. To get the epipolar channel, set
  `pointcloud.geometric_verification: true` or run `--stages verify`. Note that
  `--stages reconstruction_quality_report` on its own, with the flag on and no
  `verification.json` present, *will* run verify first and pay its cost.

  **Report-only: nothing here feeds back into the reconstruction.** No verdict,
  no grade, no cause — distributions and cumulative error only. Every block
  stamps its `grid` (`model` or `original`) and `resolution`; units are
  scale-free or normalised throughout, because 1 recon unit is not 1 metre and
  the factor differs per scene and per backbone. Pixel counts are not comparable
  across backbones, so reprojection is reported in px *and* as a fraction of
  image width.

  Per-pair columns ship as raw values, so any binning or threshold query is
  something the reader does. The one exception is the per-pixel depth residual,
  which is too large to hold (N²·H·W) and ships as `counts` + `bin_edges`. Those
  bins are over `u = r/(1+|r|)`, a monotone map onto (−1, 1): no residual can
  fall outside them however large, so nothing is clipped and nothing is dropped.
  Invert a bin edge or quantile with `u/(1−|u|)`, and query any threshold with
  `rv_histogram((counts, bin_edges)).cdf(x/(1+abs(x)))`.

`refine` (LM bundle adjustment) rewrites the reconstruction's poses in place —
COLMAP, `sparse_pc.ply`, and the pose-derived arrays in
`pointcloud.zarr`. It does NOT invalidate `mesh/`, lifted semantics, or the
localization DB built under the old poses: after `--stages refine`, re-run those
stages with `overwrite` if pose-sensitive outputs matter. Provenance for the last
refine run (BA config + LM loss history) is in `<backend>/colmap/refine.json`.

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

Preproc runs in two steps: **measure**, then **select**.

1. **Measure.** `compute_video_quality` decodes the video once and writes
   `video_quality_report.json` beside `images/` — per-frame photometry (blur,
   exposure, clipping) and per-pair motion (matches, translation, parallax). The
   report is **report-only**: it carries measurements, never thresholds and never
   a usable/unusable verdict. It is reused **by existence** — if the file is
   already there the decode pass is skipped, so delete it to re-measure.
   `extract_frames` also renders the report to two PNGs beside it
   (`photometric.png`, `motion.png`; plotters in `preproc/viz.py`), each panel with
   a marginal histogram and the kept frames marked by faint green lines. `motion.png`
   is skipped when the report has no pairs. They are written whenever
   frames are extracted and never otherwise — scenes processed before 2026-08-23
   have no PNGs until `preprocess(overwrite=True)` re-extracts.
2. **Select.** The sampler reads that report through `filter_frame_quality`,
   which cuts on a robust MAD z-score over `log(laplacian)` (`sharpness_k`, default
   2.0 — relative to the video's own sharpness spread, not an absolute value) plus an
   absolute ceiling on `clipped_low_frac + clipped_high_frac` (`max_clipped_frac`,
   default 0.25). Changing sampling policy never re-decodes the video.

`preproc.n_workers` affects **only** step 1 — it parallelises the measurement
decode and has no effect on which frames get selected. The report is byte-
identical at any worker count.

Each method has exactly one density knob. `max_frames` is the frame budget — the
target count for `uniform`, a ceiling for the other two.

| `frame_selection` | Density knob | What it holds constant |
|---|---|---|
| `fps` | `preproc.fps` | Wall-clock interval between frames — so the baseline between consecutive frames is fixed regardless of how long the video is. Count floats. |
| `uniform` | `preproc.max_frames` | Frame count. Spacing floats with video length. |
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

**A knob that belongs to another method is not silently ignored.** Each method is
its own function — `sample_fps`, `sample_uniform`, `sample_optical_flow` — so
`fps=` under `frame_selection: uniform` reaches a signature that has no such
parameter and raises.

---

## Config key reference (`base.yaml`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_path` | str | **set per run** | Absolute path to video (.MP4) or image directory |
| `output_path` | str | **set per run** | Absolute path for outputs (created if absent) |
| `preproc.frame_selection` | str | `fps` | Frame sampling: `fps`, `uniform`, or `optical_flow` |
| `preproc.fps` | float | `1.0` | `fps` method only: samples per second |
| `preproc.min_frames` | int\|null | `null` | `fps` method only: floor on the resulting count |
| `preproc.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300 — not a LoGeR limit, see below) |
| `preproc.n_workers` | int | `4` | Quality-report parallelism: decode+measure this many frame ranges at once. `1` = serial. Not auto-derived (`os.cpu_count()` reports host cores in a container). Set to `1` during a GPU eval run. |
| `preproc.undistort` | bool | `false` | Self-calibrate one shared OPENCV camera (pycolmap, ≤60 frames) and undistort every selected frame (cv2, alpha=0 crop) before `images/` is written. `images/` reuse is by existence — toggling on an existing scene needs `preprocess(overwrite=True)`. Localization query images are not undistorted. |
| `pointcloud.method` | str | `feedforward` | `feedforward` or `sfm` |
| `pointcloud.backend` | str | `vggt_omega` | feedforward: `vggt_omega`, `vggtx`, `mapanything`, or `loger`; sfm: `instantsfm` only — `colmap`/`hloc` are rejected at config validation. `ColmapCreator`/`HlocCreator` exist in `pointcloud/sfm/` but nothing dispatches to them. |
| `pointcloud.<backend>` | dict | `{}` | Per-backend creator kwargs, e.g. `pointcloud.loger.window_size`. Only the block matching `backend` is read. `max_points` is rejected here. |
| `pointcloud.instantsfm.random_seed` | int\|null | `null` | Seed InstantSfM's `RUNTIME_OPTIONS` (numpy/random/torch/cuda). Upstream `InitializeRandomPositions` draws unseeded, so two runs of one scene differ. `null` = upstream behaviour |
| `pointcloud.bundle_adjustment` | bool | `false` | Run LM bundle adjustment after pointcloud (`ValueError` with `method: sfm`) |
| `pointcloud.loop_closure` | bool | `false` | Run loop closure after pointcloud (`ValueError` with `method: sfm`) |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points |
| `semantics.enabled` | bool | `true` | Extract and lift semantic features |
| `semantics.extractor` | str | `talk2dino` | `talk2dino`, `dinov2`, or `maskclip` |
| `semantics.n_components` | int\|null | `64` | Autoencoder latent dim; null = no compression |
| `mesh.enabled` | bool | `true` | Fuse a TSDF mesh after the pointcloud stage, writing `<backend>/mesh.ply` |
| `mesh.source` | str | `feedforward` | `feedforward` fuses `pointcloud.zarr` depth lifted onto the original frames; `splats` fuses depth and color rendered from the splats stage's `ckpt.pt` (needs the splats stage, which is never auto-run) |
| `mesh.voxel_size` | float | `0.0025` | TSDF voxel edge, world units. Halving this buys finer geometry for roughly 8× the memory. Ignored when `mesh.bands` is set |
| `mesh.sdf_trunc_mult` | float | `4.0` | Truncation band as a multiple of `voxel_size`. This, not `voxel_size`, sets the thin-structure floor: a TSDF cannot resolve anything thinner than `2 × sdf_trunc`, and where a structure's front and back surface both fall inside one band they cancel and it disappears entirely. A bar seen only from the front does not cancel — it is fattened to the floor width instead, which is how a railing survives fusion as a slab and then dies as a floater. At the default the floor is `8 × voxel_size`, four times coarser than the voxel grid itself. `4.0` is Open3D's default for noisy sensor RGBD; rendered splat depth is much cleaner, so `1.5`–`2.0` recovers fence posts and railings at the same voxel size and the same memory. Must be `>= 1.0` — a band narrower than a voxel punctures the surface |
| `mesh.depth_trunc` | float | `1.5` | Ignore depth beyond this, world units. Feedforward depth is not metric, so this is in the reconstruction's own scale, not meters. Ignored when `mesh.bands` is set |
| `mesh.bands` | list\|null | `null` | `null` fuses one volume at `voxel_size`/`depth_trunc`. A list of `{depth_min, depth_trunc, voxel_size}` — ascending and contiguous — fuses the views once per band and merges the results, so the near field can be finer than the far field. TSDF memory goes as surface area ÷ voxel², and far pixels cover most of the area (one pixel at depth 100 covers 55× the world area of one at 13.7), so a near band several times finer costs a fraction of what refining the whole scene would. Bands are merged finest-first and a coarse vertex is dropped only where a finer band already covers it (within 1.5 of the coarse voxel), so a surface two bands both saw keeps its finer copy and one that only a coarse band saw is never lost |
| `mesh.conf_percentile` | float\|null | `20` | Drop depth below this global confidence percentile before fusing (`null` = off). `source: feedforward` only; a reconstruction that carries no confidence (sfm) fuses unmasked and logs that it did |
| `mesh.texture` | bool | `false` | Also decimate, UV-unwrap and project the fused views into `<backend>/texture/` (`albedo.png` beside a UV-carrying `mesh.ply`). Needs a GPU |
| `splats.enabled` | bool | `false` | Train Gaussian splats on the COLMAP poses/points + `images/` (opt-in) |
| `splats.primitive` | str | `3dgs` | `3dgs` (fast kernel, antialiased) or `2dgs` (surface-aligned) |
| `splats.max_steps` | int | `30000` | Training iterations |
| `splats.pose_opt` | bool | `true` | Refine camera poses jointly (`CameraOpt`) |
| `splats.sh_degree` | int | `3` | Max spherical-harmonics degree |
| `splats.sh_degree_interval` | int | `1000` | Steps between SH-degree increments |
| `splats.init_opacity` | float | `0.1` | Initial Gaussian opacity |
| `splats.means_lr` | float | `1.6e-4` | Means learning rate, × scene scale, decays 0.01× over the run |
| `splats.scales_lr` | float | `5.0e-3` | Scales learning rate |
| `splats.quats_lr` | float | `1.0e-3` | Quaternion learning rate |
| `splats.opacities_lr` | float | `5.0e-2` | Opacity learning rate |
| `splats.sh0_lr` | float | `2.5e-3` | DC color learning rate |
| `splats.shN_lr` | float | `1.25e-4` | Higher-order SH learning rate |
| `splats.pose_lr` | float | `1.0e-5` | Pose-opt learning rate, × scene scale |
| `splats.cap_max` | int | `1000000` | `3dgs` only: Gaussian budget for `MCMCStrategy` |
| `splats.grow_grad2d` | float | `2.0e-4` | `2dgs` only: `DefaultStrategy` densification gradient threshold (gsplat non-absgrad default; 8e-4 starved densification) |
| `splats.num_downscales` | int | `2` | Coarse-to-fine (splatfacto): train at `1/2^num_downscales` resolution first, doubling every `resolution_schedule` steps until native. `0` disables |
| `splats.resolution_schedule` | int | `3000` | Steps per coarse-to-fine resolution doubling |
| `splats.normalize_scene` | bool | `false` | Train in splatfacto's normalised frame (centre on camera mean, scale so max \|camera coord\| = 1, `scene_scale` = 1). Outputs (ply, ckpt, zarr c2w, renders) are mapped back to world units |
| `splats.log_every` | int | `500` | Steps between loss log lines |
| `splats.losses.<name>.weight` | float | see `base.yaml` | Weight of an optional loss: `depth`, `normal_consistency`, `distortion`, `opacity_reg`, `scale_reg`. Absent = off. `3dgs` wants `opacity_reg`+`scale_reg` (MCMC); `2dgs` wants `distortion` instead |
| `splats.losses.<name>.start` | int | `0` | Step at which that loss switches on |
| `splats.losses.normal_consistency.depth_ratio` | float | `0.0` | `2dgs` only: RaDe-GS blend weight on the median-depth normal — `(1-r)·d(n, dn_expected) + r·d(n, dn_median)`, `d = 1 - cos`. `0` = expected depth only. Rejected on `3dgs` and outside `[0, 1]` |
| `localization.enabled` | bool | `false` | Build the localization database (opt-in) |
| `localization.matcher` | str | `loma` | vismatch model name (`loma`, `xfeat`, `disk-lightglue`, `aliked-lightglue`, …) |
| `localization.top_k` | int | `8` | Reference frames matched per query |

**Migration (2026-09-06):** the pointcloud cleanup retired eight keys from `base.yaml`, and
they are **not refused** — a published `run_config.yaml` that still sets them deep-merges and
re-runs with the values silently ignored. Retired: `preproc.vda_context_fps`,
`pointcloud.export_max_points`, `pointcloud.clean.outlier_removal`, `pointcloud.clean.voxel_size`,
`pointcloud.clean.confidence_threshold`, `pointcloud.instantsfm.depth_align`,
`pointcloud.instantsfm.features` and `pointcloud.instantsfm.single_camera`.

`pointcloud.clean.outlier_removal` is the only one that can change output. It used to gate the
point3D deletions; the clean step now runs whenever `pointcloud.clean.enabled` is true. The old
default was `true`, so the **default path is unchanged** — but a saved config carrying
`outlier_removal: false` now deletes points and writes a **different `sparse_pc.ply`** on re-run.
Set `pointcloud.clean.enabled: false` to get the old `outlier_removal: false` behaviour.

The rest were already inert, and are listed only so a reader of an old config knows the value is
dead: `export_max_points` defaulted to `null` so the export cap never fired; `voxel_size`'s
downsampled cloud was computed and then discarded; `confidence_threshold` was annotated "NOT YET
READ by the clean step" in `base.yaml` itself; `depth_align` chose between the `scale` and
`affine` fits and the affine model is gone, so per-frame scale alignment is the only path;
`features` had one supported value (`colmap`) and was validated but never dispatched on; and `single_camera` was an `InstantSfMCreator` field that was never set
`False` — the sfm path stages one video, so the single-camera branch is now hard-coded
(`pointcloud/sfm/instantsfm.py`).

### Localization matchers

`localization.matcher` is a vismatch model name, constructed as
`LocalMatcher(name)` (`collab_splats/localization/extractors.py`) — e.g. `loma`,
`xfeat`, `disk-lightglue`, `aliked-lightglue`, `xfeat-steerers-perm`. vismatch
exposes no descriptor-level match API, so queries are matched **pairwise**: the
retrieval stage ranks reference frames and the query is matched against the top
`localization.top_k` of them.

(Historical: until 2026-08-17 this key was `localization.extractor` and also
accepted legacy in-repo registry keys — `disk`, `xfeat`, `xfeat-star`, `loma`,
`loma-g` — which took precedence over colliding vismatch names. The legacy
extractors were retired after the vismatch loma parity gate passed.)

Two blocklists in `collab_splats/localization/extractors.py` gate vismatch names:
`_VISMATCH_LICENSE_BLOCKLIST` (non-commercial licenses) and
`_VISMATCH_DEP_BLOCKLIST` (models whose deps are broken in this environment).
Blocked names raise `ValueError` at construction time with the reason.

With `pointcloud.geometric_verification: true` the same matcher's feature cache
feeds pycolmap, which requires index-stable sparse models (keypoint table indices
that survive re-extraction). Index-incapable matchers (dense/semi-dense vismatch
models, per-pair-refined variants like `xfeat-star`) hard-error at verification
time rather than silently degrading — pick an index-stable matcher or disable
verification.

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

### The `instantsfm` backend (`pointcloud.method: sfm`)

Classical global SfM instead of a feedforward model: system COLMAP SIFT + exhaustive
matching (CPU — upstream forces `CUDA_VISIBLE_DEVICES=""` on the colmap subprocess,
`instantsfm/controllers/feature_handler.py:23`, even though VDA runs on the GPU), then
InstantSfM's global mapper (rotation averaging, global positioning,
global bundle adjustment), with Video-Depth-Anything (VDA) metric depth supplying the
dense per-frame depth every downstream stage expects. Experimental — it warns at run time
and its numbers are not yet measured.

```yaml
pointcloud:
  method: sfm
  backend: instantsfm
  instantsfm:
    retriangulation: false   # GLOMAP-style post-BA refinement: denser tracks, extra runtime
    random_seed: null        # seed RUNTIME_OPTIONS; null = upstream (unseeded) behaviour
```

**Install.** `setup.sh` installs `instantsfm` from a pinned git commit with `--no-deps`
(upstream pins `numpy==1.26.4`, the lock runs numpy 2.x), plus `pyceres==2.3`,
`scikit-sparse==0.4.15` (needs `libsuitesparse-dev`) and `easydict==1.13`; it clones
Video-Depth-Anything into `third_party/Video-Depth-Anything` at commit `4f5ae23` (source
only — no weights). The metric checkpoint is pulled from the Hugging Face hub on first use
(`depth-anything/Metric-Video-Depth-Anything-Large`, ~1.5 GB) and cached under `HF_HOME`
(`/workspace/models` in the image), so a build needs no network for it and a machine that
has none at run time fails on the first sfm run with a `RuntimeError` naming the repo. A
system `colmap` binary must be on `PATH`. Plain `uv sync` prunes the `--no-deps` packages;
re-run the setup.sh block afterwards.

**Licences.** InstantSfM is CC-BY-NC-4.0 (research use only). VDA code is Apache-2.0; the
VDA metric weights are CC-BY-NC-4.0.

**Unsupported with sfm (all `ValueError` at config validation):** `bundle_adjustment: true`
(InstantSfM runs its own global BA; `refine_poses` / `--stages refine` also refuse) and
`loop_closure` (global mapper, not a sequential submap pipeline).

**Output layout** (`<backend>` is `instantsfm/`):

```
<scene>/instantsfm/
  depth_vda/images/npy/<stem>.npy ← VDA metric depth, 518-wide model res (skipped when complete)
  colmap/instantsfm.db         ← SIFT database (local build artifact, NOT pushed)
  colmap/sparse/0/*.bin        ← InstantSfM global-mapper model, image names = frame stems
  pointcloud.zarr              ← depth/images/poses/K at model res; world_points unprojected from VDA depth;
                               ←   no `confidence`, no `mv_*` (absent, never zeros); attrs: method, backend, instantsfm_version
  sparse_pc.ply
```

InstantSfM reads the scene-root `images/` store directly — nothing stages a per-run image
copy any more (`pointcloud/sfm.py`), so the COLMAP image names are the keyframe filenames.
`colmap/instantsfm.db` has its own name so it never collides with the `verify` stage's
`colmap/database.db`; both are anchored in `PUSH_EXCLUDES` and stay local. Downstream
stages — `mesh`, `splats` (depth loss), `semantics`, `localize`, `verify` — consume
`pointcloud.zarr` unchanged; consumers that read `confidence` handle its absence
(mesh fuses unmasked with a log line even when `mesh.conf_percentile` is set, the feature
lift uses uniform weights, splats depth targets are unmasked).

**Known limitation (dashboard).** `_ensure_lift_inputs` in `collab_splats/dashboard/app.py`
treats a `pointcloud.zarr` without `confidence` as a legacy scene and re-pulls its dense
members before a feature lift, so an instantsfm scene always takes that (harmless but
slow) path; once the lift is cached, `_cleanup_lift_inputs` rmtree's the pulled
`depth`/`pixel_indices` from the local copy again — pull-then-delete, once per extractor.
Not changed yet.

---

## Where outputs land

```
<output_path>/
  run_config.yaml              ← full merged config (exact settings used — for reproducibility)
  images/frame_NNNNNN.png      ← canonical decode-once keyframe store (COLMAP-style dir, lossless PNG)
  frames.json                  ← selection records + provenance for those frames
  video_quality_report.json    ← source-video quality measurements (report-only)
  photometric.png, motion.png  ← the report rendered (two files, written with images/)
  semantics/
    <extractor>.zarr           ← 2D patch cache, one per extractor (backend-agnostic)
  <backend>/                   ← e.g. vggt_omega/ (or instantsfm/ for method: sfm)
    pointcloud.zarr            ← depth maps, poses, 3D points (+ confidence when the method produces it)
    sparse_pc.ply
    mesh.ply                   ← (only if mesh.enabled=true)
    texture/                   ← mesh.ply (UV-carrying) + albedo.png (only if mesh.texture=true)
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
| `<backend>/texture/mesh.ply` + `albedo.png` | any pipeline — textured mesh (only if `mesh.texture: true`) |
| `<backend>/semantics/<extractor>_lifted.zarr` | per-point latent codes (`semantics.n_components`-D) |
| `<backend>/semantics/<extractor>_ae.pt` | decoder to full 768-D + `recon_cosine` / `recon_mse` |
| `<backend>/splats/splats.ply` | trained Gaussians (standard 3DGS PLY layout), COLMAP world frame — any splat viewer |
| `<backend>/splats/ckpt.pt` | trainer checkpoint: Gaussian params + pose-opt state, for resuming or re-rendering |
| `<backend>/splats/splats_quality_report.json` | per-view + mean train-view PSNR/SSIM, final Gaussian count, report-only |
| `<backend>/colmap/sparse/0/*.bin` | further processing inside this repo |
| `<backend>/pointcloud.zarr` | further processing inside this repo (depth, poses, confidence when the method produces it) — see below |
| `images/` + `frames.json` | the keyframes the reconstruction was built from; required to localize |
| `run_config.yaml` | exact settings used |

Every geometry-derived artifact sits under `<backend>/`, including `sparse_pc.ply` and the
per-point semantics. One scene may be reconstructed by several backends, so a per-point
latent code is only meaningful next to the point set it indexes. `images/` sits at the
scene root instead: the keyframes are decoded once from the video and shared by every
backend that reconstructs the scene.

`splats/splats.zarr` was retired on 2026-09-06 — `splats/ckpt.pt` is self-contained (model,
cameras, image size, config) and `collab_splats.splats.rendering.load_checkpoint` +
`render_views` reproduce every render. Scenes processed before that date still have the store;
nothing reads it, and it can be deleted.

**Migration (2026-09-05):** `<backend>/transforms.json` is no longer written. Because
`push_outputs` is `rclone copy` with no `--delete` and `verify_push` is `--one-way`, scenes
published before this change keep a stale remote copy frozen at that run's poses. Nothing in
this repo reads it; drop it when convenient with
`rclone delete <remote>:environments-processed --include "**/transforms.json"`.

- `<backend>/pointcloud.zarr` — the unified reconstruction artifact for every
  `pointcloud.method` (feedforward and sfm). Store attrs carry provenance:
  `method`, `backend`, and for instantsfm the installed upstream version
  (`instantsfm_version`). `confidence` and `mv_*` arrays are present only when the
  method produces them (absent, never zeros).

  **Migration (breaking, 2026-08-23):** `feedforward.zarr` was renamed with no
  fallback. Scenes written before the rename need a one-time rename or a re-run:
  - local: `mv <scene>/<backend>/feedforward.zarr <scene>/<backend>/pointcloud.zarr`
    (dashboard-built scenes are flat: `mv <scene>/feedforward.zarr <scene>/pointcloud.zarr`)
  - remote (backend-keyed, as published by the remote driver):
    `rclone moveto <remote>:environments-processed/<scene>/<backend>/feedforward.zarr \
      <remote>:environments-processed/<scene>/<backend>/pointcloud.zarr`
  - remote (flat, the layout the dashboard pulls — `pull_zarr_members` in
    `collab_splats/remote/sources.py`):
    `rclone moveto <remote>:environments-processed/<scene>/feedforward.zarr \
      <remote>:environments-processed/<scene>/pointcloud.zarr`

  Notebooks/tools reading the old name break until the scene is migrated. The tutorial
  notebooks (`docs/source/tutorials/03_splats/train_splats.ipynb`,
  `06_mesh/splats_mesh.ipynb`, `07_localization/localization.ipynb`) read
  `tutorial_config.RECON`, which already points at `pointcloud.zarr`; their stored
  *output* cells still print the old path and stay stale until re-executed.

Not pushed (`PUSH_EXCLUDES` in `collab_splats/remote/sources.py`): `/semantics/**` at the
scene root (raw 2D patch maps, regenerable from frames + extractor — note the leading slash,
which is what keeps `<backend>/semantics/**` in the push), the source video, which the remote
driver fetches into the very scene dir it later pushes and which already lives in
`environments-curated`, and the two COLMAP match databases — `<backend>/colmap/database.db`
(verify) and `<backend>/colmap/instantsfm.db` (instantsfm SIFT), both rebuildable local
artifacts. The scene-root `images/` store **is** pushed — it is the sole persistent keyframe
store, so localization or a correspondence plot against a published scene works directly,
with no re-decode of the curated video.

Comparing semantics across scenes: decode to 768-D first. Two independently-trained
autoencoders do not share a 64-D basis, so raw latent codes are not comparable; the
decoded space is. Each scene's `recon_cosine` is measured on the **training set**, not a
held-out split, so treat it as an upper bound on fidelity rather than a generalisation
estimate.

#### A published scene ships `images/`, but no `transforms.json`

Any loader that resolves `frame["file_path"]` or `data/images/{name}` off disk needs real
image files. A published scene now ships them — `images/frame_NNNNNN.png` at the scene root,
the canonical keyframe store every stage reads. What it does not ship is `transforms.json`:
nothing writes one any more, so a stock file_path-keyed dataparser still needs its poses built
first. Downstream consumers get `sparse_pc.ply` + the mesh + the features + the raw COLMAP
binaries, and read poses via `pycolmap`.

#### Splats train from the published COLMAP + images/

`--stages splats` pulls a processed scene and trains directly on `colmap/` poses + points and
the scene-root `images/` store — no transforms.json round-trip. Every `splats/` artifact is in
the COLMAP world frame; nothing is normalised. `mesh` fuses `pointcloud.zarr` by default;
`mesh.source: splats` fuses the renders instead (alpha as confidence, poses as rendered).

#### The dashboard cannot browse a scene published by the remote driver

The published tree is backend-keyed (`<scene>/<backend>/pointcloud.zarr`) and the dashboard reads
flat (`<scene>/pointcloud.zarr`, `<scene>/semantics/`, `<scene>/mesh.ply`). Pointing the
dashboard at a published scene does not just fail to load — its existence gate never trips, so the
scene re-pulls from GCS on every select and then errors in the op log. This is deliberate: the
dashboard's flat layout is what every dashboard scene already on disk uses, and unifying the read
path would orphan them all. Use the viewer or a notebook for published scenes.

#### Scenes reconstructed before the layout rename need one re-run

The TSDF output is now `<backend>/mesh.ply` (was `mesh/mesh_tsdf.ply`, then `mesh/mesh.ply`),
so older scenes make the mesh readers raise `FileNotFoundError` and the dashboard show no
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
`<scene>/pointcloud.zarr` for the pointcloud, `<scene>/semantics/<extractor>_lifted.zarr` for
semantic features, and `<scene>/mesh.ply` for the mesh. Point it at a scene the dashboard itself
built. It cannot read the backend-keyed tree the remote driver publishes — see "The dashboard
cannot browse a scene published by the remote driver" under Where outputs land.
