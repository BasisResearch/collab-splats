# Reconstruction Pipeline Configs

`base.yaml` is the single template for the reconstruction pipeline. It defines every
stage's defaults; each run sets `input_path` / `output_path` and overrides only what
differs. There is no per-dataset config file — you point the runner at video paths.

---

## Running videos

Use `reconstruct local` (console script; also `python -m collab_splats local`). Point it
at one or more videos or frame directories:

```bash
# Single video
reconstruct local --output-root /workspace/outputs scene.MP4

# Several videos + a frame directory (every image in it, filename order)
reconstruct local --output-root /workspace/outputs \
  /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4 \
  /workspace/fieldwork-data/rats/2024-07-11/frames/

# Turn on extra stages via a shared override YAML
reconstruct local --output-root /workspace/outputs \
  --config my_overrides.yaml /workspace/fieldwork-data/birds/*.MP4   # your own YAML, merged over base.yaml

# One-off dotted override (repeatable; wins over --config)
reconstruct local --output-root /workspace/outputs --set mesh.enabled=true scene.MP4

# Specific steps only
reconstruct local --output-root /workspace/outputs \
  --stages preproc,pointcloud,localize scene.MP4
```

### Steps run per video

1. **keyframe extraction** — sample sharp, well-exposed frames from the video
2. **vggt_omega pointcloud** — feed-forward 3D reconstruction (poses + depth + points)
3. **talk2dino semantics** — 2D features lifted to 3D, autoencoder-compressed
4. **localization database** — per-frame local-feature cache for camera localization

`preproc`, `pointcloud` and `reconstruction_quality_report` always run. `refine`,
`semantics`, `splats`, `mesh` and `localize` run only when enabled in the config
(`pointcloud.bundle_adjustment.enabled` / `semantics.enabled` / `splats.enabled` /
`mesh.enabled` / `localization.enabled`), or when named explicitly via `--stages`.
With loop closure on, `refine` is not planned: BA runs inside the pointcloud stage, one
solve per window.
`semantics` runs after `mesh` and also lifts onto `mesh.ply`'s vertices; after a mesh rebuild
it re-runs from the cached codes (the store's `mesh_sha256` no longer matches).
A store written before `mesh.ply` existed records no hash and counts as done: re-run
semantics with `overwrite` to add its vertex arrays.

### Where outputs land

Each input is written to `<output-root>/<name>/` — a video's stem, or a frame directory's
folder name:

```
<output-root>/C0043/
  images/frame_NNNNNN.png      ← decode-once keyframe store (COLMAP-style dir, lossless PNG)
  video_quality_report.json    ← per-frame photometry + per-pair motion of the source video
  photometric.png              ← the report rendered: blur / laplacian / exposure / clipped fractions
  motion.png                   ←   per-pair translation / parallax (failed pairs = red | at 0) / matches
  semantics/
    <extractor>_codes.zarr     ← fp16 2D codes + autoencoder.pt, one per extractor (backend-agnostic)
  <backend>/                   ← e.g. vggt_omega/ (or instantsfm/ for method: sfm)
    run_config.yaml            ← full merged config of this backend's run (exact settings used)
    pointcloud.zarr            ← depth maps, poses, 3D points (+ confidence when the method produces it)
                               ←   (+ local_features/<extractor>/reconstruction if localize ran)
    sparse_pc.ply
    mesh.ply                   ← (only if mesh.enabled=true)
    texture/                   ← mesh.obj + mesh.mtl + albedo.png (only if mesh.texture=true)
    semantics/
      <extractor>_lifted.zarr  ← lifted 3D features (N_points × latent_dim)
                                 (+ autoencoder.pt inside if semantics.n_components set)
                                 (+ mesh-vertex arrays + mesh_sha256 attr, if mesh.ply existed)
```

---

## The two drivers

`reconstruct` (`collab_splats/__main__.py`) has two subcommands. They share one scene
runner, so the stages, the config merge, and the `run_config.yaml` they leave behind are
identical — they differ only in where scenes come from and what happens afterwards.

| command | input | where a scene lands |
|---|---|---|
| `reconstruct local` | video files and/or frame directories | `<output-root>/<name>/` — video stem or directory name |
| `reconstruct remote` | scenes in the `environments-curated` GCS bucket — named scene ids, or `--all` | `<output-root>/<scene>/`, deleted again after a verified push |

Shared flags: `--output-root` (required), `--config` (override YAML merged over
`base.yaml`), `--base-config` (defaults YAML; default `configs/base.yaml`), `--stages`,
`--overwrite`, `--set key.sub=value` (repeatable, value parsed as YAML, wins over
`--config`). `local` adds `--keep-viewer` (keep the last viser viewer alive for browser
inspection). `remote` adds `--all` and `--keep-local`.

### Remote scenes

Scene ids are the curated directory names. `YYYY_MM_DD-PARENTFOLDER-VIDEONAME` is the common
convention, but any flat path-safe name is a valid scene (e.g. the
`audiomoth_only_deployments-...` deployments); dirs that fail the safety filter are named in
the driver's log.

```bash
# Named scenes
reconstruct remote --output-root /workspace/outputs \
  2026_07_20-birds-C0043 2026_07_21-rats-C0100

# Everything in the bucket
reconstruct remote --output-root /workspace/outputs --all

# Keep the local copy for inspection (skips the delete, not the push)
reconstruct remote --output-root /workspace/outputs --all --keep-local
```

`--all` re-runs every curated scene and overwrites its outputs in `environments-processed`; nothing is skipped.

A standalone `--stages refine` leaves mesh, semantics and the report stale, so re-run them; on remote the old `local_features/` stays in the bucket, because the push is `rclone copy`.

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

`--stages` naming only *leaf* stages — `refine`, `mesh`, `semantics`, `splats`, `localize`,
`reconstruction_quality_report` — pulls the scene back
out of `environments-processed` instead of rebuilding it from its curated video:

```bash
# Re-mesh every processed scene with a new voxel size
reconstruct remote --output-root /workspace/outputs \
  --stages mesh --overwrite --config remesh.yaml --all

# Add semantics to one scene reconstructed without it (no --overwrite: nothing to replace)
reconstruct remote --output-root /workspace/outputs \
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

Four things are errors rather than surprises, and each fails only its own scene:

| situation | outcome |
|---|---|
| scene has no processed outputs | `FileNotFoundError` — run the full pipeline first |
| no `--set pointcloud.backend` and not exactly one `<backend>/run_config.yaml` | `ValueError` listing the recorded backends — pick one |
| named backend has no `<backend>/run_config.yaml` | `FileNotFoundError` — that backend was never run here |
| named leaf stage's output already exists | `ValueError` — pass `--overwrite` to replace it |

That last one applies only to *leaf* stages named on `--stages`. When `--stages` is omitted the
set comes from the config's `enabled` flags, where skipping completed stages is what makes a
re-run resume rather than fail. Naming a leaf stage means asking for it; inheriting it from
config does not. Named *non-leaf* stages also still skip when already done — that is how
`--stages preproc,pointcloud,localize` resumes after a `localize` failure.

Config for a re-run is the pulled `<backend>/run_config.yaml` minus the sections of the stages being
re-run, with `base.yaml` and `--config` supplying fresh parameters for exactly those. Provenance
for every stage that is *not* re-running is preserved verbatim.

Nothing here deletes a remote object. The push is still `rclone copy`, so a re-run overwrites
the artifacts it produced and leaves everything else in place.

- `<backend>/reconstruction_quality_report.json` — reference-free scene error
  report. Columnar tables, `{column: [values]}`, beside a `scene` block (backend,
  n_frames, model_resolution, image_width, zarr). Column meanings:
  `docs/source/api/geometry.rst`.

  | table | grid | row | columns |
  |---|---|---|---|
  | `frames` | mixed | one per frame | frame_idx (null off the `frame_{idx:06d}` contract), covered_fraction (0..1, original), median_abs_rel_depth_error (model), multiview_agreement (model; share of seen pixels where ≥1 other view agrees at rel 0.05; null when unseen), confidence_median (model; backbone-native, not comparable across backbones) |
  | `depth_pairs` | model | one per ordered direction | idx1, idx2, n_pixels, median_depth, median_rel_depth_error (signed s − 1), iqr_rel_depth_error, median_parallax_deg |
  | `depth_residual_histogram` | model | — | counts, bin_edges over r/(1+\|r\|) |
  | `photometric_pairs` | original | i < j, j − i ∈ {1, 2, 5, 10, 20} | idx1, idx2, photometric_ncc (zero-mean NCC), n_pixels |

  Written atomically (`.json.tmp` then rename) by the always-on
  `reconstruction_quality_report` leaf stage; re-runnable with
  `--stages reconstruction_quality_report --overwrite`. A report on disk in the old
  format (no `frames`) raises — delete it and re-run.

  Named for what it scores. The sibling artefact `video_quality_report.json` scores
  the capture — blur, exposure, parallax — before any reconstruction exists; this one
  scores the reconstruction built from it.

  The stage runs no model and no matcher; `photometric_pairs` is null only when `images/`
  is absent.

- `<backend>/reconstruction_quality_ncc.png` — `photometric_pairs` plotted: median and p10
  NCC per frame gap, and NCC along the sequence per gap. Wide gaps expose pose drift that
  neighbor pairs hide. Not written when `photometric_pairs` is null or empty.

  **Report-only: nothing here feeds back into the reconstruction.** No verdict,
  no grade, no cause — raw per-row values only. Each table's grid (`model` or
  `original`) is fixed and listed above; units are
  scale-free or normalized throughout, because 1 recon unit is not 1 meter and
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
COLMAP, `sparse_pc.ply`, and the pose-derived arrays in `pointcloud.zarr`. After
the reproject it re-cleans the cloud (`pointcloud.clean.enabled`) and re-caps it to
`max_points`, so the point count can change: `points`, `colors` and `pixel_indices`
are rewritten. It does NOT invalidate `mesh/`, lifted semantics, or the
localization DB built under the old poses. After `--stages refine`, lifted
semantics MUST be re-run with `overwrite`: their rows index the pre-refine points
and are now misaligned. Re-run `mesh` and `localize` with `overwrite` if
pose-sensitive outputs matter. Provenance for the last
refine run (BA config + LM loss history) is in `<backend>/colmap/refine.json`.

---

## Reproducing an exact run

Every backend dir gets a `run_config.yaml` recording the exact settings used, so one scene
compared across backends keeps one record per backend. Re-run it
by passing it as `--config`; the CLI re-sets `input_path` / `output_path` from the input
and `--output-root`:

```bash
reconstruct local /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4 \
  --output-root /workspace/outputs \
  --config /workspace/outputs/C0043/vggt_omega/run_config.yaml

# Tweak a saved run and send it to a separate root
reconstruct local /workspace/fieldwork-data/birds/2024-02-06/SplatsSD/C0043.MP4 \
  --output-root /workspace/outputs_vggtx \
  --config /workspace/outputs/C0043/vggt_omega/run_config.yaml \
  --set pointcloud.backend=vggtx
```

---

## Why configs live here (not with the data)

This project separates **code + configs** (versioned in git) from **data + outputs**
(too large for git, live outside the repo):

```
/workspace/
  collab-splats/               ← this repo (configs live here)
    configs/base.yaml
    collab_splats/__main__.py  ← the `reconstruct` command
  fieldwork-data/              ← input videos (never in repo, ~GB each)
  outputs/                     ← reconstruction outputs (never in repo, ~10-50 GB per scene)
```

Absolute `/workspace/...` paths are stable across sessions in the fixed container.
Config is a small text file describing *intent* (what to run, how); output data is
large, mutable, and reproducible from `run_config.yaml` — it doesn't belong in git.

---

## Choosing a frame sampler

Preproc runs in two steps: **measure**, then **select**.

1. **Measure.** `load_video_quality` writes `video_quality_report.json` beside
   `images/`, running `compute_video_quality` (one decode pass) to fill it —
   per-frame photometry (blur, exposure, clipping) and per-pair motion (matches,
   translation, parallax). The report is **report-only**: it carries measurements,
   never thresholds and never a usable/unusable verdict. It is reused **by
   existence** — if the file is already there the decode pass is skipped, so delete
   it to re-measure. A report from an older schema (no `frames`) raises rather than
   being silently re-measured.
   `extract_frames` also renders the report to two PNGs beside it
   (`photometric.png`, `motion.png`; plotters in `preproc/viz.py`), each panel with
   a marginal histogram and the kept frames marked by faint green lines. `motion.png`
   is skipped when the report has no pairs. They are written whenever
   frames are extracted and never otherwise — scenes processed before 2026-08-23
   have no PNGs until `--stages preproc --overwrite` re-extracts.
2. **Select.** The sampler reads that report through `filter_frame_quality`,
   which cuts on a robust MAD z-score over `log(laplacian)` (`sharpness_k`, default
   2.0 — relative to the video's own sharpness spread, not an absolute value) plus an
   absolute ceiling on `clipped_low_frac + clipped_high_frac` (`max_clipped_frac`,
   default 0.25). Changing sampling policy never re-decodes the video.

`preproc.n_workers` sets the parallel decode ranges for the step 1 measurement
and for the `fps`/`uniform` selection decode in step 2. It parallelizes decode
only and never changes which frames get selected. The report is byte-identical
at any worker count.

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

**`max_frames` still dominates on long video.** At `fps: 2.0` the default
`max_frames: 300` binds past ~2.5 minutes, and beyond that the spacing is whatever
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
| `preproc.fps` | float | `2.0` | `fps` method only: samples per second |
| `preproc.min_frames` | int\|null | `null` | `fps` method only: floor on the resulting count |
| `preproc.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300 — not a LoGeR limit, see below) |
| `preproc.n_workers` | int | `16` | Decode parallelism: the quality report (step 1) and the `fps`/`uniform` selection decode (step 2) each split into this many frame ranges at once; never changes which frames are selected. `1` = serial. Not auto-derived (`os.cpu_count()` reports host cores in a container). Set to `1` during a GPU eval run. |
| `preproc.undistort` | bool | `false` | After `images/` is written, self-calibrate one shared OPENCV camera from those frames (pycolmap, ≤60 of them) and rewrite `images/` undistorted. COLMAP framing: focal kept, canvas resized to the undistorted corners, so frame dims change. `images/` reuse is by existence — toggling on an existing scene needs `--stages preproc --overwrite`. Localization query images are not undistorted. |
| `preproc.on_empty_slot` | str | `rescue` | `fps` method only: a slot with no eligible frame keeps its sharpest frame (`rescue`) or is skipped (`drop`). A preproc-level key, not under `quality`. |
| `preproc.quality.sharpness_k` | float | `2.0` | Eligibility gate for every sampler: MAD z-score cut on `log(laplacian)`; larger keeps more |
| `preproc.quality.max_clipped_frac` | float | `0.25` | Eligibility gate for every sampler: ceiling on `clipped_low_frac + clipped_high_frac`; larger keeps more |
| `pointcloud.method` | str | `feedforward` | `feedforward` or `sfm` |
| `pointcloud.backend` | str | `vggt_omega` | feedforward: `vggt_omega`, `vggtx`, `mapanything`, or `loger`; sfm: `instantsfm`, `colmap` or `hloc` (dispatched through `SFM_CREATORS` in `collab_splats.pointcloud.sfm`) |
| `pointcloud.<backend>` | dict | `{}` | Per-backend creator kwargs, e.g. `pointcloud.loger.window_size`. Only the block matching `backend` is read. `max_points`, `min_views`, `mv_rel_thresh` and `clean` are rejected here. |
| `pointcloud.instantsfm.random_seed` | int\|null | `null` | Seed InstantSfM's `RUNTIME_OPTIONS` (numpy/random/torch/cuda). Upstream `InitializeRandomPositions` draws unseeded, so two runs of one scene differ. `null` = upstream behavior |
| `pointcloud.instantsfm.pairing` | str | `exhaustive` | Same four values as `colmap.pairing`; global SfM wants every pair, so the default stays exhaustive |
| `pointcloud.instantsfm.overlap` | int | `10` | As `colmap.overlap` |
| `pointcloud.instantsfm.num_retrieved` | int | `20` | As `colmap.num_retrieved` |
| `pointcloud.instantsfm.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.instantsfm.num_threads` | int | `8` | CPU SIFT thread cap; the GPU path ignores it |
| `pointcloud.colmap.pairing` | str | `sequential+retrieval` | Image pairs matched: `sequential`, `retrieval`, `sequential+retrieval` or `exhaustive` (see "The colmap backend") |
| `pointcloud.colmap.overlap` | int | `10` | `sequential*`: pair each frame with the next N |
| `pointcloud.colmap.num_retrieved` | int | `20` | `*retrieval`: vocab-tree neighbors per image (9.5 MB tree, fetched once) |
| `pointcloud.colmap.num_threads` | int | `8` | CPU SIFT + mapper thread cap; colmap's default spawns one per host core and OOMs |
| `pointcloud.colmap.min_registered_frac` | float | `0.5` | In (0, 1]. Below this share of frames registered: `RuntimeError`; above it: subset to the registered frames, with a warning |
| `pointcloud.hloc.pairing` | str | `sequential+retrieval` | Same four values as `colmap.pairing` (see "The hloc backend") |
| `pointcloud.hloc.overlap` | int | `10` | `sequential*`: pair each frame with the next N |
| `pointcloud.hloc.num_retrieved` | int | `20` | `*retrieval`: top-k global-descriptor neighbors per image (clamped to N-1) |
| `pointcloud.hloc.retrieval_conf` | str | `netvlad` | `hloc.extract_features.confs` key for global descriptors |
| `pointcloud.hloc.feature_conf` | str | `superpoint_max` | `hloc.extract_features.confs` key for local features |
| `pointcloud.hloc.matcher_conf` | str | `superpoint+lightglue` | `hloc.match_features.confs` key. Conf keys are checked as non-empty strings only, not against hloc |
| `pointcloud.hloc.num_threads` | int | `8` | Mapper thread cap |
| `pointcloud.hloc.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.bundle_adjustment.enabled` | bool | `false` | Run LM bundle adjustment after pointcloud (`ValueError` with `method: sfm`); with `loop_closure` on it runs inside each LC window instead (first window sets the focal, later windows hold it; no `refine` stage); a bare bool sets this |
| `pointcloud.bundle_adjustment.track_source` | str | `xfeat` | Track source: `xfeat` / `loma` build matcher star tracks over the full-res `images/` frames (`geometry/tracks.py`), also inside each LC window; `vggsfm` predicts tracks on the model grid; anything else is a `ValueError` |
| `pointcloud.bundle_adjustment.track_kwargs` | dict | `{}` | Source-specific settings forwarded to `extract_tracks`; empty takes their defaults. `vggsfm`: `max_query_pts` (4096), `query_frame_num` (8), `fine_tracking` (false; true peaks at 34 GB RSS on 50 frames). `xfeat`/`loma`: `retrieval` (`dino-salad` / `megaloc`), `seed_fraction` (0.34, in (0, 1]), `window` (10), `retrieval_k` (20) and the other `build_tracks` keywords. A key the source does not take is a `TypeError`; bad values are a `ValueError` |
| `pointcloud.bundle_adjustment.vis_thresh` | float | `0.2` | Min track score for an observation (VGGSfM visibility; matcher tracks score 1.0) |
| `pointcloud.bundle_adjustment.max_reproj_error` | float\|null | `4.0` | Pre-solve pixel reprojection gate; `null` skips the filter |
| `pointcloud.bundle_adjustment.min_inliers_per_frame` | int | `64` | Frames below this inlier count sit out the solve |
| `pointcloud.bundle_adjustment.solver` | str | `schur` | `schur` eliminates points (camera-only PCG, less GPU); `lm` solves the joint system. `schur` with `refine_focal` and `shared_camera` is a `ValueError` |
| `pointcloud.bundle_adjustment.dtype` | str | `float64` | Solve precision, `float32` or `float64`; anything else is a `ValueError` |
| `pointcloud.bundle_adjustment.lm_steps` | int | `40` | Max LM steps for a solve without photometric. Ignored with `use_photometric`, which runs up to (3, 2, 1) scale re-samples x 5 IRLS steps, 30 in all |
| `pointcloud.bundle_adjustment.lm_tol` | float | `1.0e-4` | Relative loss drop below which an LM step counts as stalled; `ValueError` below 0 |
| `pointcloud.bundle_adjustment.lm_patience` | int | `2` | Stalled steps in a row that end the solve, or one photometric re-sample; `ValueError` below 1 |
| `pointcloud.bundle_adjustment.increment_size` | int | `0` | `0` = one global solve; `N` = frames added per incremental solve |
| `pointcloud.bundle_adjustment.shared_camera` | bool | `true` | One focal per scene; `false` = one per frame |
| `pointcloud.bundle_adjustment.refine_focal` | bool | `false` | Solve for focal; `false` holds the input mean focal fixed |
| `pointcloud.bundle_adjustment.use_depth` | bool | `true` | Track camera z against feedforward depth |
| `pointcloud.bundle_adjustment.depth_sigma` | float | `0.01` | Relative depth error weighted like 1 px |
| `pointcloud.bundle_adjustment.use_photometric` | bool | `true` | Brightness matching between overlapping frames; `ValueError` with `increment_size > 0` |
| `pointcloud.bundle_adjustment.device` | str\|null | `null` | CUDA device (`cuda`, `cuda:1`); `null` = auto. The pipeline never caches tracks; `tracks_cache_dir` is not a key |
| `pointcloud.loop_closure` | bool | `false` | Run loop closure after pointcloud (`ValueError` with `method: sfm`); with `bundle_adjustment.enabled` BA runs inside each window |
| `pointcloud.loop_closure.retrieval` | str | `dino-salad` | Dict form only: retrieval registry name, `dino-salad` or `megaloc`; `megaloc` needs its own `lc_retrieval_threshold` |
| `pointcloud.min_views` | int | `0` | Feedforward cross-view depth filter: keep a pixel when min(min_views, seen) other views agree. `0` = off; upstream MapAnything uses `1` |
| `pointcloud.mv_rel_thresh` | float | `0.01` | Multiview agreement tolerance, as a fraction of depth |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points, every method. sfm deletes the same points3D from the mapper's COLMAP export, which keeps its tracks and camera model |
| `semantics.enabled` | bool | `true` | Extract and lift semantic features |
| `semantics.extractors` | list[str] | `[talk2dino, ocr_lens]` | Run in order, one model on the GPU at a time, each writing `<extractor>_lifted.zarr`: `talk2dino`, `dinov2`, `maskclip`, `ocr_lens`. All share the settings below; each builds with its own defaults |
| `semantics.n_components` | int\|null | `128` | Autoencoder latent dim; null = no compression |
| `semantics.target_cosine` | float\|null | `null` | null runs every `max_epochs`; else stop once the mean reconstruction cosine reaches it. Early stops cost query fidelity (GH010238: talk2dino top-5% IoU 0.68 at a 0.95 stop vs 0.78 after 30 epochs) |
| `semantics.max_epochs` | int | `30` | Autoencoder epochs; the fit holds the whole feature set on the GPU (ocr_lens 300 frames: 3.1 GB, ~1 min) |
| `mesh.enabled` | bool | `true` | Fuse a TSDF mesh after the pointcloud stage, writing `<backend>/mesh.ply` |
| `mesh.source` | str | `feedforward` | `feedforward` fuses `pointcloud.zarr` depth lifted onto the original frames; `splats` fuses depth and color rendered from the splats stage's `ckpt.pt` (needs the splats stage, which is never auto-run) |
| `mesh.voxel_depth_px` | float | `4.0` | TSDF voxel edge in depth pixels: `voxel = voxel_depth_px × depth / fx` at the `voxel_ref_percentile` depth, `fx` on the depth's own grid (model grid for feedforward). Derived per scene, so it follows the reconstruction's scale; coarsened further when the surface blocks would exceed 8 GB. `4.0` reproduces the hand-tuned `0.0025` of the 294-frame GH010229 run |
| `mesh.voxel_ref_percentile` | float | `50` | Depth percentile the voxel footprint is taken at; lower favors near surfaces with a finer voxel |
| `mesh.sdf_trunc_mult` | float | `4.0` | Truncation band as a multiple of `voxel_size`. This, not `voxel_size`, sets the thin-structure floor: a TSDF cannot resolve anything thinner than `2 × sdf_trunc`, and where a structure's front and back surface both fall inside one band they cancel and it disappears entirely. A bar seen only from the front does not cancel — it is fattened to the floor width instead, which is how a railing survives fusion as a slab and then dies as a floater. At the default the floor is `8 × voxel_size`, four times coarser than the voxel grid itself. `4.0` is Open3D's default for noisy sensor RGBD; rendered splat depth is much cleaner, so `1.5`–`2.0` recovers fence posts and railings at the same voxel size and the same memory. Must be `>= 1.0` — a band narrower than a voxel punctures the surface |
| `mesh.depth_trunc_percentile` | float | `95` | Zero depth beyond this percentile of the scene's depth before the voxel is sized and fused. Required, in (0, 99]; `null` or more is a `ValueError`, because an uncut far tail (one stray 24-unit depth against a 0.34 median) crashes Open3D's `extract_triangle_mesh` |
| `mesh.conf_percentile` | float\|null | `20` | Drop depth below this global confidence percentile before fusing (`null` = off). `source: feedforward` only; a reconstruction that carries no confidence (sfm) fuses unmasked and logs that it did |
| `mesh.mask_sky` | bool | `true` | Zero depth wherever the sky segmenter fires before fusing, so sky never seeds floaters |
| `mesh.max_faces` | int\|null | `1500000` | Face budget for the prepared mesh: past the error-bound decimation, a second QEM pass down to this count so texturing cost stays bounded (`null` = no cap) |
| `mesh.smooth_iterations` | int | `10` | Taubin smoothing passes on the prepared mesh, after decimation and repair, before texturing (`0` = off). Moves vertices only, never removes faces; a final `make_manifold` drops the few faces it folds. `10` measured on GH010229: face-to-face angle median 22° -> 8.5°, vertices move 0.26 mm mean |
| `mesh.texture` | bool | `true` | Also decimate, UV-unwrap and project the fused views into `<backend>/texture/` (`mesh.obj` + `mesh.mtl` + `albedo.png`). Needs a GPU |
| `mesh.use_convex_hull` | bool | `true` | Trim the ragged outer edge and patch the ground out to a rounded convex hull before hole filling (`make_convex_hull`). Ground-dominated outdoor scenes only; a mesh without a dominant ground raises, so set it `false` indoors and for objects. See `docs/mesh.md` |
| `splats.enabled` | bool | `false` | Train Gaussian splats on the `pointcloud.zarr` poses/points + `images/` (opt-in) |
| `splats.primitive` | str | `3dgs` | `3dgs` (fast kernel, antialiased) or `2dgs` (surface-aligned) |
| `splats.representation` | str | `vanilla` | `vanilla` (per-Gaussian) or `scaffold` (anchors + MLP; set `losses.opacity_reg` weight 0) |
| `splats.scaffold` | dict | absent | Scaffold-only overrides, unset by default: `n_offsets` 10, `feat_dim` 32, `voxel_multiplier` 1.0 (× median kNN seed spacing), `update_from` 1500, `update_until` 15000, `refine_every` 100, `grad_threshold` 2.0e-4, `min_opacity` 0.005, `update_init_factor` 16, `success_threshold` 0.8, `appearance_dim` 32 (0 = off), `mlp_bf16` true |
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
| `localization.matcher` | str | `loma` | vismatch model name: `loma` or `xfeat`, the batch-capable models; anything else is a `ValueError` |
| `localization.retrieval` | str | `dino-salad` | Retrieval registry name, `dino-salad` or `megaloc`; stored with the DB, so queries embed with the same model |

**Migration (2026-09-06):** the pointcloud cleanup retired eight keys from `base.yaml`, and
they are **not refused** — a published `run_config.yaml` that still sets them deep-merges and
re-runs with the values silently ignored. Retired: `preproc.vda_context_fps`,
`pointcloud.export_max_points`, `pointcloud.clean.outlier_removal`, `pointcloud.clean.voxel_size`,
`pointcloud.clean.confidence_threshold`, `pointcloud.instantsfm.depth_align`,
`pointcloud.instantsfm.features` and `pointcloud.instantsfm.single_camera`.

`pointcloud.clean.outlier_removal` is the only one that can change output. It used to gate the
point3D deletions; the clean step now runs whenever `pointcloud.clean.enabled` is true, sfm included. The old
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

**Migration (2026-09-24):** `preproc.quality.on_empty_slot` moved to `preproc.on_empty_slot`;
a `run_config.yaml` still setting the old key raises a `TypeError` from `filter_frame_quality`.

**Migration (2026-09-27):** geometric verification was removed — the `verify` stage and
`pointcloud.geometric_verification`. `--stages verify` is an unknown stage (`ValueError`);
the config key is no longer read, so published `run_config.yaml` files that carry it still
load. `colmap/verification.json`, `colmap/verified/` and
`colmap/database.db` are no longer written; old files on disk are ignored.

**Migration (2026-09-26):** `pointcloud.instantsfm` now refuses unknown keys at config load. A
`run_config.yaml` carrying the retired `depth_align` / `features` / `single_camera` raises
`ValueError` naming them; delete them.

**Migration (2026-10-06):** `localization.top_k` retired, not refused; a config that sets it
runs with top_k 8 (`CameraLocalizer` default).

### Localization matchers

`localization.matcher` is a vismatch model name, constructed as
`LocalMatcher(name)` (`collab_splats/localization/extractors.py`) — `loma` or
`xfeat`. There is one match path: reference features are extracted once and cached in
the localization zarr, the query is extracted once, and `LocalMatcher.match` pairs the
query features against each reference's cached features. Retrieval (`localization.retrieval`) picks
the top `top_k` reference frames (`CameraLocalizer` default 8), or the caller passes
`refs=`; reference pixels map back to the world-point grid through `original_coords`.

(Historical: until 2026-08-17 this key was `localization.extractor` and also
accepted legacy in-repo registry keys — `disk`, `xfeat`, `xfeat-star`, `loma`,
`loma-g` — which took precedence over colliding vismatch names. The legacy
extractors were retired after the vismatch loma parity gate passed.)

`LocalMatcher` accepts only batch-capable vismatch models (`xfeat`, `loma`): it matches
cached features, so a model without vismatch `supports_batches` raises `ValueError` at
construction time.

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
ceiling is `PointcloudResult`, which holds dense per-frame images, world points, depth,
and confidence — 8.13 MB/frame at LoGeR's default `pixel_limit` of 255,000 — against a
46.6 GB container cap. The cap binds every backend; the per-frame figure is LoGeR's own,
since each backend resolves frames differently. LoGeR is merely the first backend able to
feed the buffer enough frames for the cap to matter.

### The `instantsfm` backend (`pointcloud.method: sfm`)

Classical global SfM instead of a feedforward model: pycolmap SIFT + exhaustive matching
(upstream's own step, which forces the colmap CLI onto the CPU at
`instantsfm/controllers/feature_handler.py:23`, is bypassed), then InstantSfM's global
mapper (rotation averaging, global positioning, global bundle adjustment), with
Video-Depth-Anything (VDA) metric depth supplying the dense per-frame depth every
downstream stage expects. Experimental — it warns at run time and its numbers are not
yet measured.

```yaml
pointcloud:
  method: sfm
  backend: instantsfm
  instantsfm:
    retriangulation: false   # GLOMAP-style post-BA refinement: denser tracks, extra runtime
    random_seed: null        # seed RUNTIME_OPTIONS; null = upstream (unseeded) behavior
    min_registered_frac: 0.5 # fail below this share registered; above it, subset
```

**Install.** `setup.sh` installs `instantsfm` from a pinned git commit with `--no-deps`
(upstream pins `numpy==1.26.4`, the lock runs numpy 2.x), plus `pyceres==2.3`,
`scikit-sparse==0.4.15` (needs `libsuitesparse-dev`) and `easydict==1.13`; it clones
Video-Depth-Anything into `third_party/Video-Depth-Anything` at commit `4f5ae23` (source
only — no weights). The metric checkpoint is pulled from the Hugging Face hub on first use
(`depth-anything/Metric-Video-Depth-Anything-Large`, ~1.5 GB) and cached under `HF_HOME`
(`/workspace/models` in the image), so a build needs no network for it and a machine that
has none at run time fails on the first sfm run with a `RuntimeError` naming the repo. SIFT
runs through pycolmap: GPU when the wheel is the CUDA build (`pycolmap-cuda12`) and torch sees a
GPU, otherwise CPU capped at 8 threads. Plain `uv sync` prunes the `--no-deps` packages;
re-run the setup.sh block afterwards.

**Licenses.** InstantSfM is CC-BY-NC-4.0 (research use only). VDA code is Apache-2.0; the
VDA metric weights are CC-BY-NC-4.0.

**Unsupported with any sfm backend (all `ValueError` at config validation):**
`bundle_adjustment.enabled: true` (the sfm mapper runs its own BA; the `refine` stage also
refuses) and `loop_closure` (not a sequential submap pipeline).

**Output layout** (`<backend>` is `instantsfm/`):

```
<scene>/instantsfm/
  depth_vda/images/npy/<stem>.npy ← VDA metric depth, 518-wide model res (skipped when complete)
  colmap/instantsfm.db         ← SIFT database (local build artifact, NOT pushed)
  colmap/sparse/0/*.bin        ← InstantSfM global-mapper model, image names = frame stems
  pointcloud.zarr              ← depth/images/poses/K at model res; world_points unprojected from VDA depth;
                               ←   no `confidence` (absent, never zeros); attrs: method, backend, registered_frames, total_frames, depth alignment
  sparse_pc.ply
```

InstantSfM reads the scene-root `images/` store directly — nothing stages a per-run image
copy any more (`pointcloud/sfm/instantsfm.py`), so the COLMAP image names are the keyframe filenames.
`colmap/instantsfm.db` has its own name so it never collides with a `colmap/database.db`
left by the removed geometric verification; both are anchored in `PUSH_EXCLUDES` and stay local. Downstream
stages — `mesh`, `splats` (depth loss), `semantics`, `localize`, `reconstruction_quality_report` — consume
`pointcloud.zarr` unchanged; consumers that read `confidence` handle its absence
(mesh fuses unmasked with a log line even when `mesh.conf_percentile` is set, the feature
lift uses uniform weights, splats depth targets are unmasked).

**Known limitation (dashboard).** `_ensure_lift_inputs` in `collab_splats/dashboard/app.py`
treats a `pointcloud.zarr` without `confidence` as a legacy scene and re-pulls its dense
members before a feature lift, so an instantsfm scene always takes that (harmless but
slow) path; once the lift is cached, `_cleanup_lift_inputs` rmtree's the pulled
`depth`/`pixel_indices` from the local copy again — pull-then-delete, once per extractor.
Not changed yet.

### The `colmap` backend (`pointcloud.method: sfm`)

Classical incremental SfM (decision
[018](../docs/superpowers/decisions/018-sfm-backends.md)):

- features + matches: pycolmap, on the same GPU/CPU rule as instantsfm (`num_threads` caps the CPU path)
- mapping: `pycolmap.incremental_mapping` (the 4.x wheel); the largest model is kept, with a
  warning when the scene splits
- one shared SIMPLE_RADIAL camera, refined by the mapper — same freedom as instantsfm
- VDA metric depth supplies dense depth, as for instantsfm

```yaml
pointcloud:
  method: sfm
  backend: colmap
  colmap:
    pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
    overlap: 10
    num_retrieved: 20
    num_threads: 8
    min_registered_frac: 0.5
```

**Install.** Nothing beyond the instantsfm prerequisites: the VDA clone. Any `*retrieval`
pairing (the default included) fetches COLMAP's FAISS-format
`vocab_tree_faiss_flickr100K_words32K.bin` (9.5 MB, sha256-pinned) once into
`~/.cache/collab_splats/`; no network at that point is a `RuntimeError` naming URL and path.

**Pairing.**

| `pairing` | pycolmap matcher |
|---|---|
| `sequential` | `match_sequential`, `overlap` N, `quadratic_overlap=False` (i with i+1..i+N) |
| `retrieval` | `match_vocabtree`, `num_images` = `num_retrieved` |
| `sequential+retrieval` | `match_sequential` as above + `loop_detection=True` (`loop_detection_num_images` = `num_retrieved`) |
| `exhaustive` | `match_exhaustive` |

- colmap's loop detection fires every `loop_detection_period` (10) frames, not per frame, so
  its `sequential+retrieval` is not the same pair set as hloc's

**Caching.** `colmap/colmap.db` is reused only while its image set AND its matching params
(`pairing`, plus `overlap` for sequential modes and `num_retrieved` for retrieval modes) match;
the params live in a `collab_params` table inside the DB, written after a successful build.
A knob the pairing ignores is not recorded, so changing it keeps the DB.

**Output layout** (`<backend>` is `colmap/`):

```
<scene>/colmap/
  depth_vda/images/npy/<stem>.npy ← VDA metric depth (as instantsfm)
  colmap/colmap.db             ← SIFT database (local build artifact, NOT pushed)
  colmap/sparse/0/*.bin        ← largest incremental model, image names = frame stems
  pointcloud.zarr              ← attrs: method, backend, registered_frames, total_frames, depth alignment
  sparse_pc.ply
```

**Registered subset.** Every sfm backend (instantsfm, colmap, hloc) may leave frames unregistered:

- below `min_registered_frac` of the keyframes registered: `RuntimeError` with N/M
- above it: depth, names and keyframes are filtered to the registered stems, with a warning
- `registered_frames` / `total_frames` in the zarr attrs record the split
- `images/` still holds every keyframe; downstream stages (semantics, mesh, localize,
  splats, reconstruction_quality_report) read only the frames `pointcloud.zarr` names in its
  `image_paths` attr, joined on frame index; the unregistered ones are never read
- the 2D semantics cache `semantics/<extractor>_codes.zarr` stays scene-level (every `images/`
  frame); the lift picks the pointcloud's rows out of it

### The `hloc` backend (`pointcloud.method: sfm`)

Learned-feature incremental SfM through hloc (`cvg/Hierarchical-Localization` @ `c13273b`):
local features + matches from hloc, mapping via `hloc.reconstruction.main` (pycolmap), same
single SIMPLE_RADIAL camera and VDA depth.

```yaml
pointcloud:
  method: sfm
  backend: hloc
  hloc:
    pairing: sequential+retrieval
    overlap: 10
    num_retrieved: 20
    retrieval_conf: netvlad
    feature_conf: superpoint_max
    matcher_conf: superpoint+lightglue
    num_threads: 8
    min_registered_frac: 0.5
```

**Install.** hloc is the optional `hloc` extra, an editable uv path source on
`third_party/hloc`:

- `bash setup/hloc.sh` clones it (`--recursive`) and re-pins to `c13273b`; `setup.sh` calls it
  before the sync, and `setup/hloc.sh --prefetch` caches the netvlad / SuperPoint / LightGlue
  weights
- the clone must exist before `uv lock` / `uv sync` resolve
- without it, `HlocCreator.reconstruct` raises `ImportError` naming `setup/hloc.sh`

**Licenses.** SuperPoint/SuperGlue weights are Magic Leap **non-commercial**; LightGlue is
Apache-2.0; netvlad weights come from the original authors (research use).

**Pairing.**

| `pairing` | hloc |
|---|---|
| `sequential` | in-repo pairs: frame i with i+1..i+N (`overlap`) |
| `retrieval` | `pairs_from_retrieval` on `retrieval_conf` descriptors, top `num_retrieved` (clamped to N-1) |
| `sequential+retrieval` | union of both pair sets, deduplicated |
| `exhaustive` | `pairs_from_exhaustive` |

**Caching.** hloc's h5 features and matches are reused by its own `overwrite=False` skip;
the mapper database is rebuilt every run.

**Output layout** (`<backend>` is `hloc/`):

```
<scene>/hloc/
  depth_vda/images/npy/<stem>.npy ← VDA metric depth (as instantsfm)
  colmap/hloc/                 ← h5 features/matches, pairs-*.txt, sfm/ mapper dir (NOT pushed)
  colmap/sparse/0/*.bin        ← hloc's largest model, image names = frame stems
  pointcloud.zarr              ← attrs: method, backend, registered_frames, total_frames, depth alignment
  sparse_pc.ply
```

- registered-subset behavior and `min_registered_frac`: as the colmap backend

---

## Where outputs land

```
<output_path>/
  images/frame_NNNNNN.png      ← canonical decode-once keyframe store (COLMAP-style dir, lossless PNG)
  video_quality_report.json    ← source-video quality measurements (report-only)
  photometric.png, motion.png  ← the report rendered (two files, written with images/)
  semantics/
    <extractor>_codes.zarr     ← fp16 2D codes + autoencoder.pt, one per extractor (backend-agnostic)
  <backend>/                   ← e.g. vggt_omega/ (or instantsfm/ for method: sfm)
    run_config.yaml            ← full merged config of this backend's run (exact settings used)
    pointcloud.zarr            ← depth maps, poses, 3D points (+ confidence when the method produces it)
    sparse_pc.ply
    mesh.ply                   ← (only if mesh.enabled=true)
    texture/                   ← mesh.obj + mesh.mtl + albedo.png (only if mesh.texture=true)
    semantics/
      <extractor>_lifted.zarr  ← lifted 3D features (N_points × n_components)
                                 (+ autoencoder.pt inside, needed to decode them, if n_components set)
                                 (+ mesh-vertex arrays + mesh_sha256 attr, if mesh.ply existed)
```

The 2D patch cache sits at the scene root because it depends only on the frames; the lift is
what depends on the backend, so `<extractor>_lifted.zarr` sits under `<backend>/`. Naming both
halves after the extractor lets two extractors coexist in the same scene instead of
overwriting each other.

### Processed scene layout

A processed scene (`environments-processed/<scene>/`) carries:

| path | consumer |
|---|---|
| `<backend>/sparse_pc.ply` | any pipeline — binary little-endian, float32 xyz + uchar rgb |
| `<backend>/mesh.ply` | any pipeline |
| `<backend>/texture/mesh.obj` + `mesh.mtl` + `albedo.png` | any pipeline — textured mesh (only if `mesh.texture: true`) |
| `<backend>/semantics/<extractor>_lifted.zarr` | per-point latent codes (`semantics.n_components`-D) |
| `<backend>/semantics/<extractor>_lifted.zarr` vertex arrays | per-mesh-vertex: ocr_lens `vertex_word_ids` + `vertex_word_probs` (top-64), maskclip/talk2dino `vertex_features` codes; read by `python -m collab_splats.viewer` |
| `<backend>/semantics/<extractor>_lifted.zarr/autoencoder.pt` | decoder to full 768-D + `recon_cosine` / `recon_mse` |
| `<backend>/splats/splats.ply` | trained Gaussians (standard 3DGS PLY layout), COLMAP world frame — any splat viewer |
| `<backend>/splats/ckpt.pt` | trainer checkpoint: Gaussian params + pose-opt state, for resuming or re-rendering |
| `<backend>/splats/splats_quality_report.json` | per-view + mean train-view PSNR/SSIM, final Gaussian count, report-only |
| `<backend>/colmap/sparse/0/*.bin` | further processing inside this repo |
| `<backend>/pointcloud.zarr` | further processing inside this repo (depth, poses, confidence when the method produces it) — see below |
| `images/` | the keyframes the reconstruction was built from; required to localize |
| `<backend>/run_config.yaml` | exact settings of that backend's run |

Every geometry-derived artifact sits under `<backend>/`, including `sparse_pc.ply` and the
per-point semantics. One scene may be reconstructed by several backends, so a per-point
latent code is only meaningful next to the point set it indexes. `images/` sits at the
scene root instead: the keyframes are decoded once from the video and shared by every
backend that reconstructs the scene.

`splats/splats.zarr` was retired on 2026-09-06 — `splats/ckpt.pt` is self-contained (model,
cameras, image size, config) and `collab_splats.splats.load_checkpoint` +
`render_views` reproduce every render. Scenes processed before that date still have the store;
nothing reads it, and it can be deleted.

**Migration (2026-09-05):** `<backend>/transforms.json` is no longer written. Because
`push_outputs` is `rclone copy` with no `--delete` and `verify_push` is `--one-way`, scenes
published before this change keep a stale remote copy frozen at that run's poses. Nothing in
this repo reads it; drop it when convenient with
`rclone delete <remote>:environments-processed --include "**/transforms.json"`.

- `<backend>/pointcloud.zarr` — the unified reconstruction artifact for every
  `pointcloud.method` (feedforward and sfm). Store attrs: `method`, `backend`, and for sfm
  the depth-alignment stats plus `registered_frames` / `total_frames`.
  No package versions are recorded (removed 2026-09-27: written, never read).

  `confidence` is present only when the method produces it (absent, never zeros).

  **Migration (breaking, 2026-08-23):** `feedforward.zarr` was renamed with no
  fallback. Scenes written before the rename need a one-time rename or a re-run:
  - local: `mv <scene>/<backend>/feedforward.zarr <scene>/<backend>/pointcloud.zarr`
    (dashboard-built scenes are flat: `mv <scene>/feedforward.zarr <scene>/pointcloud.zarr`)
  - remote (backend-keyed, as published by the remote driver):
    `rclone moveto <remote>:environments-processed/<scene>/<backend>/feedforward.zarr \
      <remote>:environments-processed/<scene>/<backend>/pointcloud.zarr`
  - remote (flat, the layout the dashboard pulls):
    `rclone moveto <remote>:environments-processed/<scene>/feedforward.zarr \
      <remote>:environments-processed/<scene>/pointcloud.zarr`

  Notebooks/tools reading the old name break until the scene is migrated. The tutorial
  notebooks (`docs/source/tutorials/03_splats/train_splats.ipynb`,
  `06_mesh/splats_mesh.ipynb`, `07_localization/localization.ipynb`) read
  `tutorial_config.RECON`, which already points at `pointcloud.zarr`; their stored
  *output* cells still print the old path and stay stale until re-executed.

Not pushed (`PUSH_EXCLUDES` in `collab_splats/remote.py`): `/semantics/*_features.zarr/**` at the
scene root (temporary full-width features, deleted once the codes are written; the codes store and `<backend>/semantics/**` are pushed), the source video, which the remote
driver fetches into the very scene dir it later pushes and which already lives in
`environments-curated`, and the COLMAP match databases — `<backend>/colmap/instantsfm.db` (instantsfm SIFT),
`<backend>/colmap/colmap.db` (colmap SIFT) and
`<backend>/colmap/hloc/` (hloc features, matches, pairs and mapper DB), all rebuildable local
artifacts; the
`colmap/sparse/0` model each sfm backend writes is pushed. The scene-root `images/` store **is** pushed — it is the sole persistent keyframe
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

#### Splats train from the published pointcloud.zarr + images/

`--stages splats` pulls a processed scene and trains directly on `pointcloud.zarr` poses + points
(full-res `intrinsics`) and the scene-root `images/` store — no transforms.json round-trip. Every `splats/` artifact is in
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
mesh. Semantics moved the same way: the 2D cache is `<scene>/semantics/<extractor>_codes.zarr` (was
`features/<extractor>/<extractor>.zarr`) and the lifted pair is
`<backend>/semantics/<extractor>_lifted.zarr` with `autoencoder.pt` inside (was
`<backend>/semantics/<extractor>/features.zarr` + `autoencoder.pt`). There are deliberately no
legacy fallbacks — they would restore exactly the two-name ambiguity the rename removed. Such
scenes need `--stages mesh,semantics --overwrite` run once; the 2D
cache re-extracts, which is the expensive half.

---

## Dashboard

The dashboard browses **dashboard-produced scenes**, whose tree is flat: it reads
`<scene>/pointcloud.zarr` for the pointcloud, `<scene>/semantics/<extractor>_lifted.zarr` for
semantic features, and `<scene>/mesh.ply` for the mesh. Point it at a scene the dashboard itself
built. It cannot read the backend-keyed tree the remote driver publishes — see "The dashboard
cannot browse a scene published by the remote driver" under Where outputs land.
