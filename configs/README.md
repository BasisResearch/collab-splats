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
3. **mesh** — TSDF fusion of the pointcloud's depth into a textured mesh
4. **talk2dino semantics** — 2D features lifted to 3D (and onto the mesh vertices), autoencoder-compressed
5. **localization database** — per-frame local-feature cache for camera localization

`preproc`, `pointcloud` and `reconstruction_quality_report` always run. `refine`,
`semantics`, `splats`, `mesh` and `localize` run only when enabled in the config
(`pointcloud.bundle_adjustment.enabled` / `semantics.enabled` / `splats.enabled` /
`mesh.enabled` / `localization.enabled`), or when named explicitly via `--stages`.
With loop closure on, `bundle_adjustment.strategy` picks the solve: `global` (default) plans
`refine` over every frame after LC, `window` solves inside each LC window in the pointcloud
stage, `window+global` does both.
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

The 2D patch cache sits at the scene root because it depends only on the frames; the lift is
what depends on the backend, so `<extractor>_lifted.zarr` sits under `<backend>/`. Naming both
halves after the extractor lets two extractors coexist in the same scene instead of
overwriting each other.

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
| `preproc.max_frames` | int\|null | `300` | Frame budget: the COUNT for `uniform`, a ceiling for `fps`/`optical_flow` (vggt_omega OOMs above ~300 — not a LoGeR limit, see [docs/pointcloud.md](../docs/pointcloud.md#backends)) |
| `preproc.n_workers` | int | `16` | Decode parallelism: the quality report (step 1) and the `fps`/`uniform` selection decode (step 2) each split into this many frame ranges at once; never changes which frames are selected. `1` = serial. Not auto-derived (`os.cpu_count()` reports host cores in a container). Set to `1` during a GPU eval run. |
| `preproc.undistort` | bool | `false` | After `images/` is written, self-calibrate one shared OPENCV camera from those frames (pycolmap, ≤60 of them) and rewrite `images/` undistorted. COLMAP framing: focal kept, canvas resized to the undistorted corners, so frame dims change. `images/` reuse is by existence — toggling on an existing scene needs `--stages preproc --overwrite`. Localization query images are not undistorted. |
| `preproc.on_empty_slot` | str | `rescue` | `fps` method only: a slot with no eligible frame keeps its sharpest frame (`rescue`) or is skipped (`drop`). A preproc-level key, not under `quality`. |
| `preproc.quality.sharpness_k` | float | `2.0` | Eligibility gate for every sampler: MAD z-score cut on `log(laplacian)`; larger keeps more |
| `preproc.quality.max_clipped_frac` | float | `0.25` | Eligibility gate for every sampler: ceiling on `clipped_low_frac + clipped_high_frac`; larger keeps more |
| `pointcloud.method` | str | `feedforward` | `feedforward` or `sfm` |
| `pointcloud.backend` | str | `vggt_omega` | feedforward: `vggt_omega`, `vggtx`, `mapanything`, or `loger`; sfm: `instantsfm`, `colmap` or `hloc` (dispatched through `SFM_CREATORS` in `collab_splats.pointcloud.sfm`) |
| `pointcloud.<backend>` | dict | `{}` | Per-backend creator kwargs, e.g. `pointcloud.loger.window_size`. Only the block matching `backend` is read. `max_points`, `min_views`, `mv_rel_thresh` and `clean` collide with creator kwargs the stage already passes, so setting them here fails with a `TypeError` when the pointcloud stage builds the creator. |
| `pointcloud.instantsfm.random_seed` | int\|null | `null` | Seed InstantSfM's `RUNTIME_OPTIONS` (numpy/random/torch/cuda). Upstream `InitializeRandomPositions` draws unseeded, so two runs of one scene differ. `null` = upstream behavior |
| `pointcloud.instantsfm.pairing` | str | `exhaustive` | Same four values as `colmap.pairing`; global SfM wants every pair, so the default stays exhaustive |
| `pointcloud.instantsfm.overlap` | int | `10` | As `colmap.overlap` |
| `pointcloud.instantsfm.num_retrieved` | int | `20` | As `colmap.num_retrieved` |
| `pointcloud.instantsfm.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.instantsfm.num_threads` | int | `8` | CPU SIFT thread cap; the GPU path ignores it |
| `pointcloud.colmap.pairing` | str | `sequential+retrieval` | Image pairs matched: `sequential`, `retrieval`, `sequential+retrieval` or `exhaustive` (see [docs/pointcloud.md](../docs/pointcloud.md#backends), colmap backend) |
| `pointcloud.colmap.overlap` | int | `10` | `sequential*`: pair each frame with the next N |
| `pointcloud.colmap.num_retrieved` | int | `20` | `*retrieval`: vocab-tree neighbors per image (9.5 MB tree, fetched once) |
| `pointcloud.colmap.num_threads` | int | `8` | CPU SIFT + mapper thread cap; colmap's default spawns one per host core and OOMs |
| `pointcloud.colmap.min_registered_frac` | float | `0.5` | In (0, 1]. Below this share of frames registered: `RuntimeError`; above it: subset to the registered frames, with a warning |
| `pointcloud.hloc.pairing` | str | `sequential+retrieval` | Same four values as `colmap.pairing` (see [docs/pointcloud.md](../docs/pointcloud.md#backends), hloc backend) |
| `pointcloud.hloc.overlap` | int | `10` | `sequential*`: pair each frame with the next N |
| `pointcloud.hloc.num_retrieved` | int | `20` | `*retrieval`: top-k global-descriptor neighbors per image (clamped to N-1) |
| `pointcloud.hloc.retrieval_conf` | str | `netvlad` | `hloc.extract_features.confs` key for global descriptors |
| `pointcloud.hloc.feature_conf` | str | `superpoint_max` | `hloc.extract_features.confs` key for local features |
| `pointcloud.hloc.matcher_conf` | str | `superpoint+lightglue` | `hloc.match_features.confs` key. Conf keys are checked as non-empty strings only, not against hloc |
| `pointcloud.hloc.num_threads` | int | `8` | Mapper thread cap |
| `pointcloud.hloc.min_registered_frac` | float | `0.5` | As `colmap.min_registered_frac` |
| `pointcloud.bundle_adjustment.enabled` | bool | `false` | Run LM bundle adjustment after pointcloud (`ValueError` with `method: sfm`); with `loop_closure` on, `strategy` picks where it runs; a bare bool sets this |
| `pointcloud.bundle_adjustment.strategy` | str | `global` | Read only with `loop_closure` on: `global` = one `refine` solve over every frame after LC; `window` = a solve inside each LC window (first window sets the focal, later windows hold it; no `refine` stage); `window+global` = both; anything else is a `ValueError` |
| `pointcloud.bundle_adjustment.track_source` | str | `xfeat` | Track source: `xfeat` / `loma` build matcher star tracks over the full-res `images/` frames (`geometry/tracks.py`), also inside each LC window; `vggsfm` predicts tracks on the model grid; anything else is a `ValueError` |
| `pointcloud.bundle_adjustment.track_kwargs` | dict | `{}` | Source-specific settings forwarded to `extract_tracks`; empty takes their defaults. `vggsfm`: `max_query_pts` (4096), `query_frame_num` (8), `fine_tracking` (false; true peaks at 34 GB RSS on 50 frames). `xfeat`/`loma`: `retrieval` (`dino-salad` / `megaloc`), `pairing` (`exhaustive` / `sequential+retrieval`), `seed_fraction` (0.34, in (0, 1]), `window` (10) and `retrieval_k` (20; both sequential+retrieval only) and the other `build_tracks` keywords. A key the source does not take is a `TypeError`; bad values are a `ValueError` |
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
| `pointcloud.loop_closure` | bool | `false` | Run loop closure after pointcloud (`ValueError` with `method: sfm`); with `bundle_adjustment.enabled`, `bundle_adjustment.strategy` picks per-window, global or both |
| `pointcloud.loop_closure.retrieval` | str | `dino-salad` | Dict form only: retrieval registry name, `dino-salad` or `megaloc`; `megaloc` needs its own `lc_retrieval_threshold` |
| `pointcloud.loop_closure.submap_size` | int | `20` | Dict form only: window stride; each submap holds this plus `submap_overlap` frames. Fewer frames than this falls back to the creator's own forward pass |
| `pointcloud.loop_closure.submap_overlap` | int | `1` | Dict form only: frames shared by consecutive windows (1 = VGGT-SLAM); the earlier submap owns them |
| `pointcloud.loop_closure.lc_retrieval_threshold` | float | `0.95` | Dict form only: max L2 between unit descriptors, in [0, 2], tuned for dino-salad; `<= 0` skips retrieval |
| `pointcloud.loop_closure.max_loops_per_submap` | int | `5` | Dict form only: loop candidates kept per query submap, before NMS |
| `pointcloud.loop_closure.verify_match_ratio` | float\|null | `null` | Dict form only: loop verification threshold; `null` takes the creator's `default_verify_match_ratio` |
| `pointcloud.loop_closure.nms_frame_distance` | int | `25` | Dict form only: suppression radius in frames between candidates on the same detected submap |
| `pointcloud.loop_closure.min_submap_gap` | int | `1` | Dict form only: newest submaps left out of the retrieval search |
| `pointcloud.loop_closure.conf_percentile` | float | `25.0` | Dict form only: `depth_conf` percentile gating scale points |
| `pointcloud.loop_closure.loop_edge_timing` | str | `deferred` | Dict form only: `deferred` adds all loop edges after the window loop; `live` adds each as found |
| `pointcloud.loop_closure.viz_max_points` | int | `50000` | Dict form only: viewer per-submap point cap |
| `pointcloud.max_points` | int | `500000` | Final point count cap, drawn after the outlier clean, every method. Lower it on small-GPU/RAM machines |
| `pointcloud.min_views` | int | `0` | Feedforward cross-view depth filter: keep a pixel when min(min_views, seen) other views agree. `0` = off; upstream MapAnything uses `1` |
| `pointcloud.mv_rel_thresh` | float | `0.01` | Multiview agreement tolerance, as a fraction of depth |
| `pointcloud.clean.enabled` | bool | `true` | Remove outlier points, every method. sfm deletes the same points3D from the mapper's COLMAP export, which keeps its tracks and camera model |
| `pointcloud.viz.enabled` | bool | `false` | Instantiate the viser `Viewer` during loop closure |
| `pointcloud.viz.port` | int | `8080` | viser server port |
| `semantics.enabled` | bool | `true` | Extract and lift semantic features |
| `semantics.extractors` | list[str] | `[talk2dino, ocr_lens]` | Run in order, one model on the GPU at a time, each writing `<extractor>_lifted.zarr`: `talk2dino`, `dinov2`, `maskclip`, `ocr_lens`. All share the settings below; each builds with its own defaults |
| `semantics.n_components` | int\|null | `128` | Autoencoder latent dim; null = no compression |
| `semantics.target_cosine` | float\|null | `null` | null runs every `max_epochs`; else stop once the mean reconstruction cosine reaches it. Early stops cost query fidelity (GH010238: talk2dino top-5% IoU 0.68 at a 0.95 stop vs 0.78 after 30 epochs) |
| `semantics.max_epochs` | int | `30` | Autoencoder epochs; the fit holds the whole feature set on the GPU (ocr_lens 300 frames: 3.1 GB, ~1 min) |
| `mesh.enabled` | bool | `true` | Fuse a TSDF mesh after the pointcloud stage, writing `<backend>/mesh.ply` |
| `mesh.source` | str | `feedforward` | `feedforward` fuses `pointcloud.zarr` depth lifted onto the original frames; `splats` fuses depth and color rendered from the splats stage's `ckpt.pt` (needs the splats stage, which is never auto-run) |
| `mesh.voxel_depth_px` | float | `4.0` | TSDF voxel edge in depth pixels: `voxel = voxel_depth_px × depth / fx` at the `voxel_ref_percentile` depth, `fx` on the depth's own grid (model grid for feedforward). Derived per scene, so it follows the reconstruction's scale; coarsened further when the surface blocks would exceed 8 GB. `4.0` reproduces the hand-tuned `0.0025` of the 294-frame GH010229 run |
| `mesh.voxel_ref_percentile` | float | `50` | Depth percentile the voxel footprint is taken at; lower favors near surfaces with a finer voxel |
| `mesh.sdf_trunc_mult` | float | `6.0` | Truncation band as a multiple of `voxel_size`. This, not `voxel_size`, sets the thin-structure floor: a TSDF cannot resolve anything thinner than `2 × sdf_trunc`, and where a structure's front and back surface both fall inside one band they cancel and it disappears entirely. A bar seen only from the front does not cancel — it is fattened to the floor width instead, which is how a railing survives fusion as a slab and then dies as a floater. At `6.0` the floor is `12 × voxel_size`. `6.0` beats Open3D's noisy-sensor `4.0` on GH010229 (300 frames): each voxel averages more views, so cross-view depth disagreement smooths out — smoothed face-to-neighbor angle 2.5° -> 2.2°, no fusion-time cost; `8.0` merged nearby surfaces into fins and back faces. Rendered splat depth is much cleaner, so `1.5`–`2.0` recovers fence posts and railings at the same voxel size and the same memory. Must be `>= 1.0` — a band narrower than a voxel punctures the surface |
| `mesh.depth_trunc_percentile` | float | `99` | Zero depth beyond this percentile of the scene's depth before the voxel is sized and fused. Required, in (0, 99]; `null` or more is a `ValueError`, because an uncut far tail (one stray 24-unit depth against a 0.34 median) crashes Open3D's `extract_triangle_mesh` |
| `mesh.conf_percentile` | float\|null | `10` | Drop depth below this global confidence percentile before fusing (`null` = off). `source: feedforward` only; a reconstruction that carries no confidence (sfm) fuses unmasked and logs that it did |
| `mesh.mask_sky` | bool | `true` | Zero depth wherever the sky segmenter fires before fusing, so sky never seeds floaters |
| `mesh.max_faces` | int\|null | `1500000` | Face budget for the prepared mesh: past the error-bound decimation, a second QEM pass down to this count so texturing cost stays bounded (`null` = no cap) |
| `mesh.smooth_iterations` | int | `30` | Volume-keeping relax passes (meshlib `relaxKeepVolume`) on the prepared mesh, after decimation and repair, before texturing (`0` = off). Moves vertices only, never removes faces; a final `make_manifold` drops the few faces it folds. `30` measured on GH010229 (300 frames, `sdf_trunc_mult` 6): face-to-neighbor normal angle 11.3° -> 2.2° in 10 s; Taubin 10 reached 3.8°; at trunc 4, `50` passes gave 2.3° vs 2.5° for `30` |
| `mesh.texture` | bool | `true` | Also decimate, UV-unwrap and project the fused views into `<backend>/texture/` (`mesh.obj` + `mesh.mtl` + `albedo.png`). Needs a GPU |
| `mesh.use_convex_hull` | bool | `true` | Trim the ragged outer edge and patch the ground out to a rounded convex hull before hole filling (`make_convex_hull`). Ground-dominated outdoor scenes only; a mesh without a dominant ground raises, so set it `false` indoors and for objects. See `docs/mesh.md` |
| `splats.enabled` | bool | `false` | Train Gaussian splats on the `pointcloud.zarr` poses/points + `images/` (opt-in) |
| `splats.primitive` | str | `3dgs` | `3dgs` (fast kernel, antialiased) or `2dgs` (surface-aligned) |
| `splats.representation` | str | `vanilla` | `vanilla` (per-Gaussian) or `scaffold` (anchors + MLP; set `losses.opacity_reg` weight 0) |
| `splats.scaffold` | dict | absent | Scaffold-only overrides, unset by default: `n_offsets` 10, `feat_dim` 32, `voxel_multiplier` 1.0 (× median kNN seed spacing), `update_from` 1500, `update_until` 15000, `refine_every` 100, `grad_threshold` 2.0e-4, `min_opacity` 0.005, `update_init_factor` 16, `success_threshold` 0.8, `appearance_dim` 32 (0 = off), `mlp_bf16` true |
| `splats.max_steps` | int | `30000` | Training iterations |
| `splats.pose_opt` | bool | `true` | Refine camera poses jointly (`CameraOpt`) |
| `splats.appearance_opt` | bool | `false` | Per-image affine color on train views only; regularized by `losses.appearance_reg` |
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
| `splats.appearance_lr` | float | `1.0e-3` | Learning rate of the `appearance_opt` per-image color |
| `splats.cap_max` | int | `1000000` | `3dgs` only: Gaussian budget for `MCMCStrategy` |
| `splats.grow_grad2d` | float | `2.0e-4` | `2dgs` only: `DefaultStrategy` densification gradient threshold (gsplat non-absgrad default; 8e-4 starved densification) |
| `splats.num_downscales` | int | `2` | Coarse-to-fine (splatfacto): train at `1/2^num_downscales` resolution first, doubling every `resolution_schedule` steps until native. `0` disables |
| `splats.resolution_schedule` | int | `3000` | Steps per coarse-to-fine resolution doubling |
| `splats.normalize_scene` | bool | `false` | Train in splatfacto's normalised frame (centre on camera mean, scale so max \|camera coord\| = 1, `scene_scale` = 1). Outputs (ply, ckpt, zarr c2w, renders) are mapped back to world units |
| `splats.log_every` | int | `500` | Steps between loss log lines |
| `splats.losses.<name>.weight` | float | see `base.yaml` | Weight of an optional loss: `depth`, `normal_consistency`, `distortion`, `opacity_reg`, `scale_reg`, `appearance_reg`. Absent = off. `3dgs` wants `opacity_reg`+`scale_reg` (MCMC); `2dgs` wants `distortion` instead |
| `splats.losses.<name>.start` | int | `0` | Step at which that loss switches on |
| `splats.losses.appearance_reg.weight` | float | `1.0e-3` | Regularizer on the `appearance_opt` color; no-op unless `appearance_opt` is on |
| `splats.losses.normal_consistency.depth_ratio` | float | `0.0` | `2dgs` only: RaDe-GS blend weight on the median-depth normal — `(1-r)·d(n, dn_expected) + r·d(n, dn_median)`, `d = 1 - cos`. `0` = expected depth only. Rejected on `3dgs` and outside `[0, 1]` |
| `localization.enabled` | bool | `false` | Build the localization database (opt-in) |
| `localization.matcher` | str | `loma` | vismatch model name: `loma` or `xfeat`, the batch-capable models; anything else is a `ValueError` |
| `localization.retrieval` | str | `dino-salad` | Retrieval registry name, `dino-salad` or `megaloc`; stored with the DB, so queries embed with the same model |
| `reconstruction_quality_report.min_pair_overlap` | float | `0.0` | `0` scores every ordered pair; `> 0` skips pairs whose frustum overlap is below it |

### Localization matchers

`localization.matcher` is a vismatch model name, constructed as
`LocalMatcher(name)` (`collab_splats/localization/extractors.py`) — `loma` or
`xfeat`. There is one match path: reference features are extracted once and cached in
the localization zarr, the query is extracted once, and `LocalMatcher.match` pairs the
query features against each reference's cached features. Retrieval (`localization.retrieval`) picks
the top `top_k` reference frames (`CameraLocalizer` default 8), or the caller passes
`refs=`; reference pixels map back to the world-point grid through `original_coords`.

`LocalMatcher` accepts only batch-capable vismatch models (`xfeat`, `loma`): it matches
cached features, so a model without vismatch `supports_batches` raises `ValueError` at
construction time.

Backend config blocks: [docs/pointcloud.md](../docs/pointcloud.md#backends).

---

## Processed scene layout

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

- `<backend>/pointcloud.zarr` — the unified reconstruction artifact for every
  `pointcloud.method` (feedforward and sfm). Store attrs: `method`, `backend`, and for sfm
  the depth-alignment stats plus `registered_frames` / `total_frames`.
  No package versions are recorded.

  `confidence` is present only when the method produces it (absent, never zeros).

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

---

## Dashboard

The dashboard browses **dashboard-produced scenes**, whose tree is flat: it reads
`<scene>/pointcloud.zarr` for the pointcloud, `<scene>/semantics/<extractor>_lifted.zarr` for
semantic features, and `<scene>/mesh.ply` for the mesh. Point it at a scene the dashboard itself
built. It cannot read the backend-keyed tree the remote driver publishes
(`<scene>/<backend>/pointcloud.zarr`); use the viewer or a notebook for published scenes.
