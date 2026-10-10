# Pointcloud Module

`collab_splats.pointcloud` builds a `PointcloudResult` (points, poses, depth, per-pixel
provenance) from raw video or images, via one of two backend families: feedforward
(VGGT-X, VGGT-Omega, MapAnything, LoGeR) or sfm (InstantSfM, COLMAP or hloc, + VDA depth).

---

## Depth alignment (sfm path)

`depth.py` supplies the sfm backend's metric depth: `estimate_depth` runs Video-Depth-Anything
per keyframe (cached under `depth_vda/`), and `align_depth` fits a per-frame scale from COLMAP
track observations to bring that depth to the reconstruction's world scale.

`estimate_depth` loads **Metric-Video-Depth-Anything-Large**, licensed **cc-by-nc-4.0**
(non-commercial); Metric-Small (Apache-2.0) is the commercial-safe alternative, not taken here.

## Bundle adjustment with loop closure

With `pointcloud.loop_closure` and `pointcloud.bundle_adjustment.enabled` both on,
`pointcloud.bundle_adjustment.strategy` picks the solve:

- `global` (default): plain LC windows, then the `refine` stage solves every frame at once
- `window`: BA inside each loop-closure window, right after the window's forward pass and
  before its submap enters the pose graph; the `refine` stage is skipped
- `window+global`: both, in that order

Window solves: the first refined window sets the focal and later windows hold it fixed; a
window whose solve fails keeps its feedforward poses; each window's outcome (start frame,
focal, loss, time, peak GPU memory) is stored in the `pointcloud.zarr` attrs under
`window_ba`. The LC store carries model-grid images and per-point source pixels, so `refine`
reads it like a single-pass store.

## Backends

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
[018](superpowers/decisions/018-sfm-backends.md)):

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
