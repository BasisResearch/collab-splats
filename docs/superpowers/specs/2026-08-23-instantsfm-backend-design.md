# InstantSfM pointcloud backend — design

**Date:** 2026-08-23
**Status:** Draft — pending functional smoke green (see Open risks)
**Upstream:** https://github.com/cre185/InstantSfM @ `d3e599e1a42b4c5a806a84d9f383e1005d25f61b` (v0.3.0)
**Paper:** "InstantSfM: Towards GPU-Native SfM for the Deep Learning Era" (IROS 2026, arXiv:2510.13310)

## Goal

Add the **full InstantSfM pipeline** — GPU-native **global** SfM (rotation averaging +
global positioning + retriangulation + bae-based BA), fed by Video Depth Anything (VDA)
metric depth as a constraint (`use_depths: true`) — as a third `pointcloud.method: sfm`
backend beside the existing (unwired) `colmap` and `hloc` creators.

This is a bet against the two measured pains of the feedforward backbones: **pose quality**
(poses from epipolar geometry + depth-constrained global positioning, a different objective
from the rejected reprojection-BA family) and **depth fidelity** (VDA depth maps generated
per scene; consumed by SfM in v1, available to splats/mesh in a fast-follow). Slower than
feedforward, faster than incremental COLMAP.

## Decisions (from brainstorm Q&A)

| Question | Decision |
|---|---|
| Role | Full pointcloud backend (`method: sfm`, `backend: instantsfm`) — primary bet on pose + depth quality, not a side-by-side experiment |
| Depth | **Depth path is THE path (2026-08-23 review):** VDA metric depth generated in v1, fed to SfM (`use_depths=True`, not configurable — only shipped mode). Maps persist at `backend_dir/depth_vda/` only. Downstream dense consumption (splat targets, TSDF mesh) = named fast-follow spec, NOT v1 |
| Invocation | **Their code, never CLI/subprocess** (2026-08-23 review): InstantSfM via Python API controllers; VDA via `VideoDepthAnything` class imported from cloned upstream repo (`third_party/`, not pip-installable) — zero reimplementation |
| Features | Their native feature pipeline as-is |
| Eval | In-plan: chess/seq-01 ATE + wall-clock vs 4 feedforward backends |
| License | CC-BY-NC-4.0 — user cleared as non-issue; noted in creator docstring |

## Implementation principles

- Reuse: `sfm.py` creator pattern, `PointcloudResult`, `write_pointcloud_ply`,
  existing `_run_sfm` dispatch seam, shared post-pointcloud tail (clean/PLY/transforms.json).
- Retire: nothing (the `_run_sfm` `NotImplementedError` stub is replaced, not removed).
- Minimal: no `strategy`-style config surface beyond what the smoke run proves necessary;
  no subprocess layer; no vendoring.

## Architecture

### 1. Config + dispatch

- `instantsfm` joins `_SFM_BACKENDS` in `wrapper/reconstructor.py`.
- Usage: `pointcloud: {method: sfm, backend: instantsfm}`. Output dir `out/instantsfm/`
  via the existing `backend_dir` property.
- Per-backend block `pointcloud.instantsfm:` — **ONE key**:
  - `features`: one of `colmap | dedode | disk+lightglue | superpoint+lightglue | sift`
    (default `colmap`). The creator derives BOTH their `feature_handler` AND
    `manual_config_name` from it internally — the 2026-08-23 smoke showed a mismatched
    pair (default config × SIFT db) silently breaks track filtering, so the pair is not
    user-composable.
- NOT config keys (creator dataclass fields with fixed defaults; eval overrides
  programmatically): `use_depths=True` (depth path is the only shipped mode),
  `single_camera=True` (video frames share one camera), `encoder="vitl"`.
- The existing `method='sfm' is experimental` warning stays.

### 2. VDA depth generation (`pointcloud/vda.py`)

InstantSfM does not generate depth — it consumes a `depth_vda/` folder (metric depth only;
their `data_reader.py` samples it at keypoint locations for global positioning + BA cost).

**We run their code, we do not reimplement it.** The
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) repo is not
pip-installable (no setup.py/pyproject) — upstream InstantSfM's own
`tools/video_depth_anything.py` handles this with `sys.path.insert(vda_path)` then
`from video_depth_anything.video_depth import VideoDepthAnything`. We do the same:

- VDA repo cloned to `third_party/Video-Depth-Anything` by setup.sh (pinned commit),
  Metric-Video-Depth-Anything **vitl** checkpoint downloaded beside it.
- New module `collab_splats/pointcloud/vda.py`: `generate_vda_depth(frames, out_dir) ->
  Path` — pure orchestration: sys.path shim, load `VideoDepthAnything` (their class, their
  weights, their `infer_video_depth`), write `depths.npz` in their layout. Attribution at
  the call site: both upstream repos + commits, per repo rule.
- Frames feed in **directly from `frames.zarr`** as arrays. Their tools script's
  images→mp4→re-decode roundtrip is CLI glue for people with image folders — skipped
  (their model API takes frame arrays; the roundtrip adds a lossy encode for nothing).
- Output: `backend_dir/depth_vda/depths.npz` beside the staged `images/`, exactly where
  `ins-sfm`'s data reader looks. Maps persist after reconstruction — this is the artifact
  the fast-follow spec (splat depth targets, TSDF fusion) consumes. Nothing downstream
  reads them in v1.
- VDA is a temporal model — needs a continuous ordered sequence. fps-sampled
  `frames.zarr` satisfies this; optical-flow-sampled sets with wild gaps degrade it
  (documented, not guarded).
- Metric-only: InstantSfM rejects relative depth. The mono-depth rule ("detail, never
  scale") is relaxed *inside* the SfM objective only — their global positioning estimates
  scene scale from depth values; the output COLMAP model stays the single pose authority.

### 3. `InstantSfMCreator` (`pointcloud/sfm.py`)

Dataclass sibling of `ColmapCreator`/`HlocCreator`, same contract:
`reconstruct(image_dir: Path, output_dir: Path) -> PointcloudResult`.

- Lazy-imports `instantsfm` (optional heavy dep; clear ImportError names the install extra).
- Calls their Python API (controllers: `feature_handler` → `global_mapper` →
  `reconstruction_writer`) with their `Config` loaded by the name derived from `features`,
  `use_depths=True` set on it — NOT the CLI entrypoints; no subprocess, no argv parsing.
- Reads the written COLMAP model with pycolmap → `PointcloudResult(reconstruction,
  frame=CoordinateFrame.COLMAP, image_paths=..., confidence=None)`.
- Attribution comment at the call site: upstream repo + commit + files, per repo rule.
- Docstring carries the CC-BY-NC-4.0 note.

### 4. `_run_sfm` wiring (`wrapper/reconstructor.py`)

Replaces the stub, dispatches all three sfm backends:

1. Stage images: export `frames.zarr` → `backend_dir/images/` (uint8 PNGs, skipped when
   already present and not `overwrite`). frames.zarr stays the canonical store; the images
   dir is a staging artifact for SfM backends only.
2. Run `generate_vda_depth` → `backend_dir/depth_vda/` (skipped when present and not
   `overwrite`).
3. Dispatch to `InstantSfMCreator` (kwargs from `pointcloud.instantsfm`). **`colmap` and
   `hloc` keep their `NotImplementedError` behaviour** — wiring untested backends "for
   free" was overengineering; they get wired when someone needs them.
4. Shared tail identical to the feedforward path: optional clean, `sparse_pc.ply` export,
   `transforms.json`, COLMAP binary model on disk.


### 5. Downstream contract (v1)

No `feedforward.zarr` is written. VDA depth exists on disk (`depth_vda/`) but **no stage
reads it in v1** — wiring it into splat depth targets and TSDF fusion is the fast-follow
spec (which must solve alignment/masking there, not here). Stages that require dense model
tensors must fail loud with a message naming the backend and the reason:

- `mesh` (TSDF needs dense depth), `semantics` lifting, `localize` (feature cache builds
  from feedforward.zarr), `refine` (BA track extraction needs per-frame model tensors),
  `verify` (features come from the localize cache).
- `splats` **works**: trains from COLMAP poses + sparse points + frames.zarr; depth loss
  simply has no targets (0 = no target is already the trainer contract).
- Guard location: each stage method's existing `_resolve_result` / zarr-load seam; error
  message points at `method: feedforward` for the dense pipeline.

## Install (verified 2026-08-23 on this container)

Probe already executed in the live `reconstruction` venv — results:

```
uv pip install --no-deps 'git+https://github.com/cre185/InstantSfM'   # pins d3e599e1
uv pip install pyceres==2.3
apt-get install -y libsuitesparse-dev
uv pip install scikit-sparse==0.4.15                                   # builds in ~11 s
```

- Full sfm import chain green (`instantsfm.scripts.sfm`, controllers, processors).
- **Our pins untouched after install**: numpy 2.1.3, gsplat 1.5.3, scipy 1.17.1,
  torch 2.5.1+cu121, pypose 0.7.5, cv2 4.10.0, bae 0.2.4.
- `--no-deps` is load-bearing: their pyproject pins numpy==1.26.4, unpinned gsplat,
  git pypose/fused-ssim — resolving their metadata would rewrite our lock (same class as
  the vggt-omega `--no-deps` install).
- `tensorly`, `nerfview`, `splines` deliberately NOT installed — used only by their
  `vis/` 3DGS path, which we do not call.
- **VDA is a clone, not an install**: no setup.py — `third_party/Video-Depth-Anything`
  (pinned commit) + sys.path shim, upstream InstantSfM's own pattern. Their
  requirements.txt pins torch 2.1.1 / xformers 0.0.23 — ignored; their modules must import
  against our torch 2.5.1+cu121 / py3.11 (plan task: probe import + one inference; if
  xformers is a hard import, install the wheel matching OUR torch, never their pin).
  Metric checkpoint `metric_video_depth_anything_vitl.pth` (HF) downloaded by setup.sh.
- Packaging: optional uv extra `instantsfm` (git SHA pin + pyceres + scikit-sparse) with
  `no-build-isolation`-style handling as needed; `libsuitesparse-dev` added to the Docker
  image. TRAP (memory): plain `uv sync` prunes extras — setup.sh must install the extra
  explicitly.

## Open risks (must close before the backend ships)

1. **Functional smoke not yet green.** 12-frame / 960 px probe: `ins-feat` (colmap handler,
   system COLMAP 3.10-dev CUDA binary, CPU SIFT flag) completed; `ins-sfm
   --manual_config_name colmap` established 81 tracks, then observation-count filtering
   dropped **81 → 0**, and the empty-track path crashed.
   Plan task: rerun with a realistic frame set (30–60 frames, closer spacing, native res)
   before debugging further — 12 frames × 150-frame stride is far sparser than any real run.
   If filtering still empties, instrument `min_observations` / track establishment under
   numpy 2 vs upstream's numpy 1.26.4.
2. **Upstream numpy-2 bug (report/patch upstream):** `scene/defs.py::filter_by_mask` —
   empty non-bool mask (`np.array([])`, float64) used as index; DeprecationWarning under
   numpy 1.x, hard IndexError under numpy 2. Masks the real "0 tracks" condition. Creator
   must pre-check for empty tracks and raise a clear error regardless of upstream fix.
3. **Upstream tested env divergence:** they test py3.12 / numpy 1.26.4 / torch 2.3.1; we run
   py3.11 / numpy 2.1.3 / torch 2.5.1. Imports and stage 1 are proven; numerical parity is
   not. The eval run doubles as the parity check.
4. **VDA import against our torch unverified** (see Install). Also VDA quality itself is a
   bet: "metric" VDA scale error on indoor scenes is unmeasured here — if its scale is
   badly wrong, depth constraints could *hurt* poses. The `use_depths=False` eval ablation
   (programmatic creator override, not a config key) isolates this.

## Testing

- Unit (no GPU): backend accepted by config validation; `_run_sfm` dispatch reaches the
  creator; `features` key derives a matched handler/config pair for every allowed value;
  frames.zarr → images staging writes N files and is idempotent; downstream guards raise
  with the documented messages; empty-track pre-check raises cleanly.
- Smoke (GPU, human-gated): `data/tutorial/` video end-to-end
  `preproc → pointcloud(sfm/instantsfm)` → valid COLMAP model + `sparse_pc.ply` +
  `transforms.json`; then `splats` trains from it.
- Eval (GPU, tmux, human-gated): two conditions in `evals/scripts/eval.py` —
  `instantsfm` (the shipping configuration, depth-constrained) and `instantsfm_nodepth`
  (`use_depths=False` programmatic override, isolates the VDA contribution) —
  7-Scenes chess/seq-01, ATE/RPE +
  wall-clock vs vggtx/mapanything/vggt_omega/loger; results appended to a measured report.
  This eval is the gate on the whole bet: if `instantsfm` does not beat the best
  feedforward ATE, the fast-follow (dense consumption) does not start.

## Non-goals

- **Downstream consumption of VDA depth** (splat depth targets, TSDF mesh from
  `depth_vda/`) — fast-follow spec, gated on the eval above. v1 generates and persists
  the maps; nothing reads them past SfM.
- GPU pycolmap build (system CUDA COLMAP binary already exists; pycolmap wheel stays CPU;
  speed-only change, separate infra effort if ever wanted).
- Their `ins-gs` 3DGS trainer and `vis/` stack — we have `collab_splats/splats/`.
- Feature injection from our xfeat/loma matchers.
- colmap/hloc backend wiring — stubs stay `NotImplementedError`.

## Rejected alternatives

- **Subprocess CLI into a separate venv** — dep isolation not needed (verified same-env
  install leaves every pin intact); would cost a second venv in Docker and lose
  programmatic config. Recorded as fallback only if a runtime numpy-2 incompatibility
  proves unpatchable.
- **Vendoring the SfM core** — upstream is active; huge surface for no control we need.
- **Reimplementing the VDA runner loop** — their `VideoDepthAnything` class is importable
  from a clone; our module is orchestration only (2026-08-23 review).
- **Cut as overengineering (2026-08-23 review):** `depth_encoder` / `single_camera` /
  `use_depths` config keys (always-default params — dataclass fields, not user surface);
  the `feature_handler`+`config_name` pair (must-match foot-gun → one `features` key);
  wiring colmap/hloc "for free"; the images→mp4→re-decode roundtrip from their tools
  script; persisting depth in a second location beside `frames.zarr`.
