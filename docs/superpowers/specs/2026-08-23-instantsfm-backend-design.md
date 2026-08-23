# InstantSfM pointcloud backend — design

**Date:** 2026-08-23
**Status:** Draft — pending functional smoke green (see Open risks)
**Upstream:** https://github.com/cre185/InstantSfM @ `d3e599e1a42b4c5a806a84d9f383e1005d25f61b` (v0.3.0)
**Paper:** "InstantSfM: Towards GPU-Native SfM for the Deep Learning Era" (IROS 2026, arXiv:2510.13310)

## Goal

Add InstantSfM — GPU-native **global** SfM (rotation averaging + global positioning +
retriangulation + bae-based BA) — as a third `pointcloud.method: sfm` backend beside the
existing (unwired) `colmap` and `hloc` creators. Gives the pipeline a classical-geometry
alternative to the four feedforward backbones: slower than feedforward, faster than
incremental COLMAP, poses + sparse cloud from epipolar geometry rather than a learned prior.

## Decisions (from brainstorm Q&A)

| Question | Decision |
|---|---|
| Role | Full pointcloud backend (`method: sfm`, `backend: instantsfm`) |
| Dense depth | **Sparse-only v1** — no `feedforward.zarr`; dense story deferred |
| Invocation | Same-env Python API (verified 2026-08-23, see Install) |
| Features | Their native feature pipeline as-is; our xfeat/loma injection = possible follow-on |
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
- Per-backend block `pointcloud.instantsfm:` forwarded verbatim to the creator, keys:
  - `feature_handler`: one of `colmap | dedode | disk+lightglue | superpoint+lightglue | sift`
    (default `colmap` — only handler exercised so far; also what their docker example uses).
  - `config_name`: their `manual_config_name` (default `"colmap"`; **must match the
    feature_handler family** — the 2026-08-23 smoke showed default config × SIFT db
    yields broken track filtering).
  - `single_camera`: bool, default `true` (video frames share one camera).
- The existing `method='sfm' is experimental` warning stays.

### 2. `InstantSfMCreator` (`pointcloud/sfm.py`)

Dataclass sibling of `ColmapCreator`/`HlocCreator`, same contract:
`reconstruct(image_dir: Path, output_dir: Path) -> PointcloudResult`.

- Lazy-imports `instantsfm` (optional heavy dep; clear ImportError names the install extra).
- Calls their Python API (controllers: `feature_handler` → `global_mapper` →
  `reconstruction_writer`) with their `Config` loaded by `config_name` — NOT the CLI
  entrypoints; no subprocess, no argv parsing.
- Reads the written COLMAP model with pycolmap → `PointcloudResult(reconstruction,
  frame=CoordinateFrame.COLMAP, image_paths=..., confidence=None)`.
- Attribution comment at the call site: upstream repo + commit + files, per repo rule.
- Docstring carries the CC-BY-NC-4.0 note.

### 3. `_run_sfm` wiring (`wrapper/reconstructor.py`)

Replaces the stub, dispatches all three sfm backends:

1. Stage images: export `frames.zarr` → `backend_dir/images/` (uint8 PNGs, skipped when
   already present and not `overwrite`). frames.zarr stays the canonical store; the images
   dir is a staging artifact for SfM backends only.
2. Creator map `{"colmap": ColmapCreator, "hloc": HlocCreator, "instantsfm": InstantSfMCreator}`,
   kwargs from `pointcloud.<backend>` block.
3. Shared tail identical to the feedforward path: optional clean, `sparse_pc.ply` export,
   `transforms.json`, COLMAP binary model on disk.

Only `instantsfm` is smoke-tested in this effort; colmap/hloc get the wiring for free and
stay experimental.

### 4. Sparse-only downstream contract

No `feedforward.zarr` is written. Stages that require dense model tensors must fail loud
with a message naming the backend and the reason:

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

## Testing

- Unit (no GPU): backend accepted by config validation; `_run_sfm` dispatch reaches the
  creator; frames.zarr → images staging writes N files and is idempotent; sparse-only
  guards raise with the documented messages; empty-track pre-check raises cleanly.
- Smoke (GPU, human-gated): `data/tutorial/` video end-to-end
  `preproc → pointcloud(sfm/instantsfm)` → valid COLMAP model + `sparse_pc.ply` +
  `transforms.json`; then `splats` trains from it.
- Eval (GPU, tmux, human-gated): `instantsfm` condition in `evals/scripts/eval.py`,
  7-Scenes chess/seq-01 — ATE/RPE + wall-clock vs vggtx/mapanything/vggt_omega/loger;
  results appended to a measured report.

## Non-goals

- Dense depth for this backend (hybrid FF-depth-on-SfM-poses, mono-depth fill) — deferred.
- GPU pycolmap build (system CUDA COLMAP binary already exists; pycolmap wheel stays CPU;
  speed-only change, separate infra effort if ever wanted).
- Their `ins-gs` 3DGS trainer and `vis/` stack — we have `collab_splats/splats/`.
- Feature injection from our xfeat/loma matchers — follow-on candidate.
- colmap/hloc backend validation beyond compile-level wiring.

## Rejected alternatives

- **Subprocess CLI into a separate venv** — dep isolation not needed (verified same-env
  install leaves every pin intact); would cost a second venv in Docker and lose
  programmatic config. Recorded as fallback only if a runtime numpy-2 incompatibility
  proves unpatchable.
- **Vendoring the SfM core** — upstream is active; huge surface for no control we need.
