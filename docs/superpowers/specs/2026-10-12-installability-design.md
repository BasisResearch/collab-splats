# Installability: container setup fixes + standalone semantics extra

Date: 2026-10-12 · Branch: `clean/final` · Status: approved design

## Goal

1. A user launches the runtime container and runs `bash setup.sh` with no errors, false warnings
   or post-sync churn.
2. `collab_splats[semantics]` installs and imports on its own — CPU-only Linux and macOS, no CUDA,
   no compiler, no `bae` — for downstream consumers such as collab-data's `mesh_semantics`
   (which today pins the unmerged `db-semantics-package` branch, `37f7af2d`).

## Findings (measured 2026-10-12)

Container run of `setup.sh`: exit 0, all `collab_splats` modules import, 4179 tests collect.
Defects:

- `setup.sh:197` smoke imports `VGGTOmegaCreator` from `collab_splats.pointcloud` — unexported
  since `fbef6e39`, so the creator check always prints a false `[WARN]` and verifies nothing
- runtime image lacks `jq`; collab-data's `setup_local_rclone.sh` refuses to run, so no
  `collab-data` rclone remote is configured and `remote.py` cannot reach GCS
- every `setup.sh` run uninstalls then reinstalls InstantSfM (outside the lock); a plain
  `uv sync` silently drops the backend
- WordNet downloads to `/root/nltk_data`, ephemeral on RunPod — re-fetched every launch
- `setup.sh` hardcodes `/root/.local/bin/uv`; Dockerfile `uv` / `rclone` installers unpinned

Clean CPU venv probe of `collab_splats.semantics` (torch 2.5.1+cpu + semantics-level deps):

| chain | missing | blocker? |
|---|---|---|
| `utils/image.py` | kornia | no — pure Python |
| `features/ocr_lens.py` | nltk, pyarrow | no |
| `segmentation/__init__` → `mobile_sam.py` | mobile_sam | yes — git-only dep, eager import |
| `segmentation/sky.py` | segmentation-models-pytorch, safetensors | no |
| `store.py`, `sky.py` → `preproc/__init__` | av, scikit-image, pycolmap | no — PyPI wheels |
| `lifting.py` → `geometry/__init__` → `bundle_adjustment` | pypose, bae | yes — CUDA extension |

## Design

### 1. pyproject extras

- base `dependencies`: light cross-platform core only (numpy, scipy, pillow, pyyaml, tqdm,
  requests, python-dotenv, mergedeep, matplotlib, hydra-core, omegaconf)
- `semantics`: plain `torch` / `torchvision`, transformers, timm, huggingface_hub, zarr,
  numcodecs, opencv-python, scikit-learn, kornia, nltk, pyarrow,
  segmentation-models-pytorch==0.5.0 (unpinned resolves an old release needing
  `pretrainedmodels`), safetensors, clip (git), einops, av, scikit-image — no pycolmap: the
  `pycolmap; sys_platform == 'never'` override would strip it from any in-repo sync
- `reconstruction`: everything else currently in base (gsplat, bae, nvdiffrast, fused-ssim,
  pypose, mobile_sam, maskclip_onnx, ...) — unchanged contents, new location
- `remote`: collab-data (path dep) — base install no longer needs the private repo
- `instantsfm`: instantsfm (git), pyceres, scikit-sparse, easydict — now locked; the existing
  `numpy>=2,<2.2` override neutralises upstream's `numpy==1.26.4` pin
- torch / torchvision cu121 overrides and `[tool.uv.sources]` gated to linux/win32, with
  complementary darwin entries (an override replaces the requirement outright, so a false
  marker alone would drop torch)
- `requires-python` stays `>=3.11` (pinning `==3.11.*` would block 3.12 semantics consumers)
- re-lock; gate: `uv sync --locked --all-extras --dry-run` reports no change on Linux except
  the InstantSfM packages entering the lock

### 2. Code: three lazy imports, existing patterns only

- `geometry/__init__.py`: `BundleAdjustment` / `BundleAdjustmentConfig` join the existing
  `__getattr__` (already lazy for `LoopClosure`); `import collab_splats.geometry` stops loading bae
- `preproc/__init__.py`: `calibrate_camera` / `undistort_frames` move behind a module
  `__getattr__`, same shape as geometry's; `import collab_splats.preproc` stops loading pycolmap
  (`undistort.py` is its only pycolmap importer)
- `segmentation/mobile_sam.py`: `mobile_sam` imported inside the auto strategy with a clear
  `ImportError` message — the CLAUDE.md optional-heavy-dep exception, as `sam3` / `maskclip` do

bae remains required by `bundle_adjustment`, `photometric`, `reconstructor`, loop-closure
`wrapper` and InstantSfM (lazy) — all in the `reconstruction` path, which needs CUDA anyway.

### 3. setup.sh and Docker

- smoke test resolves every creator through `get_creator(...)` for vggtx, mapanything,
  vggt_omega, loger
- `uv` from `PATH`, not `/root/.local/bin/uv`
- InstantSfM block deleted; sync failure help names `libsuitesparse-dev`
- WordNet to `NLTK_DATA=/workspace/models/nltk_data` (Dockerfile ENV + setup.sh)
- Dockerfile: `jq` in the runtime apt list; uv and rclone installers version-pinned
- README: on RunPod the `/workspace` volume shadows the image's checkout — clone into
  `/workspace`, then `bash setup.sh`
- out of repo (user-applied): drop the `apt install rclone jq` line from
  `/workspace/setup_profile.sh`; rotate the tokens in `/workspace/profile.sh`

### 4. Tests

- new `tests/semantics/test_standalone_imports.py`: AST walk of module-level imports reachable
  from `collab_splats.semantics` (following `collab_splats.*` edges, including package
  `__init__`s); fails if any third-party top-level module falls outside the `semantics` extra
  allowlist; no network, no venv
- one-off proof recorded in the plan: `uv pip install '.[semantics]'` into a clean CPU venv from
  a cwd outside the repo, then import every `collab_splats.semantics` module
- full gate: `pytest tests/` in the container venv, plus a fresh `bash setup.sh` with exit 0 and
  no `[WARN]`

## Trade-offs

- InstantSfM locked: `uv sync --all-extras` fails hard on a bare host without
  `libsuitesparse-dev` (was best-effort); the image ships it, setup.sh prints the hint
- collab-data must re-pin `collab_splats[semantics]` from `@db-semantics-package` to the landed
  commit — a change in their repo

## Out of scope

- removing bae from the reconstruction path (CPU BA)
- moving the image checkout off `/workspace`
