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
- **Talk2DINO (default extractor) cannot load**: `ImportError: This modeling file requires the
  following packages that were not found in your environment: clip`. Regression from
  `47633e39` (2026-10-08), which dropped `clip` as unused and `einops` / `omegaconf` as
  upstream-covered; the Talk2DINOv3 HF remote code imports all three and transformers'
  `check_imports` hard-fails on any missing. A repo grep cannot see these imports
- runtime rclone is v1.58.1 (2022); `install.sh` serves 1.75.2 today
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
| `lifting.py` → `geometry/projection` → `transforms` | open3d | yes — no wheels past py3.12, 400 MB |

AST reachability from every `collab_splats.semantics` module, with the §2 lazy imports applied
(prototype run 2026-10-12): third-party set is exactly PIL, av, cv2, huggingface_hub, kornia,
nltk, numpy, pyarrow, safetensors, scipy, segmentation_models_pytorch, skimage, sklearn, torch,
torchvision, tqdm, transformers, zarr — plus open3d until `transforms.py` defers it.

## Design

### 1. pyproject extras

- base `dependencies`: light cross-platform core already in base today — numpy, scipy,
  pillow, pyyaml, tqdm, matplotlib, mergedeep, python-dotenv, zarr, huggingface_hub
- `semantics` — the probe's list plus Talk2DINO's remote-code imports:
  - plain `torch` / `torchvision`, transformers, timm, opencv-python, scikit-learn, kornia,
    nltk, pyarrow, safetensors, av, scikit-image
  - `segmentation-models-pytorch==0.5.0` (unpinned resolves an old release needing
    `pretrainedmodels`)
  - `clip @ git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`
    as a PEP 508 direct URL, not a `[tool.uv.sources]` entry: pip and `--no-sources` consumers
    never read our `tool.uv` table, and bare `clip` resolves to an unrelated PyPI package; CLIP
    itself needs `ftfy`, `regex`
  - correction (Task 11 proof): uv DOES apply our `[tool.uv.sources]` to uv consumers of a
    path/git install (torch from the cu121 index on non-darwin); only `override-dependencies`
    stay project-local. CPU-only consumers pass `--no-sources` (README documents it)
  - `einops`, `omegaconf` (Talk2DINOv3 remote code)
  - CLIP's own `ftfy` / `regex` / `packaging` arrive through its requires-dist — not listed
  - no pycolmap: the `pycolmap; sys_platform == 'never'` override strips it from any in-repo
    sync, and the lazy `preproc` init (§2) removes the need
- `reconstruction`: `collab_splats[semantics]` (self-reference — av, transformers, kornia, cv2,
  ... left base for `semantics`, and the pipeline still needs them) plus everything else
  currently in base (setuptools<70, wheel, ninja, the cu121 torch pins, gsplat, fused-ssim, bae,
  pypose, nvdiffrast, nvidia-cudss, pycolmap-cuda12, mobile_sam, maskclip_onnx, torchpack,
  vismatch, ultralytics, pyvista, viser, meshlib, trimesh, pydantic, seaborn, ftfy, open3d) —
  contents unchanged, new location
  - the `+cu121` torch pins carry `; sys_platform == 'linux'`: the lock is universal, and an
    unmarked `+cu121` pin has no darwin wheel to resolve
- `remote`: collab-data (path dep) — base install no longer needs the private repo; the
  `reconstruct` CLI imports `remote.py` at top (`__main__.py:22`), so the CLI needs
  `[reconstruction,remote]` — `setup.sh` gets both via `--all-extras`
- `instantsfm`: instantsfm (git), pyceres, scikit-sparse, easydict — now locked
  - upstream pins far more than numpy: `opencv-python==4.10.0.84`, `scipy==1.13.0`,
    `tqdm==4.66.5`, `kornia==0.8.0`, `viser==0.2.11`, `torchmetrics[image]`, ...
  - so a `[[tool.uv.dependency-metadata]]` entry (the pattern bae / gsplat already use)
    declares instantsfm's requires-dist as empty — the lock equivalent of today's `--no-deps`
- torch / torchvision cu121 overrides and `[tool.uv.sources]` gated to linux/win32, with
  complementary darwin entries (an override replaces the requirement outright, so a false
  marker alone would drop torch)
- `requires-python` stays `>=3.11` (pinning `==3.11.*` would block 3.12 semantics consumers)
- no `[tool.uv] environments` added: the markers above keep darwin resolvable; the plan's
  first lock attempt proves it, and `environments` is the fallback if it does not
- re-lock; gate: `uv sync --locked --all-extras --dry-run` reports no change on Linux except
  the InstantSfM packages entering the lock

### 2. Code: four lazy imports, existing patterns only

- `geometry/__init__.py`: `BundleAdjustment` / `BundleAdjustmentConfig` join the existing
  `__getattr__` (already lazy for `LoopClosure`); `import collab_splats.geometry` stops loading bae
- `preproc/__init__.py`: `calibrate_camera` / `undistort_frames` move behind a module
  `__getattr__`, same shape as geometry's; `import collab_splats.preproc` stops loading pycolmap
  (`undistort.py` is its only pycolmap importer)
- `segmentation/mobile_sam.py`: `mobile_sam` imported inside the auto strategy with a clear
  `ImportError` message — the CLAUDE.md optional-heavy-dep exception, as `sam3` / `maskclip` do
- `geometry/transforms.py`: `open3d` imported inside `fit_dominant_plane`, its only user
  (RANSAC floor plane), with the same `ImportError` message shape; `semantics.lifting` reaches
  `transforms` via `projection`, and open3d ships no wheels past Python 3.12. open3d moves to
  `reconstruction`
- `tests/conftest.py`: the `pkg_resources.packaging` shim tolerates a missing `pkg_resources`
  (a semantics-only env may carry setuptools >= 81); `test_maskclip_onnx_importable` uses
  `pytest.importorskip` — both ported from `37f7af2d`

bae remains required by `bundle_adjustment`, `photometric`, `reconstructor`, loop-closure
`wrapper` and InstantSfM (lazy) — all in the `reconstruction` path, which needs CUDA anyway.

### 3. setup.sh and Docker

- smoke test resolves every creator through `get_creator(...)` for vggtx, mapanything,
  vggt_omega, loger
- `uv` from `PATH`, not `/root/.local/bin/uv`
- InstantSfM block deleted; sync failure help names `libsuitesparse-dev`
- WordNet to `NLTK_DATA=/workspace/models/nltk_data`: setup.sh runs
  `"$PYTHON" -m nltk.downloader -d "$NLTK_DATA" wordnet` (`-d` explicit — the downloader's
  default dir ignores the env var on some nltk versions); `ocr_lens` reads WordNet at runtime
- Dockerfile: `jq` in the runtime apt list; uv pinned to 0.12.19 (the version in this image)
  via `https://astral.sh/uv/0.12.19/install.sh`; rclone 1.75.2 from
  `https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip` — `install.sh` accepts
  only `beta`, no version argument
- `NLTK_DATA` also lands in the runtime Dockerfile `ENV` and the `.bashrc` block — the path
  must be visible to every process that reads WordNet, not only to setup.sh
- README: on RunPod the `/workspace` volume shadows the image's checkout — clone into
  `/workspace`, then `bash setup.sh`; new "semantics only" install line; drop the claim that
  `uv pip install` ignores `[tool.uv]` (false from inside the repo cwd — the probe hit it); the
  rclone line points at the pinned zip
- `docs/source/api/pointcloud.rst` instantsfm **Install.** paragraph: locked `instantsfm`
  extra, no `--no-deps` block
- out of repo (user-applied): drop the `apt install rclone jq` line from
  `/workspace/setup_profile.sh`; rotate the tokens in `/workspace/profile.sh`

### 4. Tests

- new `tests/semantics/test_standalone_imports.py`: AST walk of module-level imports reachable
  from `collab_splats.semantics` (following `collab_splats.*` edges, including package
  `__init__`s); fails if any third-party top-level module falls outside the `semantics` extra
  allowlist; no network, no venv
  - skips `if TYPE_CHECKING:` bodies (`lifting.py:22` imports `pointcloud.base` there) and
    function bodies (the lazy imports)
  - allowlist = top-level import names of base ∪ `semantics`; stdlib from
    `sys.stdlib_module_names`
  - allowlist maps import names to dists (`cv2` → opencv-python, `skimage` → scikit-image,
    `sklearn` → scikit-learn, `PIL` → pillow, `yaml` → pyyaml, ...)
- one-off proof recorded in the plan: `uv pip install '<repo>[semantics]'` into a clean CPU
  venv from a cwd outside the repo (inside it, uv applies our overrides — the probe hit this),
  then import every `collab_splats.semantics` module and construct `Talk2DinoExtractor` on CPU
- container: `Talk2DinoExtractor()` loads after the re-sync (regression check for `clip`)
- full gate: `pytest tests/` in the container venv, plus a fresh `bash setup.sh` with exit 0 and
  no `[WARN]`

## Trade-offs

- InstantSfM locked: `uv sync --all-extras` fails hard on a bare host without
  `libsuitesparse-dev` (was best-effort); the image ships it, setup.sh prints the hint
- collab-data must re-pin `collab_splats[semantics]` from `@db-semantics-package` to the landed
  commit — a change in their repo
- semantics consumers need Python >= 3.11 (collab-data declares >= 3.10)
- the self-hosted GPU CI runner reuses the full image env; the semantics-only install is
  proven by the one-off clean venv plus the AST test, not by a CI job

## Docs on completion

- CLAUDE.md In-Flight entry while open; CHANGELOG entry and In-Flight removal on landing

## Out of scope

- removing bae from the reconstruction path (CPU BA)
- moving the image checkout off `/workspace`
