# pycolmap-cuda12 4.1.1 replaces the colmap binary — design

**Date:** 2026-09-26
**Branch:** `clean/final`
**Status:** approved design, plan pending

## Problem

The runtime image copies COLMAP from `colmap/colmap:20240213.23` (3.10-dev, built 2024-02). Only one
call site uses that binary: `_generate_sift_database` in `collab_splats/pointcloud/sfm/instantsfm.py`,
which shells out to `colmap feature_extractor` + `colmap exhaustive_matcher` for GPU SIFT. Its
docstring justifies the CLI with "the wheel is CPU-only". Everything else already goes through
pycolmap 4.0.4.

- The upstream COLMAP 4 image targets ubuntu 24.04 / CUDA 12.9 / Qt6 / OpenImageIO / MKL — its
  binary cannot be copied into our ubuntu 22.04 / CUDA 12.1.1 runtime.
- `pycolmap-cuda12` now ships a GPU cp311 wheel, which removes the reason for the binary.
- Separately: pass 1 of the builder copies `README.md`, so every README edit invalidates the ~2.5 h
  CUDA compile layer. It is not needed there (verified with a dry-run: `uv sync --no-install-project`
  never reads it).

## Decision

Pin `pycolmap-cuda12==4.1.1`, move `_generate_sift_database` onto it, delete the colmap stage, and
drop `README.md` from the pass-1 COPY. One rebuild covers all of it.

### Why 4.1.1, not 4.2.0

Measured 2026-09-26 on an A40, 100 GH010229 frames (1907x1072), one shared `SIMPLE_RADIAL` camera,
no focal prior — exactly what InstantSfM builds.

Verification only, identical matches from the 3.10 binary (2,649 pairs), CPU seconds:

| pycolmap | CPU s | pairs verified | two-view estimator for our camera setup |
|---|---|---|---|
| 3.10 – 3.13 | 50 – 79 | 820 – 836 | 7-pt F LO-RANSAC + H |
| 4.0.0 / 4.0.4 | 2,654 / 2,511 | 829 / 831 | same estimator; RANSAC-loop regression |
| **4.1.1** | **77** | **841** | 7-pt F LO-RANSAC + H |
| 4.2.0 (cpu / cuda12) | 1,007 – 1,088 | 714 – 725 | 6-pt shared-focal (PoseLib) + H, MSAC |

- 4.2 routes "single camera, no focal prior, pinhole" pairs to the new
  `EstimateSharedFocalTwoViewGeometry` (`two_view_geometry.cc:626-632` @ 4.2.0): ~14x the CPU and
  ~13% fewer verified pairs. Its F path is still fast (63 CPU s when forced via per-image cameras).
- 4.0.x runs the same estimator as 3.x yet costs ~34x; fixed by 4.1.

End to end, `pycolmap-cuda12==4.1.1` vs the 3.10 binary (GPU, two reps each):

| | extract | exhaustive match | pairs |
|---|---|---|---|
| colmap 3.10 binary | 9.5 – 10.3 s | 41.1 – 43.3 s | 809 – 826 |
| pycolmap-cuda12 4.1.1 | 8.0 – 8.8 s | 24.6 – 25.0 s | 838 |

- Lock: `uv lock` changes exactly one package (`pycolmap 4.0.4` -> `pycolmap-cuda12 4.1.1`); its
  `cuda-toolkit[cudart,curand]` dep resolves to the existing 12.1.1 pins.
- Runtime: pycolmap preloads `nvidia/cuda_runtime` + `nvidia/curand` from the venv — the same
  cudart 12.1.105 torch uses. `has_cuda=True` beside torch cu121.

### Why not a CUDA upgrade

Out of scope. Only Blackwell needs it (cu128 + torch >= 2.7), and it would recompile every
extension and void the cu121-measured baselines. Separate effort if Blackwell becomes a target.

## Changes

### 1. `pyproject.toml` + `uv.lock`

```toml
# pycolmap-cuda12, pinned at 4.1.1
# - GPU SIFT wheel: replaces the colmap CLI (docs/superpowers/specs/2026-09-26-pycolmap-cuda-docker-design.md)
# - not 4.2: shared-focal two-view estimator, ~14x verification CPU for single-camera scenes
"pycolmap-cuda12==4.1.1",
```

replaces `"pycolmap>=3.1"`; `uv lock`. Both wheels install the same `pycolmap` module, so the plain
package must go. Nothing else in the lock depends on it; `pyceres==2.3` (setup.sh, non-lock) needs
only numpy.

### 2. `_generate_sift_database` on pycolmap

Same contract (`image_path`, `database_path`, `num_threads=8`; unlink partial DB and raise
`RuntimeError` on failure). Body becomes:

- `pycolmap.extract_features(database_path, image_path, camera_mode=CameraMode.SINGLE,
  reader_options=<camera_model="SIMPLE_RADIAL">, extraction_options=..., device=...)`
- `pycolmap.match_exhaustive(database_path, matching_options=..., device=...)`
- `device = Device.cuda if torch.cuda.is_available() else Device.cpu` — explicit, never `auto`
- CPU path keeps the 8-thread cap on both extraction and matching options (the 96-core OOM the
  docstring records still applies)
- drops `subprocess` / `os.environ` / `CUDA_VISIBLE_DEVICES` handling
- docstring: replace the "drives the colmap CLI" bullet with the wheel + pin rationale in one bullet;
  keep the upstream-reimplementation citation and the OOM bullet

### 3. `Dockerfile`

- delete the `colmap-source` stage and both `COPY --from=colmap-source` lines
- apt: drop COLMAP-only libs — `libboost-filesystem1.74.0 libboost-program-options1.74.0 libceres2
  libfreeimage3 libglew2.2 libgoogle-glog0v5 libqt5core5a libqt5gui5 libqt5widgets5`
- apt: add `libx11-6 libxext6 libsm6 libice6` explicitly, previously pulled in transitively by
  Qt5/GLEW; any further gap surfaces in the smoke test below, not at first run
  - open3d links `libX11` (plus `libGL`, `libudev`, `libgomp`)
  - pycolmap-cuda12 `_core.so` links `libX11`, `libXext`, `libSM`, `libICE` (found by `ldd`)
- pass-1 COPY: `pyproject.toml uv.lock LICENSE setup.sh` (no `README.md`)
- runtime smoke test: add `import open3d, cv2, pycolmap` + `assert pycolmap.has_cuda` so a missing
  system lib fails the build, not the first run (no GPU needed for either)
- header: the cache-invalidation line names the real pass-1 inputs

### 4. `README.md`

- commit the pending "5. Docker image" section (only that hunk — the intro edit at the top belongs
  to another session)
- section 4: drop `colmap` from the optional system tools
- Docker section: rebuild-trigger list matches the new pass-1 COPY

### 5. `CLAUDE.md` in-flight entry, then `CHANGELOG.md` on completion

## Testing

- **Unit:** one flat test in `tests/pointcloud/sfm/` monkeypatching `pycolmap.extract_features` /
  `match_exhaustive`: GPU path passes `Device.cuda`; CPU path passes `Device.cpu` and
  `num_threads=8`; a raised error unlinks the partial DB and surfaces as `RuntimeError`.
- **Suite:** `tests/pointcloud tests/geometry tests/wrapper tests/localization` on the relocked venv
  (4.0.4 -> 4.1.1 touches every pycolmap caller); compare against the current failure list.
- **Real run:** InstantSfM on the 100-frame GH010229 subset — registered images and track count vs
  the binary-built DB.
- **Docker (user, Mac):** rebuild; the pass-1 layer must be a cache MISS once (pyproject/uv.lock
  changed) — expected; the runtime smoke test must pass. Then on a GPU host:
  `docker run --gpus all collab-splats:release python -c "import pycolmap, torch; print(pycolmap.has_cuda, torch.cuda.get_device_name())"`.

## Out of scope

- CUDA / torch upgrade
- pycolmap >= 4.2 (revisit only with a single-camera cost measurement)
- sequential / retrieval matching for InstantSfM
- stale README section 3 (collab-data install)
