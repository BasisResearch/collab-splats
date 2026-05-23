# cu121 Migration Verification — Fixes Required

**Date:** 2026-05-22
**Branch:** refactor/cu121
**Result:** 565 passed, 0 failed

This document records every code change required to make the cu121 migration
(CUDA 11.8/torch 2.1.2/Python 3.10 → CUDA 12.1/torch 2.4.0/Python 3.11) fully
functional. Changes are grouped by root cause.

---

## 1. Setup scripts used wrong Python / pip and lacked build isolation

**Affected files:** `setup.sh`, `setup_nerfstudio.sh`, `setup_feedforward.sh`

**Symptoms:**
- Bare `python3` / `pip` resolved to system Python 3.13 instead of the nerfstudio
  conda env (Python 3.11).
- CUDA extensions (gsplat-rade, bae) failed to build because pip's build-isolation
  env had no torch — their `setup.py` files import torch at build time.
- `setup_nerfstudio.sh` failed with
  `ERROR: Could not find a version that satisfies the requirement open3d>=0.16.0`
  because nerfstudio's transitive dep resolver can't find the open3d wheel, even
  though `pip install open3d` directly succeeds (0.19.0 available).

**Fixes:**

All three scripts: added
```bash
PYTHON=/opt/conda/envs/nerfstudio/bin/python
PIP=/opt/conda/envs/nerfstudio/bin/pip
export PIP_NO_BUILD_ISOLATION=1
```
and replaced all bare `python`/`pip` calls with `$PYTHON`/`$PIP`.

`setup_nerfstudio.sh` additionally:
- Pre-installs `open3d` before `pip install -e /workspace/nerfstudio/`.
- Pre-installs gsplat-rade with `--no-build-isolation` **before** nerfstudio install.
  pip always rebuilds git-URL deps even when already installed; gsplat's `setup.py`
  imports torch — this rebuild fails without build isolation.
- Changed the nerfstudio pyproject.toml patch for gsplat from a git URL to
  `gsplat>=1.4.0` (plain floor specifier). With the git URL, nerfstudio install
  would trigger a rebuild; with a plain specifier, pip sees the requirement satisfied
  by the pre-installed wheel and skips the rebuild.

**Commits:** `2eed6bb`, `30516f2`

---

## 2. pyproject.toml used git URLs for CUDA extension deps

**Affected file:** `pyproject.toml`

**Symptoms:**
- `pip install -e .` triggered rebuilds of gsplat-rade and bae from their git URLs.
- Both packages have `setup.py` files that import torch at build time.
- Rebuild fails inside pip's build-isolation env (no torch there).

**Root cause:** git-URL deps (`pkg @ git+https://...`) are always rebuilt by pip,
even if the exact package is already installed. CUDA extensions that import torch at
build time cannot be built in an isolated env.

**Fix:** Replaced git URLs with plain version floor specifiers:
- `gsplat @ git+https://github.com/brian-xu/gsplat-rade.git` → `gsplat>=1.4.0`
- `bae @ git+https://...` → `bae>=0.2.3`

Both packages are pre-installed by Docker and the setup scripts using
`--no-build-isolation`. The plain specifier tells pip the requirement is already
satisfied.

**Commit:** `2eed6bb`

---

## 3. VGGT-X: numpy 2.x scalar assignment to CUDA tensor

**Affected file:** `third_party/VGGT-X/vggt/utils/pose_enc.py`

**Error:**
```
TypeError: can't assign a numpy.float32 to a torch.cuda.FloatTensor
  File ".../vggt/utils/pose_enc.py", line 118, in pose_encoding_to_extrinsics_and_intrinsics
    intrinsics[..., 0, 2] = W / 2
```

**Root cause:** numpy 2.x changed scalar casting behavior. `H, W = image_size_hw`
unpacks numpy float32 scalars. In numpy 1.x, `W / 2` produced a Python-compatible
scalar; in numpy 2.x it stays `numpy.float32`, which PyTorch 2.4 CUDA tensors reject
on in-place assignment.

**Fix:**
```python
# Before
H, W = image_size_hw

# After
H, W = int(image_size_hw[0]), int(image_size_hw[1])
```

After fixing the source, VGGT-X was reinstalled (`pip install ./third_party/VGGT-X --no-deps`)
and the submodule pointer updated.

**Commits:** `4d550e5` (submodule pointer bump; fix is inside the submodule)

---

## 4. Test mocks returned 2-tuple; function now returns 3-tuple

**Affected file:** `tests/pointcloud/test_vggtx_creator.py`

**Error:**
```
ValueError: not enough values to unpack (expected 3, got 2)
  collab_splats/pointcloud/feedforward/vggtx.py:322
```

**Root cause:** `unproject_and_filter_points` was updated during the migration to
return a 3-tuple `(pts3d, colors, pixel_indices)` — the `pixel_indices` array is
needed for 2D→3D feature lifting. Two existing unit tests still mocked it as a
2-tuple `(pts, colors)`.

**Fix:** Added `pixel_indices` to both mock return values:
```python
pixel_indices = np.zeros((5, 3), dtype=np.int32)
# patch return_value=(pts, colors, pixel_indices)
```

**Commit:** `23941f7`

---

## 5. maskclip_onnx import crashed at module load

**Affected files:** `collab_splats/semantics/features.py`, `tests/conftest.py` (new)

**Error:**
```
ImportError: cannot import name 'packaging' from 'pkg_resources'
```
triggered when `import collab_splats.semantics` was collected by pytest.

**Root cause:** `maskclip_onnx/clip.py` does `from pkg_resources import packaging`.
setuptools ≥ 71 removed `pkg_resources.packaging` as a submodule (it's now only
available as standalone `packaging`). The import was at module top-level in
`features.py`, so every import of `collab_splats.semantics` failed.

**Fix 1 — lazy import in `features.py`:**
Moved `import maskclip_onnx` from module top-level into `MaskCLIPExtractor.__init__`.
This means the import error only surfaces when the extractor is instantiated (correct
behaviour), not on `import collab_splats.semantics`.

**Fix 2 — `tests/conftest.py` shim:**
Added a pytest session-start shim that injects standalone `packaging` into
`pkg_resources` before any test module is collected:
```python
import packaging, pkg_resources, sys
if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = packaging
    sys.modules["pkg_resources.packaging"] = packaging
    # also register submodules (version, requirements, markers, specifiers)
```
This makes `from pkg_resources import packaging` work during the test session
regardless of setuptools version.

**Commit:** `2eb76a1`

---

## New verification tests added

| File | Count | Coverage |
|---|---|---|
| `tests/test_cu121_migration.py` | 23 | Phase 1: env guards (Python 3.11, torch 2.4+cu121, CUDA 12.1, numpy ≥2.0, scipy, pycolmap ≥4, bae editable, viser ≥0.2.23). Phase 2: full collab_splats module import sweep + flagged numpy 2.x packages. Phase 3: gsplat-rade fork symbol, gtsam SL4 manifold, bae CUDSS symbols, nerfstudio local install, mapanything compat patch, pycolmap 4.x API, nerfstudio method configs, nerfstudio patched deps, splatfacto |
| `tests/integration/test_pipeline_cu121.py` | 6 | pycolmap 4.x reconstruction round-trip, FeedforwardResult npz save/load, VGGTXCreator._postprocess, MapAnythingCreator._postprocess, Open3DTSDFFusion.create, nerfstudio method registry |
| `tests/integration/__init__.py` | — | package marker |

All tests use `/opt/conda/envs/nerfstudio/bin/python`. Integration tests use
synthetic data only — no model downloads required.
