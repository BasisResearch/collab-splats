# cu121 Notebook Verification — Additional Fixes

**Date:** 2026-05-22
**Branch:** refactor/cu121
**Context:** Follow-up to `2026-05-22-cu121-verification-fixes.md`. These fixes were
found during end-to-end notebook + training verification, after the original 565-test
pass.

---

## 6. Kernelspec update — all notebooks

**Affected:** 93 notebooks across main + `docs-site`, `bae-parity`,
`model-free-query-mesh` worktrees.

**Symptom:** All notebooks had `language_info.version: 3.10.18`. Wrong kernel
selected when opening in cu121 env (Python 3.11).

**Fix:** `scripts/update_kernelspecs.py` — walks all `.ipynb` files, rewrites
`metadata.kernelspec` and `metadata.language_info.version` to Python 3.11.

**Commit:** main branch + docs-site worktree

---

## 7. setuptools pinned at <70.0 in runtime deps

**Affected file:** `pyproject.toml`

**Symptom:** `ImportError: cannot import name 'packaging' from 'pkg_resources'`
when importing `maskclip_onnx`. setuptools 70.3.0 was installed (env drifted above
the `[build-system]` constraint `setuptools<70.0`).

**Root cause:** setuptools 70+ dropped `pkg_resources.packaging` as a submodule.
`maskclip_onnx/clip.py` uses `from pkg_resources import packaging` at module level.

**Fix:**
- Downgraded env: `pip install "setuptools==69.5.1"`
- Added `setuptools<70.0` to `[project.dependencies]` with comment explaining why
- `[build-system]` comment updated

**Commit:** `fix(deps): pin setuptools<70.0, downgrade env to 69.5.1`

---

## 8. torchpack missing from cu121 env

**Affected file:** `pyproject.toml` (runtime dep added)

**Error:**
```
ModuleNotFoundError: No module named 'torchpack'
  mobilesamv2/efficientvit/apps/trainer/base.py:9: import torchpack.distributed as dist
```
triggered at `ns-train rade-features` startup (datamanager loads mobilesamv2 for
SAM segmentation).

**Root cause:** `mobilesamv2` (loaded via torch.hub at
`/workspace/models/hub/RogerQi_MobileSAMV2_main/`) imports `torchpack.distributed`
at module level via its efficientvit trainer submodule. `torchpack` was present in
the cu118 env but not carried over to cu121.

**Fix:**
- `pip install torchpack` (installs `torchpack==0.3.1`)
- Added `"torchpack"` to `[project.dependencies]` in `pyproject.toml` with comment

**Commit:** `fix(deps): add torchpack runtime dependency`

---

## 9. pil_to_numpy broken by Pillow 12.x

**Affected file:** `nerfstudio/nerfstudio/data/utils/data_utils.py`

**Error:**
```
TypeError: function takes exactly 2 arguments (1 given)
  data_utils.py:41: e.setimage(im.im)
```
triggered at nerfstudio dataset init (every training run, every image load).

**Root cause:** Pillow 12.2.0 changed `ImagingCore.setimage` to require 2 arguments
(image + extents region). The old custom encoder-based path in `pil_to_numpy` used
internal Pillow API that was stable through Pillow 9.x but broken in 12.x.

**Fix:** Replaced entire encoder path with `np.array(im)`:
```python
def pil_to_numpy(im: PILImage) -> np.ndarray:
    im.load()
    return np.array(im)
```
`np.array(im)` handles all PIL modes (L, RGB, RGBA, F, I) with correct dtype.
Simpler, faster, future-proof.

**Commit:** `fix(data): replace pil_to_numpy encoder path with np.array(im)`
(in `/workspace/nerfstudio` repo)

---

## 10. feedforward_exploration.ipynb — VGGT cache check

**Affected file:** `docs/pointcloud/feedforward_exploration.ipynb`

**Symptom:** Cell 7 (VGGTXCreator inference) always re-ran even when
`RESULT_PATH` already existed. MapAnything cell 12 had the cache check pattern;
VGGT cell 7 did not.

**Fix:** Added `if RESULT_PATH.exists(): load else: run inference` pattern to
cell 7, matching cell 12.

**Commit:** `fix(notebooks): add VGGT-X result cache check in feedforward_exploration`

---

## Verification results

| Check | Result |
|---|---|
| keyframe_extraction.ipynb | ✅ 9/9 cells, 0 errors |
| feedforward_exploration.ipynb | ✅ 12/12 cells, 0 errors |
| hloc CLI (20 bicycle images) | ✅ 20/20 matched, transforms.json produced |
| ns-train rade-features (GPU) | ✅ Step 10 at 268ms/iter, 43% GPU, 3932 MiB |
| derive_splats.ipynb | pending (waiting for checkpoint) |
| visualization.ipynb | pending (waiting for checkpoint) |
| create_mesh.ipynb | pending (waiting for checkpoint) |
| unit tests (565) | pending re-run after pil_to_numpy fix |
