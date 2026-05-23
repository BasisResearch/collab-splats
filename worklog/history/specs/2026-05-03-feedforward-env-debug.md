# Feedforward Env Debug — Investigation Brief for Next Agent

**Date:** 2026-05-03
**Branch:** `refactor/core-modules`
**Triggered by:** end-to-end run of `docs/pointcloud/loop_closure_eval.ipynb` on bicycle scene

## Goal

Get the feedforward + loop-closure pipeline running end-to-end in the nerfstudio conda env (`/opt/conda/envs/nerfstudio/bin/python`, py3.10) on the bicycle scene at `/workspace/bicycle/images_4` (194 images).

The loop_closure_eval notebook is the test case but the env issues affect ANY caller of `MapAnythingCreator` or `VGGTXCreator` with `enable_loop_closure=True`.

## Constraint That Drives Everything

`nerfstudio` is built against CUDA 11.8, `torch==2.1.2+cu118`, `numpy==1.26.4`. These cannot change — `gsplat-rade` CUDA kernels are compiled against this exact stack. Any "just upgrade torch" fix is off the table.

This rules out swapping in newer mapanything / vggt-x versions that depend on torch ≥ 2.4 or numpy ≥ 2.0.

## What Works

- `47/47` unit tests in `tests/pointcloud/test_loop_closure_eval.py`, `test_closure_split.py`, `test_feedforward_lc_state.py`, `test_loop_closure.py`, `test_loop_closure_integration.py`, `test_pose_graph.py` pass.
- `setup_feedforward.sh` runs to completion (after timm fix below).
- `from mapanything.models import MapAnything` imports successfully.
- `from collab_splats.pointcloud.feedforward import VGGTXCreator` imports + `VGGTXCreator().load_model()` succeeds (~87s on cuda).
- Notebook parses as valid Jupyter JSON, all imports succeed.

## Bug 1: timm 0.6.7 too old (FIXED)

**Symptom:**
```
File ".../uniception/models/libs/perception_encoder/vision_encoder/pe.py", line 14
    from timm.layers import DropPath
ModuleNotFoundError: No module named 'timm.layers'
```

**Cause:** `setup_feedforward.sh` header comment says "timm stays at 0.6.7" but `uniception==0.1.7` (mapanything dep) imports `timm.layers` which only exists in `timm>=0.9`.

**Fix (applied at runtime, NOT in script yet):**
```
pip install 'timm>=1.0'
```
Currently installs `timm==1.0.26`. `nerfstudio` declares `timm==0.6.7` but never imports it at runtime — the pip-check warning is cosmetic (per existing script comment).

**Action needed:** add `pip install 'timm>=1.0' -q` to `setup_feedforward.sh` and update the header comment. Old memory at `/root/.claude/projects/-workspace-collab-splats/memory/project_feedforward_install.md` correctly noted timm needed upgrade — script regression.

## Bug 2: MapAnything calls torch.Tensor.any(dim=tuple) (BLOCKING)

**Status:** RESOLVED 2026-05-03 — fixed in `collab_splats/pointcloud/feedforward.py:_patch_mapanything_torch_compat` (called from `MapAnythingCreator._load_model`). See follow-up spec [`2026-05-03-mapanything-torch-compat.md`](2026-05-03-mapanything-torch-compat.md) for design and sunset plan.

**Symptom (during `creator.run_inference()` on bicycle):**
```
File ".../mapanything/utils/wai/intersection_check.py:254"
    batch_intersect = batch_intersect.any(dim=(1, 3))
TypeError: any() received an invalid combination of arguments - got (dim=tuple, ),
  but expected one of:
   * (int dim, bool keepdim)
```

**Cause:** Multi-dim `tensor.any(dim=tuple)` was added in PyTorch 2.4. Env has 2.1.2.

**File:** `/opt/conda/envs/nerfstudio/lib/python3.10/site-packages/mapanything/utils/wai/intersection_check.py:254`

**Workaround applied:** notebook default backend switched from `mapanything` to `vggtx` (cell 5 `BACKEND = "vggtx"`).

**Possible fixes for next agent:**
- Pin mapanything to an older release that doesn't use multi-dim `any()` — check `pip index versions mapanything` and grep older release for `intersection_check.py`.
- Monkeypatch the offending line at module-load time (intrusive but contained):
  ```python
  # Replace with: batch_intersect.any(dim=1).any(dim=-1)  # equivalent semantics
  ```
- Upstream PR to mapanything for torch <2.4 compat (long-term).

Either-or: if MapAnything backend is not required for this work, document the limitation and proceed with VGGT-X only.

## Bug 3: SALAD `x, t = x` unpacking fails (RESOLVED 2026-05-06)

**Symptom (during `ImageRetrieval` loop closure on VGGTX path):**
```
File ".../vendor/salad/models/aggregators/salad.py:120"  # approx line; verify
    x, t = x  # Extract features and token
ValueError: too many values to unpack (expected 2)
```

**Cause hypothesis:** SALAD's aggregator expects DINOv2 backbone to return `(features, token)` 2-tuple. Current DINOv2 (or whatever extractor SALAD uses) returns 3+ elements. Likely backbone version mismatch.

**File:** `/workspace/collab-splats/vendor/salad/` (cloned by setup_feedforward.sh from `https://github.com/serizba/salad.git`)

**Triggered when:** `enable_loop_closure=True` causes `ImageRetrieval(device=device)` instantiation (`feedforward.py:_run_loop_closure_inference`), which eventually calls SALAD aggregator on DINOv2 features.

**Investigation steps for next agent:**
1. Find the exact line — `grep -n 'x, t = x' /workspace/collab-splats/vendor/salad/models/aggregators/*.py`
2. Print `type(x)` and `len(x) if hasattr(x, '__len__') else 'scalar'` just before that line
3. Check `vendor/salad/main.py` or its `requirements.txt` for declared DINOv2 backbone version
4. Compare with what's actually installed: `pip show torch` for vit/dinov2, or check `vendor/salad/dinov2/` if vendored
5. Either pin DINOv2 backbone to SALAD's expected version, or patch salad to accept 3-tuple by indexing `x = x[0]; t = x[1]`

**Possible fixes:**
- Pin SALAD to specific commit known to work with installed DINOv2
- Patch SALAD aggregator to handle modern DINOv2 return shape (small, contained)
- Substitute different image-retrieval backend (would require touching `ImageRetrieval` in `collab_splats/pointcloud/loop_closure/retrieval.py`)

## Files to Read

- `setup_feedforward.sh` — install contract; needs timm fix
- `constraints_feedforward.txt` — pinned versions; do NOT modify torch/numpy
- `collab_splats/pointcloud/feedforward.py:_run_loop_closure_inference` (line ~184) — entry point that triggers all 3 bugs in sequence
- `collab_splats/pointcloud/feedforward.py:_load_model` (MapAnything line ~606, VGGTX similar) — backend model loaders
- `collab_splats/pointcloud/loop_closure/retrieval.py` — `ImageRetrieval` class that wraps SALAD
- `vendor/salad/` — vendored SALAD aggregator source
- `docs/pointcloud/loop_closure_eval.ipynb` — test notebook (BACKEND=vggtx default after this debug session)
- `worklog/history/specs/2026-04-22-feedforward-install-design.md` — original install design

## Reproducer Commands

```bash
# Verify env state (should all succeed once bugs are fixed):
/opt/conda/envs/nerfstudio/bin/python -c "import timm; assert timm.__version__ >= '1.0', timm.__version__"
/opt/conda/envs/nerfstudio/bin/python -c "from mapanything.models import MapAnything"
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.pointcloud.feedforward import VGGTXCreator; c = VGGTXCreator(); c.load_model()"

# Reproduce SALAD bug:
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import VGGTXCreator
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
from pathlib import Path
c = VGGTXCreator(camera_model='SIMPLE_PINHOLE', enable_loop_closure=True,
                 loop_closure_config=LoopClosureConfig())
c.load_model()
c.setup_inference(Path('/workspace/bicycle/images_4'))
c.run_inference()  # Bug 3 triggers here
"

# Reproduce MapAnything bug (after timm fix):
# Same as above but VGGTXCreator → MapAnythingCreator(camera_model='PINHOLE', ...)

# E2E notebook check:
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
  --to notebook --execute docs/pointcloud/loop_closure_eval.ipynb \
  --output loop_closure_eval.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
```

## Acceptance Criteria

- `setup_feedforward.sh` runs clean and includes timm upgrade
- `nbconvert` of `loop_closure_eval.ipynb` exits 0 with all cells executing successfully (PyVista cells 17-18 may fail in headless and that's OK)
- Cell 11 prints non-zero loss reduction percentage
- Cell 13 shows monotonically descending log-scale loss curve
- All 47 existing unit tests still pass

## What NOT to Do

- Do NOT upgrade `torch`, `torchvision`, `numpy`, `cuda` — breaks gsplat-rade kernels
- Do NOT delete `vendor/salad` or move LC away from DINO-SALAD without ADR
- Do NOT add `enable_loop_closure=False` as a workaround in the notebook — defeats the entire purpose of the eval
- Do NOT modify `nerfstudio` pyproject.toml

## Out of Scope

- General LC algorithm improvements (separate work)
- C0043 hloc as pseudo-GT (planned for after env stable)
- Real GT pose dataset (KITTI / outdoor)
