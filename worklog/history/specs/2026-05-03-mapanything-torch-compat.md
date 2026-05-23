# MapAnything torch ≤2.3 Compatibility — Investigation Brief for Next Agent

**Date:** 2026-05-03
**Branch:** `refactor/core-modules` (or feature branch off it)
**Predecessor work:** `2026-05-03-feedforward-env-debug.md` — Bug 2 was punted there; this brief is the follow-up.

## Goal

Make `MapAnythingCreator.run_inference()` succeed in the nerfstudio conda env (`/opt/conda/envs/nerfstudio/bin/python`, py3.10, torch==2.1.2+cu118) so that `enable_loop_closure=True` works on both backends, not just VGGT-X.

## Constraint That Drives Everything

`nerfstudio` is built against CUDA 11.8, `torch==2.1.2+cu118`, `numpy==1.26.4`. These cannot change — `gsplat-rade` CUDA kernels are compiled against this exact stack. "Just upgrade torch" is off the table.

## What Already Works

- VGGTX path (`BACKEND='vggtx'`) runs end-to-end with loop closure on `/workspace/bicycle/images_4` (`docs/pointcloud/loop_closure_eval.ipynb`).
- 48 unit tests in `tests/pointcloud/` pass.
- `setup_feedforward.sh` is clean: timm≥1.0 pinned, MapAnything import works, `MapAnything.from_pretrained` loads weights.
- `DinoSaladExtractor` (`collab_splats/semantics/retrieval.py`) is correct — DINOv2 backbone + SALAD aggregator + pretrained checkpoint. MapAnything path can reuse it as-is.

## The Bug

**Symptom (during `MapAnythingCreator.run_inference()`):**
```
File ".../mapanything/utils/wai/intersection_check.py:254"
    batch_intersect = batch_intersect.any(dim=(1, 3))
TypeError: any() received an invalid combination of arguments - got (dim=tuple, ),
  but expected one of:
   * (int dim, bool keepdim)
```

**Cause:** `tensor.any(dim=tuple)` (multi-dim reduction) was added in PyTorch 2.4. The pinned env has 2.1.2.

**File:** `/opt/conda/envs/nerfstudio/lib/python3.10/site-packages/mapanything/utils/wai/intersection_check.py:254`

**Surrounding context** (relevant block, lines ~248–254):
```python
# Reshape and check if any triangle pair intersects
batch_intersect = batch_intersect.reshape(chunk_i_size, T, chunk_j_size, T)
batch_intersect = batch_intersect.any(dim=(1, 3))
```

**Equivalent torch ≤2.3 expression:**
```python
batch_intersect = batch_intersect.any(dim=3).any(dim=1)
```
Order matters — reduce the higher-indexed dim first so the surviving indices don't shift.

## What Has Been Ruled Out

- **Older mapanything pin:** `pip index versions mapanything` shows only `1.1`. No usable older release.
- **Forking the package and uploading to PyPI:** out of scope for this branch.
- **Disabling MapAnything entirely:** orthogonal to the env-debug deliverable. Already documented as a soft punt in `feedforward.py:MapAnythingCreator._load_model` and `setup_feedforward.sh`.

## Recommended Fix: In-Tree Runtime Monkey-Patch

Targeted, contained, no install changes, no fork to maintain. Patch the offending function at `MapAnythingCreator._load_model` time, before the model is constructed.

### Implementation sketch

Place a helper at module scope in `collab_splats/pointcloud/feedforward.py`, near the top of the file (after imports, before class definitions):

```python
def _patch_mapanything_torch_compat() -> None:
    """Replace mapanything's torch>=2.4 multi-dim any() with sequential calls.

    mapanything==1.1's intersection_check uses ``tensor.any(dim=(1, 3))`` which
    requires PyTorch 2.4+. The nerfstudio env is pinned to 2.1.2+cu118 (gsplat-rade
    CUDA kernels). We rewrite the function in place at import time. Idempotent.

    TEMPORARY BRIDGE — see "Sunset Path" in
    worklog/history/specs/2026-05-03-mapanything-torch-compat.md.
    Delete this helper once the env moves to torch>=2.4.
    """
    import inspect
    import textwrap
    import mapanything.utils.wai.intersection_check as ic

    target = ic.frustum_intersection_check
    if getattr(target, "_torch_compat_patched", False):
        return

    src = textwrap.dedent(inspect.getsource(target))
    patched_src = src.replace(
        "batch_intersect.any(dim=(1, 3))",
        "batch_intersect.any(dim=3).any(dim=1)",
    )
    if patched_src == src:
        # Upstream changed — be loud rather than silently leaving torch>=2.4 code in place.
        raise RuntimeError(
            "mapanything intersection_check.py no longer matches expected pattern "
            "for torch<2.4 compat patch. Re-inspect and update _patch_mapanything_torch_compat."
        )

    ns: dict = {}
    exec(patched_src, ic.__dict__, ns)
    patched = ns["frustum_intersection_check"]
    patched._torch_compat_patched = True
    ic.frustum_intersection_check = patched
```

Call from `MapAnythingCreator._load_model`:

```python
def _load_model(self, device: str) -> Any:
    try:
        from mapanything.models import MapAnything
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e
    _patch_mapanything_torch_compat()  # in-tree workaround for torch<2.4
    model = MapAnything.from_pretrained(self.model_name)
    model = model.to(device)
    model.eval()
    return model
```

Replace the existing multi-line punt comment in `_load_model` with a one-line pointer to `_patch_mapanything_torch_compat`.

### Why this shape

- **Idempotent:** safe to call from multiple creators or tests.
- **Loud on drift:** if upstream renames or moves the line, the `RuntimeError` flags it instead of silently leaving the bug in place.
- **No global side effects until first MapAnything use:** import-only paths (e.g. unit tests that don't touch MapAnything) are unaffected.
- **No PYTHONPATH or sys.modules surgery:** uses normal attribute replacement on the module.

## Files to Modify

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward.py` | Add `_patch_mapanything_torch_compat`; call from `MapAnythingCreator._load_model`; drop the existing punt comment |
| `setup_feedforward.sh` | Remove or rephrase the "MapAnything will fail on this torch" note in the header block (it now works in-tree) |
| `worklog/history/specs/2026-05-03-feedforward-env-debug.md` | Mark Bug 2 as RESOLVED with link to this brief |
| `docs/pointcloud/loop_closure_eval.ipynb` | Optional: parametrize cell 5 to run both backends, or add a second eval block for `mapanything` |
| `tests/pointcloud/test_mapanything.py` | New file: covers patched intersection check |

## Tests to Add

1. **Unit-level patch test** (`tests/pointcloud/test_mapanything.py`):
   ```python
   import torch

   def test_patch_mapanything_torch_compat_replaces_multi_dim_any():
       from collab_splats.pointcloud.feedforward import _patch_mapanything_torch_compat
       _patch_mapanything_torch_compat()

       import inspect
       import mapanything.utils.wai.intersection_check as ic
       src = inspect.getsource(ic.frustum_intersection_check)
       assert "any(dim=(1, 3))" not in src
       assert "any(dim=3).any(dim=1)" in src
       # Idempotency
       _patch_mapanything_torch_compat()
       _patch_mapanything_torch_compat()


   def test_patched_frustum_intersection_runs_on_dummy_input():
       from collab_splats.pointcloud.feedforward import _patch_mapanything_torch_compat
       _patch_mapanything_torch_compat()
       import mapanything.utils.wai.intersection_check as ic

       # Construct a small valid input — frustum_triangles shape (B, T, 3, 3)
       # See ic.frustum_intersection_check signature; pick smallest plausible B, T.
       B, T = 2, 2
       triangles = torch.randn(B, T, 3, 3)
       out = ic.frustum_intersection_check(triangles)  # may need extra args, check signature
       assert out.shape == (B, B)
       assert out.dtype == torch.bool
   ```
   Adjust the second test once you read the actual signature — `frustum_intersection_check(frustum_triangles, chunk_size=...)` is the likely shape, but verify.

2. **Smoke test** mirroring the VGGT-X reproducer:
   ```bash
   /opt/conda/envs/nerfstudio/bin/python -c "
   from pathlib import Path
   from collab_splats.pointcloud.feedforward import MapAnythingCreator
   from collab_splats.pointcloud.loop_closure import LoopClosureConfig
   c = MapAnythingCreator(camera_model='PINHOLE', enable_loop_closure=True,
                          loop_closure_config=LoopClosureConfig())
   c.load_model()
   c.setup_inference(Path('/workspace/bicycle/images_4'))
   c.run_inference()
   print('MA_LC_INFERENCE_OK')
   "
   ```

3. **Existing unit suite** must still pass:
   ```
   /opt/conda/envs/nerfstudio/bin/pytest tests/pointcloud -q
   ```

## Acceptance Criteria

- `MapAnythingCreator.run_inference()` exits 0 on bicycle scene with `enable_loop_closure=False` (verified by `test_mapanything_run_inference_smoke`).
- The new patch unit tests pass (added to `tests/pointcloud/test_mapanything_creator.py`).
- All previously passing pointcloud tests still pass.
- All retrieval tests still pass.
- VGGTX path still passes its existing reproducer (no regression in the patch helper).
- `enable_loop_closure=True` is **not** in this spec's acceptance — see "Out of Scope" below.

## Reproducer Commands (current failure mode)

```bash
# 1. Confirm the bug still presents before patching
/opt/conda/envs/nerfstudio/bin/python -c "
from pathlib import Path
from collab_splats.pointcloud.feedforward import MapAnythingCreator
c = MapAnythingCreator(camera_model='PINHOLE')
c.load_model()
c.setup_inference(Path('/workspace/bicycle/images_4'))
c.run_inference()  # → TypeError on intersection_check.py:254
"

# 2. Inspect the offending source
/opt/conda/envs/nerfstudio/bin/python -c "
import inspect, mapanything.utils.wai.intersection_check as ic
print(inspect.getsource(ic.frustum_intersection_check))
"
```

## What NOT to Do

- Do **not** upgrade `torch`, `torchvision`, `numpy`, or `cuda` — breaks `gsplat-rade` kernels.
- Do **not** edit the installed `intersection_check.py` directly — the patch must live in our repo so a fresh `pip install` of mapanything stays compatible.
- Do **not** wrap `_patch_mapanything_torch_compat` in a `try/except: pass` — let upstream drift error out so we notice when the patch needs updating.
- Do **not** introduce a sys-wide import-time hook (e.g. via `sitecustomize.py` or an entrypoint) — keep the patch scoped to `MapAnythingCreator._load_model` so it only fires when MapAnything is actually used.
- Do **not** disable `enable_loop_closure` to dodge the bug — the eval notebook needs LC on both backends.

## Related References

- `worklog/history/specs/2026-04-22-feedforward-install-design.md` — original install design.
- `worklog/history/specs/2026-05-03-feedforward-env-debug.md` — predecessor brief; Bug 1 (timm) and Bug 3 (DinoSaladExtractor) are resolved there.
- `collab_splats/pointcloud/feedforward.py:MapAnythingCreator._load_model` — current punt site.
- `collab_splats/semantics/retrieval.py:DinoSaladExtractor` — already correct, reused unchanged for MA path.
- `vendor/salad/` — DINO-SALAD vendor source (not relevant to this fix but loaded by `enable_loop_closure=True`).

## Sunset Path / Eventual Goal

This monkey-patch is a bridge, not a permanent fix. The real solution is moving the nerfstudio env to `torch>=2.4`, at which point `tensor.any(dim=tuple)` works natively and the patch becomes dead code.

**Trigger to remove the patch:**
- `gsplat-rade` CUDA kernels rebuild cleanly against `torch>=2.4`, **or**
- the project migrates off `gsplat-rade` to a renderer that doesn't pin torch.

**Removal steps when the trigger fires:**
1. Confirm `torch.tensor([...]).any(dim=(1, 3))` works on the new env.
2. Delete `_patch_mapanything_torch_compat` and its call site in `MapAnythingCreator._load_model`.
3. Delete `tests/pointcloud/test_mapanything.py::test_patch_mapanything_torch_compat_replaces_multi_dim_any` (the dummy-input run test can stay as a regression check on the upstream function).
4. Add a follow-up note in this spec or close out a tracking issue marking the bridge retired.

**Pointer to keep in code:** the docstring of `_patch_mapanything_torch_compat` should reference this section so future readers know the helper is intentionally temporary.

## Out of Scope

- Forking mapanything to a basis-research repo.
- Submitting an upstream PR (parallel track; can be done later).
- Replacing MapAnything with a different feedforward backbone.
- Optimizing MapAnything inference time / memory.
- Upgrading torch to ≥2.4 in this branch (tracked separately under Sunset Path).
- **LC + MapAnything end-to-end:** `_run_loop_closure_inference` in `collab_splats/pointcloud/feedforward.py` reads `raw["extrinsic"]` from a single dict, but `MapAnythingCreator._forward` returns `list[dict]` (one per frame in the window). With the torch-compat patch in place, calling `MapAnythingCreator.run_inference()` with `enable_loop_closure=True` reaches this site and raises `TypeError: list indices must be integers or slices, not str` at `feedforward.py:239`. This is a pre-existing integration bug unrelated to the torch-compat fix and has its own follow-up spec. Captured in code as `tests/pointcloud/test_mapanything_creator.py::test_mapanything_run_inference_loop_closure_smoke` marked `pytest.mark.xfail(strict=True)`.
