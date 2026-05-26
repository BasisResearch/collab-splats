# Test Validation — cu121 + pyproject Upgrade Design

**Date:** 2026-05-26
**Status:** Approved
**Branch:** refactor/cu121

## Problem

`pyproject.toml` and `Dockerfile` were updated to:
- Upgrade torch 2.4 → 2.5.1 (CUDA 12.1)
- Install bae via git URL at 0.2.4 (was 0.2.1 loose pin)
- Install nerfstudio via BasisResearch fork git URL (was local `/workspace/nerfstudio` editable)
- Collapse setup.sh to single `pip install -e .`

These changes introduce at least two broken tests in `test_cu121_migration.py` that were correct for the old env but are now stale. Goal: establish a clean baseline — no failures beyond `worklog/known-test-failures.md` — before merging `refactor/cu121`.

## Success Criteria

- All `tests/test_cu121_migration.py` tests pass (hard gates per file docstring)
- `pytest tests/ -m 'not slow'` produces no failures outside `worklog/known-test-failures.md`
- `worklog/known-test-failures.md` updated to reflect cleared or newly-categorized failures
- Zero collection errors (import errors at module level block entire files)

## Out of Scope

- Fixing the 10 pre-existing failures in `worklog/known-test-failures.md` (Groups 1–5)
- Docker build validation (fresh container build)
- Feedforward / slow tests (`-m slow`)

## Design

### Stage 0 — Pre-fix known upgrade breakages

Two tests in `tests/test_cu121_migration.py` will fail due to the upgrade. Fix these before running anything else.

**`test_bae_cuda_backend`**

```python
# Before (asserts torch 2.4)
assert torch.__version__.startswith("2.4"), f"Wrong torch: {torch.__version__}"

# After (asserts torch 2.5.x)
assert torch.__version__.startswith("2.5"), f"Wrong torch: {torch.__version__}"
```

**`test_nerfstudio_installed_local`**

This test asserted nerfstudio loads from `/workspace/nerfstudio`. Nerfstudio is now installed via the BasisResearch git URL in pyproject, so it lives in site-packages, not `/workspace`. The test needs to verify the BasisResearch fork is installed (not PyPI nerfstudio) — check for a fork-specific symbol (`rasterization_2dgs_inria_wrapper` in gsplat is already covered; for nerfstudio itself, verify the patched dep versions hold via `test_nerfstudio_patched_deps`).

Fix: drop the path assertion and replace with a check that nerfstudio loads from site-packages (not from a stale `/workspace/nerfstudio` that may linger).

```python
def test_nerfstudio_installed_local():
    """nerfstudio loads from reconstruction conda env site-packages (BasisResearch fork)."""
    import nerfstudio.field_components.activations as _ns_probe
    ns_file = _ns_probe.__file__
    assert ns_file is not None
    assert "/opt/conda/envs/reconstruction" in ns_file, (
        f"nerfstudio loaded from unexpected location: {ns_file}. "
        "Expected site-packages under /opt/conda/envs/reconstruction."
    )
```

**pypose/bae conflict check**

Before running Stage 2, verify whether pypose still pins `bae==0.2` exactly:

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import importlib.metadata as m
req = [r for r in m.requires('pypose') or [] if 'bae' in r]
print(req)
"
```

If pypose still requires `bae==0.2`, Group 1 failures remain pre-existing. If pypose was updated to accept `>=0.2`, Group 1 clears — remove from known-failures.md.

### Stage 1 — Import sweep

Verify every submodule imports cleanly (catches broken `__init__` or missing transitive deps):

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import collab_splats
import collab_splats.pointcloud
import collab_splats.pointcloud.feedforward
import collab_splats.pointcloud.bundle_adjustment
import collab_splats.pointcloud.loop_closure
import collab_splats.semantics
import collab_splats.semantics.features
import collab_splats.mesh
import collab_splats.nerfstudio
import collab_splats.utils
print('import sweep OK')
"
```

Gate: zero `ImportError` / `ModuleNotFoundError`. Any failure = fix before Stage 2.

### Stage 2 — Fast unit test run + delta report

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -m 'not slow' --tb=short -q 2>&1 | tee /tmp/pytest_results.txt
```

Triage output against `worklog/known-test-failures.md`:

| Category | Action |
|---|---|
| Failure in known-failures.md | Expected — no action |
| Failure NOT in known-failures.md | **Fix it** |
| Known failure that now passes | Remove from known-failures.md |

### Stage 3 — Migration hard gates

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/test_cu121_migration.py -v
```

All tests must pass — these are merge-blocking per the file's own docstring. No failures allowed.

## Files Changed

| File | Change |
|---|---|
| `tests/test_cu121_migration.py` | Fix `test_bae_cuda_backend` (2.4→2.5), fix `test_nerfstudio_installed_local` (path assertion) |
| `worklog/known-test-failures.md` | Update: remove cleared failures, add new ones if found |

## Failure Triage Reference

Pre-existing failures from `worklog/known-test-failures.md` (2026-05-08). These are NOT regressions:

| Group | Tests | Root cause |
|---|---|---|
| 1 | test_sim3_pose_graph (collection), test_bundle_adjustment (2x) | bae/pypose version conflict — may clear with bae@0.2.4 |
| 2 | loop_closure eval API mismatch (3x) | LC eval API changed |
| 3 | VGGTXCreator model_name default (1x) | default changed upstream |
| 4 | VGGTXCreator depth shape (2x) | vggt geometry.py API change |
| 5 | _mapanything.run_mapanything missing (1x) | attribute removed upstream |
