# Known Test Failures — 2026-05-08

Run: `pytest tests/pointcloud/ --ignore=tests/pointcloud/test_sim3_pose_graph.py`
Result: **9 failed, 173 passed, 1 skipped, 1 xfailed**

Plus 1 collection error that blocks the entire `test_sim3_pose_graph.py` file.

All failures are pre-existing — none introduced by the LC/BA refactor (refactor/core-modules branch).

---

## Group 1: bae==0.2.1 / pypose version conflict (3 failures + 1 collection error)

**Affected:**
- `tests/pointcloud/test_sim3_pose_graph.py` — entire file, collection error
- `tests/pointcloud/test_bundle_adjustment.py::test_run_bundle_adjustment_early_exit_shape`
- `tests/pointcloud/test_bundle_adjustment.py::test_run_bundle_adjustment_no_reproj_filter`

**Error:**
```
ImportError: PyPose requires bae==0.2 when the optional backend is installed, but found bae==0.2.1.
Recommend running: pip install git+https://github.com/sair-lab/bae.git@0.2
```

**Root cause:** `bae` was upgraded to 0.2.1 in the conda env, but `pypose` pins to exactly `bae==0.2`. `test_sim3_pose_graph.py` imports `pypose` at module level (line 4), blocking collection. `bundle_adjustment.py:133` imports `pypose` inside `run_bundle_adjustment()`, so BA tests fail at runtime.

**Fix options:**
1. (Recommended) Downgrade bae: `pip install git+https://github.com/sair-lab/bae.git@0.2`
2. Add `pytest.importorskip("pypose")` guard at the top of `test_sim3_pose_graph.py` and `test_bundle_adjustment.py` so the tests skip gracefully instead of erroring
3. Pin `bae==0.2` in `setup.py`/`requirements.txt`

---

## Group 2: loop_closure eval API mismatch (3 failures)

**Affected:**
- `tests/pointcloud/test_loop_closure_eval.py::test_umeyama_align_raises_not_implemented`
- `tests/pointcloud/test_loop_closure_eval.py::test_ate_translation_raises_not_implemented`
- `tests/pointcloud/test_loop_closure_eval.py::test_rpe_raises_not_implemented`

**Errors:**
```
# umeyama_align and ate tests:
Failed: DID NOT RAISE <class 'NotImplementedError'>
# match="GT pose dataset"

# rpe test:
ValueError: delta=1 >= N=1, no pose pairs available
# at collab_splats/pointcloud/loop_closure/eval.py:191
```

**Root cause:** Tests expect `umeyama_align()`, `ate()`, and `rpe()` to raise `NotImplementedError("GT pose dataset")` when called without GT ground-truth data. The actual implementation has changed:
- `umeyama_align` and `ate` no longer raise `NotImplementedError` — they either succeed or raise something else
- `rpe` raises `ValueError("delta=1 >= N=1, no pose pairs available")` instead of `NotImplementedError`

Either the functions were implemented (removing the stub `NotImplementedError`) or the tests were written against a planned API that was implemented differently.

**Fix options:**
1. Update tests to match current behavior (remove `raises(NotImplementedError)`, test actual return values)
2. If these are genuinely stubs, restore `raise NotImplementedError("GT pose dataset")` in the eval functions

Inspect `collab_splats/pointcloud/loop_closure/eval.py` around the `umeyama_align`, `ate`, and `rpe` function bodies to determine which.

---

## Group 3: VGGTXCreator model_name default changed (1 failure)

**Affected:**
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_defaults`

**Error:**
```
AssertionError: assert 'facebook/VGGT-1B' == 'facebook/vggt'
  - facebook/vggt
  + facebook/VGGT-1B
```

**Root cause:** `VGGTXCreator.model_name` default was updated from `"facebook/vggt"` to `"facebook/VGGT-1B"` (the correct HuggingFace repo ID) but `test_vggtx_defaults` at line 12 still asserts `"facebook/vggt"`.

**Fix:** Update test line 12:
```python
assert c.model_name == "facebook/VGGT-1B"
```

---

## Group 4: VGGTXCreator depth tensor shape mismatch in vggt geometry (2 failures)

**Affected:**
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_postprocess_calls_global_alignment`
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_no_global_alignment_when_disabled`

**Error:**
```
ValueError: cannot select an axis to squeeze out which has size not equal to one
  at vggt/utils/geometry.py:39: depth_map[frame_idx].squeeze(-1)
  called from collab_splats/pointcloud/_vggt.py:51: unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
```

**Root cause:** `unproject_depth_map_to_point_map` calls `depth_map[frame_idx].squeeze(-1)`, which requires the last dimension of each frame's depth to be size 1. The test fixtures provide depth with shape `(N, H, W, 1)` — but something about the shape passed to the function doesn't match. Either:
- The `vggt` package updated `geometry.py` to expect a different depth layout (e.g. `(N, H, W)` without trailing dim)
- The test fixture provides depth in the wrong shape for the current vggt version

**Fix:** Check current `vggt` version's expected depth input shape for `unproject_depth_map_to_point_map`, then update the test fixtures in `test_vggtx_creator.py` to match. The fix is in the test, not the source (the actual pipeline produces correctly-shaped depth).

---

## Group 5: _mapanything.run_mapanything attribute missing (1 failure)

**Affected:**
- `tests/pointcloud/test_mapanything_creator.py::test_mapanything_run_inference_passes_inference_params`

**Error:**
```
AttributeError: <module 'collab_splats.pointcloud._mapanything'> does not have the attribute 'run_mapanything'
  at: patch("collab_splats.pointcloud._mapanything.run_mapanything", ...)
```

**Root cause:** The test patches `collab_splats.pointcloud._mapanything.run_mapanything`, but that function doesn't exist at that path. Either it was renamed, moved, or never existed under that name.

**Fix:** Run `grep -n "^def " collab_splats/pointcloud/_mapanything.py` to find the actual function name, then update the patch target in the test. Alternatively check if the inference entry point is in a different module (e.g. `MapAnythingModel` class method rather than a module-level function).

---

## Summary Table

| Group | Count | Fix type |
|-------|-------|----------|
| bae/pypose version | 3 + 1 collection | Environment: downgrade bae to 0.2 |
| loop_closure eval API | 3 | Code: update tests or restore stubs |
| VGGTXCreator model_name | 1 | Test: update expected string |
| depth shape mismatch | 2 | Test: update fixture shapes |
| _mapanything attribute | 1 | Test: update patch target |

**Easiest first:** Group 3 (1-line fix), Group 5 (find real function name + 1-line fix), Group 1 (pip downgrade).
