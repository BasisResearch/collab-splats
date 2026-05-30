# Known Test Failures — 2026-05-26

Run: `pytest tests/ -m 'not slow' --ignore=tests/test_cu121_migration.py --continue-on-collection-errors`
Result: **39 failed, 600 passed, 5 skipped, 2 collection errors**

Updated after: torch 2.4→2.5.1 upgrade, bae@0.2.4 git URL, nerfstudio BasisResearch fork via pyproject.

---

## Group 1: bae/pypose conflict — CLEARED 2026-05-26

**Previously:** pypose required `bae==0.2` exactly; bae 0.2.1 caused ImportError.

**Status:** pypose has no bae version constraint as of bae 0.2.4. Both import cleanly.
The 3 failures + 1 collection error from this group no longer occur.

---

## Group 2: loop_closure eval API mismatch (8 failures)

**Affected:**
- `tests/pointcloud/test_loop_closure_eval.py::test_apply_ba_dedup_aligns_intrinsics`
- `tests/pointcloud/test_loop_closure_integration.py::test_verify_loop_candidate_returns_tuple`
- `tests/pointcloud/test_loop_closure_integration.py::test_base_verify_raises_with_tuple_signature`
- `tests/pointcloud/test_feedforward_shared.py::test_verify_loop_candidate_rejected`
- `tests/pointcloud/test_feedforward_shared.py::test_verify_loop_candidate_accepted_no_poses`
- `tests/pointcloud/test_feedforward_shared.py::test_verify_loop_candidate_accepted_with_poses`
- `tests/pointcloud/test_feedforward_shared.py::test_verify_loop_candidate_layer_index_forwarded`
- `tests/pointcloud/test_feedforward_lc_state.py::test_lc_state_attrs_set_after_run_inference`

**Root cause:** Tests were written against an older LC verifier API. The current `_verify_loop_candidate` is a concrete method in `BaseFeedforwardCreator` that calls `extract_intermediate_features`. Tests expect tuple-return or raise behavior from an intermediate refactor that has since changed.

**Fix:** Update tests to match the current `_verify_loop_candidate` signature and return contract.

---

## Group 3: VGGTXCreator model_name default changed (1 failure)

**Affected:**
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_defaults`

**Error:**
```
AssertionError: assert 'facebook/VGGT-1B' == 'facebook/vggt'
```

**Fix:** Update `test_vggtx_defaults` line 12: `assert c.model_name == "facebook/VGGT-1B"`

---

## Group 4: VGGTXCreator depth tensor shape mismatch (2 failures)

**Affected:**
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_postprocess_calls_global_alignment`
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_no_global_alignment_when_disabled`

**Error:**
```
ValueError: cannot select an axis to squeeze out which has size not equal to one
  at vggt/utils/geometry.py:39: depth_map[frame_idx].squeeze(-1)
```

**Fix:** Check current vggt expected depth shape, update test fixtures accordingly.

---

## Group 5: _mapanything.run_mapanything attribute missing (1 failure)

**Affected:**
- `tests/pointcloud/test_mapanything_creator.py::test_mapanything_run_inference_passes_inference_params`

**Error:**
```
AttributeError: <module 'collab_splats.pointcloud._mapanything'> does not have the attribute 'run_mapanything'
```

**Fix:** Find actual function name with `grep -n "^def " collab_splats/pointcloud/_mapanything.py`, update patch target.

---

## Group 6: nerfstudio namespace shadow — tests/nerfstudio/ + tests/wrapper/ (20 failures + 2 collection errors)

**Affected:**
- `tests/nerfstudio/test_imports.py` — all 4 tests
- `tests/nerfstudio/test_datamanager_config.py` — all 11 tests
- `tests/wrapper/test_splatter_mesh.py` — all 6 tests
- `tests/test_models.py` — collection error
- `tests/wrapper/test_splatter_query.py` — collection error

**Errors:**
```
ModuleNotFoundError: No module named 'nerfstudio.cameras'
ModuleNotFoundError: No module named 'nerfstudio.configs'
ModuleNotFoundError: No module named 'nerfstudio.utils'
ModuleNotFoundError: No module named 'nerfstudio.data'
```

**Root cause:** `tests/nerfstudio/` is a Python namespace package (directory without `__init__.py`). With `pythonpath = ["."]` in pytest config, the repo root is on sys.path, which makes `tests/nerfstudio/` visible as a namespace package that shadows the installed `nerfstudio` site-package during collection/test execution. Submodules like `nerfstudio.cameras`, `nerfstudio.configs`, etc. don't exist inside `tests/nerfstudio/`, so imports fail.

The `tests/wrapper/` failures trace through `collab_splats/wrapper/splatter.py:18` → `from nerfstudio.utils.eval_utils import eval_setup` — same shadow.

**Fix:** Rename `tests/nerfstudio/` to `tests/nerfstudio_module/` (or similar) and update imports. OR add `tests/` to `norecursedirs` in pytest config (breaks test collection). Best fix: rename the directory.

---

## Group 7: test_pose_convention abstract interface mismatch (1 failure)

**Affected:**
- `tests/pointcloud/test_pose_convention.py::test_default_verifier_raises`

**Error:**
```
TypeError: Can't instantiate abstract class _DummyCreator with abstract methods _reproject, extract_intermediate_features
```
Then: test expects `NotImplementedError` matching `_verify_loop_candidate` but the base class now has a concrete implementation.

**Root cause:** `BaseFeedforwardCreator` gained two new abstract methods (`_reproject`, `extract_intermediate_features`) and `_verify_loop_candidate` was made concrete. The test's `_DummyCreator` stub doesn't implement the new abstract methods, and the test premise (verifier raises NotImplementedError) is no longer true.

**Fix:** Update `_DummyCreator` to add stubs for `_reproject` and `extract_intermediate_features`, then update the test to verify the new concrete `_verify_loop_candidate` behavior.

---

## Group 8: VGGTXCreator/MapAnythingCreator reconstruct smoke (2 failures)

**Affected:**
- `tests/pointcloud/test_vggtx_creator.py::test_vggtx_reconstruct_smoke`
- `tests/pointcloud/test_mapanything_creator.py::test_mapanything_reconstruct_smoke`

**Errors:**
```
ModuleNotFoundError: No module named 'nerfstudio.process_data'
  at collab_splats/pointcloud/base.py:119: from nerfstudio.process_data.colmap_utils import colmap_to_json
```
MapAnything: model attribute missing (Group 5 related).

**Root cause:** `nerfstudio.process_data` may have been reorganized in the BasisResearch fork, or the import in `base.py` is lazy (inside `_write_transforms`) and only triggers when `build_colmap` is called during a full reconstruct. The smoke tests run far enough to hit this code path.

**Fix:** Check if `nerfstudio.process_data.colmap_utils` exists in the fork; if not, find the new import path.

---

## Summary Table

| Group | Count | Status | Fix type |
|-------|-------|--------|----------|
| 1 bae/pypose | 0 | ✅ CLEARED 2026-05-26 | — |
| 2 LC eval API | 8 | active | Update tests for new verifier API |
| 3 VGGTXCreator model_name | 1 | active | 1-line test fix |
| 4 depth shape | 2 | active | Update test fixtures |
| 5 _mapanything attribute | 1 | active | Update patch target |
| 6 nerfstudio namespace shadow | 20 + 2 errors | active | Rename tests/nerfstudio/ |
| 7 pose_convention abstract | 1 | active | Update DummyCreator stubs + test premise |
| 8 reconstruct smoke | 2 | active | Fix nerfstudio.process_data import path |

**Migration hard gates:** `pytest tests/test_cu121_migration.py` → **24/24 PASS** ✅
