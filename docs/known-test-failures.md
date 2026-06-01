# Known Test Failures — 2026-06-01

Run (conda env `reconstruction`, py3.11):
```
/opt/conda/envs/reconstruction/bin/python -u -m pytest tests/ -m 'not slow' \
    --ignore=tests/test_cu121_migration.py --continue-on-collection-errors -q -rfE \
    -o faulthandler_timeout=120 -p no:cacheprovider
```
Result: **59 failed, 913 passed, 4 skipped, 4 deselected, 1 xpassed** in 8min. **Zero collection errors.**
Saved: `evals/results/baseline-conda-tests-0601.txt` (gitignored).

This supersedes the 2026-05-26 doc and the 2026-05-31 cu121-handoff snapshot
(`73 failed, 346 passed, 16 errors, 423 collected`). That snapshot was taken against a
**broken/incomplete env**, not the real baseline. Three things recovered ~570 tests since:

1. `panel`/`param`/`vggt`/`evo` installed by default (commits `60eec57`, feedforward setup) →
   dashboard/eval/feedforward modules now collect (no more 16 import errors).
2. `tests/nerfstudio/` → `tests/nerfstudio_methods/` rename (commit `3e4e821`) killed the
   namespace shadow (old Group 6) → `nerfstudio.*` submodule imports resolve.
3. **Infinite-hang fix** (see below) — the suite could not complete before this.

## The "11 hours at 7%" hang — FIXED 2026-06-01

The baseline appeared to run ~11h stuck at 7%. Not slow compute, not network: an **infinite loop
from an incomplete test mock**, unmasked by installing `panel`. `visualize.py:700` polls
`while proc.is_alive(): progress_queue.get(timeout=0.2)`. `test_on_run_mesh_spawns_subprocess_with_zarr_path`
and `test_on_run_mesh_reports_failure_on_nonzero_exit` mock `multiprocessing.Process` but never
stubbed `is_alive()` → `MagicMock.is_alive()` is always truthy → `while True`; the real
`progress_queue` is never fed → `.get()` raises `Empty` forever. Before `panel` was installed these
tests failed collection and never ran, so the hang was latent.

**Fix:** `proc_mock.is_alive.return_value = False` in both tests (`tests/dashboard/test_visualize.py`).
Both now pass in ~20s.

**Debug technique:** `python -u ... -o faulthandler_timeout=120` dumps the stuck main-thread frame
after the timeout, naming the exact test+line. (Torch `_inductor/.../subproc_pool._read_thread`
always shows idle-blocked in dumps — a red herring.)

---

## Failure groups (59)

Legend — **env-dep?**: does the fix belong in env/setup (matters for the uv migration) or is it
env-independent code/test drift (fails identically under conda and uv)?

| Grp | Count | Module(s) | Root cause | Fix type | Env-dep? |
|-----|-------|-----------|------------|----------|----------|
| A | 7 | `tests/scripts/test_reconstruct.py` | `SCRIPT_PATH` points to `scripts/reconstruct.py`; script moved to `docs/examples/reconstruct.py` (docstring already says so). | 1-line path fix | no |
| B | 5 | `tests/examples/test_run_c0043_pipeline.py` | Execs `examples/run_c0043_pipeline.py` — does not exist. **RETIRED 2026-06-01**: example deprecated; test file + empty `tests/examples/` package deleted. | done | no |
| C | 9 | `tests/webapp/*` | FastAPI/ASGI async tests; no async plugin was installed. **RESOLVED 2026-06-01**: app is FastAPI (not tornado — the `pytest-tornasync` hint was a red herring); `pytest-asyncio` → 9/9 pass. Added `pytest-asyncio`+`httpx` to `pyproject [dev]`. | done | **yes** (in `[dev]`) |
| D | 7 | `test_feedforward_shared`, `test_pose_convention`, `test_feedforward_lc_state`, `test_loop_closure_integration` | `BaseFeedforwardCreator` gained abstract methods (`_reproject`, `extract_intermediate_features`); stub creators (`_StubCreator`/`_DummyCreator`/`_D`) don't implement them → can't instantiate. Old Group 7, expanded. | Add stub methods; update premises | no |
| E | 5 | `tests/dashboard/test_visualize.py` | `ScenePanel` UI refactor removed/renamed `_points_options_*` rows + auto-mesh display; tests assert old attrs. | Update tests to new ScenePanel API | no |
| F | 3 | `tests/pointcloud/test_bundle_adjustment.py` | `RobustModel.forward()` missing a required positional arg — BA forward signature changed. | Update test calls (verify product intent) | no |
| G | 3 | `tests/wrapper/test_reconstructor.py`, `test_loop_closure_eval.py` | `BundleAdjustment` no longer importable from `collab_splats.pointcloud` (moved to `wrappers.py`); isinstance/identity checks on MagicMock. | Fix import path in tests | no |
| H | 3 | `tests/pointcloud/test_vggt_spark_native_similarity.py` | `.to()` called with an unsupported overload under torch 2.5 (`* (Tensor, bool non_blocking…)`). | Fix `.to()` call (product or test) | no |
| I | 2 | `tests/evals/test_metrics_auc.py` | `compute_auc()` got unexpected kwarg `align` — eval API drift. | Update test/signature | no |
| J | 2 | `tests/evals/test_eval_compare.py` | LC sim3 alignment metrics mismatch in compare emit. | Update expected metrics | no |
| Z | ~13 | mixed (`test_pgo_parity`, `test_hw_formula`, `test_graph`, `test_tsdf`, `test_mapanything_creator`, `test_pose_extraction`, `test_feedforward_logging`, `test_numpy_fix`, …) | Assorted singles: `PoseGraph.add_loop_edge` kwarg drift, sim3 submap overlap, meshlib `clean_repair` skip (effectively xfail), `_mapanything` attribute (old Group 5), `View … data_norm_type` key, numpy-warning assert. | Per-test, mostly mechanical | no |

**Key for the uv migration:** only **Group C is env-sensitive**. The other 50 failures are code/test
drift and will fail identically under conda and uv — they form the **baseline the uv env must match**
(uv must not *add* failures beyond these). Fix C in `setup/*.sh` so it survives the env rebuild.

## Suggested fix order

1. **C** — install async plugin, add to setup (only env-dep group; unblocks 9). Decide
   `pytest-tornasync` vs `pytest-asyncio` (the app is tornado → likely tornasync).
2. **A** — 1-line `SCRIPT_PATH` fix (7).
3. **B** — decide: write `examples/run_c0043_pipeline.py`, or skip the test if the example is
   deprecated (5).
4. **D, E, G** — mechanical test-API/import realignment (15).
5. **F, H, I, J, Z** — case-by-case; confirm whether each is test drift or a real product
   regression (≈23).

## Migration hard gates

`pytest tests/test_cu121_migration.py` → previously **24/24 PASS** (not re-run here; ignored via
`--ignore` in the baseline command).
