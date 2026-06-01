# cu121 Test Fixes — Design

**Date:** 2026-06-01
**Branch:** `refactor/cu121`
**Predecessor:** [cu121 test baseline handoff](2026-05-31-cu121-test-baseline-handoff.md)
**Unblocks:** [uv branch integration](2026-05-31-uv-branch-integration-design.md) Phase 0 gate

## Problem

The `reconstruction` conda env (only env present; py3.11) runs the suite far below the
documented baseline:

- **Now:** 422 collected · 16 collection errors · ~73 failed.
- **Documented** (`docs/known-test-failures.md`, 2026-05-26): ~644 collected · 2 collection
  errors · 39 failed.
- ~220 tests no longer collect — whole modules error on import and vanish, which also drops
  the passed count (600 → 346).

Investigation found **two independent root causes** plus a residual band of genuine
test-vs-code drift. The handoff's single "missing deps" hypothesis is correct but incomplete.

## Root causes (verified)

### Cause 1 — incomplete env build (14 of 16 collection errors)

Import probes in `reconstruction`: `torch, nerfstudio, pypose, bae` OK;
`vggt, evo, panel, param, mapanything, gtsam, salad` **missing**.

| Collection errors | Missing dep | Declared in | Installed by | Why missing |
|---|---|---|---|---|
| 10 × `tests/dashboard/*` | panel, param | pyproject `[dashboard]` | **nothing** | `[dashboard]` extra is orphaned — no setup script installs it |
| `evals/test_metrics_auc` | evo | pyproject `[feedforward]` + `setup/feedforward.sh` | `setup/feedforward.sh` | script never run in this env |
| `mesh/test_adapter`, `pointcloud` (agg), `test_feedforward_logging` | vggt | `third_party/VGGT-X` submodule + `setup/feedforward.sh` | `setup/feedforward.sh` | same |

Two distinct gaps:

- **vggt / evo / mapanything / gtsam / salad** — `setup/feedforward.sh` installs them, but
  `setup.sh` (default install) never calls it. CLAUDE.md treats feedforward as a separate
  manual step. This env skipped it.
- **panel / param** — declared only as the pyproject `[dashboard]` extra; **no setup script
  installs them at all**. A perfect clean rebuild still leaves them missing.

Beyond the 14 collection errors, this is the dominant effect: the ~220 vanished tests are
mostly downstream of these missing imports. **Most of the ~73 reported failures cannot be
triaged until deps are restored**, because installing them changes the collected set.

### Cause 2 — `tests/nerfstudio/` namespace shadow (2 collection errors + ~20 downstream failures)

`tests/test_models.py` and `tests/wrapper/test_splatter_query.py` **collect clean in
isolation** but error in the full run with `ModuleNotFoundError: No module named
'nerfstudio.data' / 'nerfstudio.utils'`. Dep-independent (nerfstudio is installed and imports
fine).

Mechanism — sharper than `known-test-failures.md` Group 6 states. Config already sets
`--import-mode=importlib` and `tests/nerfstudio/__init__.py` exists (added 2026-05-27). That
`__init__.py` is the **culprit, not a fix**: in importlib mode pytest derives a module's
top-level package by walking up through `__init__.py` files. `tests/nerfstudio/` has one,
`tests/` does not → the derived name for `tests/nerfstudio/test_datamanager_config.py` becomes
`nerfstudio.test_datamanager_config`. Pytest inserts `tests/` on `sys.path` and registers
`tests/nerfstudio` as `nerfstudio` in `sys.modules`, shadowing the real site-package for the
rest of the session. Every later `import nerfstudio.<submodule>` then fails.

Confirmed: running `tests/nerfstudio/ tests/test_models.py tests/wrapper/test_splatter_query.py`
together reproduces 2 collection errors; running the two victims alone passes collection.

### Residual — test-vs-code API drift (~50 failures, count unconfirmed)

In modules that **do** collect: `known-test-failures.md` Groups 2,3,4,5,7,8 (LC verifier API,
VGGTX `model_name` default, depth tensor shape, `_mapanything` attribute, `pose_convention`
abstract stubs, reconstruct-smoke `nerfstudio.process_data` import) plus handoff-reported
`evals/test_eval_gt_helpers` (14), `scripts/test_reconstruct` (7), `examples` (5),
`integration/test_pipeline_cu121` (3), `webapp/*` (9). Genuine drift between tests and current
code. **Counts are provisional** — installing deps (Cause 1) changes which modules collect, so
the residual set must be re-measured before triage.

## Solution — staged

Ordered so each stage's effect is measurable before the next.

### Stage 1 — make the env build complete by default

Per user decision: feedforward + dashboard deps install by default.

- **`setup.sh`**: after Step 3, add a step that runs `setup/feedforward.sh`, and a step that
  installs dashboard deps (`pip install -e '.[dashboard]'`). Keep `setup/feedforward.sh` itself
  callable standalone (no behavior change to it).
- Result: `vggt, evo, mapanything, gtsam, salad, panel, param` all present after a default
  `bash setup.sh`. Unlocks 14 collection errors and restores the ~220 vanished tests.
- Out of scope (deferred per user): restructuring the extras / `[dashboard]` vs `[feedforward]`
  split, and the uv migration. This stage only closes the install gaps in place.

### Stage 2 — kill the namespace shadow

- Rename `tests/nerfstudio/` → `tests/nerfstudio_methods/` (or similar non-`nerfstudio` name)
  and remove the misleading `__init__.py` so pytest cannot derive a top-level `nerfstudio`.
- Update any intra-suite references to the old path.
- Unlocks 2 collection errors + the ~20 Group-6 downstream failures.
- Correct the `known-test-failures.md` Group-6 note: root cause is the `__init__.py`-driven
  top-level derivation under importlib mode, not merely the directory name.

### Stage 3 — re-baseline

- Re-run `pytest tests/ -m 'not slow' --ignore=tests/test_cu121_migration.py
  --continue-on-collection-errors -q -rfE` in the fixed env.
- Record new collected / passed / failed / errors. Confirm collection errors → 0 (or only
  intentional).

### Stage 4 — triage residual drift

- Diff the re-baselined failures against `known-test-failures.md` Groups 2,3,4,5,7,8.
- Apply the documented per-group test fixes (mostly updating tests to current API/signatures).
- Investigate handoff-reported bands (`eval_gt_helpers`, `scripts/test_reconstruct`,
  `examples`, `integration`, `webapp`) that are not yet grouped; classify each as test drift vs
  real regression.

### Stage 5 — update `known-test-failures.md`

- Rewrite to the new reality: date, counts, surviving groups, and the corrected Group-6 cause.
- This becomes the refreshed Phase-0 baseline the uv integration resumes from.

## Success criteria

- Default `bash setup.sh` yields an env where `vggt, evo, panel, param` all import.
- 0 collection errors (or only deliberately ignored modules).
- Collected count back near the documented ~644 (±, accounting for tests added/removed since).
- Remaining failures are an explained set, all recorded in `known-test-failures.md`.

## Out of scope

- uv package-manager migration (resumes after this work; the uv worktree at `51e5a93` is
  untouched).
- Extras/dependency-structure refactor (user will address separately).
- Fixing genuine product regressions surfaced in Stage 4 beyond classifying them — those, if
  any, become their own tasks.

---

# Addendum — Stages 3–5 executed (2026-06-01, later session)

Stages 1–2 landed (commits `60eec57` deps-by-default, `3e4e821` `tests/nerfstudio_methods` rename).
This addendum records the re-baseline (Stage 3), residual triage (Stage 4), and the decisions made,
so the uv integration has a concrete target.

## Stage 3 — re-baseline result

`59 failed, 913 passed, 4 skipped, 1 xpassed`, **0 collection errors**, 8 min
(`evals/results/baseline-conda-tests-0601.txt`). Collected ~980 vs the broken-env 423 — the env
fixes + namespace rename worked. One blocker had to be cleared first:

**Infinite-hang fix ("11h at 7%").** Not deps, not network. `visualize.py:700` runs
`while proc.is_alive(): progress_queue.get(timeout=0.2)`. Two mesh-worker tests mocked
`multiprocessing.Process` but never stubbed `is_alive()` → `MagicMock` truthy → `while True`; the
real `progress_queue` is never fed → `.get()` raises `Empty` forever. Installing `panel` (Stage 1)
*unmasked* it — before, those dashboard tests failed collection and never ran. Fix:
`proc_mock.is_alive.return_value = False` in both (`tests/dashboard/test_visualize.py`). Debug via
`python -u … -o faulthandler_timeout=120` (dumps the stuck frame; torch inductor
`subproc_pool._read_thread` in the dump is a red herring).

## Decisions (Stage 4)

- **Group B — RETIRE (done).** `examples/run_c0043_pipeline.py` does not exist anywhere; the example
  is deprecated. Deleted `tests/examples/test_run_c0043_pipeline.py` + the now-empty
  `tests/examples/` package.
- **Group C — pytest-asyncio (decided empirically).** Webapp is FastAPI/ASGI; tests use
  `@pytest.mark.asyncio` + `httpx.AsyncClient`/`ASGITransport`. Installed `pytest-asyncio` →
  **9/9 webapp pass in 3.27s**. The pytest "pytest-tornasync" hint is a red herring (no tornado).
  Added `pytest-asyncio` + `httpx` to `pyproject.toml [dev]`. The uv venv installs `[dev]`, so **no
  `setup/*.sh` change is needed** — this is the only env-dependent group.
- **No product regressions.** All 59 are test-API drift, stale script paths, or missing test deps —
  fixes are test-side or 1-line product import moves, never behavior changes.

## Per-group fix plan (remaining 50 after B+C)

UPDATE = realign test to current API · DELETE = premise removed by design · all env-independent
unless noted.

| Grp | N | Action | Detail |
|-----|---|--------|--------|
| A `test_reconstruct` | 7 | UPDATE | Repoint `SCRIPT_PATH` `scripts/reconstruct.py` → `docs/examples/reconstruct.py` (1 line; docstring already says so). |
| D abstract stubs | 5 | UPDATE | Add no-op `extract_intermediate_features` + `_reproject` to `_StubCreator` (`test_feedforward_shared`) and the `test_feedforward_lc_state` stub. |
| D verifier-raises | 2 | **DELETE** | `test_pose_convention::test_default_verifier_raises`, `test_loop_closure_integration::test_base_verify_raises_with_tuple_signature` — assert the base *raises NotImplementedError*; that contract was removed (`base.py:853` is concrete → `tuple[bool, ndarray\|None]`). Deprecated. |
| E ScenePanel UI | 5 | UPDATE | Realign `test_visualize` to the refactored ScenePanel (renamed/removed `_points_options_*` rows, auto-mesh display). |
| F BA forward | 3 | UPDATE | `test_bundle_adjustment` calls `RobustModel.forward()` at old arity; update calls (product unchanged). |
| G BundleAdjustment import | 3 | UPDATE | `BundleAdjustment` now in `wrappers.py`; fix import in `test_reconstructor` + `test_loop_closure_eval`. |
| H torch `.to()` | 3 | UPDATE | `test_vggt_spark_native_similarity` mock frames' `.dtype` breaks `.to(p.device, dtype=p.dtype)` under torch 2.5; pass real tensors. |
| I `compute_auc(align=)` | 2 | UPDATE | `compute_auc` now always Sim3-aligns internally; drop the removed `align=` kwarg. The align-mode split is obsolete — keep one positive assertion. |
| J sim3 compare | 2 | UPDATE | `test_eval_compare` expected-metrics realignment to the Sim3 default. |
| Z misc singles | ~13 | UPDATE/triage | `PoseGraph.add_loop_edge` kwarg, pgo/hw_formula/graph submap math, `_mapanything` attribute (old G5), `View … data_norm_type` key, numpy-warning assert, meshlib `clean_repair` skip (leave as effective xfail). Case-by-case as touched. |

**Deprecation proposal — DELETE (7 total):** `test_run_c0043_pipeline.py` (done, 5),
`test_default_verifier_raises` (1), `test_base_verify_raises_with_tuple_signature` (1). All other
failures are realigned, not removed.

## Retire-hunt (broader sweep, beyond the failing set)

Swept all tests for: external-file loaders, skip/xfail scaffolding, `raises(NotImplementedError)`
patterns, and orphaned modules. **No additional tests to delete** — the 7 above are the complete
retire set. Zero collection errors means no test imports a dead module. Three **stale markers** found
(not deletes — they recover passing tests):

| Test | Marker | Status | Action |
|------|--------|--------|--------|
| `test_reconstruct_pane.py::test_build_creator_vggtx_returns_correct_type` | `@skip` (xfeat not importable) | **Stale** — `from collab_splats.pointcloud.feedforward import VGGTXCreator` now succeeds in-env | Remove skip; re-enable |
| `test_reconstruct_pane.py::test_build_creator_mapanything_returns_correct_type` | `@skip` (same) | **Stale** — same | Remove skip; re-enable |
| `test_numpy_fix.py::test_collab_splats_init_no_torch` | `@xfail` ("splatter.py top-level import torch; needs lazy import fix") | **Stale** — now XPASSes (the 1 xpassed in the baseline); lazy-import fix landed | Remove xfail marker |

Kept (correctly marked, not retired): `test_mapanything_creator.py::test_mapanything_run_inference_loop_closure_smoke`
(`xfail strict` tracks a real LC+MapAnything dict-vs-list[dict] bug), `test_segmentation.py`
NotImplementedError (SAM genuinely has no text-prompt support), and the env-gated
`ffmpeg`/`evo_ape`/`pypose`/`cuda` skips.

## Env-dependence for uv

Only Group C is env-sensitive, now expressed in `pyproject [dev]` (uv inherits it). The other ~48
are code/test drift, identical across conda and uv. **uv target:** after these land, the conda suite
is the reference; uv passes iff it adds no new failures beyond accepted residue (meshlib skip).

## Verification

Re-run the baseline command; expect failures ≤ accepted residue, 0 collection errors, no hangs;
refresh `known-test-failures.md` + `evals/results/baseline-conda-tests-0601.txt`.
