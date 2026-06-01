# Known Test Failures — 2026-06-01 (GREEN)

Run (conda env `reconstruction`, py3.11, **numpy 2.1.3**):
```
/opt/conda/envs/reconstruction/bin/python -u -m pytest tests/ -m 'not slow' \
    --ignore=tests/test_cu121_migration.py --continue-on-collection-errors -q -rfE \
    -o faulthandler_timeout=120 -p no:cacheprovider
```
Result: **964 passed, 0 failed, 2 skipped, 4 deselected, 3 xfailed**, 0 collection errors, 8 min.
Migration hard gate: `pytest tests/test_cu121_migration.py` → **24/24 PASS**.
Baseline saved: `evals/results/baseline-conda-tests-0601.txt` (gitignored).

The suite is green. This supersedes the 59-failure baseline earlier on 2026-06-01 and the
broken-env handoff snapshot (`73f/346p/16err`). Everything below is the record of how it got here.

## What was fixed (59 failed → 0)

Three structural unblocks (deps-by-default, `tests/nerfstudio_methods` rename, the hang fix) plus
~20 commits of test-API realignment. Highlights:

- **The "11h at 7%" hang** — `tests/dashboard/test_visualize.py` mesh-worker tests mocked
  `multiprocessing.Process` but never stubbed `is_alive()` → `while proc.is_alive()` spun forever.
  Unmasked once `panel` was installed (before, those tests failed collection). Fixed with
  `proc_mock.is_alive.return_value = False`.
- **Group C (webapp async)** — app is FastAPI/ASGI; added `pytest-asyncio` + `httpx` to
  `pyproject [dev]`. (Not tornado — the pytest "pytest-tornasync" hint was a red herring.)
- **Two real product bugs found + handled** (the "no regressions" assumption was wrong):
  - `docs/examples/reconstruct.py` `_REPO_ROOT` was `.parent.parent` after the move from
    `scripts/`, pointing `--config_dir` at the nonexistent `docs/configs`. **Fixed** to
    `.parent.parent.parent` (commit `1d331b6`).
  - BA `test_optimize_*` (CUDA+bae only) hit a real bae/pypose bug: bae's `LM.step` calls pypose
    `RobustModel.forward(input, target)` with only `input` → `missing 'target'`. **Deferred**:
    xfail(strict=False) with a follow-up reason (`a36d1d7`). Tests are correct; production needs a
    dedicated BA fix.
- **Retired (deprecated):** `tests/examples/test_run_c0043_pipeline.py` (script gone),
  `test_default_verifier_raises` + `test_base_verify_raises_with_tuple_signature` (base verifier is
  now concrete). The BA-dedup test was **rewritten** (not retired) against the relocated
  `LoopClosure.run() _dedup_rows` path.
- **Stale markers cleared:** 2 xfeat `@skip` + 1 lazy-torch `@xfail` (now pass).

## Residue (intentional, not failures)

- **3 xfailed** — BA `test_optimize_reduces_reproj_error`, `_captures_loss_history_when_flag_set`,
  `_no_loss_history_by_default`: real bae/pypose `RobustModel.forward(target)` bug, deferred. Only
  run when CUDA+bae present.
- **2 skipped** — environment-gated (e.g. ffmpeg / evo_ape CLI not installed).
- **1 flagged docstring** (no test impact) — `loop_closure/eval.py:18` `_classify_edges` still claims
  "Loop edges use Robust(Huber)"; post-`011c56f` loop and sequential edges use identical Gaussian
  noise so the function can no longer separate them. Instrumentation degraded by design; fixing
  needs a production change to track edge provenance. Owner's call.

## numpy version note (env, not a code bug)

The migration gate enforces `numpy>=2` (also required by rerun-sdk). The env had drifted to
`numpy 1.26.4` — a stray with-deps install of `vggt-omega` (which pins `numpy<2`). The documented
`setup/feedforward.sh:57` already installs vggt-omega with `--no-deps` to avoid exactly this, so a
clean build per the setup script is fine; the drift came from outside it. Restored to `numpy 2.1.3`
(vggt-omega imports and runs fine on 2.x — its `<2` pin is conservative). The uv migration inherits
the `--no-deps` install, so this won't recur.

## For the uv migration

Only Group C is env-sensitive and it lives in `pyproject [dev]` (uv installs it). Everything else is
env-independent test/code state. **Target:** the uv env passes iff it reproduces this baseline —
964 passed, the 3 BA xfails, the 2 env skips — and adds no new failures. Ensure the uv build keeps
`numpy>=2` (the `--no-deps` vggt-omega install already does).
