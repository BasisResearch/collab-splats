# vismatch upstream baseline gate

**Date:** 2026-09-25
**Repo:** `/workspace/vismatch` (fork `BasisResearch/vismatch`), branch `main`
**SHA:** `9d49b892ed21625cee00bc7797a45d352bf659a7` (= `origin/main` = `upstream/main`, v1.3.2)
**Plan:** `docs/superpowers/plans/2026-09-25-vismatch-batch-forward.md` Task 3

## Environment

- venv `/opt/venv/vismatch`: Python 3.10.12, torch 2.14.0+cu130, CUDA on A40, ruff 0.16.9,
  pytest 9.1.1, pytest-timeout 2.4.0
- install: `pip install -e .` — `.[all]` FAILED (`nvcc not found` building torch-cluster), so
  sphereglue / omniglue / zippypoint deps are absent and skip consistently
- HF cache: `/workspace/models/hub`

## Result

- `ruff check .` exit 0, `ruff format --check .` exit 0
- `pytest tests -vv -rs --timeout=300` (tmux `vm-baseline`):
  **226 passed, 15 skipped, 0 failed, 0 errors** in 1678 s, EXIT 0
- log `/workspace/logs/vismatch-baseline.log`, junit `/workspace/logs/vismatch-baseline.xml`

## Skips

Deterministic (missing optional deps — expected every run):

| test line | models | reason |
|---|---|---|
| test_matchers.py:110, :128, :166 | omniglue, zippypoint | `No module named 'tensorflow'` |
| test_matchers.py:110, :128, :166 | sift-sphereglue, superpoint-sphereglue | `No module named 'torch_geometric'` |

Environment-flaky (instantiate test only, `test_matchers.py:110`) — may pass or skip on re-run:

| model | reason |
|---|---|
| duster | download from `download.europe.naverlabs.com` failed after 5 attempts |
| master | download from `download.europe.naverlabs.com` failed after 5 attempts |
| ufm | `Disk quota exceeded (os error 122)` writing the HF cache on `/workspace/models` |

## Gate rule for feature branches

- FAILED/ERROR set must equal baseline's (empty)
- skips: the 12 deterministic ones must match; duster/master/ufm may flip skip↔pass
- passed count = 226 + new tests ± the 3 flaky
