# Final docs + configs + CI cleanup — design

**Date:** 2026-10-08
**Branch:** `clean/final` (`39005639`)
**Status:** design approved in chat, spec under review

## Goal

Last pre-release sweep of `clean/final`: drop dead docs and configs, fix the stale parts of
the docs and configs that stay, and make every CI workflow pass without changing what it does.

## Findings that shape the design

- `docs/superpowers/` is in `.gitignore`, yet 242 files under it are force-tracked.
- All 6 configs validate and carry no dead keys. Stale parts are comments, `configs/README.md`,
  one unused overlay and one finished experiment grid.
- 8 of 14 tutorial notebooks import removed names. Tutorial-rework owns them; out of scope here.
- All 5 workflows pin Python 3.10 (`requires-python >= 3.11`). `test.yml` and
  `test_notebooks.yml` cannot run on hosted runners: four CUDA extensions build at install and
  `collab-data` is private.
- `make lint` today: ruff 38 errors plus format diffs, mypy 184 errors in 43 files.

## A. `docs/superpowers/` — keep 33 tracked, untrack the rest

`git rm --cached` only. Files stay on disk; `.gitignore` already intends this.

**Keep tracked:**

| Group | Files |
|---|---|
| Record | `CHANGELOG.md`; decisions 014, 015, 017–024 |
| In flight | `specs/2026-09-06-clean-final-integration-design.md`, `plans/2026-09-06-clean-final-integration.md`, `specs/2026-09-09-clean-final-dead-code-design.md`, `specs/2026-09-07-sky-segmentation-design.md`, `plans/2026-09-07-sky-segmentation.md`, `specs/2026-09-07-sky-mask-measured-report.md`, `specs/2026-09-25-vismatch-fork-design.md`, `specs/2026-09-25-vismatch-baseline.md`, `specs/2026-09-09-tutorial-rework-design.md`, `specs/2026-09-26-consistency-design.md`, `plans/2026-09-26-consistency-phase1.md`, `plans/2026-09-26-consistency-phase2.md`, `specs/2026-10-08-dashboard-merge-design.md`, `plans/2026-10-08-dashboard-merge.md`, `specs/2026-10-08-geometry-cleanup-design.md`, this spec |
| Cited from outside | `specs/2026-09-24-preproc-release-cleanup-design.md` (release-cleanup skill), `specs/2026-08-20-video-quality-report-measured.md` (`preproc/qa.py`, contract test), `specs/2026-09-26-pycolmap-cuda-docker-design.md` and `specs/2026-08-17-vismatch-local-matcher-design.md` (`pyproject.toml`), `specs/2026-08-23-splats-measured-report.md` (train_splats notebook), `specs/2026-07-09-lc-parity-probe-results.md` (`docs/parity.md`) |

**Untrack everything else** (210): completed and superseded specs/plans, all 11 handoffs, the
mesh audit, perf-1k (work dropped 2026-10-08), the 6 `plans/scaffold-runs/*.yaml`.

Accepted cost: ~90 CHANGELOG links stop resolving on a fresh clone; they resolve locally. Links
are left as written.

## B. Configs

- Delete `configs/loop_closure.yaml` (no code/test/setup caller) and `evals/configs/rgbd_ba.yaml`
  (finished rgbd-ba grid). Drop the overlay mention in `docs/parity.md`.
- Fix comments: `configs/base.yaml:2` ("all dataset configs inherit"), `evals/configs/7scenes.yaml:3`
  and `evals/README.md` (BA + loop closure is supported since 2026-10-06).
- Trim `configs/README.md`:
  - cut migration and history notes (L463-499, L511-514, L804-808, L852-889)
  - merge the two "Where outputs land" sections
  - add mesh to "Steps run per video"
  - fill missing key-table rows (semantics, pointcloud, splats, report keys)
  - correct L378: backend-block collisions fail at creator build, not in `validate_config`
  - move the loger/instantsfm/colmap/hloc sections to `docs/pointcloud.md`

## C. User docs

- Delete `docs/benchmarks/` (one-off 08-14 research, dead paths), `docs/known-test-failures.md`
  (resolved dev log, 5 missing tests), `docs/source/tutorials/02_pointcloud/colmap_sfm.ipynb`
  (stub) and its `tutorials/index.rst` entry.
- Fix `docs/README.md` links, `api/semantics.rst` (dead `semantics.retrieval`), add API pages for
  localization, splats and `feedforward.vggt_omega`, drop the duplicate `pypose` mock in `conf.py`.
- Delete untracked local residue: `docs/examples/`, `docs/_build/`, `__pycache__` under `docs/`.
- Untouched: the 8 broken notebooks (tutorial-rework).

## D. CLAUDE.md

- Drop in-flight items that are done or superseded: gt-eval-harness, feedforward-import-cleanup,
  feedforward-mesh, docs-site, bae-vggt-parity, loma-matcher.
- Refresh the stale vismatch-fork and sky-mask notes.
- Remove the `docs/known-test-failures.md` line.

## E. CI workflows — same jobs, made to pass

| Workflow | Change |
|---|---|
| `docs.yml` | Python 3.11 |
| `lint_notebooks.yml` | Python 3.11; fix whatever notebook lint it then reports |
| `lint.yml` | Python 3.11; make `make lint` green (below) |
| `test.yml`, `test_notebooks.yml` | `runs-on: [self-hosted, gpu]`; drop the pip/torch-2.1.2/numpy<2 install steps and run in the runner's existing env; Python 3.11 |

`make lint` green:
- `ruff check --fix`, then hand-fix the rest; one `ruff format` pass over `tests/` and `collab_splats/`
- `[tool.mypy]` in `pyproject.toml`: `ignore_missing_imports`, disable the noisiest error codes;
  fix the remaining real errors in code
- `scripts/test_notebooks.sh`: replace the stale exclude list with real paths

Known after this lands:
- The self-hosted runner must be registered by the user; until then the two GPU jobs queue.
- `test_notebooks.yml` stays red until tutorial-rework fixes the 8 notebooks.

## Order and gates

One commit per section (A, B, C, D, E; E may split into lint-fix and workflow commits).
After each:
- `git grep` for every removed path returns nothing outside `docs/superpowers/`
- `pytest tests/test_docstring_contract.py tests/test_import_style.py` green
- `python -m evals.eval --config evals/configs/{7scenes,cross_model_chess}.yaml --dry_run` exits 0
- E: `make lint` exits 0 locally; full `pytest tests/` count unchanged vs pre-E reference; the
  reformat commit is AST-equal per file
- C: `make docs` builds

Commits use `git commit --only <paths>` (shared index with concurrent sessions).
