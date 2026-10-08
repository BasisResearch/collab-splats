# Tests cleanup — handoff to the consistency agent

Date: 2026-10-08 · From: tests-audit session · Spec: [2026-10-08-tests-cleanup-design.md](2026-10-08-tests-cleanup-design.md) (`9f40c081`)

## State

- Audit done and spec approved by the user. **Scope A (conservative)**, approach 1: one commit per area
  on `clean/final`, with `git commit --only <paths>`.
- No implementation plan written yet, and nothing in `tests/` has changed.
- `clean/final` has moved since the audit (HEAD was `ff462ff6` at handoff). Every `file:line` in the
  spec is from the audit snapshot, so **re-locate each by test name, not line number**.
- If your consistency commits move or rename code, re-check the matching dead-target cuts.
- Uncommitted user WIP at handoff: `README.md` and `docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md`.
  Leave both alone.

## How the audit ran

- Five read-only subagents, split by area: geometry, pointcloud+evals+integration, splats+mesh,
  semantics+localization+preproc+utils, reconstructor+dashboard+root.
- They used `--collect-only` and greps only; no coverage runs.
- So the duplicate verdicts come from reading code. The spec's per-deletion gate (re-grep or compare
  against the named surviving test) is mandatory.
- Baseline: 4282 collected tests, ~150 files, 44.5k lines. Expected after A: about −115 tests, about −27 files.

## Verified by me (not just the auditors)

- **`test_pose_graph_incremental.py` is a tautology.** `tests/geometry/loop_closure/_helpers.py::drive_pose_graph`
  runs the same `PoseGraph.add_submap` + `optimize` loop the test repeats, and the "monolith" it
  originally compared against no longer exists.
- **The `gpu` mark is unregistered.** `pyproject.toml` `[tool.pytest.ini_options]` lists only `slow`, and
  `addopts = "--import-mode=importlib"`. As a result `test_mapanything_reconstruct_smoke` downloads and
  runs the real model on every full run.
  - If you put `-m "not gpu"` in addopts, any explicit `-m` on the command line replaces it.
- **`RELEASED` in `tests/test_docstring_contract.py` omits `pointcloud`.**
  - `pytest tests/test_docstring_contract.py -k pointcloud` gives 48 passed and 80 xpassed, with 0 real failures.
  - Adding it is a free one-line change.
- **`tests/wrapper`, `tests/nerfstudio_methods`, `tests/examples` contain only `__pycache__`.**
- `collab_splats/remote/` does not exist; only `remote.py` does.
- The two viewer test files are **not** duplicates:
  - `tests/test_viewer.py` tests `collab_splats/viewer.py` (viser).
  - `tests/dashboard/test_viewer.py` tests `dashboard/viewer.py` (PyVista).
- OCR lens and `utils/progress.py` are live (`f8336c03`), so their tests stay.

## Overlap with consistency work

These findings touch convention and dedup territory, so you may want to fold them into your phases:

- **Inline-import violations in test files.**
  - `tests/test_visualization.py:17-59`
  - `tests/semantics/test_insid3_segmentation.py:9-137` (one in every test)
  - `tests/test_bae_smoke.py`
  - `semantics/test_features_guards.py::test_maskclip_onnx_importable`
  - Merges lift imports to the top; the insid3 trim itself is deferred.
- **`tests/reconstructor/_stubs.py::_stub_reconstructor` hand-writes the `mesh` config.** It already
  broke once (`docs/known-test-failures.md`). Building it from `configs/base.yaml` is a deferred
  follow-up, but it is a natural dedup item.
- **`tests/conftest.py:14-40`:** the `pkg_resources.packaging` shim is the source of the deprecation
  warnings. Check whether `maskclip_onnx` still needs it.
- **`tests/pointcloud/sfm/test_instantsfm.py`** uses `importorskip` in every test; one at module level is enough.
- **Duplicated `test_preprocess_frames_match_files`** for three backends:
  - `test_vggtx_preproc.py:69`
  - `test_mapanything_creator.py:883`
  - `test_vggt_omega_creator.py:540`
  - These could become one parametrized test in `test_feedforward_shared.py`. That is scope B, so it's deferred.
- **Probably stale:** `docs/known-test-failures.md` still has the `remote` `STATS_ARGS` note, but
  `tests/remote/test_remote.py` collects cleanly now. It also references `tests/wrapper` and cu121
  paths that this cleanup removes.
- **Stale `__pycache__`** holds bytecode for deleted tests (`test_pgsr`, `test_verification`,
  `test_sim3_pose_graph`, …). The cache is gitignored; `rm` it in the touched dirs.

## Deferred (scope B/C — do not do unless the user widens scope)

The spec's "Non-goals" section lists them all:
- splats plant/branch-check tests
- keyword-only signature tests
- the `sfm/test_creator_args` collapse
- collapsing the per-file lint params (802 + 538)
- `strict=True`
- the insid3 trim and preproc overlaps
- the metrics AST/sync test
- the `metrics_controls` fold
- BA seed reduction

## User standing rules that apply

- Report before changes: post the findings and the proposed change, and get a go-ahead before
  editing or committing (memory `feedback_report_before_changes`).
- Use `git commit --only`, because other sessions share the index. `clean/final` moves fast, so land
  each commit promptly.
- Run `isort`/`black` on touched files only, never repo-wide.
- No legacy-file checks: delete schema-refusal and "stays gone" guards.
- Python is `/opt/venv/reconstruction/bin/python`. Run the full suite in tmux and log it; don't run
  parallel heavy jobs (OOM risk).
- Next step per the superpowers flow: write the implementation plan
  (`docs/superpowers/plans/2026-10-08-tests-cleanup.md`), then execute commit 1 → 7 as the spec orders them.
