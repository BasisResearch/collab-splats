# clean/final dead-code removal — design

**Date:** 2026-09-09
**Branch:** `clean/final` (`1e7ccbe7`)
**Status:** design approved, plan not yet written

## Goal

Remove deprecated and outdated code from `clean/final`: stale tests, orphaned residue,
vendored `third_party/` clones that pip already provides, and the parked
`geometry/global_alignment.py`.

Then make the surviving tests match how the package is organised **today**. Passing is not
the same as applying: a test can reference live symbols and still sit in a directory that
stopped describing its subject three refactors ago. Part 5 covers that.

Out of scope by explicit decision: `ColmapCreator` and `HlocCreator` stay. They are unwired
into `_run_sfm` but are kept as working, unit-tested backends.

## What the measurement actually said

The request assumed broad test rot. It is not there. Every number below was measured on
`clean/final` at `1e7ccbe7` with `/opt/venv/reconstruction/bin/python`.

| Probe | Result |
|---|---|
| `pytest --collect-only` | 2566 tests, **zero collection errors** |
| Broken-reference sweep over test patch/mock target strings | 131 distinct `collab_splats.*` targets, **129 resolve** |
| The 2 that do not | both benign — `collab_splats.dummy_bridge` is a `logging.getLogger` name, `...maskclip.maskclip_onnx` is a mock patch target |
| Declared deps absent from the venv | **2** — `lomatch`, `vismatch` |
| Current failures | **17**, all in `tests/localization/test_local_matcher.py`, all from the 2 missing packages |
| `tests/remote` + `tests/wrapper` | 344 passed, 0 failed |

The suite is healthy. The debris is narrow and named, not diffuse.

Two prior notes are now stale and are corrected here: the recorded gate of
"19 failed / 2503 passed" is really 17 failed from one cause, and the "2 scene-id failures"
no longer reproduce.

## Why the obvious method does not work

The intuitive approach — delete the dead code, run the suite, remove whatever fails — is
sound as **verification** and wrong as **discovery**. Four reasons:

- **Deleting a clone yields skips, not failures.** The guards exist for exactly that.
  `tests/pointcloud/feedforward/test_loger_load_guard.py` is written to keep the registry
  importable "in a bare checkout with no `third_party/LoGeR`", and `test_registry.py:25`
  records that it is deliberately "NOT guarded on `third_party/LoGeR` being present".
  A prior fresh-tree measurement produced +6 skips and zero new failures, exit 0. Dead
  tests hide rather than announce.
- **The baseline is dirty.** 17 tests already fail for a purely environmental reason.
  Under "remove what fails" those 17 legitimate tests get deleted.
- **Failures propagate across files.** A `torch.no_grad()` leaking process-wide at a
  `yield` has previously made unrelated files fail here. One deletion produces a blast
  radius, and innocent tests fall inside it.
- **It is blind to vacuous tests.** A test that asserts nothing keeps passing after its
  subject is deleted.

Hence the ordering below: repair the environment first, discover statically, delete, then
use the run to confirm the static read.

## Part 1 — Baseline repair

Precedes every deletion. Nothing else in this design is trustworthy without it.

```bash
/root/.local/bin/uv pip install --python /opt/venv/reconstruction/bin/python \
    vismatch==1.3.1 lomatch
```

Both are already declared in `pyproject.toml` and resolved in `uv.lock`
(`uv.lock:1250`, `uv.lock:1205`). The venv has drifted from the lock — the known trap is
that a plain `uv sync` prunes packages installed outside it. This is environment repair,
not a dependency change; no manifest edit is required.

Record the resulting pass/skip/fail triple verbatim. That triple is the reference every
later step is diffed against.

**Exit condition:** 0 failures across the full suite.

## Part 2 — Test cleanup

### 2a. Orphan residue

104 `__pycache__` entries whose `.py` sources were deleted long ago, including a fully
empty `tests/nerfstudio_methods/` package directory. Representative:
`tests/pointcloud/test_sfm_creator.py` (the `sfm.py` split), `tests/preproc/test_frame_store.py`
(frames.zarr reverted to `images/` + `frames.json`), `tests/evals/test_run_vggt_long.py`,
`tests/nerfstudio_methods/test_imports.py`.

These are not tests and cannot affect a run. They are removed because they are the reason
the tree reads as rotten, and because a stale `.pyc` beside a live package is a
long-standing source of confusion.

Remove every `__pycache__` directory under `tests/`, then remove any now-empty package
directory. Verify `tests/nerfstudio_methods/` is the only directory that becomes empty.

### 2b. Migration-era gates

The first draft of this section proposed deleting three files outright. Enumerating their 31
tests proved that wrong for two of them. Recorded here because the error is instructive: a
file whose **title** names a finished migration can still hold the only coverage of a live
invariant, and nothing about the filename says so.

| File | Tests | Disposition |
|---|---|---|
| `tests/test_cu121_migration.py` | 22 | **Split, do not delete.** |
| `tests/integration/test_pipeline_cu121.py` | 6 | **Rename and relocate, do not delete.** |
| `tests/test_bae_smoke.py` | 3 | **Delete.** |

#### `test_cu121_migration.py` — split

Roughly half its tests are live invariants unrelated to the migration:

- `test_import_all_modules` — every `collab_splats` submodule must import. Package-wide
  import health; nothing else in the suite covers it.
- `test_the_module_list_covers_every_splats_module` — a self-guarding meta-test. Its own
  comment records it catching a real gap: "`collab_splats.splats.utils` was absent for two
  plans while every entry present imported cleanly."
- `test_gsplat_commit_constant_matches_pyproject` — asserts `collab_splats.splats.GSPLAT_COMMIT`
  equals the `[tool.uv.sources]` gsplat rev. Exactly the manifest-drift guard this design
  argues for elsewhere.
- `test_pycolmap_api_surface` — constructs cameras, images and rigs against the real
  pycolmap API.
- `test_gtsam_sl4_manifold`, `test_bae_use_cudss`, `test_bae_cuda_backend`,
  `test_bae_cudss_importable`, `test_collab_data_installed`.

Rename the file to `tests/test_environment_contract.py`, rewrite the module docstring to
describe a standing environment and manifest contract rather than a migration gate, and keep
those tests. The version-pin tests (`test_python_version`, `test_torch_version`,
`test_cuda_version`, `test_numpy_not_downgraded`, `test_scipy_version`,
`test_pycolmap_version`, `test_bae_installed_version`, `test_viser_version`,
`test_timm_version`, `test_flagged_package_imports`) also stay — they are the contract the
new filename names — but their docstrings lose the migration framing.

Delete exactly one test: **`test_mapanything_compat_patch_safe`**. Its docstring says
"`_patch_mapanything_torch_compat` must not raise RuntimeError at import time", and that
function no longer exists anywhere in `collab_splats/`. The body merely imports
`MapAnythingCreator`, which does exist, so the test passes and the symbol sweep in Part 1
cannot see it. Verify with `grep -rn _patch_mapanything_torch_compat collab_splats/`
returning nothing before deleting, and confirm `MapAnythingCreator`'s import is already
covered by `tests/pointcloud/feedforward/test_mapanything_creator.py`.

This is the one case in the design where staleness lives in a **name and docstring** rather
than in code. A reference sweep is structurally blind to it; only reading found it. Assume
there are others and treat any test whose docstring names a symbol as worth a `grep` for
that symbol.

#### `test_pipeline_cu121.py` — rename and relocate

Its docstring claims "numpy 2.x + torch 2.4 + pycolmap 4.0.4" and instructs running under
`/opt/conda/envs/nerfstudio/bin/python`; all three are wrong (the project pins torch
2.5.1+cu121 and `pycolmap>=3.1`, and the nerfstudio conda env is retired). But every one of
its 6 tests is behaviour coverage with `_forward` mocked, and **none** is an environment
assertion:

| Test | Destination |
|---|---|
| `test_build_pycolmap_reconstruction_roundtrip` | `tests/pointcloud/test_base.py` |
| `test_pointcloudresult_new_api` | `tests/pointcloud/test_base.py` |
| `test_feedforward_result_save_load` | `tests/pointcloud/feedforward/test_feedforward_zarr.py` |
| `test_vggtx_postprocess_pipeline` | `tests/pointcloud/feedforward/test_vggtx_creator.py` |
| `test_mapanything_postprocess_pipeline` | `tests/pointcloud/feedforward/test_mapanything_creator.py` |
| `test_tsdf_mesh_synthetic` | `tests/mesh/test_tsdf.py` |

Move the module-level helpers (`_synthetic_extrinsics`, `_synthetic_intrinsics`,
`_synthetic_vggtx_raw`, `N_FRAMES`, `H`, `W`) alongside whichever tests need them; where two
destinations need the same helper, put it in the nearest shared `conftest.py` rather than
duplicating it. Then delete the emptied file and `tests/integration/`.

Destinations assume Part 5's moves have already landed, so sequence Part 5 before this step
or the destination paths will not exist.

#### `test_bae_smoke.py` — delete

Three trivial checks, all duplicated. `test_bae_imports` and `test_bae_cuda_version` restate
`test_environment_contract.py`'s `test_bae_installed_version` and `test_cuda_version`;
`test_bundle_adjustment_module_loads` only imports `BundleAdjustmentConfig`, which
`tests/geometry/test_bundle_adjustment.py` already does as a precondition of its real
assertions.

`tests/integration/` holds only `__init__.py` and `test_pipeline_cu121.py`, so the deletion
empties it. Once the behaviour-coverage tests are relocated under `tests/pointcloud/`, remove
the directory. Part 5f's exempt list therefore does **not** name `integration` — the
directory will not exist.

### 2c. Marker registration

`pytest.mark.gpu` is applied in at least `tests/pointcloud/feedforward/test_mapanything_creator.py:548`,
`tests/pointcloud/sfm/test_colmap.py:62` and `tests/pointcloud/test_vggtx_creator.py:26`, but
`pyproject.toml` registers only `slow`. Each use emits `PytestUnknownMarkWarning`. Add to
`[tool.pytest.ini_options] markers`:

```toml
"gpu: requires a CUDA device (deselect with -m 'not gpu')",
```

## Part 3 — third_party reconcile

### 3a. Current state

`third_party/*` is gitignored (`.gitignore:7`); only `README.md` is tracked. So this is a
runtime-dependency and documentation problem, not a repository-size one. Nine directories,
2.0 GB on disk.

| Path | Size | Loaded how | Verdict |
|---|---|---|---|
| `vggt_spark/` | 181M | `sys.path` insert, `vggt_spark_creator.py:32` | **keep** — declares `name = "vggt"`, so it cannot be pip-installed alongside VGGT-X |
| `Video-Depth-Anything/` | 1.5G | `sys.path` insert, `vda.py:20` | **keep** — no `setup.py` upstream |
| `LoGeR/` | 77M | `sys.path` insert, `loger.py:84` | **keep** — no `setup.py` upstream |
| `hloc/` | 445M | editable pip install by `setup/hloc.sh` | **keep** — `HlocCreator` retained by decision |
| `VGGT-SLAM/` | 36M | `evals/scripts/run_vggt_slam.py` | **keep** — live eval consumer |
| `VGGT-X/` | 38M | nothing | **delete** — pip-installed as `vggt` via `[tool.uv.sources]` |
| `vggt-omega/` | 55M | nothing | **delete** — pip-installed as `vggt_omega`, confirmed in site-packages |
| `bae/` | 72M | nothing | **delete** — pip-installed as `bae`, confirmed in site-packages |
| `xfeat/` | 77M | nothing | **delete** — superseded by vismatch; `"xfeat"` now survives only as a vismatch **model-name string** in `FEATURE_MATCH_MODELS`, not as an import |

242M deleted. The three `sys.path` inserts in the package are the complete set of runtime
path dependencies; all three are in the keep column.

### 3b. The collision that makes vggt_spark irreducible

`third_party/VGGT-X/pyproject.toml:11` and `third_party/vggt_spark/pyproject.toml:11` both
declare `name = "vggt"`, from different upstreams (`Linketic/VGGT-X` and
`MIT-SPARK/VGGT_SPARK`). Only one can occupy site-packages. VGGT-X wins as the pip dep;
SPARK is therefore reached by a temporary `sys.path` insert that is removed immediately
after import, guarded by `_assert_loaded_from_spark`, which raises if the resolved `vggt`
module is not under the SPARK tree.

This is the single most important fact in `third_party/` and it is recorded nowhere in the
README. Making it explicit is a deliverable.

### 3c. The populator gap

Nothing creates `third_party/vggt_spark`. `setup/` contains `hloc.sh`, `loger.sh` and
`vggt_slam.sh`; `setup.sh` clones Video-Depth-Anything. No script produces vggt_spark's
181M tree at any commit. A fresh checkout therefore loses the SPARK backend silently — the
guarded import fails, the backend never registers, and `make_creator` reports an unknown
backend rather than a missing clone.

Add `setup/vggt_spark.sh`, following the idempotent shape of `setup/loger.sh`: clone
`MIT-SPARK/VGGT_SPARK` and check out the pinned commit `6e6e161`. Re-apply the pin on every
run so an existing clone cannot drift, matching what `setup.sh` already does for VDA.

### 3d. README rewrite

`third_party/README.md` is actively false. It lists five entries; two of them (`salad/`,
`xfeat/`) are attributed to `setup/feedforward.sh`, **a script that does not exist**, and
`salad/` is not even present on disk (it is a pip dep via `[tool.uv.sources]`). It omits
`vggt_spark/`, `VGGT-X/`, `vggt-omega/`, `bae/` and `VGGT-SLAM/`.

Rewrite as the single truth table over the five surviving directories, each row carrying:
path, upstream, pinned commit, populating script, consuming code path, load mechanism
(`sys.path` insert / editable install / subprocess), and licence note where one is
outstanding. Carry forward the two existing licence warnings: LoGeR has **no LICENSE file
upstream**, and the VDA vitl weights are **CC-BY-NC-4.0**.

### 3e. Enforcing test

`tests/test_third_party_contract.py`. This is what stops the rot recurring; without it this
whole part is a one-shot that decays the next time a dependency moves to pip.

Parse the README table, then assert:

1. Every README row names a populator that exists — a script in `setup/`, or `setup.sh`
   itself. Catches the current `setup/feedforward.sh` fiction.
2. Every README row names a consumer path that exists in the tree. Catches a row surviving
   the deletion of its only caller.
3. Every `third_party` path constant in `collab_splats/` appears as a README row. This is
   the direction that matters most: it catches a new `sys.path` insert added without
   documentation. Discover the constants by scanning for the literal `"third_party"` in
   package sources rather than hardcoding the three known names, so a fourth is caught.
4. No README row names a distribution that is also a `[tool.uv.sources]` entry in
   `pyproject.toml`. This is the rule that would have caught `bae`, `vggt-omega` and
   `VGGT-X` years earlier — a clone and a pip source for the same distribution is the exact
   defect being cleaned up here.

The test asserts on the README and the manifests, never on the presence of a clone. A bare
checkout with an empty `third_party/` must pass it, or it becomes a CI failure for everyone
who has not run the optional setup scripts.

## Part 4 — global_alignment removal

Delete `collab_splats/geometry/global_alignment.py` (103 lines) and every reference:

| File | Change |
|---|---|
| `collab_splats/geometry/global_alignment.py` | delete |
| `collab_splats/geometry/__init__.py:28-32` | remove the `run_global_alignment` branch from `__getattr__` |
| `collab_splats/geometry/__init__.py:50` | remove `"run_global_alignment"` from `__all__` |
| `collab_splats/pointcloud/feedforward/vggtx.py:25-28` | remove the parked-import comment block |
| `collab_splats/pointcloud/feedforward/vggtx.py:344-349` | remove the commented-out call site |
| `CLAUDE.md:73` | remove the `global_alignment.py` line from the architecture tree |

The module's only entry point was already unreachable: the sole call site is commented out
and the lazy `__getattr__` branch was the only other route in.

**Recorded consequence.** Both parked comments name **bae-vggt-parity** as the reason to
re-enable — "compare against LM BA" — and that effort is still listed in-flight in
`CLAUDE.md`. Deleting this module removes that comparison's VGGT-X-native arm. This is
accepted: the user has declared the module deprecated. The bae-vggt-parity entry in
`CLAUDE.md` is updated to state that the native-alignment comparison arm is gone and that
the effort now verifies only against upstream `zitongzhan/vggt --implementation bae`.
Recording it is what keeps a future reader from hunting for a module that was removed on
purpose.

## Part 5 — Structural realignment

`CLAUDE.md` states that `tests/` mirrors `collab_splats/`. Five refactors later it does not.
The broken-reference sweep in Part 1 proves tests point at **live symbols**; it says nothing
about whether they sit in the **right directory**. This part closes that gap.

### 5a. Measured drift

Mapping every test file to the package directory its imports and patch targets dominantly
exercise flags 40 files. Reading them splits into three classes.

**Directory-level drift.** `collab_splats/semantics/segmentation/` is a package directory
with **no corresponding test directory** — its tests sit flat in `tests/semantics/`.
`tests/preproc/` and `tests/localization/` lack `__init__.py` while every sibling test
package has one.

`tests/{docs,evals,examples,scripts}` have no package counterpart by design — they mirror
repo-root trees, not `collab_splats/`. They are correct and exempt. `tests/integration/` is
not on that list because Part 2b empties and removes it.

### 5b. Class A — unambiguous, 28 files

Name and imports agree on a single subject. Move with `git mv`.

| Destination | Count | Files |
|---|---|---|
| `tests/pointcloud/feedforward/` | 17 | `test_feature_lifting`, `test_feedforward_{density,intrinsics,preprocess_store,reproject,shared,zarr}`, `test_lc_collate_window`, `test_loger_creator`, `test_mapanything`, `test_mv_zarr_roundtrip`, `test_spark_mv_subprocess`, `test_vggt_omega_creator`, `test_vggt_spark_native_similarity`, `test_vggtx_creator`, `test_vggtx_preproc`, `test_zarr_attrs` (all from `tests/pointcloud/`) |
| `tests/semantics/segmentation/` (new) | 3 | `test_insid3_segmentation` (19/19 refs), `test_segmentation` (10/10), `test_sky_segmentation` |
| `tests/semantics/features/` | 5 | `test_extractor_preprocessing`, `test_features`, `test_positional_debiasing`, `test_query_api`, plus `tests/test_semantics_logging.py` |
| `tests/utils/` | 2 | `tests/test_visualization.py` (15/15 refs), `tests/test_heatmap.py` — both target `collab_splats/utils/visualization.py` |
| `tests/pointcloud/feedforward/` | 1 | `tests/test_feedforward_logging.py` (4/4 refs) |

`tests/test_viewer.py` and `tests/test_docstring_contract.py` stay at top level and are
correct there: `collab_splats/viewer.py` is a top-level module, and the docstring contract
is a repo-wide meta-test.

**Check while moving:** `tests/pointcloud/test_mapanything.py` and
`tests/pointcloud/feedforward/test_mapanything_creator.py` both exist and will land in the
same directory. Diff them for duplicated coverage before committing the move; merge rather
than keep two files if they overlap.

### 5c. Class B — heuristic wrong, do not move

`tests/wrapper/test_{refine,sfm,splats,verify}_stage.py` import their domain package
heavily, but their subject is `Reconstructor` stage orchestration, not the domain. They are
correctly placed. These become the seed of the contract test's exemption list.

### 5d. Class C — cross-cutting, read before placing, 6 files

- `tests/geometry/loop_closure/test_verify_threshold_resolution.py` — 4/6 refs into
  `pointcloud/feedforward`
- `tests/pointcloud/feedforward/test_verify_lc_data.py` — 5/11 refs into
  `geometry/loop_closure`

These two are near mirror images: each lives in the tree the other predominantly exercises.
At least one is misfiled; only reading them establishes which, and the answer may be that
both test a genuine seam and neither should move.

- `tests/pointcloud/test_mv_conf.py`, `tests/pointcloud/test_pose_extraction.py`
- `tests/semantics/features/test_extract_from_zarr.py`,
  `tests/semantics/test_features_guards.py`

Each gets a decision recorded in the plan: move, or stay with a one-line justification that
becomes its contract-test exemption.

### 5e. Moving is not content-neutral

Two hazards, both of which produce real state changes from a zero-diff move:

- **conftest inheritance.** `tests/pointcloud/conftest.py` and
  `tests/pointcloud/feedforward/conftest.py` both exist. `conftest.py` is hierarchical, so
  moving *down* into `feedforward/` only adds fixtures and is safe; a move that leaves a
  directory can drop a fixture the file depended on. That surfaces as a collection error,
  which is loud and detectable.
- **Execution order.** Moves change collection order, and this repo has a known class of
  test that passes in-suite but fails under a direct nodeid because the global torch RNG is
  unseeded. A move can therefore turn a test red with no content change.

Treat a red test after a move as a **discovery of a pre-existing order dependency**, not as
a regression introduced by the move. The protocol: run each moved file standalone by nodeid
**before** the move and record the result. If it was already red standalone, the move merely
exposed it — fix the seeding in a separate commit and say so. Use `git mv` throughout so
history follows the file.

### 5f. Enforcing test

Extend the contract idea from Part 3e into `tests/test_layout_contract.py`:

1. Every directory under `tests/` maps to a directory under `collab_splats/`, except an
   explicit exempt list (`docs`, `evals`, `examples`, `scripts`, and fixture dirs such as
   `preproc/data`). `integration` is absent from that list because Part 2b removes it.
2. Every package directory under `collab_splats/` has a corresponding test directory. This
   is the rule that catches `semantics/segmentation/`.
3. Every test package directory contains `__init__.py`.
4. No test file's dominant subject package differs from its own directory, except for an
   explicit exemption list seeded with Class B and whatever Class C decides. Each exemption
   carries a one-line reason in the source.

Rule 4 is the one that decays without care: an exemption list is a place to hide misfiled
tests. The plan should require that adding an exemption is a deliberate edit with a stated
reason, and the test's own docstring should say so.

## Verification protocol

The delete-then-run step is a check on the static read, not the source of it.

Before deleting, write down the prediction: which tests change state, and which skips
appear or disappear. Then run the full suite and compare against the Part 1 reference on
**both** axes:

- **Failure delta** must be zero. Any new failure means a deletion was wrong.
- **Skip delta** must match the prediction exactly. An unexplained skip counts as a miss,
  not a pass — that is the failure mode this whole design is ordered around, and the
  measurement that a green exit code will not give you.
- **Collected count** must drop by exactly the number of tests in the deleted files, minus
  any moved out of `test_pipeline_cu121.py`. A larger drop means a file stopped collecting.

For Part 5 the collected count is the primary gate, because a move should change **nothing
else**. After the moves, the count must be identical to the pre-move count. A drop means a
file stopped being collected at its new path — the most likely cause of a silently
disappearing test, and one that no failure will report.

Sequence the moves after the deletions, in their own commits, one destination directory per
commit. A move commit that also changes file contents cannot be verified by count alone.

Run the suite in chunks rather than one invocation: a full suite run under concurrent agent
activity has previously invented failures here. Never pipe to `tail` without capturing the
exit code separately — the pipe swallows it. Never use `--tb=no` during the audit, because
a failure in one file can be the cause of a failure in another and the traceback is what
distinguishes them.

## Deliverables

1. `vismatch` + `lomatch` installed; full-suite reference triple recorded.
2. 104 orphan `__pycache__` entries and `tests/nerfstudio_methods/` removed.
3. `tests/test_cu121_migration.py`, `tests/integration/test_pipeline_cu121.py`,
   `tests/test_bae_smoke.py` deleted; behaviour-coverage tests from the integration file
   relocated under `tests/pointcloud/`.
4. `gpu` marker registered in `pyproject.toml`.
5. `third_party/{VGGT-X,vggt-omega,bae,xfeat}` deleted (242M).
6. `setup/vggt_spark.sh` added, pinned to `6e6e161`.
7. `third_party/README.md` rewritten as the five-row truth table.
8. `tests/test_third_party_contract.py` added, passing on a bare checkout.
9. `collab_splats/geometry/global_alignment.py` and all references removed.
10. `CLAUDE.md` updated: architecture tree line dropped, bae-vggt-parity entry amended.
11. 28 Class A test files relocated with `git mv`; `tests/semantics/segmentation/` created;
    `__init__.py` added to `tests/preproc/` and `tests/localization/`.
12. 6 Class C files read and placed, each with a recorded decision.
13. `tests/test_layout_contract.py` added, with a justified exemption list seeded from
    Class B.
14. Collected-test count identical before and after every move commit.

## Open questions

None. All five scope decisions are settled: unwired SfM backends stay, `global_alignment`
goes, discovery is static with the run as confirmation, the third_party reconcile ships with
an enforcing test, and the test tree is realigned to the current package layout with a
layout contract to hold it there.
