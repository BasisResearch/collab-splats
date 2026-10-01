# Reconstructor Release Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `collab_splats/wrapper/` and `collab_splats/remote/` with `collab_splats/reconstructor.py`,
`collab_splats/__main__.py` and `collab_splats/remote.py`, with no logic repeated from the packages they compose.

**Architecture:** One `STAGES` dict drives dependency checks, run order and `getattr` dispatch to zero-argument stage
methods named after their stages. `run()` alone decides skip / refuse / run. Generic rclone transport moves to
collab-data's `RcloneClient` (branch `tlb-3d-tools`); `remote.py` keeps only scene knowledge (buckets, ids, excludes).

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, rclone 1.53.3, zarr 3, pycolmap, argparse,
`uv` for the lock.

Spec: [`docs/superpowers/specs/2026-09-29-reconstructor-release-cleanup-design.md`](../specs/2026-09-29-reconstructor-release-cleanup-design.md).
Rules: [017](../decisions/017-release-cleanup-rules.md).

---

## Ground rules (every task)

- **Worktree:** `WT=/workspace/collab-splats/.worktrees/reconstructor-release`, branch `clean/reconstructor-release`.
  The Bash cwd resets between calls, so prefix every command with `cd $WT &&` (spell the path out).
- **Python:** `/opt/venv/reconstruction/bin/python` only, always with `PYTHONPATH=$WT`.
- **Gate** (after every commit):

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && PYTHONPATH=/workspace/collab-splats/.worktrees/reconstructor-release /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && cd /workspace/collab-splats/.worktrees/reconstructor-release && PYTHONPATH=/workspace/collab-splats/.worktrees/reconstructor-release /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor tests/remote tests/preproc tests/pointcloud tests/mesh tests/geometry tests/dashboard tests/test_docstring_contract.py -q
  ```

  - Before Task 1 moves `tests/wrapper/` the first path is `tests/wrapper`.
  - The printed path must be inside the worktree.
  - Never pipe pytest through `| tail`. Never pass `--tb=no`.
  - Add `tests/semantics` for Tasks 3 and 13, `tests/utils` for Task 4, and `tests/evals` for Task 16.
  - Compare pass / fail / skip counts against the Task 0 baseline. A new SKIP is a failure until explained.
- **Commits:**
  - Use `git commit --only <paths>`, because other sessions share the index.
  - `docs/superpowers/**` needs `git add -f`.
  - Never amend, rebase or reset. Fix a mistake with a new commit.
  - Every message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Style** (memory overrides CLAUDE.md where they differ):
  - Every block comment is ONE plain line saying what the code does. No header+bullet runs.
  - Docstrings: `"""` on its own line, one-line summary, `- ` bullets, `Args:`/`Returns:`/`Raises:`. Private defs get
    the summary and bullets only.
  - Imports are absolute, in 4 isort groups; isort wraps at 88.
  - Blank line around every block. US spelling. No nested calls (`f(g(x))` is split one call per line).
  - No module constants for tunables (use kwargs). No layout constants or path helpers. No legacy-file checks.
- **Runs:** every real pipeline run sets `semantics: {enabled: false}`.
- **Localization:** do not edit `_localization_db_exists` or `_build_localization_db`, except the one line in Task 14.
- **collab-data push:** needs the user's explicit OK. Task 8 stops and asks.

## Spec corrections (found while planning; the plan follows these, not the spec)

1. **`PULL_EXCLUDES` moves to `collab_splats/dashboard/config.py`, not `dashboard/pipeline.py`.**
   - `app.py` must not import `pipeline` eagerly (fast-bind path).
   - `config.py` is already imported by both.
   - The kwarg stays `excludes=`, not `exclude=`, because the dashboard and its tests call it by name.
2. **The pointcloud stage does not call `to_colmap`.**
   - Every creator writes `colmap_model_dir` itself when handed a `model_dir`.
   - Only `refine` re-exports, via `to_colmap` + `write_colmap_reconstruction`.
3. **Stats stay text `--stats-one-line`.**
   - The dashboard log shows raw rclone lines, so a JSON log would regress it.
   - collab-data gains `parse_percent(line)` (regex `,\s*(\d{1,3})%`, clamped to 100).
   - `dashboard/operation_log.py` imports it from there.
4. **`RcloneClient.check` writes `--combined <tempfile>` rather than `-`.**
   - Verified on rclone 1.53.3: lines start `+ ` (missing on destination), `- ` (only on destination), `* ` (differ),
     `= ` (match), `! ` (error).
   - A transport/config fault writes no file and exits 1.
   - Any `+`/`-`/`*` line returns False. Otherwise a non-zero exit raises `RuntimeError`.
5. **Directory-input records stay `{"frame_idx": i}` (enumerate order).**
   - Provenance is `{"input_path": str(input), **preproc_cfg}`.
   - When undistorting it adds `"camera"` and `"undistorted_camera"` as `camera.todict()`.
   - `to_json_safe` writes pycolmap's pybind11 `CameraModelId` as its `.name`. It is not an `enum.Enum`: detect it by
     `hasattr(type(obj), "__members__") and hasattr(obj, "name")`.
6. **`Reconstructor.pointcloud` becomes a method, so the in-memory result is `self._result` behind a lazy `result`
   property.**
   - The property loads with `load_depth/world_points/confidence/pixel_indices=False`.
   - The `pointcloud` and `refine` stages reset `self._result = None`.
7. **`outputs` (stage → marker path):**

   | stage | marker |
   |---|---|
   | preproc | `images_dir` |
   | pointcloud | `pointcloud_zarr` |
   | refine | `backend_dir/colmap/refine.json` |
   | semantics | `lifted_store_path(backend_dir/"semantics", extractor)` |
   | splats | `backend_dir/splats/ckpt.pt` |
   | mesh | `backend_dir/mesh.ply` |
   | reconstruction_quality_report | `backend_dir/reconstruction_quality_report.json` |

   - `localize` has no single file, so `done("localize")` calls `_localization_db_exists`.
   - `pointcloud` done is `pointcloud_zarr` AND `colmap_model_dir` existing, as today.
8. **Refine rewrites the zarr with `save_zarr(extra_attrs=<attrs read before the write>)`.**
   - `save_zarr` opens `mode="w"`, so the `local_features` group (localization DB) is dropped.
   - That DB indexed the pre-refine points, so it was stale anyway.
9. **Tests deleted along with the code they guard:**
   - the mesh frame-count mismatch test (the stage reads one source, so nothing can mismatch);
   - `test_splats_stage_rejects_frames_missing_from_feedforward` (same reason);
   - the LoGeR `max_frames` advisory test and the reserved-kwarg clash test;
   - `tests/dashboard/test_pipeline.py::test_no_fourth_ae_policy_hides_in_a_parameter_default` (`_lift_and_save` is
     gone);
   - `tests/wrapper/test_verify_stage.py`, `ConfigLoader` tests, `collect_videos` and `scene_output_dir` date tests;
   - hloc conf-string validation tests (hloc resolves those names itself).
10. **Moved validation:**
    - `LoopClosureConfig` unknown knobs raise `ValueError` in `validate_config`, not at stage time.
    - The semantics extractor is instantiated even on a 2D-cache hit (accepted: `extract_feature_cache` owns the skip).
11. **Round 1 needs one non-import line in the moved `reconstructor.py`.**
    - `DEFAULT_CONFIG_DIR = Path(__file__).parents[2] / "configs"` becomes `parents[1]`, because the file moved one
      level up.
    - The byte-diff proof allows exactly that line and import lines.
12. **`collab_splats/remote/` cannot coexist with `collab_splats/remote.py`.** Round 1 moves
    `remote/rerun.py` → `wrapper/rerun.py` (deleted in Task 15) and deletes `remote/__init__.py`, whose names all live
    in `sources.py`.

## File map

| Path | Fate | Responsibility after |
|---|---|---|
| `collab_splats/reconstructor.py` | from `wrapper/reconstructor.py` | `STAGES`, `LEAF_STAGES`, localization helpers, `Reconstructor` |
| `collab_splats/__main__.py` | new | `reconstruct local ...` / `reconstruct remote ...` CLI |
| `collab_splats/remote.py` | from `remote/sources.py` | `SCENE_ID_RE`, `PUSH_EXCLUDES`, `SceneSource` over `RcloneClient` |
| `collab_splats/wrapper/` | deleted (Task 15) | — |
| `collab_splats/remote/` | deleted (Round 1) | — |
| `docs/examples/{reconstruct,run_pipeline,run_pipeline_remote}.py` | deleted (Task 15) | — |
| `collab_splats/preproc/frames.py` | modified | `frame_paths(dir, idxs=None)` |
| `collab_splats/semantics/segmentation/sky.py` | modified | uses `frame_paths(dir, idxs)` |
| `collab_splats/utils/io.py` | modified | `to_json_safe` pybind11 enum |
| `collab_splats/pointcloud/utils.py` | modified | `frame_depths` |
| `collab_splats/mesh/tsdf.py` | modified | `sdf_trunc < voxel_size` raises |
| `collab_splats/pointcloud/sfm/{base,colmap,hloc,instantsfm}.py` | modified | own-arg validation |
| `collab_splats/dashboard/{config,app,pipeline,operation_log}.py` | modified | `PULL_EXCLUDES` home; `parse_percent` import |
| `/workspace/collab-data/collab_data/data_dashboard/rclone_client.py` | modified | `run`, `run_streaming`, `copy_dir`, `check`, `parse_percent` |
| `/workspace/collab-data/tests/data_dashboard/test_rclone_client.py` | new | local-remote tests |
| `tests/reconstructor/` | from `tests/wrapper/` + `tests/examples/` + `tests/scripts/test_reconstruct.py` | stages, `run`, CLI |
| `tests/remote/test_remote.py` | from `tests/remote/test_sources.py` | `SceneSource` |
| `pyproject.toml`, `uv.lock` | modified | collab-data pin; `reconstruct` script |
| `evals/eval.py`, `configs/README.md`, `docs/known-test-failures.md`, `docs/source/api/` | modified | caller sweep |
| `tests/test_docstring_contract.py` | modified | `MODULES` += 3 files |

---

## Task 0: Preconditions and baseline

**Files:** none edited.

- [ ] **Step 1: Pull collab-data and confirm the branch** (user rule: before changing anything)

  ```bash
  cd /workspace/collab-data && git fetch origin && git checkout tlb-3d-tools && git pull --ff-only && git status --short --branch
  ```

  - Expected: `## tlb-3d-tools...origin/tlb-3d-tools`.
  - Untracked `config-local` is expected and must stay untouched.
  - If the pull is not a fast-forward, STOP and report.

- [ ] **Step 2: Confirm the worktree and symlink `third_party`**

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && git status --short --branch && git log --oneline -1 && ls third_party
  ```

  - Expected: branch `clean/reconstructor-release`, tip `ffee192c` or a later docs commit.
  - For each entry of `/workspace/collab-splats/third_party/*` that is missing here, run
    `ln -s /workspace/collab-splats/third_party/<name> third_party/<name>`. Without them, guarded tests SKIP instead of
    running.

- [ ] **Step 3: Check the gsplat version** (it has flipped between sessions)

  ```bash
  /opt/venv/reconstruction/bin/python -c "import gsplat; print(gsplat.__version__)"
  ```

  Record it in the task report.

- [ ] **Step 4: Run and record the baseline gate**

  Run the gate with `tests/wrapper tests/examples tests/scripts tests/remote tests/preproc tests/pointcloud tests/mesh tests/geometry tests/dashboard tests/semantics tests/utils tests/evals tests/test_docstring_contract.py`.
  - Record passed / failed / skipped / xfailed / xpassed for the whole run, and the failing node ids.
  - Save the summary to `$SCRATCH/baseline.txt`, where
    `SCRATCH=/tmp/claude-0/-workspace-collab-splats/c159867f-9464-48ff-a1a3-31004fa868a8/scratchpad`.
  - Every later gate is compared against this file.
  - A baseline failure listed in `docs/known-test-failures.md` stays allowed. Any other failure is reported before
    continuing.

---

## Round 1 — move and prose (no behavior change)

### Task 1: Move files, update imports

**Files:**
- Move: `collab_splats/wrapper/reconstructor.py` → `collab_splats/reconstructor.py`
- Move: `collab_splats/remote/sources.py` → `collab_splats/remote.py`
- Move: `collab_splats/remote/rerun.py` → `collab_splats/wrapper/rerun.py`
- Delete: `collab_splats/remote/__init__.py`
- Move: `tests/wrapper/*` → `tests/reconstructor/`; `tests/remote/test_sources.py` → `tests/remote/test_remote.py`
- Modify (import lines / patch-target strings only): `collab_splats/wrapper/__init__.py`, `collab_splats/wrapper/batch.py`,
  `collab_splats/wrapper/rerun.py`, `docs/examples/*.py`, `evals/eval.py`, every file under `tests/` listed below.

- [ ] **Step 1: `git mv`**

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && git mv collab_splats/wrapper/reconstructor.py collab_splats/reconstructor.py && git mv collab_splats/remote/rerun.py collab_splats/wrapper/rerun.py && git mv collab_splats/remote/sources.py collab_splats/remote.py && git rm -q collab_splats/remote/__init__.py && git mv tests/wrapper tests/reconstructor && git mv tests/remote/test_sources.py tests/remote/test_remote.py
  ```

- [ ] **Step 2: Fix the one path line** in `collab_splats/reconstructor.py`

  ```python
  DEFAULT_CONFIG_DIR = Path(__file__).parents[1] / "configs"
  ```

- [ ] **Step 3: Rewrite module paths** — a mechanical substitution over code and tests

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && grep -rlE "collab_splats\.(wrapper\.reconstructor|remote\.sources|remote\.rerun)" collab_splats tests evals docs/examples | xargs sed -i -E 's/collab_splats\.wrapper\.reconstructor/collab_splats.reconstructor/g; s/collab_splats\.remote\.sources/collab_splats.remote/g; s/collab_splats\.remote\.rerun/collab_splats.wrapper.rerun/g'
  ```

  - Then `from collab_splats.wrapper import reconstructor as X` becomes `from collab_splats import reconstructor as X`, in
    `tests/pointcloud/test_loger_creator.py:24` and `tests/dashboard/test_pipeline.py:284`.
  - `tests/reconstructor/*` imports `tests.wrapper._stubs`; rewrite each to `tests.reconstructor._stubs`
    (`grep -rn "tests.wrapper" tests`).
  - `collab_splats/wrapper/__init__.py` keeps its re-exports, pointing at `collab_splats.reconstructor`.

- [ ] **Step 4: Prove the moves are imports-only**

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && rtk proxy git diff -M --cached --stat && rtk proxy git diff -M HEAD -- collab_splats/reconstructor.py collab_splats/remote.py collab_splats/wrapper/rerun.py | grep -E '^[+-][^+-]' | grep -vE '^[+-](from |import |    [A-Za-z_]+,$|\)$)'
  ```

  - Expected: git reports each file as a rename (R0xx).
  - The filtered diff shows only the two `DEFAULT_CONFIG_DIR` lines.
  - Any other surviving line means a non-import edit: revert it.

- [ ] **Step 5: Gate** (with `tests/reconstructor` in place of `tests/wrapper`, plus `tests/examples tests/scripts`).
  Counts must equal the baseline.

- [ ] **Step 6: Commit**

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && git add -A collab_splats/reconstructor.py collab_splats/remote.py collab_splats/remote collab_splats/wrapper tests evals docs/examples && git commit --only collab_splats tests evals docs/examples -m "refactor: move reconstructor and remote up to single modules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

### Task 2: Prose round on surviving code

**Files:** `collab_splats/reconstructor.py` (path properties, `validate_config`, class docstring, module docstring),
`collab_splats/remote.py` (module, `SCENE_ID_RE`, `PUSH_EXCLUDES`, `SceneSource` and its public methods).

- [ ] **Step 1: Rewrite prose only.**
  - Docstrings per 017. Block comments become one plain line each.
  - Do not touch the localization helpers, and change no identifier.
  - Target text for the path properties:

  ```python
  @property
  def backend_dir(self) -> Path:
      """
      Per-backend output directory; every artifact after preproc lands here.
      """
      return Path(self.config["output_path"]) / self.config["pointcloud"]["backend"]

  @property
  def images_dir(self) -> Path:
      """
      Keyframe store written by preproc; shared by every backend.
      """
      return Path(self.config["output_path"]) / "images"

  @property
  def pointcloud_zarr(self) -> Path:
      """
      Reconstruction store written by the pointcloud stage.
      """
      return self.backend_dir / "pointcloud.zarr"

  @property
  def colmap_model_dir(self) -> Path:
      """
      Binary COLMAP model written beside the zarr.
      """
      return self.backend_dir / "colmap" / "sparse" / "0"

  @property
  def semantics_cache_dir(self) -> Path:
      """
      Scene-level 2D feature cache, one `<extractor>.zarr` each.

      - frames alone determine it, so every backend lifts from the same cache
      """
      return Path(self.config["output_path"]) / "semantics"
  ```

  - `PUSH_EXCLUDES` gets one comment line per entry, with no paragraphs:

  ```python
  # Never pushed: regenerable caches, build databases and the source video
  PUSH_EXCLUDES = (
      # Scene-root 2D patch cache; the leading slash keeps <backend>/semantics pushed
      "/semantics/**",
      # Source video, already in the curated bucket; bracketed because rclone globs are case sensitive
      "*.[Mm][Pp]4",
      "*.[Mm][Oo][Vv]",
      "*.[Aa][Vv][Ii]",
      # COLMAP match database left by the removed geometric verification
      "/*/colmap/database.db",
      # SIFT databases and hloc intermediates, rebuilt from images/
      "/*/colmap/instantsfm.db",
      "/*/colmap/colmap.db",
      "/*/colmap/hloc/**",
  )
  ```

  (`database.db` is dropped in Task 9, which is a code change.)

- [ ] **Step 2: AST proof plus sanity mutation**

  Write `$SCRATCH/ast_equal.py`:

  ```python
  import ast
  import subprocess
  import sys


  def strip(src: str) -> str:
      tree = ast.parse(src)
      for node in ast.walk(tree):
          body = getattr(node, "body", None)
          if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
              node.body = body[1:] or [ast.Pass()]
      return ast.dump(tree)


  for path in sys.argv[1:]:
      old = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout
      new = open(path).read()
      print(path, "EQUAL" if strip(old) == strip(new) else "DIFFERENT")
  ```

  - Run it from the worktree on both files. Expected: `EQUAL` twice.
  - Sanity mutation: temporarily change `"/semantics/**"` to `"/semantic/**"`, rerun, and expect `DIFFERENT` for
    `remote.py`. Revert the mutation.
  - Verify the file hash matches the pre-mutation hash (`sha256sum`, not `diff`: RTK's diff lies).

- [ ] **Step 3: Gate** — counts equal the baseline.

- [ ] **Step 4: Commit**

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && git commit --only collab_splats/reconstructor.py collab_splats/remote.py -m "docs(reconstructor): prose pass on surviving reconstructor and remote code

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

## Round 2 — code, one commit per change

### Task 3: `frames.frame_paths(dir, idxs)`

**Files:**
- Modify: `collab_splats/preproc/frames.py` (`frame_paths`, `read_frames`)
- Modify: `collab_splats/semantics/segmentation/sky.py:144-153`
- Test: `tests/preproc/test_frames.py`, `tests/semantics/test_sky_segmentation.py`

- [ ] **Step 1: Failing tests** — append to `tests/preproc/test_frames.py`:

  ```python
  def test_frame_paths_selects_by_source_index_in_requested_order(tmp_path):
      for idx in (4, 10, 7):
          (tmp_path / f"frame_{idx:06d}.png").touch()

      paths = frames.frame_paths(tmp_path, [10, 4])

      assert [p.name for p in paths] == ["frame_000010.png", "frame_000004.png"]


  def test_frame_paths_raises_on_an_index_the_directory_lacks(tmp_path):
      (tmp_path / "frame_000004.png").touch()

      with pytest.raises(KeyError, match=r"frame_paths: frame_idx \[5\]"):
          frames.frame_paths(tmp_path, [4, 5])


  def test_frame_paths_without_idxs_is_filename_order(tmp_path):
      for idx in (10, 4):
          (tmp_path / f"frame_{idx:06d}.png").touch()

      assert [p.name for p in frames.frame_paths(tmp_path)] == ["frame_000004.png", "frame_000010.png"]
  ```

  Run `pytest tests/preproc/test_frames.py -q -k frame_paths`. Expected: the first two FAIL with a `TypeError` (too
  many positional arguments).

- [ ] **Step 2: Implement**

  ```python
  def frame_paths(dir: Path | str, idxs: Sequence[int] | None = None) -> list[Path]:
      """
      Image paths in a frame directory, by source frame index or in filename order.

      - idxs select by source frame_idx, never by row position

      Args:
          dir: directory holding frame_NNNNNN.<ext> images.
          idxs: source frame indices, in the order wanted; None takes every image.

      Returns:
          Image paths; empty when the directory is missing or holds none.

      Raises:
          KeyError: when idxs names a frame_idx the directory does not hold.
      """
      dir = Path(dir)
      paths = sorted(p for p in dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) if dir.is_dir() else []
      if idxs is None:
          return paths

      # Map source index to path; refuse any index the directory lacks
      by_idx = {frame_idx_from_path(p): p for p in paths}
      missing = [int(i) for i in idxs if int(i) not in by_idx]
      if missing:
          raise KeyError(f"frame_paths: frame_idx {missing[:5]} not in {dir}")

      return [by_idx[int(i)] for i in idxs]
  ```

  `read_frames` body becomes:

  ```python
      paths = frame_paths(dir, idxs)
      if not paths:
          raise FileNotFoundError(f"read_frames: no frame images in {dir}")

      return np.stack([read_image(p) for p in paths])
  ```

  - Keep `read_frames`' docstring. Add `KeyError` under `Raises:`.
  - `frame_paths(dir, idxs)` raises `KeyError` before the empty check whenever `idxs` is given and the directory is
    empty. That is accepted: the message names the missing indices.

- [ ] **Step 3: `sky_masks`** — replace lines 144-153 (the `paths` / `by_idx` / `wanted` / `missing` block):

  ```python
      # Resolve wanted frames up front, so a warm cache rejects junk too
      paths = frames.frame_paths(images_dir, idxs)
      if not paths:
          raise FileNotFoundError(f"sky_masks: no frame images in {images_dir}")

      wanted = [frames.frame_idx_from_path(p) for p in paths]
      path_of = dict(zip(wanted, paths))
  ```

  - In the segment loop, change `model.segment(by_idx[idx])` to `model.segment(path_of[idx])`.
  - Then grep `tests/semantics/test_sky_segmentation.py` for `match="sky_masks: frame_idx` and change it to
    `match="frame_paths: frame_idx`.

- [ ] **Step 4: Run** `pytest tests/preproc/test_frames.py tests/semantics/test_sky_segmentation.py -q`. Expected: PASS.
  Then run the gate + `tests/semantics`.

- [ ] **Step 5: Commit** `feat(preproc): frame_paths selects by source frame index` (files: `frames.py`, `sky.py`, both
  test files).

### Task 4: `to_json_safe` pybind11 enums; `write_frames` message

**Files:** `collab_splats/utils/io.py`, `collab_splats/preproc/frames.py`, `tests/utils/test_io.py`,
`tests/preproc/test_frames.py`

- [ ] **Step 1: Failing tests**

  ```python
  # tests/utils/test_io.py
  def test_to_json_safe_writes_a_pybind11_enum_as_its_name():
      camera = pycolmap.Camera(model="OPENCV", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0])

      safe = to_json_safe(camera.todict())

      assert safe["model"] == "OPENCV"
      json.dumps(safe)
  ```

  ```python
  # tests/preproc/test_frames.py
  def test_write_frames_names_an_empty_selection(tmp_path):
      with pytest.raises(ValueError, match="no frames selected"):
          frames.write_frames(tmp_path / "images", [], [], {})
  ```

  - Add `import pycolmap` to the model-upstream import group of `test_io.py`.
  - Run both. Expected: FAIL (the `CameraModelId` is passed through and `json.dumps` raises `TypeError`; the message
    doesn't match).

- [ ] **Step 2: Implement** — in `to_json_safe`, before the `np.generic` branch:

  ```python
      if hasattr(type(obj), "__members__") and hasattr(obj, "name"):
          return obj.name
  ```

  - Add a docstring bullet: `- enums (enum.Enum and pybind11 enums alike) become their name`.
  - In `write_frames`, split the check:

  ```python
      if not records:
          raise ValueError("write_frames: no frames selected")
      if "frame_idx" not in records[0]:
          raise ValueError("write_frames: every record must contain 'frame_idx' (source video index)")
  ```

  - Update the `Raises:` entry to say "no records".

- [ ] **Step 3: Run** both test files, then the gate + `tests/utils`. Expected: PASS.

- [ ] **Step 4: Commit** `feat(utils): to_json_safe writes enums by name; write_frames names an empty selection`.

### Task 5: `pointcloud.utils.frame_depths`

**Files:** `collab_splats/pointcloud/utils.py`, `tests/pointcloud/test_utils.py`

- [ ] **Step 1: Failing tests**

  ```python
  def _depth_result(depth, confidence):
      n, h, w = depth.shape
      return SimpleNamespace(
          depth=depth,
          confidence=confidence,
          original_coords=np.tile(np.array([0, 0, w, h, w, h]), (n, 1)),
      )


  def test_frame_depths_masks_low_confidence_then_lifts_to_the_frame_grid():
      depth = np.full((1, 4, 4), 2.0, dtype=np.float32)
      confidence = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
      rgbs = np.zeros((1, 8, 8, 3), dtype=np.uint8)

      out = frame_depths(_depth_result(depth, confidence), rgbs, conf_percentile=50)

      assert out.shape == (1, 8, 8)
      assert (out == 0).any() and np.isclose(out[out > 0], 2.0).all()


  def test_frame_depths_without_confidence_keeps_every_pixel():
      depth = np.full((1, 4, 4), 2.0, dtype=np.float32)
      rgbs = np.zeros((1, 8, 8, 3), dtype=np.uint8)

      out = frame_depths(_depth_result(depth, None), rgbs, conf_percentile=50)

      assert (out > 0).all()


  def test_frame_depths_refuses_a_result_without_depth():
      with pytest.raises(ValueError, match="no depth"):
          frame_depths(_depth_result(None, None), np.zeros((1, 8, 8, 3), np.uint8))
  ```

  - `_depth_result(None, None)` needs `depth.shape`; for that case build `SimpleNamespace(depth=None, confidence=None,
    original_coords=None)` directly.
  - Read `upsample_depths`' signature (`collab_splats/utils/image.py`) before running. If it needs the
    `original_coords` layout `[x0, y0, x1, y1]` on the model grid, keep the tile above.
  - Expected: `ImportError` on `frame_depths`.

- [ ] **Step 2: Implement** (imports: `upsample_depths` from `collab_splats.utils.image`, `to_numpy` from
  `collab_splats.utils.torch_utils`; check for an import cycle by importing `collab_splats.pointcloud.utils` alone):

  ```python
  def frame_depths(result: PointcloudResult, rgbs: np.ndarray, conf_percentile: float | None = None) -> np.ndarray:
      """
      Model-grid depth, confidence-masked, lifted onto the frame grid.

      - masks on the model grid, where the confidence was predicted
      - absent confidence (sfm, some backends) fuses unmasked and logs it
      - rows follow the result's own image_paths; rgbs must match them

      Args:
          result: reconstruction carrying depth, confidence and original_coords.
          rgbs: frames in the result's row order, (N, H, W, 3) uint8.
          conf_percentile: drop depth below this confidence percentile; None keeps every pixel.

      Returns:
          (N, H, W) float32 depth on the frame grid; 0 marks no depth.

      Raises:
          ValueError: when the result carries no depth.
      """
      if result.depth is None:
          raise ValueError("frame_depths: result has no depth")

      # Zero out low-confidence depth on the model grid
      depth = np.asarray(result.depth, dtype=np.float32)
      if conf_percentile is not None and result.confidence is None:
          logger.info("conf_percentile=%s but the result has no confidence; keeping every pixel", conf_percentile)
      elif conf_percentile is not None:
          confidence = to_numpy(result.confidence)
          keep = confidence_mask(confidence, conf_percentile)
          depth = np.where(keep, depth, 0.0).astype(np.float32)

      # Lift onto the frame grid through each row's crop box
      crop_boxes = np.asarray(result.original_coords)[:, :4]
      return upsample_depths(depth, rgbs, crop_boxes)
  ```

- [ ] **Step 3: Run** `pytest tests/pointcloud/test_utils.py -q`, then the gate.

- [ ] **Step 4: Commit** `feat(pointcloud): frame_depths lifts masked model-grid depth to frames`.

### Task 6: `create_tsdf_mesh` sdf raise; sfm creators validate their args

**Files:**
- `collab_splats/mesh/tsdf.py`
- `collab_splats/pointcloud/sfm/base.py:53`, `colmap.py:42-44`, `hloc.py:52`, `instantsfm.py:54-55`
- Tests: `tests/mesh/test_tsdf.py`, `tests/pointcloud/sfm/test_creator_args.py` (new)

- [ ] **Step 1: Failing tests**

  ```python
  # tests/mesh/test_tsdf.py
  def test_create_tsdf_mesh_refuses_a_band_narrower_than_a_voxel(tmp_path):
      depths = np.ones((1, 4, 4), np.float32)
      rgbs = np.zeros((1, 4, 4, 3), np.uint8)

      with pytest.raises(ValueError, match="sdf_trunc"):
          create_tsdf_mesh(depths, rgbs, np.eye(4)[None], np.eye(3)[None], tmp_path, voxel_size=0.1, depth_trunc=5.0, sdf_trunc=0.05)
  ```

  ```python
  # tests/pointcloud/sfm/test_creator_args.py
  """
  Every sfm creator refuses a bad argument at construction.
  """

  import pytest

  from collab_splats.pointcloud.sfm import SFM_CREATORS


  @pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
  @pytest.mark.parametrize("value", [0, True, 1.5])
  def test_num_threads_must_be_a_positive_int(backend, value):
      with pytest.raises(ValueError, match="num_threads"):
          SFM_CREATORS[backend](num_threads=value)


  @pytest.mark.parametrize("backend", sorted(SFM_CREATORS))
  @pytest.mark.parametrize("value", [0.0, 1.5, True])
  def test_min_registered_frac_must_be_in_unit_interval(backend, value):
      with pytest.raises(ValueError, match="min_registered_frac"):
          SFM_CREATORS[backend](min_registered_frac=value)


  @pytest.mark.parametrize("backend", ["colmap", "hloc"])
  @pytest.mark.parametrize("key", ["overlap", "num_retrieved"])
  @pytest.mark.parametrize("value", [0, True])
  def test_pair_counts_must_be_positive_ints(backend, key, value):
      with pytest.raises(ValueError, match=key):
          SFM_CREATORS[backend](**{key: value})


  @pytest.mark.parametrize("backend", ["colmap", "hloc"])
  def test_pairing_must_be_a_known_mode(backend):
      with pytest.raises(ValueError, match="pairing"):
          SFM_CREATORS[backend](pairing="nonsense")


  @pytest.mark.parametrize("value", [-1, 2**32, 1.0])
  def test_instantsfm_random_seed_domain(value):
      with pytest.raises(ValueError, match="random_seed"):
          SFM_CREATORS["instantsfm"](random_seed=value)


  @pytest.mark.parametrize("value", [1, 0])
  def test_instantsfm_min_num_view_per_track_floor(value):
      with pytest.raises(ValueError, match="min_num_view_per_track"):
          SFM_CREATORS["instantsfm"](min_num_view_per_track=value)
  ```

  - Before writing it, check that `hloc.py` defines `pairing`. If `HlocCreator` has no `pairing` field, restrict that
    test to `["colmap"]`.
  - Constructing a creator with defaults must not import heavy deps. Confirm with
    `python -c "from collab_splats.pointcloud.sfm import SFM_CREATORS; SFM_CREATORS['hloc']()"`.
  - Move the matching tests out of `tests/reconstructor/test_sfm_config.py` (those that call `Reconstructor(...)` with a
    bad sfm block and expect `ValueError`) and delete them there. They are replaced by the file above.
  - Keep the `test_sfm_config.py` tests of method/backend/BA/LC.

- [ ] **Step 2: Implement.**
  - `create_tsdf_mesh`, after `sdf_trunc` defaults to `4 * voxel_size`:

  ```python
      # A band narrower than a voxel punctures the surface between voxels
      if sdf_trunc < voxel_size:
          raise ValueError(f"create_tsdf_mesh: sdf_trunc {sdf_trunc} is narrower than voxel_size {voxel_size}")
  ```

  - Add a `Raises:` entry. `BaseSfmCreator.__post_init__` gains, after the existing `min_registered_frac` check:

  ```python
          # bool is an int subclass, so it is refused explicitly
          if isinstance(self.num_threads, bool) or not (isinstance(self.num_threads, int) and self.num_threads >= 1):
              raise ValueError(f"num_threads must be an int >= 1, got {self.num_threads!r}")
  ```

  - Also confirm the existing `min_registered_frac` check refuses `True`; if not, add the same `bool` guard.
  - `ColmapCreator` gains `__post_init__` (and `HlocCreator` extends its own), with `PAIRINGS` imported from
    `collab_splats.pointcloud.sfm.sift_db`:

  ```python
      def __post_init__(self) -> None:
          """
          Refuse an unknown pairing mode or a non-positive pair count.
          """
          super().__post_init__()

          # Pairing must name one of sift_db's modes
          if self.pairing not in PAIRINGS:
              raise ValueError(f"pairing must be one of {PAIRINGS}, got {self.pairing!r}")

          # Pair counts are positive ints; bool is refused explicitly
          for key in ("overlap", "num_retrieved"):
              value = getattr(self, key)
              if isinstance(value, bool) or not (isinstance(value, int) and value >= 1):
                  raise ValueError(f"{key} must be an int >= 1, got {value!r}")
  ```

  - `InstantSfMCreator.__post_init__` (create it if absent, calling `super().__post_init__()` first):

  ```python
          # Seed must fit np.random.seed; a track needs two views to triangulate
          if self.random_seed is not None and not (isinstance(self.random_seed, int) and 0 <= self.random_seed < 2**32):
              raise ValueError(f"random_seed must be None or an int in [0, 2**32), got {self.random_seed!r}")
          if self.min_num_view_per_track is not None and not (isinstance(self.min_num_view_per_track, int) and self.min_num_view_per_track >= 2):
              raise ValueError(f"min_num_view_per_track must be None or an int >= 2, got {self.min_num_view_per_track!r}")
  ```

- [ ] **Step 3: Run** `pytest tests/mesh/test_tsdf.py tests/pointcloud/sfm -q`, then the gate.

- [ ] **Step 4: Commit** `feat(pointcloud,mesh): creators and create_tsdf_mesh validate their own args`.

### Task 7: collab-data `RcloneClient` additions (branch `tlb-3d-tools`)

Repo: `/workspace/collab-data`. Style there: loguru `logger`, `typing.List`, 4-space, black at 88. Commit with
`git commit --only`.

**Files:** `collab_data/data_dashboard/rclone_client.py`, `tests/data_dashboard/test_rclone_client.py` (new)

- [ ] **Step 1: Failing tests** against a local-filesystem remote. `RCLONE_CONFIG_LT_TYPE=local` registers a remote named
  `lt`:

  ```python
  """RcloneClient transport calls against a local-filesystem remote."""

  import pytest

  from collab_data.data_dashboard.rclone_client import RcloneClient, parse_percent


  @pytest.fixture
  def client(monkeypatch):
      monkeypatch.setenv("RCLONE_CONFIG_LT_TYPE", "local")
      return RcloneClient(remote_name="lt")


  @pytest.fixture
  def tree(tmp_path):
      src = tmp_path / "src"
      (src / "keep").mkdir(parents=True)
      (src / "keep" / "a.txt").write_text("a")
      (src / "skip.mp4").write_text("v")
      return src


  def test_run_returns_stdout(client, tree):
      out = client.run("lsf", f"lt:{tree}")
      assert "keep/" in out.split()


  def test_run_raises_on_failure(client, tmp_path):
      with pytest.raises(RuntimeError, match="rclone lsf"):
          client.run("lsf", "nosuchremote:x")


  def test_copy_dir_honors_excludes_and_streams_lines(client, tree, tmp_path):
      dst = tmp_path / "dst"
      lines = []
      client.copy_dir(f"lt:{tree}", str(dst), exclude=("*.mp4",), on_line=lines.append)
      assert (dst / "keep" / "a.txt").read_text() == "a"
      assert not (dst / "skip.mp4").exists()


  def test_check_true_when_destination_matches(client, tree, tmp_path):
      dst = tmp_path / "dst"
      client.copy_dir(f"lt:{tree}", str(dst))
      assert client.check(str(tree), f"lt:{dst}") is True


  def test_check_false_on_a_content_difference(client, tree, tmp_path):
      dst = tmp_path / "dst"
      client.copy_dir(f"lt:{tree}", str(dst))
      (dst / "keep" / "a.txt").write_text("changed")
      assert client.check(str(tree), f"lt:{dst}") is False


  def test_check_false_when_a_file_is_missing_remotely(client, tree, tmp_path):
      dst = tmp_path / "dst"
      client.copy_dir(f"lt:{tree}", str(dst), exclude=("*.mp4",))
      assert client.check(str(tree), f"lt:{dst}") is False
      assert client.check(str(tree), f"lt:{dst}", exclude=("*.mp4",)) is True


  def test_check_raises_when_it_cannot_run(client, tree):
      with pytest.raises(RuntimeError, match="rclone check"):
          client.check(str(tree), "nosuchremote:x")


  def test_parse_percent():
      assert parse_percent("Transferred: 1.2M / 3.4M, 42%, 1M/s") == 42
      assert parse_percent("Transferred: 1M / 1M, 100%, 1M/s") == 100
      assert parse_percent("no percent here") is None
  ```

  - Run: `cd /workspace/collab-data && python -m pytest tests/data_dashboard/test_rclone_client.py -q`, using the
    collab-data env named in its README. If there is none, use `/opt/venv/reconstruction/bin/python`, which has
    `collab_data` installed.
  - Expected: `ImportError` on `parse_percent`.

- [ ] **Step 2: Implement** (add `import re`, `import tempfile`, `from collections import deque`,
  `from pathlib import Path`, `from typing import Callable, Iterable, Optional`):

  ```python
  # rclone --stats-one-line progress: "Transferred: 1.2 GiB / 5.6 GiB, 21%, 45 MiB/s, ETA 1m"
  _PERCENT_RE = re.compile(r",\s*(\d{1,3})%")

  # Flags that make --stats lines reach stdout (rclone logs stats at INFO, below its default NOTICE)
  STATS_ARGS = ["--stats", "2s", "--stats-one-line", "--stats-log-level", "NOTICE"]


  def parse_percent(line: str) -> Optional[int]:
      """Integer transfer percentage in an rclone --stats-one-line line, or None."""
      match = _PERCENT_RE.search(line)
      if match is None:
          return None
      return min(100, int(match.group(1)))
  ```

  Methods on `RcloneClient`:

  ```python
      def run(self, *args: str) -> str:
          """Run one rclone command and return its stdout; raise RuntimeError on any non-zero exit."""
          result = subprocess.run(self._cmd(*args), capture_output=True, text=True)
          if result.returncode != 0:
              detail = (result.stderr or result.stdout).strip()
              raise RuntimeError(
                  f"rclone {args[0]} failed (exit {result.returncode}): {detail}"
              )
          return result.stdout

      def run_streaming(
          self, *args: str, on_line: Optional[Callable[[str], None]] = None
      ) -> None:
          """Run one rclone command, passing each output line to on_line; raise RuntimeError on failure."""
          # Keep the last non-progress lines, so the error carries rclone's own words
          tail: deque = deque(maxlen=5)
          with subprocess.Popen(
              self._cmd(*args),
              stdout=subprocess.PIPE,
              stderr=subprocess.STDOUT,
              text=True,
          ) as proc:
              for raw in proc.stdout or []:
                  line = raw.strip()
                  if not line:
                      continue
                  if parse_percent(line) is None:
                      tail.append(line)
                  if on_line is not None:
                      on_line(line)
              code = proc.wait()
          if code != 0:
              raise RuntimeError(f"rclone {args[0]} failed (exit {code}): {' | '.join(tail)}")

      def copy_dir(
          self,
          src: str,
          dst: str,
          exclude: Iterable[str] = (),
          on_line: Optional[Callable[[str], None]] = None,
          extra: Iterable[str] = (),
      ) -> None:
          """Copy a directory tree, skipping exclude globs; stream progress lines to on_line."""
          args = ["copy", src, dst, *extra, *STATS_ARGS]
          for pattern in exclude:
              args += ["--exclude", pattern]
          self.run_streaming(*args, on_line=on_line)

      def check(
          self,
          src: str,
          dst: str,
          exclude: Iterable[str] = (),
          extra: Iterable[str] = (),
      ) -> bool:
          """True when every src file exists in dst with equal content; raise when the check cannot run."""
          # --combined marks each file: "+" missing in dst, "-" only in dst, "*" differs, "=" equal
          with tempfile.TemporaryDirectory() as tmp:
              report = Path(tmp) / "combined.txt"
              args = ["check", src, dst, "--one-way", "--combined", str(report), *extra]
              for pattern in exclude:
                  args += ["--exclude", pattern]
              result = subprocess.run(self._cmd(*args), capture_output=True, text=True)
              lines = report.read_text().splitlines() if report.exists() else []

          mismatched = [line for line in lines if line[:2] in ("+ ", "- ", "* ")]
          if mismatched:
              logger.error(f"rclone check: {len(mismatched)} mismatched, e.g. {mismatched[:3]}")
              return False
          if result.returncode != 0:
              detail = (result.stderr or result.stdout).strip()
              raise RuntimeError(f"rclone check failed (exit {result.returncode}): {detail}")
          return True
  ```

  - `--one-way` makes `- ` lines impossible. They are still matched, so dropping `--one-way` later stays safe.
  - `extra` carries GCS-only flags (`--gcs-bucket-policy-only`, `--fast-list`) that a local remote rejects or ignores.

- [ ] **Step 3: Run** the new tests and the collab-data suite:
  `cd /workspace/collab-data && python -m pytest tests -q`. Expected: all new tests PASS and no regressions.

- [ ] **Step 4: Commit on `tlb-3d-tools`** (do NOT push)

  ```bash
  cd /workspace/collab-data && git add tests/data_dashboard/test_rclone_client.py && git commit --only collab_data/data_dashboard/rclone_client.py tests/data_dashboard/test_rclone_client.py -m "feat(rclone): run, run_streaming, copy_dir, check and parse_percent

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

### Task 8: Push collab-data (user gate) and bump the pin

- [ ] **Step 1: STOP.** Ask the user: "collab-data `tlb-3d-tools` has the RcloneClient commit `<sha>`; OK to push to
  origin?" Do nothing further until they say yes.
- [ ] **Step 2: On OK**, run `cd /workspace/collab-data && git push origin tlb-3d-tools`.
- [ ] **Step 3: Bump the pin.** In `pyproject.toml:109`, replace `file:///workspace/collab-data` with
  `collab_data @ git+https://github.com/BasisResearch/collab-data.git@<sha>`.
  - Run `cd /workspace/collab-splats/.worktrees/reconstructor-release && uv lock`.
  - Do not `uv sync`: a plain sync prunes extras from the shared venv.
  - Confirm the installed `collab_data` exposes the new API:
    `PYTHONPATH=... python -c "from collab_data.data_dashboard.rclone_client import parse_percent"`.
  - If the venv's `collab_data` is the editable `/workspace/collab-data`, it already does.
- [ ] **Step 4: Gate, then commit** `build: pin collab-data to the RcloneClient transport commit` (`pyproject.toml`,
  `uv.lock`).

### Task 9: `remote.py` on `RcloneClient`

**Files:**
- `collab_splats/remote.py`
- `collab_splats/dashboard/config.py`, `app.py:27`, `pipeline.py:38`, `operation_log.py:12`
- `tests/remote/test_remote.py`, `tests/dashboard/test_app.py:9`

- [ ] **Step 1: Move `PULL_EXCLUDES` verbatim** into `collab_splats/dashboard/config.py` (with its comment).
  - `app.py` / `pipeline.py` import it from `collab_splats.dashboard.config`.
  - `operation_log.py` imports `parse_percent` from `collab_data.data_dashboard.rclone_client` and calls
    `parse_percent(line)`.
  - `tests/dashboard/test_app.py:9` imports from `collab_splats.dashboard.config`.

- [ ] **Step 2: Rewrite `remote.py`.** Public methods keep their names, arguments and return types. New module body:

  ```python
  """
  Scene-level access to the curated and processed GCS buckets.

  - a scene id is the curated dir name; its processed outputs live under the same name
  - transport is collab-data's RcloneClient; this module knows buckets, ids and excludes
  """

  from __future__ import annotations

  import json
  import logging
  import re
  import time
  from pathlib import Path
  from typing import Callable

  from collab_data.data_dashboard.rclone_client import STATS_ARGS, RcloneClient

  logger = logging.getLogger(__name__)

  # One path segment, no leading "." or "-": ids are joined onto local output paths
  SCENE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

  # Never pushed: regenerable caches, build databases and the source video
  PUSH_EXCLUDES = (
      # Scene-root 2D patch cache; the leading slash keeps <backend>/semantics pushed
      "/semantics/**",
      # Source video, already in the curated bucket; bracketed because rclone globs are case sensitive
      "*.[Mm][Pp]4",
      "*.[Mm][Oo][Vv]",
      "*.[Aa][Vv][Ii]",
      # SIFT databases and hloc intermediates, rebuilt from images/
      "/*/colmap/instantsfm.db",
      "/*/colmap/colmap.db",
      "/*/colmap/hloc/**",
  )


  class SceneSource:
      """
      Curated scene listing and processed-output transfers for one rclone remote.

      - listings are memoized for listing_ttl seconds; transfers invalidate what they change
      - an absent GCS prefix lists as []; any rclone failure raises, so absent never looks like unreachable
      """

      def __init__(
          self,
          client: RcloneClient | None = None,
          *,
          curated: str = "environments-curated",
          processed: str = "environments-processed",
          video_exts: tuple[str, ...] = (".mp4", ".mov", ".avi"),
          listing_ttl: float = 60.0,
      ) -> None:
          """
          Bind a client and the two buckets.

          - a client that fails to construct degrades to None; check_available retries it

          Args:
              client: rclone client; None builds the default one.
              curated: bucket of input scenes, one video per scene dir.
              processed: bucket receiving pipeline outputs.
              video_exts: lower-case extensions accepted as a scene's video.
              listing_ttl: seconds a memoized listing stays fresh.
          """
          self.curated = curated
          self.processed = processed
          self.video_exts = video_exts
          self.listing_ttl = listing_ttl
          self._listing_cache: dict = {}

          # Build the default client; a broken rclone surfaces on first use, not here
          self._client = client
          if client is None:
              try:
                  self._client = RcloneClient()
              except Exception as exc:
                  logger.warning("rclone unavailable: %s", exc)
  ```

  Private helpers and bodies:

  ```python
      def _require_client(self) -> RcloneClient:
          """
          The client, or RuntimeError when rclone never came up.
          """
          if self._client is None:
              raise RuntimeError("rclone is not available")
          return self._client

      def _path(self, bucket: str, path: str = "") -> str:
          """
          `<remote>:<bucket>[/<path>]` for this client's remote.
          """
          remote = f"{self._require_client().remote_name}:{bucket}"
          return f"{remote}/{path.strip('/')}" if path else remote

      def _lsjson(self, bucket: str, path: str = "") -> list[dict]:
          """
          Entries under a bucket path; [] when absent, RuntimeError when rclone fails.
          """
          out = self._require_client().run("lsjson", self._path(bucket, path)).strip()
          return json.loads(out) if out else []
  ```

  - Keep `_cached` and `invalidate` as they are, with `self._listing_ttl` renamed to `self.listing_ttl`.
  - `check_available`, `list_scenes`, `scene_video`, `list_localization_dbs`, `list_processed_scenes` and
    `has_processed` keep their bodies. Replace `CURATED_BUCKET` / `PROCESSED_BUCKET` with `self.curated` /
    `self.processed`, and `_VIDEO_EXTS` with `self.video_exts`.
  - Transfers:

  ```python
      def fetch_video(self, scene: str, dest_dir: Path, on_line: Callable[[str], None] | None = None) -> Path:
          """
          Copy the scene's video into dest_dir.

          Args:
              scene: curated scene id.
              dest_dir: local directory, created if absent.
              on_line: receives each rclone output line.

          Returns:
              Local path of the video.
          """
          name = self.scene_video(scene)
          dest_dir = Path(dest_dir)
          dest_dir.mkdir(parents=True, exist_ok=True)
          local = dest_dir / name
          self._require_client().run_streaming(
              "copyto", self._path(self.curated, f"{scene}/{name}"), str(local), *STATS_ARGS, on_line=on_line
          )
          return local

      def pull_processed(
          self,
          scene: str,
          dest_dir: Path,
          excludes: tuple[str, ...] = (),
          on_line: Callable[[str], None] | None = None,
      ) -> Path:
          """
          Copy a scene's processed outputs into dest_dir.

          Args:
              scene: processed scene id.
              dest_dir: local directory, created if absent.
              excludes: rclone --exclude globs to skip.
              on_line: receives each rclone output line.

          Returns:
              dest_dir.
          """
          dest_dir = Path(dest_dir)
          dest_dir.mkdir(parents=True, exist_ok=True)
          self._require_client().copy_dir(self._path(self.processed, scene), str(dest_dir), exclude=excludes, on_line=on_line)
          return dest_dir

      def pull_zarr_members(
          self,
          scene: str,
          dest_dir: Path,
          members: tuple[str, ...],
          on_line: Callable[[str], None] | None = None,
      ) -> None:
          """
          Copy named pointcloud.zarr members that a pull excluded.

          Args:
              scene: processed scene id.
              dest_dir: local scene directory; members land under its pointcloud.zarr/.
              members: member names relative to the zarr root.
              on_line: receives each rclone output line.
          """
          dest = Path(dest_dir) / "pointcloud.zarr"
          dest.mkdir(parents=True, exist_ok=True)
          includes = [arg for member in members for arg in ("--include", f"{member}/**")]
          self._require_client().run_streaming(
              "copy", self._path(self.processed, f"{scene}/pointcloud.zarr"), str(dest), *includes, *STATS_ARGS, on_line=on_line
          )

      def push_outputs(self, local_dir: Path, scene: str, on_line: Callable[[str], None] | None = None) -> None:
          """
          Upload a scene directory to the processed bucket, minus PUSH_EXCLUDES.

          Args:
              local_dir: local scene directory.
              scene: processed scene id.
              on_line: receives each rclone output line.
          """
          flags = ["--gcs-bucket-policy-only", "--transfers", "8", "--retries", "3", "--timeout", "300s", "--contimeout", "60s"]
          self._require_client().copy_dir(
              str(local_dir), self._path(self.processed, scene), exclude=PUSH_EXCLUDES, on_line=on_line, extra=flags
          )

          # Drop memoized listings the push made stale
          self.invalidate(("has_processed", scene))
          self.invalidate(("list_localization_dbs", scene))
          self.invalidate(("list_processed_scenes",))

      def verify_push(self, local_dir: Path, scene: str) -> bool:
          """
          True when every pushed local file exists remotely with equal content.

          - gates the remote driver's local delete, so any failure answers False

          Args:
              local_dir: local scene directory.
              scene: processed scene id.

          Returns:
              False on a mismatch or when the check could not run.
          """
          flags = ["--gcs-bucket-policy-only", "--fast-list", "--checkers", "16"]
          logger.info("verifying %s against %s/%s", local_dir, self.processed, scene)
          try:
              return self._require_client().check(str(local_dir), self._path(self.processed, scene), exclude=PUSH_EXCLUDES, extra=flags)
          except (RuntimeError, OSError) as exc:
              logger.error("verify could not run for %s, local data kept: %s", scene, exc)
              return False
  ```

  - `verify_push` drops `on_line`: `check` reports once, at the end.
  - Grep callers: `grep -rn "verify_push(" collab_splats tests`. Drop the argument everywhere.
  - Deleted from the module: `CURATED_BUCKET`, `PROCESSED_BUCKET`, `_VIDEO_EXTS`, `PULL_EXCLUDES`, `_RCLONE_PCT_RE`,
    `_CHECK_MISMATCH_MARKER`, `_stats_args`, `parse_rclone_percent`, `_run_streaming`, and the `/*/colmap/database.db`
    exclude.
  - `collab_splats.remote` must stay torch-free. Verify:

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import sys, collab_splats.remote; print('torch' in sys.modules)"
  ```

  Expected: `False`.

- [ ] **Step 3: Migrate `tests/remote/test_remote.py`.**
  - Tests that patch `subprocess.Popen` / `subprocess.run` inside `collab_splats.remote` now inject a fake client. Add
    this to the test file:

  ```python
  class FakeClient:
      """Records every transport call; returns canned listings."""

      remote_name = "fake"

      def __init__(self, listings=None, fail=None):
          self.listings = listings or {}
          self.fail = fail
          self.calls = []

      def run(self, *args):
          self.calls.append(("run", args))
          if self.fail:
              raise RuntimeError(self.fail)
          return json.dumps(self.listings.get(args[-1], []))

      def run_streaming(self, *args, on_line=None):
          self.calls.append(("run_streaming", args))

      def copy_dir(self, src, dst, exclude=(), on_line=None, extra=()):
          self.calls.append(("copy_dir", src, dst, tuple(exclude), tuple(extra)))

      def check(self, src, dst, exclude=(), extra=()):
          self.calls.append(("check", src, dst, tuple(exclude)))
          if self.fail:
              raise RuntimeError(self.fail)
          return True
  ```

  - Keep every behavioral test: memo TTL, invalidation, absent-vs-unreachable, `SCENE_ID_RE` filtering, video pick,
    `check_available` retry, `PUSH_EXCLUDES` anchoring.
  - Rewrite each to assert on `FakeClient.calls` instead of argv.
  - Delete the tests of `parse_rclone_percent`, `_stats_args`, `_CHECK_MISMATCH_MARKER` classification,
    `_run_streaming` tail, and `PULL_EXCLUDES` (moved; its equality test moves to `tests/dashboard/test_app.py`).
  - Tests that ran real rclone against a local remote keep running. Construct them as
    `SceneSource(RcloneClient(remote_name="lt"), curated=<tmp>, processed=<tmp>)` with `RCLONE_CONFIG_LT_TYPE=local`.
  - Add:

  ```python
  def test_bucket_names_are_kwargs():
      client = FakeClient(listings={"fake:cur": [{"Name": "scene_a", "IsDir": True}]})
      source = SceneSource(client, curated="cur", processed="proc")
      assert source.list_scenes() == ["scene_a"]


  def test_verify_push_answers_false_when_the_check_cannot_run(tmp_path):
      source = SceneSource(FakeClient(fail="boom"))
      assert source.verify_push(tmp_path, "scene_a") is False
  ```

- [ ] **Step 4: Run** `pytest tests/remote tests/dashboard -q`, then the gate.

- [ ] **Step 5: Commit** `refactor(remote): SceneSource over collab-data RcloneClient; buckets become kwargs`.

### Task 10: Stage table, `outputs`, `done`, `result`, `run`

**Files:** `collab_splats/reconstructor.py`, `tests/reconstructor/test_run.py` (new),
`tests/reconstructor/test_reconstructor.py`, `tests/geometry/test_metrics.py:21,727-735`

This task renames the stage methods and rewires `run`. Stage bodies keep their old internals until Tasks 11-14, which
replace them one at a time.

- [ ] **Step 1: Failing tests** — `tests/reconstructor/test_run.py`:

  ```python
  """
  Stage table dispatch: order, dependencies, skip and refuse rules.
  """

  import pytest

  from collab_splats.reconstructor import LEAF_STAGES, STAGES, Reconstructor


  def _recon(tmp_path, monkeypatch, done=()):
      """Reconstructor whose stage methods record their calls and whose done() reads a set."""
      r = Reconstructor({"input_path": str(tmp_path / "v.mp4"), "output_path": str(tmp_path / "out"), "semantics": {"enabled": False}})
      calls = []
      for stage in STAGES:
          monkeypatch.setattr(r, stage, lambda s=stage: calls.append(s))
      monkeypatch.setattr(r, "done", lambda s: s in done)
      return r, calls


  def test_stage_order_is_dict_order():
      assert list(STAGES)[:2] == ["preproc", "pointcloud"]
      assert list(STAGES)[-1] == "reconstruction_quality_report"


  def test_leaf_stages_are_the_undepended_ones():
      assert LEAF_STAGES == {"refine", "semantics", "splats", "mesh", "localize", "reconstruction_quality_report"}


  def test_run_calls_named_stages_in_table_order(tmp_path, monkeypatch):
      r, calls = _recon(tmp_path, monkeypatch)
      r.run(["pointcloud", "preproc"])
      assert calls == ["preproc", "pointcloud"]


  def test_unknown_stage_raises(tmp_path, monkeypatch):
      r, _ = _recon(tmp_path, monkeypatch)
      with pytest.raises(ValueError, match="unknown stage"):
          r.run(["verify"])


  def test_unmet_dependency_raises(tmp_path, monkeypatch):
      r, _ = _recon(tmp_path, monkeypatch)
      with pytest.raises(ValueError, match="requires 'pointcloud'"):
          r.run(["mesh"])


  def test_dependency_on_disk_is_met(tmp_path, monkeypatch):
      r, calls = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud"})
      r.run(["mesh"])
      assert calls == ["mesh"]


  def test_named_leaf_already_done_raises_without_overwrite(tmp_path, monkeypatch):
      r, _ = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud", "mesh"})
      with pytest.raises(ValueError, match="already exists"):
          r.run(["mesh"])


  def test_named_leaf_already_done_reruns_with_overwrite(tmp_path, monkeypatch):
      r, calls = _recon(tmp_path, monkeypatch, done={"preproc", "pointcloud", "mesh"})
      r.run(["mesh"], overwrite=True)
      assert calls == ["mesh"]


  def test_done_non_leaf_is_skipped(tmp_path, monkeypatch):
      r, calls = _recon(tmp_path, monkeypatch, done={"preproc"})
      r.run(["preproc", "pointcloud"])
      assert calls == ["pointcloud"]


  def test_default_stages_follow_config(tmp_path, monkeypatch):
      r, calls = _recon(tmp_path, monkeypatch)
      r.config["mesh"]["enabled"] = False
      r.run()
      assert calls == ["preproc", "pointcloud", "reconstruction_quality_report"]


  def test_outputs_names_one_marker_per_file_stage(tmp_path):
      r = Reconstructor({"input_path": "x", "output_path": str(tmp_path)})
      assert set(r.outputs) == set(STAGES) - {"localize"}
      assert r.outputs["mesh"] == r.backend_dir / "mesh.ply"
  ```

  - `test_default_stages_follow_config` assumes base.yaml defaults: semantics on (disabled here), mesh on (disabled
    here), splats off, localization off, BA off. Read `configs/base.yaml` first and adjust the expected list if a
    default differs.
  - Expected: `ImportError` on `STAGES`.

- [ ] **Step 2: Implement.**
  - Replace `_STAGE_ORDER`, `_STAGE_DEPS` and the old `LEAF_STAGES` with:

  ```python
  # Stage -> stages it needs; dict order is run order
  STAGES: dict[str, tuple[str, ...]] = {
      "preproc": (),
      "pointcloud": ("preproc",),
      "refine": ("pointcloud",),
      "semantics": ("pointcloud",),
      "splats": ("pointcloud",),
      "mesh": ("pointcloud",),
      "localize": ("pointcloud",),
      "reconstruction_quality_report": ("pointcloud",),
  }

  # Stages nothing depends on; only these re-run alone against processed outputs
  LEAF_STAGES = frozenset(s for s in STAGES if not any(s in deps for deps in STAGES.values()))
  ```

  - Rename the methods: `preprocess` → `preproc`, `build_pointcloud` → `pointcloud`, `refine_poses` → `refine`,
    `extract_semantics` → `semantics`, `build_localization_db` → `localize`.
  - Each drops its `overwrite` and `result` parameters and its own skip check. `run` owns skipping now.
  - `preproc` keeps `shutil.rmtree(self.images_dir)` when the dir exists, because the stage always rebuilds.
  - `self.pointcloud = ...` assignments become `self._result = None`. `_resolve_result` and
    `_load_pointcloud_from_disk` are replaced by:

  ```python
      @property
      def outputs(self) -> dict[str, Path]:
          """
          Marker file per stage; a stage is done when its marker exists.

          - localize writes into pointcloud.zarr, so done() checks it separately
          """
          return {
              "preproc": self.images_dir,
              "pointcloud": self.pointcloud_zarr,
              "refine": self.backend_dir / "colmap" / "refine.json",
              "semantics": lifted_store_path(self.backend_dir / "semantics", self.config["semantics"]["extractor"]),
              "splats": self.backend_dir / "splats" / "ckpt.pt",
              "mesh": self.backend_dir / "mesh.ply",
              "reconstruction_quality_report": self.backend_dir / "reconstruction_quality_report.json",
          }

      def done(self, stage: str) -> bool:
          """
          Whether a stage's output is on disk.

          Args:
              stage: a key of STAGES.

          Returns:
              True when the marker exists; pointcloud also needs the COLMAP model.
          """
          if stage == "localize":
              return self.pointcloud_zarr.exists() and _localization_db_exists(self.pointcloud_zarr, self.config["localization"]["matcher"])
          if stage == "pointcloud":
              return self.pointcloud_zarr.exists() and self.colmap_model_dir.exists()
          return self.outputs[stage].exists()

      @property
      def result(self) -> PointcloudResult:
          """
          Points and cameras from pointcloud.zarr, loaded once per stage write.

          - dense per-frame arrays stay on disk; stages needing them load the zarr themselves
          """
          if self._result is None:
              self._result = PointcloudResult.load_zarr(
                  self.pointcloud_zarr, load_depth=False, load_world_points=False, load_confidence=False, load_pixel_indices=False
              )
          return self._result

      def run(self, stages: list[str] | None = None, overwrite: bool = False) -> None:
          """
          Run stages in table order.

          - None takes every stage the config enables; preproc, pointcloud and the report always run
          - a dependency is met by this run or by its output on disk
          - a done stage is skipped, except a named leaf, which raises unless overwrite

          Args:
              stages: stage names, any order.
              overwrite: rebuild stages whose output exists.

          Raises:
              ValueError: on an unknown stage, an unmet dependency, or a named leaf already done.
          """
          named = stages is not None

          # Default stage set from the config's enable flags
          if stages is None:
              enabled = {
                  "refine": self.config["pointcloud"]["bundle_adjustment"],
                  "semantics": self.config["semantics"]["enabled"],
                  "splats": self.config["splats"]["enabled"],
                  "mesh": self.config["mesh"]["enabled"],
                  "localize": self.config["localization"]["enabled"],
              }
              stages = [s for s in STAGES if enabled.get(s, True)]

          # Refuse unknown stages and unmet dependencies before any work
          unknown = sorted(set(stages) - set(STAGES))
          if unknown:
              raise ValueError(f"unknown stage(s) {unknown}; valid: {list(STAGES)}")

          for stage in stages:
              for dep in STAGES[stage]:
                  if dep not in stages and not self.done(dep):
                      raise ValueError(f"stage '{stage}' requires '{dep}', which is neither in this run nor on disk")

          # Run in table order; skip done stages, refuse a named done leaf
          for stage in [s for s in STAGES if s in stages]:
              if self.done(stage) and not overwrite:
                  if named and stage in LEAF_STAGES:
                      raise ValueError(f"stage '{stage}' output already exists; pass overwrite=True to replace it")
                  logger.info("stage %s done, skipping", stage)
                  continue

              logger.info("=== Stage: %s ===", stage)
              getattr(self, stage)()
  ```

  - `__init__` sets `self._result: PointcloudResult | None = None`. It keeps `self.viewer`.
  - Delete the `"verify"` special case and `_VERIFY_REMOVED`'s use in `run`. The constant goes in Task 15 with the
    config check.

- [ ] **Step 3: Migrate callers.**
  - `tests/geometry/test_metrics.py:21` becomes `from collab_splats.reconstructor import LEAF_STAGES, STAGES`.
  - Line 728 becomes `assert STAGES["reconstruction_quality_report"] == ("pointcloud",)`.
  - Line 729 becomes `assert list(STAGES).index("reconstruction_quality_report") > list(STAGES).index("pointcloud")`.
  - `evals/eval.py:254`: `recon.run_pipeline()` → `recon.run()`.
  - In `tests/reconstructor/*.py`, apply the method-rename table:

  | old | new |
  |---|---|
  | `r.run_pipeline(stages=S, overwrite=O)` | `r.run(S, overwrite=O)` |
  | `r.preprocess(overwrite=...)` | `r.preproc()` (an overwrite test now asserts via `run(["preproc"], overwrite=True)`) |
  | `r.build_pointcloud(...)` | `r.pointcloud()` |
  | `r.refine_poses(...)` | `r.refine()` |
  | `r.extract_semantics(result=..., overwrite=...)` | `r.semantics()` |
  | `r.build_localization_db(...)` | `r.localize()` |
  | `r.mesh(result=..., overwrite=...)` / `r.splats(overwrite=...)` / `r.reconstruction_quality_report(overwrite=...)` | same name, no args |
  | `r.pointcloud` used as a `PointcloudResult` | `r.result` |
  | `r._stage_output_exists(s)` | `r.done(s)` |
  | patch `Reconstructor._resolve_result` | set `r._result = <fake>` |
  | "skips when output exists" test calling a stage method directly | call `r.run([stage])` with the marker present and assert the refusal (leaf) or skip (non-leaf) |
  | `"verify" in stages` refusal test | `test_unknown_stage_raises` covers it; delete |

- [ ] **Step 4: Run** `pytest tests/reconstructor tests/geometry/test_metrics.py -q`, then the gate.

- [ ] **Step 5: Commit** `refactor(reconstructor): STAGES table with outputs/done/result and getattr dispatch`.

### Task 11: `preproc` stage body

**Files:** `collab_splats/reconstructor.py`, `tests/reconstructor/test_reconstructor_preprocess.py`,
`tests/preproc/test_undistort.py:296-360` (moves to `tests/reconstructor/test_preproc_stage.py`)

- [ ] **Step 1: Update the provenance tests first** (expected to FAIL against the old body):
  - `test_reconstructor_preprocess.py:84` and `:144`, and `test_reconstructor.py:316`: assert
    `manifest["provenance"]["input_path"] == str(video)` and `manifest["provenance"]["frame_selection"] == "fps"` in place
    of `video_path` / `method`.
  - Move `tests/preproc/test_undistort.py:296-360` (the three `extract_frames` / `_camera_provenance` tests) into
    `tests/reconstructor/test_preproc_stage.py`. Rewrite each to build a `Reconstructor` with `input_path=<frame dir>`,
    `preproc.undistort` set, then call `r.preproc()`.
  - The round-trip test asserts:

  ```python
      prov = frames.read_manifest(r.images_dir)["provenance"]
      camera = pycolmap.Camera(**prov["camera"])
      assert camera.model.name == "OPENCV"
  ```

  - If `pycolmap.Camera(**d)` refuses extra `todict()` keys, first check which keys `todict()` emits
    (`python -c "import pycolmap; print(pycolmap.Camera().todict().keys())"`). Then assert on
    `prov["camera"]["model"] == "OPENCV"` and `prov["camera"]["params"]` length instead. Do NOT reintroduce a
    provenance helper.
  - Drop the imports of `_camera_provenance` / `extract_frames` from `tests/preproc/test_undistort.py`.

- [ ] **Step 2: Replace the body.**
  - Delete `_camera_provenance`, `_apply_undistortion`, `_frames_from_dir`, `_frames_from_video` and `extract_frames`.
  - Import `plot_motion, plot_photometric` from `collab_splats.preproc.viz`, and `get_video_info` stays.

  ```python
      def preproc(self) -> None:
          """
          Select keyframes from the input into images/ with frames.json beside it.

          - video: measure quality, sample, write, plot the report with the kept frames marked
          - directory: every image, in filename order, source index = position
          - undistort self-calibrates from the written frames, then rewrites them
          """
          cfg = self.config["preproc"]
          input_path = Path(self.config["input_path"])
          provenance = {"input_path": str(input_path), **cfg}

          # Clear a previous store; the stage always rebuilds
          if self.images_dir.exists():
              shutil.rmtree(self.images_dir)

          # Directory input: take every image
          report = None
          if input_path.is_dir():
              rgbs = frames.read_frames(input_path)
              records = [{"frame_idx": i} for i in range(len(rgbs))]

          # Video input: measure the whole video, then sample from the eligible frames
          else:
              total = get_video_info(str(input_path))["total_frames"]
              report_path = self.images_dir.parent / "video_quality_report.json"
              report = load_video_quality(input_path, report_path, workers=cfg["n_workers"])
              common = {"report": report, "quality": cfg["quality"], "max_frames": cfg["max_frames"]}
              if cfg["frame_selection"] == "fps":
                  rgbs, records = sample_fps(str(input_path), fps=cfg["fps"], min_frames=cfg["min_frames"], on_empty_slot=cfg["on_empty_slot"], **common)
              elif cfg["frame_selection"] == "uniform":
                  rgbs, records = sample_uniform(str(input_path), **common)
              elif cfg["frame_selection"] == "optical_flow":
                  rgbs, records = sample_optical_flow(str(input_path), **common)
              else:
                  raise ValueError(f"preproc.frame_selection must be fps, uniform or optical_flow, got {cfg['frame_selection']!r}")

              if not len(rgbs):
                  raise ValueError(f"0 of {total} frames selected from {input_path}; see {report_path}")

          frames.write_frames(self.images_dir, rgbs, records, provenance)

          # Undistort from the written frames, then rewrite them on the new framing
          if cfg["undistort"]:
              camera = calibrate_camera(self.images_dir)
              rgbs, undistorted_camera = undistort_frames(rgbs, camera)
              provenance["camera"] = camera.todict()
              provenance["undistorted_camera"] = undistorted_camera.todict()
              frames.write_frames(self.images_dir, rgbs, records, provenance)

          # Plot the quality report with the kept frames marked
          if report is not None:
              selected = [r["frame_idx"] for r in records]
              plot_photometric(report, self.images_dir.parent, selected=selected)
              plot_motion(report, self.images_dir.parent, selected=selected)

          logger.info("preproc: %d frames in %s", len(records), self.images_dir)
  ```

  - The report path is the scene root, spelled in this stage only; there is no new property.
  - Before writing, read `configs/base.yaml`'s `preproc` block and match every key used above (`n_workers`,
    `frame_selection`, `fps`, `min_frames`, `max_frames`, `on_empty_slot`, `quality`, `undistort`) to the real names.
  - Directory input no longer uses cv2. `read_frames` reads RGB via `read_image`.
  - If `frame_paths(input_path)` filenames are not `frame_NNNNNN`, `read_frames(input_path)` (idxs=None) still works,
    because it never parses indices.
  - Do not pass `cfg` itself into `write_frames` provenance without the copy: `{**cfg}` above already copies. `quality`
    is a nested dict shared with config; `write_json` only reads it.
  - Remove imports left unused (`cv2` if no other user, `pycolmap` if no other user).

- [ ] **Step 3: Run** `pytest tests/reconstructor/test_reconstructor_preprocess.py tests/reconstructor/test_preproc_stage.py tests/preproc -q`,
  then the gate.

- [ ] **Step 4: Commit** `refactor(reconstructor): preproc stage composes preproc directly`.

### Task 12: `pointcloud` and `refine` stage bodies

**Files:** `collab_splats/reconstructor.py`, `tests/reconstructor/test_reconstructor.py`,
`test_reconstructor_loger_kwargs.py`, `test_reconstructor_mv_config.py`, `test_reconstructor_export.py`,
`test_sfm_stage.py`, `test_refine_stage.py`, `tests/pointcloud/test_loger_creator.py:821-860`

- [ ] **Step 1: Retarget the tests.**
  - Patches of `collab_splats.reconstructor._run_feedforward` become patches of `collab_splats.reconstructor.get_creator`,
    returning a stub creator class whose `create_pointcloud` returns the fake result.
  - `tests/reconstructor/_stubs.py` already has a stub creator; reuse it.
  - Tests that asserted `_run_feedforward` kwargs now assert the stub's constructor kwargs:
    `max_points`, `min_views`, `mv_rel_thresh`, `clean` and the backend block.
  - Delete the reserved-kwarg-clash test and the LoGeR `max_frames` warning test (`tests/pointcloud/test_loger_creator.py`
    near 858).
  - The LoGeR×LC refusal test near 821 moves to `tests/reconstructor/test_validate_config.py` (Task 15), unchanged in
    intent: `Reconstructor({... "pointcloud": {"backend": "loger", "loop_closure": True}})` raises `ValueError`.
  - Line 832 becomes `assert "loger" in BaseFeedforwardCreator._registry`.
  - Refine test: after `r.refine()`, the zarr attrs still carry `backend`, and `r.result.points` has the cleaned
    length. Remove any assertion on the `local_features` group surviving refine.

- [ ] **Step 2: Replace the bodies** (delete `_run_feedforward`, `_run_sfm`, `_DENSE_FIELDS`):

  ```python
      def pointcloud(self) -> None:
          """
          Reconstruct images/ into pointcloud.zarr, the COLMAP model and sparse_pc.ply.

          - feedforward optionally wraps the creator in loop closure, with a live viewer when viz is on
          - sfm is experimental; its creator writes its own subset and alignment attrs
          """
          cfg = self.config["pointcloud"]
          backend = cfg["backend"]

          # Build the feedforward creator, optionally inside loop closure
          if cfg["method"] == "feedforward":
              creator = get_creator(backend)(
                  max_points=cfg["max_points"],
                  min_views=cfg["min_views"],
                  mv_rel_thresh=cfg["mv_rel_thresh"],
                  clean=cfg["clean"]["enabled"],
                  **cfg[backend],
              )
              attrs = {"method": "feedforward", "backend": backend}

              lc = dict(cfg["loop_closure"])
              if lc.pop("enabled"):
                  creator = LoopClosure(base=creator, config=LoopClosureConfig(**lc) if lc else None)

                  # Viewer shows each loop edge live; heavy websocket dep, so imported here
                  if cfg["viz"]["enabled"]:
                      from collab_splats.viewer import Viewer

                      self.viewer = Viewer(port=cfg["viz"]["port"])
                      creator.viz = self.viewer
                      creator.config.loop_edge_timing = "live"

          # Build the sfm creator from its block
          else:
              warnings.warn("pointcloud.method='sfm' is experimental and not production-tested.", UserWarning, stacklevel=2)
              creator = SFM_CREATORS[backend](clean=cfg["clean"]["enabled"], max_points=cfg["max_points"], **cfg[backend])

          # Reconstruct; the creator writes the COLMAP model itself
          result = creator.create_pointcloud(self.images_dir, self.backend_dir, self.colmap_model_dir)
          if cfg["method"] == "sfm":
              attrs = {"backend": backend, **creator.attrs}

          result.save_zarr(self.pointcloud_zarr, extra_attrs=attrs)
          result.write_ply(self.backend_dir / "sparse_pc.ply")
          logger.info("pointcloud: %d pts in %s", len(result.points), self.pointcloud_zarr)

          # Free the model before the next stage loads its own
          del creator, result
          self._result = None
          pytorch_gc()
  ```

  - `cfg[backend]` must exist for every backend. Verify base.yaml has a block for each of
    `vggt_omega, vggtx, mapanything, loger, instantsfm, colmap, hloc`. If one is missing, add an empty `{}` block to
    base.yaml in this commit, not a `.get` fallback.
  - `lc` is already normalized to a dict with `enabled` by `validate_config`: Task 15 does that, so this task adds the
    normalization now. Move this snippet into `validate_config` in this commit:

  ```python
          # Normalize loop_closure to a dict carrying `enabled`
          lc = pc["loop_closure"]
          lc = dict(lc) if isinstance(lc, dict) else {"enabled": bool(lc)}
          lc.setdefault("enabled", True)
          unknown = set(lc) - {"enabled"} - {f.name for f in dataclasses.fields(LoopClosureConfig)}
          if unknown:
              raise ValueError(f"pointcloud.loop_closure has unknown keys {sorted(unknown)}")
          pc["loop_closure"] = lc
  ```

  - The existing `lc_enabled` line in `validate_config` becomes `lc_enabled = lc["enabled"]`.
  - Refine:

  ```python
      def refine(self) -> None:
          """
          Bundle-adjust the feedforward poses, then rewrite every pose-derived artifact.

          - points are re-derived, re-cleaned and re-capped, so their count can change
          - the zarr is rewritten whole, which drops any localization DB built on the old points
          """
          cfg = self.config["pointcloud"]
          if cfg["method"] == "sfm":
              raise ValueError("refine is not supported for pointcloud.method: sfm")

          # Refine poses, then re-derive the points under the new cameras
          ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=True)
          ba_cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir)
          ba = BundleAdjustment(ba_cfg)
          extrinsics, intrinsics = ba.refine(ff.images, ff.confidence, ff.world_points, ff.extrinsics, ff.model_intrinsics, ff.image_paths)
          ff = dataclasses.replace(ff, extrinsics=extrinsics, model_intrinsics=intrinsics, intrinsics=None)
          ff = ff.reproject()

          # Re-clean and re-cap under the refined cameras
          n_before = len(ff.points)
          ff = clean_pointcloud(ff, remove_outliers=cfg["clean"]["enabled"], max_points=cfg["max_points"])
          logger.info("refine: %d of %d pts kept after clean + cap", len(ff.points), n_before)

          # Rewrite the zarr with its provenance attrs, then the COLMAP model and PLY
          attrs = dict(zarr.open_group(str(self.pointcloud_zarr), mode="r").attrs)
          ff.save_zarr(self.pointcloud_zarr, extra_attrs=attrs)
          recon = ff.to_colmap()
          write_colmap_reconstruction(recon, self.colmap_model_dir)
          ff.write_ply(self.backend_dir / "sparse_pc.ply")
          self._result = None

          # Marker last: BA config and loss history
          marker = self.outputs["refine"]
          marker.parent.mkdir(parents=True, exist_ok=True)
          config = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(ba_cfg).items()}
          write_json(marker, {"config": config, "loss_history": ba.loss_history, "n_frames": len(ff.image_paths)})
  ```

  - Remove the now-unused `LZ4` import.

- [ ] **Step 3: Run** `pytest tests/reconstructor tests/pointcloud/test_loger_creator.py -q`, then the gate.
- [ ] **Step 4: Commit** `refactor(reconstructor): pointcloud and refine stages compose creators directly`.

### Task 13: `semantics` stage body

**Files:** `collab_splats/reconstructor.py`, `tests/reconstructor/test_reconstructor.py` (semantics tests),
`tests/dashboard/test_pipeline.py:281-295`

- [ ] **Step 1: Retarget the tests.**
  - Patches of `_extract_2d_features` / `_get_extractor` / `_lift_and_save` become patches of
    `collab_splats.reconstructor.extract_feature_cache`, `.BaseFeatureExtractor`, `.lift_features` and
    `.write_point_features`.
  - Delete `test_no_fourth_ae_policy_hides_in_a_parameter_default` from `tests/dashboard/test_pipeline.py`.

- [ ] **Step 2: Replace the body** (delete `_get_extractor`, `_extract_2d_features`, `_lift_and_save`; import
  `BaseFeatureExtractor` from `collab_splats.semantics.features` at top level):

  ```python
      def semantics(self) -> None:
          """
          Extract 2D features into the scene cache, lift them onto the points, optionally compress.

          - the 2D cache is reused when valid; extract_feature_cache owns that check
          - lifted rows follow the zarr's own frames, which may be a subset of images/
          """
          cfg = self.config["semantics"]

          # 2D features for every images/ frame
          self.semantics_cache_dir.mkdir(parents=True, exist_ok=True)
          extractor = BaseFeatureExtractor.get(cfg["extractor"])()
          cache = extract_feature_cache(extractor, self.images_dir, self.semantics_cache_dir)
          maps = load_feature_maps(cache)

          # Pick the zarr's frames out of the scene cache, in the zarr's order
          ff = PointcloudResult.load_zarr(self.pointcloud_zarr)
          all_idxs = [frames.frame_idx_from_path(p) for p in frames.frame_paths(self.images_dir)]
          row_of = {idx: row for row, idx in enumerate(all_idxs)}
          maps = [maps[row_of[frames.frame_idx_from_path(p)]] for p in ff.image_paths]
          lifted = lift_features(maps, ff)

          # Optional autoencoder compression
          ae = None
          if cfg["n_components"] is not None:
              lifted = lifted.to(get_device())
              ae = FeatureAutoencoder(input_dim=lifted.shape[-1], latent_dim=cfg["n_components"])
              ae.fit(lifted, epochs=cfg["max_epochs"], target_cosine=cfg["target_cosine"])
              lifted = ae.per_point_encode(lifted)

          write_point_features(self.backend_dir / "semantics", cfg["extractor"], to_numpy(lifted), ae)
  ```

  - A zarr frame missing from images/ raises `KeyError` from `row_of[...]`, which is the same contract `_store_rows` had.

- [ ] **Step 3: Run** `pytest tests/reconstructor tests/dashboard tests/semantics -q`, then the gate + `tests/semantics`.
- [ ] **Step 4: Commit** `refactor(reconstructor): semantics stage composes semantics directly`.

### Task 14: `mesh`, `splats`, `reconstruction_quality_report`, `localize` bodies

One commit per stage, in this order. Same loop for each: retarget tests → replace body → run
`pytest tests/reconstructor -q` → gate → commit.

**14a — mesh.** Delete `_run_tsdf_mesh` and the frame-count mismatch test (spec correction 9).
`tests/reconstructor/test_mask_sky.py` and `test_absent_confidence.py` patch `collab_splats.reconstructor.sky_masks` /
`create_tsdf_mesh` as before.

```python
    def mesh(self) -> None:
        """
        Fuse depth into a TSDF mesh, clean it, optionally texture it.

        - mesh.source picks the depth: pointcloud.zarr, or renders of the splats checkpoint
        - splats is never auto-run; source splats needs its ckpt.pt on disk
        """
        cfg = self.config["mesh"]

        # Splats source: renders carry their own frames, poses and intrinsics
        if cfg["source"] == "splats":
            from collab_splats.splats.checkpoint import render_tsdf_inputs

            ckpt = self.outputs["splats"]
            if not ckpt.exists():
                raise FileNotFoundError(f"mesh.source: splats needs {ckpt}; run the splats stage first")
            depths, rgbs, c2w, intrinsics, image_ids = render_tsdf_inputs(ckpt, self.images_dir)

        # Feedforward source: zarr depth lifted onto the zarr's own frames
        elif cfg["source"] == "feedforward":
            ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=False, load_world_points=False)
            image_ids = [frames.frame_idx_from_path(p) for p in ff.image_paths]
            rgbs = frames.read_frames(self.images_dir, image_ids)
            depths = frame_depths(ff, rgbs, cfg["conf_percentile"])
            c2w = invert_poses(ff.extrinsics)
            intrinsics = ff.intrinsics
        else:
            raise ValueError(f"mesh.source must be feedforward or splats, got {cfg['source']!r}")

        # Drop sky depth; report the share of valid depth removed
        if cfg["mask_sky"]:
            sky = sky_masks(self.images_dir, idxs=image_ids)
            if sky.shape != depths.shape:
                raise ValueError(f"sky masks {sky.shape} do not match depths {depths.shape}")
            dropped = np.count_nonzero(sky & (depths > 0)) / max(np.count_nonzero(depths), 1)
            depths = np.where(sky, 0.0, depths)
            logger.info("mesh.mask_sky: dropped %.2f%% of valid depth pixels", 100 * dropped)

        # Fuse, clean, texture
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = create_tsdf_mesh(
            depths, rgbs, c2w, intrinsics, self.backend_dir,
            voxel_size=cfg["voxel_size"], depth_trunc=cfg["depth_trunc"], sdf_trunc=cfg["sdf_trunc_mult"] * cfg["voxel_size"],
        )
        clean_repair_mesh(mesh_path, use_convex_hull=cfg["use_convex_hull"])
        if cfg["texture"]:
            create_texture_mesh(mesh_path, self.backend_dir / "texture", rgbs, c2w, intrinsics, voxel_size=cfg["voxel_size"])
```

- Mesh uses the zarr's own extrinsics (`ff.extrinsics`), not `self.result`'s. They are the same zarr, so the old
  mismatch check is gone by construction.
- Import `frame_depths` from `collab_splats.pointcloud.utils`. Drop `confidence_mask` and `upsample_depths` from the
  reconstructor imports once unused.
- Commit: `refactor(reconstructor): mesh stage composes mesh directly`.

**14b — splats.** Delete `_scene_frames` (and the `lru_cache` import). Delete
`test_splats_stage_rejects_frames_missing_from_feedforward`. Depth targets now come from `frame_depths` on the same
zarr rows as the images.

```python
    def splats(self) -> None:
        """
        Train Gaussian splats on the pointcloud's frames, poses and points.

        - depth supervision reuses the mesh stage's masked, frame-grid depth
        """
        from collab_splats.splats.trainer import SplatsConfig, train

        cfg = SplatsConfig.from_dict(self.config["splats"])
        result = self.result
        image_ids = [frames.frame_idx_from_path(p) for p in result.image_paths]
        rgbs = frames.read_frames(self.images_dir, image_ids)

        # Depth targets only when the depth loss is on
        depth_targets = None
        if "depth" in cfg.losses and cfg.losses["depth"]["weight"] > 0:
            ff = PointcloudResult.load_zarr(self.pointcloud_zarr, load_images=False, load_world_points=False)
            depth_targets = frame_depths(ff, rgbs, self.config["mesh"]["conf_percentile"])

        train(
            cfg, rgbs, result.extrinsics, result.intrinsics, result.points, result.colors, self.backend_dir / "splats",
            depth_targets=depth_targets, image_ids=image_ids,
        )
```

- `self.result` and the depth zarr are one file, so rows align.
- `frame_depths` returns float32. Check that `train` accepts float32 depth targets (the old code passed float32).
- Commit: `refactor(reconstructor): splats stage composes splats directly`.

**14c — reconstruction_quality_report.** Drop the stale-report (`"frames" not in ...`) check: it was a legacy-file
check. Delete its test.

```python
    def reconstruction_quality_report(self) -> None:
        """
        Reference-free error tables for the reconstruction, as one JSON.

        - runs no model; reads pointcloud.zarr and images/
        - the photometric table is null when images/ is empty
        """
        ff = PointcloudResult.load_zarr(self.pointcloud_zarr)

        # Keyframes in the zarr's row order, when the store has any
        images = None
        if frames.frame_paths(self.images_dir):
            image_ids = [frames.frame_idx_from_path(p) for p in ff.image_paths]
            images = frames.read_frames(self.images_dir, image_ids).astype(np.float32)

        names = [Path(str(p)).name for p in ff.image_paths]
        tables = compute_reconstruction_quality(
            ff.depth, ff.model_intrinsics, ff.intrinsics, ff.extrinsics, ff.original_coords, names, ff.confidence, images
        )
        scene = {
            "backend": self.config["pointcloud"]["backend"],
            "n_frames": len(ff.depth),
            "model_resolution": f"{ff.model_width}x{ff.model_height}",
            "image_width": int(ff.original_coords[0][4]),
            "zarr": str(self.pointcloud_zarr),
        }
        write_json(self.outputs["reconstruction_quality_report"], {"scene": scene, **tables})
```

Commit: `refactor(reconstructor): report stage reads frames by zarr rows`.

**14d — localize.** The stage body becomes:

```python
    def localize(self) -> None:
        """
        Rebuild the local-feature localization DB inside pointcloud.zarr.
        """
        cfg = self.config["localization"]
        _build_localization_db(self.pointcloud_zarr, cfg["matcher"], self.images_dir, top_k=cfg["top_k"], overwrite=True)
```

- The one permitted line in `_build_localization_db` replaces both `all_paths = ...` and
  `paths = [all_paths[row] for row in _store_rows(...)]` with:

  ```python
      paths = frames.frame_paths(images_dir, [frames.frame_idx_from_path(p) for p in ff.image_paths])
  ```

- Delete `_store_rows`.
  - **Amended (review decision):** `_store_rows` is kept as the single shared frame-id → images/ row lookup, used by semantics and `_build_localization_db`.
- `test_localization_db_overwrite.py` keeps passing: the stage always passes `overwrite=True`, and `run` owns the skip.
- Commit: `refactor(reconstructor): localize stage; drop _store_rows`.

### Task 15: `validate_config` trim, `__init__`, `__main__.py`, delete `wrapper/` and `docs/examples/`

Two commits: 15a (config) and 15b (CLI + deletions).

**15a — config.** Files: `collab_splats/reconstructor.py`, `tests/reconstructor/test_validate_config.py` (new, gathering
the config tests from `test_reconstructor.py`, `test_sfm_config.py`, `test_reconstructor_mv_config.py`).

- [ ] Delete `_VERIFY_REMOVED`, `_FEEDFORWARD_BACKENDS`, `_SFM_BACKENDS`, `_VALID_METHODS`, `_SFM_BLOCK_KEYS`, the
  `PAIRINGS` import, `_validate_sfm_block`, the `preprocessing` check, the `geometric_verification` check, the
  `sdf_trunc_mult` check and `launch_dashboard`.
- [ ] `__init__(self, config, base_config: Path | None = None)`:

  ```python
          # Deep-merge the caller's config over base.yaml, the single source of defaults
          base_config = Path(base_config) if base_config is not None else Path(__file__).parents[1] / "configs" / "base.yaml"
          defaults = yaml.safe_load(base_config.read_text()) or {}
          merged = merge({}, defaults, config)
          self.config = self.validate_config(merged)
          self._result: PointcloudResult | None = None
          self.viewer: Viewer | None = None
  ```

  Delete `DEFAULT_CONFIG_DIR`. Callers passing `config_dir=` pass `base_config=<dir>/base.yaml` (grep
  `config_dir=` in tests).
- [ ] `validate_config` body (the LC normalization from Task 12 stays):

  ```python
          # Required paths
          for key in ("input_path", "output_path"):
              if config.get(key) is None:
                  raise ValueError(f"Reconstructor config missing required field: '{key}'")

          # Method and backend must name a registered creator
          pc = config["pointcloud"]
          backends = {"feedforward": set(BaseFeedforwardCreator._registry), "sfm": set(SFM_CREATORS)}
          if pc["method"] not in backends:
              raise ValueError(f"pointcloud.method must be one of {sorted(backends)}, got {pc['method']!r}")
          if pc["backend"] not in backends[pc["method"]]:
              raise ValueError(f"pointcloud.backend must be one of {sorted(backends[pc['method']])} for method {pc['method']!r}, got {pc['backend']!r}")

          # <LC normalization block from Task 12>

          # Refuse combinations no stage can run
          if pc["bundle_adjustment"] and lc["enabled"]:
              raise ValueError("pointcloud.bundle_adjustment and pointcloud.loop_closure are mutually exclusive")
          if pc["method"] == "sfm" and pc["bundle_adjustment"]:
              raise ValueError("pointcloud.bundle_adjustment is not supported with method: sfm")
          if pc["method"] == "sfm" and lc["enabled"]:
              raise ValueError("pointcloud.loop_closure is not supported with method: sfm")
          if pc["backend"] == "loger" and lc["enabled"]:
              raise ValueError("pointcloud.loop_closure is not supported with backend: loger")

          return config
  ```

  - Keep the error-message fragments existing tests match on. Grep `match=` in the moved tests and keep each matched
    phrase in the new message.
- [ ] Tests:
  - Delete the ones for removed checks: `preprocessing` rename, `geometric_verification`, `sdf_trunc_mult`, and sfm
    block bounds (now in Task 6's file).
  - Add: `loop_closure: {"bogus": 1}` raises "unknown keys"; `loop_closure: True` normalizes to `{"enabled": True}`;
    `{"window": ...}` without `enabled` normalizes `enabled` to True (use a real `LoopClosureConfig` field name).
- [ ] Gate; commit `refactor(reconstructor): validate_config keeps only cross-field checks`.

**15b — CLI.** Files:
- Create `collab_splats/__main__.py` and `tests/reconstructor/test_cli.py`.
- Delete `collab_splats/wrapper/`, `docs/examples/{reconstruct,run_pipeline,run_pipeline_remote}.py`,
  `tests/examples/`, `tests/scripts/test_reconstruct.py`, `tests/remote/test_rerun.py`, `tests/reconstructor/test_batch.py`.
- Modify `pyproject.toml` `[project.scripts]`.

- [ ] **Step 1: Failing CLI tests** (`tests/reconstructor/test_cli.py`), using a stub `Reconstructor`:

  ```python
  """
  reconstruct local / remote argument handling with a stubbed Reconstructor.
  """

  import pytest
  import yaml

  from collab_splats import __main__ as cli


  class StubRecon:
      """Records the config and run args; optionally fails."""

      made = []

      def __init__(self, config, base_config=None):
          self.config = {**config, "pointcloud": {"backend": "vggt_omega", **config.get("pointcloud", {})}}
          self.viewer = None
          StubRecon.made.append(self)

      def run(self, stages=None, overwrite=False):
          self.ran = (stages, overwrite)
          if "fail" in self.config["input_path"]:
              raise RuntimeError("boom")


  @pytest.fixture(autouse=True)
  def stub(monkeypatch):
      StubRecon.made = []
      monkeypatch.setattr(cli, "Reconstructor", StubRecon)


  def test_local_writes_each_input_to_output_root_by_stem(tmp_path):
      video = tmp_path / "C0043.MP4"
      video.touch()
      code = cli.main(["local", str(video), "--output-root", str(tmp_path / "out")])
      assert code == cli.EXIT_OK
      assert StubRecon.made[0].config["output_path"] == str(tmp_path / "out" / "C0043")
      assert (tmp_path / "out" / "C0043" / "run_config.yaml").exists()


  def test_local_refuses_two_inputs_sharing_a_stem(tmp_path):
      (tmp_path / "a").mkdir()
      (tmp_path / "b").mkdir()
      (tmp_path / "a" / "x.mp4").touch()
      (tmp_path / "b" / "x.mov").touch()
      with pytest.raises(SystemExit):
          cli.main(["local", str(tmp_path / "a" / "x.mp4"), str(tmp_path / "b" / "x.mov"), "--output-root", str(tmp_path)])
      assert StubRecon.made == []


  def test_local_continues_past_a_failing_input(tmp_path):
      good, bad = tmp_path / "good.mp4", tmp_path / "fail.mp4"
      good.touch()
      bad.touch()
      assert cli.main(["local", str(bad), str(good), "--output-root", str(tmp_path / "o")]) == cli.EXIT_SCENE_FAILED
      assert len(StubRecon.made) == 2


  def test_set_overrides_parse_yaml_values_on_dotted_keys(tmp_path):
      video = tmp_path / "v.mp4"
      video.touch()
      cli.main(["local", str(video), "--output-root", str(tmp_path), "--set", "mesh.voxel_size=0.02", "--set", "semantics.enabled=false"])
      config = StubRecon.made[0].config
      assert config["mesh"]["voxel_size"] == 0.02 and config["semantics"]["enabled"] is False


  def test_config_file_merges_under_set(tmp_path):
      video = tmp_path / "v.mp4"
      video.touch()
      override = tmp_path / "o.yaml"
      override.write_text(yaml.safe_dump({"mesh": {"voxel_size": 0.05, "texture": True}}))
      cli.main(["local", str(video), "--output-root", str(tmp_path), "--config", str(override), "--set", "mesh.voxel_size=0.02"])
      assert StubRecon.made[0].config["mesh"] == {"voxel_size": 0.02, "texture": True}


  def test_stages_are_split_and_passed(tmp_path):
      video = tmp_path / "v.mp4"
      video.touch()
      cli.main(["local", str(video), "--output-root", str(tmp_path), "--stages", "preproc, pointcloud", "--overwrite"])
      assert StubRecon.made[0].ran == (["preproc", "pointcloud"], True)


  def test_remote_rejects_unsafe_scene_ids(tmp_path):
      with pytest.raises(SystemExit):
          cli.main(["remote", "../escape", "--output-root", str(tmp_path)])


  def test_remote_needs_ids_or_all(tmp_path):
      with pytest.raises(SystemExit):
          cli.main(["remote", "--output-root", str(tmp_path)])
  ```

  - Port every behavioral test from `tests/examples/test_run_pipeline_remote.py`, `tests/remote/test_rerun.py`,
    `tests/reconstructor/test_batch.py` and `tests/scripts/test_reconstruct.py` that still describes kept behavior:
    - fetch → run → push → verify → delete;
    - verify False keeps local;
    - `--keep-local`;
    - abort + SKIPPED rows on a dead remote;
    - `EXIT_REMOTE_UNAVAILABLE` when listing fails;
    - `EXIT_NOTHING_TO_DO`;
    - leaf re-run pulls processed and drops the re-run sections;
    - backend mismatch raises;
    - missing / empty `run_config.yaml` raises;
    - `run_config.yaml` always rewritten.
  - Retarget each to `cli._run_remote(source, scenes, args)` / `cli._prepare_scene(...)` with a fake `SceneSource`.
    `tests/remote/test_rerun.py` and `test_run_pipeline_remote.py` hold such a fake: move it.
  - Delete the tests of `collect_videos`, date-dir `scene_output_dir`, `--config-dir` and `--name`.

- [ ] **Step 2: Implement `collab_splats/__main__.py`.**

  ```python
  """
  reconstruct: run the pipeline on local inputs or on scenes in the curated bucket.

  - local: each video or frame directory writes to <output-root>/<stem>
  - remote: each scene id is fetched, run, pushed, verified, then deleted locally
  - a leaf-only --stages set re-runs remote scenes from their processed outputs
  - exit codes: 0 ok, 1 a scene failed, 2 nothing to do, 3 rclone unreachable (batch aborted)
  """

  from __future__ import annotations

  import argparse
  import logging
  import shutil
  import sys
  from pathlib import Path

  import yaml
  from mergedeep import merge

  from collab_splats.reconstructor import LEAF_STAGES, Reconstructor
  from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE, SceneSource

  logger = logging.getLogger(__name__)

  EXIT_OK = 0
  EXIT_SCENE_FAILED = 1
  EXIT_NOTHING_TO_DO = 2
  EXIT_REMOTE_UNAVAILABLE = 3


  ########################################
  # Shared helpers
  ########################################


  def _parse_overrides(pairs: list[str]) -> dict:
      """
      Nested dict from `key.sub=value` strings; each value is parsed as YAML.
      """
      out: dict = {}
      for pair in pairs:
          if "=" not in pair:
              raise ValueError(f"--set expects key=value, got {pair!r}")
          key, value = pair.split("=", 1)
          node = out
          for part in key.split(".")[:-1]:
              node = node.setdefault(part, {})
          node[key.split(".")[-1]] = yaml.safe_load(value)
      return out


  def _run_scene(config: dict, args: argparse.Namespace) -> Reconstructor:
      """
      Build one Reconstructor, record its config beside the outputs, run it.
      """
      recon = Reconstructor(config, base_config=args.base_config)
      output = Path(recon.config["output_path"])
      output.mkdir(parents=True, exist_ok=True)

      # Always rewrite: the recorded config must be the one that ran
      with open(output / "run_config.yaml", "w") as f:
          yaml.dump(recon.config, f, default_flow_style=False, sort_keys=False)

      recon.run(args.stages, overwrite=args.overwrite)
      return recon


  def _summarize(results: list[tuple[str, str, str]]) -> None:
      """
      Log one line per scene.
      """
      logger.info("==== Summary ====")
      for name, status, info in results:
          logger.info("%s: %s (%s)", status, name, info)


  ########################################
  # Local
  ########################################


  def _run_local(args: argparse.Namespace) -> int:
      """
      Run every input; one failure does not stop the rest.
      """
      results = []
      last = None
      for path in args.inputs:
          config = merge({}, args.overrides, {"input_path": str(path), "output_path": str(args.output_root / path.stem)})
          try:
              last = _run_scene(config, args)
              results.append((path.name, "OK", str(args.output_root / path.stem)))
          except Exception as exc:
              logger.exception("input failed: %s", path)
              results.append((path.name, "FAIL", str(exc)))

      _summarize(results)

      # Keep the last viser viewer up for inspection
      if args.keep_viewer and last is not None and last.viewer is not None:
          logger.info("--keep-viewer: viser server staying up (Ctrl-C to exit)")
          last.viewer.serve_forever()
      elif args.keep_viewer:
          logger.info("--keep-viewer: no viewer was created; nothing to keep alive")

      return EXIT_SCENE_FAILED if any(s == "FAIL" for _, s, _ in results) else EXIT_OK


  ########################################
  # Remote
  ########################################


  def _is_rerun(stages: list[str] | None) -> bool:
      """
      True when every stage is a leaf, so processed outputs are enough to run them.
      """
      return bool(stages) and set(stages) <= LEAF_STAGES


  def _prepare_scene(source: SceneSource, scene: str, scene_dir: Path, args: argparse.Namespace) -> dict:
      """
      Fetch a scene's inputs and return its config.

      - full runs fetch the curated video
      - leaf re-runs pull the processed scene and reuse its run_config.yaml minus the re-run sections
      """
      config = merge({}, args.overrides, {"output_path": str(scene_dir)})

      # Full run: start from the curated video
      if not _is_rerun(args.stages):
          video = source.fetch_video(scene, scene_dir, on_line=logger.info)
          config["input_path"] = str(video)
          return config

      # Leaf re-run: pull the processed scene
      if not source.has_processed(scene):
          raise FileNotFoundError(f"{scene} has no processed outputs; run the full pipeline first")
      source.pull_processed(scene, scene_dir, on_line=logger.info)

      # The pulled run_config names the backend every artifact path is built from
      run_cfg = scene_dir / "run_config.yaml"
      if not run_cfg.exists():
          raise FileNotFoundError(f"{scene}: pulled scene has no run_config.yaml; backend is unknowable")
      pulled = yaml.safe_load(run_cfg.read_text()) or {}
      backend = pulled.get("pointcloud", {}).get("backend")
      if not backend:
          raise ValueError(f"{scene}: pulled run_config.yaml has no pointcloud.backend")

      # A typed backend must agree with the data
      asked = args.overrides.get("pointcloud", {}).get("backend")
      if asked and asked != backend:
          raise ValueError(f"{scene} was built with backend '{backend}', --config asks for '{asked}'")

      # Re-run sections come fresh from base.yaml + overrides
      for stage in args.stages:
          pulled.pop("localization" if stage == "localize" else stage, None)

      logger.info("%s: re-run %s from processed (backend=%s)", scene, ",".join(args.stages), backend)
      return merge(pulled, config)


  def _remove_scene_dir(scene_dir: Path) -> bool:
      """
      Delete a scene dir; False with a warning when anything survived.
      """
      shutil.rmtree(scene_dir, ignore_errors=True)
      if scene_dir.exists():
          logger.warning("failed to remove local scene dir: %s", scene_dir)
          return False
      logger.info("removed local scene dir %s", scene_dir)
      return True


  def _run_remote(source: SceneSource, args: argparse.Namespace) -> int:
      """
      Fetch, run, push, verify and delete each scene; abort when rclone dies.
      """
      # Work list: named ids, else the bucket a leaf re-run or full run reads from
      scenes = list(args.scenes)
      if not scenes:
          try:
              scenes = source.list_processed_scenes() if _is_rerun(args.stages) else source.list_scenes()
          except Exception as exc:
              logger.error("cannot list scenes; rclone is not working: %s", exc)
              return EXIT_REMOTE_UNAVAILABLE
      if not scenes:
          logger.error("no scenes to process")
          return EXIT_NOTHING_TO_DO

      results = []
      aborted = False
      for i, scene in enumerate(scenes):
          logger.info("=== Scene: %s ===", scene)
          scene_dir = args.output_root / scene
          failure = None
          try:
              config = _prepare_scene(source, scene, scene_dir, args)
              _run_scene(config, args)
              source.push_outputs(scene_dir, scene, on_line=logger.info)

              # Delete only after a verified push
              if not source.verify_push(scene_dir, scene):
                  failure = "push verification failed; local data kept"
              elif args.keep_local:
                  logger.info("--keep-local: leaving %s in place", scene_dir)
              elif not _remove_scene_dir(scene_dir):
                  failure = f"push verified but local dir remains: {scene_dir}"
          except Exception as exc:
              logger.exception("scene failed: %s", scene)
              logger.warning("local scene dir retained for %s: %s", scene, scene_dir)
              failure = str(exc)

          if failure is None:
              results.append((scene, "OK", str(scene_dir)))
              continue
          results.append((scene, "FAIL", failure))

          # A dead remote aborts the batch; the rest are reported SKIPPED
          if not source.check_available():
              aborted = True
              logger.error("rclone is not working; aborting after %s", scene)
              results.extend((s, "SKIPPED", "not attempted; rclone unreachable") for s in scenes[i + 1 :])
              break

      _summarize(results)
      logger.info("push excluded: %s", ", ".join(PUSH_EXCLUDES))
      if aborted:
          return EXIT_REMOTE_UNAVAILABLE
      return EXIT_SCENE_FAILED if any(s == "FAIL" for _, s, _ in results) else EXIT_OK


  ########################################
  # Entry point
  ########################################


  def _parser() -> argparse.ArgumentParser:
      """
      `reconstruct local ...` and `reconstruct remote ...` with shared run options.
      """
      shared = argparse.ArgumentParser(add_help=False)
      shared.add_argument("--output-root", type=Path, required=True, help="each scene writes to <output-root>/<name>")
      shared.add_argument("--config", type=Path, help="override YAML merged over base.yaml")
      shared.add_argument("--base-config", type=Path, help="base.yaml to merge over; default configs/base.yaml")
      shared.add_argument("--stages", help="comma-separated stages; default: every enabled stage")
      shared.add_argument("--overwrite", action="store_true", help="rebuild stages whose output exists")
      shared.add_argument("--set", action="append", default=[], dest="sets", metavar="KEY=VALUE", help="dotted config override")

      parser = argparse.ArgumentParser(prog="reconstruct", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
      sub = parser.add_subparsers(dest="command", required=True)

      local = sub.add_parser("local", parents=[shared], help="run on video files or frame directories")
      local.add_argument("inputs", nargs="+", type=Path, metavar="VIDEO|DIR")
      local.add_argument("--keep-viewer", action="store_true", help="keep the last viser viewer alive")

      remote = sub.add_parser("remote", parents=[shared], help="run on curated scene ids")
      remote.add_argument("scenes", nargs="*", metavar="SCENE")
      remote.add_argument("--all", action="store_true", help="every scene in the bucket")
      remote.add_argument("--keep-local", action="store_true", help="skip the post-push delete")
      return parser


  def main(argv: list[str] | None = None) -> int:
      """
      Parse arguments and run.

      Args:
          argv: arguments after the program name; None reads sys.argv.

      Returns:
          Process exit code.
      """
      logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
      parser = _parser()
      args = parser.parse_args(argv)

      # Stages and overrides, shared by both commands
      args.stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
      overrides = yaml.safe_load(args.config.read_text()) if args.config else {}
      args.overrides = merge({}, overrides or {}, _parse_overrides(args.sets))

      if args.command == "local":
          stems = [p.stem for p in args.inputs]
          duplicates = sorted({s for s in stems if stems.count(s) > 1})
          if duplicates:
              parser.error(f"inputs share an output name: {', '.join(duplicates)}")
          return _run_local(args)

      if not args.scenes and not args.all:
          parser.error("give one or more SCENE ids, or --all")
      bad = [s for s in args.scenes if not SCENE_ID_RE.match(s)]
      if bad:
          parser.error(f"not safe scene ids: {', '.join(bad)}")
      return _run_remote(SceneSource(), args)


  if __name__ == "__main__":
      sys.exit(main())
  ```

  - A frame directory's `stem` is its folder name.
  - Tests call `main([...])` and read its return value. The console script wraps it via `sys.exit` in the entry point,
    so `[project.scripts]` must point at a wrapper that exits.
  - Use `reconstruct = "collab_splats.__main__:main"`. setuptools' generated script calls `sys.exit(main())`, so the
    return value becomes the exit code.
  - Import `collab_splats.reconstructor` at top level, not lazily. `__main__` is only imported to run.

- [ ] **Step 3: Delete** the old drivers and wrapper:

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && git rm -rq collab_splats/wrapper docs/examples/reconstruct.py docs/examples/run_pipeline.py docs/examples/run_pipeline_remote.py tests/examples tests/scripts/test_reconstruct.py tests/remote/test_rerun.py tests/reconstructor/test_batch.py && grep -rn "collab_splats.wrapper\|docs/examples/run_pipeline\|docs/examples/reconstruct" collab_splats tests evals configs docs/source pyproject.toml
  ```

  - The grep must be empty, except in `configs/README.md` and `docs/`, which Task 16 fixes.
  - If `docs/examples/` holds other files, keep them.
  - Add `reconstruct = "collab_splats.__main__:main"` under `[project.scripts]` in `pyproject.toml`.

- [ ] **Step 4: Run** `pytest tests/reconstructor tests/remote -q`, then
  `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m collab_splats local --help` (expected: usage text, exit 0).
  Then the gate.

- [ ] **Step 5: Commit** `feat(cli): reconstruct local|remote replaces wrapper batch drivers and docs/examples scripts`.

### Task 16: Caller and docs sweep

**Files:** `configs/README.md` (lines 11-28, 73-103, 151-155, 250-274, 783, 793), `docs/known-test-failures.md`,
`docs/source/api/wrapper.rst` → `docs/source/api/reconstructor.rst` (plus the toctree entry that lists it), `evals/eval.py`
(verify), `docs/source/tutorials/03_splats/train_splats.ipynb`, `06_mesh/splats_mesh.ipynb` (prose check only).

- [ ] **Step 1:**
  - Replace every `python docs/examples/run_pipeline.py ...` with `reconstruct local ...`, and
    `run_pipeline_remote.py` with `reconstruct remote ...`.
  - Replace `collab_splats/remote/sources.py` with `collab_splats/remote.py`, and `run_pipeline()` with `run()`.
  - `wrapper.rst` → `reconstructor.rst` with `automodule:: collab_splats.reconstructor`, plus
    `collab_splats.remote` if it was documented.
  - `known-test-failures.md`: `tests/wrapper/...` → `tests/reconstructor/...`. Drop entries whose test was deleted.
  - Notebook prose that names `.splats()` / `.mesh()` stays valid. Grep the notebooks for `run_pipeline`,
    `build_pointcloud` and `collab_splats.wrapper` and fix code cells that call them.
- [ ] **Step 2: Build check**

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && grep -rn "run_pipeline\|collab_splats\.wrapper\|remote/sources\|build_pointcloud\|extract_semantics\|refine_poses\|build_localization_db" configs docs/source docs/known-test-failures.md evals collab_splats tests
  ```

  - Expected: no hits, except the dashboard's own `run_pipeline` (spec 2 scope) and historical CHANGELOG/spec text.
- [ ] **Step 3:** Gate + `tests/evals`; commit `docs: reconstruct CLI and reconstructor module paths`.

### Task 17: Docstring contract

**Files:** `tests/test_docstring_contract.py:25`

- [ ] **Step 1:** `MODULES = ("utils/io.py", "reconstructor.py", "remote.py", "__main__.py")`.
- [ ] **Step 2:** `pytest tests/test_docstring_contract.py -q`. Fix every violation in those three files: docstring
  shape, missing `Args:`/`Returns:`, annotations, one-line comments. Stay prose-only where possible, and never change
  the localization helpers' bodies. If `_localization_db_exists` / `_build_localization_db` fail on docstring shape
  alone, fixing their docstrings is allowed; their code is not touched.
- [ ] **Step 3:** Gate; commit `test(contract): hold reconstructor, remote and the CLI to the docstring contract`.

### Task 18: End-to-end comparison and wrap-up

- [ ] **Step 1: Pick the clip** — the shortest tutorial video under `data/tutorial/`. Record its path.
- [ ] **Step 2: New run** (tmux; no parallel heavy jobs):

  ```bash
  cd /workspace/collab-splats/.worktrees/reconstructor-release && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m collab_splats local <clip> --output-root $SCRATCH/e2e_new --set semantics.enabled=false
  ```

- [ ] **Step 3: Reference run on `clean/final`** from the main checkout, with the old driver and the same override:

  ```bash
  cd /workspace/collab-splats && printf 'semantics:\n  enabled: false\n' > $SCRATCH/nosem.yaml && /opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py --output-root $SCRATCH/e2e_old --config $SCRATCH/nosem.yaml <clip>
  ```

  - First confirm the main checkout is on `clean/final` and print `collab_splats.__file__`.
  - If the main checkout is dirty or on another branch, make a throwaway worktree of `clean/final` instead.
- [ ] **Step 4: Compare.**
  - Frame count: `len(frames.json["frames"])`. Must be equal.
  - Point count: `len(zarr["points"])`. Must be equal.
  - Mesh vertex count: within hull run-to-run noise. Run the new side 3× if it differs by more than 1%. The hull output
    is flaky, so compare 3 runs.
  - Report numbers in a table.
- [ ] **Step 5:** `cd /workspace/collab-splats/.worktrees/reconstructor-release && graphify update .`
- [ ] **Step 6:** Final gate with every test dir from Task 0. Report pass/fail/skip against the baseline.
- [ ] **Step 7:** Append a `reconstructor-release` entry to `docs/superpowers/CHANGELOG.md` only when the user asks to
  land. Do not merge; the user decides.

---

## Self-review notes

- Spec coverage:
  - Round 1 → Tasks 1-2.
  - Round 2 items 1-12 → Tasks 3, 4, 5, 6, 7-8, 9, 10, 11-14, 15a, 15b, 16, 17.
  - End-to-end → Task 18.
  - The spec's "stats parser reads JSON log", "`check --combined -`" and "`PULL_EXCLUDES` to `pipeline.py`" are
    superseded by spec corrections 1, 3 and 4.
- The spec's pointcloud row "→ `dataclasses.replace` drops dense fields → `to_colmap`" is superseded by correction 2.
  The stage drops its whole result and relies on the lazy `result` property.
- Type names used across tasks:
  - `STAGES`, `LEAF_STAGES`, `outputs`, `done`, `result`, `_result`, `run`;
  - `frame_paths(dir, idxs)`, `frame_depths(result, rgbs, conf_percentile)`;
  - `RcloneClient.run/run_streaming/copy_dir/check`, `STATS_ARGS`, `parse_percent`;
  - `SceneSource(client, *, curated, processed, video_exts, listing_ttl)`, `verify_push(local_dir, scene)`;
  - `cli.main(argv) -> int`, `_run_local`, `_run_remote(source, args)`, `_prepare_scene(source, scene, scene_dir, args)`,
    `_parse_overrides`.
