# Consistency Phase 2: Shared Low-Level Helpers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every image decode, float-to-uint8 conversion, JSON report write and zarr validity check that
[the consistency spec](../specs/2026-09-26-consistency-design.md) phase 2 lists goes through one helper
in `collab_splats/utils/io.py`. The four phase-1 carry-overs close too. No sibling-owned line moves.

**Architecture:**
- `collab_splats/utils/io.py` is born **byte-identical** to `clean/r4-report` `4e4a8ce6`, which already
  built the JSON half (`to_json_safe`, `write_json`). Consistency then appends an images section and a
  zarr section. At the rebase onto `clean/final` the identical add replays cleanly.
- `collab_splats/utils/__init__.py` becomes docstring-only, so `import collab_splats.utils.io` never
  pulls torch.
- Call sites migrate one module group per task. Library JSON sites belong to r4-report, and files a live
  sibling rewrites are deferred to a post-rebase table. Nothing here fights a sibling's hunk.

**Tech Stack:** numpy, opencv 4.13 (`cv2.imread`), zarr 3.1.6 (`zarr.codecs.BloscCodec`), pycolmap, pytest.

---

## Conventions for every task

- **SCRATCH** = `/tmp/claude-0/-workspace-collab-splats/8ee8b55d-1af6-4884-812a-77a06f046a7b/scratchpad` (2026-09-27: `/tmp/claude-0/` was lost in a container move).

- **Worktree:** `/workspace/collab-splats/.worktrees/consistency`, branch `clean/consistency`, tip at plan
  time `ca526570`. Every command starts with `cd /workspace/collab-splats/.worktrees/consistency &&`
  because the cwd resets between Bash calls. Never touch the main checkout `/workspace/collab-splats`; it
  holds another session's uncommitted edits.
- **Python:** `PYTHONPATH=. /opt/venv/reconstruction/bin/python`
  - the venv's editable finder points at the MAIN tree; without `PYTHONPATH=.` pytest tests the wrong code
  - 2026-09-27: phase-1 stub plugin retired; the venv imports real `gsplat.losses` (1.5.3), `nvdiffrast`
    and `vismatch`. No `-p consistency_stubs`.
- **Proof line:** every gate begins with
  `PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"`,
  which must print a path under `.worktrees/consistency/`.
- **Gate form** (`$PY` below abbreviates the Python line above):
  ```bash
  cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest -p no:cacheprovider -q -rfE <paths>
  ```
  - never pipe pytest into `tail`/`head` (that eats the exit code)
  - never `--tb=no`; a failure in one file can make other files fail
  - no full-suite runs; the GPU is shared
- **Pass rule:** a package gate passes when its `FAILED`/`ERROR` node ids are a subset of the Task 0
  control list `SCRATCH/consistency-p2-control-failures.txt`.
  - 2026-09-27: the old env-failure list (gsplat 1.4.0, nvdiffrast, vismatch) is stale after the
    container move; Task 0 re-measures it. Still never run `tests/splats` (no task touches it).
- **Commit:** `git add <files> && git commit --only <files> -m "..."`. Messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - no rebase, reset, amend or stash; a mistake gets a second commit
  - check `ls .git/worktrees/consistency/sequencer 2>/dev/null` is empty before committing
- **Formatting:** never repo-wide black. Format only changed ranges:
  `black --line-ranges <a>-<b> <file>`, with isort run only on files whose import block changed.
  Line length is 120.
- **Style (CLAUDE.md):**
  - imports at the top of the file, and a block comment per logical block
  - a comment run of 3+ lines is a 5-10 word header line, then `- ` bullets
  - docstrings open with `"""` on its own line, then a one-line summary, `- ` bullets and
    `Args:`/`Returns:`/`Raises:`
  - US spelling (`color`, `normalize`, `center`)
  - flat test functions
- **TDD:** a new test that passes on the unfixed code is rejected, and every "verify it fails" step is
  mandatory. Pure-refactor tasks say so and use a pinning test that passes before and after.
- **Sibling-context rule:** before editing a file some sibling branch touches, run
  `git diff $(git merge-base HEAD <sibling>)..<sibling> -U3 -- <file>`. If the sibling changed any line
  within ±3 of your edit, skip that site and name it in the commit body under `Deferred (sibling hunk):`.
  The siblings to check are `clean/r4-report`, `clean/r4-lc`, `clean/r4-ba`, `clean/geometry-round3`,
  `clean/pointcloud-release`, `pc-lane-a`, `pc-lane-b` and `feat/sfm-backends`.

## Plan verification (2026-09-26)

What was verified against the code while writing this plan, and what was not:
- **Verified:**
  - `import collab_splats.utils` fails under `sys.modules['torch'] = None` ("import of torch halted"), so
    Task 1's RED step holds
  - `cv2`, `zarr` and `zarr.codecs.BloscCodec` import without torch
  - `git grep` finds **zero** `from collab_splats.utils import` importers; the only module-level
    reference is `tests/test_cu121_migration.py:134`, which imports `collab_splats.utils` by name
  - every site line number below was read at `ca526570`
  - every sibling claim was read from `git diff <merge-base>..<branch>`
- **Not dry-run:** the task code was not applied in a scratch copy. The RED/GREEN expectations are
  reasoned, not measured. Treat a surprising RED or GREEN as a stop-and-report, not a nudge.

## Corrections to the brief

- `clean_for_json` lives in `collab_splats/geometry/verification.py:432` at this tip, not in
  `transforms`. `metrics.py:21` imports it from there.
- The spec's `utils/io.py` JSON section **already exists** on `clean/r4-report` (`4e4a8ce6`
  "refactor(utils): one atomic nan-safe write_json for every JSON report"), under different names:
  `to_json_safe` (spec: `to_jsonable`) and `write_json` (always atomic, no `atomic=` kwarg). This plan
  adopts r4's names and bytes, so there are no two io.py files to merge.
- r4-report `4e4a8ce6` also migrates every *library* JSON writer (verification, metrics, frames, qa,
  splats/rendering, reconstructor) and `eval_verification.py`. Carry-over 4 ("replace `clean_for_json`,
  keeping NaN→null and its tests") is therefore done by r4-report. Here it only needs the rebase note in
  "Rebase notes".

## Dropped from the spec during planning

| Spec item | Why dropped |
|---|---|
| `geometry/colmap.py` | Every candidate is already a one-expression pycolmap idiom: `extrinsics_to_homogeneous(img.cam_from_world().matrix())` and `cameras[id].calibration_matrix()`. `pc-lane-b` adopted exactly those at `pointcloud/base.py`. A module wrapping one call each adds a layer and no value. |
| `camera_centers` as a new helper | r4-lc/r4-report replaced the LC wrapper's `_camera_centers_from_poses` with the idiom `invert_poses(P)[..., :3, 3]`. The branch that lands first already chose the idiom, so a second name would re-fork it. Task 12 moves the two remaining hand-rolled `-Rᵀt` copies onto that idiom instead. |
| `to_numpy` | `clean/pointcloud-release` (`4559ff3e`, `8b7a916a`) adds `utils/torch_utils.to_numpy` with tests, and `pc-lane-b` adopts it. The other two copies have different contracts: `localization/extractors.py:99` always casts to float32, and `mesh/io.py:145` is duck-typed so mesh/io stays torch-free. Adopting them is post-rebase work (deferred table). |
| `iter_images` | Its only caller would be `wrapper/reconstructor.py:769`, which r4-report rewrites. A generator over `read_image` is a one-line genexpr at the call site. |
| `stamp_valid` | A one-line `store.attrs.update(...)`. The only thing a helper would add is moving the write-validity-last rule, and that means rewriting `save_zarr`/localizer done-detection, which is phase 4. |
| JSON migration of `verification`, `metrics`, `frames`, `qa`, `splats/rendering`, `reconstructor`, `eval_verification` | Owned by r4-report `4e4a8ce6` (see Corrections). |
| Adding `utils` to docstring-contract `PACKAGES` | `utils/` as a package cannot meet the contract. `visualization.py` has inline `cv2` imports, and `image.py`, `torch_utils.py` and `progress.py` carry old-style docstrings. Task 4 instead adds a `MODULES = ("utils/io.py",)` hook, so only the new module is held to the contract, and marks `utils` RELEASED for that one file. |

## Census

"Wins" means the behavior the helper keeps. **OUT** flags an output change.

### read_image (cv2 decode, BGR→RGB, raise on None)

| Site | Today | What differs | Wins | Output |
|---|---|---|---|---|
| `preproc/frames.py:178` `read_frames` | `cv2.cvtColor(cv2.imread(p), BGR2RGB)` | missing file gives a cryptic cvtColor `!_src.empty()` | cv2 + named `FileNotFoundError` | same pixels |
| `semantics/utils.py:269` first probe | `Image.open(p).convert("RGB")` | PIL decode; ignores EXIF orientation | cv2 | PNG frames: identical. JPEG with EXIF rotation: ~~**OUT** (cv2 applies orientation)~~ identical in shape since the final review: `read_image` passes `IMREAD_IGNORE_ORIENTATION`, like PIL |
| `semantics/utils.py:287` loop | same as 269 | same | cv2 | same as 269 |
| `evals/scripts/eval.py:349` VDA frames | `np.asarray(Image.open(p).convert("RGB"))` | PIL | cv2 | as above |
| `evals/scripts/eval_similarity_calibration.py:147` | PIL, inline import at `:140` | PIL + inline import | cv2 → `Image.fromarray` for the torchvision transform | as above |
| `evals/scripts/eval_sky_mask.py:165` | `cv2.cvtColor(cv2.imread(...))` | no None check | helper | same |
| `wrapper/reconstructor.py:184`, `:769` | cv2 / PIL | | | **deferred**: r4-report rewrites reconstructor |

### to_uint8_hwc (float [0,1] → uint8 HWC, round, raise on [0,255])

| Site | Today | What differs | Wins | Output |
|---|---|---|---|---|
| `dashboard/pipeline.py:410-417` | guesses scale by `max() <= 1.0`, **truncates** (`astype`) | truncation biases down half a level | round | **OUT**: mesh vertex colors ±1 LSB |
| `localization/localizer.py:745-751` `_build_pairwise_refs` | guesses by `<= 1.5`, rounds | guess; stale comment "VGGT stores [0, 255]" | raise on > 1.5 | same for [0,1] input |
| `evals/scripts/eval_localization_parity.py:107-114` `_model_res_images` | mirror of the above | same | same | same |
| `evals/scripts/eval_splats.py:64-73` `_model_res_images` | guesses by `<= 1.0`, round, clip | guess | raise | same for [0,1] input |
| `evals/scripts/analyze_splats.py:296-297` | clip to [0,1], ×255, round | no scale check | raise | same for [0,1] input |
| `geometry/metrics.py:304-308` | guesses `rgb_scale` | | | **deferred**: r4 rewrites metrics |

The raise-on-[0,255] rule: every in-tree writer stores [0,1]. LoGeR asserts it at the source
(`loger.py:390`, the `a157421` bug class). An old zarr written before that fix now raises a named
`ValueError` instead of silently rendering a 255× blown-out image.

### JSON reports (`write_json`: nan→null, numpy→python, atomic)

All evals. Library sites belong to r4-report (see Dropped).

| Site | Today | What differs | Output |
|---|---|---|---|
| `analyze_splats.py:314` analysis.json | `write_text(json.dumps(..., indent=2))` | NaN written bare | **OUT** NaN→null |
| `ba_start_at_gt.py:180` report*.json | same | same | **OUT** NaN→null (`hist` fallback is `float("nan")`) |
| `eval.py:452` metrics.json | same | same | **OUT** NaN→null |
| `eval.py:637` IPC `_result_file` | same | same | round-trips: `np.array(None, dtype=float32)` is `nan` at `:833` |
| `eval.py:706` comparison.json | same | same | **OUT** |
| `eval.py:901` output_ate | same | same | **OUT** |
| `eval_compare.py:207` metrics.json | same | same | **OUT**; reader `format_markdown_rows` needs a None guard (Task 10) |
| `eval_localization_parity.py:258` | `default=float` | numpy scalars coerced, NaN bare | **OUT** NaN→null (carry-over 2) |
| `eval_multiview_conf.py:154` | `write_text(json.dumps)` | NaN bare | **OUT** |
| `eval_similarity_calibration.py:395` | `json.dump(f)` | NaN bare, non-atomic | **OUT** |
| `eval_sky_mask.py:242` stats.json | `write_text(json.dumps)` | NaN bare | **OUT** |
| `eval_splats.py:308` summary.json | same | same | **OUT** |
| `eval_verification.py:232` | `default=float` | | not migrated: deleted by geometry-round3 `e0d48ab8`; accept the delete at rebase (carry-over 2) |
| `eval.py:891`, `eval_verification.py:234-235` | `print(json.dumps(...))` | stdout, not a file | left as is |
| `geometry/bundle_adjustment.py:54` | `json.dumps(sort_keys=True)` cache-key hash | a hash, not a report | never touch |

### Zarr

| Site | Today | Helper | Output |
|---|---|---|---|
| `localization/localizer.py:310`, `:636` (+ import `:18`) | `lz4 = BloscCodec(cname="lz4")` | `LZ4` | same bytes |
| `semantics/utils.py:37` | `_UNREADABLE_STORE` tuple | `UNREADABLE_STORE` | same |
| `semantics/utils.py:256-264` feature-cache check | hand-rolled open, attr compare, except | `open_valid(path, {"extractor", "n_frames"})` | same decisions; log text changes |
| `semantics/utils.py:394` `point_features_cached` | `except _UNREADABLE_STORE` | `except UNREADABLE_STORE` (int compare stays; not an equality check) | same |
| `pointcloud/feedforward/base.py:163`, `geometry/bundle_adjustment.py:125-160` | BloscCodec / cache check | | **deferred**: pointcloud-release and r4 rewrite both |

### Carry-overs

| # | Site | Change | Output |
|---|---|---|---|
| 1 | `pointcloud/depth_align.py:265-269` | K by `rescale_intrinsics(K, original_coords, (h, w), to_original=False)` | ≤1 ULP float32 (float64 math, then cast) |
| 2 | `eval_localization_parity.py:258`, `eval_verification.py:232` | `write_json` | NaN→null |
| 3 | `pointcloud/feedforward/vggtx.py:48-85` portrait crop box | y scale `new_h/orig_h`, not `target/orig_w` | **OUT**: portrait box `tl_y` 423.25→421.82 and `cr_y` 1503.25→1498.18 for 1080×1920. The mesh RGB crop moves ~2-5 px, and `rescale_intrinsics` round-trips K exactly. |
| 4 | `clean_for_json` → `to_json_safe` | done by r4-report | none here |

**Carry-over 3, verified.** Upstream `load_and_preprocess_images(mode="crop")` resizes to width 518,
height `new_h = round(h * 518 / w / 14) * 14`, then crops `start_y = (new_h - 518) // 2`.
- For 1080×1920: `new_h = 924`, `start_y = 203`, and the true y scale is `924/1920 = 0.48125`. The code
  uses `518/1080 = 0.47963`, a 0.34% fy error and 3.25 px of cy.
- Landscape inputs are exact, because no height crop happens.
- It is fixed in phase 2 (Task 9), not phase 3. It is a 3-line body fix in a function no sibling touches
  (pc-lane-b's vggtx hunks start at line 116), and the phase-1 `rescale_intrinsics` gives it an exact
  round-trip test. Phase 3 is dedup, and this is a wrong number.

### Device / GC

| Site | Today | Helper |
|---|---|---|
| `localization/extractors.py:122` | `device or ("cuda" if ... else "cpu")` | `device or get_device()` |
| `localization/retrieval.py:49`, `:125` | same | same |
| `dashboard/pipeline.py:205` | `"cuda" if ... else "cpu"` | `get_device()` |
| `mesh/features.py:54` | same | same |
| `evals/scripts/eval_similarity_calibration.py:323` | same | same |
| `evals/scripts/eval_similarity_calibration.py:171` | `torch.cuda.empty_cache()` | `pytorch_gc()` (adds `gc.collect()` and a synchronize; frees the deleted SALAD model sooner) |
| `evals/scripts/refit_at_fixed_poses.py:44` | `torch.device("cuda" if ... )` | `torch.device(get_device())` |
| left as is | `vggtx.py:224`, `instantsfm.py:124`, `viewer.py:79` | these pick a device *index* or a module-specific fallback, not a `"cuda"/"cpu"` string |

## Site total

42 call sites plus the `utils/__init__` rewrite:
- JSON 12: analyze_splats 1, ba_start_at_gt 1, eval.py 4, eval_compare 1, eval_localization_parity 1,
  eval_multiview_conf 1, eval_similarity_calibration 1, eval_sky_mask 1, eval_splats 1
- `eval_verification` 1, applied as r4's exact line
- `read_image` 6
- `to_uint8_hwc` 5
- zarr 5
- carry-overs 2: depth_align K and the VGGT-X box
- `-Rᵀt` → `invert_poses` 2
- device/gc 8

## File map

| File | Change |
|---|---|
| `collab_splats/utils/__init__.py` | docstring only; no re-exports |
| `collab_splats/utils/io.py` | new; r4 JSON section verbatim, then Images and Zarr sections |
| `tests/utils/test_io.py` | new; r4 tests verbatim, then image, zarr and import-light tests |
| `tests/utils/test_utils_import_light.py` | new; the `collab_splats.utils` import-light subprocess test |
| `tests/test_docstring_contract.py` | `MODULES` hook; `utils` in RELEASED |
| `collab_splats/preproc/frames.py` | `read_frames` via `read_image` |
| `collab_splats/semantics/utils.py` | `read_image`, `open_valid`, `UNREADABLE_STORE`; drop PIL and `_UNREADABLE_STORE` |
| `collab_splats/localization/localizer.py` | `LZ4`; `_build_pairwise_refs` via `to_uint8_hwc` |
| `collab_splats/localization/extractors.py`, `retrieval.py` | `get_device` |
| `collab_splats/dashboard/pipeline.py` | `to_uint8_hwc`; `get_device` |
| `collab_splats/dashboard/localize.py` | `camera_centers` body → `invert_poses` |
| `collab_splats/mesh/features.py` | `get_device` |
| `collab_splats/pointcloud/depth_align.py` | K via `rescale_intrinsics` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | portrait crop box y scale |
| `evals/scripts/*.py` (11 files) | `write_json`, `read_image`, `to_uint8_hwc`, `get_device`/`pytorch_gc` |
| tests under `tests/{preproc,semantics,localization,dashboard,pointcloud,evals}` | one new test per behavior |

---

### Task 0: Control gate

**Files:** none

- [x] **Step 1: Confirm the tree**

```bash
cd /workspace/collab-splats/.worktrees/consistency && ls .git 2>/dev/null; git status --short | head; git log --oneline -1; ls third_party
```

Expected:
- a clean status, with tip `ca526570` or a later consistency docs commit
- `third_party/` holds the LoGeR, Video-Depth-Anything and hloc symlinks


- [x] **Step 2: Record the control failures**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && for p in tests/utils tests/preproc tests/semantics tests/localization tests/dashboard tests/pointcloud tests/geometry tests/mesh tests/evals tests/test_docstring_contract.py; do echo "== $p"; PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest -p no:cacheprovider -q -rfE $p > SCRATCH/p2-control-$(basename $p .py).log 2>&1; echo "exit=$?"; grep -E "^(FAILED|ERROR)" SCRATCH/p2-control-$(basename $p .py).log; done | tee SCRATCH/consistency-p2-control.txt; grep -hE "^(FAILED|ERROR)" SCRATCH/p2-control-*.log | sort > SCRATCH/consistency-p2-control-failures.txt; wc -l SCRATCH/consistency-p2-control-failures.txt
```

Expected:
- the proof line prints `.../.worktrees/consistency/collab_splats/__init__.py`
- the failures are only the known env ones: 5 nvdiffrast in `tests/mesh`, 17 `test_local_matcher` in
  `tests/localization`, plus whatever the logs show for the dashboard and evals stubs

Any other failure is a pre-existing break. Record it in the list and report it; don't fix it here.

---

### Task 1: `collab_splats.utils` imports without torch

**Files:**
- Modify: `collab_splats/utils/__init__.py`
- Create: `tests/utils/test_utils_import_light.py`

- [x] **Step 1: Write the failing test**

```python
"""
The utils package imports without torch, so io helpers stay usable torch-free.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _import_without_torch(module: str) -> subprocess.CompletedProcess:
    """
    Import `module` in a fresh interpreter where `import torch` raises.
    """
    code = f"import sys; sys.modules['torch'] = None; import {module}"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


def test_utils_package_imports_without_torch():
    proc = _import_without_torch("collab_splats.utils")
    assert proc.returncode == 0, proc.stderr
```

- [x] **Step 2: Verify it fails**

Run the gate on `tests/utils/test_utils_import_light.py`.
Expected: 1 failed, stderr containing `import of torch halted; None in sys.modules`.

- [x] **Step 3: Make `__init__` docstring-only**

Replace the whole of `collab_splats/utils/__init__.py` with:

```python
"""
Low-level helpers shared across collab_splats; import the submodule you need.

- io: image decode, float-to-uint8, JSON reports, zarr codec and validity (torch-free)
- torch_utils: device, GC, batching, registry
- image, progress, visualization: older helpers, imported by path
- no re-exports: a package-level import would pull torch into every torch-free caller
"""
```

- [x] **Step 4: Verify it passes and nothing imported the re-exports**

```bash
cd /workspace/collab-splats/.worktrees/consistency && rtk proxy git grep -nE "from collab_splats\.utils import|from \.utils import|from \.\.utils import|collab_splats\.utils\.(open_image|resize_image|get_device|pytorch_gc|infer_batch_size|batch_iterator|load_hf_weights|load_torchhub_model|RegistryMixin)\b" -- '*.py' '*.ipynb'
```

Expected: no output. Then run the gate on `tests/utils tests/test_cu121_migration.py`; there should be no
new failures against control.

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/utils/__init__.py tests/utils/test_utils_import_light.py && git commit --only collab_splats/utils/__init__.py tests/utils/test_utils_import_light.py -m "refactor(utils): docstring-only package init so utils imports without torch

Zero package-level importers of the re-exports (git grep).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `utils/io.py` JSON section, byte-identical to r4-report

**Files:**
- Create: `collab_splats/utils/io.py`, `tests/utils/test_io.py` (both from `4e4a8ce6`)

Pure adoption: no RED step, because the tests arrive with the code they test. Byte identity with r4 is
the point, since it makes the rebase replay a no-op add.

- [x] **Step 1: Check out r4's two files verbatim**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git show 4e4a8ce6:collab_splats/utils/io.py > collab_splats/utils/io.py && git show 4e4a8ce6:tests/utils/test_io.py > tests/utils/test_io.py && git diff --no-index --stat <(git show 4e4a8ce6:collab_splats/utils/io.py) collab_splats/utils/io.py; echo "identical=$?"
```

Expected: `identical=0`.

- [x] **Step 2: Verify the tests pass**

Run the gate on `tests/utils/test_io.py`. Expected: 4 passed.

- [x] **Step 3: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/utils/io.py tests/utils/test_io.py && git commit --only collab_splats/utils/io.py tests/utils/test_io.py -m "feat(utils): io.py JSON section — to_json_safe, atomic write_json

Byte-identical to clean/r4-report 4e4a8ce6 so the rebase onto clean/final
replays as an identical add. Later commits append images and zarr sections.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `utils/io.py` images section

**Files:**
- Modify: `collab_splats/utils/io.py`, `tests/utils/test_io.py`

- [x] **Step 1: Write the failing tests** (append to `tests/utils/test_io.py`; add `import cv2`,
  `import pytest` and the two new names to the file's top imports)

```python
########################################################################
# Images
########################################################################


def test_read_image_returns_rgb_not_bgr(tmp_path):
    """cv2 decodes BGR; a red pixel must come back red."""
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    cv2.imwrite(str(tmp_path / "r.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    out = read_image(tmp_path / "r.png")

    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    np.testing.assert_array_equal(out, rgb)


def test_read_image_names_a_missing_file(tmp_path):
    """cv2.imread returns None; the helper raises by name instead of a cvtColor assert."""
    with pytest.raises(FileNotFoundError, match="nope.png"):
        read_image(tmp_path / "nope.png")


def test_to_uint8_hwc_rounds_instead_of_truncating():
    """0.5/255 must land on 1, not 0: truncation biases every channel down half a level."""
    x = np.full((1, 3, 1, 2), 0.6 / 255.0, dtype=np.float32)
    x[0, :, 0, 1] = 1.0

    out = to_uint8_hwc(x, channels_first=True)

    assert out.shape == (1, 1, 2, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out[0, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(out[0, 0, 1], [255, 255, 255])


def test_to_uint8_hwc_channels_last_and_clip():
    x = np.array([[[-0.1, 0.5, 1.2]]], dtype=np.float32)  # (1, 1, 3) HWC

    out = to_uint8_hwc(x, channels_first=False)

    np.testing.assert_array_equal(out, [[[0, 128, 255]]])


def test_to_uint8_hwc_rejects_a_0_255_array():
    """A [0, 255] input is a stale-zarr contract break, not a scale to guess."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_uint8_hwc(np.full((1, 3, 2, 2), 200.0, dtype=np.float32), channels_first=True)
```

Top-of-file import becomes `from collab_splats.utils.io import read_image, to_json_safe, to_uint8_hwc, write_json`.

- [x] **Step 2: Verify they fail**

Run the gate on `tests/utils/test_io.py`. Expected: collection ERROR,
`ImportError: cannot import name 'read_image'`.

- [x] **Step 3: Implement**

In `collab_splats/utils/io.py`:

(a) Replace the module docstring (lines 1-6) with:

```python
"""
Torch-free IO helpers: images, JSON reports, zarr stores.

- images: read_image decodes RGB; to_uint8_hwc turns [0, 1] floats into uint8 HWC
- JSON: to_json_safe makes strict-JSON values; write_json writes them atomically
- zarr: one LZ4 codec, the unreadable-store error set, open_valid for cache checks
"""
```

(b) Add `import cv2` to the third-party import group (isort order: `cv2`, then `numpy as np`).

(c) Append after `write_json`:

```python
########################################################################
# Images
########################################################################


def read_image(path: str | Path) -> np.ndarray:
    """
    Decode one image file as RGB.

    - cv2 decode then BGR -> RGB, the same pixels read_frames has always returned
    - cv2.imread answers None for a missing or undecodable file; raised here by name

    Args:
        path: image file.

    Returns:
        (H, W, 3) uint8 RGB.

    Raises:
        FileNotFoundError: the file is missing or cv2 cannot decode it.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_uint8_hwc(images: np.ndarray, *, channels_first: bool) -> np.ndarray:
    """
    Float [0, 1] images as contiguous uint8, channels last.

    - rounds, then clips: truncation biases every channel down half a level
    - a [0, 255] input raises: the images convention is [0, 1], never a per-call guess

    Args:
        images: float images in [0, 1], (..., 3, H, W) or (..., H, W, 3).
        channels_first: True when the channel axis is third from last.

    Returns:
        (..., H, W, 3) uint8, C-contiguous.

    Raises:
        ValueError: the array's max is above 1.5, i.e. it is already [0, 255].
    """
    arr = np.asarray(images)

    # Slack above 1.0 for resize overshoot; anything past it is a [0, 255] array
    if arr.size and float(arr.max()) > 1.5:
        raise ValueError(f"images must be in [0, 1], got max {float(arr.max()):.1f} (a [0, 255] array?)")

    # Channel axis last, then round-clip-cast
    if channels_first:
        arr = np.moveaxis(arr, -3, -1)
    return np.ascontiguousarray(np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8))
```

- [x] **Step 4: Verify they pass**

Run the gate on `tests/utils`. Expected: all `test_io.py` tests pass and no new failures.

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/utils/io.py tests/utils/test_io.py && git commit --only collab_splats/utils/io.py tests/utils/test_io.py -m "feat(utils): io.read_image and io.to_uint8_hwc

read_image raises FileNotFoundError where cv2.imread returns None.
to_uint8_hwc rounds (not truncates) and rejects a [0, 255] input.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `utils/io.py` zarr section, import-light test, docstring contract

**Files:**
- Modify: `collab_splats/utils/io.py`, `tests/utils/test_io.py`, `tests/test_docstring_contract.py`

- [x] **Step 1: Write the failing tests** (append to `tests/utils/test_io.py`; add `import logging`,
  `import os`, `import subprocess`, `import sys`, `from pathlib import Path`, `import zarr` and the new
  names to the top imports)

```python
########################################################################
# Zarr
########################################################################


def _store(path: Path, **attrs) -> Path:
    """
    A one-array zarr store at `path` carrying `attrs`.
    """
    root = zarr.open(str(path), mode="w")
    root.create_array("x", shape=(2,), dtype="float32", compressors=LZ4)
    root.attrs.update(attrs)
    return path


def test_open_valid_returns_the_store_when_attrs_match(tmp_path):
    path = _store(tmp_path / "s.zarr", extractor="dino", n_frames=3)

    store = open_valid(path, {"extractor": "dino", "n_frames": 3})

    assert store is not None and store.attrs["n_frames"] == 3


def test_open_valid_rejects_a_missing_or_mismatched_store(tmp_path):
    path = _store(tmp_path / "s.zarr", extractor="dino", n_frames=3)

    assert open_valid(tmp_path / "absent.zarr", {"n_frames": 3}) is None
    assert open_valid(path, {"extractor": "dino", "n_frames": 4}) is None
    assert open_valid(path, {"never_written": 1}) is None


def test_open_valid_treats_corrupt_metadata_as_absent(tmp_path, caplog):
    path = tmp_path / "bad.zarr"
    path.mkdir()
    (path / "zarr.json").write_text("{not json")

    with caplog.at_level(logging.WARNING):
        assert open_valid(path, {"n_frames": 1}) is None
    assert "unreadable" in caplog.text


def test_open_valid_propagates_a_bug(tmp_path, monkeypatch):
    """Only UNREADABLE_STORE means stale; any other error is a bug and must surface."""
    path = _store(tmp_path / "s.zarr", n_frames=1)

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(zarr, "open", boom)
    with pytest.raises(RuntimeError, match="not a store error"):
        open_valid(path, {"n_frames": 1})


def test_io_imports_without_torch():
    """io is the torch-free layer: cv2, numpy, zarr only."""
    root = Path(__file__).resolve().parents[2]
    code = "import sys; sys.modules['torch'] = None; import collab_splats.utils.io"
    env = {**os.environ, "PYTHONPATH": str(root)}
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
```

- [x] **Step 2: Verify they fail**

Run the gate on `tests/utils/test_io.py`. Expected: collection ERROR,
`cannot import name 'LZ4'`.

(`test_io_imports_without_torch` would pass alone after Task 3. It is a guard, and a later edit that
imports torch into io.py must break it. Sanity-check it once: temporarily add `import torch` to io.py,
see it fail, and remove the line.)

- [x] **Step 3: Implement**

In `collab_splats/utils/io.py`:
- add `import logging` to the stdlib group
- add `import zarr` and `from zarr.codecs import BloscCodec` to the third-party group
- after the imports, add `logger = logging.getLogger(__name__)`

Then append:

```python
########################################################################
# Zarr
########################################################################

# One Blosc LZ4 codec for every array a pipeline store writes
LZ4 = BloscCodec(cname="lz4")

# Errors a missing, corrupt or half-written zarr store raises; anything else is a bug
UNREADABLE_STORE: tuple[type[Exception], ...] = (OSError, ValueError, KeyError, TypeError)


def open_valid(path: str | Path, expected: dict[str, Any]) -> zarr.Group | zarr.Array | None:
    """
    Open a zarr store read-only when its validity attrs match, else None.

    - writers stamp validity attrs last, so a crash mid-write leaves a store this rejects
    - missing path: None; mismatched attrs: None, logged at info
    - unreadable store (UNREADABLE_STORE): None with a warning; any other error propagates

    Args:
        path: zarr store directory.
        expected: attrs the store must carry, each compared with ==.

    Returns:
        The opened store, or None when it must be rebuilt.
    """
    path = Path(path)
    if not path.exists():
        return None

    # Open and read attrs; only store-shaped errors mean "rebuild"
    try:
        store = zarr.open(str(path), mode="r")
        attrs = dict(store.attrs)
    except UNREADABLE_STORE:
        logger.warning("Store at %s is corrupt or unreadable, treating it as absent", path)
        return None

    # Every expected attr must match exactly
    stale = {k: attrs.get(k) for k, v in expected.items() if attrs.get(k) != v}
    if stale:
        logger.info("Store at %s is stale (%s), treating it as absent", path, stale)
        return None
    return store
```

- [x] **Step 4: Hold io.py to the docstring contract**

In `tests/test_docstring_contract.py`:

(a) Below `PACKAGES = (...)` (line 22) add:

```python
# Single modules held to the contract before their whole package is
MODULES = ("utils/io.py",)
```

(b) In `_sources()`, before `return out`, add:

```python
    out += [ROOT / "collab_splats" / m for m in MODULES]
```

(c) At line 194 change `RELEASED` to `frozenset({"preproc", "semantics", "geometry", "utils"})` and add
this comment above it: `# utils: only its MODULES entries are sourced, never the whole package`.

(d) Update the module docstring bullet on line 10 to
`- extend PACKAGES (or MODULES, for one file) as the remaining modules are cleaned up`.

2026-09-27 as built (`87f24a99`): (c) was not applied. `clean/pointcloud-release` and `pc-lane-b` edit
`FIXED_FACTS` three lines below `RELEASED`, so the sibling-context rule skips that line. Instead
`_release_params` treats a `MODULES` entry as released (`released = pkg in RELEASED or str(rel) in MODULES`).

- [x] **Step 5: Verify**

Run the gate on `tests/utils tests/test_docstring_contract.py`. Expected:
- every `test_io.py` test passes
- every `collab_splats/utils/io.py::*` contract param passes (not xfail)
- no new failures against control

If a contract check fails on io.py, fix io.py. Do not loosen the check.

- [x] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/utils/io.py tests/utils/test_io.py tests/test_docstring_contract.py && git commit --only collab_splats/utils/io.py tests/utils/test_io.py tests/test_docstring_contract.py -m "feat(utils): io zarr section — LZ4, UNREADABLE_STORE, open_valid

io.py joins the docstring contract via a MODULES hook; utils as a whole
cannot yet (visualization inline imports, old-style docstrings).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: preproc — `read_frames` through `read_image`

**Files:**
- Modify: `collab_splats/preproc/frames.py:178`, `tests/preproc/test_frames.py`

Sibling check: r4-report `4e4a8ce6` edits frames.py's `_jsonable` (line 45) and import block. Line 178
is clear of it, but the import-block merge at rebase is trivial.

- [x] **Step 1: Write the failing test** (append to `tests/preproc/test_frames.py`; make sure `import pytest`
  is at the top)

```python
def test_read_frames_names_an_undecodable_frame(tmp_path):
    """A zero-byte PNG must raise by path, not as an opaque cvtColor assert."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame_000000.png").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="frame_000000.png"):
        fr.read_frames(images)
```

- [x] **Step 2: Verify it fails**

Run the gate on `tests/preproc/test_frames.py`.
Expected: 1 failed, `cv2.error ... !_src.empty()` rather than `FileNotFoundError`.

- [x] **Step 3: Implement**

- `frames.py:178` becomes `return np.stack([read_image(p) for p in paths])`
- add `from collab_splats.utils.io import read_image` to the first-party import group
- keep `import cv2`, since `write_frames` still encodes with it

- [x] **Step 4: Verify**

Run the gate on `tests/preproc tests/test_docstring_contract.py`. There should be no new failures.

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/preproc/frames.py tests/preproc/test_frames.py && git commit --only collab_splats/preproc/frames.py tests/preproc/test_frames.py -m "refactor(preproc): read_frames decodes through utils.io.read_image

An undecodable frame now raises FileNotFoundError naming the file.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`92699535`): RED 1 failed / 13 passed (`cv2.error ... !_src.empty()`); gate
`tests/preproc tests/test_docstring_contract.py` 500 passed, 20 xfailed, 55 xpassed, 0 failed. The
`read_image` import sits in the slot where `clean/geometry-round3` (ex r4-report) adds `write_json`; the
commit body carries the one-line merge note.

---

### Task 6: semantics — `read_image`, `open_valid`, `UNREADABLE_STORE`

**Files:**
- Modify: `collab_splats/semantics/utils.py` (lines 24, 37, 256-264, 269, 287, 394),
  `tests/semantics/test_semantics_utils.py`

No sibling touches `semantics/utils.py`.

- [x] **Step 1: Write the failing tests** (append to the "Store checks" section)

```python
def test_extract_feature_cache_hands_the_extractor_rgb_arrays(tmp_path):
    """Frames reach forward() as decoded RGB ndarrays, not lazily-decoded PIL handles."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.full((4, 4, 3), (200, 10, 10), dtype=np.uint8)).save(images / "frame_000000.png")

    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    extractor.forward.return_value = [torch.zeros(3, 2, 2)]
    su.extract_feature_cache(extractor, images, tmp_path)

    [frame] = extractor.forward.call_args.args[0]
    assert isinstance(frame, np.ndarray)
    np.testing.assert_array_equal(frame[0, 0], [200, 10, 10])


def test_extract_feature_cache_reextracts_on_frame_count_mismatch(tmp_path):
    """A store whose n_frames disagrees with the directory is stale: re-extract."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / "frame_000000.png")
    stale = zarr.open(str(tmp_path / "fake.zarr"), mode="w")
    stale.attrs.update({"extractor": "fake", "n_frames": 7})

    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    extractor.forward.return_value = [torch.zeros(3, 2, 2)]
    su.extract_feature_cache(extractor, images, tmp_path)

    assert extractor.forward.called
    assert zarr.open(str(tmp_path / "fake.zarr"), mode="r").attrs["n_frames"] == 1
```

- [x] **Step 2: Verify**

Run the gate on `tests/semantics/test_semantics_utils.py`. Expected:
- `..._hands_the_extractor_rgb_arrays` FAILS (`isinstance(frame, np.ndarray)` is False: a PIL Image)
- `..._reextracts_on_frame_count_mismatch` PASSES today; it is the pinning test for the `open_valid`
  swap. Note that in the report.

- [x] **Step 3: Implement**

1. Delete `from PIL import Image` (line 24). No other use: `grep -n "Image\." collab_splats/semantics/utils.py` must print nothing afterwards.
2. Delete the `_UNREADABLE_STORE` comment and constant (lines 36-37).
3. Add `from collab_splats.utils.io import UNREADABLE_STORE, open_valid, read_image` to the first-party group.
4. Replace lines 256-264 (the "A cache is valid ..." block) with:

```python
    # A cache is valid when the extractor name and frame count both match
    if open_valid(zarr_path, {"extractor": extractor.name, "n_frames": N}) is not None:
        logger.info("Feature cache valid, skipping extraction: %s", zarr_path)
        return zarr_path
```

5. Line 269 becomes `first_frame = read_image(paths[0])`.
6. Line 287 becomes `frame = read_image(paths[i])`, and line 289's `extractor.forward([pil_img])` becomes
   `extractor.forward([frame])`.
7. Line 394's `except _UNREADABLE_STORE:` becomes `except UNREADABLE_STORE:`.

`test_extract_feature_cache_propagates_unexpected_errors` still holds: it patches `su.zarr.open`, which is
the same `zarr` module object `io.open_valid` calls.

- [x] **Step 4: Verify**

Run the gate on `tests/semantics tests/test_docstring_contract.py`. Both new tests pass, and there are no
new failures.

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/semantics/utils.py tests/semantics/test_semantics_utils.py && git commit --only collab_splats/semantics/utils.py tests/semantics/test_semantics_utils.py -m "refactor(semantics): feature cache via utils.io read_image and open_valid

Drops the module-local _UNREADABLE_STORE and the PIL decode; frames reach
forward() as RGB ndarrays (open_image accepts them).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`b3158af5`): RED `..._rgb_arrays` failed (PIL Image), `..._frame_count_mismatch`
passed (pinning test, as planned). Gate `tests/semantics tests/test_docstring_contract.py` 513 passed,
20 xfailed, 55 xpassed, 0 failed.

---

### Task 7: localization — `LZ4`, `_build_pairwise_refs`; eval_localization_parity mirror and JSON

**Files:**
- Modify: `collab_splats/localization/localizer.py` (lines 18, 310, 636 and uses, 745-751)
- Modify: `evals/scripts/eval_localization_parity.py` (107-114, 258)
- Test: `tests/localization/test_localizer.py`

No sibling touches either file.

- [x] **Step 1: Write the failing test** (append to `tests/localization/test_localizer.py`; `pytest`,
  `numpy` and `torch` are already imported there, so check and add any that are missing at the top)

```python
def test_build_pairwise_refs_rejects_0_255_images(monkeypatch):
    """The old <=1.5 guess silently rescaled; a [0, 255] tensor is now a named error."""
    from collab_splats.localization import localizer as loc

    monkeypatch.setattr(loc.BaseRetrievalExtractor, "get", lambda name: pytest.fail("retrieval reached"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        loc.CameraLocalizer._build_pairwise_refs(torch.full((1, 3, 4, 4), 200.0))
```

If `test_localizer.py` imports `localizer` at the top, use that name instead of the inline import. The
inline form is written here only because the file's import style was not re-read for this plan.

- [x] **Step 2: Verify it fails**

Run the gate on `tests/localization/test_localizer.py`.
Expected: FAIL. Today the 200.0 tensor passes the `<= 1.5` check as "already [0, 255]" and reaches
retrieval, so `pytest.fail("retrieval reached")` fires.

- [x] **Step 3: Implement localizer**

1. Delete `from zarr.codecs import BloscCodec` (line 18) and the two `lz4 = BloscCodec(cname="lz4")`
   lines (310, 636).
2. Replace `compressors=lz4` with `compressors=LZ4` in both functions, by hand-checked sed:
   `sed -i 's/compressors=lz4\b/compressors=LZ4/g' collab_splats/localization/localizer.py`, then
   `grep -n "lz4\b" collab_splats/localization/localizer.py` must print nothing.
3. Add `from collab_splats.utils.io import LZ4, to_uint8_hwc` to the first-party group.
4. Replace lines 745-751 with:

```python
        # ff.images is (N, 3, H, W) float in [0, 1] -> HWC uint8 RGB per frame
        imgs = ff_images.detach().cpu().numpy() if torch.is_tensor(ff_images) else np.asarray(ff_images)
        ref_images = list(to_uint8_hwc(imgs, channels_first=True))
```

This deletes the stale "VGGT stores [0, 255]" comment. Every creator stores [0, 1]
(`loger.py:390` asserts it).

- [x] **Step 4: Implement eval_localization_parity**

1. Replace the body of `_model_res_images` (107-114) with:

```python
def _model_res_images(ff_images) -> list[np.ndarray]:
    """
    ff.images (N, 3, H, W) in [0, 1] -> list of HWC uint8 RGB, as _build_pairwise_refs builds them.
    """
    imgs = ff_images.detach().cpu().numpy() if torch.is_tensor(ff_images) else np.asarray(ff_images)
    return list(to_uint8_hwc(imgs, channels_first=True))
```

2. Line 258 `out.write_text(json.dumps(report, indent=2, default=float))` becomes
   `write_json(out, report)`.
3. Add `from collab_splats.utils.io import to_uint8_hwc, write_json`. Then check the file still needs
   `json`: `grep -n "json\." evals/scripts/eval_localization_parity.py`. If that prints only the removed
   line, delete `import json`.
4. Smoke: `$PY -c "import sys; sys.path.insert(0, 'evals/scripts'); import eval_localization_parity"` exits 0.

- [x] **Step 5: Verify**

Run the gate on `tests/localization tests/evals`. The new test passes, and there are no new failures
(the 17 `vismatch` failures are control).

- [x] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/localization/localizer.py evals/scripts/eval_localization_parity.py tests/localization/test_localizer.py && git commit --only collab_splats/localization/localizer.py evals/scripts/eval_localization_parity.py tests/localization/test_localizer.py -m "refactor(localization): utils.io LZ4 and to_uint8_hwc for the pairwise refs

A [0, 255] images tensor now raises instead of being guessed at.
eval_localization_parity mirrors it and writes its report with write_json
(nan -> null; was default=float with bare NaN).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`9c284ea1`): both tensor branches lift with `.detach().float().cpu().numpy()`
(not `.detach().cpu().numpy()`), so an in-memory VGGT-X bf16 tensor no longer raises TypeError; pinned by
an extra `test_build_pairwise_refs_accepts_bf16_images` (RED: `Got unsupported ScalarType BFloat16`). The
new test imports `BaseRetrievalExtractor`/`CameraLocalizer` from the file's top-level imports. RED 2
failed; gate `tests/localization tests/evals` 232 passed, 0 failed (the 17 `vismatch` failures no longer
occur: vismatch is installed). `import json` deleted from eval_localization_parity.

---

### Task 8: dashboard pipeline — mesh RGB through `to_uint8_hwc`

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py:410-417`, `tests/dashboard/test_pipeline.py`

Sibling check: pc-lane-b changes one line at `pipeline.py:491`, well clear of 410-417. Rerun the
sibling-context command anyway, because pc-lane-b is live.

**Output change:** mesh vertex colors move from truncation to rounding (+0 or +1 per channel).

- [x] **Step 1: Write the failing test**

Read how `test_pipeline.py` drives the mesh branch (`grep -n "fuse_tsdf" tests/dashboard/test_pipeline.py`)
and copy that test's setup. With a `result.images` of constant `0.6 / 255` float32, assert the `rgbs`
argument captured by the patched `fuse_tsdf` is all `1` (truncation gives `0`). Name it
`test_mesh_rgb_rounds_not_truncates`.

If no existing test patches `fuse_tsdf`, extract lines 407-417 into a module-level
`_mesh_rgbs(images) -> np.ndarray`. Unit-test that directly:
`_mesh_rgbs(torch.full((1, 3, 2, 2), 0.6 / 255)).max() == 1`, and give it a house docstring. Keep the
tensor→cpu float32 step inside it.

- [x] **Step 2: Verify it fails**

Run the gate on `tests/dashboard/test_pipeline.py`. Expected: the new test fails with 0 != 1.

- [x] **Step 3: Implement**

Replace lines 413-416 (the rgb_scale guess comment, the `rgb_scale` line and the `rgbs = ...astype` line) with:

```python
            # images is [0, 1] on every backend; round to uint8 HWC once for the fusion
            rgbs = to_uint8_hwc(images, channels_first=True)
```

Keep lines 408-412, the "Land it on the CPU as float32" tensor step. Add
`from collab_splats.utils.io import to_uint8_hwc`.

- [x] **Step 4: Verify**

Run the gate on `tests/dashboard`. There should be no new failures.

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/dashboard/pipeline.py tests/dashboard/test_pipeline.py && git commit --only collab_splats/dashboard/pipeline.py tests/dashboard/test_pipeline.py -m "fix(dashboard): mesh RGB rounds via utils.io.to_uint8_hwc

Was a per-array [0,1]/[0,255] guess plus truncation; mesh colors shift
by at most +1 per channel.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`54336566`): `fuse_tsdf` is patched by
`test_run_pipeline_orders_steps_and_pushes`, so the new test copies that setup (no `_mesh_rgbs` extraction).
RED 1 failed (all 0); gate `tests/dashboard` 228 passed, 0 failed.

---

### Task 9: carry-overs 1 and 3 — depth_align K, VGGT-X portrait crop box

**Files:**
- Modify: `collab_splats/pointcloud/depth_align.py:265-269, 301-303`
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py:73-80`
- Test: `tests/pointcloud/test_depth_align.py`, `tests/pointcloud/test_feedforward_intrinsics.py`

Sibling check (done while planning; redo it): no depth_align hunk past line 223, and no vggtx hunk in
48-115, on pointcloud-release, pc-lane-a, pc-lane-b or feat/sfm-backends.

- [x] **Step 1: Write the tests**

(a) Append a pinning test to `tests/pointcloud/test_depth_align.py`, with
`from collab_splats.geometry.transforms import rescale_intrinsics` at the top. It passes before and
after, because this half is a refactor:

```python
def test_result_from_reconstruction_k_is_rescale_intrinsics_of_its_box():
    """
    Stored K equals rescale_intrinsics(original K, original_coords): one convention, not two.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    out, _ = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    fx, fy, cx, cy = K_PARAMS
    K_orig = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    expected = rescale_intrinsics(K_orig, out.original_coords[0], (DEPTH_H, DEPTH_W), to_original=False)
    assert out.intrinsics.dtype == np.float32
    np.testing.assert_allclose(out.intrinsics, np.broadcast_to(expected, out.intrinsics.shape), rtol=1e-6)
```

(b) Append a failing test to `tests/pointcloud/test_feedforward_intrinsics.py`, adding
`from collab_splats.geometry.transforms import rescale_intrinsics` and
`from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords` at the top:

```python
def test_vggtx_portrait_box_round_trips_the_model_k():
    """
    Upstream crop mode: resize to (518, round(h*518/w/14)*14), then center-crop 518 rows.

    - the y scale is new_h / orig_h (924/1920), not 518/orig_w
    - the wrong scale left fy 0.34% off and cy 3.25 px off for 1080x1920
    """
    box = _compute_vggtx_crop_coords([(1080, 1920)], target_size=518)[0]

    # Model K from the true upstream transform
    new_h = round(1920 * 518 / 1080 / 14) * 14
    start_y = (new_h - 518) // 2
    sx, sy = 518 / 1080, new_h / 1920
    K = np.array([[1000.0, 0.0, 540.0], [0.0, 1000.0, 960.0], [0.0, 0.0, 1.0]])
    K_model = np.array([[1000.0 * sx, 0.0, 540.0 * sx], [0.0, 1000.0 * sy, 960.0 * sy - start_y], [0.0, 0.0, 1.0]])

    back = rescale_intrinsics(K_model, box, (518, 518), to_original=True)
    np.testing.assert_allclose(back, K, atol=1e-2)
```

- [x] **Step 2: Verify**

Run the gate on `tests/pointcloud/test_depth_align.py tests/pointcloud/test_feedforward_intrinsics.py`.
Expected:
- (a) passes
- (b) fails with `fy` about 996.6 against 1000 and `cy` about 963.25 against 960

- [x] **Step 3: Implement vggtx**

In `_compute_vggtx_crop_coords`, replace the `if new_h > target_size:` branch body (lines 76-79) with:

```python
        if new_h > target_size:
            # Height crop applied; the resized height is new_h, so y scales by new_h / orig_h
            sy = new_h / orig_h
            start_y_resized = (new_h - target_size) // 2
            tl_y = start_y_resized / sy
            cr_y = (start_y_resized + target_size) / sy
```

In the docstring, change the second paragraph's "resizes width→target_size then center-crops height" to
"resizes to (target_size, new_h) with new_h rounded to a multiple of 14, then center-crops height". Leave
the rest of the docstring alone, because pc-lane-b reformats docstrings nearby.

- [ ] **Step 4: Implement depth_align** (DEFERRED, sibling hunk: see as-built note and the deferred table)

1. Move the `original_coords = ...` block (the comment and line 303) to just after the `cam_dims` check
   (after line 264).
2. Replace lines 265-269 with:

```python
    # Depth-grid scales for the sparse-point pixel indices below
    sx, sy = w / orig_w, h / orig_h

    # K from the original camera through the full-frame box: the rescale_intrinsics convention
    intrinsics = np.stack([reconstruction.cameras[im.camera_id].calibration_matrix() for im in images_sorted])
    intrinsics = rescale_intrinsics(intrinsics, original_coords, (h, w), to_original=False).astype(np.float32)
```

3. Add `from collab_splats.geometry.transforms import rescale_intrinsics` to the first-party imports.

- [x] **Step 5: Verify**

Run the gate on `tests/pointcloud tests/geometry tests/mesh tests/test_docstring_contract.py`. Both new
tests pass; `test_vggtx_crop_coords_portrait/landscape/cr_x` still pass; there are no new failures.

- [x] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/pointcloud/depth_align.py collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_depth_align.py tests/pointcloud/test_feedforward_intrinsics.py && git commit --only collab_splats/pointcloud/depth_align.py collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_depth_align.py tests/pointcloud/test_feedforward_intrinsics.py -m "fix(pointcloud): VGGT-X portrait crop box y scale; depth_align K via rescale_intrinsics

Portrait box used 518/orig_w for y; upstream crop mode resizes height to
round(h*518/w/14)*14, so y scales by new_h/orig_h. 1080x1920: tl_y
423.25 -> 421.82, cr_y 1503.25 -> 1498.18; the model K now round-trips.
Existing VGGT-X portrait zarrs keep the old box until re-run.

depth_align: K through rescale_intrinsics on its own full-frame box
(<= 1 ULP float32 vs the hand-scaled rows).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`b3496862`): vggtx half only; **carry-over 1 (depth_align K) deferred**.
- sibling re-check was stale: `clean/pointcloud-release` rewrites `depth_align.py:16` (adds its own
  `geometry.transforms` import + `full_frame_coords`) and `:303`; the `rescale_intrinsics` import can't
  land outside that hunk, so Step 4 moved to the deferred table. Step 1(a) pin test landed anyway (passes on
  the hand-scaled K, guards the post-rebase swap).
- vggtx `75-79` clear (sibling hunks end at 66, resume at 88); docstring edit skipped (sibling rewrites 48-66).
- (b) became two parametrized tests: K round-trip over 1080x1920, 1000x1040, 1920x1080, 1000x1000, and a
  gray-ramp frame through the real `load_and_preprocess_images` (100x104, 192x108, 100x100).
- test imports are function-local, like the file's existing vggtx tests: the top import block is within 3
  lines of sibling hunks in both test files.
- plan's RED numbers were wrong in fy's sign: old box returns fy 1003.38 (not 996.6), cy 963.24.
- RED 3 failed / 33 passed (the three portrait cases); gate `tests/pointcloud` + docstring contract
  687 passed / 2 skipped / 20 xfailed / 55 xpassed, `tests/geometry tests/mesh` 398 passed, 0 failed.
- no local VGGT-X portrait zarr exists (6 zarrs scanned); remote processed scenes not checked.

---

### Task 10: evals JSON reports through `write_json`

**Files:**
- Modify (all under `evals/scripts/`): `analyze_splats.py:314`, `ba_start_at_gt.py:180`,
  `eval.py:452,637,706,901`, `eval_compare.py:207` and `format_markdown_rows` (~246-256),
  `eval_multiview_conf.py:154`, `eval_similarity_calibration.py:393-395`, `eval_sky_mask.py:242`,
  `eval_splats.py:308`, `eval_verification.py:232`
- Test: `tests/evals/test_eval_compare.py`

Sibling check, required for `eval.py`, `ba_start_at_gt.py` and `eval_verification.py` (all three are
rewritten by r4-*). At plan time:
- eval.py's r4 hunks are at base lines 383-417, 547-568, 629, 675, 768 and 819-821. The sites are at 452,
  637, 706 and 901. 637 is 8 lines from the 629 hunk: clear of ±3, but recheck.
- ba_start_at_gt's r4 hunks are 32-38, 47-52 and 131-170. Site 180 is 10 lines clear.

**Output change:** every NaN/Inf in these reports becomes `null`. Readers of these files:
- `eval.py:833`: `np.array(..., dtype=float32)` maps null to nan
- `eval_compare.collect_grid_metrics` → `format_markdown_rows`: fixed in this task
- `docs/source/tutorials/evals/ground_truth_evals.ipynb`: reads metrics.json into Python/pandas, where
  None displays as None/NaN; no crash path found. Recheck with
  `grep -n "rmse\]" docs/source/tutorials/evals/ground_truth_evals.ipynb`.

- [x] **Step 1: Write the failing test** (append to `tests/evals/test_eval_compare.py`)

```python
def test_format_markdown_rows_renders_a_null_metric_as_nan():
    """write_json turns a nan ATE into null; the table must print nan, not raise on None."""
    from eval_compare import format_markdown_rows

    rows = [{"_cell": "c", "ate": {"rmse": None}, "rpe": {}, "auc": {"auc_30": 0.5}}]

    table = format_markdown_rows(rows)

    assert "| c | nan |" in table
```

- [x] **Step 2: Verify it fails**

Run the gate on `tests/evals/test_eval_compare.py`.
Expected: FAIL with `TypeError: unsupported format string passed to NoneType.__format__`.

- [x] **Step 3: Guard the reader**

In `format_markdown_rows`, build each row's four values through a local helper placed above the function:

```python
def _metric(section: dict, key: str) -> float:
    """
    One metric from a report section; missing or null (a write_json nan) reads as nan.
    """
    value = section.get(key)
    return float("nan") if value is None else value
```

Then `ate = _metric(r.get("ate", {}), "rmse")`, and the same for `rpe_t`, `rpe_r` and `auc`.

- [x] **Step 4: Migrate the writers**

In each file, add `from collab_splats.utils.io import write_json` to the first-party imports. Files
that `sys.path.insert` before first-party imports keep the `# noqa: E402` style of their neighbors.
Then apply:

| Site | New line |
|---|---|
| `analyze_splats.py:314` | `write_json(out_path, analysis)` |
| `ba_start_at_gt.py:180` | `write_json(OUT / ("report_nofilter.json" if no_filter else "report.json"), report)` |
| `eval.py:452` | `write_json(output_dir / "metrics.json", metrics_json)` |
| `eval.py:637-643` | `write_json(args._result_file, {...})`, same dict literal |
| `eval.py:706` | `write_json(cfg.output_dir / "comparison.json", rows)` |
| `eval.py:901` | `write_json(args.output_ate, ate_by_condition)` |
| `eval_compare.py:207` | `write_json(metrics_path, payload)` |
| `eval_multiview_conf.py:154` | `write_json(args.out, {"zarr": str(args.zarr), "seq": str(args.seq), "scale": s, "rows": rows})` |
| `eval_similarity_calibration.py:393-394` | the `with open(...)` block becomes `write_json(out_path, {"mode": args.mode, "layer_index_override": args.layer_index, "results": results})` |
| `eval_sky_mask.py:242` | `write_json(args.results / "stats.json", {"off": off, "on": on})` |
| `eval_splats.py:308` | `write_json(summary_path, summary)` |
| `eval_verification.py:232-233` | r4's exact text: the comment `# nan-safe atomic write: a degenerate pair's nan error lands as null, not a bare NaN` above the existing `mkdir`, then `write_json(args.out, report)`. The import goes at r4's position, after `from collab_splats.preproc import frames as fr`. |

`write_json` needs the parent directory to exist. Every site above already has it: a preceding `mkdir`,
or the directory the script just wrote into. For `eval.py:637`, check that `args._result_file`'s parent
is the tmpdir the parent process created.

After each file, drop `import json` if `grep -n "json\." <file>` prints nothing. eval.py still has
`json.loads` (833) and `print(json.dumps)` (891), so it keeps the import.

- [x] **Step 5: Smoke-import every touched script**

```bash
cd /workspace/collab-splats/.worktrees/consistency && for s in analyze_splats ba_start_at_gt eval eval_compare eval_multiview_conf eval_similarity_calibration eval_sky_mask eval_splats eval_verification; do PYTHONPATH=.:evals:evals/scripts /opt/venv/reconstruction/bin/python -c "import $s" >/dev/null 2>&1 && echo "ok $s" || echo "FAIL $s"; done
```

Expected: `ok` for each. Scripts whose import fails on the control commit for env reasons
(`git stash` is banned, so check with `git show HEAD~0:<file>` in a scratch dir) are noted, not fixed.

- [x] **Step 6: Verify**

Run the gate on `tests/evals`. The new test passes, and there are no new failures.

- [x] **Step 7: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && F="evals/scripts/analyze_splats.py evals/scripts/ba_start_at_gt.py evals/scripts/eval.py evals/scripts/eval_compare.py evals/scripts/eval_multiview_conf.py evals/scripts/eval_similarity_calibration.py evals/scripts/eval_sky_mask.py evals/scripts/eval_splats.py evals/scripts/eval_verification.py tests/evals/test_eval_compare.py" && git add $F && git commit --only $F -m "refactor(evals): every eval JSON report through utils.io.write_json

nan/inf -> null, numpy -> python, atomic; closes the default=float sites
in eval_localization_parity (previous commit) and eval_verification (r4's
exact line). format_markdown_rows reads a null metric as nan.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`18846f7c`): RED 1 failed / 9 passed (`TypeError ... NoneType.__format__`), gate
`tests/evals tests/utils` 182 passed, 0 failed; all nine scripts import-smoke ok.
- **eval_verification skipped**: deleted by geometry-round3 `e0d48ab8` (with `geometry/verification.py` and
  its tests); accept the delete at rebase. The commit body's "already migrated" reason is superseded.
- bytes: indent 2, no trailing newline, no sort_keys, as before; only non-finite floats change (→ null).
  `eval.py` `_result_file` was compact `json.dumps` and is now indent 2 (whitespace only, a temp IPC file).
- the new `write_json` imports in `eval.py`, `ba_start_at_gt.py` and `eval_similarity_calibration.py` sit
  beside sibling import hunks (pc-lane-a/pointcloud-release, geometry-round3); the write sites are all
  clear of ±3. Rebase keeps both import lines.
- `_metric` got an `Args:`/`Returns:` docstring (CLAUDE.md shape); `import json` dropped from five scripts.

---

### Task 11: evals images and uint8 — `read_image`, `to_uint8_hwc`

**Files:**
- Modify: `evals/scripts/eval.py:52,349`, `eval_similarity_calibration.py:140,147`, `eval_sky_mask.py:165`,
  `eval_splats.py:64-73`, `analyze_splats.py:296-297`
- Test: `tests/evals/test_eval_splats.py`, `tests/evals/test_analyze_splats.py`

Sibling check: eval.py:349 sits 8 lines above pc-lane-a's `:357` hunk, and r4's `@@ -52,0 +54 @@` inserts
after `from PIL import Image`. Deleting line 52 therefore conflicts trivially at rebase: keep r4's added
line and drop PIL. If the ±3 rule flags 349 on the live pc-lane-a, defer it.

- [x] **Step 1: Write the failing test** (append to `tests/evals/test_eval_splats.py`)

```python
def test_model_res_images_rejects_a_0_255_zarr():
    """The <=1.0 guess read a [0, 255] zarr as already scaled; now it is a named error."""
    result = SimpleNamespace(images=torch.full((1, 3, 2, 2), 200.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        eval_splats._model_res_images(result)
```

Match the file's existing import names for `eval_splats`, `SimpleNamespace`, `torch` and `pytest`
(`grep -n "^import\|^from\|sys.path" tests/evals/test_eval_splats.py`).

- [x] **Step 2: Verify it fails**

Expected: FAIL, because no ValueError is raised: 200 > 1.0 skips the ×255 and the result is clipped to 200.

- [x] **Step 3: Implement**

| Site | Change |
|---|---|
| `eval.py:349` | `frames = np.stack([read_image(p) for p in image_paths])`; delete `from PIL import Image` (line 52) if `grep -n "Image\." evals/scripts/eval.py` is then empty |
| `eval_similarity_calibration.py:140,147` | delete the inline `from PIL import Image`, add it at the top; line 147 becomes `img = Image.fromarray(read_image(p))` |
| `eval_sky_mask.py:165` | `rgb = read_image(path)` (`cv2` stays for the sheet) |
| `eval_splats.py:64-73` | body becomes `return to_uint8_hwc(result.images.numpy(), channels_first=True)`; docstring bullet `- [0, 1] float CHW -> uint8 HWC; a [0, 255] zarr raises` |
| `analyze_splats.py:296-297` | `rgbs = to_uint8_hwc(np.asarray(ff.images), channels_first=True)`; keep the comment above, reworded to `# images is (N, 3, H, W) float in [0, 1]; fuse_tsdf takes (N, H, W, 3) uint8` |

Add the matching `from collab_splats.utils.io import ...` line to each file.

- [x] **Step 4: Verify**

Rerun the Task 10 Step 5 smoke loop, then the gate on `tests/evals`. The new test passes, and there are
no new failures (`test_analyze_splats.py` included).

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && F="evals/scripts/eval.py evals/scripts/eval_similarity_calibration.py evals/scripts/eval_sky_mask.py evals/scripts/eval_splats.py evals/scripts/analyze_splats.py tests/evals/test_eval_splats.py" && git add $F && git commit --only $F -m "refactor(evals): image decode and uint8 conversion through utils.io

read_image replaces PIL/cv2 decodes; to_uint8_hwc replaces three
[0,1]/[0,255] guesses. eval_similarity_calibration's inline PIL import
moves to the top.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`2dad18c6`): RED 1 failed / 6 passed (`DID NOT RAISE`), gate `tests/evals tests/utils`
183 passed, 0 failed; smoke loop all ok.
- **eval.py deferred**: `:349` is 3 lines above pc-lane-a/pointcloud-release's `:352-354` hunk (the plan
  only counted `:357`), and `:52` is adjacent to their `:51` delete and geometry-round3's insert. `PIL`
  stays in eval.py until `:349` moves. Added to the deferred table.
- **plan gap**: `test_inputs_uint8_hwc_for_both_image_scales` pinned [0, 255] acceptance, and the
  `_write_ff_zarr` fixture defaulted to `scale=255.0` (also breaking `test_inputs_native_from_images_dir`'s
  model-res call). Fixture default is now 1.0; the test became `test_inputs_uint8_hwc_from_a_0_1_zarr`
  (exact `rint` values plus a [0, 255] raise through `inputs_from_pointcloud_zarr`).

---

### Task 12: camera centers through `invert_poses`

**Files:**
- Modify: `collab_splats/dashboard/localize.py:39-43`, `evals/scripts/ba_start_at_gt.py:58-61`
- Test: `tests/dashboard/test_localize_page.py`

This is the idiom r4 chose for the LC wrapper: `invert_poses(P)[..., :3, 3]`. The names
`camera_centers` / `cam_centres` stay, because callers and the dashboard test import them. Only the
bodies change.

- [x] **Step 1: Write the test** (append next to `test_camera_centers_inverts_world_to_camera`)

The existing fixture uses the identity rotation, which cannot tell `-Rᵀt` from `-Rt`. This test can:

```python
def test_camera_centers_uses_the_rotation_transpose():
    """A non-symmetric rotation: -R^T t and -R t differ, only the first is the center."""
    c = np.array([1.0, 2.0, 3.0])
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
    ext = np.eye(4)[None].copy()
    ext[0, :3, :3] = R
    ext[0, :3, 3] = -R @ c

    np.testing.assert_allclose(camera_centers(ext)[0], c, atol=1e-12)
```

- [x] **Step 2: Run it**

Expected: PASS on the current einsum body. It is a pinning test for the swap. Also verify it would catch
a regression: temporarily change the einsum to `"nij,nj->ni"`, see it fail, then revert.

- [x] **Step 3: Implement**

- `dashboard/localize.py`: the body becomes `return invert_poses(extrinsics)[:, :3, 3]`. Add
  `from collab_splats.geometry.transforms import invert_poses`, and replace the one-line docstring with
  a house docstring: summary `World-space camera centers from (N, 4, 4) world-to-camera transforms.`,
  bullet `- the translation column of the inverse pose, C = -R^T t`, plus `Args:`/`Returns:`.
- `ba_start_at_gt.py:58-61`: the body becomes `return invert_poses(poses_w2c)[:, :3, 3]`, and the
  import on line 38 becomes `from collab_splats.geometry.transforms import invert_poses, umeyama_sim3  # noqa: E402`.
  - sibling check: r4's hunks are at 32-38 (a different import) and 47-52. Line 38 is itself the
    transforms import, which r4 leaves unchanged, but it sits within ±3 of r4's 32-38 hunk.
  - if the rule flags it, skip ba_start_at_gt entirely and defer it.

- [x] **Step 4: Verify**

Run the gate on `tests/dashboard/test_localize_page.py tests/evals`, then import-smoke
`ba_start_at_gt`. There should be no new failures.

- [x] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && F="collab_splats/dashboard/localize.py evals/scripts/ba_start_at_gt.py tests/dashboard/test_localize_page.py" && git add $F && git commit --only $F -m "refactor(dashboard): camera centers via invert_poses, the idiom r4 chose

Same math (C = -R^T t); adds a non-identity-rotation test the old
identity fixture could not fail.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`29774d62`): pinning test uses two cameras with dense, non-symmetric rotations
(Rz·Rx·Ry, Ry·Rx) and a fixture guard that `-R t` and `t` miss the centers. It passes on the einsum body
and after the swap; scratch-copy mutants `nij` (no transpose) and `-t` fail it while the old identity
test passes both. Gate `tests/dashboard` 229 passed, docstring contract 314 passed / 0 failed.
- **ba_start_at_gt deferred**: its transforms import (`:37`) is 1 line from geometry-round3's
  bundle_adjustment import hunk (`:34-36`). Added to the deferred table.
- **deviation**: the `invert_poses` import is lazy inside `camera_centers`, not at module top.
  `collab_splats.geometry` pulls BA + warp, measured ~4.3 s on top of the page import, and the page's
  documented rule is heavy imports lazy so it renders at launch.

---

### Task 13: adopt `get_device` / `pytorch_gc`

**Files:**
- Modify: `collab_splats/localization/extractors.py:122`, `collab_splats/localization/retrieval.py:49,125`,
  `collab_splats/dashboard/pipeline.py:205`, `collab_splats/mesh/features.py:54`,
  `evals/scripts/eval_similarity_calibration.py:171,323`, `evals/scripts/refit_at_fixed_poses.py:44`

Pure refactor with identical strings. The only behavior change is at `eval_similarity_calibration:171`,
where `pytorch_gc` adds `synchronize()` + `gc.collect()` after `del salad`. There is no RED step.

- [x] **Step 1: No new test**

Every site builds a model (weights download) before its device is observable, so a unit test would
need to mock the whole constructor to see a string that `get_device()` already returns. The gate is:
- the no-new-failures run in Step 3
- the grep in Step 3, which must list only the deferred and left-as-is sites

- [x] **Step 2: Implement**

| Site | New code |
|---|---|
| `extractors.py:122` | `self._device = device or get_device()` |
| `retrieval.py:49`, `:125` | same |
| `pipeline.py:205` | `device = get_device()` |
| `mesh/features.py:54` | `device = get_device()` |
| `eval_similarity_calibration.py:323` | `device = get_device()` |
| `eval_similarity_calibration.py:171` | `pytorch_gc()` |
| `refit_at_fixed_poses.py:44` | `dev = torch.device(get_device())` |

Each file gets `from collab_splats.utils.torch_utils import get_device` (and `pytorch_gc` where used).
`torch` stays imported where other code uses it.

- [x] **Step 3: Verify**

```bash
cd /workspace/collab-splats/.worktrees/consistency && rtk proxy git grep -n '"cuda" if torch.cuda.is_available() else "cpu"' -- collab_splats evals
```

Expected (re-run 2026-09-27): `bundle_adjustment.py:330`, `:560`, `:629`, `feedforward/base.py:1162`,
`pointcloud/utils.py:277`, `eval_similarity_calibration.py:322` (deferred / left-as-is) and the
definition itself, `utils/torch_utils.py:17`. Then run the gate on `tests/localization tests/dashboard tests/mesh tests/evals`; there
should be no new failures.

- [x] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && F="collab_splats/localization/extractors.py collab_splats/localization/retrieval.py collab_splats/dashboard/pipeline.py collab_splats/mesh/features.py evals/scripts/eval_similarity_calibration.py evals/scripts/refit_at_fixed_poses.py" && git add $F && git commit --only $F -m "refactor: adopt torch_utils.get_device / pytorch_gc at eight inline sites

Sibling-owned sites (BA, feedforward base, LC wrapper, reconstructor,
vda, pointcloud/utils, eval.py) deferred to post-rebase.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

2026-09-27 as built (`05156acc`): six sites, not eight. `get_device()` is exactly
`"cuda" if torch.cuda.is_available() else "cpu"` (no mps), and `device or get_device()` keeps a
caller-passed device winning, so every string is identical. Gates: `tests/localization` 91,
`tests/mesh` 72, `tests/evals` 143, `tests/dashboard` 229 passed (+2 FAILED = Task 14's uncommitted RED
tests, shape `(20, 40, 3)`), docstring contract 0 failed. The Step 3 grep lists only deferred sites plus
`torch_utils.py:17` itself.
- **eval_similarity_calibration deferred** (`:170` `pytorch_gc`, `:322` `get_device`): its new
  `torch_utils` import would sit within 3 lines of pointcloud-release/pc-lane-a's inserted
  `pointcloud.utils` import above `utils.image`. Added to the deferred table.
- `refit_at_fixed_poses.py` has no tests and runs at import (loads an npz): its import statements were
  exec'd alone and `py_compile`d.

---

### Task 14: sweep for leftovers

**Files:** only what the sweep finds

- [x] **Step 1: Search**

```bash
cd /workspace/collab-splats/.worktrees/consistency && rtk proxy git grep -nE 'BloscCodec\(cname="lz4"\)|_UNREADABLE_STORE|cv2\.imread|\.convert\("RGB"\)|default=float|rgb_scale|<= 1\.5|json\.dumps\([^)]*indent' -- collab_splats evals
```

Every remaining hit must be one of:
- a deferred-table row below
- a site listed as "left as is"
- a non-report `json.dumps`: a print or a cache hash

Known census misses, in scope for this sweep (found in the T5-T8 review, 2026-09-27):
- `collab_splats/dashboard/pipeline.py:~572`: localization ref images via
  `np.asarray(open_image(p).convert("RGB"))`
- `collab_splats/dashboard/localize.py:~693`: correspondence ref image, same PIL idiom (inline
  `open_image` import at `:679`)
- why (as planned): PIL ignores EXIF orientation, cv2 applies it; `wrapper/reconstructor.py:769` already decodes the
  same frames with cv2, so the dashboard and the reconstructor can disagree on a rotated JPEG
- superseded in the final review: `read_image` now ignores EXIF (`IMREAD_IGNORE_ORIENTATION`), matching
  PIL and the feedforward loaders; the migration stays for one decode path, not for orientation
- migrate both to `read_image`, each with a test, as in Task 6

- [x] **Step 2: Fix or record**

A hit that is none of those is a census miss. If it is in a non-sibling file, migrate it with a test,
the same way as its group's task. If it is sibling-owned, add it to the deferred table in this plan.

- [x] **Step 3: Commit** (only if Step 2 changed anything; message
  `refactor: utils.io sweep — <what>`, trailer as above)

2026-09-27 as built (`360ace9a`): both dashboard PIL decodes moved to `read_image`. Tests write an
EXIF-orientation-6 20x40 JPEG and require the 40x20 `read_image` array. RED 2 failed (`(20, 40, 3) ==
(40, 20, 3)`), GREEN `tests/dashboard` 231 passed, docstring contract 0 failed. The `localize.py` import
stays lazy at the old `open_image` site: `utils.io` (zarr + cv2) measured ~0.8-1.4 s at module top.
Final-review correction: EXIF-following was the wrong target (semantics would rotate against the
feedforward depth grid). `read_image` now ignores orientation; both tests were renamed
`..._ignoring_exif_orientation` and require the stored-pixel 20x40 array. The "back to PIL" mutant
now survives by design; dropping `IMREAD_IGNORE_ORIENTATION` kills both, plus the new
`test_read_image_ignores_exif_orientation_like_pil` in `tests/utils/test_io.py`.
Every other sweep hit is accounted for:
- deferred rows: `bundle_adjustment.py:154`, `metrics.py:307-308`, `feedforward/base.py:163`,
  `reconstructor.py:184,769`, `eval.py:350`, `eval_verification.py:232`
- r4-report library JSON (Dropped table): `verification.py:476`, `frames.py:149`, `qa.py:552`,
  `splats/rendering.py:398`
- non-report: `eval_verification.py:234` print; `feedforward/base.py:975` is docstring text
- not an RGB decode: `sky.py:171` and `eval_verification.py:102` read grayscale / 16-bit depth
  (`read_image` is color-only)
- **new deferred rows**: `feedforward/base.py:1034` `_decode_dir_to_frames` PIL decode (sibling hunks in
  the same function); `open_image(...).convert("RGB")` at `semantics/features/base.py:143` and
  `semantics/segmentation/sky.py:81` (polymorphic path/ndarray/PIL inputs, so the fix is `open_image`'s
  path branch, not the call site)

---

### Task 15: final gate, graph, bookkeeping

- [x] **Step 1: Full package gate against control**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && for p in tests/utils tests/preproc tests/semantics tests/localization tests/dashboard tests/pointcloud tests/geometry tests/mesh tests/evals tests/test_docstring_contract.py; do PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest -p no:cacheprovider -q -rfE $p > SCRATCH/p2-final-$(basename $p .py).log 2>&1; echo "$p exit=$?"; done; grep -hE "^(FAILED|ERROR)" SCRATCH/p2-final-*.log | sort > SCRATCH/consistency-p2-final-failures.txt; comm -13 SCRATCH/consistency-p2-control-failures.txt SCRATCH/consistency-p2-final-failures.txt
```

Expected: the `comm` output (new failures) is empty. Also check the SKIP counts against the control
logs. A rise means a guarded dependency went missing, which makes the gate vacuous.

- [x] **Step 2: Graph**

`cd /workspace/collab-splats/.worktrees/consistency && graphify update .` (AST only).

- [x] **Step 3: Tick this plan, fill the fork-point line, commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add -f docs/superpowers/plans/2026-09-26-consistency-phase2.md && git commit --only docs/superpowers/plans/2026-09-26-consistency-phase2.md -m "docs(plans): consistency phase 2 complete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

CHANGELOG and CLAUDE.md in-flight edits belong to whoever closes the whole consistency effort, not this
phase.

**2026-09-27 as built** (tip after the G5 nit commit `1e9f5245`; proof line printed
`.worktrees/consistency/collab_splats/__init__.py`; logs `SCRATCH/consistency-p2-final-<pkg>.log`):

| Package | Control | Final | Exit | Drift explained by |
|---|---|---|---|---|
| utils | 20 | 40 | 0 | +20: new `test_io.py` (18 tests + 1 parametrized x2) and `test_utils_import_light.py` (1) |
| preproc | 185 | 186 | 0 | +1: `test_read_frames_names_an_undecodable_frame` |
| semantics | 197 | 199 | 0 | +2: `extract_feature_cache` RGB-array + frame-count-mismatch pins |
| localization | 89 | 91 | 0 | +2: `build_pairwise_refs` rejects 0-255 / accepts bf16 |
| dashboard | 227 | 231 | 0 | +4: camera_centers, 2 EXIF `read_image` pins, `test_mesh_rgb_rounds_not_truncates` |
| pointcloud | 365 + 2 skip | 375 + 2 skip | 0 | +10: depth_align K test (1) + 2 crop-box tests parametrized x9 total |
| geometry | 326 | 326 | 0 | none |
| mesh | 72 | 72 | 0 | none |
| evals | 141 | 143 | 0 | +3 new, 1 replaced (`inputs_uint8_hwc_for_both_image_scales` split) |
| contract | 306 / 20xf / 55xp | 314 / 20xf / 55xp | 0 | +8: `MODULES = ("utils/io.py",)` adds 8 file-parametrized ids |

- FAILED/ERROR: none; `consistency-p2-final-failures.txt` and the control list both empty, `comm` empty
- SKIP: 2 in pointcloud, same as control; no other package skips
- drift verified by `+def test_` counts in `git diff ca526570 HEAD -- tests/<pkg>` and `--collect-only`
- graph: `graphify update .` run (graphify-out is gitignored)

---

## Deferred until after the rebase onto `clean/final`

Each row is sibling-owned today. Migrate it once that sibling has landed, in the phase-3 dedup pass.

| Site | Helper | Owner |
|---|---|---|
| `wrapper/reconstructor.py:184`, `:769` image reads | `read_image` | r4-report |
| `wrapper/reconstructor.py:499-503` (inline torch import), `:581`, `:1111` | `pytorch_gc` | r4-report |
| `geometry/metrics.py:304-308` rgb_scale guess | `to_uint8_hwc` | r4-report (still guesses at `metrics.py:246` there) |
| `geometry/bundle_adjustment.py` BloscCodec (~154), tracks-cache check (125-160), device (330/560/629) | `LZ4`, `open_valid`, `get_device` | r4-ba/r4-report (r4-report still catches `(KeyError, ValueError, OSError)` without TypeError) |
| `pointcloud/feedforward/base.py:163` BloscCodec; `:592`, `:1162`, `:1221` device/gc | `LZ4`, `get_device`, `pytorch_gc` | pointcloud-release / pc-lane-b |
| `geometry/loop_closure/wrapper.py:292`, `:566` | `pytorch_gc` | r4-lc |
| `pointcloud/vda.py:162` | `pytorch_gc` | pc-lane-a |
| `pointcloud/utils.py:277` | `get_device` | pc-lane-a (already adopted there) |
| `evals/scripts/eval.py:352-354` | `pytorch_gc` | pc-lane-a (`:357` hunk) |
| `evals/scripts/eval.py:350` VDA frames decode, `:52` `from PIL import Image` (Task 11) | `read_image` | pc-lane-a / pointcloud-release (`:352-354` hunk, `:51` delete) |
| `evals/scripts/eval_verification.py:232` (Task 10) | none | deleted by geometry-round3 `e0d48ab8`; accept the delete at rebase |
| `mesh/io.py:145`, `localization/extractors.py:99` | `to_numpy` | after pointcloud-release lands it; extractors keeps its float32 cast |
| COLMAP IO: `colmap/sparse/0` literal ×7; weak `cameras.bin`-only done check at `wrapper/reconstructor.py:1618` | new `utils/colmap.py` (may import pycolmap; io.py stays light): `SPARSE_SUBDIR`/`sparse_dir(root)`, `is_complete` (all three `.bin`), atomic `write_model` (temp dir + rename), `read_model`; pose/K one-liners stay inline | pointcloud-release (5 of 7 sites in sfm/ff files it is reorganizing) |
| `pointcloud/depth_align.py:265-269` hand-scaled K (Task 9 carry-over 1) | `rescale_intrinsics(K, full_frame_coords(orig_w, orig_h, 1)[0], (h, w), to_original=False)`; pin test `test_result_from_reconstruction_k_is_rescale_intrinsics_of_its_box` already landed | pointcloud-release (`:16` import, `:303`) |
| `feedforward/vggtx.py:53-54` `_compute_vggtx_crop_coords` docstring | none: state `new_h = round(h*518/w/14)*14` and the crop-row scale `sy = new_h/orig_h` | after pointcloud-release lands (it rewrites 48-66) |
| pre-existing: mixed-orientation VGGT-X batches are white-padded to the largest shape upstream (`load_fn.py:271-292`), so per-row boxes no longer match the model grid | gap: upstream offsets each smaller image by `pad_top`/`pad_left`, and `_compute_vggtx_crop_coords` ignores both; the box must add them | pointcloud-release rewrite of `feedforward/vggtx.py` (~48-66) |
| `transforms.intrinsics_to_original` (r4-report) vs `rescale_intrinsics` (phase 1) | one of them | reconcile in phase 3 |
| `evals/scripts/ba_start_at_gt.py:58-61` `cam_centres` (Task 12) | `invert_poses(P)[:, :3, 3]` | geometry-round3 (`:34-36` import hunk beside the `:37` transforms import) |
| `evals/scripts/eval_similarity_calibration.py:170`, `:322` (Task 13) | `pytorch_gc`, `get_device` | pointcloud-release / pc-lane-a (inserted `pointcloud.utils` import beside the import block) |
| `pointcloud/feedforward/base.py:1034` `_decode_dir_to_frames` PIL decode (Task 14) | `read_image` | pointcloud-release / pc-lane-b (hunks in the same function, PIL import rewritten) |
| `open_image(...).convert("RGB")` at `semantics/features/base.py:143`, `semantics/segmentation/sky.py:81` (Task 14) | `open_image`'s str/Path branch decodes via `read_image` (EXIF ignored, same as PIL) | phase 3; check other `open_image` callers relying on non-RGB modes first |
| `utils/io.py` `write_json` (byte-identical to r4 `4e4a8ce6`, so not changed here) | none: no atomicity test (a direct `write_text` mutant passes), the fixed `.json.tmp` name races concurrent writers, no fsync; a crash between write and `os.replace` leaves `metrics.json.tmp` (or `<name>_alignment.json.tmp`), which `eval_compare.scan_results_dir` (~113-119) rejects as "unexpected file" on the next run — fix upstream with a unique tmp name or skip `.tmp` in the scan | raise with r4-report |

## Risks

- **On-disk artifacts change** in four places:
  - Eval JSON NaN→null (Task 7, Task 10). In-tree readers are handled. An out-of-tree notebook doing
    `f"{x:.3f}"` on a null raises.
  - VGGT-X portrait `original_coords` (Task 9). New runs write the corrected box. Existing portrait
    zarrs keep the stale one, 2-5 px off in y. Mesh RGB crop and metrics read the box, so a re-run is
    the fix, and nothing detects the stale box.
  - Dashboard mesh colors (Task 8): up to +1 per channel.
  - A pre-`a157421` zarr with [0, 255] images now raises in localization, eval_splats, analyze_splats
    and the dashboard mesh, where before it was guessed at.
- **JPEG decode:** `read_image` (cv2 + `IMREAD_IGNORE_ORIENTATION`) ignores EXIF orientation, as PIL,
  the upstream VGGT-X/MapAnything loaders and `feedforward/base.py` `_decode_dir_to_frames` do; pixels
  may still differ by ±1 from PIL's libjpeg on JPEGs
  - no stage migrated here rotates; semantics features stay aligned with the feedforward depth grid
  - one EXIF-following decode remains: the directory ingest `wrapper/reconstructor.py:184` (plain
    `cv2.imread`, deferred to r4-report) bakes orientation into the PNG store; migrating it to
    `read_image` closes it
  - pipeline frames are PNG (`images/frame_NNNNNN.png`), which carry no EXIF tag
- **Sibling overlap:** checked with `git branch -a` and per-branch diffs while planning. The live
  branches are pc-lane-a (`e8520014`), pc-lane-b (`78e88271`, still moving) and r4-* (`clean/r4-report`
  and siblings), all of which can move again. The sibling-context rule in Conventions is the defense.
  Rerun it at every task, not once.
- **Docstring contract:** r4 adds two checks (nested defs, docstring on the quote line) that will run
  against `utils/io.py` after the rebase. The code in Tasks 3-4 has no nested defs and opens every
  docstring on its own line, but rerun the contract after the rebase.

## Rebase notes (for whoever rebases `clean/consistency` onto `clean/final`)

- `utils/io.py` and `tests/utils/test_io.py`: Task 2's add is byte-identical to r4-report `4e4a8ce6`,
  so it replays as no change. Tasks 3-4 then apply as plain appends plus the docstring-header swap.
- `eval_verification.py`: deleted by geometry-round3 `e0d48ab8` (with `geometry/verification.py` and
  `tests/geometry/test_verification.py`); accept the delete at rebase, including phase 1's `2664c4c0`
  `calibration_matrix` edit to it.
- `preproc/frames.py`: the import blocks conflict (r4 adds `write_json`, this branch adds `read_image`).
  Merge them to `from collab_splats.utils.io import read_image, write_json`.
- `eval.py:52`: Task 11 left `from PIL import Image` in place (deferred with `:350`), so r4's insert after
  it replays cleanly.
- Phase-1 `clean_for_json` tests: `tests/geometry/test_verification.py` is deleted by geometry-round3
  `e0d48ab8` with `clean_for_json`'s module; accept the delete. geometry-round3's
  `tests/geometry/test_metrics.py` no longer references `clean_for_json` (`:25, 407, 983-984`); take its
  side. NaN→null, numpy→python and tuple coverage lives in `tests/utils/test_io.py` (`to_json_safe`).
  That is where carry-over 4 finally closes.
- `tests/test_docstring_contract.py`: Task 4 edits lines 10, 22-23, 44-46 and 194, and r4 edits 363+.
  They should not conflict.

Fork point for phase 2: `ca526570` (phase 1/1b tip; fork of the whole branch `b9c5843a`).
