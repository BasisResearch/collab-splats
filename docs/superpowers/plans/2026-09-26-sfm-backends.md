# SfM backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `pointcloud: {method: sfm, backend: colmap | hloc}` runs end to end through `Reconstructor`, like `instantsfm`, with instantsfm output byte-identical.

**Architecture:** The system-colmap SIFT helpers move out of `sfm/instantsfm.py` into a shared `sfm/sift_db.py`, which gains pairing modes, a params sidecar, vocab-tree fetch and `largest_model`. `ColmapCreator` (colmap CLI SIFT + `pycolmap.incremental_mapping`) and `HlocCreator` (hloc @ c13273b) are rewritten to instantsfm's `reconstruct(data_dir, images_dir) -> pycolmap.Reconstruction` contract. `_run_sfm` dispatches through `SFM_CREATORS`, then subsets to the registered frames above a floor.

**Tech Stack:** pycolmap 4.0.4 (CPU wheel), system `colmap` 3.10-dev CLI (CUDA), hloc @ `c13273b` (editable uv path source), pytest with mocks.

**Spec:** `docs/superpowers/specs/2026-09-26-sfm-backends-design.md`

---

## Standing rules (every task, every subagent)

- Work only in `/workspace/collab-splats/.worktrees/sfm-backends` (branch `feat/sfm-backends`).
- Every Bash call must start with `cd /workspace/collab-splats/.worktrees/sfm-backends && ` and run python as `PYTHONUTF8=1 PYTHONPATH=/workspace/collab-splats/.worktrees/sfm-backends /opt/venv/reconstruction/bin/python`.
  - The venv's editable finder hardcodes the main tree, so without both parts pytest tests the WRONG tree and reports false green.
- The first pytest call of each task prints the proof line:
  - `python -c "import collab_splats; print(collab_splats.__file__)"`
  - It must show `.worktrees/sfm-backends/`.
- Pytest rules:
  - never pipe pytest through `tail`/`head`;
  - never `--tb=no`;
  - never the full suite;
  - always `-p no:cacheprovider`.
- **Never** run `pip install`, `uv sync`, `uv lock`, or `uv pip`. The venv is shared, and the user runs these.
- **Never** `git commit --amend`, `rebase`, `reset`, `stash`, `push`, or `merge`.
- Commit with `git add <paths>`, then `git commit --only <paths> -m ...` (`git add -f` for `docs/superpowers/**`). Use a conventional-commit subject, and end the body with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not touch notebooks, or `.worktrees/tutorial-rework`.
- Code style is CLAUDE.md's:
  - `"""` docstrings with the summary on the next line, `- ` bullets, and `Args:`/`Returns:`;
  - every param annotated;
  - comment runs of 3+ lines are a header line plus `- ` bullets;
  - US spelling;
  - `logger`, never `print`.
- `tests/test_docstring_contract.py` enforces these on `collab_splats/pointcloud`.
- Format touched files only: `black <files> && isort <files>`. Never run them repo-wide; the venv black is newer.

## Execution waves (subagent-driven, parallel where file sets are disjoint)

Everything happens in the worktree. That includes this plan, which is committed on `feat/sfm-backends`, not on clean/final.

Parallel agents share one git index, so every commit is `git commit --only <own paths>`. Never run `git add -A` or `git add .`.

| Wave | Tasks | Why together |
|---|---|---|
| W0 | 0 | worktree, baseline gate, instantsfm reference runs A/B (tmux, heavy — nothing else runs) |
| W1 | 1 | creates `sift_db.py`; everything after imports it |
| W2 | 2 ∥ 5 ∥ 8 ∥ 9 | disjoint files: `sift_db` pairing / `hloc.py`+`setup/hloc.sh` / `depth_align`+`sources` / pyproject+setup.sh+Dockerfile |
| W3 | 3 ∥ 4 | 3 needs `sift_db.PAIRINGS` (Task 2); 4 needs Task 2's helpers; files disjoint (reconstructor+base.yaml vs colmap.py) |
| W4 | 6 | needs 4 + 5 (`SFM_CREATORS` imports both) |
| W5 | 7 ∥ 10 | 7 needs 3 + 6; docs are independent of code edits |
| W6 | 11 → 12 → 13 | gate, then heavy runs serially |

- Each wave ends with a spec-compliance review, then a code-quality review, before the next wave starts.
- `test_registry.py::test_all_creators_are_instantiable` is red between W3 and W4 by construction (Task 4 note).

## File map

| File | Change |
|---|---|
| `collab_splats/pointcloud/sfm/sift_db.py` | NEW: `sfm_image_dir`, `sift_database_valid`, `build_sift_database`, `write_database_params`, `rename_images_to_stems`, `largest_model`, `fetch_vocab_tree`, `colmap_cli_version` |
| `collab_splats/pointcloud/sfm/instantsfm.py` | helpers removed, imports from `sift_db` (no behavior change) |
| `collab_splats/pointcloud/sfm/colmap.py` | rewrite: `ColmapCreator` |
| `collab_splats/pointcloud/sfm/hloc.py` | rewrite: `HlocCreator`, `HLOC_PIN`, `sequential_pairs`, `union_pairs` |
| `collab_splats/pointcloud/sfm/__init__.py` | `SFM_CREATORS`, docstring |
| `collab_splats/pointcloud/__init__.py` | `_REGISTRY` feedforward only, docstrings |
| `collab_splats/pointcloud/base.py` | docstrings only |
| `collab_splats/pointcloud/depth_align.py` | backend-neutral messages/docstrings |
| `collab_splats/wrapper/reconstructor.py` | `_SFM_BACKENDS`, validation, `_run_sfm` dispatch/subset/attrs, `_registered_rows` |
| `collab_splats/remote/sources.py` | `PUSH_EXCLUDES` += `colmap.db`, `colmap/hloc/` |
| `configs/base.yaml` | `colmap:` + `hloc:` blocks, backend comment |
| `setup/hloc.sh`, `setup.sh`, `Dockerfile`, `pyproject.toml` | hloc dependency wiring |
| `tests/pointcloud/sfm/test_sift_db.py` | NEW |
| `tests/pointcloud/sfm/test_instantsfm.py` | SIFT/rename/image-dir tests move to `test_sift_db.py` |
| `tests/pointcloud/sfm/test_colmap.py`, `test_hloc.py` | rewrite |
| `tests/pointcloud/test_registry.py` | colmap/hloc -> `KeyError` |
| `tests/wrapper/test_sfm_config.py`, `tests/wrapper/test_sfm_stage.py` | flips + new wiring tests |
| `tests/remote/test_sources.py` | excludes |
| docs | decision 018, `third_party/README.md`, `configs/README.md`, `docs/source/api/pointcloud.rst`, `CLAUDE.md` |

---

### Task 0: Worktree, baseline gate, instantsfm reference runs

**Files:** none (setup only)

- [ ] **Step 1: Create the worktree off the current clean/final tip**

```bash
cd /workspace/collab-splats && git worktree add -b feat/sfm-backends .worktrees/sfm-backends clean/final
cd /workspace/collab-splats/.worktrees/sfm-backends && for d in LoGeR Video-Depth-Anything hloc; do ln -s /workspace/collab-splats/third_party/$d third_party/$d; done && ls -la third_party
```

Expected: three symlinks plus `README.md`.

- [ ] **Step 2: Proof line**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
```

Expected: `/workspace/collab-splats/.worktrees/sfm-backends/collab_splats/__init__.py`

- [ ] **Step 3: Baseline gate (before any edit), saved to the scratchpad**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD timeout 3000 /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper tests/geometry tests/remote tests/test_docstring_contract.py -q -p no:cacheprovider --continue-on-collection-errors -rfEs > $SCRATCH/baseline_gate.txt 2>&1; echo EXIT $?
```

`$SCRATCH` = `/tmp/claude-0/-workspace-collab-splats/9df0e669-b4c9-48e2-8fb2-c13034260060/scratchpad`.
- Record the final summary line (failed/passed/skipped/errors) and every FAILED/ERROR/SKIPPED id.
- Run the same command from the main tree (`cd /workspace/collab-splats`, `PYTHONPATH=/workspace/collab-splats`) into `main_gate.txt`. The SKIP counts must match; a worktree surplus means a missing third_party symlink.

- [ ] **Step 4: instantsfm reference runs A and B (baseline code, seeded)**

A tmux session with nothing else running. Preproc runs once, and both runs read the same keyframes.
- Write `$SCRATCH/insfm_seed.yaml`:

```yaml
pointcloud: {method: sfm, backend: instantsfm, instantsfm: {random_seed: 0}}
semantics: {enabled: false}
```

- Then run:

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && tmux new -d -s sfm-base "PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python docs/examples/run_pipeline.py --output-root /workspace/outputs/sfm-backends/base-A --config $SCRATCH/insfm_seed.yaml --stages preproc,pointcloud data/tutorial/tutorial_example-video.mp4 > $SCRATCH/base-A.log 2>&1; echo DONE >> $SCRATCH/base-A.log"
```

- When A is done: copy the scene's `images/` (plus its `frames.json`) into `/workspace/outputs/sfm-backends/base-B/<scene>/`. Run B with `--stages pointcloud` only, so both use identical keyframes.
- Then compare A vs B with `$SCRATCH/compare_sfm.py`:

```python
"""Compare two sfm scene dirs: sparse/0 binaries byte-for-byte, pointcloud.zarr arrays + attrs."""
import hashlib, sys
from pathlib import Path
import numpy as np, zarr

a, b = Path(sys.argv[1]), Path(sys.argv[2])
for f in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
    pa, pb = a / "colmap/sparse/0" / f, b / "colmap/sparse/0" / f
    if pa.exists() or pb.exists():
        ha = hashlib.sha256(pa.read_bytes()).hexdigest()
        hb = hashlib.sha256(pb.read_bytes()).hexdigest()
        print(f, "SAME" if ha == hb else "DIFF")
za, zb = zarr.open(str(a / "pointcloud.zarr"), mode="r"), zarr.open(str(b / "pointcloud.zarr"), mode="r")
print("attrs", "SAME" if dict(za.attrs) == dict(zb.attrs) else f"DIFF {dict(za.attrs)} {dict(zb.attrs)}")
for k in sorted(za.array_keys()):
    same = k in zb and np.array_equal(np.asarray(za[k]), np.asarray(zb[k]))
    print(k, "SAME" if same else "DIFF")
print("db", hashlib.sha256((a / "colmap/instantsfm.db").read_bytes()).hexdigest()[:12],
      hashlib.sha256((b / "colmap/instantsfm.db").read_bytes()).hexdigest()[:12])
```

- The scene dir is `<output-root>/<scene>/instantsfm`.
- If A == B on every line: the tip gate (Task 11) is full byte identity.
- Otherwise the gate downgrades to identical `instantsfm.db` build argv (Task 1 snapshot test) plus DB hash on a fresh build. Record which gate applies.

---

### Task 1: Move the SIFT helpers into `sfm/sift_db.py` (pure move)

**Files:**
- Create: `collab_splats/pointcloud/sfm/sift_db.py`
- Modify: `collab_splats/pointcloud/sfm/instantsfm.py` (delete `_sfm_image_dir`, `_sift_database_valid`, `_generate_sift_database`, `_rename_images_to_stems`; import the new names)
- Create: `tests/pointcloud/sfm/test_sift_db.py`
- Modify: `tests/pointcloud/sfm/test_instantsfm.py` (remove the moved test sections)

- [ ] **Step 1: Write the snapshot test for today's exhaustive argv, against the OLD function first**

This test is the "before the move" snapshot. It is written against `instantsfm._generate_sift_database` and run BEFORE any code moves. Put it in `tests/pointcloud/sfm/test_sift_db.py`:

```python
"""
Shared system-colmap SIFT database helpers (sfm/sift_db.py).
"""

import subprocess

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm import instantsfm

########################################################
########## SIFT database build: argv ###################
########################################################


def _capture_argv(monkeypatch, *, gpu):
    """
    Patch subprocess.run and torch.cuda; returns the list every colmap argv lands in.
    """
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.setattr("torch.cuda.is_available", lambda: gpu)
    return calls


# Today's instantsfm argv, verbatim: the byte-identical instantsfm gate rests on it
_EXHAUSTIVE_CPU = [
    ["colmap", "feature_extractor", "--image_path", "IMG", "--database_path", "DB",
     "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
     "--SiftExtraction.use_gpu", "0", "--SiftExtraction.num_threads", "8"],
    ["colmap", "exhaustive_matcher", "--database_path", "DB", "--SiftMatching.use_gpu", "0",
     "--SiftMatching.num_threads", "8"],
]
_EXHAUSTIVE_GPU = [
    ["colmap", "feature_extractor", "--image_path", "IMG", "--database_path", "DB",
     "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
     "--SiftExtraction.use_gpu", "1"],
    ["colmap", "exhaustive_matcher", "--database_path", "DB", "--SiftMatching.use_gpu", "1"],
]


@pytest.mark.parametrize("gpu,expected", [(False, _EXHAUSTIVE_CPU), (True, _EXHAUSTIVE_GPU)])
def test_exhaustive_argv_matches_the_pre_move_snapshot(monkeypatch, gpu, expected):
    calls = _capture_argv(monkeypatch, gpu=gpu)
    instantsfm._generate_sift_database("IMG", "DB")
    assert calls == expected
```

`torch.cuda.is_available` is patched by dotted string, which is the same object `instantsfm` imports.

- [ ] **Step 2: Run it against the old code; it must PASS (that is the snapshot)**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_sift_db.py -q -p no:cacheprovider
```

Expected: 2 passed. If it fails, fix the literal to what the old code emits; never change old code in this step.

- [ ] **Step 3: Create `sift_db.py` holding the four moved helpers, renamed public, wording neutral**

```python
"""
System-colmap SIFT database + COLMAP model helpers shared by the sfm creators.

- SIFT extraction/matching drives the CUDA-built `colmap` CLI; the pycolmap wheel is CPU-only
- used by InstantSfMCreator and ColmapCreator; not re-exported from sfm/__init__.py
- output contract helpers: stem image names, largest incremental model
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pycolmap
import torch

logger = logging.getLogger(__name__)


########################################
# Image directory
########################################


def sfm_image_dir(images_dir: Path) -> Path:
    """
    Image directory an sfm creator reads.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        The same directory — the store IS the COLMAP image layout, so nothing is staged.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"sfm expects an image directory at {images_dir} — run preprocess first")
    return images_dir
```

- Then move `_sift_database_valid` VERBATIM as `sift_database_valid(database_path: Path, image_names: list[str]) -> bool`, body and docstring unchanged except:
  - Args `database_path: the scene's SIFT database (colmap/instantsfm.db or colmap/colmap.db).`
  - `ReadColmapDatabase` → "the mapper".
- Move `_generate_sift_database` VERBATIM as `build_sift_database(image_path: Path, database_path: Path, *, num_threads: int = 8) -> None`. Only the log line changes: `logger.info("colmap: running %s %s (%s)", ...)`.
- Move `_rename_images_to_stems` as `rename_images_to_stems`. In its docstring, "InstantSfM registers images" → "COLMAP mappers register images".
- The section dividers are `# SIFT feature database (system colmap)` and `# Output contract`.
- In `instantsfm.py`:
  - delete the four functions;
  - drop the now-unused `os` and `subprocess` imports only if nothing else uses them (grep first);
  - add `from collab_splats.pointcloud.sfm.sift_db import build_sift_database, rename_images_to_stems, sfm_image_dir, sift_database_valid`;
  - replace the four call sites (lines ~472, 503, 506, 556), and the docstring mention at ~392 (`_generate_sift_database above` → `sift_db.build_sift_database`).
- Keep `torch` in instantsfm only if still used (it is: `_patch_*`; grep).

- [ ] **Step 4: Repoint the snapshot test and move the old tests**

- In `test_sift_db.py`: `from collab_splats.pointcloud.sfm import sift_db` replaces the `instantsfm` import; the call becomes `sift_db.build_sift_database("IMG", "DB")`.
- Move from `tests/pointcloud/sfm/test_instantsfm.py` into `test_sift_db.py`, verbatim except for `instantsfm._x` → `sift_db.x`:
  - the `SIFT database validity` section (`_DB_NAMES`, `_sift_db`, 7 tests);
  - the `Stem rename` section (`_recon`, 1 test);
  - the `Image directory` section (2 tests).
- Delete them from `test_instantsfm.py`, and drop that file's imports that become unused.
- `test_sfm_image_dir_refuses_a_missing_directory` matches `"image directory"`, which still holds.

- [ ] **Step 5: Run**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm tests/test_docstring_contract.py -q -p no:cacheprovider
```

Expected: every test passes. The moved-test count equals the removed count (12 = 2 snapshot + 7 + 1 + 2 in `test_sift_db.py`).

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && black collab_splats/pointcloud/sfm/sift_db.py collab_splats/pointcloud/sfm/instantsfm.py tests/pointcloud/sfm/test_sift_db.py tests/pointcloud/sfm/test_instantsfm.py && isort collab_splats/pointcloud/sfm/sift_db.py collab_splats/pointcloud/sfm/instantsfm.py tests/pointcloud/sfm/test_sift_db.py tests/pointcloud/sfm/test_instantsfm.py && git add collab_splats/pointcloud/sfm/sift_db.py tests/pointcloud/sfm/test_sift_db.py && git commit --only collab_splats/pointcloud/sfm/sift_db.py collab_splats/pointcloud/sfm/instantsfm.py tests/pointcloud/sfm/test_sift_db.py tests/pointcloud/sfm/test_instantsfm.py -m "refactor(pointcloud): move system-colmap SIFT helpers into sfm/sift_db.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `sift_db` gains pairing, params sidecar, vocab tree, `largest_model`, CLI version

**Files:**
- Modify: `collab_splats/pointcloud/sfm/sift_db.py`
- Test: `tests/pointcloud/sfm/test_sift_db.py`

Pinned constants, measured 2026-09-26:
- Both mirrors serve the same 15,229,678-byte file with sha256 `d37d8f19ee0a49705c4c0b06967a08cedfed5cf86519eada3271497256732bc2`.
- colmap 3.10-dev loads it: `vocab_tree_matcher` and `sequential_matcher --SequentialMatching.loop_detection 1` both exit 0 on 12 GH010229 frames.

- [ ] **Step 1: Failing tests (append to `test_sift_db.py`)**

```python
@pytest.mark.parametrize(
    "pairing,matcher_tail",
    [
        ("exhaustive", ["exhaustive_matcher", "--database_path", "DB", "--SiftMatching.use_gpu", "1"]),
        (
            "sequential",
            ["sequential_matcher", "--database_path", "DB", "--SiftMatching.use_gpu", "1",
             "--SequentialMatching.overlap", "5", "--SequentialMatching.quadratic_overlap", "0"],
        ),
        (
            "sequential+retrieval",
            ["sequential_matcher", "--database_path", "DB", "--SiftMatching.use_gpu", "1",
             "--SequentialMatching.overlap", "5", "--SequentialMatching.quadratic_overlap", "0",
             "--SequentialMatching.loop_detection", "1",
             "--SequentialMatching.loop_detection_num_images", "7",
             "--SequentialMatching.vocab_tree_path", "VT"],
        ),
        (
            "retrieval",
            ["vocab_tree_matcher", "--database_path", "DB", "--SiftMatching.use_gpu", "1",
             "--VocabTreeMatching.num_images", "7", "--VocabTreeMatching.vocab_tree_path", "VT"],
        ),
    ],
)
def test_matcher_argv_per_pairing(monkeypatch, pairing, matcher_tail):
    calls = _capture_argv(monkeypatch, gpu=True)
    sift_db.build_sift_database("IMG", "DB", pairing=pairing, overlap=5, num_retrieved=7, vocab_tree="VT")
    assert calls[0] == _EXHAUSTIVE_GPU[0]
    assert calls[1] == ["colmap"] + matcher_tail


def test_retrieval_pairing_without_a_vocab_tree_is_refused(monkeypatch):
    _capture_argv(monkeypatch, gpu=True)
    with pytest.raises(ValueError, match="vocab_tree"):
        sift_db.build_sift_database("IMG", "DB", pairing="retrieval")


def test_unknown_pairing_is_refused(monkeypatch):
    _capture_argv(monkeypatch, gpu=True)
    with pytest.raises(ValueError, match="pairing"):
        sift_db.build_sift_database("IMG", "DB", pairing="spatial")


_PARAMS = {"pairing": "sequential", "overlap": 10, "num_retrieved": 20}


def test_params_none_ignores_the_sidecar(tmp_path):
    # instantsfm's gate: no sidecar written, none read — today's behavior byte for byte
    db = tmp_path / "instantsfm.db"
    _sift_db(db, verified_pair=True)
    assert sift_db.sift_database_valid(db, _DB_NAMES) is True


def test_params_given_requires_a_matching_sidecar(tmp_path):
    db = tmp_path / "colmap.db"
    _sift_db(db, verified_pair=True)
    assert sift_db.sift_database_valid(db, _DB_NAMES, params=_PARAMS) is False  # no sidecar yet

    sift_db.write_database_params(db, _PARAMS)
    assert sift_db.sift_database_valid(db, _DB_NAMES, params=_PARAMS) is True
    assert sift_db.sift_database_valid(db, _DB_NAMES, params=dict(_PARAMS, pairing="exhaustive")) is False


def test_write_database_params_lands_beside_the_db(tmp_path):
    db = tmp_path / "colmap.db"
    sift_db.write_database_params(db, _PARAMS)
    assert (tmp_path / "colmap.db.json").is_file()


def test_largest_model_picks_by_registered_count_not_key():
    small, big = _recon(["a.png"]), _recon(["b.png", "c.png", "d.png"])
    assert sift_db.largest_model({0: small, 1: big}) is big


def test_largest_model_refuses_an_empty_result():
    with pytest.raises(RuntimeError, match="no model"):
        sift_db.largest_model({})


def test_fetch_vocab_tree_uses_a_cached_file_with_the_pinned_hash(tmp_path, monkeypatch):
    cached = tmp_path / sift_db.VOCAB_TREE_NAME
    cached.write_bytes(b"x")
    monkeypatch.setattr(sift_db, "VOCAB_TREE_SHA256", __import__("hashlib").sha256(b"x").hexdigest())
    monkeypatch.setattr(sift_db.urllib.request, "urlretrieve", lambda *a, **k: pytest.fail("downloaded"))
    assert sift_db.fetch_vocab_tree(tmp_path) == cached


def test_fetch_vocab_tree_rejects_a_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sift_db.urllib.request, "urlretrieve", lambda url, dst: Path(dst).write_bytes(b"bad"))
    with pytest.raises(RuntimeError, match="sha256"):
        sift_db.fetch_vocab_tree(tmp_path)
    assert not (tmp_path / sift_db.VOCAB_TREE_NAME).exists()


def test_colmap_cli_version_reads_the_banner(monkeypatch):
    banner = "COLMAP 3.10-dev -- Structure-from-Motion and Multi-View Stereo\n(Commit 879a296a on 2024-02-13 with CUDA)\n\nUsage:\n"
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=banner, stderr="")
    )
    assert sift_db.colmap_cli_version() == (
        "COLMAP 3.10-dev -- Structure-from-Motion and Multi-View Stereo (Commit 879a296a on 2024-02-13 with CUDA)"
    )
```

Add `from pathlib import Path` to the test imports.

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_sift_db.py -q -p no:cacheprovider
```

Expected: the new tests FAIL (`TypeError: unexpected keyword 'pairing'`, `AttributeError: write_database_params`, etc.). The moved tests still pass.

- [ ] **Step 3: Implement in `sift_db.py`**

- New imports: `hashlib`, `json`, `urllib.request`.
- Constants, under a `# Constants` divider, placed after `logger`:

```python
PAIRINGS = ("sequential", "retrieval", "sequential+retrieval", "exhaustive")

# COLMAP's flickr100K 32K-word vocab tree, the pre-3.12 format a 3.10 binary reads
# - measured 2026-09-26: both URLs serve the same 15,229,678 bytes; colmap 3.10-dev loads it
VOCAB_TREE_NAME = "vocab_tree_flickr100K_words32K.bin"
VOCAB_TREE_URL = f"https://github.com/colmap/colmap/releases/download/3.11.1/{VOCAB_TREE_NAME}"
VOCAB_TREE_SHA256 = "d37d8f19ee0a49705c4c0b06967a08cedfed5cf86519eada3271497256732bc2"
VOCAB_TREE_CACHE = Path.home() / ".cache" / "collab_splats"
```

- `build_sift_database` new signature and matcher section. The extractor command and the error handling are unchanged:

```python
def build_sift_database(
    image_path: Path,
    database_path: Path,
    *,
    pairing: str = "exhaustive",
    overlap: int = 10,
    num_retrieved: int = 20,
    vocab_tree: Path | None = None,
    num_threads: int = 8,
) -> None:
```

- Docstring:
  - keep the existing bullets;
  - add `Args:` for all seven params: `pairing` is one of PAIRINGS; `overlap` is sequential neighbors; `num_retrieved` is vocab-tree neighbors per image; `vocab_tree` is required when pairing retrieves;
  - add a bullet: `- sequential sets quadratic_overlap 0 so "sequential" means i with i+1..i+N, as hloc's generator does`.
- Body, at the top:

```python
    # Pairing is checked before any subprocess: a bad value must not leave a half-built DB
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    if "retrieval" in pairing and vocab_tree is None:
        raise ValueError(f"pairing {pairing!r} needs a vocab_tree (see fetch_vocab_tree)")
```

- Matcher command replaces the fixed `exhaustive_matcher` list:

```python
    # Matcher per pairing; the GPU flag stays right after the DB path, as before the split
    matcher = {"exhaustive": "exhaustive_matcher", "retrieval": "vocab_tree_matcher"}.get(pairing, "sequential_matcher")
    matcher_cmd = ["colmap", matcher, "--database_path", str(database_path), "--SiftMatching.use_gpu", "1" if use_gpu else "0"]
    if pairing.startswith("sequential"):
        matcher_cmd += ["--SequentialMatching.overlap", str(overlap), "--SequentialMatching.quadratic_overlap", "0"]
    if pairing == "sequential+retrieval":
        matcher_cmd += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.loop_detection_num_images", str(num_retrieved),
            "--SequentialMatching.vocab_tree_path", str(vocab_tree),
        ]
    if pairing == "retrieval":
        matcher_cmd += ["--VocabTreeMatching.num_images", str(num_retrieved), "--VocabTreeMatching.vocab_tree_path", str(vocab_tree)]
```

Note: `loop_detection_num_images` = `num_retrieved` is how the spec's "num_retrieved used whenever pairing retrieves" reaches colmap's sequential+retrieval mode.

- `sift_database_valid(database_path, image_names, *, params: dict | None = None)`:
  - Add a docstring Arg: `params: matching params the DB must have been built with, compared against the colmap.db.json sidecar; None skips the check (instantsfm).`
  - Insert after the exists() check, before opening the DB:

```python
    # Matching params live in a sidecar: the DB itself does not record which matcher ran
    if params is not None:
        sidecar = database_params_path(database_path)
        if not sidecar.is_file() or json.loads(sidecar.read_text()) != params:
            logger.info("SIFT database %s was matched with other params — rebuilding", database_path)
            return False
```

- New helpers:

```python
def database_params_path(database_path: Path) -> Path:
    """
    Sidecar path for a DB's matching params: <db>.json beside it.

    Args:
        database_path: the SIFT database.

    Returns:
        database_path with `.json` appended to its name.
    """
    database_path = Path(database_path)
    return database_path.with_name(database_path.name + ".json")


def write_database_params(database_path: Path, params: dict) -> None:
    """
    Record the matching params a finished DB was built with.

    - written only after a successful build, so a crash never leaves a sidecar vouching for it

    Args:
        database_path: the SIFT database just built.
        params: JSON-serializable matching params (pairing, overlap, num_retrieved).
    """
    database_params_path(database_path).write_text(json.dumps(params, sort_keys=True))


def fetch_vocab_tree(cache_dir: Path | None = None) -> Path:
    """
    Path to the pinned COLMAP vocab tree, downloaded once into the cache.

    - sha256-checked on every call; a mismatched download is deleted, never used

    Args:
        cache_dir: where the file lives; defaults to ~/.cache/collab_splats.

    Returns:
        The local vocab tree path.
    """
    cache_dir = Path(cache_dir or VOCAB_TREE_CACHE)
    path = cache_dir / VOCAB_TREE_NAME

    # Download on a cache miss; network or hash failure raises with the URL and the path
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("colmap: downloading %s -> %s", VOCAB_TREE_URL, path)
        try:
            urllib.request.urlretrieve(VOCAB_TREE_URL, path)
        except OSError as err:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"vocab tree download failed ({err}): {VOCAB_TREE_URL} -> {path}") from err

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != VOCAB_TREE_SHA256:
        path.unlink()
        raise RuntimeError(f"vocab tree sha256 {digest} != pinned {VOCAB_TREE_SHA256} ({VOCAB_TREE_URL}, {path})")
    return path


def largest_model(recons: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction:
    """
    The incremental-mapping model with the most registered images.

    - incremental mapping returns one model per disconnected component, in no size order

    Args:
        recons: pycolmap.incremental_mapping's {index: model} result.

    Returns:
        The largest model; warns when there was more than one.
    """
    if not recons:
        raise RuntimeError("incremental mapping produced no model — too little overlap between frames")
    best = max(recons.values(), key=lambda r: r.num_reg_images())
    if len(recons) > 1:
        sizes = sorted((r.num_reg_images() for r in recons.values()), reverse=True)
        logger.warning("incremental mapping split the scene into %d models %s — keeping the largest", len(recons), sizes)
    return best


def colmap_cli_version() -> str:
    """
    The system colmap's version banner, for zarr provenance.

    Returns:
        The first two lines of `colmap -h` joined (version + commit), or "unknown".
    """
    try:
        out = subprocess.run(["colmap", "-h"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return "unknown"
    return " ".join(line.strip() for line in out.splitlines()[:2] if line.strip()) or "unknown"
```

`colmap_cli_version` takes two lines because line 1 has no commit; the spec said "first line", and this is a deliberate refinement.

- [ ] **Step 4: Run**

Command as in Step 2. Expected: all pass.

- [ ] **Step 5: Commit** (same pattern; message `feat(pointcloud): sift_db pairing modes, params sidecar, vocab tree, largest_model`)

---

### Task 3: Config — `base.yaml` blocks + validation

**Files:**
- Modify: `configs/base.yaml:33-34` (backend comment), add the blocks after the `instantsfm:` block (line ~89)
- Modify: `collab_splats/wrapper/reconstructor.py` (validation only; `_SFM_BACKENDS` stays `{"instantsfm"}` until Task 7)
- Test: `tests/wrapper/test_sfm_config.py`

- [ ] **Step 1: `base.yaml`**

Line 34 becomes `# sfm: instantsfm | colmap | hloc`.

After the instantsfm block:

```yaml
  # COLMAP incremental SfM (method: sfm, backend: colmap): colmap CLI SIFT + pycolmap mapper.
  colmap:
    pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
    overlap: 10              # sequential: pair each frame with the next N
    num_retrieved: 20        # retrieval: vocab-tree neighbors per image (15 MB tree, fetched once)
    num_threads: 8           # CPU SIFT/mapper thread cap; colmap's default spawns 96 and OOMs
    min_registered_frac: 0.5 # fail below this share of frames registered; above it, subset
  # hloc incremental SfM (method: sfm, backend: hloc): learned features + pycolmap mapper.
  # Needs the optional `hloc` extra (setup/hloc.sh, then the user-run uv lock + sync).
  hloc:
    pairing: sequential+retrieval
    overlap: 10
    num_retrieved: 20        # retrieval: top-k global-descriptor neighbors per image
    retrieval_conf: netvlad  # hloc.extract_features.confs key
    feature_conf: superpoint_max          # hloc.extract_features.confs key
    matcher_conf: superpoint+lightglue    # hloc.match_features.confs key
    num_threads: 8
    min_registered_frac: 0.5
```

- [ ] **Step 2: Failing tests (append to `tests/wrapper/test_sfm_config.py`)**

```python
_COLMAP_DEFAULTS = {
    "pairing": "sequential+retrieval",
    "overlap": 10,
    "num_retrieved": 20,
    "num_threads": 8,
    "min_registered_frac": 0.5,
}


def test_base_yaml_has_colmap_and_hloc_blocks():
    cfg = _base_config()
    assert cfg["pointcloud"]["colmap"] == _COLMAP_DEFAULTS
    assert cfg["pointcloud"]["hloc"] == dict(
        _COLMAP_DEFAULTS,
        retrieval_conf="netvlad",
        feature_conf="superpoint_max",
        matcher_conf="superpoint+lightglue",
    )


def _sfm_cfg(backend, **block):
    """
    base.yaml with method: sfm, the given backend, and `block` merged into its sub-block.
    """
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = backend
    cfg["pointcloud"][backend].update(block)
    return cfg


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
@pytest.mark.parametrize("pairing", ["sequential", "retrieval", "sequential+retrieval", "exhaustive"])
def test_every_pairing_validates(backend, pairing):
    Reconstructor._validate_sfm_block(_sfm_cfg(backend, pairing=pairing)["pointcloud"], backend)


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
@pytest.mark.parametrize(
    "key,bad",
    [
        ("pairing", "spatial"),
        ("pairing", None),
        ("overlap", 0),
        ("overlap", True),
        ("overlap", 2.0),
        ("num_retrieved", 0),
        ("num_retrieved", False),
        ("num_threads", -1),
        ("num_threads", "8"),
        ("min_registered_frac", 0),
        ("min_registered_frac", 1.5),
        ("min_registered_frac", True),
        ("min_registered_frac", "0.5"),
        ("typo_key", 1),
    ],
)
def test_bad_sfm_block_values_are_rejected(backend, key, bad):
    with pytest.raises(ValueError, match=f"pointcloud.{backend}"):
        Reconstructor._validate_sfm_block(_sfm_cfg(backend, **{key: bad})["pointcloud"], backend)


@pytest.mark.parametrize("key", ["retrieval_conf", "feature_conf", "matcher_conf"])
@pytest.mark.parametrize("bad", ["", None, 3])
def test_hloc_conf_keys_must_be_non_empty_strings(key, bad):
    with pytest.raises(ValueError, match=f"pointcloud.hloc.{key}"):
        Reconstructor._validate_sfm_block(_sfm_cfg("hloc", **{key: bad})["pointcloud"], "hloc")


def test_min_registered_frac_accepts_one_and_an_int_one():
    for good in (1.0, 1, 0.01):
        Reconstructor._validate_sfm_block(_sfm_cfg("colmap", min_registered_frac=good)["pointcloud"], "colmap")


def test_colmap_rejects_hloc_only_keys():
    with pytest.raises(ValueError, match="pointcloud.colmap"):
        Reconstructor._validate_sfm_block(_sfm_cfg("colmap", feature_conf="sift")["pointcloud"], "colmap")
```

The tests call a static helper `Reconstructor._validate_sfm_block(pc, backend)` directly, because until Task 7 `validate_config` still rejects the backend itself. Task 7 adds end-to-end `validate_config` acceptance tests.

- [ ] **Step 3: Run to confirm failure** (`AttributeError: _validate_sfm_block`)

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_config.py -q -p no:cacheprovider
```

- [ ] **Step 4: Implement**

- Constants block in `reconstructor.py`, after `_SFM_BACKENDS`:

```python
# colmap / hloc sub-block keys; anything else is a typo that would TypeError in the creator
_SFM_BLOCK_KEYS = {
    "colmap": {"pairing", "overlap", "num_retrieved", "num_threads", "min_registered_frac"},
    "hloc": {
        "pairing",
        "overlap",
        "num_retrieved",
        "retrieval_conf",
        "feature_conf",
        "matcher_conf",
        "num_threads",
        "min_registered_frac",
    },
}
```

- `_SFM_PAIRINGS`: add `from collab_splats.pointcloud.sfm.sift_db import PAIRINGS as _SFM_PAIRINGS` to the top imports. Reconstructor already imports the sfm chain through `InstantSfMCreator`, so this costs nothing at config load, and one tuple stays the source of truth.

- Static method on `Reconstructor`, placed right after `validate_config`:

```python
    @staticmethod
    def _validate_sfm_block(pc: dict, backend: str) -> None:
        """
        Bounds-check a colmap / hloc sub-block at config load.

        - every key is consumed after SIFT or the mapper has started, so a typo here would
          otherwise cost a whole run
        - hloc conf names are NOT checked against hloc.*.confs: that would import the optional
          extra at config load

        Args:
            pc: the config's pointcloud section.
            backend: "colmap" or "hloc".
        """
        block = pc.get(backend) or {}

        # Unknown keys would reach the creator constructor as a TypeError, mid-run
        unknown = set(block) - _SFM_BLOCK_KEYS[backend]
        if unknown:
            raise ValueError(f"pointcloud.{backend} has unknown keys {sorted(unknown)}")

        pairing = block.get("pairing")
        if pairing not in _SFM_PAIRINGS:
            raise ValueError(f"pointcloud.{backend}.pairing must be one of {_SFM_PAIRINGS}, got {pairing!r}")

        # Counts: bool is an int subclass, so it is rejected explicitly
        for key in ("overlap", "num_retrieved", "num_threads"):
            value = block.get(key)
            if isinstance(value, bool) or not (isinstance(value, int) and value >= 1):
                raise ValueError(f"pointcloud.{backend}.{key} must be an int >= 1, got {value!r}")

        frac = block.get("min_registered_frac")
        if isinstance(frac, bool) or not (isinstance(frac, (int, float)) and 0 < frac <= 1):
            raise ValueError(f"pointcloud.{backend}.min_registered_frac must be a number in (0, 1], got {frac!r}")

        # hloc conf names: non-empty strings; hloc itself resolves them at run time
        if backend == "hloc":
            for key in ("retrieval_conf", "feature_conf", "matcher_conf"):
                value = block.get(key)
                if not (isinstance(value, str) and value):
                    raise ValueError(f"pointcloud.hloc.{key} must be a non-empty string, got {value!r}")
```

- In `validate_config`, after the instantsfm sub-block check (line ~911):

```python
        # colmap / hloc sub-block bounds check
        if method == "sfm" and backend in _SFM_BLOCK_KEYS:
            Reconstructor._validate_sfm_block(pc, backend)
```

- Also make the BA/LC refusal messages backend-neutral:
  - `"... with method: sfm — every sfm backend runs its own bundle adjustment"`
  - `"... with method: sfm — sfm backends map the whole frame set at once, not in sequential submaps"`
  - The comment above the BA one becomes `# SfM path: every sfm mapper runs its own BA — refuse the flag`.
- Update `test_sfm_rejects_bundle_adjustment` docstring: `bundle_adjustment is the sfm mapper's own job — refused at validation.` The `match=` args are unchanged.

- [ ] **Step 5: Run.** Expected: all pass, including the two unchanged refusal tests (they match on the key name).

- [ ] **Step 6: Commit** `feat(wrapper): colmap + hloc config blocks and validation` (files: `configs/base.yaml`, `collab_splats/wrapper/reconstructor.py`, `tests/wrapper/test_sfm_config.py`).

---

### Task 4: `ColmapCreator`

**Files:**
- Rewrite: `collab_splats/pointcloud/sfm/colmap.py`
- Rewrite: `tests/pointcloud/sfm/test_colmap.py`

- [ ] **Step 1: Failing tests (replace the file)**

```python
"""
ColmapCreator: colmap CLI SIFT DB + pycolmap incremental mapping, all heavy legs mocked.
"""

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm import colmap as colmap_mod
from collab_splats.pointcloud.sfm.colmap import ColmapCreator
from collab_splats.preproc import frames as fr

NAMES = ["frame_000000.png", "frame_000009.png", "frame_000030.png"]


def _recon(names):
    """
    One PINHOLE camera, one image per name, one point3D observed in every image.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


def _scene(tmp_path):
    """
    A scene dir whose images/ holds NAMES as real PNG keyframes.
    """
    data_dir = tmp_path / "colmap_backend"
    images_dir = tmp_path / "images"
    fr.write_frames(
        images_dir,
        [np.zeros((8, 8, 3), np.uint8)] * len(NAMES),
        [{"frame_idx": int(n[6:12]), "blur_score": 1.0} for n in NAMES],
        {"video_path": "x.mp4", "method": "uniform"},
    )
    return data_dir, images_dir


@pytest.fixture
def mocked(monkeypatch):
    """
    Stub the DB build, vocab fetch and mapper; records their calls.
    """
    calls = {"build": [], "map": []}

    def build(image_path, db_path, **kw):
        calls["build"].append(kw)
        Path(db_path).write_bytes(b"db")

    def mapping(db, image_dir, out, options):
        calls["map"].append({"db": db, "image_dir": image_dir, "out": out, "options": options})
        return {0: _recon(NAMES[:1]), 1: _recon(NAMES)}

    monkeypatch.setattr(colmap_mod, "build_sift_database", build)
    monkeypatch.setattr(colmap_mod, "fetch_vocab_tree", lambda: Path("/vt.bin"))
    monkeypatch.setattr(colmap_mod, "sift_database_valid", lambda *a, **k: False)
    monkeypatch.setattr(colmap_mod.pycolmap, "incremental_mapping", mapping)
    return calls


def test_reconstruct_writes_the_largest_model_to_sparse_0_with_stems(tmp_path, mocked):
    data_dir, images_dir = _scene(tmp_path)
    recon = ColmapCreator().reconstruct(data_dir, images_dir=images_dir)

    assert sorted(im.name for im in recon.images.values()) == [Path(n).stem for n in NAMES]
    reread = pycolmap.Reconstruction(str(data_dir / "colmap" / "sparse" / "0"))
    assert reread.num_reg_images() == len(NAMES)
    assert not (data_dir / "colmap" / "mapper").exists()


def test_reconstruct_passes_pairing_params_and_thread_cap(tmp_path, mocked):
    data_dir, images_dir = _scene(tmp_path)
    ColmapCreator(pairing="retrieval", overlap=3, num_retrieved=4, num_threads=2).reconstruct(
        data_dir, images_dir=images_dir
    )
    assert mocked["build"] == [
        {"pairing": "retrieval", "overlap": 3, "num_retrieved": 4, "vocab_tree": Path("/vt.bin"), "num_threads": 2}
    ]
    assert mocked["map"][0]["options"] == {"num_threads": 2}
    assert mocked["map"][0]["db"] == str(data_dir / "colmap" / "colmap.db")


def test_sequential_pairing_never_fetches_the_vocab_tree(tmp_path, mocked, monkeypatch):
    monkeypatch.setattr(colmap_mod, "fetch_vocab_tree", lambda: pytest.fail("fetched"))
    data_dir, images_dir = _scene(tmp_path)
    ColmapCreator(pairing="sequential").reconstruct(data_dir, images_dir=images_dir)
    assert mocked["build"][0]["vocab_tree"] is None


def test_a_valid_db_is_reused_and_a_rebuild_writes_the_sidecar(tmp_path, mocked, monkeypatch):
    data_dir, images_dir = _scene(tmp_path)
    ColmapCreator().reconstruct(data_dir, images_dir=images_dir)
    sidecar = data_dir / "colmap" / "colmap.db.json"
    assert sidecar.is_file()

    monkeypatch.setattr(colmap_mod, "sift_database_valid", lambda *a, **k: True)
    ColmapCreator().reconstruct(data_dir, images_dir=images_dir)
    assert len(mocked["build"]) == 1


def test_stale_sparse_tree_is_removed(tmp_path, mocked):
    data_dir, images_dir = _scene(tmp_path)
    stale = data_dir / "colmap" / "sparse" / "1"
    stale.mkdir(parents=True)
    ColmapCreator().reconstruct(data_dir, images_dir=images_dir)
    assert not stale.exists()


def test_no_model_raises(tmp_path, mocked, monkeypatch):
    monkeypatch.setattr(colmap_mod.pycolmap, "incremental_mapping", lambda *a, **k: {})
    data_dir, images_dir = _scene(tmp_path)
    with pytest.raises(RuntimeError, match="no model"):
        ColmapCreator().reconstruct(data_dir, images_dir=images_dir)
```

- [ ] **Step 2: Run to confirm failure** (the old class takes `image_dir, output_dir`)

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_colmap.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement (replace `colmap.py`)**

```python
"""
COLMAP incremental SfM: colmap CLI SIFT + pycolmap incremental mapping.

- `pointcloud: {method: sfm, backend: colmap}`; Reconstructor._run_sfm dispatches here
- SIFT DB at <data_dir>/colmap/colmap.db (reused while its image set + matching params hold)
- output contract (shared with InstantSfMCreator): stem image names, model at colmap/sparse/0
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from collab_splats.pointcloud.sfm.sift_db import (
    build_sift_database,
    database_params_path,
    fetch_vocab_tree,
    largest_model,
    rename_images_to_stems,
    sfm_image_dir,
    sift_database_valid,
    write_database_params,
)
from collab_splats.preproc.frames import frame_paths

logger = logging.getLogger(__name__)


@dataclass
class ColmapCreator:
    """
    Classical incremental SfM on a scene directory.

    - features/matches: the CUDA colmap CLI (3.10); mapping: the pycolmap wheel (4.x)
    - one shared SIMPLE_RADIAL camera, refined by the mapper — same freedom as instantsfm

    Args:
        pairing: sequential | retrieval | sequential+retrieval | exhaustive.
        overlap: sequential neighbors per frame.
        num_retrieved: vocab-tree neighbors per frame when pairing retrieves.
        num_threads: CPU SIFT + mapper thread cap.
    """

    pairing: str = "sequential+retrieval"
    overlap: int = 10
    num_retrieved: int = 20
    num_threads: int = 8

    def reconstruct(self, data_dir: Path, images_dir: Path | None = None) -> pycolmap.Reconstruction:
        """
        Run SIFT + incremental mapping over the scene's keyframes.

        Args:
            data_dir: backend working directory; colmap/ lives here.
            images_dir: COLMAP-shaped image directory; defaults to data_dir/images.

        Returns:
            The largest model, image names renamed to stems, also written to colmap/sparse/0.
        """
        data_dir = Path(data_dir)
        image_dir = sfm_image_dir(data_dir / "images" if images_dir is None else images_dir)
        colmap_dir = data_dir / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(colmap_dir / "sparse", ignore_errors=True)

        # SIFT DB: reuse only when the image set AND the matching params both match
        db_path = colmap_dir / "colmap.db"
        params = {"pairing": self.pairing, "overlap": self.overlap, "num_retrieved": self.num_retrieved}
        names = [p.name for p in frame_paths(image_dir)]
        if not sift_database_valid(db_path, names, params=params):
            db_path.unlink(missing_ok=True)
            database_params_path(db_path).unlink(missing_ok=True)
            logger.info("colmap: building SIFT database (%s)", self.pairing)
            build_sift_database(
                image_dir,
                db_path,
                pairing=self.pairing,
                overlap=self.overlap,
                num_retrieved=self.num_retrieved,
                vocab_tree=fetch_vocab_tree() if "retrieval" in self.pairing else None,
                num_threads=self.num_threads,
            )
            write_database_params(db_path, params)

        # Incremental mapping writes one mapper/<idx>/ per component, so it cannot target sparse/
        mapper_dir = colmap_dir / "mapper"
        shutil.rmtree(mapper_dir, ignore_errors=True)
        mapper_dir.mkdir()
        recons = pycolmap.incremental_mapping(
            str(db_path), str(image_dir), str(mapper_dir), options={"num_threads": self.num_threads}
        )
        recon = largest_model(recons)

        # Contract layout: stems, sparse/0; the per-component scratch goes
        sparse_dst = colmap_dir / "sparse" / "0"
        sparse_dst.mkdir(parents=True)
        rename_images_to_stems(recon, sparse_dst)
        shutil.rmtree(mapper_dir)
        logger.info("colmap: %d/%d registered, %d points3D", recon.num_reg_images(), len(names), recon.num_points3D())
        return recon
```

`database_params_path` is Task 2's public helper; the rebuild uses it, never an inline `with_name`.

- [ ] **Step 4: Run** (Step 2's command). Expected: 6 passed.

- [ ] **Step 5: Commit** `feat(pointcloud): ColmapCreator on the shared sfm contract` (files: `colmap.py`, `test_colmap.py`).

Note: `pointcloud/__init__.py` still registers `ColmapCreator` as a `BasePointcloudCreator` until Task 6. `test_registry.py::test_all_creators_are_instantiable` fails from here until Task 6. Tasks 4-6 land together before the gate, and Task 6's commit is where it goes green again. Record this in the task report.

---

### Task 5: `HlocCreator` + `setup/hloc.sh`

**Files:**
- Rewrite: `collab_splats/pointcloud/sfm/hloc.py`, `tests/pointcloud/sfm/test_hloc.py`, `setup/hloc.sh`

- [ ] **Step 1: Rewrite `setup/hloc.sh` (the pin source the test reads)**

```bash
#!/usr/bin/env bash
# hloc (cvg/Hierarchical-Localization) clone for the `hloc` extra
# - editable uv path source (pyproject [tool.uv.sources]); the clone must exist before any resolve
# - --recursive: hloc's superpoint/superglue modules sys.path-append third_party/ submodules
# - re-pinned every run, as setup/loger.sh does; installing is uv's job, never pip's
set -euo pipefail

HLOC_DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/hloc"
COMMIT="c13273bd0ecc2917a35910fd843712a1c6243193"

echo "==> hloc @ ${COMMIT:0:7}"
if [ ! -d "$HLOC_DIR/.git" ]; then
    git clone --recursive https://github.com/cvg/Hierarchical-Localization "$HLOC_DIR"
fi
git -C "$HLOC_DIR" fetch --all --quiet
git -C "$HLOC_DIR" checkout --quiet "$COMMIT"
git -C "$HLOC_DIR" submodule update --init --recursive --quiet

# Weight prefetch: netvlad + LightGlue download on first use otherwise, mid-run
# - best-effort: a deps-only Docker pass has no venv yet; run again after the sync
PYTHON=/opt/venv/reconstruction/bin/python
if [ "${1:-}" = "--prefetch" ]; then
    "$PYTHON" - <<'EOF' || echo "WARN: hloc weight prefetch failed — weights will download on first use."
from hloc import extract_features, match_features
from hloc.utils.base_model import dynamic_load
from hloc import extractors, matchers

for conf, pkg in ((extract_features.confs["netvlad"], extractors), (extract_features.confs["superpoint_max"], extractors),
                  (match_features.confs["superpoint+lightglue"], matchers)):
    dynamic_load(pkg, conf["model"]["name"])(conf["model"])
print("hloc weights cached")
EOF
fi
echo "==> hloc done"
```

- Before relying on the prefetch snippet, check `hloc/utils/base_model.py` for `dynamic_load` at the pin (`git -C third_party/hloc show c13273b:hloc/utils/base_model.py | grep -n dynamic_load`). The snippet cannot run yet, because hloc is not installed.
- Leave the prefetch unexecuted, and state that in the report.

- [ ] **Step 2: Failing tests (replace `tests/pointcloud/sfm/test_hloc.py`)**

```python
"""
HlocCreator with a fake `hloc` package: pairs, confs, reconstruction contract, pin.
"""

import re
import sys
import types
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm import hloc as hloc_mod
from collab_splats.pointcloud.sfm.hloc import HLOC_PIN, HlocCreator, sequential_pairs, union_pairs
from collab_splats.preproc import frames as fr

REPO = Path(__file__).parents[3]
NAMES = ["frame_000000.png", "frame_000009.png", "frame_000030.png", "frame_000057.png"]


def test_sequential_pairs_link_each_frame_to_the_next_n():
    assert sequential_pairs(["a", "b", "c", "d"], 2) == [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d")]


def test_union_pairs_drops_duplicates_in_either_orientation():
    assert union_pairs([("a", "b"), ("b", "c")], [("b", "a"), ("c", "d")]) == [("a", "b"), ("b", "c"), ("c", "d")]


def test_hloc_pin_matches_setup_script():
    pin = re.search(r'COMMIT="([0-9a-f]+)"', (REPO / "setup" / "hloc.sh").read_text()).group(1)
    assert HLOC_PIN == pin


def _recon(names):
    """
    One PINHOLE camera, one image per name, one point3D observed in every image.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


@pytest.fixture
def fake_hloc(monkeypatch):
    """
    Install a fake `hloc` package in sys.modules; returns the recorded calls.
    """
    calls = {"extract": [], "retrieval": [], "exhaustive": [], "match": [], "recon": []}
    confs = {k: {"output": f"out-{k}"} for k in ("netvlad", "superpoint_max", "superpoint+lightglue", "sift")}

    def extract(conf, image_dir, export_dir, image_list=None):
        calls["extract"].append((conf["output"], image_list))
        path = Path(export_dir) / f"{conf['output']}.h5"
        path.touch()
        return path

    def retrieval(descriptors, output, num_matched):
        calls["retrieval"].append(num_matched)
        Path(output).write_text("frame_000000.png frame_000057.png\n")

    def exhaustive(output, image_list=None):
        calls["exhaustive"].append(image_list)
        Path(output).write_text("")

    def match(conf, pairs, features, export_dir):
        calls["match"].append((conf["output"], Path(pairs).read_text()))
        path = Path(export_dir) / "matches.h5"
        path.touch()
        return path

    def recon_main(sfm_dir, image_dir, pairs, features, matches, **kw):
        calls["recon"].append(kw)
        (Path(sfm_dir) / "models" / "0").mkdir(parents=True)
        return _recon(NAMES)

    mods = {
        "hloc": types.ModuleType("hloc"),
        "hloc.extract_features": types.SimpleNamespace(confs=confs, main=extract),
        "hloc.match_features": types.SimpleNamespace(confs=confs, main=match),
        "hloc.pairs_from_retrieval": types.SimpleNamespace(main=retrieval),
        "hloc.pairs_from_exhaustive": types.SimpleNamespace(main=exhaustive),
        "hloc.reconstruction": types.SimpleNamespace(main=recon_main),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            setattr(mods["hloc"], name.split(".")[1], mod)
    return calls


def _scene(tmp_path):
    """
    A scene dir whose images/ holds NAMES as real PNG keyframes.
    """
    images_dir = tmp_path / "images"
    fr.write_frames(
        images_dir,
        [np.zeros((8, 8, 3), np.uint8)] * len(NAMES),
        [{"frame_idx": int(n[6:12]), "blur_score": 1.0} for n in NAMES],
        {"video_path": "x.mp4", "method": "uniform"},
    )
    return tmp_path / "hloc_backend", images_dir


def test_reconstruct_writes_sparse_0_with_stems(tmp_path, fake_hloc):
    data_dir, images_dir = _scene(tmp_path)
    recon = HlocCreator().reconstruct(data_dir, images_dir=images_dir)
    assert sorted(im.name for im in recon.images.values()) == [Path(n).stem for n in NAMES]
    assert pycolmap.Reconstruction(str(data_dir / "colmap" / "sparse" / "0")).num_reg_images() == len(NAMES)


def test_reconstruct_uses_one_simple_radial_camera_and_the_thread_cap(tmp_path, fake_hloc):
    data_dir, images_dir = _scene(tmp_path)
    HlocCreator(num_threads=3).reconstruct(data_dir, images_dir=images_dir)
    kw = fake_hloc["recon"][0]
    assert kw["camera_mode"] == pycolmap.CameraMode.SINGLE
    assert kw["image_options"] == {"camera_model": "SIMPLE_RADIAL"}
    assert kw["mapper_options"] == {"num_threads": 3}
    assert kw["image_list"] == NAMES


def test_sequential_plus_retrieval_matches_the_deduplicated_union(tmp_path, fake_hloc):
    data_dir, images_dir = _scene(tmp_path)
    HlocCreator(overlap=1, num_retrieved=20).reconstruct(data_dir, images_dir=images_dir)
    pairs = fake_hloc["match"][0][1].split("\n")
    assert pairs == [
        "frame_000000.png frame_000009.png",
        "frame_000009.png frame_000030.png",
        "frame_000030.png frame_000057.png",
        "frame_000000.png frame_000057.png",
        "",
    ]
    # topk over 4 images cannot ask for 20 neighbors: clamped to N-1
    assert fake_hloc["retrieval"] == [3]


def test_sequential_pairing_skips_retrieval(tmp_path, fake_hloc):
    data_dir, images_dir = _scene(tmp_path)
    HlocCreator(pairing="sequential").reconstruct(data_dir, images_dir=images_dir)
    assert fake_hloc["retrieval"] == []
    assert [out for out, _ in fake_hloc["extract"]] == ["out-superpoint_max"]


def test_exhaustive_pairing_uses_hloc_generator(tmp_path, fake_hloc):
    data_dir, images_dir = _scene(tmp_path)
    HlocCreator(pairing="exhaustive").reconstruct(data_dir, images_dir=images_dir)
    assert fake_hloc["exhaustive"] == [NAMES]


def test_no_model_raises(tmp_path, fake_hloc, monkeypatch):
    monkeypatch.setattr(sys.modules["hloc.reconstruction"], "main", lambda *a, **k: None)
    data_dir, images_dir = _scene(tmp_path)
    with pytest.raises(RuntimeError, match="no model"):
        HlocCreator().reconstruct(data_dir, images_dir=images_dir)


def test_missing_hloc_names_the_setup_script(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hloc", None)
    data_dir, images_dir = _scene(tmp_path)
    with pytest.raises(ImportError, match="setup/hloc.sh"):
        HlocCreator().reconstruct(data_dir, images_dir=images_dir)
```

- [ ] **Step 3: Run to confirm failure** (`ImportError: cannot import name 'HLOC_PIN'`)

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm/test_hloc.py -q -p no:cacheprovider
```

- [ ] **Step 4: Implement (replace `hloc.py`)**

```python
"""
hloc incremental SfM: learned features + matches, pycolmap incremental mapping.

- `pointcloud: {method: sfm, backend: hloc}`; Reconstructor._run_sfm dispatches here
- hloc is the optional `hloc` extra (editable path source on third_party/hloc, setup/hloc.sh)
- intermediates under <data_dir>/colmap/hloc/; output contract shared with InstantSfMCreator
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from collab_splats.pointcloud.sfm.sift_db import rename_images_to_stems, sfm_image_dir
from collab_splats.preproc.frames import frame_paths

logger = logging.getLogger(__name__)

# The hloc commit setup/hloc.sh pins; hloc.__version__ is "1.5" there and identifies nothing
HLOC_PIN = "c13273bd0ecc2917a35910fd843712a1c6243193"


########################################
# Pair lists
########################################


def sequential_pairs(names: list[str], overlap: int) -> list[tuple[str, str]]:
    """
    Pair each frame with the next `overlap` frames.

    - hloc has no sequential generator; this matches colmap's sequential_matcher with
      quadratic_overlap 0

    Args:
        names: image names in capture order.
        overlap: forward neighbors per frame.

    Returns:
        (earlier, later) pairs, ordered by the earlier frame.
    """
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 : i + 1 + overlap]]


def union_pairs(*pair_lists: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Concatenate pair lists, dropping a pair already seen in either orientation.

    Args:
        pair_lists: pair lists, kept in order.

    Returns:
        The first occurrence of each unordered pair.
    """
    seen, out = set(), []
    for pairs in pair_lists:
        for a, b in pairs:
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                out.append((a, b))
    return out


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    """
    Parse an hloc pairs file.

    Args:
        path: one "name0 name1" pair per line.

    Returns:
        The pairs in file order.
    """
    return [tuple(line.split()) for line in Path(path).read_text().splitlines() if line.strip()]


########################################
# Creator
########################################


@dataclass
class HlocCreator:
    """
    Learned-feature incremental SfM via hloc on a scene directory.

    - one shared SIMPLE_RADIAL camera, refined by the mapper — same freedom as instantsfm
    - h5 features/matches are reused across runs by hloc's own skip (overwrite=False)

    Args:
        pairing: sequential | retrieval | sequential+retrieval | exhaustive.
        overlap: sequential neighbors per frame.
        num_retrieved: global-descriptor neighbors per frame when pairing retrieves.
        retrieval_conf: hloc.extract_features.confs key for global descriptors.
        feature_conf: hloc.extract_features.confs key for local features.
        matcher_conf: hloc.match_features.confs key.
        num_threads: mapper thread cap.
    """

    pairing: str = "sequential+retrieval"
    overlap: int = 10
    num_retrieved: int = 20
    retrieval_conf: str = "netvlad"
    feature_conf: str = "superpoint_max"
    matcher_conf: str = "superpoint+lightglue"
    num_threads: int = 8

    def reconstruct(self, data_dir: Path, images_dir: Path | None = None) -> pycolmap.Reconstruction:
        """
        Extract, pair, match and map the scene's keyframes with hloc.

        Args:
            data_dir: backend working directory; colmap/ lives here.
            images_dir: COLMAP-shaped image directory; defaults to data_dir/images.

        Returns:
            hloc's largest model, image names renamed to stems, also written to colmap/sparse/0.
        """
        # Optional extra, imported late so config load and the other backends never need it
        try:
            from hloc import (
                extract_features,
                match_features,
                pairs_from_exhaustive,
                pairs_from_retrieval,
                reconstruction,
            )
        except ImportError as err:
            raise ImportError(
                "hloc is not installed — run `bash setup/hloc.sh`, then `uv lock` and `bash setup.sh` "
                "(the `hloc` extra is an editable path source on third_party/hloc)"
            ) from err

        data_dir = Path(data_dir)
        image_dir = sfm_image_dir(data_dir / "images" if images_dir is None else images_dir)
        colmap_dir = data_dir / "colmap"
        hloc_dir = colmap_dir / "hloc"
        hloc_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(colmap_dir / "sparse", ignore_errors=True)
        names = [p.name for p in frame_paths(image_dir)]

        # Local features for every keyframe
        feature_conf = extract_features.confs[self.feature_conf]
        features = extract_features.main(feature_conf, image_dir, hloc_dir, image_list=names)

        # Pair list per pairing mode
        # - retrieval top-k clamps to N-1: torch.topk fails when k exceeds the image count
        # - exhaustive delegates to hloc's own generator
        pairs_path = hloc_dir / f"pairs-{self.pairing}.txt"
        if self.pairing == "exhaustive":
            pairs_from_exhaustive.main(pairs_path, image_list=names)
        else:
            pairs = sequential_pairs(names, self.overlap) if "sequential" in self.pairing else []
            if "retrieval" in self.pairing:
                retrieval_conf = extract_features.confs[self.retrieval_conf]
                descriptors = extract_features.main(retrieval_conf, image_dir, hloc_dir, image_list=names)
                retrieval_path = hloc_dir / "pairs-retrieval.txt"
                pairs_from_retrieval.main(
                    descriptors, retrieval_path, num_matched=min(self.num_retrieved, len(names) - 1)
                )
                pairs = union_pairs(pairs, _read_pairs(retrieval_path))
            pairs_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))

        matches = match_features.main(
            match_features.confs[self.matcher_conf], pairs_path, feature_conf["output"], hloc_dir
        )

        # Map from a fresh sfm_dir: stale models/<idx>/ would inflate the component count
        # - hloc keeps the largest model itself and only logs the split; the warning is ours
        # - returns None (logged, not raised) when nothing reconstructs
        sfm_dir = hloc_dir / "sfm"
        shutil.rmtree(sfm_dir, ignore_errors=True)
        recon = reconstruction.main(
            sfm_dir,
            image_dir,
            pairs_path,
            features,
            matches,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_list=names,
            image_options={"camera_model": "SIMPLE_RADIAL"},
            mapper_options={"num_threads": self.num_threads},
        )
        if recon is None:
            raise RuntimeError("hloc produced no model — too little overlap between frames")
        models = [p for p in (sfm_dir / "models").iterdir() if p.is_dir()]
        if len(models) > 1:
            logger.warning("hloc split the scene into %d models — keeping the largest", len(models))

        # Contract layout: stems, sparse/0
        sparse_dst = colmap_dir / "sparse" / "0"
        sparse_dst.mkdir(parents=True)
        rename_images_to_stems(recon, sparse_dst)
        logger.info("hloc: %d/%d registered, %d points3D", recon.num_reg_images(), len(names), recon.num_points3D())
        return recon
```

The fake's `match(conf, pairs, features, export_dir)` must match the real positional order `(conf, pairs, features, export_dir)` at `hloc/match_features.py:155-163` @ c13273b. It does.

- [ ] **Step 5: Run.** Expected: 10 passed.

- [ ] **Step 6: Commit** `feat(pointcloud): HlocCreator on the shared sfm contract; pin hloc c13273b` (files: `hloc.py`, `test_hloc.py`, `setup/hloc.sh`).

---

### Task 6: Registry — `SFM_CREATORS`, feedforward-only `_REGISTRY`

**Files:**
- Modify: `collab_splats/pointcloud/sfm/__init__.py`, `collab_splats/pointcloud/__init__.py`, `collab_splats/pointcloud/base.py` (docstrings)
- Test: `tests/pointcloud/test_registry.py`

- [ ] **Step 1: Failing test edits in `test_registry.py`**

```python
from collab_splats.pointcloud.sfm import SFM_CREATORS, ColmapCreator, HlocCreator, InstantSfMCreator


@pytest.mark.parametrize("name", ["colmap", "hloc"])
def test_sfm_backends_left_the_feedforward_registry(name):
    with pytest.raises(KeyError, match="unknown pointcloud backend"):
        get_creator(name)


def test_sfm_creators_maps_every_sfm_backend():
    assert SFM_CREATORS == {"instantsfm": InstantSfMCreator, "colmap": ColmapCreator, "hloc": HlocCreator}
```

- Delete `test_get_creator_colmap` and `test_get_creator_hloc`.
- `test_all_creators_are_instantiable` loops `("mapanything", "vggtx")`.

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_registry.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement**

- `sfm/__init__.py`:

```python
"""
SfM backends for `pointcloud.method: sfm`: InstantSfM (global), COLMAP and hloc (incremental).

- every creator: reconstruct(data_dir, images_dir) -> pycolmap.Reconstruction
- model at <data_dir>/colmap/sparse/0, image names are filename stems
- SFM_CREATORS is what Reconstructor._run_sfm dispatches on
"""

from .colmap import ColmapCreator
from .hloc import HlocCreator
from .instantsfm import InstantSfMCreator

SFM_CREATORS = {"instantsfm": InstantSfMCreator, "colmap": ColmapCreator, "hloc": HlocCreator}

__all__ = ["SFM_CREATORS", "ColmapCreator", "HlocCreator", "InstantSfMCreator"]
```

- `pointcloud/__init__.py`:
  - Module docstring first bullet becomes `- feedforward backbones (vggtx, mapanything, vggt_omega, loger) resolve through get_creator / make_creator; sfm backends dispatch through sfm.SFM_CREATORS`.
  - Remove `"colmap"`/`"hloc"` from `_REGISTRY`.
  - The `get_creator` docstring loses the "not the sfm allowlist" bullet, and its `name:` Arg becomes `mapanything or vggtx, plus vggt_omega / loger when their optional deps are installed.`
  - Keep `ColmapCreator`/`HlocCreator` imported and in `__all__` (public names); `_REGISTRY`'s type hint stays.
- `base.py`:
  - Line 6 becomes `- BasePointcloudCreator is the one abstract method every feedforward backend implements; sfm creators share a different contract (sfm/__init__.py)`.
  - Line 179 becomes `- concrete feedforward backends resolve through the registry in this package's __init__`.
  - The class docstring's first line becomes `Abstract base every feedforward pointcloud backend implements.`

- [ ] **Step 4: Run** `tests/pointcloud/test_registry.py tests/pointcloud/sfm tests/test_docstring_contract.py`. Expected: all pass.

- [ ] **Step 5: Commit** `refactor(pointcloud): sfm creators dispatch via SFM_CREATORS, not the feedforward registry`

---

### Task 7: `_run_sfm` dispatch, registered subset, provenance attrs

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (imports ~line 37, constants 72-78, `_run_sfm` 1059-1141)
- Test: `tests/wrapper/test_sfm_config.py`, `tests/wrapper/test_sfm_stage.py`

- [ ] **Step 1: Failing tests**

In `test_sfm_config.py`:
- Replace `test_instantsfm_is_the_only_sfm_backend` with:

```python
def test_sfm_backends_are_the_sfm_creators():
    assert _SFM_BACKENDS == {"instantsfm", "colmap", "hloc"}
```

- Replace `test_sfm_rejects_unwired_backends_at_config_load` with:

```python
@pytest.mark.parametrize("backend", ["colmap", "hloc"])
def test_colmap_and_hloc_validate_at_config_load(backend):
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = backend
    Reconstructor.validate_config(cfg)  # must not raise


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
def test_colmap_and_hloc_refuse_ba_and_lc(backend):
    for key in ("bundle_adjustment", "loop_closure"):
        cfg = _base_config()
        cfg["pointcloud"]["method"] = "sfm"
        cfg["pointcloud"]["backend"] = backend
        cfg["pointcloud"][key] = True
        with pytest.raises(ValueError, match=key):
            Reconstructor.validate_config(cfg)


def test_bad_colmap_block_fails_config_load():
    cfg = _base_config()
    cfg["pointcloud"]["method"] = "sfm"
    cfg["pointcloud"]["backend"] = "colmap"
    cfg["pointcloud"]["colmap"]["overlap"] = 0
    with pytest.raises(ValueError, match="pointcloud.colmap.overlap"):
        Reconstructor.validate_config(cfg)
```

- Delete `test_run_sfm_still_guards_non_instantsfm_backends`; the guard goes.
- Update the module docstring to `"""Tests for the sfm config surface: backend allowlist, sub-blocks, BA/LC/refine rejection."""`.

In `test_sfm_stage.py`:
- Extend `_sfm_reconstructor` with a `backend="instantsfm"` kwarg that sets `config["pointcloud"]["backend"] = backend`.
- Extend `_patched_sfm` with:

```python
        "SFM_CREATORS": patch.dict(f"{RECONSTRUCTOR}.SFM_CREATORS", {"colmap": MagicMock(), "hloc": MagicMock()}),
        "colmap_cli_version": patch(f"{RECONSTRUCTOR}.colmap_cli_version", return_value="COLMAP test"),
```

- `patch.dict` yields the dict itself, so the tests read `started["SFM_CREATORS"]["colmap"]`.
- Then add:

```python
def _registered(recon, stems):
    """
    Make the patched creator return a model registering exactly `stems`.
    """
    images = {i: SimpleNamespace(name=s) for i, s in enumerate(stems, start=1)}
    return SimpleNamespace(images=images, reg_image_ids=lambda: list(images))


def _run_backend(tmp_path, backend, stems, *, floor=0.5):
    """
    Run _run_sfm for colmap/hloc with a creator registering `stems`; returns the started mocks.
    """
    recon = _sfm_reconstructor(tmp_path, backend=backend)
    recon.config["pointcloud"][backend]["min_registered_frac"] = floor
    patches = _patched_sfm(recon)
    model = _registered(recon, stems)
    with ExitStack() as stack:
        started = {name: stack.enter_context(p) for name, p in patches.items()}
        started["SFM_CREATORS"][backend].return_value.reconstruct.return_value = model
        started["generate_vda_depth"].return_value = np.arange(len(FRAME_IDX), dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
        recon._run_sfm()
    return started


@pytest.mark.parametrize("backend", ["colmap", "hloc"])
def test_run_sfm_builds_the_creator_from_the_block_minus_the_floor(tmp_path, backend):
    started = _run_backend(tmp_path, backend, [f"frame_{i:06d}" for i in FRAME_IDX])
    kwargs = started["SFM_CREATORS"][backend].call_args.kwargs
    assert "min_registered_frac" not in kwargs
    assert kwargs["pairing"] == "sequential+retrieval"
    started["InstantSfMCreator"].assert_not_called()


def test_run_sfm_subsets_to_registered_frames_in_order(tmp_path):
    kept = [f"frame_{i:06d}" for i in (0, 30, 57)]
    started = _run_backend(tmp_path, "colmap", kept)
    _recon_arg, depths, keyframes, names = started["result_from_reconstruction"].call_args.args
    assert names == [f"{s}.png" for s in kept]
    assert depths[:, 0, 0].tolist() == [0.0, 2.0, 3.0]
    assert keyframes.shape[0] == 3


def test_run_sfm_raises_below_the_registered_floor(tmp_path):
    with pytest.raises(RuntimeError, match="1/4"):
        _run_backend(tmp_path, "colmap", ["frame_000000"], floor=0.5)


def test_run_sfm_stamps_backend_provenance(tmp_path):
    started = _run_backend(tmp_path, "colmap", [f"frame_{i:06d}" for i in (0, 9, 30)])
    attrs = started["result_from_reconstruction"].return_value[0].save_zarr.call_args.kwargs["extra_attrs"]
    assert attrs["backend"] == "colmap"
    assert attrs["colmap_cli_version"] == "COLMAP test"
    assert attrs["pycolmap_version"] == "0.0.0"
    assert (attrs["registered_frames"], attrs["total_frames"]) == (3, 4)


def test_run_sfm_hloc_stamps_the_pin(tmp_path):
    from collab_splats.pointcloud.sfm.hloc import HLOC_PIN

    started = _run_backend(tmp_path, "hloc", [f"frame_{i:06d}" for i in FRAME_IDX])
    attrs = started["result_from_reconstruction"].return_value[0].save_zarr.call_args.kwargs["extra_attrs"]
    assert attrs["hloc_commit"] == HLOC_PIN
    assert "colmap_cli_version" not in attrs


def test_run_sfm_instantsfm_attrs_are_unchanged(tmp_path):
    started = _run_sfm_with_mocks(_sfm_reconstructor(tmp_path))
    attrs = started["result_from_reconstruction"].return_value[0].save_zarr.call_args.kwargs["extra_attrs"]
    assert list(attrs) == ["method", "backend", "instantsfm_version"]
```

- The inline `from ... import HLOC_PIN` in a test violates the imports-at-top rule, so move it to the file's top imports.
- The store image extension comes from `fr.write_frames`. Before writing the `names ==` assertion, check the extension with `ls` on a store (`frame_000000.png` is expected).

- [ ] **Step 2: Run to confirm failure**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && PYTHONUTF8=1 PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_config.py tests/wrapper/test_sfm_stage.py -q -p no:cacheprovider
```

- [ ] **Step 3: Implement in `reconstructor.py`**

- Imports:
  - Replace line 37 with `from collab_splats.pointcloud.sfm import SFM_CREATORS, InstantSfMCreator`.
  - Add `from collab_splats.pointcloud.sfm.hloc import HLOC_PIN`.
  - Extend the Task 3 `sift_db` import to `PAIRINGS as _SFM_PAIRINGS, colmap_cli_version`.
  - Keep isort order.
- Constants (replacing lines 75-78):

```python
# Every sfm creator _run_sfm can dispatch; config load rejects anything else
_SFM_BACKENDS = set(SFM_CREATORS)
```

- Module-level helper, placed in the helpers section above the class. Find the `########` section before `class Reconstructor`; if there is none, add `# Helpers` there:

```python
def _registered_rows(recon: pycolmap.Reconstruction, names: list[str], min_frac: float, backend: str) -> list[int]:
    """
    Rows of `names` whose stems the model registered, in order.

    - incremental mappers may drop frames; the dense result is built on the registered subset

    Args:
        recon: the sfm model, image names already stems.
        names: keyframe filenames in store order.
        min_frac: floor on the registered share; below it the run fails.
        backend: sfm backend name, for messages.

    Returns:
        Indices into `names` of registered frames.
    """
    registered = {recon.images[i].name for i in recon.reg_image_ids()}
    rows = [row for row, name in enumerate(names) if Path(name).stem in registered]
    if len(rows) < min_frac * len(names):
        raise RuntimeError(
            f"{backend} registered {len(rows)}/{len(names)} frames, below min_registered_frac "
            f"{min_frac} — raise overlap or use pairing: exhaustive"
        )
    if len(rows) < len(names):
        logger.warning("%s registered %d/%d frames — continuing on the registered subset", backend, len(rows), len(names))
    return rows
```

- `_run_sfm` edits:
  - Docstring: `SfM pointcloud path: staged keyframes -> VDA metric depth -> the configured sfm mapper.`, and the bullets say "the sfm creator" rather than "InstantSfM"/"InstantSfMCreator".
  - Delete the NotImplementedError block, lines 1071-1076, with its comment.
  - Replace the creator construction (lines 1108-1115) with:

```python
        # Mapper per backend; writes colmap/<db> + colmap/sparse/0 with stem image names
        # - instantsfm: its three knobs, constructed exactly as before
        # - colmap / hloc: the whole block but min_registered_frac, the floor applied below
        if backend == "instantsfm":
            creator = InstantSfMCreator(
                retriangulation=pc_cfg["instantsfm"]["retriangulation"],
                random_seed=pc_cfg["instantsfm"]["random_seed"],
                min_num_view_per_track=pc_cfg["instantsfm"]["min_num_view_per_track"],
            )
        else:
            kwargs = {k: v for k, v in pc_cfg[backend].items() if k != "min_registered_frac"}
            creator = SFM_CREATORS[backend](**kwargs)
        recon = creator.reconstruct(backend_dir, images_dir=images_dir)
```

  - After the keyframe re-read, before `result_from_reconstruction`:

```python
        # Incremental mappers may drop frames: build on the registered subset above the floor
        # - instantsfm stays strict; result_from_reconstruction refuses a partial model
        subset_attrs = {}
        if backend != "instantsfm":
            rows = _registered_rows(recon, names, pc_cfg[backend]["min_registered_frac"], backend)
            subset_attrs = {"registered_frames": len(rows), "total_frames": len(names)}
            names = [names[row] for row in rows]
            depths = depths[rows]
            keyframes = keyframes[rows]
```

  - Replace the `extra_attrs` dict:

```python
        # Provenance per backend; instantsfm's keys and order are unchanged
        # - colmap: SIFT from the CLI, mapping from the wheel — two different COLMAPs
        if backend == "instantsfm":
            version_attrs = {"instantsfm_version": importlib.metadata.version("instantsfm")}
        elif backend == "colmap":
            version_attrs = {
                "pycolmap_version": importlib.metadata.version("pycolmap"),
                "colmap_cli_version": colmap_cli_version(),
            }
        else:
            version_attrs = {"pycolmap_version": importlib.metadata.version("pycolmap"), "hloc_commit": HLOC_PIN}

        zarr_path = backend_dir / "pointcloud.zarr"
        outputs.save_zarr(
            zarr_path,
            extra_attrs={"method": "sfm", "backend": backend, **version_attrs, **subset_attrs, **align_attrs},
        )
```

  - Update the comment above the VDA block, "The scene's images/ is already the COLMAP image layout InstantSfM wants", to "every sfm creator wants".
  - `test_run_sfm_instantsfm_attrs_are_unchanged` asserts exact key order with `align_attrs` stubbed to `{}`.

- [ ] **Step 4: Run** the Step 2 command, plus `tests/wrapper -q -p no:cacheprovider`. Expected: all pass, except baseline-known failures (compare with `baseline_gate.txt`).

- [ ] **Step 5: Commit** `feat(wrapper): dispatch colmap + hloc through _run_sfm with a registered-frame floor`

---

### Task 8: `depth_align` wording + `PUSH_EXCLUDES`

**Files:**
- Modify: `collab_splats/pointcloud/depth_align.py:3,199,208,220-224`, `collab_splats/remote/sources.py:66-77`
- Test: `tests/remote/test_sources.py`, plus any `depth_align` test that matches on `"InstantSfM registered"` (grep `tests/pointcloud/test_depth_align.py`)

- [ ] **Step 1: Failing test in `tests/remote/test_sources.py`**

Place it beside the existing `PUSH_EXCLUDES` tests (~line 545), reusing their anchor-stripping style:

```python
@pytest.mark.parametrize(
    "name",
    (
        "colmap/colmap/colmap.db",
        "colmap/colmap/colmap.db.json",
        "hloc/colmap/hloc/feats-superpoint-n4096-rmax1600.h5",
        "hloc/colmap/hloc/sfm/database.db",
    ),
)
def test_push_excludes_cover_the_colmap_and_hloc_build_artifacts(name):
    """Rebuildable SIFT DBs and hloc caches; anchors stripped as test_push_excludes_raw_feature_maps does."""
    assert any(fnmatch.fnmatchcase(name, p.lstrip("/")) for p in PUSH_EXCLUDES), name


def test_push_excludes_keep_the_colmap_model():
    """The model the dense result was built on must reach processed."""
    assert not any(fnmatch.fnmatchcase("hloc/colmap/sparse/0/images.bin", p.lstrip("/")) for p in PUSH_EXCLUDES)
```

- Existing tests (~line 530) model rclone's leading-slash anchor with `p.lstrip("/")` on root-relative names; same here.
- That file documents each test with a one-line docstring; follow it.

- [ ] **Step 2: Run to confirm failure**

- [ ] **Step 3: Implement**

`PUSH_EXCLUDES`, after the instantsfm entry:

```python
    # ColmapCreator's SIFT database + its params sidecar (pointcloud/sfm/colmap.py). Rebuilt from
    # the staged images.
    "/*/colmap/colmap.db",
    "/*/colmap/colmap.db.json",
    # HlocCreator's h5 features/matches, pair lists and mapper DB (pointcloud/sfm/hloc.py).
    # Rebuildable; the model it produced is colmap/sparse/0, which IS pushed.
    "/*/colmap/hloc/**",
```

`depth_align.py`:
- Module docstring: `Build a FeedforwardResult from an sfm COLMAP model + VDA depth, at the COLMAP world scale.`
- Function summary: `Build a COLMAP-scale FeedforwardResult from an sfm model + VDA depth maps.`
- Arg: `reconstruction: Registered sfm model; its image stems must equal names' stems (callers subset names to the registered frames first).`
- Error: `f"the sfm model registered {len(reconstruction.images)}/{len(stems)} frames — names must be subset to the registered frames first"`
- Lines 30 and 284 stay; they describe InstantSfM's writer patch specifically.
- Update any test `match=` on the old error text.

- [ ] **Step 4: Run** `tests/remote/test_sources.py tests/pointcloud tests/test_docstring_contract.py`. Expected: pass.

- [ ] **Step 5: Commit** `fix(remote): exclude colmap/hloc build artifacts from push; neutral depth_align wording`

---

### Task 9: Dependency wiring — pyproject, setup.sh, Dockerfile

**Files:** `pyproject.toml`, `setup.sh`, `Dockerfile`

No test beyond `bash -n`; nothing here is executed. `uv lock` and the sync are the user's.

- [ ] **Step 1: `pyproject.toml`**

- In `[project.optional-dependencies]`, after `feedforward`:

```toml
# hloc @ c13273b for the `hloc` sfm backend: editable path source on the setup/hloc.sh clone
# - editable, not a git install: hloc's superpoint/superglue sys.path-append ../../third_party
hloc = ["hloc"]
```

- In `[tool.uv.sources]`:

```toml
hloc = { path = "third_party/hloc", editable = true }
```

- hloc's `setup.py` reads `requirements.txt` at build time, so uv can build its metadata. No `[[tool.uv.dependency-metadata]]` entry is needed. Note that in the report.

- [ ] **Step 2: `setup.sh`**

Insert right after the collab-data block (after line 80), before the `uv sync` comment:

```bash
# hloc: editable path source (pyproject [tool.uv.sources]), so the clone must exist before
# any resolve — including the deps-only Docker pass below
bash "$SCRIPT_DIR/setup/hloc.sh"
```

After the full sync (after the vismatch prefetch block):

```bash
bash "$SCRIPT_DIR/setup/hloc.sh" --prefetch
```

Check that `SCRIPT_DIR` is defined above line 80 (`grep -n SCRIPT_DIR setup.sh`).

- [ ] **Step 3: `Dockerfile`**

Line 72 area:

```dockerfile
COPY pyproject.toml uv.lock README.md LICENSE setup.sh /workspace/collab-splats/
COPY setup/hloc.sh /workspace/collab-splats/setup/hloc.sh
```

- The runtime stage already copies the builder's `/workspace/collab-splats` whole (line 113), and that includes `third_party/hloc` cloned by `setup.sh`, so the editable finder's target ships.
- Add one comment line above that COPY: `# - includes third_party/hloc, the hloc extra's editable source`.

- [ ] **Step 4: Syntax check**

```bash
cd /workspace/collab-splats/.worktrees/sfm-backends && bash -n setup.sh && bash -n setup/hloc.sh && /opt/venv/reconstruction/bin/python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['project']['optional-dependencies']['hloc'], d['tool']['uv']['sources']['hloc'])"
```

Expected: `['hloc'] {'path': 'third_party/hloc', 'editable': True}`

- [ ] **Step 5: Commit** `build(hloc): hloc extra as an editable path source; clone before the deps-only sync`. The commit body states that `uv.lock` is NOT regenerated; the user runs `uv lock`.

---

### Task 10: Docs

**Files:**
- Create: `docs/superpowers/decisions/018-sfm-backends.md`
- Modify: `third_party/README.md`, `configs/README.md` (~363, 491-551, 622-624, 649-650), `docs/source/api/pointcloud.rst`, `CLAUDE.md` (architecture line 67 + in-flight entry), `docs/superpowers/specs/2026-09-26-sfm-backends-design.md` (status line)

- [ ] **Step 1: Decision record.** Match the shape of `017-release-cleanup-rules.md` (read it first). Content:
  - **Context:** colmap/hloc creators existed unwired, and instantsfm was the only sfm backend.
  - **Decision:**
    - both wired through `SFM_CREATORS`;
    - incremental only (the system colmap 3.10 has no `global_mapper`, and instantsfm covers global);
    - one shared `pairing` vocabulary;
    - one SIMPLE_RADIAL camera;
    - a registered-subset floor for colmap/hloc only.
  - **Dependency:**
    - hloc is an editable uv path source on `third_party/hloc` @ c13273b, not a git install. SuperPoint/SuperGlue `sys.path`-append `../../third_party` (`hloc/extractors/superpoint.py:8`, `hloc/matchers/superglue.py:6` @ c13273b).
    - vismatch was rejected: not in the venv, and its COLMAP export is mid-rework.
  - **Licenses:**
    - SuperPoint/SuperGlue weights are Magic Leap non-commercial;
    - LightGlue is Apache-2.0;
    - netvlad weights come from the original authors (research);
    - same footing as InstantSfM/VDA.
  - **No GLOMAP:** needs COLMAP ≥ 3.11/4.x CLI; follow-up is a COLMAP 4.x image.
  - **Consequences:**
    - colmap's `sequential+retrieval` ≠ hloc's (periodic loop detection vs per-frame retrieval);
    - features from colmap 3.10 CLI, mapping from pycolmap 4.0.4;
    - `uv.lock` regenerated by the user.
  - **Follow-ups** from the spec.

- [ ] **Step 2: `third_party/README.md`**
  - Line 11 becomes `bash setup/hloc.sh     # hloc clone @ c13273b (also called by setup.sh; --prefetch caches weights)`.
  - The hloc row becomes: `` `hloc/` | `cvg/Hierarchical-Localization` @ `c13273b` (recursive: SuperGlue, d2net, r2d2, deep-image-retrieval) | `setup/hloc.sh` (called by `setup.sh` before the sync) | `hloc` sfm backend (`collab_splats/pointcloud/sfm/hloc.py`); editable uv path source, extra `hloc`. **SuperPoint/SuperGlue weights: Magic Leap non-commercial.** ``
  - The "re-pinned" bullet adds hloc.

- [ ] **Step 3: `configs/README.md`**
  - Row 363: `sfm: instantsfm, colmap or hloc`.
  - Add rows for each `pointcloud.colmap.*` / `pointcloud.hloc.*` key, with type/default/meaning from base.yaml.
  - In the instantsfm section (491-551), add sibling sections `### The colmap backend` and `### The hloc backend`. Each gets:
    - install;
    - output layout (`colmap/colmap.db` + `.json`, or `colmap/hloc/`);
    - pairing table;
    - registered-subset behavior and the attrs.
  - Fix `pointcloud/sfm.py` at 541 → `pointcloud/sfm/instantsfm.py`.
  - Lines 622-624 attrs list adds the colmap/hloc keys.
  - 649-650 excludes paragraph adds `colmap.db(.json)` and `colmap/hloc/`.
  - US spelling: fix `behaviour` at 365 while there.

- [ ] **Step 4: `docs/source/api/pointcloud.rst`.** Read it. If it has an sfm section, add `ColmapCreator`/`HlocCreator`/`SFM_CREATORS` there, in its existing autodoc style. Otherwise add an `SfM backends` section with `automodule` entries for `collab_splats.pointcloud.sfm`, `.colmap`, `.hloc` and `.instantsfm` in the style of the page's existing entries.

- [ ] **Step 5: `CLAUDE.md`**
  - Architecture comment line 67: `#   + ColmapCreator / HlocCreator (incremental; SFM_CREATORS dispatch; sift_db.py shared SIFT)`.
  - Add an in-flight bullet:
    `- **sfm-backends** — wire ColmapCreator + HlocCreator into pointcloud.method sfm on feat/sfm-backends; hloc integration waits on the user's uv lock + sync ([spec](docs/superpowers/specs/2026-09-26-sfm-backends-design.md) · [plan](docs/superpowers/plans/2026-09-26-sfm-backends.md))`
  - The PreToolUse guard hook checks size; keep it short.

- [ ] **Step 6: Spec status line.** `approved design, revised 2026-09-26; implementation on feat/sfm-backends (plan 2026-09-26-sfm-backends.md)`.

- [ ] **Step 7: Commit** `docs(decisions): 018 sfm backends; configs/third_party/api docs for colmap + hloc` (`git add -f` for `docs/superpowers/**`).

---

### Task 11: Gate

- [ ] **Step 1: Proof line + scoped gate** (Task 0 Step 3's command, output to `$SCRATCH/tip_gate.txt`).
- [ ] **Step 2: Diff against the baseline**
  - Every FAILED/ERROR at tip must also be in `baseline_gate.txt`.
  - Explain the passed-count delta test by test: tests added in Tasks 1-8, minus the deleted ones.
    - Deleted in Task 6: `test_get_creator_colmap`, `test_get_creator_hloc`.
    - Deleted in Task 7: `test_run_sfm_still_guards_non_instantsfm_backends`.
    - Changed: the parametrized unwired-backend test (2 cases) became the acceptance test (2 cases).
    - The rewritten `test_colmap.py`/`test_hloc.py` replace the old counts. Record old vs new per file.
  - The SKIP count is unchanged.
- [ ] **Step 3: `black --check` + `isort --check`** on every file touched on the branch (`git diff --name-only clean/final...HEAD -- '*.py'`).
- [ ] **Step 4:** `graphify update .` in the worktree (AST-only).

---

### Task 12: instantsfm byte-identical at tip

- [ ] Run the tip on a copy of run A's `images/` + `frames.json` into `/workspace/outputs/sfm-backends/tip/<scene>/`.
  - Use `--stages pointcloud` with `insfm_seed.yaml`, in tmux, from the worktree with `PYTHONPATH`.
  - Then `compare_sfm.py base-A/<scene>/instantsfm tip/<scene>/instantsfm`.
- [ ] The gate chosen in Task 0 must hold:
  - full identity if A == B;
  - otherwise the argv snapshot test plus the DB hash.
- [ ] Report which gate applied, and why.

---

### Task 13: Integration runs

tmux, nothing else running, the same keyframes as run A (copy `images/` + `frames.json`), `semantics: {enabled: false}`.

- [ ] **colmap:**
  - overrides `pointcloud: {method: sfm, backend: colmap}`, `--stages pointcloud,splats` (`splats` is in `_STAGE_ORDER`), into `/workspace/outputs/sfm-backends/colmap/`;
  - instantsfm needs the same `pointcloud,splats` run for the PSNR comparison (the splats budget from base.yaml, identical for all).
- [ ] **Report per backend:**
  - registered N/M (`pointcloud.zarr` attrs, or `num_reg_images`);
  - point count;
  - mean reprojection error (`pycolmap.Reconstruction(...).compute_mean_reprojection_error()`);
  - pointcloud-stage runtime (log timestamps);
  - splat PSNR (splats stage log/metrics).
- [ ] **Tolerance check:** if colmap registered fewer than M, confirm splats ran without error on the subset (`images/` holds frames the model omits).
- [ ] **hloc:** BLOCKED until the user runs `uv lock` + `bash setup.sh`.
  - Stop and ask.
  - After the sync: `bash setup/hloc.sh --prefetch`, then the same run with `backend: hloc`.
- [ ] Append results to decision 018 under `## Measured` and commit.

---

## Stop points for the user

1. After Task 9: the user runs `uv lock` (and `bash setup.sh`) to install the `hloc` extra. The hloc integration waits on this; everything else proceeds.
2. Docker rebuild (Mac-only): the user's.
3. No merge, no push: the branch stays local on `feat/sfm-backends`.
