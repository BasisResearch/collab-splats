# Pointcloud Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One implementation per operation in `collab_splats/pointcloud/`: one `PointcloudResult` every creator returns, COLMAP as interop only, every geometric operation in `geometry/`, one sfm skeleton.

**Architecture:**
- Refactors land first; each keeps outputs equal and is net-negative in lines.
- Parity changes P1–P8 (P5 dropped, see Amendments) land last, one commit each, measured against the refactored baseline.
- New shared modules: `utils/colmap.py`, `geometry/projection.py`, `pointcloud/depth.py`, `pointcloud/sfm/base.py`, `semantics/lifting.py`.
- Each new module lands in the same commit that deletes the copies it replaces.

**Tech Stack:** Python 3.11, numpy, torch, pycolmap 4.x, zarr 3, pytest, uv. Venv `/opt/venv/reconstruction`.

**Spec:** `docs/superpowers/specs/2026-09-27-pointcloud-reorg-design.md` (commit `f77e1126`).

---

## Amendments to the spec (found while planning; confirmed by the user 2026-09-27)

| # | Spec says | Plan does | Why |
|---|---|---|---|
| A1 | `depth_residual(...) -> (residual, in_frame)` | `-> (residual, depth, in_frame)`, where `depth` is the projected z in camera j | multiview needs z for its `abs + rel·z` tolerance and for `collect`'s `median_depth` |
| A2 | `unproject(depth, w2c, K)` | `unproject(depth, w2c, K, *, pixel_offset=0.0)` | pgsr samples at pixel centers `(u+0.5, v+0.5)`; the pipeline and vggt use integer pixels `(u, v)` — **VOID 2026-09-27: PGSR removed (decision 019); `pixel_offset` and public `pixel_rays` dropped** |
| A3 | P5: VGGT-X box y-scale | dropped | `clean/consistency` b3496862 already fixed it (`new_h/orig_h`) |
| A4 | `center_crop_box(orig_wh, resized_wh, crop_wh)` | adds `*, scale=None` | consistency's `_mapanything_crop_coords` keeps a uniform scale on purpose |
| A5 | `subsample_points` refactor | lands only as P3 | the array form and `_limit_trues` draw in different index spaces, so any collapse changes the LC sample |
| A6 | `depth_residual` as a refactor | lands only as P4 | multiview's inlier test `|z−s|<tol` becomes `|r|·z<tol`, which can flip boundary pixels at ulp level |
| A7 | zarr attrs `depth_scale*` | deleted in Task 9 | the only reader is the attr gate, which becomes a field-presence check |

---

## Conventions for every task

```bash
WT=/workspace/collab-splats/.worktrees/pointcloud-release
SP=/workspace/scratch/pc-release
PY=/opt/venv/reconstruction/bin/python
```

**Worktree trap.** The venv's editable finder points at the MAIN tree.

- Every Python command runs as `cd $WT && PYTHONPATH=$WT $PY ...`.
- Every test run first prints the proof line:

  ```bash
  cd $WT && PYTHONPATH=$WT $PY -c "import collab_splats;print('PROOF', collab_splats.__file__)"
  ```

  Expected: `PROOF /workspace/collab-splats/.worktrees/pointcloud-release/collab_splats/__init__.py`.
- A path under `/workspace/collab-splats/collab_splats/` means the run is void.

**GATE** (run after every commit, never skipped):

```bash
bash $SP/gprime.sh $WT $SP/reorg_T<N>.log      # expect "IDS IDENTICAL to baseline"
cd $WT && PYTHONPATH=$WT $PY $SP/parity.py --check > $SP/reorg_T<N>_parity.log 2>&1; tail -3 $SP/reorg_T<N>_parity.log   # expect PASS
```

- For a parity commit (Tasks P1–P8), `parity.py --check` is EXPECTED to fail on the named cases.
  - Record the diff in the task's measurement file.
  - Re-save only for those cases: `parity.py --save --cases <case>`.
  - Re-save only after the user approves the measurement.
- New test ids added by a task are appended to nothing: `baseline_a29.txt` lists FAILURES only.

**Net-negative check** (every refactor commit):

```bash
cd $WT && git diff --cached --numstat -- collab_splats evals | awk '{a+=$1;d+=$2} END {print "added",a,"deleted",d; exit (a>=d)}'
```

- Exit 0 is required.
- Tests and docs are excluded from the count.
- If it fails, a duplicate was left behind: find and delete it, never pad.

**Commit.** Shared index, so commit by path only:

```bash
cd $WT && git add <paths> && git commit --only <paths> -m "<msg>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- Never `--amend`, `rebase`, `reset` or `stash` while any agent holds dirty state.
- Check `ls .git/worktrees/pointcloud-release/sequencer 2>/dev/null` before committing.

**Code style** (enforced by `tests/test_docstring_contract.py` for `pointcloud`, `geometry`, `semantics`):

- Imports at the top of the module.
- Docstring: `"""` on its own line, a one-line summary, `- ` bullets, then `Args:`/`Returns:`.
- Every parameter and return is annotated.
- `#` runs of 3+ lines are a header line, then `- ` bullets.
- `########` section dividers.

**Disclosures.** Anything surprising goes to `$SP/final_report_notes.md` under a `## reorg T<N>` heading: skipped gates, ulp drift, assumptions.

**Notebooks are off-limits.** Every notebook break goes to `$SP/tutorial_breaks.md` (handed to tutorial-rework).

**Anchors.** Line numbers below are from `4520452a` and WILL shift after Task 0's rebase.

- Locate code by symbol name, never by line number.
- `$SP/reorg_anchors.md` (written in Task 0) has the post-rebase numbers.

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `collab_splats/utils/colmap.py` | create | the one `sparse/0` spelling; completeness check; read; atomic write with stem names |
| `collab_splats/geometry/projection.py` | create | torch `pixel_rays`, `unproject`, `project`, `depth_residual` |
| `collab_splats/geometry/transforms.py` | modify | + `intrinsics_to_original` (Task 3), `center_crop_box` (Task 4); `rescale_intrinsics` / `shift_intrinsics` came from consistency |
| `collab_splats/pointcloud/base.py` | rewrite | `PointcloudResult` (ex-`FeedforwardResult`) + `BasePointcloudCreator.create` |
| `collab_splats/pointcloud/depth.py` | create | `load_vda_model`, `estimate_depth` (replaces `vda.py`) |
| `collab_splats/pointcloud/utils.py` | modify | `clean_pointcloud`, `confidence_mask`, `subsample_points(mask, …)`; `lift_features` and `reproject_pixels` leave |
| `collab_splats/pointcloud/feedforward/base.py` | shrink | template method + multiview confidence only |
| `collab_splats/pointcloud/feedforward/{vggtx,vggt_omega,mapanything,loger}.py` | shrink | model load/forward + ported sizing lines |
| `collab_splats/pointcloud/sfm/base.py` | create | `BaseSfmCreator` (create/densify/sift_database), vocab tree fetch |
| `collab_splats/pointcloud/sfm/{colmap,hloc,instantsfm}.py` | shrink | `reconstruct` only |
| `collab_splats/pointcloud/sfm/{common,sift_db}.py`, `pointcloud/{vda,depth_align}.py` | delete | absorbed |
| `collab_splats/semantics/lifting.py` | create | `lift_features` (moved) |
| `collab_splats/splats/pgsr.py` | modify | imports `pixel_rays`/`unproject`/`project` from `geometry/projection` |
| `collab_splats/geometry/loop_closure/*` | modify | + `cross_frame_attention_ratio`, `mean_top_quarter`; callers of renamed fields |
| `collab_splats/wrapper/reconstructor.py` | modify | `_run_sfm` → `create()`; one PLY write; field-presence gates; stored `camera_model` |
| `evals/scripts/*.py`, `collab_splats/dashboard/*` | modify | renamed fields/types only |
| `tests/utils/test_colmap.py`, `tests/geometry/test_projection.py`, `tests/geometry/test_center_crop_box.py`, `tests/pointcloud/test_depth.py`, `tests/pointcloud/sfm/test_base.py`, `tests/semantics/test_lifting.py` | create | per-module tests |

---

## Task 0: Rebase onto the landed dependencies; re-record baselines

**Files:**
- Create: `$SP/reorg_anchors.md`
- Overwrite: `$SP/baseline_a29.txt`, `$SP/parity_baseline/*`

- [x] **Step 1: Check that both dependencies have landed in `clean/final`** — DONE 2026-09-27: round3 f612e69d + consistency 93cacd75 on clean/final; `rescale_intrinsics`, `shift_intrinsics`, `write_json` present; `scale_intrinsics_to_original` / `intrinsics_to_original` both gone.

```bash
cd $WT && git fetch origin 2>/dev/null; for b in clean/geometry-round3 clean/consistency; do echo "== $b"; git cherry clean/final $b | grep -c '^+'; done
```

- Expected: `0` for both, meaning every patch of each branch is in `clean/final`. Use patch-id, never `--is-ancestor`, because a squash records no ancestry.
- If either count is non-zero: STOP. Report "blocked: <branch> has N unlanded patches" and do nothing else.
- Agreed order (2026-09-27): round3 landed; consistency lands on `clean/final` next; this branch rebases once, after that.
- Consistency phase 3's `utils/colmap.py` is superseded by Task 1. If phase 3 landed anyway, Task 1 extends that module instead of creating it.
- Also confirm that consistency phase 1 is in (phases 2–3 must NOT be required):

```bash
cd $WT && git show clean/final:collab_splats/geometry/transforms.py | grep -n "def rescale_intrinsics\|def intrinsics_to_original\|def scale_intrinsics_to_original"
cd $WT && git show clean/final:collab_splats/utils/io.py | grep -n "def write_json"
```

- Expected: `rescale_intrinsics` and `write_json` are present.
- Record which of `intrinsics_to_original` / `scale_intrinsics_to_original` still exist. Task 3 deletes whichever remain.

- [x] **Step 2: Rebase (no agent may hold dirty state; confirm with the user first)** — DONE 2026-09-27: squashed (35838a00) then rebased → `f5ae80f9`; history in `backup/pointcloud-release-pre-rebase-20260927`; resolutions logged in `final_report_notes.md`.

```bash
cd $WT && git status --porcelain | grep -v '^??' && echo "DIRTY — stop" || git rebase clean/final
```

- Expected: a clean rebase.
- On conflict, resolve toward the `clean/final` side for anything consistency or round3 touched. Log each resolution in `final_report_notes.md`.

- [ ] **Step 3: Re-survey the anchors**

```bash
cd $WT && rtk proxy grep -rn "original_coords\|image_paths\|model_width\|model_height\|FeedforwardResult\|PointcloudResult\|lift_features\|_raw_to_world_points\|build_pycolmap_reconstruction\|write_colmap\|sparse/0\|\"sparse\", \"0\"\|reproject_pixels\|_limit_trues\|subsample_points\|unproject_depth_map_to_point_map\|scale_intrinsics_to_original\|intrinsics_to_original\|depth_scale\|provenance\|_crop_box\|_crop_coords\|full_frame_coords\|_compute_target_size" collab_splats evals tests > $SP/reorg_anchors.md; wc -l $SP/reorg_anchors.md
```

- [ ] **Step 4: Re-record G′ and parity at the rebased tip**

```bash
cd $WT && bash $SP/gprime.sh $WT $SP/reorg_T0.log; cp $SP/reorg_T0.log $SP/baseline_a29.txt   # file name kept: gprime.sh reads it
cd $WT && PYTHONPATH=$WT $PY $SP/parity.py --save > $SP/reorg_T0_parity_save.log 2>&1
cd $WT && PYTHONPATH=$WT $PY $SP/parity.py --check | tail -3            # PASS
cd $WT && PARITY_MUTATE=1 PYTHONPATH=$WT $PY $SP/parity.py --check | tail -3   # must FAIL (sanity)
```

- Expected: `--check` passes and the mutated run fails.
- If the mutated run passes, parity is vacuous: STOP.

- [ ] **Step 5: Fixture scene snapshots for the zarr-equality gates**

```bash
cd $WT && PYTHONPATH=$WT $PY - <<'EOF'
# Snapshot pointcloud.zarr arrays for one ff and one sfm fixture run
# - uses the wrapper tests' synthetic scenes; paths recorded for later diffs
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/wrapper/test_sfm_stage.py", "tests/wrapper/test_pointcloud_stage.py", "--basetemp", "/workspace/scratch/pc-release/zarr_snap_T0"]))
EOF
ls $SP/zarr_snap_T0
```

- Keep `$SP/zarr_snap_T0`; Tasks 6–9 diff against it with `$SP/zarr_eq.py` (Step 6).
- If `test_pointcloud_stage.py` does not exist, use the wrapper test that runs `_run_feedforward` with a stub creator. Find it with `rtk proxy grep -ln "_run_feedforward" tests/wrapper`.

- [ ] **Step 6: Write the zarr-equality checker `$SP/zarr_eq.py`**

```python
"""
Compare every array under two pointcloud.zarr roots; field renames mapped.
"""

import sys

import numpy as np
import zarr

# Old on-disk key -> new on-disk key (identity until P7 renames keys)
RENAMES = {"original_coords": "crop_box", "image_paths": "image_names"}


def main(old: str, new: str) -> int:
    a, b = zarr.open_group(old, mode="r"), zarr.open_group(new, mode="r")
    bad = 0
    for key in sorted(a.array_keys()):
        nkey = RENAMES.get(key, key) if RENAMES.get(key, key) in b else key
        if nkey not in b:
            print("MISSING", key)
            bad += 1
            continue
        if not np.array_equal(np.asarray(a[key]), np.asarray(b[nkey]), equal_nan=True):
            print("DIFF", key)
            bad += 1
    print("EQUAL" if not bad else f"{bad} differ")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 7: No commit** (everything here is scratch). Append a `## reorg T0` block to `final_report_notes.md` with the rebased tip SHA and the baseline counts.

---

## Task 1: `utils/colmap.py` — one COLMAP IO module

**Files:**
- Create: `collab_splats/utils/colmap.py`, `tests/utils/test_colmap.py`
- Modify:
  - `collab_splats/pointcloud/feedforward/base.py` (`write_colmap` body)
  - `collab_splats/pointcloud/sfm/{colmap,hloc,instantsfm}.py` (`write_sfm_model` calls)
  - `collab_splats/pointcloud/base.py` (`from_colmap`)
  - `collab_splats/wrapper/reconstructor.py` (done-check on `cameras.bin`)
  - `collab_splats/remote/sources.py`, `evals/scripts/eval.py` (`sparse/0` spellings)
- Delete: `rename_images_to_stems`, `write_sfm_model` from `sfm/common.py`, and their tests in `tests/pointcloud/sfm/test_common.py`

- [ ] **Step 1: Write the failing tests** — `tests/utils/test_colmap.py`

```python
import numpy as np
import pycolmap
import pytest

from collab_splats.utils.colmap import SPARSE_SUBDIR, is_complete, read_colmap, sparse_dir, write_colmap


def _tiny_recon(names: list[str]) -> pycolmap.Reconstruction:
    # Two posed images on one PINHOLE camera plus one 3D point
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    for i, name in enumerate(names, start=1):
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.1 * i, 0.0, 0.0]))
        recon.add_image_with_trivial_frame(pycolmap.Image(name=name, camera_id=1, image_id=i), pose)
    recon.add_point3D(np.array([0.0, 0.0, 2.0]), pycolmap.Track(), np.array([10, 20, 30], np.uint8))
    return recon


def test_sparse_dir_is_the_one_spelling(tmp_path):
    assert SPARSE_SUBDIR == "sparse/0"
    assert sparse_dir(tmp_path) == tmp_path / "sparse" / "0"


def test_write_renames_to_stems_and_is_complete(tmp_path):
    write_colmap(_tiny_recon(["frame_000000.png", "frame_000001.png"]), tmp_path)
    assert is_complete(tmp_path)
    names = sorted(im.name for im in read_colmap(tmp_path).images.values())
    assert names == ["frame_000000", "frame_000001"]


def test_is_complete_needs_all_three_files(tmp_path):
    write_colmap(_tiny_recon(["a.png"]), tmp_path)
    (sparse_dir(tmp_path) / "points3D.bin").unlink()
    assert not is_complete(tmp_path)


def test_write_replaces_atomically_and_leaves_no_temp(tmp_path):
    write_colmap(_tiny_recon(["a.png"]), tmp_path)
    write_colmap(_tiny_recon(["a.png", "b.png"]), tmp_path)
    assert len(read_colmap(tmp_path).images) == 2
    assert sorted(p.name for p in (tmp_path / "sparse").iterdir()) == ["0"]


def test_read_missing_model_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="sparse/0"):
        read_colmap(tmp_path)
```

- [ ] **Step 2: Run the tests; they should fail**

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/utils/test_colmap.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'collab_splats.utils.colmap'`

- [ ] **Step 3: Implement `collab_splats/utils/colmap.py`**

```python
"""
COLMAP model IO: the one place the sparse/0 layout is spelled.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pycolmap

SPARSE_SUBDIR = "sparse/0"
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def sparse_dir(root: Path) -> Path:
    """
    Model directory under a COLMAP root.

    Args:
        root: COLMAP root (the dir holding sparse/).

    Returns:
        root / sparse / 0.
    """
    return Path(root) / SPARSE_SUBDIR


def is_complete(root: Path) -> bool:
    """
    Whether all three binary model files exist.

    Args:
        root: COLMAP root.

    Returns:
        True when cameras.bin, images.bin and points3D.bin are all present.
    """
    return all((sparse_dir(root) / f).is_file() for f in _MODEL_FILES)


def read_colmap(root: Path) -> pycolmap.Reconstruction:
    """
    Binary model under root/sparse/0.

    Args:
        root: COLMAP root.

    Returns:
        The loaded reconstruction.

    Raises:
        FileNotFoundError: the model is missing or incomplete.
    """
    if not is_complete(root):
        raise FileNotFoundError(f"no complete COLMAP model at {sparse_dir(root)}")
    return pycolmap.Reconstruction(str(sparse_dir(root)))


def write_colmap(recon: pycolmap.Reconstruction, root: Path) -> None:
    """
    Write a model to root/sparse/0 with image names reduced to stems, atomically.

    - stems match the pipeline's frame ids (frame_NNNNNN), whatever extension the mapper saw
    - written to a sibling temp dir, then renamed over the old model; a crash never leaves half a model

    Args:
        recon: model to write; its image names are renamed in place.
        root: COLMAP root.
    """
    # Image names -> stems
    for image in recon.images.values():
        image.name = Path(image.name).stem

    # Write beside the target, then swap in
    target = sparse_dir(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".0.", dir=target.parent))
    recon.write_binary(str(tmp))
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
```

- [ ] **Step 4: Run the new tests; they should pass**

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/utils/test_colmap.py -q`
Expected: `5 passed`

- [ ] **Step 5: Replace every copy**

1. `feedforward/base.py::write_colmap`: keep the build + rescale for now (Task 6 moves them). Replace the last four lines (`sparse_dir = ...` through `write_binary`) with `write_colmap(recon, Path(output_dir) / "colmap")`.
   - Rename the ff function to `_export_colmap` to avoid the name clash. Update its callers from `rtk proxy grep -rn "write_colmap(" collab_splats evals`.
   - Import: `from collab_splats.utils.colmap import write_colmap`.
2. `sfm/common.py`: delete `rename_images_to_stems` and `write_sfm_model`.
   - In each backend replace `write_sfm_model(recon, colmap_dir, label, n)` with the inline form below. The info log is kept at the call site, one line, because `write_sfm_model` logged it:

     ```python
     write_colmap(recon, colmap_dir)
     logger.info("%s: registered %d/%d frames, %d points", label, recon.num_reg_images(), n, recon.num_points3D())
     ```

   - `write_sfm_model` re-read the model after writing; callers used the return value. Switch them to the in-memory `recon`, which already has stem names after `write_colmap`.
3. `pointcloud/base.py::PointcloudResult.from_colmap`: `pycolmap.Reconstruction(str(root / "sparse" / "0"))` → `read_colmap(root)`.
4. `wrapper/reconstructor.py`: the done-check `(... / "sparse" / "0" / "cameras.bin").exists()` → `is_complete(<colmap root>)`.
   - Every other `"sparse" / "0"` or `"sparse/0"` in `collab_splats/` and `evals/` → `sparse_dir(root)`.
   - Tests may keep literal paths.
5. Delete the `test_common.py` tests for the two deleted functions (spec item 7). Move any assertion about stem renaming into `tests/utils/test_colmap.py` if it is not already covered.

```bash
cd $WT && rtk proxy grep -rn '"sparse" / "0"\|"sparse/0"\|sparse", "0"' collab_splats evals   # expected: only utils/colmap.py
```

- [ ] **Step 6: Behavior-change test for the done-check** (`is_complete` is stricter than `cameras.bin`-only)

Add to `tests/wrapper/test_sfm_stage.py` (or the file whose test covers the done-check; find it with `rtk proxy grep -ln "cameras.bin" tests/wrapper`):

```python
def test_done_check_rejects_partial_model(tmp_path):
    # A cameras.bin alone is a crashed write, not a finished stage
    sparse = tmp_path / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(b"")
    from collab_splats.utils.colmap import is_complete
    assert not is_complete(tmp_path / "colmap")
```

(The import sits at the module top in the real edit; it is shown inline here only to keep the snippet self-contained.)

- [ ] **Step 7: GATE + net-negative + commit**

```bash
cd $WT && git add collab_splats/utils/colmap.py tests/utils/test_colmap.py <modified paths> && git commit --only <all paths> -m "refactor(pointcloud): utils/colmap.py is the one COLMAP IO module

- sparse_dir / is_complete / read_colmap / write_colmap replace write_sfm_model, rename_images_to_stems and four sparse/0 spellings
- done-check requires all three .bin files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 2: `geometry/projection.py` — move pgsr's projection; delete `reproject_pixels`

**Files:**
- Create: `collab_splats/geometry/projection.py`, `tests/geometry/test_projection.py`
- Modify: `collab_splats/splats/pgsr.py` (delete `pixel_rays`, `unproject`, `project`; import them)
- Modify: `collab_splats/pointcloud/utils.py` (delete `reproject_pixels`)
- Delete tests: the `reproject_pixels` tests (`rtk proxy grep -rln reproject_pixels tests`)

Refactor only: no pipeline caller adopts `unproject` here (that is P1), and none adopts `depth_residual` (that is P4).

- [ ] **Step 1: Write the failing tests** — `tests/geometry/test_projection.py`

```python
import numpy as np
import torch

from collab_splats.geometry.projection import pixel_rays, project, unproject


def _pose(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    # Non-identity w2c and K: an identity fixture cannot tell R from R.T
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = q, rng.normal(size=3)
    K = np.array([[300.0, 0, 31.0], [0, 280.0, 22.0], [0, 0, 1]])
    return torch.from_numpy(w2c), torch.from_numpy(K)


def test_unproject_then_project_round_trips_integer_pixels():
    w2c, K = _pose()
    depth = torch.full((1, 6, 8), 2.5, dtype=torch.float64)
    pts = unproject(depth, w2c[None], K[None])                      # (1, 6, 8, 3)
    px, cam = project(pts.reshape(-1, 3), w2c, K)
    v, u = torch.meshgrid(torch.arange(6.0), torch.arange(8.0), indexing="ij")
    assert torch.allclose(px, torch.stack([u, v], -1).reshape(-1, 2).double(), atol=1e-9)
    assert torch.allclose(cam[:, 2], torch.full((48,), 2.5, dtype=torch.float64))


def test_pixel_offset_half_matches_pgsr_centers():
    _, K = _pose()
    rays = pixel_rays(2, 3, K[None], pixel_offset=0.5)
    assert torch.allclose(rays[0, 0, 0], torch.tensor([(0.5 - 31.0) / 300.0, (0.5 - 22.0) / 280.0, 1.0]).double())


def test_unproject_keeps_input_dtype_and_batch():
    w2c, K = _pose()
    depth = torch.ones((3, 4, 5), dtype=torch.float32)
    out = unproject(depth, w2c.float()[None].expand(3, 4, 4), K.float()[None].expand(3, 3, 3))
    assert out.shape == (3, 4, 5, 3) and out.dtype == torch.float32


def test_project_clamps_depth_behind_camera():
    w2c, K = _pose()
    px, cam = project(torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64), torch.eye(4, dtype=torch.float64), K)
    assert torch.isfinite(px).all() and cam[0, 2] < 0
```

- [ ] **Step 2: Run them; they should fail**

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/geometry/test_projection.py -q`
Expected: `ModuleNotFoundError: No module named 'collab_splats.geometry.projection'`

- [ ] **Step 3: Implement `collab_splats/geometry/projection.py`**

The bodies are pgsr's, generalized over leading batch dims and pixel offset. Keep pgsr's matmul form `(p - t) @ R` so pgsr outputs stay equal.

```python
"""
Pinhole projection in torch: pixel rays, depth unprojection, point projection.

- works in the input dtype; numpy callers pass torch.from_numpy
- poses are w2c OpenCV; K is on the depth map's own pixel grid
"""

from __future__ import annotations

import torch
from torch import Tensor


def pixel_rays(height: int, width: int, intrinsics: Tensor, *, pixel_offset: float = 0.0) -> Tensor:
    """
    Camera-frame ray per pixel with unit z: ((u + o - cx)/fx, (v + o - cy)/fy, 1).

    - pixel_offset 0.0: integer pixel coords (vggt, the pointcloud pipeline)
    - pixel_offset 0.5: pixel centers (splats/pgsr)
    - not normalized: pgsr's plane_depth relies on z == 1

    Args:
        height: grid height in pixels.
        width: grid width in pixels.
        intrinsics: (..., 3, 3) camera matrices.
        pixel_offset: added to every integer pixel coordinate.

    Returns:
        (..., H, W, 3) camera-frame directions.
    """
    # Pixel grid in the intrinsics' dtype and device
    u = torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype) + pixel_offset
    v = torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype) + pixel_offset
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    # Broadcast per-camera K over the grid
    fx = intrinsics[..., 0, 0, None, None]
    fy = intrinsics[..., 1, 1, None, None]
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    ray_x = (grid_u - cx) / fx
    ray_y = (grid_v - cy) / fy
    return torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)


def unproject(depth: Tensor, world_to_cam: Tensor, intrinsics: Tensor, *, pixel_offset: float = 0.0) -> Tensor:
    """
    World point of every depth pixel.

    - x_world = R^T (d * ray - t), written as (p - t) @ R

    Args:
        depth: (..., H, W) z-depth.
        world_to_cam: (..., 4, 4) or (..., 3, 4) w2c.
        intrinsics: (..., 3, 3) camera matrices.
        pixel_offset: see pixel_rays.

    Returns:
        (..., H, W, 3) world points.
    """
    rays = pixel_rays(depth.shape[-2], depth.shape[-1], intrinsics, pixel_offset=pixel_offset)
    points_cam = rays * depth[..., None]

    # Camera -> world
    rotation = world_to_cam[..., None, :3, :3]
    translation = world_to_cam[..., None, None, :3, 3]
    return (points_cam - translation) @ rotation


def project(
    points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor, *, min_depth: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """
    Pixel coordinates of world points in one camera.

    Args:
        points_world: (..., 3) world points.
        world_to_cam: (4, 4) or (3, 4) w2c.
        intrinsics: (3, 3) camera matrix.
        min_depth: perspective-divide floor; a point at or behind the camera divides by it.

    Returns:
        (pixels (..., 2), camera-frame points (..., 3)).
    """
    rotation = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    points_cam = points_world @ rotation.transpose(-1, -2) + translation

    # Clamped divide: unclamped, a mask's 0 * inf poisons the backward
    depth = points_cam[..., 2].clamp(min=min_depth)
    u = points_cam[..., 0] * intrinsics[0, 0] / depth + intrinsics[0, 2]
    v = points_cam[..., 1] * intrinsics[1, 1] / depth + intrinsics[1, 2]
    return torch.stack([u, v], dim=-1), points_cam
```

- [ ] **Step 4: Run the new tests; they should pass**

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/geometry/test_projection.py -q`
Expected: `4 passed`

- [ ] **Step 5: Rewire pgsr; delete its copies and `reproject_pixels`**

In `collab_splats/splats/pgsr.py`:

- Delete `pixel_rays`, `unproject` and `project`.
- Add at the top: `from collab_splats.geometry.projection import pixel_rays, project, unproject`.
- At each call site, keep pgsr's `(1, 4, 4)` inputs and `(H*W, 3)` outputs:

```python
# was: pixel_rays(h, w, K)
pixel_rays(h, w, K, pixel_offset=0.5)
# was: unproject(depth, w2c, K)            depth (1,H,W,1)
unproject(depth[..., 0], w2c, K, pixel_offset=0.5).reshape(-1, 3)
# was: project(points, w2c, K, min_depth=m)  w2c (1,4,4), K (1,3,3)
project(points, w2c[0], K[0], min_depth=m)
```

- `pixel_grid` stays in pgsr: its other callers (`patch_ncc`, `sample_at_pixels`) are not projection.
- Its docstring line "`pixel_rays` shares this grid" becomes "`geometry.projection.pixel_rays(..., pixel_offset=0.5)` builds the same grid".

In `collab_splats/pointcloud/utils.py`, delete `reproject_pixels`. Its only callers are tests; delete those tests too.

```bash
cd $WT && rtk proxy grep -rn "reproject_pixels\|pgsr.unproject\|pgsr.project\|pgsr import.*\(unproject\|project\|pixel_rays\)" collab_splats evals tests   # expected: empty
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/splats -q -k "pgsr" 2>&1 | tail -2    # same pass count as before the edit
```

- [ ] **Step 6: GATE + `tests/splats -k pgsr` unchanged + net-negative + commit**

`refactor(geometry): projection.py owns pixel_rays/unproject/project; pgsr imports; reproject_pixels deleted`

---

## Task 3: `intrinsics_to_original` is the one crop-box K undo

Post-rebase state (clean/final 93cacd75): consistency shipped `rescale_intrinsics(K, src_hw, dst_hw)` and `shift_intrinsics(K, offset_xy)`; `scale_intrinsics_to_original` is gone and `depth_align` already calls `rescale_intrinsics`. What is left is the same four-line box → K undo written three times.

**Files:**
- Modify: `collab_splats/geometry/transforms.py` (+`intrinsics_to_original`)
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`_rescale_reconstruction_to_original_dimensions`), `collab_splats/geometry/metrics.py` (lift branch of `compute_reconstruction_quality`), `evals/scripts/eval_splats.py` (K undo after the size check)
- Test: `tests/geometry/test_transforms.py`

- [ ] **Step 1: Write the failing test**

```python
def test_intrinsics_to_original_equals_rescale_then_shift_on_cropped_boxes():
    # Cropped, non-square, fractional boxes: a full-frame box cannot see the crop offset
    rng = np.random.default_rng(3)
    K = np.tile(np.array([[400.0, 0, 259.0], [0, 410.0, 180.0], [0, 0, 1]]), (4, 1, 1))
    K[:, 0, 0] += rng.normal(size=4)
    box = np.array([[0, 37.5, 1920, 1042.5, 1920, 1080], [120, 0, 1800, 1080, 1920, 1080],
                    [0, 0, 1920, 1080, 1920, 1080], [10.25, 3.5, 1010.25, 753.5, 1024, 768]], np.float64)
    model_hw = (364, 518)
    crop_hw = np.stack([box[:, 3] - box[:, 1], box[:, 2] - box[:, 0]], axis=-1)
    expected = rescale_intrinsics(K, model_hw, crop_hw)
    expected = shift_intrinsics(expected, box[:, :2])
    np.testing.assert_array_equal(intrinsics_to_original(K, box, model_hw), expected)
```

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/geometry/test_transforms.py -q -k intrinsics_to_original`
Expected: FAIL (`ImportError: cannot import name 'intrinsics_to_original'`).

- [ ] **Step 2: Implement in `geometry/transforms.py`, below `shift_intrinsics`**

```python
def intrinsics_to_original(K: np.ndarray, crop_box: np.ndarray, model_hw: tuple[int, int]) -> np.ndarray:
    """
    Undo a crop-then-resize: K from the model grid back to original pixels.

    - resize undone first (model grid -> crop size), then the crop's top-left shift
    - crop_box rows are [tl_x, tl_y, br_x, br_y, orig_w, orig_h]; only the first four are read

    Args:
        K: (N, 3, 3) intrinsics on the model grid.
        crop_box: (N, 6) crop box per frame, original pixels.
        model_hw: model grid (height, width).

    Returns:
        (N, 3, 3) float64 K in original pixels.
    """
    # Crop box per frame and its (H, W) size, in original pixels
    box = np.asarray(crop_box, dtype=np.float64)[:, :4]
    crop_hw = np.stack([box[:, 3] - box[:, 1], box[:, 2] - box[:, 0]], axis=-1)

    # Undo the resize (model grid -> crop size), then the crop's top-left offset
    K_crop = rescale_intrinsics(K, model_hw, crop_hw)
    return shift_intrinsics(K_crop, box[:, :2])
```

Run the Step 1 test. Expected: PASS.

- [ ] **Step 3: Swap the three call sites**

- `feedforward/base.py::_rescale_reconstruction_to_original_dimensions`: the per-image `crop_hw` + `rescale_intrinsics` + `shift_intrinsics` lines become `K = pycamera.calibration_matrix()` then `K = intrinsics_to_original(K[None], box[None], model_hw)[0]`.
- `geometry/metrics.py`: the `box` / `crop_hw` / `lifted_K` block (two comments, three statements) becomes `lifted_K = intrinsics_to_original(intrinsics, original_coords, (model_h, model_w))`. Keep `box = original_coords[:, :4]` only if `upsample_depths` still needs it (it takes `original_coords[:, :4]` directly today).
- `evals/scripts/eval_splats.py`: same, `intrinsics = intrinsics_to_original(result.intrinsics, result.original_coords, model_hw)`, then `intrinsics = intrinsics.astype(np.float32)` on its own line.
- Drop `rescale_intrinsics` / `shift_intrinsics` imports that become unused.
- One call per line (feedback: no nested calls): never `f(g(x))`; the `.astype(np.float32)` cast in eval_splats goes on its own line.

- [ ] **Step 4: Equality check**

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/geometry/test_transforms.py tests/geometry/test_metrics.py tests/pointcloud/test_feedforward_intrinsics.py tests/evals -q`
Expected: pass with no assertion loosened. The composition is the same float64 arithmetic in the same order, so outputs are bit-identical; any diff is a bug.

- [ ] **Step 5: GATE + net-negative + commit**

`refactor(geometry): intrinsics_to_original is the one crop-box K undo`

---

## Task 4: `center_crop_box` replaces the four crop-box functions

**Files:**
- Modify: `collab_splats/geometry/transforms.py` (+`center_crop_box`)
- Modify: `feedforward/vggtx.py` (`_compute_vggtx_crop_coords`), `feedforward/vggt_omega.py` (`_compute_omega_original_coords`), `feedforward/mapanything.py` (`_mapanything_crop_coords`, post-consistency name), `feedforward/base.py` + `feedforward/loger.py` (`full_frame_coords`)
- Test: `tests/geometry/test_center_crop_box.py`

- [ ] **Step 1: Snapshot the old outputs over a size grid** (before any edit)

```bash
cd $WT && PYTHONPATH=$WT $PY - <<'EOF'
# Old crop-box outputs over a size grid, for the equality gate
import numpy as np
from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords
from collab_splats.pointcloud.feedforward import vggt_omega, mapanything, base
sizes = [(w, h) for w in (320, 518, 640, 1024, 1280, 1920, 3840) for h in (240, 364, 480, 720, 1080, 2160)]
out = {"sizes": np.array(sizes)}
out["vggtx"] = _compute_vggtx_crop_coords(sizes)
out["full"] = np.stack([base.full_frame_coords(w, h, 1)[0] for w, h in sizes])
# omega / mapanything: call each with its real signature, one size at a time; record the model dims used
np.savez("/workspace/scratch/pc-release/crop_grid_T4.npz", **out)
print({k: v.shape for k, v in out.items()})
EOF
```

- Before running, open `vggt_omega._compute_omega_original_coords` and `mapanything._mapanything_crop_coords`.
- Add their per-size calls to the script with the model dims each creator passes by default (read them from the creator dataclass fields).
- Store the calls as `out["omega"]` and `out["mapanything"]`.

- [ ] **Step 2: Write the failing tests** — `tests/geometry/test_center_crop_box.py`

```python
import numpy as np

from collab_splats.geometry.transforms import center_crop_box


def test_no_crop_is_full_frame():
    np.testing.assert_array_equal(center_crop_box((1920, 1080), (1920, 1080), (1920, 1080)), [0, 0, 1920, 1080, 1920, 1080])


def test_height_crop_per_axis_scale():
    # 1920x1080 -> 518x294 resize, centre crop to 518x280: 7 px off top, in original px 7*1080/294
    box = center_crop_box((1920, 1080), (518, 294), (518, 280))
    np.testing.assert_allclose(box, [0, 7 * 1080 / 294, 1920, 287 * 1080 / 294, 1920, 1080])


def test_uniform_scale_override():
    # mapanything: one scale s for both axes, box offsets divided by s
    s = 0.3
    box = center_crop_box((1000, 800), (300, 240), (280, 224), scale=(s, s))
    np.testing.assert_allclose(box, [10 / s, 8 / s, 290 / s, 232 / s, 1000, 800])


def test_equals_old_functions_on_size_grid():
    snap = np.load("/workspace/scratch/pc-release/crop_grid_T4.npz")
    # Each creator's new call reproduces its old function over every grid size
    from collab_splats.pointcloud.feedforward.vggtx import vggtx_crop_box
    got = np.stack([vggtx_crop_box(w, h) for w, h in snap["sizes"]])
    np.testing.assert_array_equal(got.astype(np.float32), snap["vggtx"])
```

- The last test reads a scratch file. It is a migration gate: keep it until Step 5 passes, then replace `snap[...]` with inline literals for 4 representative sizes per creator, so the committed test has no scratch dependency.
- Add the same assertion for omega, mapanything and full-frame.

- [ ] **Step 3: Implement `center_crop_box` in `geometry/transforms.py`**

```python
def center_crop_box(
    orig_wh: tuple[float, float],
    resized_wh: tuple[float, float],
    crop_wh: tuple[float, float],
    *,
    scale: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Centred crop of a resized frame, as a box in original pixels.

    - resize orig -> resized, then centre-crop resized -> crop; the box inverts both
    - offsets are integer in the resized grid ((resized - crop) // 2), as every upstream loader does
    - scale defaults to per-axis resized / orig; mapanything passes one uniform (s, s)

    Args:
        orig_wh: original frame (width, height).
        resized_wh: frame size after the resize, before the crop.
        crop_wh: model input size after the crop.
        scale: (sx, sy) original -> resized; None = resized / orig per axis.

    Returns:
        (6,) float64 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
    """
    ow, oh = orig_wh
    rw, rh = resized_wh
    cw, ch = crop_wh
    sx, sy = scale if scale is not None else (rw / ow, rh / oh)

    # Integer centre offsets in the resized grid, mapped back to original pixels
    left, top = (rw - cw) // 2, (rh - ch) // 2
    return np.array([left / sx, top / sy, (left + cw) / sx, (top + ch) / sy, ow, oh], dtype=np.float64)
```

- [ ] **Step 4: Replace each old function with the model's sizing lines plus one call**

`vggtx.py`: `_compute_vggtx_crop_coords` becomes `vggtx_crop_box(orig_w, orig_h)`. Keep the upstream citation block verbatim above the sizing lines.

```python
def vggtx_crop_box(orig_w: int, orig_h: int) -> np.ndarray:
    """
    Crop box of upstream VGGT-X crop mode, in original pixels.

    Args:
        orig_w: original frame width.
        orig_h: original frame height.

    Returns:
        (6,) float64 box; see geometry.transforms.center_crop_box.
    """
    # Upstream crop-mode target width, px
    # - Linketic/VGGT-X @ 26d1b95, vggt/utils/load_fn.py:211 (target_size = 518)
    # - crop mode resizes width to it (:238-242) and center-crops a taller height to it (:249-251)
    target = 518
    new_h = round(orig_h * (target / orig_w) / 14) * 14
    return center_crop_box((orig_w, orig_h), (target, new_h), (target, min(new_h, target)))
```

- Do the same for omega (crop-then-resize: `resized = orig`, crop = the aspect-band window), mapanything (`scale=(s, s)`), and full frame (`center_crop_box((w, h), (w, h), (w, h))`).
- `full_frame_coords` is deleted; callers use `np.tile(center_crop_box(...), (n, 1))`.
- The consistency fix b3496862 means per-axis scale IS the VGGT-X behavior, so the default `scale=None` applies.
- The `_preprocess` call sites become `np.stack([vggtx_crop_box(int(f.shape[1]), int(f.shape[0])) for f in frames]).astype(np.float32)`. Keep the float32 cast wherever the old function returned float32.

- [ ] **Step 5: Run the grid gate**

Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/geometry/test_center_crop_box.py tests/pointcloud -q -k "crop or coords or full_frame"`
Expected: all pass.

- If any creator differs on any grid size, STOP: that creator's old function was not a centred crop.
- Record the size and both boxes in `final_report_notes.md`, and keep that creator's function. The user decides.

- [ ] **Step 6: Freeze the grid literals into the test (see Step 2), then GATE + parity + net-negative + commit**

`refactor(geometry): center_crop_box replaces four per-model crop-box functions`

---

## Task 5: Field renames on `FeedforwardResult` (mechanical sweep)

**Files:** every file listed in `$SP/reorg_anchors.md` for these names:
- `original_coords` → `crop_box`
- `image_paths` → `image_names` (type `list[Path]` → `list[str]` of stems)
- `model_width` / `model_height` → `model_hw: tuple[int, int]` as `(H, W)`

The zarr on-disk keys DO NOT change here (P7 changes them). `save_zarr`/`load_zarr` map new attribute names to the old keys.

- [ ] **Step 1: Rename by AST-aware sweep** (not `sed`: `image_paths` also names locals in `localization/`)

```bash
cd $WT && PYTHONPATH=$WT $PY - <<'EOF'
# Rename FeedforwardResult attribute access and constructor kwargs only
# - rope-free: libcst matches Attribute(value=*, attr=name) and Arg(keyword=name) in ff-result contexts
# - prints every touched file for review
EOF
```

- Use `libcst` if it is installed (`$PY -c "import libcst"`). Otherwise edit by hand, file by file, from `reorg_anchors.md`.
- Rules for `image_paths`:
  - rename `.image_paths` / `image_paths=` ONLY where the object is a ff/pointcloud result;
  - `localization/localizer.py` and LC `submap`/`graph` take their own `image_paths` arguments, so check each hit's type;
  - every `[p.name for p in r.image_paths]` becomes `r.image_names`;
  - every `Path(...)` wrapper that was needed only for `.name`/`.stem` goes.
- Rules for `model_width` / `model_height`:
  - `r.model_width` → `r.model_hw[1]`;
  - `r.model_height` → `r.model_hw[0]`;
  - `(r.model_width, r.model_height)` → `r.model_hw[::-1]`;
  - constructor `model_width=w, model_height=h` → `model_hw=(h, w)`.
- `image_names` holds STEMS. Every producer that built `Path(frame_name(idx))` now yields `frame_name(idx)`; check that `frame_name` returns a stem with `rtk proxy grep -n "def frame_name" -A12 collab_splats/preproc/frames.py`.
  - If it includes `.png`, producers use `Path(frame_name(idx)).stem`.
  - Log which form applies.

- [ ] **Step 2: Keep the on-disk keys**

In `FeedforwardResult.save_zarr` / `load_zarr`:

- write `crop_box` under the existing key `original_coords`;
- write `image_names` under the existing key and in the existing format (if paths were stored with an extension, re-append it on save and strip it on load; log the choice);
- write `model_hw` as the existing `model_width`/`model_height` attrs.

- [ ] **Step 3: Proof that only names changed**

```bash
cd $WT && git diff --stat | tail -1
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/pointcloud/test_zarr_attrs.py tests/pointcloud/test_mv_zarr_roundtrip.py tests/pointcloud/feedforward -q
```

- Then run the Task 0 fixture tests into `$SP/zarr_snap_T5` (same command as Task 0 Step 5 with a new `--basetemp`).
- `$PY $SP/zarr_eq.py <T0 zarr> <T5 zarr>` → `EQUAL` for both the ff and sfm fixtures.

- [ ] **Step 4: GATE + parity + commit**

`refactor(pointcloud): rename result fields — crop_box, image_names (stems), model_hw`

Net-negative is not required here (a pure rename); state that in the commit body.

---

## Task 6: One `PointcloudResult`; `to_colmap` / `from_colmap`

**Files:**
- Rewrite: `collab_splats/pointcloud/base.py`
- Modify: `collab_splats/pointcloud/feedforward/base.py` (delete `FeedforwardResult`, `build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`, `_export_colmap`)
- Modify: `collab_splats/pointcloud/__init__.py`, `feedforward/__init__.py`
- Modify (callers of either old type): wrapper, geometry (`metrics`, `bundle_adjustment`, LC), localization, dashboard, `utils/visualization`, evals scripts
- Test: `tests/pointcloud/test_base.py` (rewrite)

- [ ] **Step 1: K-resolution census** (the trap: old `PointcloudResult.intrinsics` was ORIGINAL-res; the new one is MODEL-res)

```bash
cd $WT && rtk proxy grep -rn "PointcloudResult\|\.reproject(\|from_colmap\|_resolve_result\|_load_pointcloud_from_disk" collab_splats evals > $SP/reorg_T6_census.md
```

For every site that reads `.intrinsics` or `.reproject()` from an OLD `PointcloudResult`, decide in the census file:

- (a) The site needs original-res K: add `intrinsics_to_original(r.intrinsics, r.crop_box, r.model_hw)`.
- (b) It uses model-res K with model-res arrays: no change.

Known (a) sites from the pre-rebase survey:

- `_run_tsdf_mesh` (took a PointcloudResult with original K);
- the splats stage camera load;
- `evals/scripts/eval.py` reproject.

Every site gets a row: `path:line — a|b — why`. No site is left undecided.

- [ ] **Step 2: Write the failing tests** — add to `tests/pointcloud/test_base.py` (delete the old pycolmap-wrapper tests)

```python
import numpy as np
import pytest

from collab_splats.geometry.transforms import intrinsics_to_original
from collab_splats.pointcloud.base import PointcloudResult


def _result(camera_model: str = "PINHOLE") -> PointcloudResult:
    # Cropped box + non-identity poses: to_colmap must undo the crop
    n, h, w = 2, 28, 42
    ext = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    ext[1, :3, 3] = [0.3, -0.1, 0.05]
    K = np.tile(np.array([[40.0, 0, 21.0], [0, 41.0, 14.0], [0, 0, 1]], np.float32), (n, 1, 1))
    box = np.tile(np.array([0, 10, 420, 290, 420, 300], np.float32), (n, 1))
    return PointcloudResult(
        extrinsics=ext, intrinsics=K, image_names=["frame_000000", "frame_000001"], crop_box=box,
        model_hw=(h, w), camera_model=camera_model, camera_params=np.zeros((n, 0), np.float32),
        points=np.array([[0, 0, 2.0]], np.float32), colors=np.array([[1, 2, 3]], np.uint8),
        pixel_indices=np.array([[0, 14, 21]], np.int32),
    )


def test_to_colmap_is_original_resolution():
    r = _result()
    recon = r.to_colmap()
    cam = recon.cameras[recon.images[1].camera_id]
    assert (cam.width, cam.height) == (420, 300)
    np.testing.assert_allclose(
        cam.calibration_matrix(), intrinsics_to_original(r.intrinsics[:1], r.crop_box[:1], r.model_hw)[0], rtol=1e-6
    )


def test_to_colmap_from_colmap_round_trip_poses_and_points(tmp_path):
    from collab_splats.utils.colmap import write_colmap
    write_colmap(_result().to_colmap(), tmp_path)
    back = PointcloudResult.from_colmap(tmp_path)
    np.testing.assert_allclose(back.extrinsics, _result().extrinsics, atol=1e-6)
    np.testing.assert_allclose(back.points, _result().points)
    assert back.model_hw == (300, 420) and back.depth is None


def test_camera_model_is_recorded_not_rederived():
    assert _result("SIMPLE_PINHOLE").to_colmap().cameras[1].model.name == "SIMPLE_PINHOLE"


def test_zarr_round_trip(tmp_path):
    r = _result()
    r.save_zarr(tmp_path / "pointcloud.zarr")
    back = PointcloudResult.load_zarr(tmp_path / "pointcloud.zarr")
    assert back.image_names == r.image_names and back.model_hw == r.model_hw and back.camera_model == r.camera_model
    np.testing.assert_array_equal(back.crop_box, r.crop_box)
```

- Imports go at the module top in the real file.
- Run: `cd $WT && PYTHONPATH=$WT $PY -m pytest tests/pointcloud/test_base.py -q`. Expected: FAIL (`TypeError: unexpected keyword 'camera_model'` or an import error).

- [ ] **Step 3: Rewrite `pointcloud/base.py`**

1. Delete the old pycolmap-wrapping `PointcloudResult` (its `extrinsics`/`intrinsics` properties, `write_ply`).
   - Correction 2026-09-27: the old class has NO `reproject`; the earlier `reproject(frame)` via `project` had no caller and is dropped.
   - The one `reproject()` is the moved `FeedforwardResult.reproject()` (after BA: stored pixels + depth -> world points); it moves verbatim, P1 rewrites it on `unproject`.
2. Move the `FeedforwardResult` dataclass from `feedforward/base.py` here and rename it `PointcloudResult`.
   - Move `save_zarr` / `load_zarr` verbatim; they keep the old on-disk keys until P7.
   - Add fields `camera_model: str = "PINHOLE"` and `camera_params: np.ndarray | None = None`.
   - `save_zarr` writes `camera_model` as an attr and `camera_params` as an array when not None.
   - `load_zarr` defaults `camera_model` to `"PINHOLE"` when the attr is absent, because every existing zarr is either ff or sfm-densified at pinhole K.
3. Add the methods below. `to_colmap` merges `build_pycolmap_reconstruction` + `_rescale_reconstruction_to_original_dimensions`.
   - It keeps today's SIMPLE_PINHOLE rule: `mean(fx, fy)` at build, then `max` after rescale. That rule is equivalent to `max(fx, fy)` of the rescaled K only when fx == fy; reproduce it exactly (P6 removes it).

```python
    def to_colmap(self) -> pycolmap.Reconstruction:
        """
        COLMAP model at original resolution; points carry no tracks.

        - K undone through the crop box: intrinsics_to_original
        - one camera per frame, model = self.camera_model

        Returns:
            In-memory reconstruction; utils.colmap.write_colmap writes it.

        Raises:
            ValueError: camera_model is not PINHOLE or SIMPLE_PINHOLE.
        """
        K = intrinsics_to_original(self.intrinsics, self.crop_box, self.model_hw)
        recon = pycolmap.Reconstruction()

        # Points: no feature tracks in a dense-derived set
        for xyz, rgb in zip(self.points, self.colors):
            recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb)

        # One camera + image per frame at original size
        for i, name in enumerate(self.image_names):
            params = _camera_params(self.camera_model, self.intrinsics[i], K[i])
            camera = pycolmap.Camera(
                model=self.camera_model, width=int(self.crop_box[i, 4]), height=int(self.crop_box[i, 5]),
                params=params, camera_id=i + 1,
            )
            recon.add_camera_with_trivial_rig(camera)
            pose = pycolmap.Rigid3d(
                pycolmap.Rotation3d(self.extrinsics[i, :3, :3].astype(np.float64)), self.extrinsics[i, :3, 3].astype(np.float64)
            )
            recon.add_image_with_trivial_frame(pycolmap.Image(name=name, camera_id=i + 1, image_id=i + 1), pose)
        return recon

    @classmethod
    def from_colmap(cls, root: Path) -> "PointcloudResult":
        """
        Sparse fields of a COLMAP model; no dense arrays.

        - K stays at camera resolution: model_hw = camera size, crop_box = full frame
        - images sorted by name, the pipeline's frame order

        Args:
            root: COLMAP root holding sparse/0.

        Returns:
            Result with poses, K, points, colors; depth/world_points/images are None.
        """
        recon = read_colmap(root)
        images = sorted(recon.images.values(), key=lambda im: im.name)
        cams = [recon.cameras[im.camera_id] for im in images]
        pids = sorted(recon.points3D)
        return cls(
            extrinsics=extrinsics_to_homogeneous(np.stack([im.cam_from_world().matrix() for im in images])).astype(np.float32),
            intrinsics=np.stack([c.calibration_matrix() for c in cams]).astype(np.float32),
            image_names=[im.name for im in images],
            crop_box=np.stack([center_crop_box((c.width, c.height), (c.width, c.height), (c.width, c.height)) for c in cams]).astype(np.float32),
            model_hw=(cams[0].height, cams[0].width),
            camera_model=cams[0].model.name,
            camera_params=np.zeros((len(cams), 0), np.float32),
            points=np.array([recon.points3D[p].xyz for p in pids], np.float32).reshape(-1, 3),
            colors=np.array([recon.points3D[p].color for p in pids], np.uint8).reshape(-1, 3),
            pixel_indices=np.zeros((0, 3), np.int32),
        )

```

The module-level helper holds today's two-step rule. It is a copy of the deleted code's arithmetic; P6 deletes the SIMPLE_PINHOLE branch:

```python
def _camera_params(camera_model: str, K_model: np.ndarray, K_orig: np.ndarray) -> list[float]:
    """
    pycolmap params for one camera, reproducing the pre-reorg build-then-rescale rule.

    Args:
        camera_model: PINHOLE or SIMPLE_PINHOLE.
        K_model: model-grid K.
        K_orig: original-resolution K.

    Returns:
        PINHOLE [fx, fy, cx, cy] or SIMPLE_PINHOLE [f, cx, cy].

    Raises:
        ValueError: any other model.
    """
    if camera_model == "PINHOLE":
        return [K_orig[0, 0], K_orig[1, 1], K_orig[0, 2], K_orig[1, 2]]
    if camera_model == "SIMPLE_PINHOLE":
        # Old path: f = mean(fx, fy) on the model grid, then max over the per-axis rescale of that f
        f = (K_model[0, 0] + K_model[1, 1]) / 2.0
        sx, sy = K_orig[0, 0] / K_model[0, 0], K_orig[1, 1] / K_model[1, 1]
        return [max(f * sx, f * sy), K_orig[0, 2], K_orig[1, 2]]
    raise ValueError(f"camera_model must be PINHOLE or SIMPLE_PINHOLE, got {camera_model!r}")
```

- Before relying on this, verify it against the old path. Write a scratch script that builds the same result both ways at the Task 5 tip (`git stash`-free: run the old code from `git show HEAD:...` into `$SP/old_ffbase.py`) and asserts the params are equal for 5 random Ks with fx ≠ fy.
- If they differ, fix `_camera_params` until they are equal; do NOT change the old semantics here.

4. `BasePointcloudCreator`: an abstract `create(self, images_dir: Path, out_dir: Path) -> PointcloudResult`.
   - `BaseFeedforwardCreator` implements it by renaming its current entry point. The wrapper calls `create`.
   - `camera_model` is set on the result by each creator: the ff creator's `camera_model` field, or the sfm camera's model.
5. The ff `write_colmap(result, out_dir, camera_model)` call → `write_colmap(result.to_colmap(), out_dir / "colmap")`.
6. `depth_align.result_from_reconstruction` builds `PointcloudResult(camera_model=<the COLMAP camera model name>, ...)` with `camera_params` holding the non-pinhole params:
   - SIMPLE_RADIAL: `params[3:]`, i.e. `[k1]`;
   - PINHOLE: empty (0 columns).

- [ ] **Step 4: Update every caller from the census**

- The wrapper `_load_pointcloud_from_disk` / `_resolve_result`:
  - stage 2+ loads `PointcloudResult.load_zarr(out/pointcloud.zarr)`;
  - `from_colmap` stays only where a stage reads a COLMAP model that has no zarr (census rows say which).
- Apply every (a) row's `intrinsics_to_original(...)`.
- `rtk proxy grep -rn "FeedforwardResult" collab_splats evals tests` → empty. Update the exports in both `__init__.py`.

- [ ] **Step 5: Run the tests + the zarr-equality gate**

```bash
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/pointcloud tests/wrapper -q 2>&1 | tail -3
```

- Then refresh the fixture snapshot into `$SP/zarr_snap_T6`, and `zarr_eq.py` T0 vs T6 → `EQUAL`.
- New attrs (`camera_model`) are not arrays; `zarr_eq.py` ignores them.
- The exported COLMAP model is equal as well:

```bash
cd $WT && PYTHONPATH=$WT $PY -c "
import pycolmap,sys,numpy as np
a,b=[pycolmap.Reconstruction(p) for p in sys.argv[1:3]]
for i in a.images: assert np.allclose(a.images[i].cam_from_world().matrix(), b.images[i].cam_from_world().matrix()); assert np.allclose(a.cameras[a.images[i].camera_id].params, b.cameras[b.images[i].camera_id].params)
print('COLMAP EQUAL', len(a.images), a.cameras[1].model.name)" <T0 colmap/sparse/0> <T6 colmap/sparse/0>
```

- [ ] **Step 6: GATE + parity + net-negative + commit**

`refactor(pointcloud): one PointcloudResult for every creator; to_colmap/from_colmap replace the pycolmap wrapper`

---

## Task 7: `pointcloud/depth.py` replaces `vda.py`

**Files:**
- Create: `collab_splats/pointcloud/depth.py`, `tests/pointcloud/test_depth.py`
- Delete: `collab_splats/pointcloud/vda.py`, `tests/pointcloud/test_vda.py` (tests moved)
- Modify: callers of `generate_vda_depth` / `vda_depth_complete` / `_vda_npy_dir` (`sfm/instantsfm.py`, `wrapper/reconstructor.py`, evals)

- [ ] **Step 1: Write the failing tests** — `tests/pointcloud/test_depth.py`. The model is stubbed with `monkeypatch`, as `test_vda.py` does today; port its fixtures.

```python
import numpy as np

from collab_splats.pointcloud import depth as depth_mod


def _fake_infer(frames: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, None]:
    # Depth = frame index + 1, constant per frame
    return np.stack([np.full(f.shape[:2], i + 1.0, np.float32) for i, f in enumerate(frames)]), None


def _write_frames(images_dir, names, hw=(36, 64)):
    import cv2
    images_dir.mkdir()
    for n in names:
        cv2.imwrite(str(images_dir / f"{n}.png"), np.zeros((*hw, 3), np.uint8))


def test_estimate_depth_caches_and_reuses(tmp_path, monkeypatch):
    names = ["frame_000000", "frame_000001"]
    _write_frames(tmp_path / "images", names)
    calls = []
    monkeypatch.setattr(depth_mod, "load_vda_model", lambda device: type("M", (), {"infer_video_depth": lambda s, f, *a, **k: (calls.append(1), _fake_infer(f))[1]})())
    d1 = depth_mod.estimate_depth(tmp_path / "images", names, tmp_path)
    d2 = depth_mod.estimate_depth(tmp_path / "images", names, tmp_path)
    assert len(calls) == 1 and sorted(p.name for p in (tmp_path / "depth").iterdir()) == [f"{n}.npy" for n in names]
    np.testing.assert_array_equal(d1, d2)
    assert d1.shape[2] == 518


def test_partial_cache_is_wiped_and_regenerated(tmp_path, monkeypatch):
    names = ["frame_000000", "frame_000001"]
    _write_frames(tmp_path / "images", names)
    (tmp_path / "depth").mkdir()
    np.save(tmp_path / "depth" / "frame_000000.npy", np.zeros((2, 2), np.float32))
    np.save(tmp_path / "depth" / "stale_frame.npy", np.zeros((2, 2), np.float32))
    monkeypatch.setattr(depth_mod, "load_vda_model", lambda device: type("M", (), {"infer_video_depth": lambda s, f, *a, **k: _fake_infer(f)})())
    depth_mod.estimate_depth(tmp_path / "images", names, tmp_path)
    assert sorted(p.name for p in (tmp_path / "depth").iterdir()) == [f"{n}.npy" for n in names]
```

- Port the remaining `test_vda.py` assertions (resize width, nearest interpolation, fp32 flag) onto `estimate_depth`.
- Run it and expect `ModuleNotFoundError`.

- [ ] **Step 2: Implement `collab_splats/pointcloud/depth.py`**

Move `_load_vda_model` → `load_vda_model` and the body of `generate_vda_depth` + `vda_depth_complete` → `estimate_depth`. Keep the arithmetic line for line (resize, `input_size=518`, `fp32`); only the IO shape changes.

```python
"""
Per-frame depth priors from Video-Depth-Anything, cached as .npy per frame.

- depth is metric-model output used up to scale: sfm densify fits a per-frame scale, no shift
- depth, not disparity
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from collab_splats.utils.torch_utils import get_device

# + the Video-Depth-Anything import lines copied unchanged from vda.py's header

logger = logging.getLogger(__name__)

DEPTH_WIDTH = 518
DEPTH_SUBDIR = "depth"


def load_vda_model(device: str) -> torch.nn.Module:
    """
    Metric Video-Depth-Anything ViT-L in eval mode.

    Args:
        device: target device.

    Returns:
        The VDA model on device.
    """
    # body moved verbatim from vda._load_model/_load_vda_model


def estimate_depth(images_dir: Path, names: list[str], out_dir: Path) -> np.ndarray:
    """
    Depth per frame at width 518, from cache or one VDA pass.

    - cache: out_dir/depth/<name>.npy; hit iff the file set equals names exactly
    - miss: wipe the dir, decode images_dir/<name>.png, run VDA once, write every file

    Args:
        images_dir: keyframe store (frame_NNNNNN.png).
        names: frame stems, in pipeline order.
        out_dir: scene output root.

    Returns:
        (N, h, 518) float32 depth, rows in names order.
    """
    cache = Path(out_dir) / DEPTH_SUBDIR
    want = {f"{n}.npy" for n in names}

    # Cache hit: exactly the requested files
    if cache.is_dir() and {p.name for p in cache.iterdir()} == want:
        return np.stack([np.load(cache / f"{n}.npy") for n in names])

    # Miss: wipe, decode, infer, write
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    frames = np.stack([cv2.cvtColor(cv2.imread(str(Path(images_dir) / f"{n}.png")), cv2.COLOR_BGR2RGB) for n in names])
    # the inference + resize lines of generate_vda_depth, verbatim, producing `depths` (N, h, 518)
    for n, d in zip(names, depths):
        np.save(cache / f"{n}.npy", d)
    return depths
```

- The two "verbatim" markers are MOVE instructions, not placeholders. Copy the exact lines from `vda.py` at the Task 6 tip and delete them from `vda.py` in the same commit.
- `get_device()` is the device source, as in `generate_vda_depth` today.
- Match the frame decode to how the wrapper decodes today. If `generate_vda_depth` received in-memory frames from `preproc.frames`, use the same reader (`rtk proxy grep -n "def " collab_splats/preproc/frames.py`) instead of `cv2.imread`, and log the choice.

- [ ] **Step 3: Rewire the callers**

- InstantSfM gets `depth_path = out_dir/depth`; upstream accepts `depth/` (`data_reader.py:40`, spec).
- Delete `vda.py`.

```bash
cd $WT && rtk proxy grep -rn "vda\b\|generate_vda_depth\|vda_depth_complete\|depth_vda" collab_splats evals tests   # expected: empty
```

- [ ] **Step 4: Equality**

- sfm fixture zarr T6 vs T7 → `EQUAL` (the depth values are unchanged; only the cache path moved).
- The first run after this commit regenerates the cache once. Log that in `final_report_notes.md` as a known one-time cost.

- [ ] **Step 5: GATE + parity + net-negative + commit**

`refactor(pointcloud): depth.py — load_vda_model + estimate_depth replace vda.py`

---

## Task 8: `sfm/base.BaseSfmCreator` — skeleton, `densify`, `sift_database`

**Files:**
- Create: `collab_splats/pointcloud/sfm/base.py`, `tests/pointcloud/sfm/test_base.py`
- Delete: `sfm/common.py`, `sfm/sift_db.py`, `pointcloud/depth_align.py`; move their tests into `test_base.py`
- Modify: `sfm/{colmap,hloc,instantsfm}.py` (`reconstruct` only; delete `provenance`), `sfm/__init__.py`, `wrapper/reconstructor.py::_run_sfm`, `evals/scripts/eval.py::_run_instantsfm`

- [ ] **Step 1: Write the failing skeleton test** — `tests/pointcloud/sfm/test_base.py`

```python
import numpy as np
import pycolmap

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.sfm.base import BaseSfmCreator
from collab_splats.utils.colmap import is_complete


class _FakeSfm(BaseSfmCreator):
    # reconstruct returns a canned model; create must write it and densify
    def reconstruct(self, images_dir, colmap_dir, names):
        return _canned_recon(names)


def test_create_writes_colmap_and_returns_densified(tmp_path, monkeypatch):
    names = ["frame_000000", "frame_000001", "frame_000002"]
    monkeypatch.setattr("collab_splats.pointcloud.sfm.base.estimate_depth", lambda d, n, o: _canned_depth(len(n)))
    monkeypatch.setattr("collab_splats.pointcloud.sfm.base.load_frames", lambda d, n: _canned_frames(len(n)))
    result = _FakeSfm().create(tmp_path / "images", tmp_path)
    assert isinstance(result, PointcloudResult) and result.depth is not None
    assert is_complete(tmp_path / "colmap")
    assert result.camera_model == "SIMPLE_RADIAL" and result.camera_params.shape == (3, 1)
```

- Build `_canned_recon`, `_canned_depth` and `_canned_frames` from the existing `tests/pointcloud/test_depth_align.py` fixtures. Move them here; that file is deleted in Step 6.
- The canned model uses a SIMPLE_RADIAL camera with NON-identity poses.
- The existing depth_align tests become `densify` tests unchanged except for the call: `result_from_reconstruction(recon, depths, images, names)` → `_FakeSfm().densify(recon, depths, images)`.
- Their assertions stay as they are: they are the equality gate for the move.

- [ ] **Step 2: Run them; they should fail** (`ModuleNotFoundError: collab_splats.pointcloud.sfm.base`)

- [ ] **Step 3: Implement `sfm/base.py`**

```python
"""
SfM creator skeleton: depth priors, backend mapping, COLMAP write, densify.

- backends implement reconstruct only
- densify aligns depth priors to the mapper's world with a per-frame median scale
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import urllib.request
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from collab_splats.geometry.transforms import center_crop_box, extrinsics_to_homogeneous, rescale_intrinsics
from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.depth import estimate_depth
from collab_splats.preproc.frames import load_frames   # the store reader _run_sfm uses today; keep its real name
from collab_splats.utils.colmap import write_colmap
from collab_splats.utils.torch_utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class BaseSfmCreator(BasePointcloudCreator):
    """
    Template: estimate_depth -> reconstruct -> write_colmap -> densify.

    Attributes:
        min_obs: track observations a frame needs for its own depth scale.
    """

    min_obs: int = 20

    def create(self, images_dir: Path, out_dir: Path) -> PointcloudResult:
        """
        Run the sfm pipeline on a keyframe store.

        Args:
            images_dir: keyframe store.
            out_dir: scene output root; colmap/ and depth/ land under it.

        Returns:
            Densified result at COLMAP scale, registered frames only.
        """
        names = sorted(p.stem for p in Path(images_dir).glob("*.png"))
        depths = estimate_depth(images_dir, names, out_dir)
        recon = self.reconstruct(Path(images_dir), Path(out_dir) / "colmap", names)
        write_colmap(recon, Path(out_dir) / "colmap")

        # Registered subset: depth + frame rows for the frames the mapper kept
        registered = sorted(im.name for im in recon.images.values())
        rows = [names.index(n) for n in registered]
        return self.densify(recon, depths[rows], load_frames(images_dir, registered))

    @abstractmethod
    def reconstruct(self, images_dir: Path, colmap_dir: Path, names: list[str]) -> pycolmap.Reconstruction:
        """
        Largest registered model from the backend's mapper.

        Args:
            images_dir: keyframe store.
            colmap_dir: backend scratch root (databases, mapper output).
            names: frame stems.

        Returns:
            In-memory model; image names may carry extensions (write_colmap strips them).
        """
```

`densify` is `result_from_reconstruction`'s body moved into a method, with three edits:

1. K via `rescale_intrinsics(K_orig, orig_hw, depth_hw)` — already in `depth_align` since the 2026-09-27 rebase.
2. The crop box via `center_crop_box` (full frame), tiled. Task 4 already did this.
3. The return is `PointcloudResult(..., camera_model=cam.model.name, camera_params=<non-pinhole params>)`, and the `attrs` dict is DROPPED (amendment A7).

The helpers `_pixel_indices_from_reconstruction`, `_depth_correspondences` and `_fit_depth_scales` move in as private module functions, unchanged. Their stats go to `logger.info` as today.

`sift_database` is `ensure_sift_database` + `build_sift_database` + `_database_holds` + `fetch_vocab_tree` moved verbatim as module functions (the sidecar JSON stays until P8). `_MATCHERS` / `PAIRINGS` go with them.

- [ ] **Step 4: Backends shrink to `reconstruct`**

For each of `sfm/colmap.py`, `sfm/hloc.py`, `sfm/instantsfm.py`:

- the class inherits `BaseSfmCreator`;
- the old entry point body minus `prepare_sfm_dirs`, `write_sfm_model` and `provenance` becomes `reconstruct(self, images_dir, colmap_dir, names)`;
- `prepare_sfm_dirs` did three things; inline the one each backend needs:
  - `colmap_dir.mkdir(parents=True, exist_ok=True)`;
  - `sfm_image_dir(d)` → `images_dir`;
  - the rmtree of the stale `colmap/sparse` is no longer needed, because `write_colmap` swaps atomically;
- the largest-model pick stays inline in `colmap.py`: `max(maps.values(), key=lambda r: r.num_reg_images())`;
- `hloc.py`: `sequential_pairs` → private `_sequential_pairs`; keep `HLOC_PIN` until Task 11;
- `instantsfm.py`: reads priors from `colmap_dir.parent / "depth"`, written by `create` before `reconstruct`;
- delete every `provenance()`;
- `sfm/__init__.py`: `SFM_CREATORS` maps name → `"module:Class"`, and `get_sfm_creator(name)` importlib-loads it. That lets each backend import its deps at module top (hloc's inline import goes to the top of `hloc.py`):

```python
"""
SfM backends by name; each backend module is imported only when requested.
"""

import importlib

from .base import BaseSfmCreator

SFM_CREATORS = {
    "instantsfm": "collab_splats.pointcloud.sfm.instantsfm:InstantSfMCreator",
    "colmap": "collab_splats.pointcloud.sfm.colmap:ColmapCreator",
    "hloc": "collab_splats.pointcloud.sfm.hloc:HlocCreator",
}


def get_sfm_creator(name: str) -> type[BaseSfmCreator]:
    """
    Backend class by name.

    Args:
        name: key of SFM_CREATORS.

    Returns:
        The creator class.

    Raises:
        KeyError: unknown backend.
    """
    module, cls = SFM_CREATORS[name].split(":")
    return getattr(importlib.import_module(module), cls)
```

- [ ] **Step 5: Callers**

- `Reconstructor._run_sfm`:

  ```python
  result = get_sfm_creator(backend)(**block).create(images_dir, out_dir)
  result.save_zarr(out_dir / "pointcloud.zarr")
  ```

  - `_registered_rows` is deleted, because `create` subsets.
  - The `depth_scale` attr gates in mesh and splats become `if result.depth is None: raise ValueError("pointcloud.zarr has no depth; re-run the pointcloud stage")`.
- `_SFM_BLOCK_KEYS` / `_validate_sfm_block`:
  - move the bounds checks into each creator's `__post_init__` (`ValueError` with the same messages);
  - unknown-key rejection is the dataclass `TypeError`;
  - port the wrapper tests for these messages to `tests/pointcloud/sfm/test_base.py` (or the backend test files).
- `evals/scripts/eval.py::_run_instantsfm` → `InstantSfMCreator(**cfg).create(images_dir, out_dir)`.

- [ ] **Step 6: Delete the absorbed modules; run the gates**

```bash
cd $WT && git rm -q collab_splats/pointcloud/sfm/common.py collab_splats/pointcloud/sfm/sift_db.py collab_splats/pointcloud/depth_align.py tests/pointcloud/test_depth_align.py tests/pointcloud/sfm/test_common.py
cd $WT && rtk proxy grep -rn "sfm.common\|sift_db\|depth_align\|result_from_reconstruction\|provenance\|_registered_rows\|_SFM_BLOCK_KEYS" collab_splats evals tests   # expected: empty
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/pointcloud/sfm tests/wrapper -q 2>&1 | tail -3
```

- sfm fixture zarr T7 vs T8 → `EQUAL`.

- [ ] **Step 7: GATE + parity + net-negative + commit**

`refactor(sfm): BaseSfmCreator owns the skeleton, densify and SIFT DB; common/sift_db/depth_align deleted`

---

## Task 9: InstantSfM in-memory conversion (test first)

**Files:**
- Modify: `collab_splats/pointcloud/sfm/instantsfm.py` (delete `_patch_instantsfm_colmap_write` + the `WriteGlomapReconstruction` → read-back path; add `_to_pycolmap`)
- Test: `tests/pointcloud/sfm/test_instantsfm_convert.py`

- [ ] **Step 1: Capture the golden output of the patched writer** (at the Task 8 tip, before any edit)

- Run the existing InstantSfM fixture test that exercises `SolveGlobalMapper`. Find it with `rtk proxy grep -ln "SolveGlobalMapper\|InstantSfMCreator" tests`.
- Monkeypatch its `reconstruct` to also pickle `(cameras, images, tracks)` right after `SolveGlobalMapper` into `$SP/isfm_golden/state.pkl`.
- Copy the written `sparse/0` model to `$SP/isfm_golden/model`.
- If no test runs the mapper (it needs CUDA + upstream), run the pipeline on a 20-frame chess seq-01 subset in tmux instead, using the same capture. Log which one was used.

- [ ] **Step 2: Write the failing test**

```python
import pickle
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud.sfm.instantsfm import _to_pycolmap

GOLDEN = Path("/workspace/scratch/pc-release/isfm_golden")


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden capture is scratch-only")
def test_in_memory_conversion_equals_patched_writer():
    cameras, images, tracks, image_dir = pickle.loads((GOLDEN / "state.pkl").read_bytes())
    got = _to_pycolmap(cameras, images, tracks, image_dir)
    want = pycolmap.Reconstruction(str(GOLDEN / "model"))
    assert sorted(i.name for i in got.images.values()) == sorted(i.name for i in want.images.values())
    by_name = {i.name: i for i in want.images.values()}
    for im in got.images.values():
        np.testing.assert_allclose(im.cam_from_world().matrix(), by_name[im.name].cam_from_world().matrix(), atol=1e-9)
    assert got.num_points3D() == want.num_points3D()
    got_obs = sorted((got.images[e.image_id].name, e.point2D_idx) for p in got.points3D.values() for e in p.track.elements)
    want_obs = sorted((want.images[e.image_id].name, e.point2D_idx) for p in want.points3D.values() for e in p.track.elements)
    assert got_obs == want_obs
```

- The skip is scratch-only, so this test does not gate CI.
- Also add a committed, CPU-only test on a hand-built 3-image `(cameras, images, tracks)` triple using upstream's own classes. Build it the way upstream's tests or `ReadColmapDatabase` do; read their constructors at `d3e599e`.
- The committed test asserts that the conversion loads in pycolmap and keeps `min_track_length=3`.

- [ ] **Step 3: Implement `_to_pycolmap`**

Mirror upstream `WriteGlomapReconstruction` → `ExportReconstruction` at `d3e599e`:
`FilterRigCompleteness` → largest cluster → `Reconstruction(cameras, images, tracks)` → `filter_by_cluster` → `filter_registered_only` → `build_correspondences(min_track_length=3)`.

Then populate a `pycolmap.Reconstruction` from that upstream object instead of calling `write_binary`.

- Read upstream `Reconstruction.write_binary` at the pin first: `cd /opt/venv/reconstruction/lib/python3.11/site-packages/instantsfm && rtk proxy grep -n "def write_binary\|def build_correspondences\|def filter_\|def extract_colors" -A40 <file>`.
- Field names from the survey:
  - `_selected_indices`;
  - `_point3d_ids[idx]` (−1 = none);
  - `tracks.xyzs`, `tracks.colors`, `tracks.observations[tid]` as `(img_idx, feat_idx)`;
  - `images.world2cams`, `cam_ids`, `filenames`, `features`;
  - `cam.model_id`, `width`, `height`, `params`.
- Confirm each one against the source before use.

```python
def _to_pycolmap(cameras, images, tracks, image_dir: Path) -> pycolmap.Reconstruction:
    """
    Upstream global-mapper state as an in-memory pycolmap model, largest cluster only.

    - mirrors upstream ExportReconstruction @ d3e599e minus the write_binary
    - replaces WriteGlomapReconstruction + read-back, and the colmap-write patch

    Args:
        cameras: upstream camera list.
        images: upstream Images after SolveGlobalMapper.
        tracks: upstream Tracks.
        image_dir: frames, for point colors.

    Returns:
        Registered images of the largest cluster with tracks of length >= 3.
    """
    # Upstream filtering, reused as-is
    FilterRigCompleteness(images)
    cluster = _largest_cluster(images)
    upstream = Reconstruction(cameras, images, tracks)
    upstream.filter_by_cluster(cluster)
    upstream.filter_registered_only()
    upstream.build_correspondences(min_track_length=3)

    # Cameras and posed images with keypoints
    recon = pycolmap.Reconstruction()
    for cam_id, cam in enumerate(cameras):
        recon.add_camera_with_trivial_rig(pycolmap.Camera(
            model=pycolmap.CameraModelId(cam.model_id.value), width=cam.width, height=cam.height,
            params=np.asarray(cam.params, np.float64), camera_id=cam_id + 1))
    for idx in upstream._selected_indices:
        w2c = np.asarray(images.world2cams[idx], np.float64)
        image = pycolmap.Image(name=images.filenames[idx], camera_id=int(images.cam_ids[idx]) + 1, image_id=int(idx) + 1)
        image.points2D = pycolmap.ListPoint2D([pycolmap.Point2D(np.asarray(xy, np.float64)) for xy in images.features[idx]])
        recon.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3]))

    # Points with their tracks; add_point3D links each observation's Point2D
    for tid in upstream.kept_track_ids():   # the real accessor from write_binary
        track = pycolmap.Track()
        for img_idx, feat_idx in tracks.observations[tid]:
            track.add_element(int(img_idx) + 1, int(feat_idx))
        recon.add_point3D(np.asarray(tracks.xyzs[tid], np.float64), track, np.asarray(tracks.colors[tid], np.uint8))
    recon.extract_colors_for_all_images(str(image_dir))
    return recon
```

- `kept_track_ids()` stands for whatever expression `write_binary` uses to enumerate the tracks it writes. Replace it with that exact expression.
- `_largest_cluster` is the cluster rule the current `reconstruct` applies after the write (it checks `cluster_ids`); move that rule here.
- The Step 2 golden test is the arbiter of every one of these details.

- [ ] **Step 4: Run the golden test until it passes; then delete the writer path**

```bash
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/pointcloud/sfm/test_instantsfm_convert.py -q
```

Delete `_patch_instantsfm_colmap_write`, the `WriteGlomapReconstruction` call, the scratch read-back and the cluster checks that moved.

- [ ] **Step 5: GATE + parity + net-negative + commit**

`refactor(sfm): InstantSfM converts mapper state to pycolmap in memory; colmap-write patch deleted`

---

## Task 10: `lift_features` → `semantics/lifting.py`

**Files:**
- Create: `collab_splats/semantics/lifting.py`; move `tests/pointcloud/test_feature_lifting.py` → `tests/semantics/test_lifting.py`
- Modify: `collab_splats/pointcloud/utils.py` (delete `lift_features`), callers (`wrapper/reconstructor.py::_lift_and_save`, `dashboard/viewer`, `dashboard/pipeline`)

- [ ] **Step 1: Move the function body unchanged, then fix its imports and callers**

```bash
cd $WT && rtk proxy grep -rn "lift_features" collab_splats evals tests
```

- [ ] **Step 2: AST-equal proof**

```bash
cd $WT && PYTHONPATH=$WT $PY - <<'EOF'
# The moved function body is AST-identical to the old one
import ast, subprocess
old_src = subprocess.check_output(["git", "show", "HEAD:collab_splats/pointcloud/utils.py"], text=True)
new_src = open("collab_splats/semantics/lifting.py").read()
get = lambda s: next(n for n in ast.walk(ast.parse(s)) if isinstance(n, ast.FunctionDef) and n.name == "lift_features")
print("AST EQUAL" if ast.dump(get(old_src)) == ast.dump(get(new_src)) else "AST DIFFER")
EOF
```

Expected: `AST EQUAL`.

- [ ] **Step 3: Run the moved tests, then GATE (+ `tests/semantics`) + parity + commit**

`refactor(semantics): lift_features moves to semantics/lifting.py`

`tests/semantics` joins this task's gate because the gate script does not cover it:

```bash
cd $WT && PYTHONPATH=$WT $PY -m pytest tests/semantics -q 2>&1 | tail -2
```

---

## Task 11: Feedforward boilerplate collapse + hloc install

**Files:**
- Modify: `feedforward/base.py`, `vggtx.py`, `vggt_omega.py`, `mapanything.py`, `loger.py`
- Modify: `geometry/loop_closure/<consumer file>`; `pyproject.toml`, `uv.lock`, `setup.sh`, `setup/hloc.sh`

Each item below is its own commit, with GATE + parity + net-negative.

- [ ] **Step 1: `extract_intermediate_features` → one base method**

The vggtx / omega / mapanything bodies differ only in:

- (a) the attention module path;
- (b) the decode call.

Plan:

1. Add two per-class hooks to `BaseFeedforwardCreator`:

   ```python
   def _lc_attention(self) -> torch.nn.Module:            # e.g. self.model.aggregator.global_blocks[self._lc_layer_index].attn
   def _lc_decode(self, predictions: dict, hw: tuple[int, int]) -> dict:   # e.g. _decode_depth_head(predictions, hw, pose_encoding_to_extri_intri)
   ```

2. Move one of the three bodies into the base as `extract_intermediate_features`, calling the hooks.
3. Delete the three copies. The existing LC tests for each creator are the gate.
   - If a creator's body differs in anything else (dtype, batch shape), diff the three first with `diff <(sed -n '/def extract_intermediate_features/,/return captured/p' vggtx.py) <(... omega ...)`.
   - Keep that difference as a hook argument, not a branch.

Commit: `refactor(feedforward): one extract_intermediate_features with per-model hooks`

- [ ] **Step 2: Synthetic names + PIL source + mapanything pose inversion**

- `image_names = [frame_name(i) for i in frame_idxs]` is done once in the base template, not in each `_preprocess`.
- `loader_names` + `frames_as_pil_source`:
  - move `frames_as_pil_source` to `collab_splats/preproc/frames.py`, its only domain;
  - the three `loader_names` lines become one base helper `_pil_names(image_names)`.
- mapanything's three c2w→w2c inversions → `geometry.transforms.invert_poses`, which already exists.

Commit: `refactor(feedforward): shared synthetic names, PIL source in preproc, invert_poses for mapanything`

- [ ] **Step 3: LC-only helpers move to their consumer**

`cross_frame_attention_ratio` and `mean_top_quarter` → the one `geometry/loop_closure/*.py` file that calls them (`rtk proxy grep -rn "cross_frame_attention_ratio\|mean_top_quarter" collab_splats`).

Commit: `refactor(geometry): LC attention helpers live with loop closure`

- [ ] **Step 4: hloc via uv git source**

In `pyproject.toml`, replace the editable path source with:

```toml
[tool.uv.sources]
hloc = { git = "https://github.com/cvg/Hierarchical-Localization", rev = "c13273bd" }
```

(Use the full SHA from `HLOC_PIN`.)

```bash
cd $WT && uv lock 2>&1 | tail -3 && uv sync --frozen --inexact 2>&1 | tail -3
cd $WT && PYTHONPATH=$WT $PY -c "import hloc, hloc.extractors.superpoint, hloc.matchers.superglue; print('HLOC OK', hloc.__file__)"
```

- If both work:
  - delete `setup/hloc.sh`, its `setup.sh` call and `HLOC_PIN`;
  - the hloc fixture model is equal (run `tests/pointcloud/sfm/test_hloc.py`).
- If superpoint/superglue fail to import (their `sys.path` append to `../../third_party` needs the submodules):
  - revert `pyproject.toml`/`uv.lock` to HEAD (`git checkout HEAD -- pyproject.toml uv.lock`);
  - keep `setup/hloc.sh`;
  - delete ONLY `HLOC_PIN` from `hloc.py`, since the pin lives in `setup/hloc.sh`;
  - log the import error verbatim.
- The `uv lock` changes the shared venv only through `uv sync`. Ask the user before `uv sync` if any other agent is running tests: plain `uv sync` prunes extras (memory: splats gsplat pin), which is why `--inexact` is used.

Commit: `build(hloc): install from git at c13273b via uv; setup/hloc.sh + HLOC_PIN deleted`

---

## Task 12: Wrapper and eval callers

**Files:** `collab_splats/wrapper/reconstructor.py`, `evals/scripts/*.py`, `collab_splats/dashboard/*`

Each step below is a commit.

- [ ] **Step 1: PLY written once**

- Delete the ff creator's PLY write (`feedforward/base.py`, `write_ply` call) and refine's write.
- `Reconstructor.build_pointcloud` writes the one PLY from `result.points`/`result.colors` after the stage finishes, for both methods and after refine.
- Test: in `tests/wrapper`, count `write_ply` calls with a `monkeypatch` counter across one ff run and one refine run. Expected: 1 per stage run.

Commit: `refactor(wrapper): write pointcloud.ply once per stage`

- [ ] **Step 2: refine reads the stored `camera_model`**

- `get_creator(backend)(**block).camera_model` → `result.camera_model`.
- `write_colmap(result.to_colmap(), out/colmap)`.
- Refine's vggt unprojection stays until P1.

Commit: `refactor(wrapper): refine uses the result's recorded camera_model`

- [ ] **Step 3: Tutorial break list**

```bash
cd $WT && rtk proxy grep -rln "FeedforwardResult\|original_coords\|image_paths\|model_width\|lift_features\|pointcloud.vda\|depth_align\|sift_db\|sfm.common\|build_pycolmap_reconstruction" docs --include=*.ipynb >> $SP/tutorial_breaks.md
```

No commit (scratch).

---

## Parity changes — one commit each, in this order

Every parity task:

- (1) writes a test that sees the new behavior on a non-identity, cropped fixture;
- (2) runs GATE, expecting `parity.py --check` to fail ONLY on the named cases;
- (3) writes the measurement to `$SP/reorg_P<n>.md`;
- (4) STOPS for user approval before `parity.py --save --cases <...>` and the commit.

### Task P1: Unprojection through `geometry/projection.unproject`, in the input dtype

**Files:** `feedforward/base.py` (`_raw_to_world_points`, `_verify_geometry`, `unproject_and_filter_points`), `sfm/base.py::densify`, `wrapper/reconstructor.py` refine, `geometry/loop_closure/{wrapper,graph}.py`

**Added 2026-09-27 (user: "move all unprojection onto the shared function"):**

- `PointcloudResult.reproject()` (ex-`FeedforwardResult`, moved in Task 6) + `pointcloud/utils.py::reproject_pixels` (and its tests in `tests/pointcloud/test_feature_lifting.py`)
  - Task 2 could not delete it: `reproject()` calls it from BA refine and `evals/scripts/eval.py`
  - same pinhole math as `unproject`, but sparse at `pixel_indices`, numpy, float64 inside
  - rewrite: unproject per frame, gather at `pixel_indices`; never the dense N×H×W×3 stack (~1.6 GB at 500×518²)
- `geometry/metrics.py::compute_photometric_ncc` inline `inv(K) @ pix * depth` unprojection

- [ ] **Step 1: Test** — `tests/geometry/test_projection.py`

```python
def test_unproject_matches_vggt_to_float32_tolerance():
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    w2c, K = _pose(5)
    depth = np.random.default_rng(5).uniform(0.5, 4, size=(2, 7, 9)).astype(np.float32)
    ext = np.stack([w2c.numpy()[:3]] * 2).astype(np.float32)
    Ks = np.stack([K.numpy()] * 2).astype(np.float32)
    want = unproject_depth_map_to_point_map(depth[..., None], ext, Ks)
    got = unproject(torch.from_numpy(depth), torch.from_numpy(ext), torch.from_numpy(Ks)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
```

Move the `vggt` import to the module top when committing.

- [ ] **Step 2: Replace every vggt / inline unprojection**

- Pattern: `unproject(torch.from_numpy(depth), torch.from_numpy(w2c), torch.from_numpy(K)).numpy()`.
- LC's stride-8 grid: unproject the full grid, then slice `[:, ::8, ::8]`.
- Delete `_raw_to_world_points` if nothing else remains in it. Its LC callers import `unproject` directly.

- [ ] **Step 3: Measure** — `$SP/reorg_P1.md`

- Max |Δ| of `world_points` and `points`, T12 zarr vs P1 zarr, for ff (all 4 backends via parity) and sfm.
- LC `rotation_only` parity delta.
- Expected: max drift ≤ 1e-4 relative. Anything larger: STOP.

- [ ] **Step 4: User approval → `parity.py --save --cases <failed>` → commit**

`fix(pointcloud): unproject via geometry.projection in the input dtype (P1)`

### Task P3: `subsample_points(mask, max_points, seed=0) -> mask`

**Files:** `pointcloud/utils.py`, `feedforward/base.py` (`_limit_trues`, `_mask_to_points`), LC callers of the array form

- [ ] **Step 1: Tests** — `tests/pointcloud/test_utils_subsample.py` (replace the array-form tests)

```python
def test_subsample_points_keeps_mask_shape_and_cap():
    mask = np.zeros((3, 4, 5), bool)
    mask[:, 1:, 2:] = True
    out = subsample_points(mask, 7)
    assert out.shape == mask.shape and out.sum() == 7 and not (out & ~mask).any()


def test_subsample_points_under_cap_is_identity_and_rng_private():
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    mask = np.ones(5, bool)
    assert subsample_points(mask, 10) is mask
    assert np.random.random() == before
```

- [ ] **Step 2: Implement**

- The `_limit_trues` body under the name `subsample_points`, with `seed` a real keyword.
- `_mask_to_points` inlines to `m = subsample_points(mask, max_points); pixel_indices = np.argwhere(m).astype(np.int32); points[m], colors[m]`.
- Each LC caller of the array form builds a mask over its point array: `pts[subsample_points(np.ones(len(pts), bool), cap)]`.

- [ ] **Step 3: Measure** — `$SP/reorg_P3.md`

- LC parity `rotation_only` for the four backbones.
- chess seq-01 ATE before/after, run in tmux via `evals/scripts/eval.py`.
- The ff point sample is unchanged (same function); the LC sample changes.

- [ ] **Step 4: User approval → save → commit** `fix(pointcloud): one mask-space subsample_points (P3)`

### Task P2: Order becomes conf mask → clean (SOR) → subsample

**Files:** `feedforward/base.py::unproject_and_filter_points` / `_postprocess`

- [ ] **Step 1: Test.** `clean_pointcloud` runs on the full conf-masked set.

- Patch `clean_pointcloud` with a recorder.
- Assert that it received `mask.sum()` points (> `max_points`) and that the final count is ≤ `max_points`.

- [ ] **Step 2: Implement the reorder**

- [ ] **Step 3: Measure** — `$SP/reorg_P2.md`

- Point count.
- SOR runtime + peak RSS on a 300-frame scene, under `/usr/bin/time -v` in tmux, against the 46.6 GB cap.
- Splat PSNR on C0043.
- If peak RSS > 30 GB or SOR > 5 min, implement the spec fallback instead: clean a `4 × max_points` sample, then cap. Re-measure.

- [ ] **Step 4: User approval → save → commit** `fix(pointcloud): clean before subsampling the sparse set (P2)`

### Task P4: `depth_residual` + one tolerance for multiview and lifting

**Files:** `geometry/projection.py` (+`depth_residual`), `feedforward/base.py` multiview loop, `semantics/lifting.py`, `tests/pointcloud/test_mv_conf.py` (identity extrinsics → non-identity, spec item 6)

- [ ] **Step 1: Tests** — `tests/geometry/test_projection.py`

```python
def test_depth_residual_is_signed_relative_error():
    w2c, K = _pose(7)
    depth_j = torch.full((6, 8), 2.0, dtype=torch.float64)
    pts = unproject(depth_j[None], w2c[None], K[None])[0].reshape(-1, 3)
    res, z, in_frame = depth_residual(pts, depth_j * 1.1, K, w2c)
    assert in_frame.all()
    torch.testing.assert_close(res, torch.full_like(res, 0.1))
    torch.testing.assert_close(z, torch.full_like(z, 2.0))


def test_depth_residual_flags_out_of_frame_and_behind():
    w2c, K = _pose(7)
    far = torch.tensor([[1e4, 0, 1.0]], dtype=torch.float64) @ w2c[:3, :3] - w2c[:3, 3] @ w2c[:3, :3]
    _, _, in_frame = depth_residual(far, torch.ones(6, 8, dtype=torch.float64), K, w2c)
    assert not in_frame.any()
```

Also run a multiview count-flip check on a seeded non-identity fixture: compare old vs new `inlier_count` and `valid_count`. The measurement records the number of flipped pixels.

- [ ] **Step 2: Implement `depth_residual`**

It samples the way multiview does today: nearest, `align_corners=True`, inclusive bounds.

```python
def depth_residual(
    points_world: Tensor, depth_j: Tensor, intrinsics_j: Tensor, world_to_cam_j: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Signed relative depth disagreement of world points with camera j's depth map.

    - project each point into j: pixel (u, v) and camera-frame depth z
    - sample j's depth map at the nearest pixel: s
    - residual = (s - z) / z; negative = something nearer occludes the point, positive = free space
    - a pixel with no depth (s == 0) gives residual -1; callers treat s > 0 as residual > -1

    Args:
        points_world: (P, 3) world points.
        depth_j: (H, W) camera j depth, 0 = missing.
        intrinsics_j: (3, 3) K of camera j on depth_j's grid.
        world_to_cam_j: (4, 4) w2c of camera j.

    Returns:
        (residual (P,), z (P,), in_frame (P,) bool: pixel within [0, W-1] x [0, H-1] and z > 0).
    """
    h, w = depth_j.shape
    pixels, points_cam = project(points_world, world_to_cam_j, intrinsics_j)
    z = points_cam[:, 2]

    # Nearest sample, align_corners=True grid: pixel 0 -> -1, pixel W-1 -> 1
    grid = torch.stack([pixels[:, 0] / (w - 1) * 2 - 1, pixels[:, 1] / (h - 1) * 2 - 1], dim=-1)
    in_frame = (grid.abs() <= 1).all(dim=-1) & (z > 0)
    sampled = F.grid_sample(depth_j[None, None], grid[None, None], mode="nearest", padding_mode="zeros", align_corners=True)[0, 0, 0]

    # Relative residual; z floored like project's divide
    return (sampled - z) / z.clamp(min=1e-6), z, in_frame
```

- [ ] **Step 3: Adopt it in multiview and lifting with one tolerance**

- Multiview inner loop:
  - `pts_world` comes from `unproject` (P1);
  - `res, z, ok = depth_residual(pts_world, depth_t[j], K[j], E[j])`;
  - `valid = src_valid & ok`;
  - `has_depth = res > -1`;
  - `tol_rel = abs_thresh / z.abs() + rel_thresh`;
  - `inlier = (res.abs() < tol_rel) & valid & has_depth`;
  - `occluded = (res < -tol_rel) & valid & has_depth`.
- The `collect` branch uses `res[sel]` for `rel` and `z[sel]` for `median_depth`.
- Lifting replaces its inline projection + `depth_ok` with `depth_residual` and the same `abs/z + rel` tolerance. Its bounds become multiview's inclusive `[0, W-1]` with nearest rounding (it truncated before: a real change, which is why this is P4).

- [ ] **Step 4: Measure** — `$SP/reorg_P4.md`

- Multiview inlier/valid count deltas: pixels flipped per scene.
- Lifted-feature cosine old vs new on C0043: mean and p10.

- [ ] **Step 5: User approval → save → commit** `fix(geometry): depth_residual is the one cross-view depth test (P4)`

### Task P6: ff `to_colmap` writes PINHOLE from the stored K

- [ ] **Step 1: Test**

- `_result("SIMPLE_PINHOLE")` with fx ≠ fy: after P6, the exported camera is PINHOLE with `params == [fx, fy, cx, cy]` of the rescaled K.
- ff creators set `camera_model = "PINHOLE"`. VGGT-X's `SIMPLE_PINHOLE` default goes, and the mean/max rule in `_camera_params` is deleted.

- [ ] **Step 2: Measure** — `$SP/reorg_P6.md`: per-camera K before/after on the exported model (VGGT-X C0043).

- [ ] **Step 3: User approval → commit** `fix(pointcloud): feedforward COLMAP export is PINHOLE from the stored K (P6)`

### Task P7: zarr schema clean break

- [ ] **Step 1: Tests**

- `save_zarr` writes the keys `crop_box`, `image_names`, `model_hw`, `camera_model`, `camera_params`, plus a `schema: 2` attr.
- `load_zarr` on a zarr without `schema: 2` raises `ValueError("stale pointcloud.zarr (pre-reorg schema); delete it and re-run the pointcloud stage")`. That is the `preproc.qa.load_video_quality` pattern; read it first (`rtk proxy grep -n "stale" -B3 -A3 collab_splats/preproc/qa.py`).

- [ ] **Step 2: Implement**

- Delete the old-key mapping from Task 5.
- Update `$SP/zarr_eq.py` `RENAMES` for the snapshot diffs.

- [ ] **Step 3: Measure** — `$SP/reorg_P7.md`

- An old zarr raises the stale error.
- The new zarr round-trips.
- List every on-disk zarr consumer outside `pointcloud/` that reads keys directly: `rtk proxy grep -rn "\"original_coords\"\|'original_coords'\|\"image_paths\"" collab_splats evals` → empty.

- [ ] **Step 4: User approval → commit** `feat(pointcloud)!: pointcloud.zarr schema 2 — renamed keys, stale-schema error (P7)`

### Task P8: SIFT DB reuse keyed on DB contents

- [ ] **Step 1: Tests** — `tests/pointcloud/sfm/test_base.py`

```python
def test_sift_database_reuses_on_matching_params(tmp_path, fake_sift):
    sift_database(tmp_path / "img", tmp_path / "db.db", NAMES, pairing="exhaustive")
    sift_database(tmp_path / "img", tmp_path / "db.db", NAMES, pairing="exhaustive")
    assert fake_sift.builds == 1 and not (tmp_path / "db.db.json").exists()


def test_sift_database_rebuilds_on_changed_params(tmp_path, fake_sift):
    sift_database(tmp_path / "img", tmp_path / "db.db", NAMES, pairing="sequential", overlap=10)
    sift_database(tmp_path / "img", tmp_path / "db.db", NAMES, pairing="sequential", overlap=5)
    assert fake_sift.builds == 2
```

`fake_sift` is ported from the existing sidecar tests' extraction stub.

- [ ] **Step 2: Implement**

- After a build, write one row to a `collab_params(json TEXT)` table in the DB:

  ```python
  with sqlite3.connect(db) as c:
      c.execute("CREATE TABLE IF NOT EXISTS collab_params(json TEXT)")
      c.execute("DELETE FROM collab_params")
      c.execute("INSERT INTO collab_params VALUES (?)", (json.dumps(params, sort_keys=True),))
  ```

- Reuse iff `_database_holds(db, names)` AND the stored json equals the current params.
- A DB with no table (pre-reorg) rebuilds once.
- Delete the sidecar read/write.

- [ ] **Step 3: Measure** — `$SP/reorg_P8.md`: a rerun reuses (timing), a param change rebuilds, and the first run after the upgrade rebuilds once.

- [ ] **Step 4: User approval → commit** `fix(sfm): SIFT DB reuse keyed on a params row inside the DB (P8)`

---

## Task 13: Docs, changelog, in-flight entry

**Files:** `docs/pointcloud.md` (or the module doc path under `docs/`), `docs/superpowers/CHANGELOG.md`, `CLAUDE.md` (Architecture tree + in-flight list), the spec (status line)

- [ ] **Step 1: Module doc**

- Update to the resulting layout (spec § Resulting layout).
- Any mention of `FeedforwardResult`, `vda.py`, `depth_align.py`, `sift_db.py`, `common.py`, `reproject_pixels` → current names.

```bash
cd $WT && rtk proxy grep -rn "FeedforwardResult\|depth_align\|sift_db\|sfm/common\|reproject_pixels\|vda.py\|original_coords" docs --include=*.md | grep -v superpowers   # expected: empty
```

- [ ] **Step 2: `CLAUDE.md`**

- Replace the `pointcloud/` block of the Architecture tree with the new layout.
- Remove the `pointcloud-reorg` in-flight line.
- Add a Recently Completed bullet.
- The hook guard enforces the size limit.

- [ ] **Step 3: CHANGELOG entry** with commits, gates and the P-measurements summary. The spec status becomes `implemented`.

- [ ] **Step 4: Commit** `docs(pointcloud): reorg complete — layout, changelog, in-flight cleared`

- [ ] **Step 5: Final report** to the user from `final_report_notes.md`: every disclosure, and `tutorial_breaks.md`. No merge, no push; the user decides.

---

## Self-review record

- **Spec coverage:**
  - §1 → Tasks 1, 5, 6, P2, P3, P7, 10.
  - §2 → Tasks 7, 8, 9, 11.4, P8.
  - §3 → Tasks 2, 4, 11.1–11.3, P1, P4, P6.
  - §4 gates → each task's GATE and measurement.
  - Carried items 5/6/7 → P3, P4, Task 1 Step 5.
  - The omega `_CropProbe` → Task 4.
- **Not in any task, by decision:** P5 (A3), plus the spec's out-of-scope list.
- **Type consistency:**
  - `PointcloudResult` fields: Task 5 renames, Task 6 adds `camera_model`/`camera_params`, P7 persists them.
  - `unproject` is `(depth (..., H, W), w2c (..., 4|3, 4), K (..., 3, 3), *, pixel_offset)` throughout.
  - `project` is single-camera throughout.
  - `depth_residual` returns 3 values in Task P4 and in amendment A1.
  - `get_sfm_creator` is used by Task 8 Step 5.
