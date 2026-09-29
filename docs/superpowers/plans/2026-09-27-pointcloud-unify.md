# Pointcloud Unify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one result type (`PointcloudResult`), one depth module, one creator path (`create(images_dir, out_dir, model_dir)`) and one implementation per listed operation, with net-negative lines. Implements [spec](../specs/2026-09-27-pointcloud-unify-design.md).

**Architecture:**
- `FeedforwardResult` moves into `pointcloud/base.py` as `PointcloudResult`. It stores K twice: `intrinsics` at full resolution and `model_intrinsics` on the model grid.
- COLMAP becomes an export only (`to_colmap` → `utils/colmap.write_colmap_reconstruction`). Stage 2+ always reload from `pointcloud.zarr`.
- Feedforward and sfm creators share `create()`. The sfm skeleton lives once, in `sfm/base.py`.

**Tech Stack:** Python 3.11, numpy, torch, pycolmap 4.x, zarr 3, pytest.

---

## Conventions (every task)

- `WT=/workspace/collab-splats/.worktrees/pointcloud-release` (branch `clean/pointcloud-release`)
- `SP=/workspace/scratch/pc-release`
- `PY=/opt/venv/reconstruction/bin/python`
- Every Bash command starts with `cd $WT &&`. Print `collab_splats.__file__` (must be under `$WT`) before any pytest or parity run.
- Unit tests: `cd $WT && PYTHONPATH=$WT PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider <paths>`.
- **G′ gate** is `$SP/gprime.sh $WT $SP/unify_T<n>.log` and must print `IDS IDENTICAL`.
  - It uses the baseline `$SP/baseline_unify.txt` (Task 0).
  - Run it in the foreground with a Bash timeout of 600000, one gate per commit, and make no edits while it runs.
  - Then run `tests/utils tests/splats tests/semantics`. Failures must match the Task 0 list.
- **Parity gate:** `cd $WT && PARITY_WT=$WT PYTHONPATH=$WT PYTHONUTF8=1 $PY $SP/parity.py --check`.
  - Refactor tasks must be bit-exact.
  - Parity-moving tasks (P1–P4, P6–P8, K split, clean move) may fail only on the fields the task names.
    - Paste the diff into the commit body.
    - Then chain the baseline with `PARITY_BASE=$SP/parity_baseline_unify … --save`. This writes a separate dir; the original `parity_baseline/` is never overwritten.
- **Commits:**
  - `git add <paths> && git commit --only <paths>`; docs under `docs/superpowers` need `git add -f`.
  - Conventional subject with scope.
  - Trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Never:**
  - amend, rebase, reset, merge, push, bare `git stash`;
  - pip or uv;
  - edit notebooks;
  - `--tb=no`;
  - pipe pytest into `tail` or `head`.
- **Style (CLAUDE.md):**
  - imports at top; no `f(g(x))`, one call per line; US spelling;
  - comment runs of three lines or more are a header plus `- ` bullets;
  - docstring contract: `"""` on its own line, one summary line, `Args:` / `Returns:` for public defs;
  - block comments with a blank line above.
- **Contract** (grep before adding, renaming or deleting a name): the spec's "Contract" section. Specifically:
  - K undo is two commented lines, never a wrapper;
  - no layout constants or one-liner path helpers;
  - ADR 017: a public helper needs two real callers; decode with `read_image` / `read_frames`;
  - LC wrapper gets call-site edits only.
- Line numbers below are from `1ac95ee0` and drift. Locate code by content.
- If the "before" text is missing or a gate disagrees, restore the files and report BLOCKED with evidence. Do not improvise.
- A task is done only when every changed file is in the commit and the notebooks are untouched.

## Order note

P6 (PINHOLE export) runs as Task 2, before the result merge, instead of in the P-group:
- Today's SIMPLE_PINHOLE export averages fx/fy at model resolution and takes `max` after rescaling. That value cannot be computed from a full-res K.
- Landing P6 first lets `to_colmap()` read `intrinsics` with no camera-model branch and gives it a bit-exact old-vs-new test.
- Every other task follows the spec Order.

## File map

| File | Change |
|---|---|
| `collab_splats/utils/colmap.py` | new: `write_colmap_reconstruction`, `read_colmap_reconstruction` |
| `collab_splats/pointcloud/base.py` | `PointcloudResult` (moved from `feedforward/base.py`), `BasePointcloudCreator.create` |
| `collab_splats/pointcloud/depth.py` | new: `_load_vda_model`, `estimate_depth`, `align_depth` |
| `collab_splats/pointcloud/sfm/base.py` | new: `BaseSfmCreator`, `_IncrementalSfmCreator` |
| `collab_splats/pointcloud/feedforward/base.py` | loses result, builders, `write_colmap`; gains `center_crop_coords`; `create` |
| `collab_splats/semantics/lifting.py` | new: `lift_features` + two private samplers |
| `collab_splats/geometry/projection.py` | lane B, plus `depth_residual` (P4) |
| `collab_splats/wrapper/reconstructor.py` | `colmap_model_dir` property; `create` → `save_zarr` → `write_ply`; `load_zarr` reload |
| deleted | `pointcloud/vda.py`, `pointcloud/depth_align.py`, `pointcloud/sfm/common.py`, `splats/pgsr.py` |

---

### Task 0: Baseline

**Files:** scratch only (`$SP/gprime.sh`, `$SP/parity.py`)

- [ ] **Step 1: Make the baseline file an env var in `gprime.sh`.** Replace `"$SP/baseline_a29.txt"` with `"${GPRIME_BASE:-$SP/baseline_unify.txt}"`.
- [ ] **Step 2: Make the parity baseline dir an env var.** In `parity.py` line 55, replace `BASE_DIR = SP / "parity_baseline"` with `BASE_DIR = Path(os.environ.get("PARITY_BASE", SP / "parity_baseline"))`.
- [ ] **Step 3: Record the G′ baseline at the tip.** Run: `cd $WT && git log --oneline -1 && $SP/gprime.sh $WT $SP/baseline_unify_raw.log; cp $SP/baseline_unify_raw.log $SP/baseline_unify.txt`
  - Expect the proof line under `$WT`.
  - Record the pass/fail counts and the FAILED ids in the plan's run log.
- [ ] **Step 4: Record the side-suite baseline.** Run: `cd $WT && PYTHONPATH=$WT PYTHONUTF8=1 $PY -m pytest -q -p no:cacheprovider tests/utils tests/splats tests/semantics > $SP/side_unify_base.log 2>&1; grep -aE "^(FAILED|ERROR)|passed|failed" $SP/side_unify_base.log`
- [ ] **Step 5: Seed the chained parity baseline.**
  - Run `parity.py --check`; it must be bit-exact against `parity_baseline/`.
  - Then run `cp -r $SP/parity_baseline $SP/parity_baseline_unify`.
  - Export `PARITY_BASE=$SP/parity_baseline_unify` for every later task.

### Task 1: COLMAP IO (lane A, trimmed)

**Files:**
- Create: `collab_splats/utils/colmap.py`, `tests/utils/test_colmap.py`
- Modify:
  - `collab_splats/pointcloud/sfm/common.py` (`write_sfm_model` body → the writer)
  - `collab_splats/pointcloud/feedforward/base.py` (`write_colmap` → the writer)
  - `collab_splats/wrapper/reconstructor.py` (`colmap_model_dir` property; done check)
  - `tests/pointcloud/sfm/test_common.py`, `tests/wrapper/test_reconstructor.py`
- Reference only: `git show f7caf1a1:collab_splats/utils/colmap.py`. Do not cherry-pick it: its constant, `sparse_dir` and `is_complete` are contract violations.

- [ ] **Step 1: Write the failing tests** in `tests/utils/test_colmap.py`:

```python
"""
Tests for utils/colmap.py: stem names, atomic swap, round trip.
"""

import numpy as np
import pycolmap
import pytest

from collab_splats.utils.colmap import read_colmap_reconstruction, write_colmap_reconstruction


def _recon(names: list[str]) -> pycolmap.Reconstruction:
    recon = pycolmap.Reconstruction()
    for i, name in enumerate(names, start=1):
        camera = pycolmap.Camera(model="PINHOLE", width=64, height=48, params=[50.0, 50.0, 32.0, 24.0], camera_id=i)
        recon.add_camera_with_trivial_rig(camera)
        image = pycolmap.Image(name=name, camera_id=i, image_id=i)
        recon.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
    recon.add_point3D(np.array([0.0, 0.0, 1.0]), pycolmap.Track(), np.array([1, 2, 3], dtype=np.uint8))
    return recon


def test_write_names_are_stems(tmp_path):
    model_dir = tmp_path / "colmap" / "sparse" / "0"
    write_colmap_reconstruction(_recon(["frame_000001.png", "frame_000002.jpg"]), model_dir)
    names = sorted(img.name for img in read_colmap_reconstruction(model_dir).images.values())
    assert names == ["frame_000001", "frame_000002"]


def test_write_replaces_whole_model(tmp_path):
    model_dir = tmp_path / "m"
    write_colmap_reconstruction(_recon(["a.png", "b.png"]), model_dir)
    write_colmap_reconstruction(_recon(["c.png"]), model_dir)
    assert [img.name for img in read_colmap_reconstruction(model_dir).images.values()] == ["c"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m"]


def test_write_clears_crash_leftovers(tmp_path):
    model_dir = tmp_path / "m"
    (tmp_path / ".m.tmp").mkdir()
    (tmp_path / ".m.old").mkdir()
    write_colmap_reconstruction(_recon(["a.png"]), model_dir)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m"]


def test_read_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_colmap_reconstruction(tmp_path / "absent")
```

- [ ] **Step 2: Confirm they fail.** Run: `… -m pytest tests/utils/test_colmap.py -v`. Expected: `ModuleNotFoundError: collab_splats.utils.colmap`.
- [ ] **Step 3: Implement `collab_splats/utils/colmap.py`:**

```python
"""
COLMAP binary model IO: stem-named, atomically swapped writes and the matching read.

- callers own the layout: both functions take the model dir itself
"""

import shutil
from pathlib import Path

import pycolmap


def write_colmap_reconstruction(recon: pycolmap.Reconstruction, model_dir: Path) -> None:
    """
    Write a binary model to exactly `model_dir`, image names reduced to stems, swapped in whole.

    - stems match the pipeline's frame ids (frame_NNNNNN), whatever extension the mapper saw
    - written to a hidden sibling, the old model moved aside, then the new one renamed in
    - model_dir is always a whole model or absent; a crash's leftover siblings are cleared next write

    Args:
        recon: model to write; its image names are renamed in place.
        model_dir: directory that holds cameras.bin / images.bin / points3D.bin afterwards.
    """
    # Image names -> stems
    for image in recon.images.values():
        image.name = Path(image.name).stem

    # Write a fresh hidden sibling; write_binary needs the dir to exist
    model_dir = Path(model_dir)
    tmp = model_dir.with_name(f".{model_dir.name}.tmp")
    old = model_dir.with_name(f".{model_dir.name}.old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    recon.write_binary(str(tmp))

    # Move the old model aside, rename the new one in, drop the old
    shutil.rmtree(old, ignore_errors=True)
    if model_dir.exists():
        model_dir.rename(old)
    tmp.rename(model_dir)
    shutil.rmtree(old, ignore_errors=True)


def read_colmap_reconstruction(model_dir: Path) -> pycolmap.Reconstruction:
    """
    Binary model in `model_dir`; the inverse of write_colmap_reconstruction.

    Args:
        model_dir: directory holding the three .bin files.

    Returns:
        The loaded reconstruction.

    Raises:
        FileNotFoundError: model_dir does not exist.
    """
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"no COLMAP model at {model_dir}")
    return pycolmap.Reconstruction(str(model_dir))
```

- [ ] **Step 4: Rewire the callers.**
  - `sfm/common.py:write_sfm_model` keeps its signature. Its body renames nothing and calls `write_colmap_reconstruction(recon, Path(data_dir) / "colmap" / "sparse" / "0")`. The rename-to-stems moves into the writer. Delete `rename_images_to_stems` if it is left with no caller.
  - `feedforward/base.py:write_colmap`: replace the `mkdir` + `write_binary` block with `write_colmap_reconstruction(recon, Path(output_dir) / "colmap" / "sparse" / "0")`. Both files keep their layout spelling until Tasks 3 and 6 delete them.
  - `wrapper/reconstructor.py`:
    - add a `colmap_model_dir` property next to `pointcloud_zarr`: `return self.backend_dir / "colmap" / "sparse" / "0"`, with a one-line docstring;
    - `_stage_output_exists("pointcloud")` → `return self.pointcloud_zarr.exists() and self.colmap_model_dir.exists()`;
    - `_load_pointcloud_from_disk` → `from_colmap(self.colmap_model_dir.parent.parent, ...)` if `from_colmap` takes the root. Check the signature; this line dies in Task 3.
  - Update `tests/pointcloud/sfm/test_common.py` and `tests/wrapper/test_reconstructor.py` where they built `cameras.bin` as the done marker: now the dir itself is the marker.
- [ ] **Step 5: Run the tests.** Run `tests/utils/test_colmap.py tests/pointcloud/sfm tests/wrapper -q`. All pass except known baseline ids.
- [ ] **Step 6: Run the gates.** G′ `unify_T1` → IDS IDENTICAL; side suite; parity bit-exact.
- [ ] **Step 7: Commit.** `refactor(utils): colmap.py is the one COLMAP model writer/reader`.

### Task 2: P6 — feedforward COLMAP export is always PINHOLE

**Files:**
- Modify:
  - `feedforward/base.py`: `build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`, `write_colmap`, the `camera_model` field and docstring at ~1104/1122, `build_colmap`
  - `vggtx.py:82,94`, `vggt_omega.py:86,99`, `loger.py:111,123` (drop the `camera_model` field and its docstring line)
  - `wrapper/reconstructor.py:1240-1241` (refine: drop the `get_creator(...).camera_model` lookup)
  - `evals/` if any hit
- Test: `tests/pointcloud/test_feedforward_intrinsics.py` (or wherever SIMPLE_PINHOLE is asserted; grep `SIMPLE_PINHOLE` under `tests/`)

- [ ] **Step 1: Write the failing test.**
  - Every camera written by `write_colmap` is `PINHOLE`, and its params equal `rescale_intrinsics` then `shift_intrinsics` applied to the result's K.
  - Use a cropped box (`original_coords` row `[0, 60, 640, 420, 640, 480]`) and fx ≠ fy.

```python
def test_write_colmap_pinhole_matches_two_line_undo(tmp_path):
    # Fixture: 2 frames, 64x48 model grid, cropped box, fx != fy
    result = _tiny_result(fx=40.0, fy=44.0, box=[0.0, 60.0, 640.0, 420.0, 640.0, 480.0])
    recon = write_colmap(result, tmp_path)
    for image_id, image in recon.images.items():
        camera = recon.cameras[image.camera_id]
        box = result.original_coords[image_id - 1]
        crop_hw = (box[3] - box[1], box[2] - box[0])
        K = rescale_intrinsics(result.intrinsics[image_id - 1], (48, 64), crop_hw)
        K = shift_intrinsics(K, box[:2])
        assert camera.model.name == "PINHOLE"
        np.testing.assert_array_equal(camera.params, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
```

  `_tiny_result` builds a `FeedforwardResult` inline in the test module: 2 frames, 3 points, identity-ish extrinsics, `image_paths=[Path("frame_000000.png"), Path("frame_000001.png")]`, `model_width=64`, `model_height=48`.
- [ ] **Step 2: Confirm it fails** for the VGGT-X default (SIMPLE_PINHOLE today): run it with a creator-less call; `write_colmap` still takes `camera_model`, so the call itself fails.
- [ ] **Step 3: Implement.**
  - Remove the `camera_model` parameter from `build_pycolmap_reconstruction` and `write_colmap`, and the field from creators. The camera is always `PINHOLE` with `params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]`.
  - In the rescale, drop the SIMPLE_PINHOLE branch and the ValueError; always write the 4 params.
  - Delete any test that asserted SIMPLE_PINHOLE.
- [ ] **Step 4: Run the gates.**
  - Test passes; G′; side suite; parity. `write_colmap` is not in parity, so expect bit-exact.
  - Also run `grep -rn SIMPLE_PINHOLE collab_splats evals tests`. The only hits allowed are LoGeR's comment, if it still reads true, and sfm `SIMPLE_RADIAL`.
- [ ] **Step 5: Commit.** `fix(pointcloud): feedforward COLMAP export is PINHOLE from the stored K (P6)`. The body notes that VGGT-X's `colmap/sparse/0` camera moves from `max(f)` to `(fx, fy)`.

### Task 3: One `PointcloudResult`, K stored twice, P7, wrapper flow

This is the biggest task. It is one commit because the rename must not leave two types alive.

**Files:**
- Modify:
  - `collab_splats/pointcloud/base.py`: replace the old `PointcloudResult` with the moved `FeedforwardResult` class, renamed.
  - `collab_splats/pointcloud/feedforward/base.py`:
    - delete `FeedforwardResult`, `build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`, `write_colmap`;
    - `reconstruct` → `create`;
    - `build_colmap` → `write_colmap_model(model_dir)`, one creator method;
    - `postprocess` sets full-res `intrinsics`.
  - `collab_splats/pointcloud/__init__.py`, `feedforward/__init__.py` (exports), `feedforward/mapanything.py`, `feedforward/loger.py`
  - `collab_splats/pointcloud/depth_align.py`: construct `PointcloudResult`; set `intrinsics` = COLMAP K and `model_intrinsics` = depth-res K.
  - `collab_splats/pointcloud/utils.py` (`lift_features` / `reproject_pixels` type hints)
  - `collab_splats/geometry/loop_closure/wrapper.py`: call sites only. `reconstruct` return annotation; `build_colmap(output_dir)` → `write_colmap_model(model_dir)`.
  - `collab_splats/geometry/bundle_adjustment.py`, `collab_splats/localization/localizer.py`, `collab_splats/dashboard/{viewer,pipeline,app}.py`, `collab_splats/utils/visualization.py`
  - `collab_splats/wrapper/reconstructor.py`: every `FeedforwardResult` / `PointcloudResult` / `from_colmap` site (list below).
  - `evals/scripts/{eval,eval_splats,eval_localization_parity,analyze_splats}.py`
  - tests: every file `grep -rln "FeedforwardResult\|from_colmap\|reconstruction=" tests` lists.
  - `$SP/parity.py`: import rename; `_fields` encodes `model_intrinsics` as well.

**K rule:**
- Sites on the model grid read `model_intrinsics`: `reproject`, `_multiview`, `compute_multiview_depth_confidence` callers, BA / `check_model_resolution`, lifting, the quality report, the refine `unproject` of `world_points`.
- Sites that want the full-res frame read `intrinsics`: mesh `_run_tsdf_mesh`, `splats()`, `to_colmap`, localization, dashboard.
- For each site you touch, write which K it reads in the commit body table.

- [ ] **Step 1: Write the failing tests** in `tests/pointcloud/test_base.py`. Replace the old wrapper tests:

```python
def test_postprocess_sets_full_res_intrinsics_on_cropped_box():
    # Cropped box: full-frame boxes would make both Ks equal and hide a swap
    result = _tiny_result(box=[0.0, 60.0, 640.0, 420.0, 640.0, 480.0])  # model_intrinsics set, intrinsics None
    creator = _StubCreator(outputs_from=result)                          # _postprocess returns result
    creator.postprocess()
    out = creator.outputs
    box = out.original_coords[0]
    crop_hw = (box[3] - box[1], box[2] - box[0])
    K = rescale_intrinsics(out.model_intrinsics[0], (out.model_height, out.model_width), crop_hw)
    K = shift_intrinsics(K, box[:2])
    np.testing.assert_array_equal(out.intrinsics[0], K)


def test_to_colmap_equals_old_write_colmap(tmp_path):
    # Old path frozen at 1ac95ee0+P6: build at model res then rescale
    result = _full_res_result()
    recon = result.to_colmap()
    for image_id, image in recon.images.items():
        np.testing.assert_array_equal(recon.cameras[image.camera_id].calibration_matrix(), result.intrinsics[image_id - 1])
        assert recon.cameras[image.camera_id].width == int(result.original_coords[image_id - 1][4])


def test_load_zarr_refuses_pre_unify_schema(tmp_path):
    path = tmp_path / "pointcloud.zarr"
    _full_res_result().save_zarr(path)
    store = zarr.open(str(path), mode="r+")
    del store["model_intrinsics"]
    with pytest.raises(ValueError, match="re-run the pointcloud stage"):
        PointcloudResult.load_zarr(path)


def test_write_ply_round_trip(tmp_path):
    result = _full_res_result()
    result.write_ply(tmp_path / "sparse_pc.ply")
    ply = trimesh.load(tmp_path / "sparse_pc.ply")  # or plyfile, whichever the repo already imports
    np.testing.assert_allclose(np.asarray(ply.vertices), result.points, rtol=0, atol=1e-6)
```

  Also freeze an old-vs-new K equality script, `$SP/unify_T3_k_eq.py`. It is not committed:
  - At `HEAD~0`, before your edits, run `VGGTXCreator`, `VGGTOmegaCreator`, `MapAnythingCreator` and `LoGeRCreator._postprocess` through `$SP/parity.py`'s synthetic cases. Save `write_colmap(...)` camera K per frame to npz.
  - After your edits, assert the new `result.intrinsics` and `to_colmap()` K equal it bit-for-bit, and `model_intrinsics` equals the old `intrinsics`.
  - Paste the output into the commit body.
- [ ] **Step 2: Confirm they fail.** Expected: ImportError, or the `model_intrinsics` attribute is missing.
- [ ] **Step 3: Move the class.**
  - Cut `FeedforwardResult` from `feedforward/base.py` into `pointcloud/base.py` as `PointcloudResult`. Keep the dataclass order.
  - Add `model_intrinsics: np.ndarray` right after `intrinsics`. Docstrings:
    - `intrinsics`: "(N, 3, 3) K on the full-res frame";
    - `model_intrinsics`: "(N, 3, 3) K on the model grid, matching depth / world_points / pixel_indices".
  - `save_zarr` writes both. Its codec is `utils.io.LZ4` if that name exists at HEAD (grep `LZ4` in `utils/io.py`); else keep `BloscCodec(cname="lz4")` and note it.
  - `load_zarr`: before reading, `if "model_intrinsics" not in store: raise ValueError(f"{path} predates the model_intrinsics schema — re-run the pointcloud stage")`.
  - `reproject` uses `self.model_intrinsics`.
  - Delete the old `PointcloudResult` class (and `from_colmap`); keep `BasePointcloudCreator` in `base.py`.
- [ ] **Step 4: Add `to_colmap()` and `write_ply()`.**
  - `to_colmap()` is the body of `build_pycolmap_reconstruction`, with these changes:
    - K = `self.intrinsics[i]`;
    - width/height = `int(self.original_coords[i][4])`, `int(self.original_coords[i][5])`;
    - names = `[p.name for p in self.image_paths]`.
  - It returns the recon and writes nothing.
  - `write_ply(path)` writes `points` + `colors` as a binary PLY. Use the writer the repo already has: grep `def write_ply\|PlyData\|export_PLY` in `collab_splats`, and prefer an existing helper. If none, use `plyfile`, which is in the venv (check with `$PY -c "import plyfile"`).
  - Delete `build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions` and `write_colmap`.
- [ ] **Step 5: Change the ff creator flow.**
  - `postprocess()`: after `self.outputs = self._postprocess(self.raw_outputs)`, add:

```python
        # Full-res K from the model grid, once, for every full-frame consumer
        # - rescale undoes the resize to each crop box, shift adds the box origin
        outputs = self.outputs
        crop_hw = np.stack([outputs.original_coords[:, 3] - outputs.original_coords[:, 1],
                            outputs.original_coords[:, 2] - outputs.original_coords[:, 0]], axis=-1)
        K = rescale_intrinsics(outputs.model_intrinsics, (outputs.model_height, outputs.model_width), crop_hw)
        K = shift_intrinsics(K, outputs.original_coords[:, :2])
        self.outputs = dataclasses.replace(outputs, intrinsics=K)
```

    - First check that `rescale_intrinsics` / `shift_intrinsics` accept batched `(N, 3, 3)` K and `(N, 2)` hw. Read `geometry/transforms.py`. If they do not batch, loop per frame and `np.stack`, one call per line. Keep dtype float64 or the old float32 to match `unify_T3_k_eq.py`.
    - `_postprocess` (base and MapAnything's override) constructs `PointcloudResult(..., model_intrinsics=intrinsic, intrinsics=None)`. If the field must be non-None, make it `np.ndarray | None = None`, set by `postprocess`.
  - `write_colmap_model(self, model_dir: Path) -> None`: `write_colmap_reconstruction(self.outputs.to_colmap(), model_dir)`. It replaces `build_colmap`, which also wrote the PLY; the PLY moves to the wrapper.
  - `create(self, images_dir: Path, out_dir: Path, model_dir: Path) -> PointcloudResult` replaces `reconstruct`:
    - check existence, `mkdir out_dir`;
    - `load_model`, `setup_inference`, `run_inference`, `postprocess`, `write_colmap_model(model_dir)`;
    - `return self.outputs`.
  - Put `create(images_dir, out_dir, model_dir) -> PointcloudResult` as the abstract method on `BasePointcloudCreator`, and drop `reconstruct` there.
  - `run()` stays.
- [ ] **Step 6: Change the wrapper flow.**
  - `_run_feedforward`:
    - `result = creator.create(images_dir, output_dir, model_dir)`, where `model_dir` is passed in from `self.colmap_model_dir`;
    - LC path: `LoopClosure.reconstruct(...)`, then its existing COLMAP write call changes to `creator.write_colmap_model(model_dir)`. Call-site edit only; read `loop_closure/wrapper.py:189-208,650-700` for how it calls `build_colmap` today;
    - `save_zarr(pointcloud_zarr, extra_attrs=...)` stays here;
    - returns `result`, a `PointcloudResult`.
  - `build_pointcloud`:
    - SOR clean block: keep it this task, now on `result.points` / `result.colors`, `pixel_indices` through `dataclasses.replace` on the mask; Task 4 moves it.
    - `result.write_ply(self.backend_dir / "sparse_pc.ply")`.
  - `_run_sfm`: `outputs, align_attrs = result_from_reconstruction(...)`; `write_colmap_reconstruction(recon, self.colmap_model_dir)` stays in the creator via `write_sfm_model`; return `outputs`.
  - `_load_pointcloud_from_disk` → `return PointcloudResult.load_zarr(self.pointcloud_zarr)`. Keep the load flags minimal: no images, no world_points, unless a caller needs them.
  - `refine_poses`:
    - BA uses `ff.model_intrinsics`;
    - after BA, `dataclasses.replace(ff, extrinsics=..., model_intrinsics=...)`, then recompute full-res `intrinsics` with the same two lines. Extract no helper: the ADR-017 two-caller rule and the no-wrapper rule both apply, and this is the second caller. Name it only if the reviewer asks; the contract says never a wrapper;
    - write `result.to_colmap()` through `write_colmap_reconstruction(..., self.colmap_model_dir)`;
    - store both Ks in the zarr.
  - `mesh` / `_run_tsdf_mesh` / `splats` / `reconstruction_quality_report` / `_lift_and_save` / `_build_localization_db`: every `FeedforwardResult.load_zarr` → `PointcloudResult.load_zarr`; K per the K rule above.
- [ ] **Step 7: Update the remaining callers and tests.**
  - Update every remaining import site: dashboard, localization, evals and tests. Run `grep -rn "FeedforwardResult\|from_colmap\|build_colmap\|write_colmap\b\|\.reconstruct(" collab_splats evals tests`. Expect 0 hits, except LC's own `LoopClosure.reconstruct` method name.
  - Update `$SP/parity.py`: `FeedforwardResult` → `PointcloudResult` (import from `collab_splats.pointcloud.base`).
- [ ] **Step 8: Run the gates.**
  - New tests pass; `unify_T3_k_eq.py` prints all-equal; G′; side suite.
  - Parity: expected moves are only `intrinsics` (now full-res; baseline had model-grid) and the new `model_intrinsics` field. Verify `model_intrinsics` equals the baseline `intrinsics` bit-for-bit with a 5-line check in `unify_T3_k_eq.py`. Then `--save` into `parity_baseline_unify`.
- [ ] **Step 9: Commit.** `refactor(pointcloud): one PointcloudResult; K stored full-res + model grid; COLMAP export-only (P7)`. The body has the K-site table and the parity diff.

### Task 4: SOR clean moves into the creators

**Files:** `feedforward/base.py` (`postprocess`, new `clean: bool = True` field), `wrapper/reconstructor.py` (`build_pointcloud` clean block deleted; `_run_feedforward` passes `clean=pc_cfg["clean"]["enabled"]` into the creator kwargs), the `_run_sfm` sfm path, tests.

- [ ] **Step 1: Write the failing test** (`tests/pointcloud/test_base.py`): after `postprocess()` with `clean=True` on a fixture holding 1 far outlier among 200 clustered points, the outlier is gone from `outputs.points`, `colors` and `pixel_indices`, and the three lengths agree. With `clean=False`, all points are kept.
- [ ] **Step 2: Implement.**
  - At the end of `postprocess()`, after the K block:

```python
        # Statistical outlier removal on the final cloud; per-point arrays filtered together
        if self.clean:
            keep = clean_pointcloud(self.outputs.points)
            self.outputs = dataclasses.replace(
                self.outputs,
                points=self.outputs.points[keep],
                colors=self.outputs.colors[keep],
                pixel_indices=self.outputs.pixel_indices[keep],
            )
```

  - Check the `PointcloudResult` fields at HEAD: every per-point array must be filtered by `keep`. Grep for fields shaped `(P, ...)`.
  - sfm (`_run_sfm`): before `write_sfm_model`, drop the cleaned-out `points3D` from `recon` (the loop that `build_pointcloud` has today). Also filter `outputs` points when `pc_cfg["clean"]["enabled"]`.
    - Today `build_pointcloud` cleaned the recon's `points3D` (sparse SfM points), while the zarr holds dense depth points.
    - Keep that meaning: clean `recon.points3D` before the write, and clean the zarr `points` separately with the same `clean_pointcloud`.
    - State this in the commit body.
  - Delete the clean block from `build_pointcloud`.
  - The LC wrapper goes through `postprocess` and gets the clean for free. Verify by reading `loop_closure/wrapper.py`: if the LC wrapper calls `_postprocess` directly instead of `postprocess`, stop and report.
- [ ] **Step 3: Run the gates.**
  - G′, side suite, parity.
  - Expected parity moves: `points`, `colors`, `pixel_indices` (and `mv_*` per-point, if any) of the ff cases whose synthetic cloud has SOR outliers.
  - Chain-save.
- [ ] **Step 4: Commit.** `fix(pointcloud): SOR clean inside the creator — zarr, COLMAP, PLY hold one point set`. The body names the latent bug fixed and the parity diff.

### Task 5: Refine re-cleans after reproject

**Files:** `wrapper/reconstructor.py:refine_poses`, `tests/wrapper/test_refine_stage.py`

- [ ] **Step 1: Write the failing test.** After `refine_poses` on the existing refine fixture plus one injected far depth pixel, `pointcloud.zarr` points, the COLMAP model's `points3D` count and `sparse_pc.ply` all hold the same cleaned count.
- [ ] **Step 2: Implement.**
  - After `.reproject()`, apply the same keep-filter block as Task 4 (`clean_pointcloud`, then filter the per-point arrays). This is its second caller; if the reviewer flags duplication, extract a `PointcloudResult.select(keep)` method used by both.
  - Then `subsample_points`, as `_mask_to_points` did, when `len(points) > max_points`.
  - Then the zarr rewrite (`points`, `colors`, `pixel_indices` resized: zarr arrays of a new length must be rewritten, not slice-assigned), `to_colmap` write, PLY.
- [ ] **Step 3: Run the gates.** G′, side suite, parity (refine is not in parity; expect bit-exact).
- [ ] **Step 4: Commit.** `fix(wrapper): refine re-cleans and re-caps the cloud after reproject`.

### Task 6: `pointcloud/depth.py`

**Files:**
- Create: `collab_splats/pointcloud/depth.py`, `tests/pointcloud/test_depth.py` (from `test_vda.py` + `test_depth_align.py`, `git mv` then edit)
- Delete: `pointcloud/vda.py`, `pointcloud/depth_align.py`
- Modify: `wrapper/reconstructor.py:_run_sfm`, `sfm/instantsfm.py` docstrings, `evals/scripts/eval.py`, `$SP/parity.py` (`depth_align` → `depth`, `result_from_reconstruction` → `align_depth`)

- [ ] **Step 1: Move the tests first.**
  - `git mv tests/pointcloud/test_depth_align.py tests/pointcloud/test_depth.py`; append `test_vda.py`'s tests; `git rm tests/pointcloud/test_vda.py`.
  - Rename the imports to `from collab_splats.pointcloud.depth import _load_vda_model, align_depth, estimate_depth`.
  - Add one test: `estimate_depth` with a partial `depth_vda/` cache (one stem missing) wipes the dir and regenerates. Monkeypatch `_load_vda_model` with the stub the old vda test used.
- [ ] **Step 2: Confirm they fail** (ModuleNotFoundError).
- [ ] **Step 3: Implement the merge.**
  - `depth.py` is `vda.py` + `depth_align.py` concatenated under `########` dividers: VDA model, estimate, align.
  - `generate_vda_depth` → `estimate_depth`. Its body starts with the cache-miss wipe that `_run_sfm` does today:

```python
    # A partial cache is dropped whole: estimation only adds maps
    if not _depth_cache_complete(out_dir, names):
        shutil.rmtree(Path(out_dir) / "depth_vda", ignore_errors=True)
```

    `vda_depth_complete` becomes private `_depth_cache_complete`; its one caller is here.
  - `result_from_reconstruction` → `align_depth`. It sets `intrinsics` = COLMAP K at full res (`camera.calibration_matrix()`) and `model_intrinsics` = K rescaled to the depth grid (today's single `intrinsics`).
  - Replace its numpy homogeneous transforms with `transforms.transform_points` where one exists: grep `np.einsum\|@ .*\.T\|hstack.*ones` in the moved code. Bit-exact is required: if `transform_points` changes bits, keep the numpy form and note it.
  - `_run_sfm` loses its wipe block and calls `estimate_depth`.
  - `eval.py` imports follow.
- [ ] **Step 4: Run the gates.** Tests pass; G′ (test ids renamed → `IDS DIFFER` expected only by the renamed test file path; list the mapping in the commit body); side suite; parity bit-exact after the import rename in `parity.py`.
- [ ] **Step 5: Commit.** `refactor(pointcloud): depth.py merges vda + depth_align (estimate_depth, align_depth)`.

### Task 7: sfm skeleton — `sfm/base.py`; `_run_sfm` and eval shrink; provenance removed

**Files:**
- Create: `collab_splats/pointcloud/sfm/base.py`, `tests/pointcloud/sfm/test_base.py`
- Delete: `collab_splats/pointcloud/sfm/common.py`, `tests/pointcloud/sfm/test_common.py` (surviving tests move into `test_base.py`)
- Modify:
  - `sfm/{instantsfm,colmap,hloc}.py`: `reconstruct` → `_map`; `provenance` deleted; `HLOC_PIN` deleted
  - `sfm/__init__.py` (dict unchanged)
  - `wrapper/reconstructor.py`: `_run_sfm` shrinks; `_registered_rows` moves out; `_SFM_BLOCK_KEYS` special case goes
  - `evals/scripts/eval.py`: `_run_instantsfm` → `creator.create`; PIL → `read_image`
  - `docs/superpowers/decisions/018-*.md`: superseding note on provenance
  - tests referencing `provenance`, `HLOC_PIN`, `pycolmap_version`, `hloc_commit`

- [ ] **Step 1: Write the failing tests** in `tests/pointcloud/sfm/test_base.py`, with a fake backend whose `_map` returns a pycolmap model built in-test:
  - `test_create_writes_model_and_returns_aligned_result`: the model dir exists with stem names; the result has `len(image_paths)` equal to the registered count and `attrs["method"] == "sfm"`.
  - `test_strict_base_refuses_partial_model`: base (instantsfm semantics) raises when 1 of 3 frames is unregistered.
  - `test_incremental_floor`: `_IncrementalSfmCreator(min_registered_frac=0.5)` keeps 2 of 3 with a warning and raises below the floor. `min_registered_frac=1.5` raises in `__post_init__`.
  - `test_clean_drops_points3d_before_write`: one far `points3D` outlier is absent from the written model.
  - Monkeypatch `estimate_depth` to return constant depth, so no VDA runs.
- [ ] **Step 2: Confirm they fail.**
- [ ] **Step 3: Implement `sfm/base.py`:**

```python
@dataclass
class BaseSfmCreator(BasePointcloudCreator):
    """
    SfM skeleton: VDA depth, backend mapper, registered subset, clean, COLMAP write, depth align.

    - backends implement _map only
    - strict: every frame must register (instantsfm); _IncrementalSfmCreator relaxes it

    Attributes:
        clean: SOR-clean the mapper's points3D and the aligned cloud.
    """

    clean: bool = True
    attrs: dict = field(default_factory=dict, init=False, repr=False)

    def create(self, images_dir: Path, out_dir: Path, model_dir: Path) -> PointcloudResult:
        """
        ...Args/Returns per contract...
        """
        # Keyframe names; VDA metric depth, cached by stem
        names = [p.name for p in frames.frame_paths(images_dir)]
        depths = estimate_depth(frames.read_frames(images_dir), out_dir, names)

        # Backend mapper; frees its models before the align
        recon = self._map(images_dir, out_dir, names)
        pytorch_gc()

        # Registered subset from the in-memory model
        rows = self._registered_rows(recon, names)
        subset_attrs = {} if len(rows) == len(names) else {"registered_frames": len(rows), "total_frames": len(names)}
        names = [names[row] for row in rows]

        # SOR clean on the mapper's points, then the COLMAP write
        if self.clean:
            ...delete cleaned-out point3D ids (the block Task 4 put in _run_sfm)...
        write_colmap_reconstruction(recon, model_dir)

        # Dense result at depth resolution in the COLMAP world
        keyframes = frames.read_frames(images_dir, [frames.frame_idx_from_path(n) for n in names])
        depths = depths[rows]
        result, align_attrs = align_depth(recon, depths, keyframes, names)
        self.attrs = {"method": "sfm", **subset_attrs, **align_attrs}
        return result

    @abstractmethod
    def _map(self, images_dir: Path, out_dir: Path, names: list[str]) -> pycolmap.Reconstruction:
        """
        Backend mapper: one model over `names`, image names as in images_dir.
        """

    def _registered_rows(self, recon: pycolmap.Reconstruction, names: list[str]) -> list[int]:
        """
        Rows of `names` the model registered; strict — every frame or ValueError.
        """
```

  - Implementation notes:
    - No clean after `align_depth` (review of Task 4):
      - the zarr points are built from the already-cleaned `recon.points3D`
      - SOR is not idempotent, so a second pass would drop more points and split zarr from COLMAP
    - Drop track-less points3D before the SOR (review of Task 4):
      - `align_depth` skips observation-less points (InstantSfM sub-min-track-length exports), the COLMAP write keeps them
      - deleting them first makes the COLMAP and zarr counts match, and keeps them out of the SOR neighborhoods
    - Before writing, check the `subset_attrs` rule against today's `_run_sfm`: today non-instantsfm always writes `registered_frames` / `total_frames`. Keep today's behavior exactly: `_IncrementalSfmCreator` always sets them; the base never does.
    - `_IncrementalSfmCreator(BaseSfmCreator)`:
      - `min_registered_frac: float = <today's default in configs/base.yaml>`;
      - `__post_init__` range check;
      - `_registered_rows` = the wrapper's `_registered_rows` moved verbatim (floor + warning), reading registered names from `recon.reg_image_ids()` → `recon.images[i].name` stems.
    - `prepare_sfm_dirs`, the largest-model pick and `sfm_image_dir` fold in where their one caller is. Read `common.py` for each and inline it at that caller.
    - `ColmapCreator` / `HlocCreator` subclass `_IncrementalSfmCreator`; `InstantSfMCreator` subclasses `BaseSfmCreator`. Each `reconstruct` body becomes `_map` minus the `write_sfm_model` call, returning the recon.
  - Wrapper:

```python
    def _run_sfm(self) -> PointcloudResult:
        # Mapper + depth align; the creator writes the COLMAP model
        pc_cfg = self.config["pointcloud"]
        backend = pc_cfg["backend"]
        creator = SFM_CREATORS[backend](clean=pc_cfg["clean"]["enabled"], **pc_cfg[backend])
        result = creator.create(self.images_dir, self.backend_dir, self.colmap_model_dir)
        result.save_zarr(self.pointcloud_zarr, extra_attrs={"backend": backend, **creator.attrs})
        return result
```

  - Delete `provenance()` from all three, `HLOC_PIN`, and their tests.
  - ADR 018: append the section `## Superseded in part (2026-09-27, pointcloud-unify)` with one bullet: provenance attrs removed (written, never read). Keep everything else.
  - `eval.py:_run_instantsfm` → `InstantSfMCreator(...).create(images_dir, out_dir, out_dir / "colmap" / "sparse" / "0")`. Its PIL decode → `read_image`.
- [ ] **Step 4: Run the gates.** G′ (renamed and deleted test ids listed in the body), side suite, parity (the `depth` case is bit-exact).
- [ ] **Step 5: Commit.** `refactor(pointcloud): sfm/base.py owns the sfm skeleton; provenance removed`.

### Task 8: InstantSfM in-memory export

**Files:** `sfm/instantsfm.py`, `tests/pointcloud/sfm/test_instantsfm.py`

- [ ] **Step 1: Write the failing test first.**
  - `_to_pycolmap(cameras, images, tracks)` on InstantSfM-shaped inputs must give the same model as today's `_patch_instantsfm_colmap_write` + `WriteGlomapReconstruction` + read-back: same cameras (model, params), image names, poses (bit-exact), `points3D` xyz/color/track elements.
  - Build the inputs with the upstream types. Read `third_party/InstantSfM` (or wherever `instantsfm` imports from) for `Cameras` / `Images` / `Tracks`.
  - The old path runs in-test to produce the expected model. The test stays, as the equality proof, until the old path is deleted in step 3. Then the expected model is frozen to a committed tiny `.npz`, or rebuilt from a hand-written dict; pick the smaller.
- [ ] **Step 2: Implement `_to_pycolmap`.**
  - Keep cluster-0 selection (ADR 018).
  - `_to_pycolmap` may carry track-less points3D; `BaseSfmCreator.create` drops them before the SOR (Task 7).
  - Delete `_patch_instantsfm_colmap_write`, the `WriteGlomapReconstruction` call and the read-back.
  - Fix the stale "verify stage" comment.
- [ ] **Step 3: Run the gates.** G′, side suite, parity.
- [ ] **Step 4: Commit.** `refactor(sfm): InstantSfM result converted in memory, no write + read-back`.

### Task 9: `center_crop_coords` shared by VGGT-X / Omega / MapAnything

**Files:** `feedforward/base.py` (new public function next to `full_frame_coords`), `vggtx.py`, `vggt_omega.py`, `mapanything.py`, `tests/pointcloud/feedforward/test_center_crop_coords.py`

- [ ] **Step 1: Write the test.**
  - It holds verbatim copies of the three current functions (`_compute_vggtx_crop_coords`, `_compute_omega_original_coords`, `_mapanything_crop_coords` from `1ac95ee0`) as private `_old_*` references. They are test-only frozen oracles, with a comment citing the commit.
  - Sweep: all sizes `w, h in range(64, 4097, 97) × range(64, 4097, 89)` plus common frames (1920×1080, 1080×1920, 3840×2160, 640×480, 518×518). That is ~3k sizes.
  - MapAnything runs on its 4 grids (grep its resolution table).
  - Assert `np.array_equal` new vs old, dtype included.
- [ ] **Step 2: Implement.**
  - The helper, adapted from lane C `a6fd4ed6` (`git show a6fd4ed6:collab_splats/geometry/transforms.py`, `center_crop_box`) but in `feedforward/base.py`:

```python
def center_crop_coords(
    orig_wh: tuple[float, float],
    resized_wh: tuple[float, float],
    crop_wh: tuple[float, float],
    scale: tuple[float, float],
) -> list[float]:
    """
    Centered crop of a resized frame, as a box in original pixels.

    - resize orig -> resized, then center-crop resized -> crop; the box inverts both
    - integer offsets in the resized grid, (resized - crop) // 2, as every upstream loader crops

    Args:
        orig_wh: original frame (width, height).
        resized_wh: frame size after the resize, before the crop.
        crop_wh: model input size after the crop.
        scale: (sx, sy) original -> resized.

    Returns:
        [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
    """
    ow, oh = orig_wh
    rw, rh = resized_wh
    cw, ch = crop_wh
    sx, sy = scale

    # Integer center offsets in the resized grid, mapped back to original pixels
    left, top = (rw - cw) // 2, (rh - ch) // 2
    return [left / sx, top / sy, (left + cw) / sx, (top + ch) / sy, ow, oh]
```

  - Each backend keeps a loop with only its resized-size rule, then `np.array(rows, dtype=np.float32)`:
    - VGGT-X: `resized=(518, new_h)`, `crop=(518, min(new_h, 518))`, `scale=(518/w, new_h/h)`;
      - check the no-crop branch: `cr_y = orig_h`, and with `new_h <= 518`, `top = (new_h - new_h)//2 = 0`, `cr_y = new_h / (new_h/h) = h`. The float path can differ by 1 ulp, and the sweep test decides;
    - Omega: `resized=(w, h)`, `crop=(crop_width, crop_height)`, `scale=(1, 1)`;
    - MapAnything: `scale=(s, s)` with its `+ 1e-8`, `resized=(floor(w*s), floor(h*s))`, `crop=(model_w, model_h)`.
  - If any backend is not bit-exact, keep that backend's old function, note it in the body, and still land the other two.
  - Land `d19abb0e`'s docstring wording into VGGT-X's rule comment where it still applies.
- [ ] **Step 3: Run the gates.** Sweep test passes; G′; parity bit-exact; lines net-negative (`git diff --stat HEAD~1` excluding the test).
- [ ] **Step 4: Commit.** `refactor(pointcloud): center_crop_coords shared by the three center-crop backends`.

### Task 10: Feedforward boilerplate

**Files:** `feedforward/{base,vggtx,vggt_omega,mapanything,loger}.py`

Keep each sub-item only if it is net-negative; else drop it and say so.

- [ ] **Step 1: Synthetic `image_paths` ×4.** Grep `Path(f"frame_` / `frame_name(` in `_preprocess` implementations and hoist the shared line into `setup_inference`, if all 4 are identical.
- [ ] **Step 2: `extract_intermediate_features` near-twins (vggtx / omega).** Diff them. If they differ only in the attribute path of the aggregator, move the body to `BaseFeedforwardCreator` with a ClassVar for that difference.
- [ ] **Step 3: 4 numpy transform sites in `feedforward/base.py` → `transforms.transform_points`.** These are `_raw_to_world_points`, `_frustum_world_aabbs` and two others; grep `@ .*T +\|einsum`. Keep only the bit-exact ones (parity decides).
- [ ] **Step 4: `_decode_dir_to_frames`: `PIL.Image.open` → `read_image`.** It must be bit-exact; `read_image` ignores EXIF and the old path did too (verify).
- [ ] **Step 5: `get_device` / `pytorch_gc` sites.** Apply the phase-2 deferred rows: `grep -n "Deferred" -A40 docs/superpowers/plans/2026-09-26-consistency-phase2.md`, apply the pointcloud rows only.
- [ ] **Step 6: Run the gates and commit.** G′, parity bit-exact, net-negative. `refactor(pointcloud): feedforward boilerplate dedup`.

### Task 11: `lift_features` → `semantics/lifting.py`

**Files:**
- Create: `collab_splats/semantics/lifting.py`
- Modify:
  - `pointcloud/utils.py`: remove `lift_features`, `_grid_sample_at_pixels`, `_sample_at_source_pixels`
  - `wrapper/reconstructor.py` import; the `lift_features` tests (`git mv tests/pointcloud/test_feature_lifting.py tests/semantics/test_lifting.py`)
  - `tests/test_docstring_contract.py`: semantics is already in PACKAGES; confirm.
- [ ] **Step 1: Move the code.** The body must be AST-equal: prove it with `$PY -c "import ast,…"` comparing `ast.dump` of the three defs before and after, and paste the output into the body.
- [ ] **Step 2: Check for import cycles.** `semantics` must not import `wrapper`. Check with `$PY -c "import collab_splats.semantics.lifting"`.
- [ ] **Step 3: Run the gates and commit.** G′ (renamed test path listed); side suite (it now includes the moved test); parity. `refactor(semantics): lift_features lives in semantics/lifting.py`.

### Task 12: Lane B + P1 caller migration

**Files:**
- Cherry-pick content of `7a6ce23a`, `f338b732`, `c2202a71` (`git cherry-pick -n` each, resolve, commit as one or three).
- Then migrate the callers:
  - `_raw_to_world_points`, `_verify_geometry`, the multiview loop, `_frustum_world_aabbs`;
  - `depth.align_depth`;
  - `reproject_pixels`: deleted; `PointcloudResult.reproject` uses `project` with `model_intrinsics`;
  - `semantics/lifting.lift_features`, `geometry/metrics` NCC, `reconstructor` refine unproject, BA near :674;
  - `evals/scripts/{analyze_splats,refit_at_fixed_poses}.py`.
- Exempt: `mesh/texture.py`, `mesh/tsdf.py`.

- [ ] **Step 1: Cherry-pick lane B** as 3 commits, keeping the subjects.
  - After `c2202a71`, confirm `transforms.py`'s hand-formatted matrix survived: `git show c2202a71 -- collab_splats/geometry/transforms.py` shows only annotation changes.
  - Run G′ + side + parity after each. `f338b732` deletes `splats/pgsr.py`; its known-failures edits must apply cleanly.
- [ ] **Step 2: Write the P1 test first.**
  - `tests/geometry/test_projection.py` gains `test_project_unproject_round_trip_non_identity` if missing: non-identity poses, non-square cropped K.
  - Each migrated caller gets an old-vs-new check in a scratch script (`$SP/unify_T12_p1_eq.py`), with old code copied from HEAD.
  - Expected: allclose at 1e-6 and not bit-exact (torch vs numpy). This is the parity-moving part.
- [ ] **Step 3: Migrate the callers, one commit** `refactor(geometry): P1 — projection callers use unproject/project`.
  - Parity moves are allowed on `points`, `world_points`, `colors`, `pixel_indices`, `mv_*`, and `depth`-case fields.
  - Paste the diff; chain-save.

### Tasks 13–16: P3, P2, P4, P8 — one commit each

Each task follows the same steps:
1. Write a failing unit test on a fixture that can see the change.
2. Implement.
3. Run G′, the side suite, and parity (paste expected-field moves, chain-save).
4. Commit.

**Task 13 — P3 `subsample_points(mask, max_points, seed=0) -> mask`**
- Replaces `_limit_trues` (`feedforward/base.py:~396`) and the array form (`pointcloud/utils.py:98`).
- Callers:
  - `_mask_to_points`;
  - LC wrapper :504, :665 (call-site edit: `mask = subsample_points(np.ones(len(points), bool), max_points)`, then index);
  - refine (Task 5).
- Test: the mask form returns ≤ max_points Trues, all within the input Trues, and is deterministic under the same seed.
- Subject: `refactor(pointcloud): one mask-form subsample_points (P3)`.

**Task 14 — P2 clean before subsample**
- In `postprocess`, the order becomes conf mask → SOR clean → subsample.
- This means `unproject_and_filter_points` stops capping, and the cap applies after the clean in `postprocess`. Also apply it in `BaseSfmCreator.create` and refine.
- LC `_assemble_result` subsamples first, then `LoopClosure.postprocess` cleans:
  - either move LC to clean → subsample too, adding it to the callers above
  - or keep its order for the `to_colmap` memory cap and record why in the commit body
- Test: on a fixture with outliers and `max_points` < n, the output count == `max_points` (today it is fewer after the clean) and no outlier survives.
- Subject: `fix(pointcloud): clean before subsample (P2)`. The body lists the risk-4 consequences (splat seed and lifted semantics move; TSDF and localization unchanged).

**Task 15 — P4 `projection.depth_residual`**
- One cross-view depth test, shared by `compute_multiview_depth_confidence` and `lift_features`' visibility check.
- Test: known residuals on a two-view fixture with a non-identity relative pose.
- Subject: `refactor(geometry): depth_residual shared by multiview + lifting (P4)`.

**Task 16 — P8 SIFT DB reuse**
- `ensure_sift_database` reuses iff the DB image names == `names` AND a params row stored in the DB equals the call's params.
- Store the params in a table the function creates (`CREATE TABLE IF NOT EXISTS collab_params (json TEXT)` via `sqlite3`).
- Delete the `<db>.json` sidecar code and its tests.
- Test: same names + same params → no rebuild (monkeypatch `build_sift_database` counter); changed `overlap` → rebuild; a legacy DB with no params table → rebuild.
- Subject: `fix(sfm): SIFT DB reuse keyed on DB contents, sidecar deleted (P8)`.

### Task 17: C2 docstring, smoke, CHANGELOG

- [ ] **Step 1: C2 docstring.** If Task 9 did not already absorb `d19abb0e`'s wording, apply it (`git cherry-pick -n d19abb0e`, resolve against the new VGGT-X rule) and commit `docs(pointcloud): …`.
- [ ] **Step 2: Real smoke, in tmux, no other heavy process running.**
  - Find the C0043 input path: `grep -rn C0043 configs docs/superpowers/CHANGELOG.md | head`.
  - Run `run_pipeline.py` with an overrides file: `semantics: {enabled: false}`, stages `preproc,pointcloud,mesh,localize,reconstruction_quality_report`, output under `$SP/smoke_ff`.
  - Repeat with `pointcloud: {method: sfm, backend: instantsfm}` under `$SP/smoke_sfm`.
  - Check that every artifact exists: `pointcloud.zarr` has `model_intrinsics`; `colmap/sparse/0` holds 3 `.bin`; `sparse_pc.ply`, `mesh.ply` and the report json exist.
  - Check that the COLMAP `points3D` count == the PLY vertex count on the ff path.
  - No quality judgment.
- [ ] **Step 3: CHANGELOG.** Append a `pointcloud-unify` entry to `docs/superpowers/CHANGELOG.md`: what landed, the commit range, and the parity chain (`parity_baseline_unify`, never re-saved over the original). Update the CLAUDE.md in-flight row to "landed on clean/pointcloud-release, not merged". Commit `docs(changelog): pointcloud-unify`.
- [ ] **Step 4: Stop.** No merge, no push. Report to the user.
