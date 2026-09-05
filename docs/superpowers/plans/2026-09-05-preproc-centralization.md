# preproc Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `frames.zarr` with a COLMAP-style `images/` directory, make the quality filter actually filter, and replace three hand-rolled subsystems (undistortion framing, ffmpeg decode, MAD) with library calls — cutting `collab_splats/preproc/` from 2268 to ~1485 lines.

**Architecture:** Five sequential phases. A swaps the storage format behind a flat function API with byte-identical frame selection. B changes which frames are selected and is gated on an A/B measurement. C and D replace undistortion and decode with pycolmap and PyAV. E deletes dead code and rewrites docstrings. Each phase leaves the test suite green and the dashboard smoke-passing.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), OpenCV, pycolmap 4.0.4, PyAV 17.0.1, scipy, numpy, pytest.

**Spec:** [2026-09-05-preproc-centralization-design.md](../specs/2026-09-05-preproc-centralization-design.md)

---

## Ground Rules

Read these before Task 1. They apply to every task.

- **Python is `/opt/venv/reconstruction/bin/python`.** The base shell `python` is 3.13 and wrong for this project. Every command in this plan spells the interpreter out.
- **Format before every commit:** `black . && isort .` — but never repo-wide `black` on unrelated files; stage only what the task touched.
- **`docs/superpowers/` is gitignored.** Committing anything under it needs `git add -f`.
- **Commit with `git commit --only <paths>`**, never bare `git commit -a`. Other sessions share this working tree and a bare commit sweeps their staged work.
- **The dashboard smoke gate is mandatory** before any commit that touches `collab_splats/dashboard/`: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke`.
- **Two hard gates block phases.** Phase B does not land until Task 14 (the GH010229 sampling A/B) passes and the user has read it. Phase D does not land until Task 19 (the rotated-video fixture) exists and is green against the current ffmpeg code. Do not skip them.
- **Colour convention:** every function in `preproc/frames.py` takes and returns **RGB**. `cv2` reads and writes BGR, so conversions happen inside `frames.py` and nowhere else.

---

## Parallel Execution Schedule

The 28 tasks are not a straight line. Their file sets are disjoint in two big
places, and exploiting that collapses the critical path from 28 slots to ~18.

**Every wave runs in its own git worktree, one per task.** The main checkout at
`/workspace/collab-splats` is shared with other sessions and its git index is
shared with every worktree — two agents committing there at once sweep each
other's staged AND unstaged work. A worktree gives each agent its own index, so
broad `git commit --only <dir>` paths become safe again.

### Worktree protocol

```bash
# integration branch, forked once per wave from the previous wave's merge
git worktree add /workspace/collab-splats/.worktrees/preproc-t<N> -b preproc/t<N> <base-sha>
```

Inside a worktree, **`PYTHONPATH` is mandatory**. The venv installs
`collab_splats` editable through a finder that hardcodes
`/workspace/collab-splats`, so a bare `pytest` in a worktree silently tests the
MAIN tree's code and reports success on work that was never applied:

```bash
cd /workspace/collab-splats/.worktrees/preproc-t<N>
PYTHONPATH=/workspace/collab-splats/.worktrees/preproc-t<N> \
  /opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v
```

Every `pytest`, every `python -c`, every dashboard smoke run inside a worktree
carries that prefix. A task that reports green without it has verified nothing.

### The waves

| wave | tasks | notes |
|---|---|---|
| **0** | **25, 26, 19, 1, 24a** | fully independent; nothing here depends on anything else |
| **1** | **2, 4, 5, 6, 7, 8** | all need Task 1's `frames.py`; file sets disjoint from each other |
| 2 | 3 (reference, no code), 9 | 9 needs all of 4-8 merged |
| 3 | 10 -> 11 -> 12 -> 13 -> 14 | serial: one file, each builds on the last's helpers |
| 4 | 15 -> 16 -> 17 -> 18 | serial: `undistort.py` then `reconstructor.py` |
| 5 | 20 -> 21 -> 22 -> 23 -> 24b | serial: one file, cumulative |
| 6 | 27 -> 28 | 27 sweeps every module the earlier waves touched |

**Task 25 moves to wave 0, and this is not only a scheduling choice.**
`preproc/viz.py` imports `FrameStore` and `plot_quality_examples` is its only
consumer. No Phase A task converts `viz.py` — Task 7 is semantics/geometry/mesh,
Task 8 is dashboard/evals/notebooks — so Task 9's step-1 verification grep would
return `collab_splats/preproc/viz.py` and block. Deleting the function first
closes the gap for free. `plot_selection` and `plot_frame_extremes` never touch
the store.

**Task 24 splits.** 24a declares `av` in `pyproject.toml` and runs at wave 0 —
it already resolves transitively, so declaring it early breaks nothing and makes
wave 5's imports legal. 24b sweeps the dead `info=` and needs Tasks 21 and 22.

### What cannot be parallelised, and why

Four single-file chains, each task consuming the previous one's helpers:
`sampling.py` (10-13), `video.py` (20-23), `undistort.py` (15-16).
`reconstructor.py` is the cross-lane lock — Tasks 4, 13, 17 and 18 all edit it
across three different waves, so those four never overlap.

Tasks 15 and 16 have no file overlap with the wave-3 chain and could run beside
it. They stay after Task 14 anyway: the gate exists so a human approves the
selection change before more breaking work lands, and jumping it buys one slot.

### Merging a wave

Each task's branch merges back into the wave's integration branch, in task
order, then the full suite and the dashboard smoke run **once** on the merged
result before the next wave forks. Individual task branches are green in
isolation; only the merge proves they are green together.

Serialise the heavy ones even within a wave: Task 8 re-runs six notebooks and
binds the dashboard smoke port, and Task 15's tests run real SIFT plus
incremental mapping. The container cap is 46.6 GB.

---

## File Structure

**Created:**

| file | responsibility |
|---|---|
| `collab_splats/preproc/frames.py` | The `images/` + `frames.json` store: write, read, list paths, read manifest. Replaces `frame_store.py`. |
| `scripts/migrate_frames_zarr.py` | One-shot converter: existing `frames.zarr` -> `images/` + `frames.json`, no video decode. |
| `tests/preproc/test_frames.py` | Directory-store tests. Replaces `test_frame_store.py`. |
| `tests/preproc/data/make_rotated_fixture.py` | Generates `tests/preproc/data/rotated_90.mp4`, the fixture phase D is gated on. |
| `tests/preproc/test_docstrings.py` | Lints the Args/Returns contract across `preproc.__all__`. |

**Deleted:**

| file | reason |
|---|---|
| `collab_splats/preproc/frame_store.py` | Replaced by `frames.py`. |
| `tests/preproc/test_frame_store.py` | Replaced by `test_frames.py`. |
| `tests/preproc/test_sampling_parity.py` | Asserts the window-argmax substitution the eligible pool replaces; its premise is deleted. |

**Heavily modified:**

| file | change |
|---|---|
| `collab_splats/preproc/sampling.py` | `filter_frame_quality` rewritten; `_sample_by_quality` deleted; three samplers take an eligible pool; gains `context_indices`. |
| `collab_splats/preproc/undistort.py` | `DistortionProfile` deleted; `calibrate_camera` + `undistort_frames` over `pycolmap.Camera`. |
| `collab_splats/preproc/video.py` | ffmpeg decode subprocesses replaced by PyAV; `context_indices` and `decode_context` move out. |
| `collab_splats/preproc/qa.py` | Three pair-motion functions collapse into `compute_pair_motion`. |
| `collab_splats/preproc/viz.py` | Two dead plots deleted. |
| `collab_splats/wrapper/reconstructor.py` | `extract_frames` writes `images/`, then splits by source; `_apply_undistortion` rewritten. |
| `collab_splats/pointcloud/sfm.py` | Gains `decode_context`; `image_path` points at `<scene>/images`. |

---

# Phase A — Storage

**Invariant for this whole phase: frame selection does not change.** Every frame chosen before the phase is chosen after it. Any behavioural difference is a bug introduced here, not an intended change.

---

### Task 1: `preproc/frames.py` — the directory store

**Files:**
- Create: `collab_splats/preproc/frames.py`
- Create: `tests/preproc/test_frames.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/preproc/test_frames.py`:

```python
"""
Directory-backed keyframe store: images/frame_NNNNNN.png + frames.json.
"""

import json

import numpy as np
import pytest

from collab_splats.preproc import frames as fr


def _frames(n=3, h=8, w=12):
    """
    n deterministic RGB frames, each a different flat colour.
    """
    return [np.full((h, w, 3), i * 40 + 5, dtype=np.uint8) for i in range(n)]


def _records(idxs):
    return [{"frame_idx": int(i), "blur_score": float(i) * 1.5} for i in idxs]


def test_frame_idx_from_path_reads_the_padded_stem():
    assert fr.frame_idx_from_path("images/frame_000042.png") == 42


def test_write_then_read_round_trips_rgb(tmp_path):
    images = tmp_path / "images"
    written = fr.write_frames(images, _frames(3), _records([0, 5, 11]), {"method": "uniform"})

    assert [p.name for p in written] == ["frame_000000.png", "frame_000005.png", "frame_000011.png"]

    out = fr.read_frames(images)
    assert out.shape == (3, 8, 12, 3)
    assert out.dtype == np.uint8

    # PNG is lossless and the store is RGB at both boundaries
    np.testing.assert_array_equal(out, np.stack(_frames(3)))


def test_read_frames_selects_by_frame_idx_not_position(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})

    out = fr.read_frames(images, idxs=[11, 0])
    np.testing.assert_array_equal(out[0], _frames(3)[2])
    np.testing.assert_array_equal(out[1], _frames(3)[0])


def test_read_frames_raises_on_an_index_the_directory_does_not_hold(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})

    with pytest.raises(KeyError, match="7"):
        fr.read_frames(images, idxs=[7])


def test_frame_paths_is_sorted_and_extension_filtered(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([11, 0, 5]), {})
    (images / "notes.txt").write_text("ignore me")

    assert [p.name for p in fr.frame_paths(images)] == [
        "frame_000000.png",
        "frame_000005.png",
        "frame_000011.png",
    ]


def test_frame_paths_on_a_missing_directory_is_empty(tmp_path):
    assert fr.frame_paths(tmp_path / "nope") == []


def test_manifest_carries_records_and_provenance(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(2), _records([0, 5]), {"method": "fps", "fps": 2.0})

    manifest = fr.read_manifest(images)
    assert manifest["schema_version"] == 2
    assert manifest["provenance"] == {"method": "fps", "fps": 2.0}
    assert [r["frame_idx"] for r in manifest["frames"]] == [0, 5]

    # frames.json sits beside images/, not inside it
    assert (tmp_path / "frames.json").exists()
    assert not (images / "frames.json").exists()


def test_manifest_converts_nan_to_null(tmp_path):
    images = tmp_path / "images"
    records = [{"frame_idx": 0, "blur_score": float("nan")}]
    fr.write_frames(images, _frames(1), records, {})

    raw = (tmp_path / "frames.json").read_text()
    assert "NaN" not in raw
    assert json.loads(raw)["frames"][0]["blur_score"] is None


def test_write_frames_clears_a_previous_longer_run(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})
    fr.write_frames(images, _frames(2), _records([0, 5]), {})

    assert [p.name for p in fr.frame_paths(images)] == ["frame_000000.png", "frame_000005.png"]


def test_write_frames_rejects_records_without_frame_idx(tmp_path):
    with pytest.raises(ValueError, match="frame_idx"):
        fr.write_frames(tmp_path / "images", _frames(1), [{"blur_score": 1.0}], {})


def test_write_frames_rejects_a_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="against"):
        fr.write_frames(tmp_path / "images", _frames(3), _records([0]), {})


def test_read_manifest_names_the_migration_script_when_absent(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="migrate_frames_zarr"):
        fr.read_manifest(tmp_path / "images")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.preproc.frames'`.

- [ ] **Step 3: Write the implementation**

Create `collab_splats/preproc/frames.py`:

```python
"""
Canonical keyframe store: a COLMAP-style images/ directory plus frames.json.

The preprocess stage decodes a video once and writes images/frame_NNNNNN.png
(lossless, PNG compression 1) beside frames.json, which holds the selection
records and provenance COLMAP has no slot for. Every pixel consumer reads the
directory; path-locked consumers take the directory itself, so nothing stages
a second copy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# The repo's single image-extension listing — reconstructor and feedforward both defer here
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

_MANIFEST_NAME = "frames.json"

# OpenCV's default. Level 9 costs 10x the time for 11% of the size (measured, spec 2.2).
_PNG_COMPRESSION = 1


def _manifest_path(dir) -> Path:
    """
    frames.json, which sits beside the images directory rather than inside it.
    """
    return Path(dir).parent / _MANIFEST_NAME


def _jsonable(value):
    """
    numpy scalar or NaN -> a plain JSON value (NaN becomes null).
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if np.isnan(value) else value
    return value


def frame_idx_from_path(path) -> int:
    """
    Source frame index encoded in a frame_{idx:06d}.<ext> filename.

    Args:
        path: path whose stem ends in the zero-padded source index.

    Returns:
        The source video frame index.
    """
    return int(Path(path).stem.split("_")[-1])


def frame_paths(dir) -> list[Path]:
    """
    Image paths in a frame directory, in filename order.

    Args:
        dir: directory holding frame_NNNNNN.<ext> images.

    Returns:
        Sorted image paths; empty when the directory is missing or holds none.
    """
    dir = Path(dir)
    if not dir.is_dir():
        return []
    return sorted(p for p in dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def write_frames(dir, frames, records, provenance) -> list[Path]:
    """
    Write frames as PNGs and the manifest beside them.

    Args:
        dir: images directory to create; stale frame images in it are removed first.
        frames: RGB uint8 (H, W, 3) frames, one per record.
        records: selection records, each carrying an int 'frame_idx' (source index).
        provenance: descriptive dict stamped into frames.json.

    Returns:
        Written image paths, in record order.
    """
    dir = Path(dir)
    if len(frames) != len(records):
        raise ValueError(f"write_frames: {len(frames)} frames against {len(records)} records")
    if not records or "frame_idx" not in records[0]:
        raise ValueError("write_frames: every record must contain 'frame_idx' (source video index)")

    dir.mkdir(parents=True, exist_ok=True)

    # A re-run selecting fewer frames must not leave the previous run's extras behind,
    # where frame_paths would serve them as if they were this run's selection
    for stale in frame_paths(dir):
        stale.unlink()

    # Store is RGB at the boundary; cv2 writes BGR
    paths: list[Path] = []
    for frame, record in zip(frames, records):
        path = dir / f"frame_{int(record['frame_idx']):06d}.png"
        cv2.imwrite(
            str(path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION],
        )
        paths.append(path)

    # Row-oriented: a reader wants one frame's record, not one column
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "frames": [{k: _jsonable(v) for k, v in record.items()} for record in records],
    }
    _manifest_path(dir).write_text(json.dumps(manifest, indent=2))

    logger.info("frames: wrote %d PNGs to %s", len(paths), dir)
    return paths


def read_frames(dir, idxs=None) -> np.ndarray:
    """
    Read frames from an images directory as one RGB stack.

    Args:
        dir: images directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices to read, in the order given; None reads every
            frame in filename order.

    Returns:
        (N, H, W, 3) uint8 RGB.
    """
    paths = frame_paths(dir)
    if not paths:
        raise FileNotFoundError(f"read_frames: no frame images in {dir}")

    # idxs select by source frame_idx, never by row position — a caller holding a
    # frame_idx from a record must not have to know where it landed in the directory
    if idxs is not None:
        by_idx = {frame_idx_from_path(p): p for p in paths}
        missing = [int(i) for i in idxs if int(i) not in by_idx]
        if missing:
            raise KeyError(f"read_frames: frame_idx {missing[:5]} not in {dir}")
        paths = [by_idx[int(i)] for i in idxs]

    return np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])


def read_manifest(dir) -> dict:
    """
    Selection records and provenance written beside an images directory.

    Args:
        dir: images directory; frames.json sits in its parent.

    Returns:
        {'schema_version', 'provenance', 'frames'}.
    """
    path = _manifest_path(dir)
    if not path.exists():
        raise FileNotFoundError(
            f"read_manifest: {path} not found. A scene written before this format holds "
            "frames.zarr — convert it with scripts/migrate_frames_zarr.py."
        )
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/frames.py tests/preproc/test_frames.py
isort collab_splats/preproc/frames.py tests/preproc/test_frames.py
git commit --only collab_splats/preproc/frames.py tests/preproc/test_frames.py \
  -m "feat(preproc): images/ + frames.json store to replace FrameStore"
```

---

### Task 2: `scripts/migrate_frames_zarr.py`

**Files:**
- Create: `scripts/migrate_frames_zarr.py`
- Test: `tests/preproc/test_frames.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/preproc/test_frames.py`:

```python
def test_migrate_converts_a_zarr_store_without_decoding(tmp_path):
    """
    The migration script reads frames.zarr and writes images/ + frames.json.
    """
    import zarr

    from scripts.migrate_frames_zarr import migrate_scene

    # Build a minimal frames.zarr by hand — the same shape FrameStore.create wrote
    scene = tmp_path / "scene"
    scene.mkdir()
    imgs = np.stack(_frames(3))
    store = zarr.open(str(scene / "frames.zarr"), mode="w")
    store.create_array("images", data=imgs, chunks=(1, *imgs.shape[1:]))
    store.create_array("frame_idx", data=np.array([0, 5, 11]))
    store.create_array("blur_score", data=np.array([1.0, 2.0, 3.0]))
    store.attrs["record_keys"] = ["blur_score", "frame_idx"]
    store.attrs["provenance"] = {"method": "fps", "fps": 2.0}
    store.attrs["schema_version"] = 1

    migrate_scene(scene)

    out = fr.read_frames(scene / "images")
    np.testing.assert_array_equal(out, imgs)

    manifest = fr.read_manifest(scene / "images")
    assert manifest["provenance"] == {"method": "fps", "fps": 2.0}
    assert [r["frame_idx"] for r in manifest["frames"]] == [0, 5, 11]
    assert manifest["frames"][1]["blur_score"] == 2.0
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py::test_migrate_converts_a_zarr_store_without_decoding -v
```

Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.migrate_frames_zarr'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/migrate_frames_zarr.py`:

```python
"""
Convert a scene's frames.zarr into images/ + frames.json with no video decode.

Existing processed scenes hold frames.zarr and no images/. Re-running preproc
would re-decode the source video (98 s cold per scene); this reads the store
instead. The old store is left in place — delete it once the scene reads back.

Usage:
    python scripts/migrate_frames_zarr.py <scene_dir> [<scene_dir> ...]
    python scripts/migrate_frames_zarr.py --all <processed_root>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import zarr

from collab_splats.preproc import frames as fr

logger = logging.getLogger(__name__)


def migrate_scene(scene_dir: Path) -> int:
    """
    Write images/ + frames.json from a scene's frames.zarr.

    Args:
        scene_dir: directory holding frames.zarr.

    Returns:
        Number of frames written.
    """
    scene_dir = Path(scene_dir)
    store_path = scene_dir / "frames.zarr"
    if not store_path.exists():
        raise FileNotFoundError(f"no frames.zarr in {scene_dir}")

    store = zarr.open(str(store_path), mode="r")
    images = store["images"][:]
    keys = list(store.attrs.get("record_keys", ["frame_idx"]))

    # Columnar arrays back into row dicts, one per selected frame
    columns = {k: store[k][:] for k in keys}
    records = [{k: columns[k][row] for k in keys} for row in range(images.shape[0])]

    provenance = dict(store.attrs.get("provenance", {}))
    fr.write_frames(scene_dir / "images", [np.asarray(f) for f in images], records, provenance)
    return int(images.shape[0])


def main(argv=None) -> int:
    """
    CLI entry point.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path, help="scene directories, or a root with --all")
    parser.add_argument("--all", action="store_true", help="treat each argument as a root of scene directories")
    args = parser.parse_args(argv)

    # --all expands each root into the scenes under it that still hold a store
    targets: list[Path] = []
    for arg in args.scenes:
        if args.all:
            targets += sorted(p.parent for p in Path(arg).glob("*/frames.zarr"))
        else:
            targets.append(Path(arg))

    for scene in targets:
        n = migrate_scene(scene)
        logger.info("migrated %s (%d frames)", scene, n)

    logger.info("migrated %d scene(s)", len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Format and commit**

```bash
black scripts/migrate_frames_zarr.py tests/preproc/test_frames.py
isort scripts/migrate_frames_zarr.py tests/preproc/test_frames.py
git commit --only scripts/migrate_frames_zarr.py tests/preproc/test_frames.py \
  -m "feat(preproc): frames.zarr -> images/ migration script"
```

---

### Task 3: The call-site translation table

**Files:**
- Read only: this task produces no code. It is the reference Tasks 4-9 apply.

Every `FrameStore` use in the repo is one of these shapes. Apply the right-hand side mechanically. `fr` is `from collab_splats.preproc import frames as fr`, and `images_dir` is `<scene>/images`.

| old | new |
|---|---|
| `FrameStore.create(z, frames, records, provenance=prov)` | `fr.write_frames(images_dir, frames, records, prov)` |
| `FrameStore.open(z)` | *(delete — there is no handle any more)* |
| `store.images()` | `fr.read_frames(images_dir)` |
| `store.images(idxs)` | `fr.read_frames(images_dir)[idxs]` — **positions**, so index the stack |
| `store.image(i)` | `fr.read_frames(images_dir)[i]`, or `cv2.imread` on `fr.frame_paths(images_dir)[i]` in a loop |
| `store.image_by_frame_idx(fi)` | `fr.read_frames(images_dir, idxs=[fi])[0]` |
| `store.has_frame_idx(fi)` | `fi in {fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)}` |
| `store.frame_indices()` | `np.array([fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)])` |
| `store.record(i)` | `fr.read_manifest(images_dir)["frames"][i]` |
| `store.provenance()` | `fr.read_manifest(images_dir)["provenance"]` |
| `len(store)` | `len(fr.frame_paths(images_dir))` |
| `store.export(tmp)` / `store.export(tmp, ext="jpg")` | *(delete the call and the tmpdir; pass `images_dir` itself)* |
| `FrameStore.frame_idx_from_path(p)` | `fr.frame_idx_from_path(p)` |
| a `FrameStore \| Path` parameter | `Path` |
| `sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})` | `fr.frame_paths(d)` |

**Two traps:**

1. `store.images(idxs)` took **row positions**; `fr.read_frames(dir, idxs=...)` takes **source frame indices**. They are the same list only when every frame was selected. Translate positional calls to `fr.read_frames(dir)[idxs]`, never to the `idxs=` keyword.
2. `read_frames` reads the whole directory each call. A loop that called `store.image(i)` per iteration must hoist one `read_frames` above the loop, or iterate `fr.frame_paths` and `cv2.imread` one path at a time — not call `read_frames` per iteration.

- [ ] **Step 1: Confirm the surface before starting**

```bash
git grep -ln 'FrameStore' -- 'collab_splats/*' 'evals/*' 'scripts/*'
```

Expected: 11 files — `dashboard/localize.py`, `dashboard/pipeline.py`, `geometry/loop_closure/wrapper.py`, `geometry/metrics.py`, `mesh/utils.py`, `pointcloud/feedforward/base.py`, `preproc/__init__.py`, `preproc/frame_store.py`, `preproc/viz.py`, `semantics/features/base.py`, `wrapper/reconstructor.py`, plus `evals/datasets.py`, `evals/scripts/eval_splats.py`, `evals/scripts/eval_verification.py`.

If the list differs, a concurrent session has moved things — reconcile before continuing.

---

### Task 4: `reconstructor.extract_frames` writes `images/`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`extract_frames`, `_apply_undistortion`, `_LazyFrames`, and the `frames_zarr` parameter name)
- Test: `tests/wrapper/test_reconstructor_preprocess.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor_preprocess.py`:

```python
def test_extract_frames_writes_an_images_dir_and_manifest(tmp_path, monkeypatch):
    """
    Image-directory input lands as images/frame_NNNNNN.png + frames.json.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc import frames as fr
    from collab_splats.wrapper.reconstructor import extract_frames

    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"img_{i}.png"), np.full((8, 12, 3), i * 40 + 5, np.uint8))

    scene = tmp_path / "scene"
    scene.mkdir()

    n = extract_frames(src, scene / "images", "uniform", None, None, 10)

    assert n == 3
    assert [p.name for p in fr.frame_paths(scene / "images")] == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]
    assert fr.read_manifest(scene / "images")["provenance"]["method"] == "dir"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py::test_extract_frames_writes_an_images_dir_and_manifest -v
```

Expected: FAIL — `extract_frames` still writes a zarr store, so `frame_paths` returns `[]`.

- [ ] **Step 3: Rename the parameter and swap the writer**

In `collab_splats/wrapper/reconstructor.py`:

1. Rename the `frames_zarr: Path` parameter of `extract_frames` to `images_dir: Path` throughout the function, and at every call site inside the module. The sibling paths it derives change with it:

```python
# was: frames_zarr.parent / "video_quality_report.json"
report_path = images_dir.parent / "video_quality_report.json"
```

2. Replace both `FrameStore.create(...)` calls (the directory branch and the video branch) with:

```python
frames.write_frames(images_dir, frame_arrays, records, prov)
```

3. Replace the directory-branch extension listing:

```python
# was: exts = {".jpg", ".jpeg", ".png"}
#      frames = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
source_paths = frames.frame_paths(input_path)
if not source_paths:
    raise ValueError(f"No images ({list(frames.IMAGE_EXTS)}) found in directory {input_path}")
frame_arrays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in source_paths]
```

4. Update the imports at the top of the module:

```python
from collab_splats.preproc import frames
```

and delete `from collab_splats.preproc.frame_store import FrameStore`.

5. Replace `_LazyFrames` with a cached reader. Find the class and substitute:

```python
@lru_cache(maxsize=1)
def _scene_frames(images_dir: Path) -> np.ndarray:
    """
    Every frame of a scene as one RGB stack, cached per directory.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        (N, H, W, 3) uint8 RGB.
    """
    return frames.read_frames(images_dir)
```

with `from functools import lru_cache` at the top. Replace each `_LazyFrames(...)` construction with `_scene_frames(images_dir)` and each `lazy[i]` subscript with `_scene_frames(images_dir)[i]`.

- [ ] **Step 4: Run the wrapper suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
```

Expected: the new test passes. Other tests in this file that assert on `frames.zarr` will fail — fix them in the same task by applying the Task 3 table to their assertions.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/
isort collab_splats/wrapper/reconstructor.py tests/wrapper/
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/ \
  -m "refactor(wrapper): extract_frames writes images/ + frames.json"
```

---

### Task 5: `pointcloud/sfm.py` drops the JPEG staging step

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py`
- Test: `tests/wrapper/test_sfm_result.py`

- [ ] **Step 1: Find the staging block**

```bash
git grep -n 'export\|instantsfm/images\|image_path' -- collab_splats/pointcloud/sfm.py
```

The block writes `<scene>/instantsfm/images/frame_NNNNNN.jpg` from the store and then hands that directory to InstantSfM as `image_path`.

- [ ] **Step 2: Write the failing test**

Append to `tests/wrapper/test_sfm_result.py`:

```python
def test_sfm_points_at_the_scene_images_dir_and_stages_nothing(tmp_path):
    """
    InstantSfM reads <scene>/images directly; no instantsfm/images/ copy is written.
    """
    from collab_splats.pointcloud.sfm import _sfm_image_dir

    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)

    assert _sfm_image_dir(scene / "images") == scene / "images"
    assert not (scene / "instantsfm" / "images").exists()
```

- [ ] **Step 3: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_result.py::test_sfm_points_at_the_scene_images_dir_and_stages_nothing -v
```

Expected: FAIL, `ImportError: cannot import name '_sfm_image_dir'`.

- [ ] **Step 4: Delete the staging loop**

In `collab_splats/pointcloud/sfm.py`, delete the loop that writes `instantsfm/images/*.jpg` and the `FrameStore.open(...)` above it, and replace the whole block with:

```python
def _sfm_image_dir(images_dir: Path) -> Path:
    """
    Image directory InstantSfM reads.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        The same directory — the store IS the COLMAP image layout, so nothing is staged.
    """
    return Path(images_dir)
```

Then pass `_sfm_image_dir(images_dir)` where the staged directory was passed. VDA's depth naming (`depth_vda/images/npy/<stem>.npy`) is stem-keyed and needs no change, because the stems are still `frame_NNNNNN`.

- [ ] **Step 5: Run the test and the sfm suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_result.py tests/pointcloud/ -v
```

Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
black collab_splats/pointcloud/sfm.py tests/wrapper/test_sfm_result.py
isort collab_splats/pointcloud/sfm.py tests/wrapper/test_sfm_result.py
git commit --only collab_splats/pointcloud/sfm.py tests/wrapper/test_sfm_result.py \
  -m "refactor(sfm): read <scene>/images directly, drop JPEG staging"
```

---

### Task 6: `feedforward/base.py` drops the transient export

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`_decode_source`, and the listing at ~L968)
- Test: `tests/pointcloud/feedforward/test_preprocess_frames.py`, `tests/pointcloud/test_feedforward_preprocess_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/feedforward/test_preprocess_frames.py`:

```python
def test_decode_source_takes_an_images_dir_and_makes_no_tempdir(tmp_path, monkeypatch):
    """
    Path-locked model preprocessing reads the scene's images/ directly.
    """
    import tempfile

    import cv2
    import numpy as np

    from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator

    images = tmp_path / "images"
    images.mkdir()
    for i in (0, 5):
        cv2.imwrite(str(images / f"frame_{i:06d}.png"), np.full((8, 12, 3), i + 5, np.uint8))

    # Any TemporaryDirectory here means a copy is still being staged
    monkeypatch.setattr(
        tempfile, "TemporaryDirectory", lambda *a, **k: pytest.fail("staged a temporary copy")
    )

    paths = BaseFeedforwardCreator._source_paths(images)
    assert [p.name for p in paths] == ["frame_000000.png", "frame_000005.png"]
```

Add `import pytest` at the top of the file if it is not already there.

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_preprocess_frames.py::test_decode_source_takes_an_images_dir_and_makes_no_tempdir -v
```

Expected: FAIL, `AttributeError: type object 'BaseFeedforwardCreator' has no attribute '_source_paths'`.

- [ ] **Step 3: Replace the export with a direct read**

In `collab_splats/pointcloud/feedforward/base.py`:

1. Add the static helper, replacing the duplicated extension listing at ~L968:

```python
@staticmethod
def _source_paths(images_dir: Path) -> list[Path]:
    """
    Frame image paths a path-locked model preprocessor reads.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        Image paths in filename order.
    """
    return frames.frame_paths(images_dir)
```

with `from collab_splats.preproc import frames` at the top.

2. In `_decode_source`, change the parameter from a `FrameStore` to `images_dir: Path`, delete the `with tempfile.TemporaryDirectory() as tmp:` block and the `store.export(tmp)` inside it, and use `self._source_paths(images_dir)` where the exported paths were used. Un-indent the body that was inside the `with`.

3. Delete the now-unused `import tempfile` if nothing else in the module uses it.

- [ ] **Step 4: Run the feedforward suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ -v
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/pointcloud/feedforward/base.py tests/pointcloud/
isort collab_splats/pointcloud/feedforward/base.py tests/pointcloud/
git commit --only collab_splats/pointcloud/feedforward/base.py tests/pointcloud/ \
  -m "refactor(feedforward): read images/ directly, drop transient export"
```

---

### Task 7: `semantics`, `geometry` and `mesh` call sites

**Files:**
- Modify: `collab_splats/semantics/features/base.py`, `collab_splats/geometry/metrics.py`, `collab_splats/geometry/loop_closure/wrapper.py`, `collab_splats/mesh/utils.py`
- Test: `tests/semantics/features/test_extract_from_zarr.py`, `tests/geometry/test_metrics.py`, `tests/mesh/test_utils.py`, `tests/mesh/test_absent_confidence.py`

- [ ] **Step 1: Find every use**

```bash
git grep -n 'FrameStore' -- collab_splats/semantics collab_splats/geometry collab_splats/mesh
```

- [ ] **Step 2: Apply the Task 3 table**

Mechanical, one file at a time. Two specifics:

- `geometry/loop_closure/wrapper.py` has a `FrameStore | Path` union parameter. It collapses to `Path`; delete the `isinstance` branch that handled the store and keep the path branch.
- `geometry/metrics.py` uses `FrameStore.frame_idx_from_path` for its source-frame join. That becomes `frames.frame_idx_from_path`, imported from `collab_splats.preproc.frames`.

- [ ] **Step 3: Run the affected suites**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/ tests/geometry/ tests/mesh/ -v
```

Expected: PASS. Test files that construct a `FrameStore` fixture switch to `fr.write_frames(tmp_path / "images", ...)`.

- [ ] **Step 4: Format and commit**

```bash
black collab_splats/semantics collab_splats/geometry collab_splats/mesh tests/semantics tests/geometry tests/mesh
isort collab_splats/semantics collab_splats/geometry collab_splats/mesh tests/semantics tests/geometry tests/mesh
git commit --only collab_splats/semantics collab_splats/geometry collab_splats/mesh tests/semantics tests/geometry tests/mesh \
  -m "refactor(semantics,geometry,mesh): read images/ in place of FrameStore"
```

---

### Task 8: `dashboard`, `evals` and the notebooks

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py`, `collab_splats/dashboard/localize.py`, `evals/datasets.py`, `evals/scripts/eval_splats.py`, `evals/scripts/eval_verification.py`, `docs/source/tutorials/notebook_utils.py`
- Test: `tests/dashboard/`, `tests/evals/`, `tests/docs/test_notebook_utils.py`

- [ ] **Step 1: Find every use**

```bash
git grep -n 'FrameStore' -- collab_splats/dashboard evals docs/source/tutorials/notebook_utils.py
```

- [ ] **Step 2: Apply the Task 3 table**

One extra deletion: `dashboard/localize.py`'s `_local_ref_paths` writes a thumbnail directory that duplicates what `images/` now holds. Delete the function and point its caller at `frames.frame_paths(images_dir)`.

- [ ] **Step 3: Update the notebooks**

Six tutorial notebooks reference `FrameStore`. For each, replace the import and the store calls per the Task 3 table:

```bash
git grep -ln 'FrameStore' -- 'docs/source/tutorials/**/*.ipynb'
```

Edit each with the NotebookEdit tool, not by hand-editing JSON. Re-run each edited notebook top to bottom and commit the executed output — a notebook with a stale traceback is how `plot_quality_examples` stayed broken.

- [ ] **Step 4: Run the suites and the mandatory smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ tests/evals/ tests/docs/ -v
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke
```

Expected: tests PASS; smoke exits 0.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/dashboard evals docs/source/tutorials/notebook_utils.py tests/dashboard tests/evals tests/docs
isort collab_splats/dashboard evals docs/source/tutorials/notebook_utils.py tests/dashboard tests/evals tests/docs
git commit --only collab_splats/dashboard evals docs/source/tutorials tests/dashboard tests/evals tests/docs \
  -m "refactor(dashboard,evals,docs): read images/ in place of FrameStore"
```

---

### Task 9: Delete `frame_store.py` and close out Phase A

**Files:**
- Delete: `collab_splats/preproc/frame_store.py`, `tests/preproc/test_frame_store.py`
- Modify: `collab_splats/preproc/__init__.py`, `collab_splats/remote/sources.py`, `configs/base.yaml`, `docs/source/api/preproc.rst`

- [ ] **Step 1: Verify nothing imports it**

```bash
git grep -n 'frame_store\|FrameStore' -- 'collab_splats/*' 'evals/*' 'scripts/*' 'tests/*'
```

Expected: only `collab_splats/preproc/frame_store.py`, `collab_splats/preproc/__init__.py` and `tests/preproc/test_frame_store.py`. Anything else is an unconverted call site — go back and convert it.

If `collab_splats/preproc/viz.py` appears, Task 25 has not run. It is the only `FrameStore` consumer no Phase A task converts; run Task 25 rather than writing a conversion for a function that is about to be deleted.

- [ ] **Step 2: Delete and rewire**

```bash
git rm collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py
```

In `collab_splats/preproc/__init__.py`, replace the `FrameStore` import with:

```python
from collab_splats.preproc.frames import (
    frame_idx_from_path,
    frame_paths,
    read_frames,
    read_manifest,
    write_frames,
)
```

and update `__all__` — remove `"FrameStore"`, add `"frame_idx_from_path"`, `"frame_paths"`, `"read_frames"`, `"read_manifest"`, `"write_frames"`, keeping the list alphabetically sorted. Update the module docstring's `frame_store.py` mention to `frames.py`.

In `collab_splats/remote/sources.py`, rewrite `PUSH_EXCLUDES` / `PULL_EXCLUDES` and their comment block: every `frames.zarr/**` pattern becomes `images/**`, and the comment explaining why the store is excluded now says "the images/ directory is regenerable from the source video, and is the largest thing in a scene".

In `configs/base.yaml`, rewrite the `preproc:` block's leading comment:

```yaml
preproc:
  # Output: <output_path>/images/frame_NNNNNN.png (COLMAP-style image directory,
  # lossless PNG) + <output_path>/frames.json (selection records + provenance).
```

and change the `undistort:` comment's `frames.zarr is written` to `images/ is written`, and its `frames.zarr reuse is by EXISTENCE` to `images/ reuse is by EXISTENCE`.

In `docs/source/api/preproc.rst`, replace the `frame_store` automodule entry with `frames`.

- [ ] **Step 3: Run the whole suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke
```

Expected: the suite is green apart from anything already listed in `docs/known-test-failures.md`; smoke exits 0.

- [ ] **Step 4: Update the graph and commit**

```bash
graphify update .
black collab_splats/preproc configs
isort collab_splats/preproc
git commit --only collab_splats/preproc collab_splats/remote/sources.py configs/base.yaml docs/source/api/preproc.rst \
  -m "refactor(preproc)!: delete FrameStore, images/ is the store

BREAKING: a scene holding only frames.zarr raises. Convert it with
scripts/migrate_frames_zarr.py."
```

**Phase A gate:** the suite is green, the dashboard smoke passes, and frame selection is byte-identical to before the phase. If a selection changed, it is a bug in Phase A — find it before starting Phase B.

---

## Phase B — the quality filter fires, and sampling reads from the pool it makes

Invariant for this phase: **selection changes on purpose**, and Task 14 is the gate that proves the change is the intended one. Nothing in Phase B may land on a branch that has not passed Task 14.

---

### Task 10: `filter_frame_quality` — robust MAD on log(laplacian)

**Files:**
- Modify: `collab_splats/preproc/sampling.py:32-76`
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
def _report(laplacian, *, clipped_low=None, clipped_high=None):
    """
    Minimal quality report carrying only the columns the filter reads.
    """
    n = len(laplacian)
    return {
        "frames": {
            "laplacian": list(laplacian),
            "clipped_low_frac": list(clipped_low if clipped_low is not None else [0.0] * n),
            "clipped_high_frac": list(clipped_high if clipped_high is not None else [0.0] * n),
        }
    }


def test_filter_cuts_the_soft_frame_in_an_otherwise_sharp_run():
    """
    One frame two orders of magnitude softer than its neighbours is cut.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    mask = filter_frame_quality(_report([400.0] * 20 + [3.0] + [400.0] * 20))

    assert mask[20] == False  # noqa: E712 — the soft frame
    assert mask.sum() == 40


def test_filter_is_scale_free():
    """
    Multiplying every laplacian by a constant cannot change the mask —
    that is the whole point of a robust z-score on the log.
    """
    import numpy as np

    from collab_splats.preproc.sampling import filter_frame_quality

    lap = [400.0, 380.0, 410.0, 3.0, 395.0, 405.0] * 8
    a = filter_frame_quality(_report(lap))
    b = filter_frame_quality(_report([x * 1000.0 for x in lap]))

    assert np.array_equal(a, b)


def test_filter_keeps_everything_when_sharpness_is_uniform():
    """
    Zero MAD must not divide-by-zero into an all-False mask.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    assert filter_frame_quality(_report([250.0] * 30)).all()


def test_filter_cuts_a_clipped_frame():
    """
    Clipping is an absolute rule: >25% destroyed pixels is out regardless of sharpness.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    lap = [400.0] * 10
    low = [0.0] * 9 + [0.30]
    mask = filter_frame_quality(_report(lap, clipped_low=low))

    assert mask[9] == False  # noqa: E712
    assert mask[:9].all()


def test_filter_clipping_is_the_sum_of_both_tails():
    """
    0.15 crushed + 0.15 blown is 0.30 destroyed, over the 0.25 ceiling.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    mask = filter_frame_quality(
        _report([400.0] * 4, clipped_low=[0.0, 0.0, 0.0, 0.15], clipped_high=[0.0, 0.0, 0.0, 0.15])
    )

    assert mask[3] == False  # noqa: E712


def test_filter_handles_an_empty_report():
    """
    A report with no rows returns an empty mask, not an exception.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    assert filter_frame_quality(_report([])).shape == (0,)


def test_filter_no_longer_takes_the_deleted_thresholds():
    """
    laplacian_min et al are gone; passing one is a TypeError, not a silent no-op.
    """
    import pytest

    from collab_splats.preproc.sampling import filter_frame_quality

    for dead in ("laplacian_min", "exposure_mean_range", "exposure_min_std", "blur_max"):
        with pytest.raises(TypeError):
            filter_frame_quality(_report([400.0] * 5), **{dead: 1})
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k filter -v
```

Expected: the scale-free, soft-frame, clipping and dead-kwarg tests FAIL — the current filter uses a fixed `laplacian_min=50` and reads no clipping columns.

- [ ] **Step 3: Replace the filter**

In `collab_splats/preproc/sampling.py`, replace the whole of `filter_frame_quality` (lines 32-76) with:

```python
def filter_frame_quality(
    report: dict,
    *,
    sharpness_k: float = 2.0,
    max_clipped_frac: float = 0.25,
) -> np.ndarray:
    """
    Per-frame usability mask over a quality report's photometry columns.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        sharpness_k: robust z-score cut on log(laplacian); larger keeps more.
        max_clipped_frac: ceiling on clipped_low_frac + clipped_high_frac.

    Returns:
        (N,) bool, True = keep, indexed by source frame index.
    """
    f = report["frames"]
    lap = np.asarray(f["laplacian"], dtype=float)
    if lap.size == 0:
        return np.zeros(0, dtype=bool)

    # Sharpness is relative: laplacian variance scales with resolution, texture and
    # content, so the cut is a robust z-score on the log rather than an absolute value.
    log_lap = np.log(np.clip(lap, 1e-6, None))
    centre = np.median(log_lap)
    spread = float(median_abs_deviation(log_lap, scale="normal"))

    # A zero MAD means every frame is equally sharp — nothing to cut, and the
    # z-score would be a division by zero.
    sharp = np.ones_like(lap, dtype=bool) if spread == 0.0 else log_lap >= centre - sharpness_k * spread

    # Clipping is absolute: a pixel at 0 or 255 recorded nothing recoverable.
    clipped = np.asarray(f["clipped_low_frac"], dtype=float) + np.asarray(f["clipped_high_frac"], dtype=float)

    return sharp & (clipped <= max_clipped_frac)
```

Add the import at the top of the module:

```python
from scipy.stats import median_abs_deviation
```

- [ ] **Step 4: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k filter -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit --only collab_splats/preproc/sampling.py tests/preproc/test_sampling.py \
  -m "feat(preproc)!: robust MAD sharpness cut + absolute clipping cut

BREAKING: filter_frame_quality drops laplacian_min, exposure_mean_range,
exposure_min_std and blur_max. It now cuts frames; the old defaults never did."
```

The rest of the suite will now fail wherever `_sample_by_quality` consumed the old mask. That is expected — Tasks 11-13 fix it.

---

### Task 11: the eligible pool and `sample_uniform`

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (add `_eligible` and `_decode_selection`, rewrite `sample_uniform`)
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
def test_eligible_intersects_the_mask_with_the_candidate_grid():
    """
    The quality mask and the VDA context grid are one restriction, not two.
    """
    import numpy as np

    from collab_splats.preproc.sampling import _eligible

    report = _report([400.0] * 20 + [3.0] + [400.0] * 9)
    pool = _eligible(report, quality=None, candidates=[0, 10, 20, 25])

    # 20 is condemned by the mask, so the grid loses it
    assert np.array_equal(pool, np.array([0, 10, 25]))


def test_eligible_raises_when_the_pool_is_empty():
    """
    An empty pool is a config error, not an empty scene written silently.
    """
    import pytest

    from collab_splats.preproc.sampling import _eligible

    with pytest.raises(ValueError, match="no eligible frames"):
        _eligible(_report([400.0] * 10), quality=None, candidates=[])


def test_sample_uniform_spans_the_eligible_pool(monkeypatch, tmp_path):
    """
    Picks are evenly spaced in POOL index, and never land on a condemned frame.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    # 30 frames, the middle 10 blurred out
    report = _report([400.0] * 10 + [2.0] * 10 + [400.0] * 10)

    monkeypatch.setattr(sampling, "get_video_info", lambda p: {"total_frames": 30, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i, np.uint8)) for i in indices],
    )

    frames, records = sampling.sample_uniform(str(tmp_path / "v.mp4"), max_frames=4, report=report)

    picked = [r["frame_idx"] for r in records]
    assert len(frames) == 4
    assert all(i < 10 or i >= 20 for i in picked), picked
    assert picked == sorted(picked)


def test_sample_uniform_returns_the_whole_pool_when_it_is_short(monkeypatch, tmp_path, caplog):
    """
    A pool smaller than max_frames returns the pool and logs the shortfall.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    report = _report([400.0] * 3)
    monkeypatch.setattr(sampling, "get_video_info", lambda p: {"total_frames": 3, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i, np.uint8)) for i in indices],
    )

    with caplog.at_level("WARNING"):
        frames, records = sampling.sample_uniform(str(tmp_path / "v.mp4"), max_frames=10, report=report)

    assert [r["frame_idx"] for r in records] == [0, 1, 2]
    assert "eligible" in caplog.text
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "eligible or uniform" -v
```

Expected: FAIL, `ImportError: cannot import name '_eligible'`.

- [ ] **Step 3: Add the two helpers**

In `collab_splats/preproc/sampling.py`, delete `_sample_by_quality` in full (lines 238-336) and put these in its place:

```python
def _eligible(report: dict, *, quality: dict | None, candidates: Sequence[int] | None) -> np.ndarray:
    """
    Source frame indices a sampler may select from.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        quality: overrides for filter_frame_quality's thresholds.
        candidates: optional index grid every pick must be a member of (the VDA context grid).

    Returns:
        (M,) int64, ascending.
    """
    pool = np.flatnonzero(filter_frame_quality(report, **(quality or {})))

    # The mask and the context grid are the same kind of restriction, so they intersect
    if candidates is not None:
        pool = np.intersect1d(pool, np.asarray(sorted({int(c) for c in candidates}), dtype=np.int64))

    if pool.size == 0:
        raise ValueError(
            "no eligible frames: the quality filter and the candidate grid have no index in common"
        )

    return pool


def _decode_selection(
    video_path: str,
    chosen: Sequence[int],
    *,
    report: dict,
    on_progress,
    desc: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Decode exactly the selected frames, in one pass, as RGB.

    Args:
        video_path: source video.
        chosen: ascending source frame indices.
        report: the quality report the blur_score column is read from.
        on_progress: optional (done, total) callback.
        desc: progress-bar label.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    # One ffmpeg select pass over exactly the frames we keep
    decoded = dict(iter_frames(video_path, indices=list(chosen)))

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx in progress(chosen, total=len(chosen), desc=desc, on_progress=on_progress):
        bgr = decoded.get(idx)
        if bgr is None:
            continue  # ffmpeg dropped the frame (should not happen)

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # blur_score comes from the report, not a recompute — same measurement,
        # and it is the column the record has always carried.
        records.append({"frame_idx": int(idx), "blur_score": float(laplacian[idx])})

    return frames, records
```

- [ ] **Step 4: Rewrite `sample_uniform`**

Replace `sample_uniform` (through its `return _sample_by_quality(...)`) with:

```python
def sample_uniform(
    video_path: str,
    *,
    max_frames: int,
    report: dict,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Exactly max_frames evenly-spaced picks from the eligible pool.

    Args:
        video_path: source video.
        max_frames: how many frames to keep — the COUNT is the contract.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.
        candidates: optional index grid every pick must be a member of.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    if max_frames <= 0:
        return [], []

    pool = _eligible(report, quality=quality, candidates=candidates)

    # Spacing is even in POOL index, not in time: budget is not spent inside
    # footage the filter just condemned.
    if pool.size <= max_frames:
        logger.warning(
            "max_frames=%d but only %d eligible frames; keeping the whole pool", max_frames, pool.size
        )
        chosen = pool.tolist()
    else:
        chosen = pool[np.linspace(0, pool.size - 1, max_frames).round().astype(int)].tolist()

    return _decode_selection(
        video_path, chosen, report=report, on_progress=on_progress, desc="Uniform sampling"
    )
```

Note the deleted `get_video_info` call: the pool comes from the report, whose length already is the video's frame count, so uniform sampling no longer probes the video at all.

- [ ] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "eligible or uniform" -v
```

Expected: PASS. `test_sample_uniform_spans_the_eligible_pool`'s monkeypatched `get_video_info` is now unused for `sample_uniform` — leave it, `sample_fps` in the next task needs it.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit --only collab_splats/preproc/sampling.py tests/preproc/test_sampling.py \
  -m "refactor(preproc)!: sample_uniform picks from the eligible pool

BREAKING: deletes _sample_by_quality and sample_uniform's search_radius."
```

---

### Task 12: `sample_fps` snaps to the pool, and `context_indices` moves here

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (`sample_fps`), `collab_splats/preproc/video.py:189-211` (delete `context_indices`)
- Test: `tests/preproc/test_sampling.py`, `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
def test_sample_fps_snaps_targets_to_the_nearest_eligible_frame(monkeypatch, tmp_path):
    """
    Constant-rate targets land on the closest eligible index, never on a condemned one.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    # 60 frames at 30 fps; frames 10-14 blurred out. fps=3 targets 0,10,20,...
    report = _report([400.0] * 10 + [2.0] * 5 + [400.0] * 45)

    monkeypatch.setattr(sampling, "get_video_info", lambda p, **k: {"total_frames": 60, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i % 251, np.uint8)) for i in indices],
    )

    frames, records = sampling.sample_fps(str(tmp_path / "v.mp4"), fps=3.0, report=report)
    picked = [r["frame_idx"] for r in records]

    # target 10 is condemned; 9 and 15 are equidistant-ish, 9 is nearer
    assert 10 not in picked
    assert 9 in picked
    assert picked == sorted(set(picked))


def test_sample_fps_respreads_outside_the_band(monkeypatch, tmp_path, caplog):
    """
    A count outside [min_frames, max_frames] re-spreads over the whole pool, never truncates.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    report = _report([400.0] * 60)
    monkeypatch.setattr(sampling, "get_video_info", lambda p, **k: {"total_frames": 60, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i % 251, np.uint8)) for i in indices],
    )

    with caplog.at_level("WARNING"):
        frames, records = sampling.sample_fps(
            str(tmp_path / "v.mp4"), fps=15.0, report=report, max_frames=6
        )

    picked = [r["frame_idx"] for r in records]
    assert len(picked) == 6
    assert picked[-1] >= 55, "re-spread must still span the video, not truncate at frame 6"
    assert "re-spread" in caplog.text


def test_context_indices_lives_in_sampling():
    """
    It computes a selection grid, so it belongs to this module.
    """
    from collab_splats.preproc.sampling import context_indices

    assert context_indices("x.mp4", target_fps=2.0, info={"total_frames": 10, "fps": 10.0}) == [0, 5]
```

And in `tests/preproc/test_video.py`, delete every `context_indices` test and add:

```python
def test_video_no_longer_exports_context_indices():
    """
    context_indices moved to sampling; video.py decodes, it does not select.
    """
    from collab_splats.preproc import video

    assert not hasattr(video, "context_indices")
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "fps or context" tests/preproc/test_video.py -k context -v
```

Expected: FAIL, `ImportError: cannot import name 'context_indices' from 'collab_splats.preproc.sampling'`.

- [ ] **Step 3: Move `context_indices`**

Cut `context_indices` out of `collab_splats/preproc/video.py` (lines 189-211) and paste it into `collab_splats/preproc/sampling.py` above `_eligible`, with the docstring rewritten to house style:

```python
def context_indices(video_path: str | Path, *, target_fps: float, info: dict | None = None) -> list[int]:
    """
    Source frame indices on a constant-rate grid at target_fps.

    Args:
        video_path: source video.
        target_fps: grid rate; must be positive.
        info: a get_video_info dict, to hoist the probe out of a loop.

    Returns:
        Ascending source frame indices. Stride floors at 1 — a rate above the
        source rate cannot sample sub-frame.
    """
    # target_fps is the contract here, so an absent one is a config error, not a default
    if target_fps is None or target_fps <= 0:
        raise ValueError(f"context_indices needs a positive target_fps, got {target_fps!r}")

    # Reuse a caller's probe when given — a fresh one costs a container parse
    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]
    if total == 0:
        return []

    native_fps = info["fps"] or 30.0
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, total, step))
```

Add `from pathlib import Path` to `sampling.py` if absent. Then update the imports:

- `sampling.py`: `from collab_splats.preproc.video import get_video_info, iter_frames` (drop `context_indices`).
- Anything importing `context_indices` from `video`:

```bash
git grep -n 'context_indices'
```

Expected callers: `collab_splats/wrapper/reconstructor.py` (VDA context grid) and `collab_splats/preproc/__init__.py` if it lists it. Repoint them at `collab_splats.preproc.sampling`.

- [ ] **Step 4: Rewrite `sample_fps`**

Replace the body of `sample_fps` from `info = get_video_info(...)` to its `return`, and drop `search_radius` from the signature:

```python
def sample_fps(
    video_path: str,
    *,
    fps: float,
    report: dict,
    min_frames: int | None = None,
    max_frames: int | None = None,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    One frame every 1/fps seconds, each snapped to the nearest eligible frame.

    Args:
        video_path: source video.
        fps: target rate — the SPACING is the contract, the count floats.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        min_frames: floor; a count below it re-spreads over the whole video.
        max_frames: ceiling; a count above it re-spreads over the whole video.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.
        candidates: optional index grid every pick must be a member of.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    # fps is the contract here, so an absent one is a config error, not a default
    if fps is None or fps <= 0:
        raise ValueError(f"sample_fps needs a positive fps, got {fps!r}")

    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []

    pool = _eligible(report, quality=quality, candidates=candidates)

    # One source of truth for the stride: a context grid built at this same rate
    # contains these targets by construction, not by coincidence
    targets = context_indices(video_path, target_fps=fps, info=info)

    # Clamp the floating count into the band by re-spreading over the pool, never by
    # truncating — truncation would hand the reconstructor half a scene
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = pool[np.linspace(0, pool.size - 1, min(bounded, pool.size)).round().astype(int)].tolist()
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            (info["fps"] or 30.0) * len(targets) / total,
        )

    # Snap each target to the nearest eligible frame, then dedup: two targets either
    # side of an excised stretch can snap to the same survivor.
    pos = np.clip(np.searchsorted(pool, targets), 1, pool.size - 1)
    left, right = pool[pos - 1], pool[pos]
    snapped = np.where(np.abs(targets - left) <= np.abs(right - targets), left, right)
    chosen = sorted(set(snapped.tolist()))

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="fps sampling")
```

The `np.clip(..., 1, pool.size - 1)` is deliberate: it makes `pos - 1` and `pos` both valid for a single-element pool, where they collapse to the same index and the `where` picks it either way.

- [ ] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py tests/preproc/test_video.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc tests/preproc
isort collab_splats/preproc tests/preproc
git commit --only collab_splats/preproc tests/preproc \
  -m "refactor(preproc)!: sample_fps snaps to the eligible pool; context_indices moves to sampling

BREAKING: preproc.video.context_indices is now preproc.sampling.context_indices."
```

---

### Task 13: retire `search_radius` everywhere

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (`sample_optical_flow`), `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml`
- Test: `tests/preproc/test_sampling.py`, `tests/wrapper/test_reconstructor.py`, `tests/preproc/test_sampling_parity.py`

- [ ] **Step 1: Find every mention**

```bash
git grep -n 'search_radius'
```

Expected: `collab_splats/preproc/sampling.py`, `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml` (where it says `7`), `tests/preproc/test_sampling.py`, `tests/preproc/test_sampling_parity.py`, `tests/wrapper/test_reconstructor*.py`, and the notebooks.

- [ ] **Step 2: Write the failing test**

Append to `tests/preproc/test_sampling.py`:

```python
def test_samplers_no_longer_take_search_radius():
    """
    The window search is gone; the pool replaced it.
    """
    import inspect

    from collab_splats.preproc import sampling

    for fn in (sampling.sample_uniform, sampling.sample_fps, sampling.sample_optical_flow):
        assert "search_radius" not in inspect.signature(fn).parameters, fn.__name__
```

- [ ] **Step 3: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py::test_samplers_no_longer_take_search_radius -v
```

Expected: FAIL — `sample_optical_flow` never had it, but the reconstructor still passes it and `base.yaml` still sets it.

- [ ] **Step 4: Delete it**

1. `collab_splats/preproc/sampling.py` — `sample_optical_flow` keeps its signature but swaps its gate to the shared pool. Replace its `usable = filter_frame_quality(...)` line and the loop's gate with:

```python
    pool = set(_eligible(report, quality=quality, candidates=None).tolist())
```

and in the loop:

```python
        # Pool gate first: an ineligible frame never reaches the selector, so it
        # cannot become the reference the next frames are scored against.
        if idx not in pool:
            continue
```

Rewrite its docstring to the `Args:`/`Returns:` house form while you are in there.

2. `collab_splats/wrapper/reconstructor.py` — delete the `search_radius: int = 3` parameter from `extract_frames`, every `search_radius=search_radius` it forwards to a sampler, and the `cfg.preproc.search_radius` read at the call site.

3. `configs/base.yaml` — delete the `search_radius: 7` line and its comment.

4. Delete `search_radius` from every test and notebook the grep found.

- [ ] **Step 5: Run the preproc and wrapper suites**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/wrapper/ -v
git grep -n 'search_radius'
```

Expected: tests PASS; the grep returns nothing.

`tests/preproc/test_sampling_parity.py` asserts the old window-argmax behaviour. Its premise is deleted, so delete the file — do not rewrite it into a test of the new behaviour, Tasks 11-12 already cover that.

- [ ] **Step 6: Commit**

```bash
black collab_splats tests configs
isort collab_splats tests
git commit --only collab_splats/preproc collab_splats/wrapper/reconstructor.py configs/base.yaml tests/preproc tests/wrapper \
  -m "refactor(preproc)!: retire search_radius

BREAKING: preproc.search_radius leaves base.yaml and the samplers. It said 7 in
config and defaulted to 3 in code — nothing depended on either."
```

---

### Task 14: **HARD GATE** — the GH010229 A/B

**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/ada04594-773b-4b10-b55a-fa414346cead/scratchpad/ab_sampling.py`

Nothing in Phase B ships until this runs and a human reads it. Phase B changes which frames a scene is built from — that is the point — and this is what turns "the filter fires" into a number someone signed off on.

- [ ] **Step 1: Write the harness**

```python
"""
A/B the old window-argmax selection against the new eligible-pool selection.

Run against GH010229 (13,115 frames, the video whose 5.1% cut motivated the change)
and against the tutorial video (14.5% cut).
"""

import json
import sys
from pathlib import Path

import numpy as np

from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import _eligible, context_indices, filter_frame_quality


def main(report_path: str, fps: float, max_frames: int) -> None:
    report = load_video_quality(Path(report_path))
    lap = np.asarray(report["frames"]["laplacian"], dtype=float)
    n = lap.size

    mask = filter_frame_quality(report)
    pool = _eligible(report, quality=None, candidates=None)

    # New fps selection: constant-rate targets snapped to the pool
    targets = np.asarray(context_indices("", target_fps=fps, info={"total_frames": n, "fps": 30.0}))
    pos = np.clip(np.searchsorted(pool, targets), 1, pool.size - 1)
    left, right = pool[pos - 1], pool[pos]
    new = np.unique(np.where(np.abs(targets - left) <= np.abs(right - targets), left, right))

    # Old fps selection: the same targets, unfiltered (the old mask never fired)
    old = np.unique(targets)

    print(f"frames               {n}")
    print(f"cut by filter        {100 * (1 - mask.mean()):.1f}%")
    print(f"old picks            {old.size}")
    print(f"new picks            {new.size}")
    print(f"unchanged            {np.intersect1d(old, new).size}")
    print(f"max index shift      {int(np.abs(new - old[: new.size]).max()) if new.size else 0}")
    print(f"old mean laplacian   {lap[old].mean():.1f}")
    print(f"new mean laplacian   {lap[new].mean():.1f}")
    print(f"old min laplacian    {lap[old].min():.1f}")
    print(f"new min laplacian    {lap[new].min():.1f}")

    # Largest gap in source frames — the bunching the spec predicts
    print(f"old max gap          {int(np.diff(old).max()) if old.size > 1 else 0}")
    print(f"new max gap          {int(np.diff(new).max()) if new.size > 1 else 0}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
```

- [ ] **Step 2: Run it on both videos**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/ada04594-773b-4b10-b55a-fa414346cead/scratchpad
/opt/venv/reconstruction/bin/python $SCRATCH/ab_sampling.py <gh010229 scene>/video_quality_report.json 2.0 300
/opt/venv/reconstruction/bin/python $SCRATCH/ab_sampling.py <tutorial scene>/video_quality_report.json 2.0 300
```

If no report exists for GH010229, generate one first:

```bash
/opt/venv/reconstruction/bin/python -c "
from pathlib import Path
from collab_splats.preproc.qa import compute_video_quality
compute_video_quality('<path to GH010229.MP4>', output_path=Path('$SCRATCH/gh_report.json'))
"
```

- [ ] **Step 3: Read the numbers against these expectations**

| line | expected | what a miss means |
|---|---|---|
| `cut by filter` GH010229 | ~5.1% | a different number means the filter is not the one measured in the spec |
| `cut by filter` tutorial | ~14.5% | same |
| `new min laplacian` | strictly above `old min laplacian` | the filter is not removing the soft frames it exists to remove |
| `new max gap` | larger than `old max gap` | the bunching the spec predicts; a gap of hundreds of frames means a long excised stretch — check it really is blurry footage |
| `unchanged` | most picks | a low overlap means the snap is moving frames it should not |

- [ ] **Step 4: Present to the user and stop**

Paste both tables. State plainly which frames moved and by how much. **Do not start Phase C until the user reads this and says go.** If `new max gap` is large, spot-check the excised stretch by writing those frames out and looking at them — a filter that condemns a correctly-exposed static shot is a bug, not a feature.

---

## Phase C — undistortion: pycolmap picks the camera, cv2 moves the pixels

`undistort` defaults to `false`, so nothing published depends on this phase. That is exactly why the framing change (COLMAP's fixed-focal expanding canvas, not cv2's fixed-canvas shrinking focal) can land without a toggle.

---

### Task 15: `calibrate_camera` returns a `pycolmap.Camera`

**Files:**
- Rewrite: `collab_splats/preproc/undistort.py` (delete `DistortionProfile` and `estimate_camera_distortion`)
- Test: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Write the failing tests**

Replace the `DistortionProfile` and `estimate_camera_distortion` tests in `tests/preproc/test_undistort.py` with:

```python
def test_calibrate_camera_returns_a_pycolmap_camera(tmp_path):
    """
    Calibration's output type IS pycolmap's, so nothing round-trips through a dataclass.
    """
    import pycolmap

    from collab_splats.preproc.undistort import calibrate_camera

    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=12, width=320, height=240)

    cam = calibrate_camera(images, max_frames=12)

    assert isinstance(cam, pycolmap.Camera)
    assert cam.model.name == "OPENCV"
    assert (cam.width, cam.height) == (320, 240)


def test_calibrate_camera_stages_no_image_copies(tmp_path, monkeypatch):
    """
    pycolmap.extract_features(image_names=...) reads the scene's images/ in place.
    """
    import cv2
    import pytest

    from collab_splats.preproc.undistort import calibrate_camera

    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=12, width=320, height=240)

    # The old code staged JPEG copies into a tempdir; any write here means it still does.
    # The COLMAP database still gets a tempdir — that is scratch, not a copy of the images.
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: pytest.fail("staged a temporary image copy"))
    calibrate_camera(images, max_frames=12)


def test_calibrate_camera_raises_when_registration_is_thin(tmp_path):
    """
    A featureless sequence cannot calibrate, and says so rather than returning nonsense.
    """
    import cv2
    import numpy as np
    import pytest

    from collab_splats.preproc.undistort import calibrate_camera

    images = tmp_path / "images"
    images.mkdir()
    for i in range(12):
        cv2.imwrite(str(images / f"frame_{i:06d}.png"), np.full((240, 320, 3), 128, np.uint8))

    with pytest.raises(RuntimeError, match="registered"):
        calibrate_camera(images, max_frames=12)
```

`_write_textured_sequence` is the existing helper in this file that renders a moving textured pattern; keep it. If it does not exist, add it:

```python
def _write_textured_sequence(out_dir, *, n, width, height):
    """
    n frames of a high-frequency pattern translating a few pixels per frame.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (height + 4 * n, width + 4 * n, 3), dtype=np.uint8)
    for i in range(n):
        crop = canvas[2 * i : 2 * i + height, 2 * i : 2 * i + width]
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), crop)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -k calibrate -v
```

Expected: FAIL, `ImportError: cannot import name 'calibrate_camera'`.

- [ ] **Step 3: Rewrite the calibration half of `undistort.py`**

Delete `DistortionProfile` (the whole frozen dataclass, `to_dict`, `from_dict`, `K`, `dist_coeffs`) and `estimate_camera_distortion`. Put this in their place:

```python
"""
Camera calibration and undistortion.

pycolmap estimates the camera and picks the undistorted framing; cv2 moves the
pixels. The camera IS a pycolmap.Camera — there is no local dataclass mirroring it.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

# SIFT reads the cgroup's core count, not the container's cap: 96 host cores on a
# 46.6 GB cgroup OOMs. See project_splatfacto_parity.
_SIFT_NUM_THREADS = 8


def calibrate_camera(images_dir: Path, *, max_frames: int = 60) -> pycolmap.Camera:
    """
    Estimate one shared OPENCV camera from a scene's images.

    Args:
        images_dir: the scene's images/ directory.
        max_frames: how many evenly-spaced images to calibrate from.

    Returns:
        A pycolmap.Camera (model OPENCV) carrying f, pp and k1 k2 p1 p2.
    """
    paths = frames.frame_paths(images_dir)
    if len(paths) < 8:
        raise ValueError(f"calibrate_camera needs at least 8 images, found {len(paths)} in {images_dir}")

    # Evenly-spaced subset: calibration wants baseline, not every frame
    idxs = np.linspace(0, len(paths) - 1, min(max_frames, len(paths))).round().astype(int)
    names = [paths[i].name for i in np.unique(idxs)]

    # The database is scratch; the images are read from images_dir in place
    with tempfile.TemporaryDirectory(prefix="calib_db_") as tmp:
        database = Path(tmp) / "database.db"

        pycolmap.extract_features(
            database,
            images_dir,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(camera_model="OPENCV"),
            extraction_options=pycolmap.FeatureExtractionOptions(num_threads=_SIFT_NUM_THREADS),
        )
        pycolmap.match_exhaustive(database)

        recons = pycolmap.incremental_mapping(database, images_dir, Path(tmp) / "sparse")

    if not recons:
        raise RuntimeError(f"calibration failed: no reconstruction from {len(names)} images in {images_dir}")

    recon = recons[0]

    # A reconstruction over a minority of the input has not seen the lens
    if len(recon.images) < 0.6 * len(names):
        raise RuntimeError(
            f"calibration too thin: {len(recon.images)} of {len(names)} images registered "
            f"(need 60%); the footage may be featureless or the motion degenerate"
        )

    camera = next(iter(recon.cameras.values()))
    logger.info("calibrated %s from %d/%d images: %s", images_dir, len(recon.images), len(names), camera)
    return camera
```

The `TemporaryDirectory` that survives holds the COLMAP database and sparse output — scratch, not copies of the images. What is deleted is the old `cv2.imwrite` JPEG staging loop that wrote every calibration frame out a second time; `image_names=` makes it unnecessary, and that is what the test above pins.

`from collab_splats.preproc import frames` is already in the import block shown above.

- [ ] **Step 3b: Rewire the package exports**

`collab_splats/preproc/__init__.py` still imports the two deleted names. Replace:

```python
from collab_splats.preproc.undistort import DistortionProfile, estimate_camera_distortion, undistort_frames
```

with:

```python
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
```

and in `__all__`, drop `"DistortionProfile"` and `"estimate_camera_distortion"`, add `"calibrate_camera"`, keeping the list sorted. With Task 9's five `frames` names this brings `__all__` to the 17 entries the spec's section 13 lists.

Then check nothing else imports them:

```bash
git grep -n 'DistortionProfile\|estimate_camera_distortion'
```

Expected after this task: nothing outside `docs/`. `collab_splats/wrapper/reconstructor.py` still has hits at this point — Task 17 clears them, and the tree is red between the two. Do Tasks 15, 16 and 17 back to back.

- [ ] **Step 4: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -k calibrate -v
```

Expected: PASS. These run real SIFT + mapping on 12 tiny images; budget ~20 s.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
isort collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
git commit --only collab_splats/preproc/undistort.py tests/preproc/test_undistort.py \
  -m "refactor(preproc)!: calibrate_camera returns a pycolmap.Camera

BREAKING: DistortionProfile and estimate_camera_distortion are deleted."
```

---

### Task 16: `undistort_frames` uses `pycolmap.undistort_camera`

**Files:**
- Modify: `collab_splats/preproc/undistort.py` (`undistort_frames`)
- Test: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_undistort.py`:

```python
def _distorted_camera(width=1920, height=1080):
    """
    A barrel-distorted OPENCV camera, k1=-0.25.
    """
    import pycolmap

    return pycolmap.Camera(
        model="OPENCV",
        width=width,
        height=height,
        params=[1190.4, 1190.4, width / 2, height / 2, -0.25, 0.05, 0.0, 0.0],
    )


def test_undistort_frames_keeps_the_focal_and_grows_the_canvas():
    """
    COLMAP's framing: focal is preserved, the canvas expands to hold the corners.
    """
    import numpy as np

    from collab_splats.preproc.undistort import undistort_frames

    cam = _distorted_camera()
    frames_in = np.zeros((2, 1080, 1920, 3), np.uint8)

    out, new_cam = undistort_frames(frames_in, cam)

    assert new_cam.model.name == "PINHOLE"
    assert new_cam.focal_length_x == cam.focal_length_x
    assert (new_cam.width, new_cam.height) > (cam.width, cam.height)
    assert out.shape == (2, new_cam.height, new_cam.width, 3)


def test_undistort_frames_straightens_a_line():
    """
    A row of dots bowed by barrel distortion comes back collinear.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc.undistort import undistort_frames

    cam = _distorted_camera(width=640, height=480)

    # Project a straight world line THROUGH the distortion, so undistorting must straighten it
    xs = np.linspace(-0.35, 0.35, 9)
    img = np.zeros((480, 640, 3), np.uint8)
    for x in xs:
        u, v = cam.img_from_cam(np.array([[x, -0.2, 1.0]]))[0]
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (255, 255, 255), -1)

    out, _ = undistort_frames(img[None], cam)

    # Centroid of each blob in the output; a straight line has near-zero y spread
    gray = cv2.cvtColor(out[0], cv2.COLOR_RGB2GRAY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats((gray > 128).astype(np.uint8))
    ys = sorted(c[1] for c in centroids[1:])

    assert n - 1 >= 7, "lost blobs — the warp is dropping content"
    assert max(ys) - min(ys) < 2.0, f"line still bowed: y spread {max(ys) - min(ys):.2f} px"


def test_undistort_frames_returns_a_pinhole_camera_with_no_distortion():
    """
    The returned camera has no distortion params left to apply twice.
    """
    from collab_splats.preproc.undistort import undistort_frames

    import numpy as np

    _, new_cam = undistort_frames(np.zeros((1, 1080, 1920, 3), np.uint8), _distorted_camera())

    assert list(new_cam.params[4:]) == []
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -k undistort_frames -v
```

Expected: FAIL — the current `undistort_frames` takes a `DistortionProfile` and returns `(frames, K, roi)`.

- [ ] **Step 3: Rewrite `undistort_frames`**

Replace the whole function with:

```python
def undistort_frames(
    frames_in: np.ndarray, camera: pycolmap.Camera
) -> tuple[np.ndarray, pycolmap.Camera]:
    """
    Undistort a stack of frames onto COLMAP's undistorted framing.

    Args:
        frames_in: (N, H, W, 3) uint8; H, W must match the camera.
        camera: a distorted pycolmap.Camera from calibrate_camera.

    Returns:
        ((N, H', W', 3) uint8, PINHOLE camera). The focal length is preserved and
        the canvas grows to hold the corners — the centre stays 1:1, nothing is
        resampled down to fit the original frame size.
    """
    frames_in = np.asarray(frames_in)
    if frames_in.ndim != 4 or frames_in.shape[1:3] != (camera.height, camera.width):
        raise ValueError(
            f"undistort_frames: frames are {frames_in.shape[1:3]}, camera is "
            f"{(camera.height, camera.width)}"
        )

    # COLMAP picks the framing: focal fixed, canvas sized to hold the corners
    new_cam = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), camera)

    # cv2 moves the pixels: one dst->src map, built once, reused for every frame
    map1, map2 = cv2.initUndistortRectifyMap(
        camera.calibration_matrix(),
        np.asarray(camera.params[4:], dtype=np.float64),
        None,
        new_cam.calibration_matrix(),
        (new_cam.width, new_cam.height),
        cv2.CV_32FC1,
    )
    out = np.stack([cv2.remap(f, map1, map2, cv2.INTER_LINEAR) for f in frames_in])

    logger.info(
        "undistorted %d frames: %dx%d -> %dx%d, f=%.1f preserved",
        len(out),
        camera.width,
        camera.height,
        new_cam.width,
        new_cam.height,
        new_cam.focal_length_x,
    )
    return out, new_cam
```

Deleted with it: `getOptimalNewCameraMatrix`, the even-ROI rounding (`w -= w % 2`), the crop, the `K_out[0,2] -= x` principal-point shift, and the `roi` return value. The principal-point shift is the classic silent bug in this pattern — a crop that moves the optical centre without updating `cx, cy` produces poses that are subtly wrong everywhere. It cannot happen now because there is no crop.

- [ ] **Step 4: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
isort collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
git commit --only collab_splats/preproc/undistort.py tests/preproc/test_undistort.py \
  -m "refactor(preproc)!: undistort_camera picks the framing, cv2.remap moves the pixels

BREAKING: undistort_frames takes a pycolmap.Camera and returns (frames, camera);
the roi return is gone. Output is now larger than the input at native focal
length, where it used to be input-sized at ~0.81x focal."
```

---

### Task 17: the reconstructor's undistort call site and provenance

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`_apply_undistortion`)
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor.py`:

```python
def test_undistort_provenance_records_the_camera_not_a_profile(tmp_path, monkeypatch):
    """
    Provenance carries pycolmap's camera dicts, and no roi.
    """
    import numpy as np
    import pycolmap

    from collab_splats.wrapper import reconstructor

    cam = pycolmap.Camera(
        model="OPENCV", width=64, height=48, params=[60.0, 60.0, 32.0, 24.0, -0.2, 0.0, 0.0, 0.0]
    )
    monkeypatch.setattr(reconstructor, "calibrate_camera", lambda d, **k: cam)

    prov = {}
    out = reconstructor._apply_undistortion(np.zeros((3, 48, 64, 3), np.uint8), tmp_path, prov)

    assert "roi" not in prov["undistort"]
    assert prov["undistort"]["camera"]["model"] == "OPENCV"
    assert prov["undistort"]["undistorted_camera"]["model"] == "PINHOLE"
    assert out.shape[1:3] == (
        prov["undistort"]["undistorted_camera"]["height"],
        prov["undistort"]["undistorted_camera"]["width"],
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_undistort_provenance_records_the_camera_not_a_profile -v
```

Expected: FAIL — `_apply_undistortion` still calls `estimate_camera_distortion` and stamps `profile`/`K_new`/`roi`.

- [ ] **Step 3: Rewrite `_apply_undistortion`**

```python
def _apply_undistortion(frame_arrays: np.ndarray, images_dir: Path, prov: dict) -> np.ndarray:
    """
    Calibrate from the extracted frames and undistort them in place.

    Args:
        frame_arrays: (N, H, W, 3) uint8 RGB, as selected.
        images_dir: where the frames were written — calibration reads them from here.
        prov: provenance dict, stamped with both cameras.

    Returns:
        (N, H', W', 3) uint8 RGB on the undistorted framing.
    """
    camera = calibrate_camera(images_dir)
    undistorted, new_camera = undistort_frames(frame_arrays, camera)

    # Both cameras, as COLMAP writes them — no local mirror of the same numbers
    prov["undistort"] = {
        "camera": camera.todict(),
        "undistorted_camera": new_camera.todict(),
    }
    return undistorted
```

Update the import line to `from collab_splats.preproc.undistort import calibrate_camera, undistort_frames`, and update the call site: `_apply_undistortion` now runs **after** the frames are written (calibration reads `images_dir`), so the sequence in `extract_frames` becomes write, calibrate-and-undistort, rewrite:

```python
    # Write once so calibration has images to read, then rewrite the undistorted stack
    frames.write_frames(images_dir, frame_arrays, records, prov)
    if undistort:
        frame_arrays = _apply_undistortion(frame_arrays, images_dir, prov)
        frames.write_frames(images_dir, frame_arrays, records, prov)
```

The double write is deliberate and cheap relative to SIFT + mapping. It is also the only ordering that lets `calibrate_camera` take a directory rather than an array.

If `pycolmap.Camera.todict()` is absent in 4.0.4, use `{"model": camera.model.name, "width": camera.width, "height": camera.height, "params": list(camera.params)}` — check first:

```bash
/opt/venv/reconstruction/bin/python -c "import pycolmap; print(hasattr(pycolmap.Camera, 'todict'))"
```

- [ ] **Step 3b: Repoint the VDA context decode at the camera**

`DistortionProfile` is gone, so `decode_context` and its one caller change with it. In `collab_splats/preproc/video.py`, `decode_context`'s `profile: DistortionProfile | None = None` becomes `camera: pycolmap.Camera | None = None`, and the undistort call inside its chunk loop becomes:

```python
            if camera is not None:
                chunk, _ = undistort_frames(chunk, camera)
```

The lazy import at the top of that function becomes `from collab_splats.preproc.undistort import undistort_frames`.

In `collab_splats/wrapper/reconstructor.py`, `_ensure_vda_depth` reads the camera back out of provenance instead of rebuilding a profile:

```python
                        # Decode with the same camera images/ was written with, or the
                        # context frames and the keyframes disagree on the framing
                        camera = None
                        if provenance.get("undistort"):
                            camera = pycolmap.Camera(**provenance["undistort"]["camera"])

                        # (the existing logger.info about the context stream stays here)
                        context_frames = decode_context(video_path, grid, camera=camera)
```

with `import pycolmap` at the top and `DistortionProfile` dropped from the import line. Note it reads `["camera"]` — the DISTORTED camera, because `decode_context` undistorts raw decoded frames, exactly as it did with the distorted profile before.

- [ ] **Step 4: Run the test and the wrapper suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
isort collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py \
  -m "refactor(wrapper)!: undistort provenance carries both pycolmap cameras

BREAKING: prov['undistort'] loses 'profile', 'K_new' and 'roi'."
```

---

### Task 18: split `extract_frames` and drop the `method` alias

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`extract_frames`)
- Test: `tests/wrapper/test_reconstructor_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_frames_has_no_method_alias():
    """
    `method = frame_selection` was a no-op rename inside the function.
    """
    import inspect

    from collab_splats.wrapper import reconstructor

    src = inspect.getsource(reconstructor.extract_frames)
    assert "method = frame_selection" not in src
    assert hasattr(reconstructor, "_frames_from_dir")
    assert hasattr(reconstructor, "_frames_from_video")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py::test_extract_frames_has_no_method_alias -v
```

Expected: FAIL, `AssertionError` on the alias.

- [ ] **Step 3: Split the function**

Lift the two branches out of `extract_frames` into module-level helpers with the same bodies:

```python
def _frames_from_dir(input_path: Path) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Every image in a directory, in filename order.

    Args:
        input_path: directory of images.

    Returns:
        (frames, records, provenance).
    """
    # Body is the directory branch of extract_frames, moved unchanged: the
    # frames.frame_paths listing, the empty-directory ValueError, the imread/cvtColor
    # comprehension, the records built from enumerate, and the {"method": "dir", ...}
    # provenance dict.


def _frames_from_video(
    input_path: Path,
    *,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    report: dict,
    n_workers: int,
    vda_context_fps: float | None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Frames selected from a video by one of the three sampling methods.

    Args:
        input_path: source video.
        frame_selection: "fps" | "uniform" | "optical_flow".
        fps: target rate for frame_selection="fps".
        min_frames: floor for the fps re-spread band.
        max_frames: cap; the contract for frame_selection="uniform".
        report: quality report from qa.compute_video_quality.
        n_workers: worker count for the quality pass.
        vda_context_fps: rate for the VDA context grid every keyframe must be a member of.

    Returns:
        (frames, records, provenance).
    """
    # Body is the video branch of extract_frames, moved unchanged: the report load or
    # compute, the vda_context_fps grid, the three-way frame_selection dispatch to
    # sample_fps / sample_uniform / sample_optical_flow, and the provenance dict.
```

`extract_frames` then reduces to: pick the branch, call it, write, optionally undistort and rewrite, plot. Delete the `method = frame_selection` line and use `frame_selection` throughout.

- [ ] **Step 4: Run the wrapper suite and the smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke
```

Expected: tests PASS; smoke exits 0.

- [ ] **Step 5: Commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/
isort collab_splats/wrapper/reconstructor.py tests/wrapper/
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/ \
  -m "refactor(wrapper): split extract_frames into _frames_from_dir/_frames_from_video"
```

---

## Phase D — PyAV replaces the ffmpeg subprocess

Measured on GH010229 (13,115 frames): full linear scan **76.2 s** against the ffmpeg pipe's **89.2 s** (1.17×), scattered index reads 2.19×, single-frame seek 10×, output pixel-identical (maxdiff 0). PyAV is never slower than what it replaces. The one strategy *inside* PyAV that is slower — seek-per-index over a scattered list, 97.4 s, because every seek lands on a keyframe and re-decodes forward — is not used; scattered reads go through the linear scan.

`ffprobe` stays, for rotation only. PyAV 17 cannot read a container display matrix: `av.sidedata` exposes `['encparams', 'motionvectors', 'sidedata']` with no stream-level accessor, and a decoded frame's `side_data` carries only `SEI_UNREGISTERED`. That is a missing capability, not a speed question — PyAV reads the rest of the metadata in 110 ms against ffprobe's 3.5 s.

---

### Task 19: **HARD GATE** — synthesize a rotated-video fixture

**Files:**
- Create: `tests/preproc/data/make_rotated_fixture.py`, `tests/preproc/data/rotated_90.mp4`

No fixture in the repo carries a rotation. The tutorial video is natively portrait (1080×1920) with `tags.rotate: None` and empty `side_data_list`, so it exercises nothing. Phase D changes the code path that reads rotation. Without this fixture that change is untested, and a rotated GoPro clip would silently decode transposed.

- [ ] **Step 1: Write the generator**

```python
"""
Generate tests/preproc/data/rotated_90.mp4 — a landscape clip carrying a 90-degree
display matrix, so its display dimensions are portrait.

Run once, commit the .mp4. Regenerate with:
    /opt/venv/reconstruction/bin/python tests/preproc/data/make_rotated_fixture.py
"""

import subprocess
from pathlib import Path

OUT = Path(__file__).parent / "rotated_90.mp4"


def main() -> None:
    # A 320x180 landscape clip with a moving bar, so a transposed decode is obvious
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=size=320x180:rate=10:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-metadata:s:v:0", "rotate=90",
            str(OUT),
        ],
        check=True,
        capture_output=True,
    )

    # Verify the display matrix actually landed — -metadata rotate= is silently
    # ignored by some muxers, in which case the fixture tests nothing
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
         "-show_streams", str(OUT)],
        capture_output=True, text=True, check=True,
    )
    if "Display Matrix" not in probe.stdout and '"rotate"' not in probe.stdout:
        raise SystemExit(
            f"{OUT} carries no rotation metadata — the muxer dropped it. Try "
            f"`-display_rotation 90` on the input instead."
        )

    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
/opt/venv/reconstruction/bin/python tests/preproc/data/make_rotated_fixture.py
```

Expected: `wrote .../rotated_90.mp4 (<some bytes>)`. If it exits with the muxer complaint, retry with `-display_rotation 90` before `-i` (ffmpeg 6+), which writes a real display matrix rather than a tag.

- [ ] **Step 3: Write the fixture's own test**

Append to `tests/preproc/test_video.py`:

```python
ROTATED = Path(__file__).parent / "data" / "rotated_90.mp4"


def test_rotated_fixture_reports_display_dimensions():
    """
    The fixture is 320x180 stored, 180x320 displayed. get_video_info reports displayed.
    """
    from collab_splats.preproc.video import get_video_info

    info = get_video_info(ROTATED)

    assert (info["width"], info["height"]) == (180, 320)


def test_rotated_fixture_decodes_upright():
    """
    ffmpeg auto-rotates, and whatever replaces it must too: decoded frames match
    the reported display dimensions.
    """
    from collab_splats.preproc.video import get_video_info, iter_frames

    info = get_video_info(ROTATED)
    _, frame = next(iter(iter_frames(ROTATED)))

    assert frame.shape[:2] == (info["height"], info["width"]) == (320, 180)
```

- [ ] **Step 4: Run them against the CURRENT code**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k rotated -v
```

Expected: **PASS**. This is the point of the gate — the tests must pass on today's ffmpeg implementation, so that when they fail after Task 20 or 21, the failure is unambiguously the new code.

If they fail here, the fixture is wrong, not the code. Fix the fixture.

- [ ] **Step 5: Commit and report to the user**

```bash
git add tests/preproc/data/make_rotated_fixture.py tests/preproc/data/rotated_90.mp4
git commit --only tests/preproc/data/make_rotated_fixture.py tests/preproc/data/rotated_90.mp4 tests/preproc/test_video.py \
  -m "test(preproc): rotated-video fixture, green against the ffmpeg implementation"
```

**Tell the user the fixture exists and passes on current code before starting Task 20.** Phase D has no other guard against a transposed decode.

---

### Task 20: `get_video_info` reads metadata through PyAV

**Files:**
- Modify: `collab_splats/preproc/video.py:56-100` (`get_video_info`), `_rotation_degrees`
- Test: `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_get_video_info_frame_count_is_exact(tiny_video):
    """
    PyAV reads stream.frames from the container — no full demux, no packet count.
    """
    from collab_splats.preproc.video import get_video_info

    info = get_video_info(tiny_video)

    assert info["total_frames"] > 0
    assert info["fps"] > 0
    assert abs(info["duration_s"] - info["total_frames"] / info["fps"]) < 0.05


def test_get_video_info_has_no_count_frames_flag():
    """
    The -count_packets full demux is gone; there is no cheap/expensive split left.
    """
    import inspect

    from collab_splats.preproc.video import get_video_info

    assert "count_frames" not in inspect.signature(get_video_info).parameters


def test_get_video_info_returns_zeros_for_a_non_video(tmp_path):
    """
    An unprobeable file returns the zeros dict rather than raising.
    """
    from collab_splats.preproc.video import get_video_info

    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nope")

    assert get_video_info(bad)["total_frames"] == 0
```

Delete any existing test that passes `count_frames=`.

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k get_video_info -v
```

Expected: the `count_frames` test FAILS; the others may pass already.

- [ ] **Step 3: Rewrite the probe**

Replace `get_video_info` with:

```python
def get_video_info(video_path: str | Path) -> dict:
    """
    Video metadata: PyAV for everything except rotation.

    Args:
        video_path: source video.

    Returns:
        {total_frames, fps, duration_s, width, height}, all zeros if unprobeable.
        Width/height are DISPLAY dims (rotation applied), matching what decode yields.
    """
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            fps = float(stream.average_rate) if stream.average_rate else 0.0
            width, height = stream.codec_context.width, stream.codec_context.height

            # stream.frames is the container's own count; fall back to duration x rate
            total = int(stream.frames)
            if total == 0 and stream.duration and stream.time_base:
                total = int(round(float(stream.duration * stream.time_base) * fps))
    except Exception:
        logger.debug("PyAV could not open %s", video_path, exc_info=True)
        return zeros

    # PyAV 17 cannot see the container display matrix, so rotation alone still
    # costs one ffprobe. Everything else above came from a 110 ms container parse.
    if _rotation_degrees(video_path) in (90, 270):
        width, height = height, width

    duration_s = total / fps if fps > 0 else 0.0
    return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}
```

and rewrite `_rotation_degrees` to take a path and do its own probe:

```python
def _rotation_degrees(video_path: str | Path) -> int:
    """
    CW display rotation, via ffprobe.

    Args:
        video_path: source video.

    Returns:
        0, 90, 180 or 270. Two metadata locations: legacy tags.rotate (CW) and
        Display Matrix side data (modern GoPro/iPhone; ffprobe reports CCW).

    PyAV 17 exposes neither, which is the only reason ffprobe survives in this module.
    """
    if shutil.which("ffprobe") is None:
        logger.debug("ffprobe missing; assuming no rotation for %s", video_path)
        return 0

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation", str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe rotation probe failed for %s", video_path, exc_info=True)
        return 0

    for s in streams:
        rotate = s.get("tags", {}).get("rotate")
        if rotate:
            return int(rotate) % 360

        for sd in s.get("side_data_list", []):
            if sd.get("rotation") is not None:
                return int(-sd["rotation"]) % 360

    return 0
```

Add `import av` at the top, delete `_require_ffmpeg` (nothing needs the ffmpeg *binary* any more; the rotation probe checks `shutil.which("ffprobe")` itself), and update the module docstring — it currently opens "the only module that shells out to ffmpeg/ffprobe", which is still true but for a different reason. Say so:

```python
"""
Video decode and probe.

PyAV decodes and reads metadata. ffprobe survives for exactly one field —
container display rotation, which PyAV 17 does not expose — and is skipped
entirely when it is not on PATH.

Colour convention: iter_frames yields BGR (what cv2 wants), extract_frame
returns RGB (what its consumers store). Both are uint8 HWC.
"""
```

- [ ] **Step 4: Fix the `count_frames=` callers**

```bash
git grep -n 'count_frames'
```

Every hit is a caller passing `count_frames=False` for the cheap path. Delete the kwarg — the cheap path is now the only path.

- [ ] **Step 5: Run the video suite, rotated fixture included**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -v
```

Expected: PASS, **including `test_rotated_fixture_reports_display_dimensions`**. If that one fails, the rotation probe is broken — do not proceed.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/video.py tests/preproc/test_video.py
isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py \
  -m "perf(preproc)!: get_video_info reads metadata via PyAV (3.5s -> 110ms)

BREAKING: count_frames= is gone; the count is exact and free."
```

---

### Task 21: `iter_frames` decodes through PyAV

**Files:**
- Modify: `collab_splats/preproc/video.py` (`iter_frames`)
- Test: `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_iter_frames_yields_bgr(tiny_video):
    """
    Colour convention is unchanged: iter_frames is BGR, extract_frame is RGB.
    """
    import numpy as np

    from collab_splats.preproc.video import extract_frame, iter_frames

    _, bgr = next(iter(iter_frames(tiny_video)))
    rgb = extract_frame(tiny_video, 0)

    assert np.array_equal(bgr[..., ::-1], rgb) or np.abs(
        bgr[..., ::-1].astype(int) - rgb.astype(int)
    ).max() <= 2


def test_iter_frames_indices_yields_exactly_those_indices(tiny_video):
    """
    Scattered reads come back in ascending order, once each.
    """
    from collab_splats.preproc.video import iter_frames

    got = [i for i, _ in iter_frames(tiny_video, indices=[7, 2, 2, 5])]

    assert got == [2, 5, 7]


def test_iter_frames_start_count_window(tiny_video):
    """
    start/count is a contiguous window in source frame indices.
    """
    from collab_splats.preproc.video import iter_frames

    got = [i for i, _ in iter_frames(tiny_video, start=4, count=3)]

    assert got == [4, 5, 6]


def test_iter_frames_stops_early_without_decoding_the_tail(tiny_video):
    """
    The last requested index ends the scan; the rest of the file is not decoded.
    """
    from collab_splats.preproc.video import get_video_info, iter_frames

    total = get_video_info(tiny_video)["total_frames"]
    assert total > 20

    got = [i for i, _ in iter_frames(tiny_video, indices=[1, 3])]
    assert got == [1, 3]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k iter_frames -v
```

Expected: PASS on the current implementation — these are the behaviours PyAV must preserve, written down before the rewrite so the rewrite has something to violate. Add `test_iter_frames_uses_no_subprocess`:

```python
def test_iter_frames_uses_no_subprocess(tiny_video, monkeypatch):
    """
    Decode is in-process; no ffmpeg pipe.
    """
    import subprocess

    import pytest

    from collab_splats.preproc.video import iter_frames

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("spawned an ffmpeg pipe"))

    assert len(list(iter_frames(tiny_video, indices=[0, 2]))) == 2
```

That one FAILS now.

- [ ] **Step 3: Rewrite `iter_frames` as a linear scan**

```python
def iter_frames(
    video_path: str | Path,
    *,
    indices: Sequence[int] | None = None,
    start: int = 0,
    count: int | None = None,
    info: dict | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, BGR uint8 HWC) in one decode pass.

    Args:
        video_path: source video.
        indices: yield only these source frame indices, ascending and deduped.
        start: first index of a contiguous window.
        count: length of that window; None runs to the end.
        info: accepted and ignored — kept so callers need not change.

    Returns:
        An iterator of (source frame index, BGR frame).

    One linear scan, always. Seeking to each requested index is SLOWER on a
    scattered list (97.4 s against 76.2 s on a 13k-frame clip) because every seek
    lands on a keyframe and re-decodes forward from it.
    """
    wanted = sorted({int(i) for i in indices}) if indices is not None else None
    if wanted is not None and not wanted:
        return

    # The scan stops at the last frame anyone asked for
    if wanted is not None:
        last = wanted[-1]
    elif count is not None:
        last = start + count - 1
    else:
        last = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        # Let PyAV use every core it is allowed; decode is the whole cost here
        stream.thread_type = "AUTO"

        cursor = 0
        pending = iter(wanted) if wanted is not None else None
        target = next(pending, None) if pending is not None else None

        for frame in container.decode(stream):
            if last is not None and cursor > last:
                break

            if wanted is not None:
                if target is None:
                    break
                take = cursor == target
                if take:
                    target = next(pending, None)
            else:
                take = cursor >= start and (count is None or cursor < start + count)

            if take:
                # to_ndarray already applies the container's rotation
                yield cursor, frame.to_ndarray(format="bgr24")

            cursor += 1
```

`info=` is kept and ignored on purpose: it exists in a dozen call sites purely to hoist an ffprobe that no longer happens. Deleting the parameter would be a wider change than this task, and Task 24 sweeps it.

- [ ] **Step 4: Run the video suite and the rotated fixture**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v
```

Expected: PASS, **including `test_rotated_fixture_decodes_upright`**. If that fails, `to_ndarray` is not applying the display matrix and the decode needs an explicit rotate — do not paper over it by transposing in the caller.

- [ ] **Step 5: Confirm the speed claim on real footage**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/ada04594-773b-4b10-b55a-fa414346cead/scratchpad
/opt/venv/reconstruction/bin/python -c "
import time
from collab_splats.preproc.video import iter_frames
t = time.perf_counter()
n = sum(1 for _ in iter_frames('<path to GH010229.MP4>'))
print(f'{n} frames in {time.perf_counter() - t:.1f} s')
"
```

Expected: ~76 s for 13,115 frames. Materially slower means `thread_type = "AUTO"` did not take effect.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/video.py tests/preproc/test_video.py
isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py \
  -m "perf(preproc): iter_frames decodes in-process via PyAV (1.17x scan, 2.19x scattered)"
```

---

### Task 22: `extract_frame` seeks through PyAV

**Files:**
- Modify: `collab_splats/preproc/video.py` (`extract_frame`)
- Test: `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_frame_matches_the_scan(tiny_video):
    """
    Seeking to frame N returns the same pixels the scan yields at N.
    """
    import numpy as np

    from collab_splats.preproc.video import extract_frame, iter_frames

    scanned = dict(iter_frames(tiny_video, indices=[12]))
    seeked = extract_frame(tiny_video, 12)

    assert np.abs(scanned[12][..., ::-1].astype(int) - seeked.astype(int)).max() <= 2


def test_extract_frame_rejects_an_out_of_range_index(tiny_video):
    """
    Past the end is an error, not a silently-clamped last frame.
    """
    import pytest

    from collab_splats.preproc.video import extract_frame, get_video_info

    total = get_video_info(tiny_video)["total_frames"]
    with pytest.raises(ValueError, match="out of range"):
        extract_frame(tiny_video, total + 5)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k extract_frame -v
```

Expected: the match test may pass; the range test's message may differ. Adjust the `match=` to the message you actually write, not the other way round.

- [ ] **Step 3: Rewrite `extract_frame`**

```python
def extract_frame(video_path: str | Path, frame_idx: int, *, info: dict | None = None) -> np.ndarray:
    """
    Decode one frame by seeking to it.

    Args:
        video_path: source video.
        frame_idx: source frame index.
        info: a get_video_info dict, to skip the range check's own probe.

    Returns:
        (H, W, 3) uint8 RGB.

    Seek-then-scan-forward, which is 10x a full scan for ONE frame and the reason
    this is a separate function from iter_frames. It is exact on CFR video and may
    land one frame off near a keyframe on a VFR source.
    """
    info = info if info is not None else get_video_info(video_path)
    total, fps = info["total_frames"], info["fps"]

    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"frame {frame_idx} out of range for {video_path} ({total} frames)")
    if not fps:
        raise ValueError(f"cannot seek {video_path}: missing fps")

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Seek lands on the keyframe at or before the target; decode forward from there
        target_pts = int(frame_idx / fps / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.pts is None or frame.pts >= target_pts:
                return frame.to_ndarray(format="rgb24")

    raise ValueError(f"decode of {video_path} ended before reaching frame {frame_idx}")
```

- [ ] **Step 4: Run the video suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/video.py tests/preproc/test_video.py
isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py \
  -m "perf(preproc): extract_frame seeks in-process via PyAV (10x)"
```

---

### Task 23: move `decode_context` to the module that uses it

**Files:**
- Modify: `collab_splats/preproc/video.py` (delete `decode_context`), `collab_splats/pointcloud/sfm.py` (add it), `collab_splats/wrapper/reconstructor.py` (import)
- Test: move the `decode_context` tests from `tests/preproc/test_video.py` to `tests/pointcloud/test_sfm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_decode_context_lives_with_vda():
    """
    It exists to feed generate_vda_depth; it is not general video decode.
    """
    from collab_splats.pointcloud import sfm
    from collab_splats.preproc import video

    assert hasattr(sfm, "decode_context")
    assert not hasattr(video, "decode_context")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_sfm.py::test_decode_context_lives_with_vda -v
```

Expected: FAIL, `AttributeError`.

- [ ] **Step 3: Move it**

Cut `decode_context` (with the `camera=` signature from Task 17's Step 3b) out of `collab_splats/preproc/video.py` and paste it into `collab_splats/pointcloud/sfm.py`, above `generate_vda_depth`. Its imports move with it:

```python
from collab_splats.preproc.undistort import undistort_frames
from collab_splats.preproc.video import get_video_info, iter_frames
```

The lazy import inside the function becomes a top-level one — the cycle it was avoiding (`video` → `undistort`) does not exist from `sfm`.

Update `collab_splats/wrapper/reconstructor.py`:

```python
from collab_splats.pointcloud.sfm import decode_context
from collab_splats.preproc.sampling import context_indices
```

Move the `decode_context` tests from `tests/preproc/test_video.py` into `tests/pointcloud/test_sfm.py`, adjusting only their import line.

- [ ] **Step 4: Run both suites**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/ tests/wrapper/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats tests
isort collab_splats tests
git commit --only collab_splats/preproc/video.py collab_splats/pointcloud/sfm.py collab_splats/wrapper/reconstructor.py tests/preproc tests/pointcloud \
  -m "refactor(preproc)!: decode_context moves to pointcloud.sfm, next to its only caller

BREAKING: preproc.video.decode_context is now pointcloud.sfm.decode_context."
```

---

### Task 24: promote `av` to a direct dependency and sweep the dead `info=`

**Files:**
- Modify: `pyproject.toml`, `collab_splats/preproc/video.py`, every `info=` call site

- [ ] **Step 1: Confirm av's provenance**

```bash
/opt/venv/reconstruction/bin/python -c "import av; print(av.__version__)"
grep -n '^ *"av' uv.lock | head
```

Expected: `17.0.1`, arriving transitively. A module we now import directly must be declared directly.

- [ ] **Step 2: Declare it**

In `pyproject.toml`, add to `dependencies` (alphabetical position, before `black` or wherever the list orders it):

```toml
    "av>=17.0",
```

- [ ] **Step 3: Sync and verify**

```bash
uv sync
/opt/venv/reconstruction/bin/python -c "import av; print(av.__version__)"
```

Expected: still `17.0.1`, now pinned by us. **Do not run a bare `uv sync` that prunes extras** — this project's extras carry VGGT-X and MapAnything; use the same invocation `setup.sh` uses if `uv sync` alone drops them.

- [ ] **Step 4: Delete the dead `info=` parameter**

`iter_frames` ignores it, and `extract_frame` uses it only to skip a now-cheap probe.

```bash
git grep -n 'info=info\|info=\(get_video_info\|_info\)'
```

Remove `info=` from `iter_frames` entirely (signature and every call site). Keep it on `extract_frame` — it still saves a probe per call in a loop, and the probe is 110 ms, not free.

- [ ] **Step 5: Run everything**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke
```

Expected: green apart from `docs/known-test-failures.md`; smoke exits 0.

- [ ] **Step 6: Commit**

```bash
black collab_splats tests
isort collab_splats tests
git commit --only pyproject.toml uv.lock collab_splats tests \
  -m "build: declare av as a direct dependency; drop iter_frames' dead info="
```

---

## Phase E — delete the dead, collapse the redundant, fix the docstrings

---

### Task 25: delete `plot_disparity_sensitivity` and `plot_quality_examples`

> **Runs in wave 0, before Task 9.** `plot_quality_examples` is the only `FrameStore`
> consumer in `viz.py`, and no Phase A task converts that file. Deleting it here is
> what lets Task 9's verification grep come back clean.

**Files:**
- Modify: `collab_splats/preproc/viz.py:102-188`, `tests/preproc/test_viz.py`, `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`, `docs/known-test-failures.md`

`plot_quality_examples` has been broken since the FrameStore rework — the tutorial notebook carries its traceback in committed output, and `docs/known-test-failures.md` lists it. Its "Rejected: soft" split hardcodes `lap < 50.0`, the threshold Task 10 deleted. `plot_disparity_sensitivity` re-thresholds records through a formula that ignores the selector's stateful keyframe updates, so its counts are approximate by its own docstring and nothing consumes them.

- [ ] **Step 1: Write the failing test**

Replace the `plot_disparity_sensitivity` and `plot_quality_examples` tests in `tests/preproc/test_viz.py` with:

```python
def test_dead_plots_are_gone():
    """
    Both were broken or approximate, and nothing outside the notebook called them.
    """
    from collab_splats.preproc import viz

    assert not hasattr(viz, "plot_disparity_sensitivity")
    assert not hasattr(viz, "plot_quality_examples")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py::test_dead_plots_are_gone -v
```

Expected: FAIL, `AssertionError`.

- [ ] **Step 3: Delete**

Remove both functions from `collab_splats/preproc/viz.py` (lines 102-188) and the imports they alone needed — `OpticalFlowFrameSelector` and `filter_frame_quality`, if nothing else in the module uses them:

```bash
git grep -n 'OpticalFlowFrameSelector\|filter_frame_quality' -- collab_splats/preproc/viz.py
```

- [ ] **Step 4: Fix the notebook**

`docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` has three call cells (`plot_quality_examples(video_lookup, frame_scores)`, `plot_quality_examples(video_lookup, demo_scores)`, `plot_disparity_sensitivity(frame_scores, disparity_values)`), their imports, the `_VideoFrameLookup` shim that existed only to fake a store for them, the `disparity_values` definition, and the surrounding markdown. Delete all of it with the NotebookEdit tool, then re-run the notebook end to end and commit the executed output.

The notebook keeps `plot_selection` and `plot_frame_extremes`, which work.

- [ ] **Step 5: Drop the known-failure entry**

Remove the `plot_quality_examples` row from `docs/known-test-failures.md`. Deleting the function is the fix.

- [ ] **Step 6: Run and commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -v
black collab_splats/preproc/viz.py tests/preproc/test_viz.py
isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py docs/known-test-failures.md docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb \
  -m "refactor(preproc)!: delete plot_disparity_sensitivity and plot_quality_examples

BREAKING: both leave preproc.viz. plot_quality_examples had been raising since
the frame-store rework; plot_disparity_sensitivity's counts were approximate."
```

---

### Task 26: collapse the qa pair-motion four into one

**Files:**
- Modify: `collab_splats/preproc/qa.py` (`detect_orb`, `match_descriptors`, `compute_translation`, `compute_parallax` → `compute_pair_motion`)
- Test: `tests/preproc/test_qa.py`

The four are called in exactly one place, always in the same order, always on the same two frames. `detect_orb` stays separate — the worker caches its result to detect once per frame rather than twice per pair, a measured 1.34× win. The other three collapse.

- [ ] **Step 1: Write the failing test**

```python
def test_compute_pair_motion_returns_all_three_measures():
    """
    One call per pair, replacing match + translation + parallax.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc.qa import compute_pair_motion, detect_orb

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (300, 400), dtype=np.uint8)
    a, b = canvas[:240, :320], canvas[10:250, 8:328]

    row = compute_pair_motion(detect_orb(a), detect_orb(b))

    assert set(row) == {"n_matches", "translation_px", "parallax"}
    assert row["n_matches"] > 20
    assert 8.0 < row["translation_px"] < 20.0


def test_compute_pair_motion_on_an_unmatchable_pair():
    """
    Two unrelated frames give zero matches and zeroed measures, not an exception.
    """
    import numpy as np

    from collab_splats.preproc.qa import compute_pair_motion, detect_orb

    a = np.zeros((240, 320), np.uint8)
    b = np.full((240, 320), 255, np.uint8)

    row = compute_pair_motion(detect_orb(a), detect_orb(b))

    assert row["n_matches"] == 0
    assert row["translation_px"] == 0.0
    assert row["parallax"] == 0.0


def test_the_three_collapsed_helpers_are_gone():
    """
    match_descriptors/compute_translation/compute_parallax had one caller between them.
    """
    from collab_splats.preproc import qa

    for dead in ("match_descriptors", "compute_translation", "compute_parallax"):
        assert not hasattr(qa, dead), dead
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k pair_motion -v
```

Expected: FAIL, `ImportError: cannot import name 'compute_pair_motion'`.

- [ ] **Step 3: Collapse**

In `collab_splats/preproc/qa.py`, replace `match_descriptors`, `compute_translation` and `compute_parallax` with one function whose body is their three bodies in sequence:

```python
def compute_pair_motion(feat_a: tuple, feat_b: tuple, *, ransac_thresh_px: float = 3.0) -> dict:
    """
    Match two frames' ORB features and measure the motion between them.

    Args:
        feat_a: (keypoints, descriptors) from detect_orb, the earlier frame.
        feat_b: (keypoints, descriptors) from detect_orb, the later frame.
        ransac_thresh_px: homography inlier threshold for the parallax estimate.

    Returns:
        {n_matches, translation_px, parallax}. All zero when the pair does not match.
    """
    # match_descriptors' body: BFMatcher(NORM_HAMMING, crossCheck=True), the empty-descriptor
    # guard, and the two (N, 2) float32 point arrays it returns.
    # then compute_translation's body: median L2 norm of (pts_b - pts_a).
    # then compute_parallax's body: findHomography(RANSAC, ransac_thresh_px) and the median
    # residual of pts_b against the homography-warped pts_a.
```

The three bodies move in unchanged, and their three separate empty-match guards collapse into one at the top:

```python
    if feat_a[1] is None or feat_b[1] is None or len(feat_a[1]) == 0 or len(feat_b[1]) == 0:
        return {"n_matches": 0, "translation_px": 0.0, "parallax": 0.0}
```

This is a call-shape change, not a behaviour change. Keep `detect_orb` exactly as it is.

- [ ] **Step 4: Update the worker call site**

In `collab_splats/preproc/qa.py` (~lines 305-335):

```python
        pending[idx] = detect_orb(gray_small)
        partner = idx - stride
        if partner in pending:
            pair_rows.append(
                {
                    "frame_idx_a": partner,
                    "frame_idx_b": idx,
                    **compute_pair_motion(pending[partner], pending[idx]),
                }
            )
            del pending[partner]
```

- [ ] **Step 5: Run the qa suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v
```

Expected: PASS. Delete the tests of the three collapsed helpers — their behaviour is now covered by `compute_pair_motion`'s.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/qa.py tests/preproc/test_qa.py
isort collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "refactor(preproc)!: collapse pair motion into compute_pair_motion

BREAKING: match_descriptors, compute_translation and compute_parallax leave
preproc.qa. detect_orb stays — the worker caches it per frame."
```

---

### Task 27: the docstring pass, and the rule that keeps it

**Files:**
- Modify: every public function in `collab_splats/preproc/`, `CLAUDE.md`
- Test: `tests/preproc/test_docstrings.py` (create)

The brief's third item: docstrings state what a function does, what its inputs are, and what its outputs are — not paragraphs. Tasks 10-26 already wrote the new and rewritten functions in this form. This task sweeps whatever they did not touch and pins the rule so it does not erode.

- [ ] **Step 1: Write the enforcing test**

Create `tests/preproc/test_docstrings.py`:

```python
"""
The docstring contract for preproc's public surface.

Every public function documents its inputs and its outputs. This is a lint, not
a behaviour test — it exists because the alternative is a slow drift back to
paragraphs that describe neither.
"""

import inspect

import pytest

from collab_splats import preproc

PUBLIC = [
    (name, getattr(preproc, name))
    for name in preproc.__all__
    if inspect.isfunction(getattr(preproc, name))
]


@pytest.mark.parametrize("name,fn", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_public_function_documents_its_inputs_and_outputs(name, fn):
    doc = inspect.getdoc(fn)
    assert doc, f"{name} has no docstring"

    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    if params:
        assert "Args:" in doc, f"{name} takes {params} and documents none of them"
        for p in params:
            assert f"{p}:" in doc, f"{name} does not document '{p}'"

    if "-> None" not in str(inspect.signature(fn)):
        assert "Returns:" in doc, f"{name} returns something and documents nothing"


@pytest.mark.parametrize("name,fn", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_summary_is_one_line(name, fn):
    doc = inspect.getdoc(fn)
    summary = doc.split("\n\n")[0]

    assert "\n" not in summary, f"{name}'s summary spans multiple lines"
    assert len(summary) <= 100, f"{name}'s summary is {len(summary)} chars"
    assert not summary.startswith(name), f"{name}'s summary restates its own name"
```

- [ ] **Step 2: Run it and read the failures**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_docstrings.py -v
```

Expected: a list of exactly which functions still need work. That list is this task's checklist.

- [ ] **Step 3: Fix each one**

For every failure, rewrite the docstring in the house form:

```python
def f(a, b):
    """
    One line saying what it does.

    Args:
        a: what a is.
        b: what b is.

    Returns:
        What comes back.
    """
```

Deleting is as valid as rewriting: a bullet that restates the signature, explains a parameter that no longer exists, or narrates history goes.

- [ ] **Step 4: Run it clean**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_docstrings.py -v
```

Expected: PASS, every parametrized case.

- [ ] **Step 5: Pin the rule in `CLAUDE.md`**

In the **Code Style** section, replace the existing `**Docstrings:**` bullet with:

```markdown
- **Docstrings:** every public function and class gets a one-line summary docstring. The `"""` open and close on their own lines — summary starts on the line after the opening quotes, never on the same line. Multi-line docstrings put a blank line between the summary and what follows. A function with parameters documents every one under `Args:`; a function that returns something documents it under `Returns:`. Bullets or Args/Returns, not prose blocks. No restating the function name. No padding. `tests/preproc/test_docstrings.py` enforces this for `preproc`; extend it as other modules are cleaned up.
```

- [ ] **Step 6: Run everything, one last time**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke
graphify update .
```

Expected: green apart from `docs/known-test-failures.md`; smoke exits 0.

- [ ] **Step 7: Commit**

```bash
black collab_splats/preproc tests/preproc
isort collab_splats/preproc tests/preproc
git commit --only collab_splats/preproc tests/preproc CLAUDE.md \
  -m "docs(preproc): Args/Returns on every public function, enforced by test"
```

---

### Task 28: close the work out

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/CHANGELOG.md`

- [ ] **Step 1: Count what happened**

```bash
/opt/venv/reconstruction/bin/python -c "
from pathlib import Path
total = 0
for p in sorted(Path('collab_splats/preproc').glob('*.py')):
    n = len(p.read_text().splitlines())
    total += n
    print(f'{n:5d}  {p}')
print(f'{total:5d}  TOTAL  (was 2268)')
"
```

Expected: ~1485. A number materially above that means something the spec said to delete is still there — find it before writing the changelog.

- [ ] **Step 2: Write the changelog entry**

Append to `docs/superpowers/CHANGELOG.md`, following the format of the entries above it: what changed per module, what broke, and the measured numbers — the filter's 5.1%/14.5% cuts from Task 14, PyAV's 1.17×/2.19×/10×, undistortion's framing change, and the line count.

- [ ] **Step 3: Update `CLAUDE.md`**

Remove `preproc-centralization` from **In-Flight Work** if it was listed. Update the `preproc/` block of the architecture tree:

```
  preproc/                 # video preprocessing: measure (qa) then select (sampling)
    video.py               # PyAV decode: get_video_info, iter_frames, extract_frame
    qa.py                  # report-only capture quality: compute_video_quality, load_video_quality
    sampling.py            # context_indices + sample_fps | sample_uniform | sample_optical_flow, all from filter_frame_quality's eligible pool
    frames.py              # images/frame_NNNNNN.png + frames.json: the COLMAP-style keyframe store
    undistort.py           # calibrate_camera (pycolmap) + undistort_frames (pycolmap framing, cv2 pixels)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
```

- [ ] **Step 4: Commit**

```bash
git commit --only CLAUDE.md docs/superpowers/CHANGELOG.md \
  -m "docs: preproc centralization changelog"
```

- [ ] **Step 5: Tell the user what shipped**

Report the line count, the two gate results, and the four breaking changes a caller outside this repo would hit: `FrameStore` is gone, `filter_frame_quality`'s kwargs changed, `undistort_frames` takes a camera and returns a larger image, and `preproc.video.context_indices` / `decode_context` moved.
