# preprocess_gdrive_videos.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `scripts/flatten_dataset.py` with a single script that pairs each DaVinci Resolve export against its camera original, flattens the Drive tree into `environments-curated/`, and carries capture metadata across — static tags plus a retimed `gpmd` track injected into the curated mp4, a full-rate telemetry sidecar, and a two-column CSV index.

**Architecture:** One script, `scripts/preprocess_gdrive_videos.py`, with `########` section dividers matching the file it replaces. Stages run in order `pair → copy → align → extract → inject`, with `push` behind `--push`. All video and metadata I/O shells out to `ffmpeg`, `ffprobe` and `exiftool`; the pure logic (pairing, DMS formatting, offset solving, GPMF sample expansion, CSV emission) is written as importable module-level functions so it is directly testable without decoding a real video.

**Tech Stack:** Python 3, `numpy`, `scipy.signal` (FFT cross-correlation), `pyarrow` (Parquet), stdlib `csv`/`json`/`subprocess`/`argparse`, external `ffmpeg`/`ffprobe`/`exiftool`, `rclone` via the existing `scripts/push_curated.sh`.

**Spec:** `docs/superpowers/specs/2026-08-11-preprocess-gdrive-videos-design.md`

## Global Constraints

- Single file: `scripts/preprocess_gdrive_videos.py`. No new package under `collab_splats/`. This was considered and rejected as over-complication.
- Repo style (`CLAUDE.md`): imports at top, `########` section dividers, one-line docstrings, block-level comments explaining *why*, `logging` not `print()`, hard imports (no `try: import`), no premature abstraction.
- Line length 120 (`[tool.black] line-length = 120`, `[tool.flake8] max-line-length = 120`).
- `scripts/` is **not** covered by `scripts/lint.sh` (it lints `tests/ collab_splats/` only). Run `ruff check scripts/preprocess_gdrive_videos.py` by hand before each commit.
- `scripts/` is not an importable package. Tests load the module by file path with `importlib.util.spec_from_file_location`, exactly as the existing test file does.
- pytest is configured with `--import-mode=importlib`, `testpaths = ["./tests"]`, `pythonpath = ["."]`. Run a single test file with `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v`.
- **Nothing on disk is renamed.** `collab_splats/dashboard/app.py:87` matches curated video stems against processed-output stems; renaming a curated mp4 breaks it.
- Alignment acceptance is a single gate: normalized cross-correlation `r >= 0.95`. Constant `DEFAULT_ALIGN_MIN_R = 0.95`, flag `--align-min-r`.
- Audio for alignment is decoded mono at 8000 Hz: `ffmpeg -vn -ac 1 -ar 8000 -f f32le -`.
- GPS in the CSV formats as `42 deg 21' 11.52" N, 71 deg 3' 57.24" W`. Missing GPS is a **blank cell** — never `0,0`, never a sentinel.
- The CSV has exactly two columns: `unique_id,gps`.
- `scripts/push_curated.sh` is not modified. It is invoked via `subprocess`.
- `docs/superpowers/` is gitignored in this repo, so the spec and this plan are deliberately untracked. Do not `git add -f` them.
- Commit after each task. Do not amend or squash across tasks.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/preprocess_gdrive_videos.py` | Everything: pairing, copy, alignment, extraction, injection, indexing, CLI |
| `tests/scripts/test_preprocess_gdrive_videos.py` | All tests; renamed from `test_flatten_dataset.py` |
| `scripts/flatten_dataset.py` | Deleted in Task 11 |
| `scripts/push_curated.sh` | Unchanged; called via subprocess |
| `pyproject.toml` | Add `pyarrow` to `[project] dependencies` |
| `README.md` | Rewrite the curated-video block, lines ~92-103 |

---

### Task 1: Scaffold the script and port the pure naming helpers

Ports `sanitize`, `find_videos` and `flat_dir_name` unchanged from `flatten_dataset.py`, and renames the test file. The old script stays on disk until Task 11 so the repo never has a broken README reference mid-plan.

**Files:**
- Create: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py` (renamed from `tests/scripts/test_flatten_dataset.py`)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `sanitize(name: str) -> str`
  - `find_videos(folder: Path) -> list[Path]`
  - `flat_dir_name(date: str, parent: str, video: Path) -> str`
  - Constants `_VIDEO_EXTS: set[str]`, `_DATE_RE: re.Pattern`, `DEFAULT_SOURCE_ROOT: Path`, `DEFAULT_OUTPUT_ROOT: Path`

- [ ] **Step 1: Rename the test file with git so history follows**

```bash
git mv tests/scripts/test_flatten_dataset.py tests/scripts/test_preprocess_gdrive_videos.py
```

- [ ] **Step 2: Repoint the module loader and docstring at the top of the renamed test**

Replace lines 1-12 of `tests/scripts/test_preprocess_gdrive_videos.py` with:

```python
"""Tests for scripts/preprocess_gdrive_videos.py."""

import csv
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not an importable package, so load the module by file path
_SCRIPT = Path(__file__).parents[2] / "scripts" / "preprocess_gdrive_videos.py"
_spec = importlib.util.spec_from_file_location("preprocess_gdrive_videos", _SCRIPT)
preproc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preproc)
```

Then replace every remaining `flatten_dataset.` prefix in the file with `preproc.`:

```bash
sed -i '' 's/flatten_dataset\./preproc./g' tests/scripts/test_preprocess_gdrive_videos.py
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v`
Expected: collection error — `FileNotFoundError` or `AttributeError: 'NoneType' object has no attribute 'loader'`, because `scripts/preprocess_gdrive_videos.py` does not exist yet.

- [ ] **Step 4: Create the script with its header and the three ported helpers**

Create `scripts/preprocess_gdrive_videos.py`:

```python
#!/usr/bin/env python3
"""Flatten the Google Drive capture tree and carry capture metadata into the curated videos.

DaVinci Resolve strips every timed data track and container tag on export: the camera
original carries ``tmcd + gpmd + fdsc``, GPS and the GoPro UserData block, while the export
carries ``tmcd`` alone. This script pairs each export against its original, copies it into
the flat curated layout, solves the trim offset from audio, and writes the metadata back —
as static tags plus a retimed ``gpmd`` track inside the mp4, and as a full-rate Parquet
sidecar next to it.

Source layout, where the parent folder name is not stable across dates (``GoproSplat``,
``Goprosplat``, ``splats``, ``Phone pics and splat videos``)::

    <source-root>/<YYYY-MM-DD>/<parent>/<stem>.mp4       <- Resolve export (the edit)
                                       /src/<stem>.MP4   <- camera original

A video is an *edit* if and only if ``<parent>/src/<stem>.*`` exists, matched on stem only,
case-insensitively, across any video extension. Only pairs are processed; an original with
no export is skipped and logged, and enters scope automatically once it is edited.

Output layout — one flat folder per pair::

    <output-root>/<YYYY_MM_DD>-<parent>-<stem>/
        <stem>.mp4                  edit + static tags + retimed gpmd track
        <stem>_metadata.json        static tags, provenance, alignment scores
        <stem>_telemetry.parquet    ~200 Hz IMU and GPS samples (only when IMU exists)
    <output-root>/index.csv         unique_id,gps — regenerated from the sidecars

Usage:
    python scripts/preprocess_gdrive_videos.py --dry-run
    python scripts/preprocess_gdrive_videos.py
    python scripts/preprocess_gdrive_videos.py --only GH010234
    python scripts/preprocess_gdrive_videos.py --index-only
    python scripts/preprocess_gdrive_videos.py --push
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

########
# Constants
########

_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_SOURCE_ROOT = _REPO_ROOT.parent / "gdrive-src"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT.parent / "environments-curated"
PUSH_SCRIPT = _REPO_ROOT / "scripts" / "push_curated.sh"

_VIDEO_EXTS = {".mp4", ".mov", ".avi"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Audio is decoded mono at this rate purely to solve the trim offset; one sample is 125 us,
# far finer than a 59.94 fps frame, so the offset is frame-exact.
AUDIO_RATE = 8000

# The edit's audio is a literal subsegment of the source, so a true match correlates near
# 1.0 and a wrong lag lands near 1/sqrt(N). See the spec for why the gap is this wide.
DEFAULT_ALIGN_MIN_R = 0.95

INDEX_NAME = "index.csv"


########
# Naming
########


def sanitize(name):
    """Replace runs of non-alphanumeric characters with a single underscore."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def find_videos(folder):
    """Return videos sitting directly inside `folder`, sorted; never recurses."""
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _VIDEO_EXTS)


def flat_dir_name(date, parent, video):
    """Build the flat folder name YYYY_MM_DD-PARENT-VIDEO for one video.

    The video stem is carried over verbatim so the folder is traceable back to the original
    filename (e.g. PXL_20260630_002106958.TS keeps its dot). Only the parent folder is
    sanitized. The stem is last, so any hyphens it contains stay unambiguous when splitting
    the name on the first two hyphens.
    """
    return f"{date.replace('-', '_')}-{sanitize(parent)}-{video.stem}"
```

- [ ] **Step 5: Run the naming tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "sanitize or flat_dir_name"`
Expected: 12 PASS (8 `sanitize` parametrizations, 4 `flat_dir_name` cases).

The `plan_copies` and `copy_video` tests will still fail with `AttributeError` — that is expected and gets fixed in Tasks 2 and 3.

- [ ] **Step 6: Lint and commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): scaffold preprocess_gdrive_videos with ported naming helpers"
```

---

### Task 2: Pairing rule — `plan_pairs`

This is the core replacement for `plan_copies` and the reason the old script produced a wrong tree. Two bugs are fixed here: date-folder-root videos were invisible, and videos in folders with no `src/` sibling were misclassified as exports when they are actually camera originals.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `sanitize`, `find_videos`, `flat_dir_name`, `_VIDEO_EXTS`, `_DATE_RE`
- Produces:
  - `class Pair(NamedTuple): edit: Path; source: Path; name: str`
  - `find_source(video: Path) -> Path | None`
  - `plan_pairs(source_root: Path) -> list[Pair]`

- [ ] **Step 1: Delete the three obsolete `plan_copies` tests**

Remove `test_plan_copies_skips_src_only_folders`, `test_plan_copies_ignores_undated_top_level_dirs`, `test_plan_copies_raises_on_name_collision` and the `tree` fixture from `tests/scripts/test_preprocess_gdrive_videos.py`. They test a rule that no longer exists — in particular the first one asserts `GH010229` is included, but `GH010229` has no `src/` counterpart, so it is now skipped by design.

- [ ] **Step 2: Write the failing tests**

Add to `tests/scripts/test_preprocess_gdrive_videos.py`, replacing the deleted block:

```python
########
# plan_pairs
########


def _touch(path):
    """Create an empty file, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def tree(tmp_path):
    """Miniature capture tree mirroring the real gdrive-src shapes."""
    root = tmp_path / "gdrive-src"
    # Pair with an extension AND case difference: .mp4 edit, .MP4 original
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010228.mp4")
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    # Edit with no original in src/: skipped
    _touch(root / "2026-07-15" / "Goprosplat" / "GH010229.mp4")
    # Original with no edit: skipped
    _touch(root / "2026-06-29" / "GoproSplat" / "src" / "GH010221.MP4")
    # Spaces in the parent name, and a .mp4 edit against a .MOV original
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "IMG_4085.mp4")
    _touch(root / "2026-06-29" / "Phone pics and splat videos" / "src" / "IMG_4085.MOV")
    # Videos directly in a date folder with no src/ anywhere: camera originals, skipped
    _touch(root / "2026-06-03" / "GH010218.MP4")
    # Noise that must never be picked up
    _touch(root / "2026-07-15" / "Goprosplat" / ".DS_Store")
    _touch(root / "notes.txt")
    _touch(root / "scratch" / "whatever.mp4")
    return root


def test_plan_pairs_returns_only_videos_with_a_src_counterpart(tree):
    names = [p.name for p in preproc.plan_pairs(tree)]
    assert names == [
        "2026_06_29-Phone_pics_and_splat_videos-IMG_4085",
        "2026_07_15-Goprosplat-GH010228",
    ]


def test_plan_pairs_matches_across_case(tree):
    pair = next(p for p in preproc.plan_pairs(tree) if p.name.endswith("GH010228"))
    assert pair.edit.name == "GH010228.mp4"
    assert pair.source.name == "GH010228.MP4"


def test_plan_pairs_matches_across_extension(tree):
    pair = next(p for p in preproc.plan_pairs(tree) if p.name.endswith("IMG_4085"))
    assert pair.edit.suffix == ".mp4"
    assert pair.source.suffix == ".MOV"


def test_plan_pairs_ignores_undated_top_level_dirs(tree):
    assert all("scratch" not in p.edit.parts for p in preproc.plan_pairs(tree))


def test_plan_pairs_walks_videos_directly_in_a_date_folder(tmp_path):
    # The old plan_copies never looked here, so 2026-06-03/GH010218-220.MP4 were invisible
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-06-03" / "GH010218.mp4")
    _touch(root / "2026-06-03" / "src" / "GH010218.MP4")
    assert [p.name for p in preproc.plan_pairs(root)] == ["2026_06_03-2026_06_03-GH010218"]


def test_plan_pairs_never_treats_a_src_video_as_an_edit(tmp_path):
    # A video inside src/ is a camera original even if src/src/ somehow existed
    root = tmp_path / "gdrive-src"
    _touch(root / "2026-07-15" / "Goprosplat" / "src" / "GH010228.MP4")
    assert preproc.plan_pairs(root) == []


def test_plan_pairs_raises_on_name_collision(tmp_path):
    # "gopro splat" and "gopro-splat" both sanitize to gopro_splat
    root = tmp_path / "gdrive-src"
    for parent in ("gopro splat", "gopro-splat"):
        _touch(root / "2026-07-15" / parent / "GH010228.mp4")
        _touch(root / "2026-07-15" / parent / "src" / "GH010228.MP4")
    with pytest.raises(ValueError, match="collision"):
        preproc.plan_pairs(root)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k plan_pairs`
Expected: 7 FAIL with `AttributeError: module 'preprocess_gdrive_videos' has no attribute 'plan_pairs'`.

- [ ] **Step 4: Implement `Pair`, `find_source` and `plan_pairs`**

Add to `scripts/preprocess_gdrive_videos.py`, in a new `########  Pairing` section after the naming section:

```python
########
# Pairing
########


class Pair(NamedTuple):
    """One Resolve export and the camera original it was cut from."""

    edit: Path
    source: Path
    name: str


def find_source(video):
    """Return the camera original for `video`, or None if it has no src/ counterpart.

    Matches on stem only: the real tree differs by case (GH010234.mp4 vs src/GH010234.MP4)
    and by extension (IMG_4085.mp4 vs src/IMG_4085.MOV), so neither can be compared directly.
    """
    src_dir = video.parent / "src"
    if not src_dir.is_dir():
        return None
    stem = video.stem.lower()
    for candidate in find_videos(src_dir):
        if candidate.stem.lower() == stem:
            return candidate
    return None


def plan_pairs(source_root):
    """Walk the capture tree and return every (edit, source) pair, sorted by flat name.

    A video is an edit iff <parent>/src/<stem>.* exists. Videos with no counterpart are
    camera originals that have not been edited yet; they are skipped and logged, and enter
    scope automatically once exported. Raises ValueError on a flat-name collision, which
    would otherwise silently overwrite one capture with another.
    """
    pairs = []
    skipped = 0
    for date_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if not _DATE_RE.fullmatch(date_dir.name):
            logger.info("skip %s: not a YYYY-MM-DD directory", date_dir.name)
            continue
        # Videos may sit directly in the date folder or one level down in a named parent.
        # The old plan_copies only looked one level down, so the former were invisible.
        folders = [date_dir] + sorted(p for p in date_dir.iterdir() if p.is_dir() and p.name != "src")
        for folder in folders:
            for video in find_videos(folder):
                source = find_source(video)
                if source is None:
                    logger.info("skip %s: no src/ counterpart (unedited original)", video.name)
                    skipped += 1
                    continue
                pairs.append(Pair(video, source, flat_dir_name(date_dir.name, folder.name, video)))

    seen = {}
    for pair in pairs:
        if pair.name in seen:
            raise ValueError(f"flat name collision {pair.name!r}: {seen[pair.name]} and {pair.edit}")
        seen[pair.name] = pair.edit

    logger.info("%d pairs, %d originals skipped", len(pairs), skipped)
    return sorted(pairs, key=lambda p: p.name)
```

Note on `test_plan_pairs_walks_videos_directly_in_a_date_folder`: when the video sits in the date folder itself, `folder.name` is the date, so the flat name is `2026_06_03-2026_06_03-GH010218`. That is redundant but unambiguous and collision-free, and it is what the assertion expects.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k plan_pairs`
Expected: 7 PASS.

- [ ] **Step 6: Lint and commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): pair each edit against its src/ original"
```

---

### Task 3: Copy with source-fingerprint idempotency

`copy_video` ports over unchanged, keeping all three existing tests green. The idempotency fix lives in a *separate* gate, `needs_copy`, which `main()` consults first. This matters: injection makes the curated mp4 larger than its source, so `copy_video`'s own size check would fail forever and re-copy over the injected file on every run.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `Pair`
- Produces:
  - `copy_video(video: Path, dest_dir: Path, force: bool = False) -> int`
  - `metadata_path(dest_dir: Path, video: Path) -> Path`
  - `read_metadata(dest_dir: Path, video: Path) -> dict | None`
  - `write_metadata(dest_dir: Path, video: Path, payload: dict) -> Path`
  - `source_fingerprint(path: Path) -> dict` returning `{"size_bytes": int, "mtime": float}`
  - `needs_copy(pair: Pair, dest_dir: Path, force: bool = False) -> bool`

- [ ] **Step 1: Write the failing tests**

Add to `tests/scripts/test_preprocess_gdrive_videos.py` (the three ported `copy_video` tests already exist and stay as-is):

```python
########
# Idempotency
########


def test_source_fingerprint_records_size_and_mtime(tmp_path):
    src = tmp_path / "GH010228.MP4"
    src.write_bytes(b"video")
    fp = preproc.source_fingerprint(src)
    assert fp["size_bytes"] == 5
    assert fp["mtime"] == pytest.approx(src.stat().st_mtime)


def test_metadata_round_trips(tmp_path):
    dest = tmp_path / "2026_07_15-Goprosplat-GH010228"
    dest.mkdir(parents=True)
    video = Path("GH010228.mp4")
    preproc.write_metadata(dest, video, {"unique_id": "x", "source": {"size_bytes": 5}})
    assert preproc.metadata_path(dest, video).name == "GH010228_metadata.json"
    assert preproc.read_metadata(dest, video)["source"]["size_bytes"] == 5


def test_read_metadata_returns_none_when_absent(tmp_path):
    assert preproc.read_metadata(tmp_path, Path("GH010228.mp4")) is None


def test_needs_copy_is_true_on_a_fresh_destination(tmp_path):
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    pair = preproc.Pair(edit, src, "scene")
    assert preproc.needs_copy(pair, tmp_path / "out") is True


def test_needs_copy_is_false_when_the_recorded_source_still_matches(tmp_path):
    # The trap: injection grows the curated mp4 past its source size, so a destination-size
    # check would re-copy forever. The gate keys off the fingerprint in the sidecar instead.
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "GH010228.mp4").write_bytes(b"video plus an injected gpmd track")
    pair = preproc.Pair(edit, src, "scene")
    preproc.write_metadata(dest, edit, {"source": preproc.source_fingerprint(edit)})

    assert preproc.needs_copy(pair, dest) is False
    assert preproc.needs_copy(pair, dest, force=True) is True


def test_needs_copy_is_true_when_the_source_changed(tmp_path):
    src = tmp_path / "src" / "GH010228.MP4"
    edit = tmp_path / "GH010228.mp4"
    _touch(src)
    edit.write_bytes(b"video")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "GH010228.mp4").write_bytes(b"video")
    pair = preproc.Pair(edit, src, "scene")
    preproc.write_metadata(dest, edit, {"source": preproc.source_fingerprint(edit)})

    edit.write_bytes(b"a re-exported, different edit")
    assert preproc.needs_copy(pair, dest) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "fingerprint or metadata or needs_copy or copy_video"`
Expected: the 6 new tests FAIL with `AttributeError`, and the 3 ported `copy_video` tests FAIL with `AttributeError: ... has no attribute 'copy_video'`.

- [ ] **Step 3: Implement the copy and idempotency section**

Add a `########  Copy and idempotency` section to `scripts/preprocess_gdrive_videos.py`:

```python
########
# Copy and idempotency
########


def copy_video(video, dest_dir, force=False):
    """Copy `video` into `dest_dir` under its original filename; return bytes copied.

    Writes to a `.partial` sidecar and renames, so an interrupted run cannot leave a
    truncated file that looks complete. Returns 0 when an up-to-date copy already exists.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / video.name
    size = video.stat().st_size

    # Existing copy of the right size is treated as done unless forced
    if dest.exists() and not force and dest.stat().st_size == size:
        logger.info("exists  %s", dest.relative_to(dest_dir.parent))
        return 0

    logger.info("copying %s (%.1f GB)", dest.relative_to(dest_dir.parent), size / 1e9)
    partial = dest.with_suffix(dest.suffix + ".partial")
    shutil.copy2(video, partial)
    os.replace(partial, dest)
    return size


def metadata_path(dest_dir, video):
    """Return the JSON sidecar path for `video` inside `dest_dir`."""
    return dest_dir / f"{video.stem}_metadata.json"


def read_metadata(dest_dir, video):
    """Return the JSON sidecar as a dict, or None when it does not exist."""
    path = metadata_path(dest_dir, video)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_metadata(dest_dir, video, payload):
    """Write the JSON sidecar for `video` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(dest_dir, video)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def source_fingerprint(path):
    """Return the size and mtime used to decide whether a curated copy is still current."""
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime": stat.st_mtime}


def needs_copy(pair, dest_dir, force=False):
    """Return True when the curated copy is missing or its recorded source has changed.

    The comparison is against the fingerprint stored in the JSON sidecar, never against the
    destination's own size. Injection adds a gpmd track and static tags, so the curated mp4
    ends up larger than the file it was copied from; a destination-size check would fail on
    every subsequent run, clobber the injected file, and re-upload ~7 GB each time.
    """
    if force:
        return True
    if not (dest_dir / pair.edit.name).is_file():
        return True
    recorded = (read_metadata(dest_dir, pair.edit) or {}).get("source")
    if not recorded:
        return True
    return recorded != source_fingerprint(pair.edit)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "fingerprint or metadata or needs_copy or copy_video"`
Expected: 9 PASS (6 new, 3 ported).

- [ ] **Step 5: Lint and commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): key copy idempotency off the recorded source fingerprint"
```

---

### Task 4: DMS formatting

A pure function, tested exhaustively, because it is the one piece of output a human reads directly.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `format_dms(value: float, axis: str) -> str` — `axis` is `"lat"` or `"lon"`
  - `format_gps(lat: float | None, lon: float | None) -> str` — returns `""` when either is None

- [ ] **Step 1: Write the failing tests**

```python
########
# DMS formatting
########


@pytest.mark.parametrize(
    "value,axis,expected",
    [
        (42.3532, "lat", "42 deg 21' 11.52\" N"),
        (-71.0659, "lon", "71 deg 3' 57.24\" W"),
        (-33.8688, "lat", "33 deg 52' 7.68\" S"),
        (151.2093, "lon", "151 deg 12' 33.48\" E"),
        (0.0, "lat", "0 deg 0' 0.00\" N"),
        (0.0, "lon", "0 deg 0' 0.00\" E"),
    ],
)
def test_format_dms(value, axis, expected):
    assert preproc.format_dms(value, axis) == expected


def test_format_dms_never_emits_a_leading_minus():
    # The hemisphere letter carries the sign; a minus as well would be double-signed
    assert "-" not in preproc.format_dms(-71.0659, "lon")


def test_format_gps_joins_both_axes():
    assert preproc.format_gps(42.3532, -71.0659) == "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W"


@pytest.mark.parametrize("lat,lon", [(None, -71.0659), (42.3532, None), (None, None)])
def test_format_gps_is_blank_without_a_fix(lat, lon):
    # Blank, never 0,0 and never a sentinel string
    assert preproc.format_gps(lat, lon) == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "dms or format_gps"`
Expected: 11 FAIL with `AttributeError: ... has no attribute 'format_dms'`.

- [ ] **Step 3: Implement**

Add a `########  GPS formatting` section:

```python
########
# GPS formatting
########


def format_dms(value, axis):
    """Format a signed decimal degree as degrees, minutes and 2-decimal seconds.

    The sign becomes the hemisphere letter, so the output never carries a leading minus:
    -71.0659 with axis="lon" gives 71 deg 3' 57.24" W. This matches the exiftool convention
    already in use when reading these files by hand.
    """
    hemisphere = ("N", "S") if axis == "lat" else ("E", "W")
    letter = hemisphere[0] if value >= 0 else hemisphere[1]
    magnitude = abs(value)
    degrees = int(magnitude)
    minutes_full = (magnitude - degrees) * 60
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60
    return f"{degrees} deg {minutes}' {seconds:.2f}\" {letter}"


def format_gps(lat, lon):
    """Format a fix as a single human-readable string, or "" when there is no fix."""
    if lat is None or lon is None:
        return ""
    return f"{format_dms(lat, 'lat')}, {format_dms(lon, 'lon')}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "dms or format_gps"`
Expected: 11 PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): format GPS fixes as human-readable DMS"
```

---

### Task 5: CSV index built from the sidecars

The index is derived state, rebuilt by scanning the curated tree. That is what makes a partial run safe: `--only GH010234` rewrites one clip's JSON, and every other clip's JSON is still on disk, so the regenerated CSV is still complete.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `format_gps`, `INDEX_NAME`
- Produces:
  - `index_rows(curated_root: Path) -> list[tuple[str, str]]`
  - `write_index(curated_root: Path) -> Path`
- Sidecar contract this task establishes, which Task 7 must satisfy: the JSON has a top-level `"unique_id"` (str) and a top-level `"gps"` object that is either `null` or `{"latitude": float, "longitude": float}`.

- [ ] **Step 1: Write the failing tests**

```python
########
# CSV index
########


def _curated(root, unique_id, stem, gps):
    """Write a minimal curated folder with just the JSON sidecar the index reads."""
    folder = root / unique_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_metadata.json").write_text(json.dumps({"unique_id": unique_id, "gps": gps}))
    return folder


def test_index_rows_are_sorted_by_unique_id(tmp_path):
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    _curated(tmp_path, "2026_06_29-splats-GH010221", "GH010221", {"latitude": 42.0, "longitude": -71.0})
    rows = preproc.index_rows(tmp_path)
    assert [r[0] for r in rows] == ["2026_06_29-splats-GH010221", "2026_07_22-splats-GH010234"]


def test_index_rows_format_gps_as_dms(tmp_path):
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    assert preproc.index_rows(tmp_path)[0][1] == "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W"


def test_index_rows_leave_gps_blank_without_a_fix(tmp_path):
    _curated(tmp_path, "2026_06_29-Phone-PXL_20260629_225753909.TS", "PXL_20260629_225753909.TS", None)
    assert preproc.index_rows(tmp_path)[0][1] == ""


def test_write_index_has_exactly_two_columns(tmp_path):
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    path = preproc.write_index(tmp_path)
    assert path.name == "index.csv"
    rows = list(csv.reader(io.StringIO(path.read_text())))
    assert rows[0] == ["unique_id", "gps"]
    assert len(rows[1]) == 2


def test_write_index_quotes_the_inch_mark_so_it_round_trips(tmp_path):
    # The DMS string contains both a comma and a double quote; csv must survive both
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    path = preproc.write_index(tmp_path)
    rows = list(csv.reader(io.StringIO(path.read_text())))
    assert rows[1][1] == "42 deg 21' 11.52\" N, 71 deg 3' 57.24\" W"


def test_write_index_survives_a_partial_run(tmp_path):
    # Rebuilding from sidecars means an untouched clip still appears in the index
    _curated(tmp_path, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    _curated(tmp_path, "2026_06_29-splats-GH010221", "GH010221", None)
    preproc.write_index(tmp_path)
    rows = list(csv.reader(io.StringIO((tmp_path / "index.csv").read_text())))
    assert len(rows) == 3  # header + 2 clips
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "index"`
Expected: 6 FAIL with `AttributeError: ... has no attribute 'index_rows'`.

- [ ] **Step 3: Implement**

Add a `########  CSV index` section:

```python
########
# CSV index
########


def index_rows(curated_root):
    """Scan the curated tree and return (unique_id, gps) rows sorted by unique_id.

    Derived entirely from the JSON sidecars, so the index can never drift from them and a
    partial run still produces a complete file: clips this run did not touch still have
    their sidecar on disk.
    """
    rows = []
    for sidecar in sorted(curated_root.glob("*/*_metadata.json")):
        payload = json.loads(sidecar.read_text())
        gps = payload.get("gps") or {}
        rows.append(
            (
                payload.get("unique_id", sidecar.parent.name),
                format_gps(gps.get("latitude"), gps.get("longitude")),
            )
        )
    return sorted(rows)


def write_index(curated_root):
    """Regenerate <curated-root>/index.csv from the sidecars and return its path."""
    rows = index_rows(curated_root)
    path = curated_root / INDEX_NAME
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unique_id", "gps"])
        writer.writerows(rows)
    logger.info("index: %d rows -> %s", len(rows), path)
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "index"`
Expected: 6 PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): rebuild the two-column CSV index from the sidecars"
```

---

### Task 6: Audio alignment by normalized cross-correlation

The offset solver is separated from audio decoding so it can be tested on synthesized arrays with no ffmpeg call and no real video.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `AUDIO_RATE`, `DEFAULT_ALIGN_MIN_R`
- Produces:
  - `class Alignment(NamedTuple): offset_s: float; r: float; ok: bool`
  - `solve_offset(edit: np.ndarray, source: np.ndarray, rate: int = AUDIO_RATE, min_r: float = DEFAULT_ALIGN_MIN_R) -> Alignment`
  - `decode_audio(path: Path, rate: int = AUDIO_RATE) -> np.ndarray`
  - `align(pair: Pair, rate: int = AUDIO_RATE, min_r: float = DEFAULT_ALIGN_MIN_R) -> Alignment`

- [ ] **Step 1: Write the failing tests**

```python
########
# Alignment
########


def _rng():
    """Deterministic generator so alignment tests never flake."""
    return np.random.default_rng(20260811)


def test_solve_offset_recovers_a_known_trim():
    rng = _rng()
    body = rng.standard_normal(preproc.AUDIO_RATE * 3).astype(np.float32)
    head = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    tail = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = np.concatenate([head, body, tail])

    result = preproc.solve_offset(body, source)
    assert result.ok is True
    assert result.r == pytest.approx(1.0, abs=1e-4)
    # Within one audio sample of the true 1.0 s offset
    assert abs(result.offset_s - 1.0) <= 1.0 / preproc.AUDIO_RATE


def test_solve_offset_is_unaffected_by_a_gain_change():
    # Normalized correlation is scale-free, so a level difference must not lower r
    rng = _rng()
    body = rng.standard_normal(preproc.AUDIO_RATE * 2).astype(np.float32)
    source = np.concatenate([rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32), body])

    result = preproc.solve_offset(body * 0.25, source)
    assert result.ok is True
    assert result.r == pytest.approx(1.0, abs=1e-4)


def test_solve_offset_rejects_uncorrelated_audio():
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result.ok is False
    assert result.r < preproc.DEFAULT_ALIGN_MIN_R


def test_solve_offset_rejects_an_edit_longer_than_its_source():
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)

    result = preproc.solve_offset(edit, source)
    assert result.ok is False
    assert result.offset_s == 0.0


def test_solve_offset_rejects_silence():
    # A constant signal has zero variance, so correlation is undefined rather than perfect
    source = np.zeros(preproc.AUDIO_RATE * 3, dtype=np.float32)
    result = preproc.solve_offset(np.zeros(preproc.AUDIO_RATE, dtype=np.float32), source)
    assert result.ok is False


def test_solve_offset_threshold_is_overridable():
    rng = _rng()
    edit = rng.standard_normal(preproc.AUDIO_RATE).astype(np.float32)
    source = rng.standard_normal(preproc.AUDIO_RATE * 5).astype(np.float32)

    assert preproc.solve_offset(edit, source, min_r=0.0).ok is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "solve_offset"`
Expected: 6 FAIL with `AttributeError: ... has no attribute 'solve_offset'`.

- [ ] **Step 3: Implement**

Add an `########  Alignment` section:

```python
########
# Alignment
########


class Alignment(NamedTuple):
    """Where the edit sits inside its source, and how well the audio matched."""

    offset_s: float
    r: float
    ok: bool


def solve_offset(edit, source, rate=AUDIO_RATE, min_r=DEFAULT_ALIGN_MIN_R):
    """Locate `edit` inside `source` by normalized cross-correlation.

    Resolve cuts and colour-corrects but never alters audio content, so the edit's audio is
    a literal subsegment of the source and a true match correlates near 1.0. A wrong lag
    over N samples lands near 1/sqrt(N) — about 0.0006 for a minute at 8 kHz — so the two
    cases are three orders of magnitude apart and the 0.95 gate is not delicate.

    The lag search is restricted to feasible positions, so an offset that would run the edit
    past the end of the source cannot be returned at all.
    """
    n, m = len(edit), len(source)
    if n == 0 or n > m:
        logger.warning("alignment impossible: edit has %d samples, source has %d", n, m)
        return Alignment(0.0, 0.0, False)

    # Remove the DC component from both so the correlation measures shape, not offset
    e = np.asarray(edit, dtype=np.float64) - np.mean(edit)
    s = np.asarray(source, dtype=np.float64) - np.mean(source)
    e_energy = float(np.sqrt(np.dot(e, e)))
    if e_energy == 0.0:
        logger.warning("alignment impossible: the edit's audio has no variance (silence)")
        return Alignment(0.0, 0.0, False)

    # Correlation at every feasible lag; "valid" gives exactly m - n + 1 positions
    numerator = signal.fftconvolve(s, e[::-1], mode="valid")
    # Energy of each length-n window of the source, via a cumulative sum
    cumulative = np.concatenate(([0.0], np.cumsum(s * s)))
    window_energy = np.sqrt(np.maximum(cumulative[n:] - cumulative[:-n], 0.0))
    denominator = window_energy * e_energy
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denominator > 0, numerator / denominator, 0.0)

    lag = int(np.argmax(r))
    peak = float(r[lag])
    return Alignment(lag / rate, peak, peak >= min_r)


def decode_audio(path, rate=AUDIO_RATE):
    """Decode `path` to a mono float32 array at `rate` Hz via ffmpeg."""
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def align(pair, rate=AUDIO_RATE, min_r=DEFAULT_ALIGN_MIN_R):
    """Solve where the edit sits inside its camera original, from audio alone."""
    result = solve_offset(decode_audio(pair.edit, rate), decode_audio(pair.source, rate), rate, min_r)
    if result.ok:
        logger.info("%s: offset %.3f s (r=%.4f)", pair.name, result.offset_s, result.r)
    else:
        logger.warning("%s: alignment rejected (r=%.4f < %.2f)", pair.name, result.r, min_r)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "solve_offset"`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): solve the trim offset by normalized audio cross-correlation"
```

---

### Task 7: Probe helpers and static metadata extraction

Reads the camera original with `exiftool` and `ffprobe`, and produces the sidecar payload the index contract from Task 5 depends on.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `Alignment`, `source_fingerprint`, `write_metadata`
- Produces:
  - `exif_dump(path: Path) -> list[dict]` — the `-json` array, one dict per document
  - `probe_duration(path: Path) -> float`
  - `find_gpmd_index(path: Path) -> int | None`
  - `static_tags(dump: list[dict]) -> dict`
  - `first_fix(dump: list[dict]) -> dict | None` returning `{"latitude": float, "longitude": float, "source_time": float}`
  - `build_payload(pair: Pair, dump: list[dict], alignment: Alignment, duration_s: float, unique_id: str, has_imu: bool, fingerprint: dict) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
########
# Metadata extraction
########

# One 1 Hz GPMF chunk per document, shaped the way exiftool -ee -json -G3 -n returns them
_DUMP = [
    {
        "Main:Make": "GoPro",
        "Main:Model": "GoPro Max",
        "Main:SerialNumber": "C123456789",
        "Main:FirmwareVersion": "H19.03.02.00",
        "Main:CreateDate": "2026:07:22 14:31:08",
        "Main:FieldOfView": "Wide",
    },
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.001,
        "Doc1:GPSLatitude": 0.0,
        "Doc1:GPSLongitude": 0.0,
        "Doc1:GPSDateTime": "2026:07:22 14:31:08",
    },
    {
        "Doc2:SampleTime": 1.001,
        "Doc2:SampleDuration": 1.001,
        "Doc2:GPSLatitude": 42.3532,
        "Doc2:GPSLongitude": -71.0659,
        "Doc2:GPSDateTime": "2026:07:22 14:31:09",
    },
]


def test_static_tags_strips_the_group_prefix():
    tags = preproc.static_tags(_DUMP)
    assert tags["Model"] == "GoPro Max"
    assert tags["SerialNumber"] == "C123456789"
    assert tags["CreateDate"] == "2026:07:22 14:31:08"


def test_first_fix_skips_the_unlocked_zero_zero_sample():
    # A GoPro emits 0,0 before satellite lock; treating that as a fix would put every early
    # clip in the Gulf of Guinea
    fix = preproc.first_fix(_DUMP)
    assert fix["latitude"] == pytest.approx(42.3532)
    assert fix["longitude"] == pytest.approx(-71.0659)
    assert fix["source_time"] == pytest.approx(1.001)


def test_first_fix_returns_none_when_nothing_locked():
    assert preproc.first_fix([{"Doc1:SampleTime": 0.0, "Doc1:GPSLatitude": 0.0, "Doc1:GPSLongitude": 0.0}]) is None


def test_first_fix_reads_a_container_location_when_there_is_no_track():
    # Pixel and iPhone carry a single point and no GPS track at all
    dump = [{"Main:GPSLatitude": 42.3532, "Main:GPSLongitude": -71.0659}]
    fix = preproc.first_fix(dump)
    assert fix["latitude"] == pytest.approx(42.3532)
    assert fix["source_time"] == 0.0


def test_first_fix_honours_the_trim_window():
    # A fix before the cut starts is not where this clip was shot
    fix = preproc.first_fix(_DUMP, start_s=1.5)
    assert fix is None


def test_build_payload_satisfies_the_index_contract():
    pair = preproc.Pair(Path("/x/GH010234.mp4"), Path("/x/src/GH010234.MP4"), "2026_07_22-splats-GH010234")
    payload = preproc.build_payload(
        pair,
        _DUMP,
        preproc.Alignment(4.2, 0.997, True),
        duration_s=131.4,
        unique_id="2026_07_22-splats-GH010234",
        has_imu=True,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["unique_id"] == "2026_07_22-splats-GH010234"
    assert payload["gps"]["latitude"] == pytest.approx(42.3532)
    assert payload["alignment"] == {"offset_s": 4.2, "r": 0.997, "ok": True}
    assert payload["source"] == {"size_bytes": 5, "mtime": 1.0}
    assert payload["has_imu"] is True
    # Cuts plus colour correction only, so lens geometry still describes the exported pixels
    assert payload["intrinsics"]["valid_for_edit"] is True


def test_build_payload_records_a_missing_fix_as_null():
    pair = preproc.Pair(Path("/x/PXL.mp4"), Path("/x/src/PXL.mp4"), "2026_06_29-Phone-PXL")
    payload = preproc.build_payload(
        pair,
        [{"Main:Model": "Pixel 9 Pro"}],
        preproc.Alignment(0.0, 0.99, True),
        duration_s=44.2,
        unique_id="2026_06_29-Phone-PXL",
        has_imu=False,
        fingerprint={"size_bytes": 5, "mtime": 1.0},
    )
    assert payload["gps"] is None
    assert payload["has_imu"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "static_tags or first_fix or build_payload"`
Expected: 7 FAIL with `AttributeError: ... has no attribute 'static_tags'`.

- [ ] **Step 3: Implement**

Add an `########  Metadata extraction` section:

```python
########
# Metadata extraction
########


def _ungrouped(doc):
    """Strip exiftool's `-G3` document prefix, so Doc12:SampleTime becomes SampleTime."""
    return {key.split(":", 1)[-1]: value for key, value in doc.items()}


def exif_dump(path):
    """Return exiftool's full JSON dump for `path`, timed metadata included.

    -ee walks the embedded documents, which is what surfaces the per-chunk GPMF payloads;
    -n gives raw numbers rather than formatted strings; -G3 tags each value with the
    document it came from, which is how samples stay grouped by 1 Hz chunk.
    """
    command = ["exiftool", "-ee", "-api", "LargeFileSupport=1", "-json", "-n", "-G3", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def probe_duration(path):
    """Return the container duration of `path` in seconds."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ]
    return float(subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip())


def find_gpmd_index(path):
    """Return the stream index of the gpmd data track, or None when there is not one.

    GPMF rides a `bin_data` stream tagged `gpmd`; Sony's rtmd and the phone tracks are
    different tags, so matching on the tag is what keeps injection GoPro-only for now.
    """
    command = [
        "ffprobe", "-v", "error", "-select_streams", "d",
        "-show_entries", "stream=index,codec_tag_string", "-of", "json", str(path),
    ]
    streams = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    for stream in streams.get("streams", []):
        if stream.get("codec_tag_string") == "gpmd":
            return int(stream["index"])
    return None


def static_tags(dump):
    """Return the container-level tags that are true of the edit regardless of the trim."""
    wanted = (
        "Make", "Model", "SerialNumber", "FirmwareVersion", "CreateDate", "MediaCreateDate",
        "FieldOfView", "LensProjection", "ProjectionType", "ElectronicImageStabilization",
        "GPSCoordinates", "GPSAltitude",
    )
    tags = {}
    for doc in dump:
        flat = _ungrouped(doc)
        for key in wanted:
            if key in flat and key not in tags:
                tags[key] = flat[key]
    return tags


def first_fix(dump, start_s=0.0):
    """Return the first locked GPS fix at or after `start_s`, or None when there is none.

    A GoPro emits 0,0 before satellite lock, so those samples are discarded rather than
    trusted — 4 of the 33 originals never lock at all. Clips with no GPS track (Pixel,
    iPhone) fall back to the single container fix, which has no time and is reported at 0.
    """
    best = None
    for doc in dump:
        flat = _ungrouped(doc)
        lat, lon = flat.get("GPSLatitude"), flat.get("GPSLongitude")
        if lat is None or lon is None:
            continue
        if float(lat) == 0.0 and float(lon) == 0.0:
            continue
        when = float(flat.get("SampleTime", 0.0))
        if when < start_s:
            continue
        if best is None or when < best["source_time"]:
            best = {"latitude": float(lat), "longitude": float(lon), "source_time": when}
    return best


def build_payload(pair, dump, alignment, duration_s, unique_id, has_imu, fingerprint):
    """Assemble the JSON sidecar for one pair.

    `gps` is the first locked fix inside the trimmed edit when alignment succeeded, and the
    first fix anywhere in the source when it did not — the edit window is unknown in that
    case, and `gps_source_anchored` records it so the value is never mistaken for exact.
    """
    anchored = not alignment.ok
    fix = first_fix(dump, start_s=0.0 if anchored else alignment.offset_s)
    return {
        "unique_id": unique_id,
        "edit": str(pair.edit),
        "source": fingerprint,
        "source_path": str(pair.source),
        "duration_s": duration_s,
        "has_imu": has_imu,
        "gps": fix,
        "gps_source_anchored": anchored and fix is not None,
        "alignment": {"offset_s": alignment.offset_s, "r": alignment.r, "ok": alignment.ok},
        "static_tags": static_tags(dump),
        # The edits are cuts plus colour correction with no geometric transform, so lens
        # geometry still describes the exported pixels. If a clip is ever reframed or
        # stabilized, this flag is what has to flip.
        "intrinsics": {"valid_for_edit": True, "reason": "cuts and colour correction only, no reframe"},
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "static_tags or first_fix or build_payload"`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): extract static tags and the first locked GPS fix"
```

---

### Task 8: GPMF sample expansion and the Parquet sidecar

GPMF arrives as one value-set per 1 Hz chunk with all ~200 samples concatenated. This task expands those back to per-sample rows on a real time axis and writes them to Parquet.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`, `pyproject.toml`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `_ungrouped`, `Alignment`
- Produces:
  - `expand_gpmf(dump: list[dict], key: str, components: int) -> tuple[list[float], list[tuple]]`
  - `telemetry_table(dump: list[dict], alignment: Alignment, duration_s: float) -> pa.Table | None`
  - `write_telemetry(table: pa.Table, dest_dir: Path, video: Path) -> Path`
  - Constant `_GPMF_STREAMS: dict[str, tuple[str, int]]` mapping column prefix to (exiftool key, component count)

- [ ] **Step 1: Declare `pyarrow` as a real dependency**

`pyarrow` is currently present in `uv.lock` only transitively via gradio, so it is not a dependency this project can rely on. Add it to `[project] dependencies` in `pyproject.toml`, immediately after the `"numcodecs",` line:

```toml
    "numcodecs",
    "pyarrow",          # columnar telemetry sidecars written by scripts/preprocess_gdrive_videos.py
    "scipy>=1.11",
```

- [ ] **Step 2: Write the failing tests**

```python
########
# Telemetry
########

# Two 1 Hz chunks, each holding 3 accelerometer triplets — the shape exiftool returns
_IMU_DUMP = [
    {
        "Doc1:SampleTime": 0.0,
        "Doc1:SampleDuration": 1.0,
        "Doc1:Accelerometer": "1 2 3 4 5 6 7 8 9",
        "Doc1:Gyroscope": "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
    },
    {
        "Doc2:SampleTime": 1.0,
        "Doc2:SampleDuration": 1.0,
        "Doc2:Accelerometer": "10 11 12 13 14 15 16 17 18",
        "Doc2:Gyroscope": "1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9",
    },
]


def test_expand_gpmf_spreads_samples_across_the_chunk_duration():
    times, values = preproc.expand_gpmf(_IMU_DUMP, "Accelerometer", 3)
    assert len(times) == 6
    assert values[0] == (1.0, 2.0, 3.0)
    assert values[3] == (10.0, 11.0, 12.0)
    # Three samples spread over a 1 s chunk land at 0, 1/3, 2/3
    assert times[1] == pytest.approx(1 / 3)
    assert times[3] == pytest.approx(1.0)


def test_expand_gpmf_accepts_a_list_payload():
    # exiftool returns a list rather than a space-joined string for some tags
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": [1, 2, 3]}]
    times, values = preproc.expand_gpmf(dump, "Accelerometer", 3)
    assert values == [(1.0, 2.0, 3.0)]
    assert times == [0.0]


def test_expand_gpmf_returns_empty_for_a_missing_key():
    assert preproc.expand_gpmf(_IMU_DUMP, "Gravity", 3) == ([], [])


def test_expand_gpmf_skips_a_ragged_chunk():
    # A truncated payload cannot be split into whole triplets; dropping it beats guessing
    dump = [{"Doc1:SampleTime": 0.0, "Doc1:SampleDuration": 1.0, "Doc1:Accelerometer": "1 2 3 4"}]
    assert preproc.expand_gpmf(dump, "Accelerometer", 3) == ([], [])


def test_telemetry_table_carries_both_time_axes():
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
    columns = table.column_names
    assert "source_time" in columns and "edit_time" in columns and "in_edit" in columns
    assert "accl_x" in columns and "gyro_z" in columns
    edit_time = table.column("edit_time").to_pylist()
    source_time = table.column("source_time").to_pylist()
    # edit_time is source_time shifted by the solved offset
    assert edit_time[3] == pytest.approx(source_time[3] - 0.5)


def test_telemetry_table_marks_samples_outside_the_cut():
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
    in_edit = table.column("in_edit").to_pylist()
    # The cut runs 0.5 s to 1.5 s in source time, so the first and last samples fall outside
    assert in_edit[0] is False
    assert in_edit[3] is True
    assert in_edit[-1] is False


def test_telemetry_table_nulls_edit_time_when_alignment_failed():
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.0, 0.2, False), duration_s=1.0)
    assert set(table.column("edit_time").to_pylist()) == {None}
    assert set(table.column("in_edit").to_pylist()) == {None}


def test_telemetry_table_is_none_without_imu():
    assert preproc.telemetry_table([{"Main:Model": "Pixel 9 Pro"}], preproc.Alignment(0.0, 0.99, True), 44.2) is None


def test_write_telemetry_names_the_file_after_the_video(tmp_path):
    table = preproc.telemetry_table(_IMU_DUMP, preproc.Alignment(0.5, 0.99, True), duration_s=1.0)
    path = preproc.write_telemetry(table, tmp_path, Path("GH010234.mp4"))
    assert path.name == "GH010234_telemetry.parquet"
    assert path.is_file()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "expand_gpmf or telemetry"`
Expected: 9 FAIL with `AttributeError: ... has no attribute 'expand_gpmf'`.

- [ ] **Step 4: Implement**

Add a `########  Telemetry` section:

```python
########
# Telemetry
########

# Column prefix -> (exiftool tag, component count). GPS9 is read as 5-wide because only the
# first five components (lat, lon, altitude, 2D speed, 3D speed) are stable across firmware.
#
# VERIFY AGAINST REAL FOOTAGE: the exiftool tag names below are the documented GPMF stream
# names, but exiftool renames some of them per firmware. Run
#   exiftool -ee -api LargeFileSupport=1 -json -n -G3 <original> | python -c \
#     "import json,sys; print(sorted({k.split(':',1)[-1] for d in json.load(sys.stdin) for k in d}))"
# on one GoPro original and correct any mismatch before relying on the output. A wrong name
# yields an empty column, not an exception, so this fails quietly if left unchecked.
_GPMF_STREAMS = {
    "accl": ("Accelerometer", 3),
    "gyro": ("Gyroscope", 3),
    "grav": ("GravityVector", 3),
    "cori": ("CameraOrientation", 4),
    "iori": ("ImageOrientation", 4),
    "gps": ("GPSTrack", 5),
}
_AXES = {3: ("x", "y", "z"), 4: ("w", "x", "y", "z"), 5: ("lat", "lon", "alt", "speed2d", "speed3d")}


def expand_gpmf(dump, key, components):
    """Expand 1 Hz GPMF chunks into per-sample (times, values) on the source time axis.

    exiftool returns one value-set per chunk with every sample concatenated — a single
    Accelerometer chunk is a flat run of triplets — plus SampleTime and SampleDuration.
    Samples are spread evenly across the chunk, which is the best placement available:
    GPMF carries no per-sample timestamp, only a per-chunk one.
    """
    times, values = [], []
    for doc in dump:
        flat = _ungrouped(doc)
        raw = flat.get(key)
        if raw is None:
            continue
        numbers = [float(x) for x in (raw if isinstance(raw, list) else str(raw).split())]
        if not numbers or len(numbers) % components:
            logger.warning("skipping ragged %s chunk: %d values for %d components", key, len(numbers), components)
            continue
        count = len(numbers) // components
        start = float(flat.get("SampleTime", 0.0))
        step = float(flat.get("SampleDuration", 1.0)) / count
        for i in range(count):
            times.append(start + i * step)
            values.append(tuple(numbers[i * components:(i + 1) * components]))
    return times, values


def telemetry_table(dump, alignment, duration_s):
    """Build the per-sample telemetry table, or None when the clip carries no IMU.

    Every stream is resampled onto the union of sample times so one table holds all of them;
    GPS at ~18 Hz shares the table with nulls elsewhere, which Parquet run-length encodes to
    nothing, so one file beats two. `edit_time` and `in_edit` are null when alignment failed:
    the cut window is unknown, so any value would be a guess.
    """
    streams = {}
    for prefix, (key, components) in _GPMF_STREAMS.items():
        times, values = expand_gpmf(dump, key, components)
        if times:
            streams[prefix] = (times, values, components)
    if not streams:
        return None

    axis = sorted({t for times, _, _ in streams.values() for t in times})
    columns = {"source_time": axis}
    for prefix, (times, values, components) in streams.items():
        lookup = dict(zip(times, values))
        for i, name in enumerate(_AXES[components]):
            columns[f"{prefix}_{name}"] = [lookup[t][i] if t in lookup else None for t in axis]

    if alignment.ok:
        columns["edit_time"] = [t - alignment.offset_s for t in axis]
        end = alignment.offset_s + duration_s
        columns["in_edit"] = [alignment.offset_s <= t <= end for t in axis]
    else:
        columns["edit_time"] = [None] * len(axis)
        columns["in_edit"] = [None] * len(axis)

    return pa.table(columns)


def write_telemetry(table, dest_dir, video):
    """Write the telemetry table beside the curated video and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{video.stem}_telemetry.parquet"
    pq.write_table(table, path, compression="zstd")
    logger.info("telemetry: %d rows -> %s", table.num_rows, path.name)
    return path
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "expand_gpmf or telemetry"`
Expected: 9 PASS.

If `pyarrow` is not installed in the active environment, install it first: `python -m pip install pyarrow`.

- [ ] **Step 6: Commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py pyproject.toml
git commit -m "feat(preproc): expand GPMF chunks into a Parquet telemetry sidecar"
```

---

### Task 9: Injection — gpmd remux and static tags

Order is load-bearing: ffmpeg rewrites the container, so it must run before exiftool writes tags into it. Reversed, ffmpeg drops the tags.

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: `Alignment`, `find_gpmd_index`
- Produces:
  - `gpmd_command(curated: Path, source: Path, offset_s: float, duration_s: float, gpmd_index: int, out: Path) -> list[str]`
  - `tag_command(curated: Path, tags: dict) -> list[str]`
  - `inject(curated: Path, source: Path, alignment: Alignment, duration_s: float, tags: dict) -> bool`

**No separate "already injected" guard.** The spec describes one, but it is unreachable given
the fingerprint gate from Task 3: `process_pair` either returns early without injecting, or
re-copies the pristine edit with `force=True` first. Injection therefore always operates on
freshly copied bytes and can never be applied twice to the same file. The `-Software`
provenance tag is still written, as a marker for humans inspecting a curated mp4.

- [ ] **Step 1: Write the failing tests**

These test the command *construction*, not ffmpeg itself — the argument order is the part that has been wrong before, and it is checkable without decoding anything.

```python
########
# Injection
########


def test_gpmd_command_trims_the_source_before_mapping_it():
    command = preproc.gpmd_command(
        Path("/out/GH010234.mp4"), Path("/src/GH010234.MP4"), 4.2, 131.4, 3, Path("/out/tmp.mp4")
    )
    # -ss and -t must precede the -i they apply to, or ffmpeg trims the wrong input
    assert command.index("-ss") < command.index("/src/GH010234.MP4")
    assert command[command.index("-ss") + 1] == "4.2"
    assert command[command.index("-t") + 1] == "131.4"


def test_gpmd_command_copies_every_curated_stream_and_only_the_gpmd_track():
    command = preproc.gpmd_command(
        Path("/out/GH010234.mp4"), Path("/src/GH010234.MP4"), 4.2, 131.4, 3, Path("/out/tmp.mp4")
    )
    assert "-map" in command and "0" in command
    assert "1:3" in command
    # -c copy keeps the edit's pixels untouched; -copy_unknown is what lets bin_data through
    assert "-c" in command and "copy" in command
    assert "-copy_unknown" in command


def test_tag_command_overwrites_in_place():
    command = preproc.tag_command(Path("/out/GH010234.mp4"), {"Model": "GoPro Max"})
    assert "-overwrite_original" in command
    assert "-Model=GoPro Max" in command
    # QuickTimeUTC stops exiftool reinterpreting the capture time in local time
    assert "-api" in command and "QuickTimeUTC" in command


def test_tag_command_skips_empty_values():
    command = preproc.tag_command(Path("/out/GH010234.mp4"), {"Model": "GoPro Max", "SerialNumber": None})
    assert not any(arg.startswith("-SerialNumber") for arg in command)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "gpmd_command or tag_command"`
Expected: 4 FAIL with `AttributeError: ... has no attribute 'gpmd_command'`.

- [ ] **Step 3: Implement**

Add an `########  Injection` section:

```python
########
# Injection
########

# Written into the curated file so a re-run can tell an injected mp4 from a fresh copy
PROVENANCE_TAG = "preprocess_gdrive_videos"


def gpmd_command(curated, source, offset_s, duration_s, gpmd_index, out):
    """Build the ffmpeg call that grafts the trimmed gpmd track onto the curated video.

    -ss and -t precede the second -i so they trim that input rather than the first. -c copy
    leaves the edit's pixels untouched, and -copy_unknown is what allows the bin_data gpmd
    stream through at all — without it ffmpeg silently drops unrecognized track types.
    """
    return [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(curated),
        "-ss", f"{offset_s}", "-t", f"{duration_s}", "-i", str(source),
        "-map", "0", "-map", f"1:{gpmd_index}",
        "-c", "copy", "-copy_unknown",
        str(out),
    ]


def tag_command(curated, tags):
    """Build the exiftool call that writes static tags into the finished container."""
    command = ["exiftool", "-overwrite_original", "-api", "QuickTimeUTC"]
    for key, value in tags.items():
        if value in (None, ""):
            continue
        command.append(f"-{key}={value}")
    command.append(f"-Software={PROVENANCE_TAG}")
    command.append(str(curated))
    return command


def inject(curated, source, alignment, duration_s, tags):
    """Write metadata back into the curated video; return True when a gpmd track landed.

    Order matters. ffmpeg runs first because it rewrites the container; exiftool runs second
    to write tags into the final one. Reversed, ffmpeg drops the tags. The remux goes to a
    temporary file that is only moved into place on success, so a failure leaves the curated
    video exactly as it was.
    """
    injected = False
    gpmd_index = find_gpmd_index(source) if alignment.ok else None
    if gpmd_index is not None:
        temp = curated.with_suffix(curated.suffix + ".inject")
        command = gpmd_command(curated, source, alignment.offset_s, duration_s, gpmd_index, temp)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            os.replace(temp, curated)
            injected = True
        else:
            temp.unlink(missing_ok=True)
            logger.error("gpmd injection failed for %s: %s", curated.name, result.stderr.strip().splitlines()[-1:])
    elif not alignment.ok:
        logger.warning("%s: no gpmd injected, alignment was rejected", curated.name)

    subprocess.run(tag_command(curated, tags), capture_output=True, text=True, check=True)
    return injected
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "gpmd_command or tag_command"`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): inject the retimed gpmd track and static tags"
```

---

### Task 10: CLI wiring, stage orchestration and the run summary

**Files:**
- Modify: `scripts/preprocess_gdrive_videos.py`
- Test: `tests/scripts/test_preprocess_gdrive_videos.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `require_binaries() -> None` — raises `SystemExit` naming any missing tool
  - `process_pair(pair: Pair, output_root: Path, min_r: float, force: bool) -> dict` — returns a one-line summary record
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

```python
########
# CLI
########


def test_require_binaries_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: None if name == "exiftool" else "/usr/bin/" + name)
    with pytest.raises(SystemExit, match="exiftool"):
        preproc.require_binaries()


def test_require_binaries_passes_when_everything_is_present(monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    assert preproc.require_binaries() is None


def test_main_dry_run_copies_nothing(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    out = tmp_path / "curated"
    assert preproc.main(["--source-root", str(tree), "--output-root", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_main_index_only_rebuilds_without_touching_video(tmp_path, monkeypatch):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    out = tmp_path / "curated"
    _curated(out, "2026_07_22-splats-GH010234", "GH010234", {"latitude": 42.3532, "longitude": -71.0659})
    assert preproc.main(["--output-root", str(out), "--index-only"]) == 0
    rows = list(csv.reader(io.StringIO((out / "index.csv").read_text())))
    assert rows[1][0] == "2026_07_22-splats-GH010234"


def test_main_only_filters_to_one_clip(tree, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(preproc.shutil, "which", lambda name: "/usr/bin/" + name)
    out = tmp_path / "curated"
    with caplog.at_level("INFO"):
        preproc.main(["--source-root", str(tree), "--output-root", str(out), "--only", "GH010228", "--dry-run"])
    assert "GH010228" in caplog.text
    assert "IMG_4085" not in caplog.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "require_binaries or main"`
Expected: 5 FAIL with `AttributeError: ... has no attribute 'require_binaries'`.

- [ ] **Step 3: Implement**

Add `########  Pipeline` and `########  Entry point` sections:

```python
########
# Pipeline
########


def require_binaries():
    """Exit before any work when a required external tool is missing."""
    missing = [name for name in ("ffmpeg", "ffprobe", "exiftool") if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required tools: {', '.join(missing)}")
    return None


def process_pair(pair, output_root, min_r, force):
    """Run copy, align, extract and inject for one pair; return a summary record."""
    dest_dir = output_root / pair.name
    curated = dest_dir / pair.edit.name

    if needs_copy(pair, dest_dir, force=force):
        copy_video(pair.edit, dest_dir, force=True)
    else:
        logger.info("%s: up to date", pair.name)
        if not force:
            return {"name": pair.name, "status": "skipped"}

    alignment = align(pair, min_r=min_r)
    dump = exif_dump(pair.source)
    duration_s = probe_duration(curated)

    table = telemetry_table(dump, alignment, duration_s)
    if table is not None:
        write_telemetry(table, dest_dir, pair.edit)

    payload = build_payload(
        pair, dump, alignment, duration_s, pair.name, table is not None, source_fingerprint(pair.edit)
    )
    injected = inject(curated, pair.source, alignment, duration_s, payload["static_tags"])
    payload["gpmd_injected"] = injected
    write_metadata(dest_dir, pair.edit, payload)

    return {
        "name": pair.name,
        "status": "processed",
        "aligned": alignment.ok,
        "r": alignment.r,
        "imu": table is not None,
        "injected": injected,
    }


########
# Entry point
########


def main(argv=None):
    """Parse args and run the curation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="nested capture tree to read")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="flat curated tree to write")
    parser.add_argument("--only", help="process only pairs whose flat name contains this substring")
    parser.add_argument("--dry-run", action="store_true", help="log the plan and total size, change nothing")
    parser.add_argument("--force", action="store_true", help="re-copy and re-inject even when up to date")
    parser.add_argument("--index-only", action="store_true", help="rebuild index.csv from the sidecars and stop")
    parser.add_argument("--push", action="store_true", help="run scripts/push_curated.sh after processing")
    parser.add_argument("--align-min-r", type=float, default=DEFAULT_ALIGN_MIN_R, help="minimum correlation to accept")
    args = parser.parse_args(argv)

    require_binaries()

    # --index-only never reads the source tree, so it works on a machine with no Drive mount
    if args.index_only:
        write_index(args.output_root)
        return 0

    if not args.source_root.is_dir():
        parser.error(f"source root does not exist: {args.source_root}")

    pairs = plan_pairs(args.source_root)
    if args.only:
        pairs = [p for p in pairs if args.only in p.name]
    if not pairs:
        logger.warning("no pairs found under %s", args.source_root)
        return 0

    if args.dry_run:
        total = sum(p.edit.stat().st_size for p in pairs)
        for pair in pairs:
            logger.info("%s/%s  <- %s", pair.name, pair.edit.name, pair.source)
        logger.info("dry run: %d pairs, %.1f GB", len(pairs), total / 1e9)
        return 0

    records = [process_pair(pair, args.output_root, args.align_min_r, args.force) for pair in pairs]
    write_index(args.output_root)

    processed = [r for r in records if r["status"] == "processed"]
    unaligned = [r["name"] for r in processed if not r["aligned"]]
    logger.info(
        "done: %d processed, %d skipped, %d with IMU, %d injected",
        len(processed),
        len(records) - len(processed),
        sum(1 for r in processed if r["imu"]),
        sum(1 for r in processed if r["injected"]),
    )
    # Surfaced loudly: these clips carry source-time telemetry and no injected track
    for name in unaligned:
        logger.warning("alignment rejected, telemetry left in source time: %s", name)

    if args.push:
        subprocess.run([str(PUSH_SCRIPT)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v -k "require_binaries or main"`
Expected: 5 PASS.

- [ ] **Step 5: Run the whole test file**

Run: `python -m pytest tests/scripts/test_preprocess_gdrive_videos.py -v`
Expected: all PASS, roughly 60 tests.

- [ ] **Step 6: Commit**

```bash
ruff check scripts/preprocess_gdrive_videos.py
git add scripts/preprocess_gdrive_videos.py tests/scripts/test_preprocess_gdrive_videos.py
git commit -m "feat(preproc): wire the CLI, stage ordering and run summary"
```

---

### Task 11: Retire `flatten_dataset.py` and update the README

**Files:**
- Delete: `scripts/flatten_dataset.py`
- Modify: `README.md` lines ~92-103

- [ ] **Step 1: Confirm nothing else references the old script**

Run: `grep -rn "flatten_dataset" --include="*.py" --include="*.md" --include="*.sh" --include="*.toml" .`
Expected: hits only in `scripts/flatten_dataset.py` itself and `README.md`. If anything else references it, stop and report — that reference needs repointing first.

- [ ] **Step 2: Delete the superseded script**

```bash
git rm scripts/flatten_dataset.py
```

- [ ] **Step 3: Rewrite the README curation block**

Replace the paragraph beginning `**Curated environment videos.**` and the shell block that follows it with:

```markdown
**Curated environment videos.** `scripts/preprocess_gdrive_videos.py` turns the nested Drive
export into one folder per video and carries the capture metadata across: DaVinci Resolve
strips every timed data track on export, so the script pairs each export against its camera
original in `src/`, solves the trim offset from audio, and writes the metadata back as
static tags plus a retimed `gpmd` track inside the mp4, with a full-rate Parquet sidecar
beside it. `scripts/push_curated.sh` uploads that tree to the `environments-curated` bucket.
Both are re-runnable and skip work already done.

Only videos that have a `src/` counterpart are processed; an unedited camera original is
skipped and enters scope automatically once it is exported.

```sh
python scripts/preprocess_gdrive_videos.py --dry-run   # ../gdrive-src -> ../environments-curated
python scripts/preprocess_gdrive_videos.py
python scripts/preprocess_gdrive_videos.py --only GH010234   # one clip
python scripts/preprocess_gdrive_videos.py --index-only      # rebuild index.csv alone

./scripts/push_curated.sh --dry-run            # -> collab-data:environments-curated
./scripts/push_curated.sh
```

Each curated folder holds the video, a `_metadata.json` sidecar, and a `_telemetry.parquet`
sidecar when the camera recorded IMU. `environments-curated/index.csv` is a two-column
`unique_id,gps` table regenerated from the sidecars on every run.

`exiftool` is required alongside `ffmpeg` for this pipeline.
```

- [ ] **Step 4: Add exiftool to the system requirements list**

In the `### 4. System requirements` section, change the optional-tools line to:

```markdown
- Optional: `colmap`, `ffmpeg`, `exiftool`, `rclone` for the COLMAP and data pipelines.
```

- [ ] **Step 5: Run the full test suite**

Run: `./scripts/test.sh`
Expected: no failures introduced by this branch. Pre-existing failures elsewhere in the suite are not this plan's concern — note them rather than fixing them.

- [ ] **Step 6: Commit**

```bash
git add -A scripts/ README.md
git commit -m "refactor(preproc): retire flatten_dataset.py in favour of preprocess_gdrive_videos"
```

---

## Manual verification on real footage

Run after Task 11. These are the checks that need actual video and cannot live in the test suite.

```bash
# 1. Plan only — expect 19 pairs, 29 originals skipped with reasons
python scripts/preprocess_gdrive_videos.py --dry-run

# 2. One clip end to end
python scripts/preprocess_gdrive_videos.py --only GH010234

# 3. gpmd survived injection. Before: tmcd alone. After: tmcd AND gpmd.
ffprobe -v error -show_entries stream=codec_tag_string -of csv=p=0 \
  ../environments-curated/2026_07_22-splats-GH010234/GH010234.mp4

# 4. Static tags landed — a 2026-07-22 capture date, not the Resolve export date
exiftool -G1 -GPSCoordinates -CreateDate -Model -SerialNumber \
  ../environments-curated/2026_07_22-splats-GH010234/GH010234.mp4

# 5. Telemetry sane: edit_time near 0 at the cut, rows ~ duration x 200 Hz, no NaNs in accl
python -c "
import pandas as pd
d = pd.read_parquet('../environments-curated/2026_07_22-splats-GH010234/GH010234_telemetry.parquet')
print(d.shape); print(d[['source_time','edit_time','in_edit']].describe())
print(d.filter(like='accl_').isna().sum())"

# 6. Idempotency — the trap. A second run must copy 0 bytes and re-inject nothing.
python scripts/preprocess_gdrive_videos.py --only GH010234

# 7. Index is two columns and pastes into Notion cleanly
head -3 ../environments-curated/index.csv

# 8. Full run, then push
python scripts/preprocess_gdrive_videos.py
./scripts/push_curated.sh --dry-run
python scripts/preprocess_gdrive_videos.py --push
```

**Expect on the first full run:** `2026_07_08-GoproSplat-GH010223` appears for the first time and the curated tree goes from 18 folders to 19. Every curated mp4 is rewritten, so the next `push_curated.sh` re-uploads roughly 7 GB once, then rclone is stable again.

**Watch for:** any clip logged as `alignment rejected`. The spec predicts a true match correlates near 1.0, so a rejection at 0.90-0.94 means the AAC re-encode degrades the match more than expected. Lower the bar with `--align-min-r` and record the observed values — do not silently widen the default.
