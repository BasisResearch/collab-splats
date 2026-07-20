# Frame Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a chunked `frames.zarr` the single canonical decode-once store of selected keyframes, remove the duplicate `images/` JPG dir and the `feedforward.zarr` `images` array, migrate every pixel consumer onto the store, and clean up `preproc`.

**Architecture:** The preprocess stage decodes the video once via `sample_frames` and writes `frames.zarr` (chunked-per-frame images + columnar records + provenance attrs). Consumers that need pixels read arrays from the store; consumers with path-locked APIs (VGGT-X/MapAnything model preprocessing) get a transient `FrameStore.export(dir)` deleted after use; the browser webapp encodes JPG bytes on demand. COLMAP/nerfstudio/splatter are unaffected (names/arrays or raw-video only). `feedforward.zarr` keeps a `frame_idx` reference instead of an `images` copy.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), zarr 3.x (`store.create_array`, `compressors=[BloscCodec(cname="lz4")]`), numpy, opencv (`cv2`), pytest. Spec: `docs/superpowers/specs/2026-07-20-keyframe-store-design.md`.

**Run tests with:** `/opt/venv/reconstruction/bin/python -m pytest`
**Format before every commit:** `black . && isort .`

---

## File Structure

- Create: `collab_splats/preproc/frame_store.py` — `FrameStore` (create/open/accessor/export/is_stale). One responsibility: persist + serve decoded keyframes.
- Create: `tests/preproc/test_frame_store.py` — store round-trip, partial read, provenance, export.
- Modify: `collab_splats/preproc/sampling.py` — cleanup (dead `stats`, weight params, W/H probe split, annotations); retire re-decode helpers; rename `extract_frame_fast`→`extract_frame`.
- Modify: `collab_splats/preproc/__init__.py` — export `FrameStore`; drop retired names.
- Modify: `collab_splats/wrapper/reconstructor.py` — `_extract_frames`/`preprocess` write `frames.zarr`; `_extract_2d_features`, localize DB build read store.
- Modify: `collab_splats/pointcloud/feedforward/{base,vggtx,mapanything}.py` — `_preprocess` reads store (export tmp dir); `save_zarr`/`load_zarr` drop `images`, add `frame_idx` ref.
- Modify: `collab_splats/localization/{localizer.py,viz.py}` — `cv2.imread(path)`→store arrays.
- Modify: `collab_splats/preproc/viz.py` — `load_frames`→store.
- Modify: `collab_splats/webapp/routers/{preprocess,localize,semantics,visualize}.py` — thumbnails/preview from store; `visualize.py` `images` shape read → store.
- Modify: `evals/datasets.py` — `_load_video` double-decode fix.

**Task ordering:** foundation (Tasks 1–3) → producer (Task 4) → primary consumer feedforward + dedup (Tasks 5–6) → remaining consumers (Tasks 7–9) → retire + docs (Tasks 10–11). Each task ends green and committed.

---

### Task 1: FrameStore module

**Files:**
- Create: `collab_splats/preproc/frame_store.py`
- Test: `tests/preproc/test_frame_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/preproc/test_frame_store.py
import numpy as np
import pytest
from collab_splats.preproc.frame_store import FrameStore


def _frames_and_records(n=3, h=8, w=12):
    # Distinct per-frame content so index mixups are caught
    frames = [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]
    records = [{"frame_idx": i * 10, "blur_score": float(i)} for i in range(n)]
    return frames, records


def test_create_open_roundtrip(tmp_path):
    frames, records = _frames_and_records()
    prov = {"video_path": "v.mp4", "video_mtime": 1.0, "method": "uniform", "max_frames": None}
    FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance=prov)
    store = FrameStore.open(tmp_path / "frames.zarr")
    assert len(store) == 3
    # image(i) returns the i-th selected frame; content matches
    np.testing.assert_array_equal(store.image(1), frames[1])
    assert store.record(1)["frame_idx"] == 10
    np.testing.assert_array_equal(store.frame_indices(), np.array([0, 10, 20]))


def test_image_by_frame_idx(tmp_path):
    frames, records = _frames_and_records()
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "s.zarr")
    # Look up by SOURCE video frame index, not position
    np.testing.assert_array_equal(store.image_by_frame_idx(20), frames[2])
    with pytest.raises(KeyError):
        store.image_by_frame_idx(999)


def test_images_subset(tmp_path):
    frames, records = _frames_and_records()
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "s.zarr")
    out = store.images([0, 2])
    assert out.shape == (2, 8, 12, 3)
    np.testing.assert_array_equal(out[1], frames[2])


def test_is_stale(tmp_path):
    frames, records = _frames_and_records()
    prov = {"video_path": "v.mp4", "video_mtime": 1.0, "method": "uniform", "max_frames": None}
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance=prov)
    store = FrameStore.open(tmp_path / "s.zarr")
    assert store.is_stale({"video_path": "v.mp4", "video_mtime": 2.0, "method": "uniform", "max_frames": None})
    assert not store.is_stale(prov)


def test_export_writes_jpgs(tmp_path):
    frames, records = _frames_and_records()
    FrameStore.create(tmp_path / "s.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "s.zarr")
    out = store.export(tmp_path / "exported")
    # One file per selected frame, named by source frame_idx, readable back
    assert len(out) == 3
    assert all(p.exists() for p in out)
    import cv2
    back = cv2.cvtColor(cv2.imread(str(out[2])), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(back, frames[2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frame_store.py -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.preproc.frame_store`.

- [ ] **Step 3: Write the FrameStore implementation**

```python
# collab_splats/preproc/frame_store.py
"""Canonical decode-once store of selected keyframes (chunked zarr).

The preprocess stage decodes a video exactly once and writes frames.zarr:
chunked-per-frame RGB images, columnar selection records, and provenance
attrs. All pixel consumers read from here instead of re-decoding the video.
Path-locked consumers (model preprocessing) use export() for a transient dir.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import zarr
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

# Provenance keys that decide whether a stored run may be reused as-is.
_STALENESS_KEYS = ("video_path", "video_mtime", "method", "max_frames")


class FrameStore:
    """Persist and serve selected keyframes from a chunked frames.zarr store."""

    def __init__(self, path: Path, store):
        self.path = Path(path)
        self._store = store
        # frame_idx -> row position, for source-index lookups
        self._idx_to_row = {int(fi): row for row, fi in enumerate(store["frame_idx"][:])}

    @classmethod
    def create(cls, path, frames, records, *, provenance) -> "FrameStore":
        """Write frames + records + provenance to a new frames.zarr and return it open."""
        path = Path(path)
        imgs = np.stack(frames).astype(np.uint8)  # (N, H, W, 3) RGB
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(path), mode="w")
        # Chunk one frame per chunk so a consumer reads a single keyframe alone
        store.create_array("images", data=imgs, chunks=(1, *imgs.shape[1:]), compressors=lz4)
        # Columnar records: every key present on any record becomes an array
        keys = sorted({k for r in records for k in r})
        for k in keys:
            col = np.array([r.get(k, np.nan) for r in records])
            store.create_array(k, data=col, chunks=col.shape, compressors=lz4)
        store.attrs["record_keys"] = keys
        store.attrs["provenance"] = {k: provenance.get(k) for k in provenance}
        store.attrs["schema_version"] = 1
        return cls(path, zarr.open(str(path), mode="r"))

    @classmethod
    def open(cls, path) -> "FrameStore":
        """Open an existing frames.zarr read-only."""
        return cls(path, zarr.open(str(path), mode="r"))

    def __len__(self) -> int:
        return int(self._store["images"].shape[0])

    def image(self, i: int) -> np.ndarray:
        """i-th selected frame (H, W, 3) uint8 RGB — single-chunk partial read."""
        return self._store["images"][i]

    def image_by_frame_idx(self, frame_idx: int) -> np.ndarray:
        """Frame by SOURCE video index; KeyError if that index was not selected."""
        if frame_idx not in self._idx_to_row:
            raise KeyError(f"frame_idx {frame_idx} not in store {self.path}")
        return self.image(self._idx_to_row[frame_idx])

    def images(self, idxs=None) -> np.ndarray:
        """Stack of selected frames (all, or the given row positions)."""
        if idxs is None:
            return self._store["images"][:]
        return np.stack([self._store["images"][i] for i in idxs])

    def record(self, i: int) -> dict:
        """Selection record dict for the i-th selected frame."""
        keys = list(self._store.attrs["record_keys"])
        return {k: self._store[k][i] for k in keys}

    def records(self) -> list[dict]:
        """All selection records, in selection order."""
        return [self.record(i) for i in range(len(self))]

    def frame_indices(self) -> np.ndarray:
        """Source video indices of the selected frames, in order."""
        return self._store["frame_idx"][:].astype(int)

    def is_stale(self, provenance: dict) -> bool:
        """True if stored provenance differs from `provenance` on any staleness key."""
        stored = dict(self._store.attrs.get("provenance", {}))
        return any(stored.get(k) != provenance.get(k) for k in _STALENESS_KEYS)

    def export(self, out_dir, *, ext="jpg") -> list[Path]:
        """Write frames to out_dir as frame_NNNNNN.<ext> (source-idx named); return paths.

        Transient bridge for path-locked consumers (model preprocessing); the
        caller deletes out_dir after use. Derived from the store, no re-decode.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for row, fi in enumerate(self.frame_indices()):
            p = out_dir / f"frame_{int(fi):06d}.{ext}"
            # Store holds RGB; cv2 writes BGR
            cv2.imwrite(str(p), cv2.cvtColor(self.image(row), cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frame_store.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py
isort collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py
git add collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py
git commit -m "feat(preproc): FrameStore — canonical decode-once frames.zarr + accessor"
```

---

### Task 2: preproc/sampling.py cleanup (dead stats, weight params, annotations)

**Files:**
- Modify: `collab_splats/preproc/sampling.py`
- Test: `tests/preproc/test_sampling.py` (existing — update any weight-param assertions)

- [ ] **Step 1: Update tests to the trimmed API**

In `tests/preproc/test_sampling.py`, remove any call passing `motion_weight=`/`coverage_weight=` to `sample_frames`, `score_frames`, or `OpticalFlowFrameSelector`, and delete any assertion touching `selector.stats`. Add a guard test:

```python
def test_selector_has_no_stats_attr():
    from collab_splats.preproc.sampling import OpticalFlowFrameSelector
    sel = OpticalFlowFrameSelector(min_disparity=50.0)
    assert not hasattr(sel, "stats")
```

- [ ] **Step 2: Run to verify the new guard fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py::test_selector_has_no_stats_attr -v`
Expected: FAIL — `stats` still present.

- [ ] **Step 3: Delete `stats` (sampling.py:254-259, 280-282)**

Remove the `self.stats = {...}` block in `OpticalFlowFrameSelector.__init__` and the three `self.stats[...].append(...)` lines in `score_frame`. Update the class docstring line that mentions `.stats`.

- [ ] **Step 4: Remove `motion_weight` / `coverage_weight` (always default)**

- `_combine_scores` (sampling.py:206-226): drop the two params; hardcode `motion_weight, coverage_weight = 0.6, 0.4` as local constants (keep the normalisation math intact).
- `OpticalFlowFrameSelector.__init__` (237-250): drop both params + the weight-validation branch (244-247); drop `self.motion_weight`/`self.coverage_weight`; the `score_frame` call to `_combine_scores` (283-290) drops the two kwargs.
- `sample_frames` (367-378), `_sample_optical_flow` (503-518, 514-517), `score_frames` (540-559): drop both params and stop threading them.
- Keep `min_disparity` everywhere — it is varied by callers.

- [ ] **Step 5: Bare type annotations (sampling.py:595, 608)**

Change `extract_frame(video_path: "str | Path", ...)` and `extract_frame_fast(video_path: "str | Path", ...)` to bare `video_path: str | Path` (the file already has `from __future__ import annotations`).

- [ ] **Step 6: Run the full sampling suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: PASS.

- [ ] **Step 7: Format and commit**

```bash
black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "refactor(preproc): drop dead selector.stats + always-default weight params"
```

---

### Task 3: Split cheap W/H probe from the full-demux ffprobe

**Files:**
- Modify: `collab_splats/preproc/sampling.py`
- Test: `tests/preproc/test_sampling.py`

**Why:** `_iter_frames` needs only width/height but calls `get_video_info`, which always runs `ffprobe -count_packets` (demuxes the whole file). Split a cheap dims-only probe so every decode stops paying a full demux.

- [ ] **Step 1: Write the failing test**

```python
def test_probe_dims_matches_full_info(tmp_path):
    # Reuses the sample video fixture already used in this file (see top of module).
    from collab_splats.preproc.sampling import get_video_info, _probe_dims
    video = SAMPLE_VIDEO  # existing fixture/const in test_sampling.py
    info = get_video_info(str(video))
    w, h = _probe_dims(str(video))
    assert (w, h) == (info["width"], info["height"])
```

If `test_sampling.py` has no shared sample-video constant, reuse the fixture the existing decode tests use (search the file for the video path they pass to `_iter_frames`/`sample_frames`) and mirror it here.

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py::test_probe_dims_matches_full_info -v`
Expected: FAIL — `_probe_dims` not defined.

- [ ] **Step 3: Add `_probe_dims` and use it in `_iter_frames`**

Add after `get_video_info` (sampling.py:114):

```python
def _probe_dims(video_path: str) -> tuple[int, int]:
    """Display (width, height) via a cheap ffprobe — no packet count / full demux."""
    _require_ffmpeg()
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-select_streams", "v:0", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe dims failed for %s", video_path, exc_info=True)
        return 0, 0
    for s in streams:
        if s.get("codec_type") != "video":
            continue
        w, h = int(s.get("width") or 0), int(s.get("height") or 0)
        # Match get_video_info: 90/270 rotation swaps display W/H
        if _rotation_degrees(s) in (90, 270):
            w, h = h, w
        return w, h
    return 0, 0
```

In `_iter_frames` (sampling.py:123), replace `info = get_video_info(str(video_path)); w, h = info["width"], info["height"]` with `w, h = _probe_dims(str(video_path))`.

- [ ] **Step 4: Run the sampling suite (decode paths still work)**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py tests/preproc/test_extract_frame.py tests/preproc/test_extract_frame_fast.py -v`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "perf(preproc): cheap W/H probe for _iter_frames (skip full-file demux)"
```

---

### Task 4: Producer — preprocess stage writes frames.zarr

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`_extract_frames` def:43, `preprocess` def:415, `images_dir`/new `frames_zarr` path property)
- Test: `tests/wrapper/test_reconstructor_preprocess.py` (create; mirror existing wrapper test style — flat functions)

**Design:** `preprocess` samples once and writes `frames.zarr` under `output_path`. It no longer writes `images/` JPGs as the canonical artifact. Downstream stages that still need file paths call `FrameStore.export(...)` (Task 5). Provenance = `{video_path, video_mtime, method, max_frames}`; reuse store when `not is_stale`.

- [ ] **Step 1: Write the failing test**

```python
# tests/wrapper/test_reconstructor_preprocess.py
from pathlib import Path
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.wrapper.reconstructor import Reconstructor  # adjust import to actual


def test_preprocess_writes_frames_zarr(tmp_path, sample_video):
    # sample_video: path to a short test video (reuse the preproc test fixture)
    cfg = {
        "input_path": str(sample_video),
        "output_path": str(tmp_path / "out"),
        "preprocessing": {"frame_selection": "uniform", "max_frames": 5},
    }
    r = Reconstructor(cfg)
    r.preprocess()
    store = FrameStore.open(Path(cfg["output_path"]) / "frames.zarr")
    assert 0 < len(store) <= 5
    # Canonical images/ JPG dir is NOT the source of truth anymore
    assert not (Path(cfg["output_path"]) / "images").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py -v`
Expected: FAIL — `frames.zarr` absent (stage still writes `images/`).

- [ ] **Step 3: Add a `frames_zarr` path property + rewrite `_extract_frames` to build a store**

Add near `images_dir` (reconstructor.py:401):

```python
@property
def frames_zarr(self) -> Path:
    """Canonical decode-once keyframe store for this run."""
    return self.output_path / "frames.zarr"
```

Rewrite `_extract_frames` (reconstructor.py:43) so the video branch (72/84) returns `(frames, records)` from `sample_frames` and writes a `FrameStore` instead of JPGs; the image-dir branch (62-68) builds records from the copied files and writes the store too (so every input yields a `frames.zarr`). Provenance from the input path stat:

```python
prov = {
    "video_path": str(input_path),
    "video_mtime": input_path.stat().st_mtime if input_path.is_file() else None,
    "method": method,
    "max_frames": max_frames,
}
FrameStore.create(self.frames_zarr, frames, records, provenance=prov)
```

Delete the `cv2.imwrite frame_%04d.jpg` loop (90-95). `preprocess` (415) reuses the store when it exists and `not FrameStore.open(self.frames_zarr).is_stale(prov)`.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py -v`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor_preprocess.py
isort collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor_preprocess.py
git add collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor_preprocess.py
git commit -m "feat(wrapper): preprocess stage writes canonical frames.zarr"
```

---

### Task 5: feedforward `_preprocess` reads the store (export tmp dir)

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`setup_inference` / `_preprocess` contract), `vggtx.py:240-252`, `mapanything.py:198-228`
- Modify: `collab_splats/wrapper/reconstructor.py` (`_run_feedforward`/`reconstruct` call at ~132 passes the store, not `images_dir`)
- Test: `tests/pointcloud/test_feedforward_preprocess_store.py` (create)

**Design:** VGGT-X `load_and_preprocess_images` and MapAnything `PIL.open` are path-locked. Lowest-risk: the feedforward creator receives the `FrameStore`, calls `store.export(tmp_dir)` into a `tempfile.TemporaryDirectory`, runs existing path-based preprocessing over the exported files, and the temp dir is deleted on exit. `image_paths` become the exported names (basenames stay `frame_NNNNNN`, preserving COLMAP naming).

- [ ] **Step 1: Write the failing test**

```python
# tests/pointcloud/test_feedforward_preprocess_store.py
import numpy as np
from collab_splats.preproc.frame_store import FrameStore


def test_preprocess_from_store_matches_dir(tmp_path, small_creator):
    # small_creator: a lightweight BaseFeedforwardCreator subclass fixture used in
    # existing feedforward tests; adjust to the real fixture name.
    frames = [np.full((32, 48, 3), i, dtype=np.uint8) for i in range(4)]
    records = [{"frame_idx": i, "blur_score": 1.0} for i in range(4)]
    FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance={"video_path": "v"})
    store = FrameStore.open(tmp_path / "frames.zarr")
    views, image_paths, coords = small_creator._preprocess_from_store(store)
    assert len(image_paths) == 4
    # Names preserve source frame_idx for COLMAP registration
    assert [p.name for p in image_paths] == [f"frame_{i:06d}.jpg" for i in range(4)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_preprocess_store.py -v`
Expected: FAIL — `_preprocess_from_store` not defined.

- [ ] **Step 3: Add store-based preprocessing**

In `base.py`, add a concrete helper on `BaseFeedforwardCreator` that exports the store to a temp dir and delegates to the existing abstract `_preprocess(image_dir)`:

```python
def _preprocess_from_store(self, store):
    """Export the FrameStore to a temp dir and run the existing path-based _preprocess."""
    import tempfile
    # Keep the temp dir alive for the whole run; inference reads the exported files
    self._frame_export = tempfile.TemporaryDirectory()
    export_dir = Path(self._frame_export.name)
    store.export(export_dir)  # frame_NNNNNN.jpg named by source idx
    return self._preprocess(export_dir)
```

Change `setup_inference` (base.py:787-790) to accept a `FrameStore` and call `_preprocess_from_store` when given one (keep `_preprocess(image_dir)` for the legacy dir path so `vggtx`/`mapanything` need no change). Update `reconstruct`/`_run_feedforward` (reconstructor.py:132) to pass `FrameStore.open(self.frames_zarr)`.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_preprocess_store.py -v`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_feedforward_preprocess_store.py
isort collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_feedforward_preprocess_store.py
git add collab_splats/pointcloud/feedforward/base.py collab_splats/wrapper/reconstructor.py tests/pointcloud/test_feedforward_preprocess_store.py
git commit -m "feat(pointcloud): feedforward _preprocess reads frames.zarr via temp export"
```

---

### Task 6: Drop `images` from feedforward.zarr; migrate its readers

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`save_zarr:171-180`, `load_zarr:234-235`)
- Modify: `collab_splats/webapp/routers/visualize.py:102,177`, `collab_splats/webapp/routers/localize.py:97`, `collab_splats/dashboard/viewer.py:53`
- Test: `tests/pointcloud/test_save_zarr_no_images.py` (create)

**Design:** `save_zarr` stops writing the `images` array and instead persists `frame_idx` (source indices, already in `image_paths` names). Readers needing pixels open `frames.zarr`; the one shape-only reader (`visualize.py:177`) reads `frames.zarr` `images.shape`.

- [ ] **Step 1: Write the failing test**

```python
# tests/pointcloud/test_save_zarr_no_images.py
import numpy as np
import zarr
from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _minimal_result():
    n = 2
    return FeedforwardResult(
        points=np.zeros((10, 3), np.float32), colors=np.zeros((10, 3), np.uint8),
        extrinsics=np.tile(np.eye(4), (n, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (n, 1, 1)).astype(np.float32),
        original_coords=np.zeros((n, 4), np.float32),
        image_paths=[__import__("pathlib").Path(f"frame_{i:06d}.jpg") for i in range(n)],
        model_width=48, model_height=32,
    )


def test_save_zarr_writes_no_images_array(tmp_path):
    _minimal_result().save_zarr(tmp_path / "ff.zarr")
    store = zarr.open(str(tmp_path / "ff.zarr"), mode="r")
    assert "images" not in store
    # frame_idx reference persisted instead
    assert "frame_idx" in store
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_save_zarr_no_images.py -v`
Expected: FAIL — `images` still written / `frame_idx` absent.

- [ ] **Step 3: Edit save_zarr / load_zarr**

- `save_zarr` (base.py:171-180): delete the `images` block. Add: write a `frame_idx` array from `[int(p.stem.split("_")[-1]) for p in self.image_paths]` (basenames are `frame_NNNNNN`).
- `load_zarr` (base.py:186,234-235): remove the `load_images` param and the `images = ...` line; leave `images=None` on the constructed result (field stays for back-compat, always None from zarr).
- Update every `load_zarr(..., load_images=True)` caller to drop the kwarg: `reconstructor.py:214,252,307`, `webapp/routers/visualize.py:102`, `webapp/routers/localize.py:97`. Any that then needed pixels handled in Step 4.

- [ ] **Step 4: Repoint the pixel/shape readers to frames.zarr**

- `visualize.py:177` `img_shape = store["images"].shape` → open `frames.zarr` and read `store["images"].shape` (H,W from `(N,H,W,3)`), or `FrameStore.open(frames_zarr)` and use `.image(0).shape`.
- `visualize.py:102`, `localize.py:97`, `dashboard/viewer.py:53`: if they used the loaded `images` tensor for display/lift, switch to reading needed frames from `frames.zarr` (`FrameStore.open(run_dir/'frames.zarr').images(...)`). If they only passed it through, no pixel read is needed.

- [ ] **Step 5: Run the affected suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_save_zarr_no_images.py tests/pointcloud -k "zarr or feature_lifting" -v`
Expected: PASS. (Update `tests/pointcloud/test_feature_lifting.py` if it asserts `load_images=True`.)

- [ ] **Step 6: Format and commit**

```bash
black collab_splats/pointcloud/feedforward/base.py collab_splats/webapp/routers/visualize.py collab_splats/webapp/routers/localize.py collab_splats/dashboard/viewer.py tests/pointcloud/test_save_zarr_no_images.py
isort <same files>
git add -A
git commit -m "refactor(pointcloud): drop images from feedforward.zarr; readers use frames.zarr"
```

---

### Task 7: Migrate localization + preproc.viz off file reads

**Files:**
- Modify: `collab_splats/localization/localizer.py:208,489` (`cv2.imread`), `collab_splats/localization/viz.py:96`
- Modify: `collab_splats/preproc/viz.py:135` (`load_frames`)
- Modify: `collab_splats/wrapper/reconstructor.py:170-184` (`_extract_2d_features`)
- Test: extend `tests/localization/` + `tests/preproc/test_viz.py`

- [ ] **Step 1: Write failing tests**

For `preproc/viz.py`, `plot_quality_examples` currently calls `load_frames(video_path, idxs)`. Add a test that it accepts a `FrameStore` and reads frames from it:

```python
def test_plot_quality_examples_uses_store(tmp_path, monkeypatch):
    import numpy as np
    from collab_splats.preproc.frame_store import FrameStore
    from collab_splats.preproc import viz
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in range(3)]
    recs = [{"frame_idx": i, "blur_score": 1.0, "score": 0.9, "selected": True} for i in range(3)]
    store = FrameStore.create(tmp_path / "s.zarr", frames, recs, provenance={"video_path": "v"})
    # Should not raise and should not call any video decoder
    monkeypatch.setattr(viz, "load_frames", None)  # ensure the old path is gone
    viz.plot_quality_examples(store, recs, n_examples=2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py::test_plot_quality_examples_uses_store -v`
Expected: FAIL.

- [ ] **Step 3: Migrate the readers**

- `preproc/viz.py:135`: change `plot_quality_examples(video_path, ...)` signature to `plot_quality_examples(store, ...)`; replace `frame_by_idx = dict(zip(all_idxs, load_frames(video_path, all_idxs)))` with `frame_by_idx = {i: store.image_by_frame_idx(i) for i in all_idxs}`.
- `localizer.py:208,489`: replace `cv2.imread(str(path))` with a frame pulled from the run's `frames.zarr` by source idx (`FrameStore.open(run_dir/'frames.zarr').image_by_frame_idx(idx)`), converting RGB→BGR only where the original code assumed BGR. Confirm colour order against the extractor.
- `localization/viz.py:96`: same store read.
- `reconstructor.py:170-184` `_extract_2d_features`: if `extractor.extract_and_cache` requires file paths, pass `FrameStore.open(self.frames_zarr).export(tmp)`; if it can take arrays, pass `store.images()`.

- [ ] **Step 4: Run localization + viz suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization tests/preproc/test_viz.py -v`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black <touched files> && isort <touched files>
git add -A
git commit -m "refactor(localization,preproc): read pixels from frames.zarr, not files"
```

---

### Task 8: Migrate webapp thumbnails/preview to the store

**Files:**
- Modify: `collab_splats/webapp/routers/{preprocess.py:35-52,89, localize.py:36-48,108, semantics.py:113}`
- Test: `tests/webapp/` (extend existing router tests; if none, add a thin one)

**Design:** The browser needs JPG *bytes*, not a persistent dir. Serve them by reading `frames.zarr` and `cv2.imencode(".jpg", ...)` on demand. Reconcile the webapp's separate `frames/` dir and reconstructor's old `images/` dir — both become the single `frames.zarr`.

- [ ] **Step 1: Write the failing test**

```python
def test_thumbnail_served_from_store(tmp_path, client):
    # client: existing webapp test client fixture; adjust to real setup.
    import numpy as np
    from collab_splats.preproc.frame_store import FrameStore
    frames = [np.full((16, 16, 3), 7, dtype=np.uint8)]
    FrameStore.create(tmp_path / "frames.zarr", frames, [{"frame_idx": 0}], provenance={"video_path": "v"})
    # Point the session at tmp_path; request the first thumbnail
    resp = client.get("/preprocess/thumbnail/0?session=...")  # adjust route
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/webapp -k thumbnail -v`
Expected: FAIL.

- [ ] **Step 3: Migrate the routers**

Replace `glob("*.jpg")` / `frames/` reads in `preprocess.py`, `localize.py`, `semantics.py` with `FrameStore.open(session_dir/'frames.zarr')`; serve a frame via `cv2.imencode(".jpg", cv2.cvtColor(store.image(i), cv2.COLOR_RGB2BGR))[1].tobytes()` behind the existing route. Remove the now-dead `_write_frames`/thumbnail-dir writers.

- [ ] **Step 4: Run webapp suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/webapp -v`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/webapp/routers/*.py && isort collab_splats/webapp/routers/*.py
git add -A
git commit -m "refactor(webapp): serve frames from frames.zarr (encode on demand)"
```

---

### Task 9: evals `_load_video` double-decode fix

**Files:**
- Modify: `evals/datasets.py:245-256`
- Test: `tests/evals/test_datasets.py` (extend or create)

- [ ] **Step 1: Write the failing test**

```python
def test_load_video_single_decode(tmp_path, sample_video, monkeypatch):
    import evals.datasets as ds
    calls = {"n": 0}
    real = ds.sample_frames
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(ds, "sample_frames", counting)
    monkeypatch.setattr(ds, "extract_frames", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-decode")))
    ds._load_video(sample_video, max_frames=3, fps=1.0)
    assert calls["n"] == 1  # decoded once, no extract_frames re-decode
```

- [ ] **Step 2: Run to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_datasets.py::test_load_video_single_decode -v`
Expected: FAIL — `extract_frames` re-decode fires.

- [ ] **Step 3: Rewrite `_load_video` to build a store then export**

Replace lines 251-253:

```python
frames, records = sample_frames(str(seq_dir), method="uniform", fps=fps)
frames, records = frames[:max_frames], records[:max_frames]
store = FrameStore.create(frames_dir / "frames.zarr", frames, records,
                          provenance={"video_path": str(seq_dir), "method": "uniform", "max_frames": max_frames})
images = store.export(frames_dir)  # write JPEGs once, from the in-memory frames
```

Add `from collab_splats.preproc.frame_store import FrameStore` to the imports at the top of `evals/datasets.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_datasets.py::test_load_video_single_decode -v`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black evals/datasets.py tests/evals/test_datasets.py && isort evals/datasets.py tests/evals/test_datasets.py
git add evals/datasets.py tests/evals/test_datasets.py
git commit -m "fix(evals): _load_video decodes once via FrameStore (no extract_frames re-decode)"
```

---

### Task 10: Retire re-decode helpers; rename extract_frame_fast; trim exports

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (delete `load_frames:590`, old `extract_frame:595`, `extract_frames:652`, `_iter_frames_at:143`; rename `extract_frame_fast:608`→`extract_frame`)
- Modify: `collab_splats/preproc/__init__.py`
- Delete: `tests/preproc/test_extract_frame.py` (tested the deleted exact decoder), retarget `tests/preproc/test_extract_frame_fast.py`→`test_extract_frame.py`

- [ ] **Step 1: Confirm no live callers remain**

Run: `/opt/venv/reconstruction/bin/python -m pytest -q` first to ensure Tasks 4–9 removed every caller. Then:
Run: `grep -rn "load_frames\|extract_frames\|_iter_frames_at\|extract_frame_fast" collab_splats evals`
Expected: only definitions in `sampling.py` and `__init__.py` remain (no external callers). If a caller remains, migrate it before deleting.

- [ ] **Step 2: Delete the retired functions**

Remove `load_frames` (590-592), the old exact `extract_frame` (595-605), `extract_frames` (652-661), and `_iter_frames_at` (143-158) from `sampling.py`.

- [ ] **Step 3: Rename the kept seek reader**

Rename `extract_frame_fast` (608) → `extract_frame`. Update its docstring to drop the "use extract_frame where exactness matters" clause (the exact one is gone). Its internal fallback `return extract_frame(video_path, frame_idx)` (621) — the unprobeable-video branch — now recurses; replace that branch with a `raise ValueError(f"cannot probe {video_path} for seek decode")` since the streaming exact decoder no longer exists.

- [ ] **Step 4: Update `__init__.py`**

```python
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.sampling import (
    check_frame_quality,
    compute_blur_score,
    extract_frame,
    get_video_info,
    sample_frames,
    score_frames,
)

__all__ = [
    "FrameStore",
    "sample_frames",
    "score_frames",
    "get_video_info",
    "extract_frame",
    "compute_blur_score",
    "check_frame_quality",
]
```

(`check_frame_quality`/`compute_blur_score` stay public — tutorial API per `project_tutorial_keyframe_rework`.)

- [ ] **Step 5: Retarget the tests**

Delete `tests/preproc/test_extract_frame.py` (covered the deleted exact decoder). Rename `tests/preproc/test_extract_frame_fast.py` → `tests/preproc/test_extract_frame.py`; update its imports/calls from `extract_frame_fast` to `extract_frame`.

- [ ] **Step 6: Run the full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest -q`
Expected: PASS (no import errors from the removed names).

- [ ] **Step 7: Format and commit**

```bash
black collab_splats/preproc tests/preproc && isort collab_splats/preproc tests/preproc
git add -A
git commit -m "refactor(preproc): retire re-decode helpers; extract_frame_fast -> extract_frame"
```

---

### Task 11: Example config + docs

**Files:**
- Modify: `docs/examples/run_scenes.py` config (frame_selection/max_frames stays; note frames.zarr output)
- Modify: `docs/` module doc mirroring `preproc/` (add FrameStore + frames.zarr artifact); `docs/known-test-failures.md` if any migrated test is quarantined.

- [ ] **Step 1: Update the example + preproc doc**

Document that preprocess now emits `output_path/frames.zarr` as the canonical keyframe artifact (no `images/` dir), and that consumers read pixels from it. Add a one-paragraph `FrameStore` usage snippet (`open` / `image_by_frame_idx` / `export`).

- [ ] **Step 2: Full suite + smoke**

Run: `/opt/venv/reconstruction/bin/python -m pytest -q`
Expected: PASS.
If the dashboard was touched transitively, run its gate: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke` → `SMOKE PASS`.

- [ ] **Step 3: Commit**

```bash
git add -f docs/
git commit -m "docs(preproc): document frames.zarr as canonical keyframe artifact"
```

---

## Verification (end-to-end)

1. **Decode-once:** run `docs/examples/run_scenes.py` on a short clip; confirm `output_path/frames.zarr` exists, `output_path/images/` does NOT, and the video is decoded a single time (instrument `_iter_frames`/`_probe_dims` call counts, or check timing).
2. **No dup:** confirm `feedforward.zarr` has no `images` array (`zarr.open(...).array_keys()`), only a `frame_idx` reference.
3. **Parity:** the reconstruction output (points/poses/COLMAP) on the short clip matches the pre-migration baseline within tolerance.
4. **Consumers:** localization run, webapp thumbnail request, and `plot_quality_examples` all render from the store with no video re-decode.
5. **Full suite green:** `/opt/venv/reconstruction/bin/python -m pytest -q`; dashboard `--smoke` prints `SMOKE PASS` if touched.
