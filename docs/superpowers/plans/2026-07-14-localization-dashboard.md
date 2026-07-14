# Localization Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a localization page to the splats dashboard: pick an rgb_X field-camera video frame, localize it against an existing reconstruction's feature DB, visualize match lines + camera pose on mesh + inlier distribution, and grow the DB with provenance.

**Architecture:** Tabbed shell (`pn.Tabs(dynamic=True)`) hosts the existing splats page and a new `LocalizePage` in one Panel session. A new `run_localization()` in `dashboard/pipeline.py` chains existing package parts (`FeedforwardResult.load_zarr` → `CameraLocalizer.from_feedforward` → `localize` → `add_localized_frame`) with `OperationLog` progress. Provenance lives in zarr group attrs; no new config files in gcloud.

**Tech Stack:** Panel + PyVista (existing dashboard stack), zarr v3, pycolmap (already wired), matplotlib, rclone via `SessionSource`, ffmpeg for single-frame decode.

**Spec:** `docs/superpowers/specs/2026-07-14-localization-dashboard-design.md`

**Conventions (from CLAUDE.md):**
- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Run tests as `/opt/venv/reconstruction/bin/python -m pytest ...`
- Imports at top of file; exception: optional heavy deps inside the function with a clear ImportError path.
- Flat test functions; `tests/` mirrors `collab_splats/`.
- `logging`, not `print`. Block-level comments. `git add -f` is NOT needed for code, only for `docs/superpowers/`.
- Commit scopes used below: `feat(localization)`, `feat(dashboard)`, `feat(preproc)`, `refactor(dashboard)`, `test(...)`.

**Pre-existing facts the implementer must NOT rediscover:**
- `LocalizationResult` fields: `pose (4,4)|None`, `n_correspondences`, `n_inliers`, `pts2d (M,2)`, `pts3d_matched (M,3)`, `inlier_mask (M,)`, `pts2d_ref (M,2)`, `ref_frame_indices (M,) int32`, `query_features: LocalFeatures` (`collab_splats/localization/localizer.py:22`).
- `localize()` already refines focal length by default: `refinement_options.refine_focal_length` defaults `True` via the config dict (`localizer.py:823-826`). **No localizer change needed for focal refinement.** Pose refinement is single-pose only; the DB is never modified.
- `CameraLocalizer.from_feedforward(result, extractor=, extractor_name=, zarr_path=, progress_callback=)` loads the zarr feature cache if present, else builds on GPU and saves (`localizer.py:643`).
- `add_localized_frame(image_path, pose, intrinsics, features, zarr_path=, extractor_name=)` appends to `local_features/<extractor>/localized/` (`localizer.py:497`).
- Extractor registry keys: `"disk"`, `"xfeat"`, `"loma"`, `"loma-g"`; instantiate via `BaseLocalExtractor.get(name)()` (RegistryMixin, `utils/torch_utils.py:119`).
- `GpuWorker.submit(job, on_done, doc)` — job runs on the worker thread, `on_done(result_or_exception)` runs on the IOLoop (usage pattern: `app.py:377-417`).
- `OperationLog`: `start_op(name)`, `update_progress(pct, msg, log=True)`, `append_line`, `finish_op`, `error_op`, `attach_logging("collab_splats")`, `render_html()` polled every 300 ms (`app.py:541-551`).
- `SessionSource` uses `RcloneClient` from `collab_data.data_dashboard.rclone_client`; `client.list_directory(bucket, path)` returns dicts with `"Name"` and `"IsDir"`; `client._cmd(verb, *args)` builds an rclone argv; `client.remote_name` prefixes remotes (`sources.py`).
- Reconstruction outputs land at `<base_dir>/<session>/<stem>/` with `feedforward.zarr`, `frames/*.jpg` (zero-padded `%05d.jpg`), `mesh/mesh_tsdf.ply`, `run_config.yaml`. `pull_processed`/`push_outputs` move the whole tree incrementally (rclone `copy` is idempotent/incremental).
- `pn.pane.VTK(plotter.ren_win, ...)` is the render pattern (`viewer.py:84-87`); call `pane.synchronize()` after mutating the plotter scene.
- zarr v3 note (project memory): use `compressors=[BloscCodec(...)]`-style codecs; `store.create_array(...)` — see existing calls in `localizer.py:278-296` and copy them.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/preproc/sampling.py` | Modify | add `extract_frame(video, idx)` — ffmpeg single-frame decode |
| `collab_splats/localization/localizer.py` | Modify | provenance attrs in `save_index` / `add_localized_frame` |
| `collab_splats/localization/intrinsics.py` | Create | `estimate_intrinsics` (experimental) |
| `collab_splats/localization/viz.py` | Modify | `plot_correspondences` refactor; `plot_inlier_distribution` |
| `collab_splats/localization/__init__.py` | Modify | export new names |
| `collab_splats/dashboard/config.py` | Modify | `LocalizationConfig` dataclass |
| `collab_splats/dashboard/sources.py` | Modify | field-session listing + remote DB discovery |
| `collab_splats/dashboard/pipeline.py` | Modify | `run_localization()` orchestrator |
| `collab_splats/dashboard/localize.py` | Create | `LocalizePage`, `SceneCache`, camera-viz helpers |
| `collab_splats/dashboard/app.py` | Modify | split `SplatsApp.view()` into `sidebar()`/`main()`; keep `view()` |
| `collab_splats/dashboard/shell.py` | Create | tabbed shell + entry point wiring |
| `collab_splats/dashboard/__main__.py` | Modify | serve the shell instead of the single page |
| `tests/preproc/test_extract_frame.py` | Create | ffmpeg frame decode |
| `tests/localization/test_provenance.py` | Create | attrs round-trip |
| `tests/localization/test_viz_distribution.py` | Create | both plot functions |
| `tests/localization/test_intrinsics.py` | Create | estimate + rescale with fake creator |
| `tests/dashboard/test_sources_field.py` | Create | field listing + DB discovery with fake client |
| `tests/dashboard/test_localization_config.py` | Create | config defaults |
| `tests/dashboard/test_run_localization.py` | Create | orchestrator with fakes |
| `tests/dashboard/test_localize_page.py` | Create | pure page logic (preselect, subsample) |
| `tests/dashboard/test_shell.py` | Create | shell composition |

---

### Task 1: `extract_frame` — ffmpeg single-frame decode

The preproc package is ffmpeg-only by design (no cv2.VideoCapture). The localize page needs one exact frame by index.

**Files:**
- Modify: `collab_splats/preproc/sampling.py`
- Modify: `collab_splats/preproc/__init__.py` (export)
- Test: `tests/preproc/test_extract_frame.py`

- [ ] **Step 1: Write the failing test**

Create `tests/preproc/test_extract_frame.py`:

```python
"""Tests for single-frame ffmpeg decode."""
import shutil
import subprocess

import numpy as np
import pytest

from collab_splats.preproc import extract_frame

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """10-frame 64x48 synthetic video whose frame index is encoded in the red channel."""
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    # Each frame's red value = frame_index * 20 → decodable assertion signal
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", "color=black:size=64x48:rate=10:duration=1",
            "-vf", "geq=r='N*20':g=0:b=0",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


def test_extract_frame_shape_and_dtype(tiny_video):
    frame = extract_frame(tiny_video, 0)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8


def test_extract_frame_selects_correct_index(tiny_video):
    # Red channel encodes frame index * 20; codec noise allows a loose tolerance
    f0 = extract_frame(tiny_video, 0)
    f5 = extract_frame(tiny_video, 5)
    assert abs(int(f0[..., 0].mean()) - 0) < 15
    assert abs(int(f5[..., 0].mean()) - 100) < 15


def test_extract_frame_out_of_range_raises(tiny_video):
    with pytest.raises(ValueError, match="frame 999"):
        extract_frame(tiny_video, 999)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_extract_frame.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_frame'`

- [ ] **Step 3: Implement `extract_frame`**

In `collab_splats/preproc/sampling.py`, add near the other ffmpeg helpers (the file already imports `subprocess`, `json`, `numpy as np`, `logging` — verify and reuse; add any of these that are missing to the top-of-file imports):

```python
def extract_frame(video_path: "str | Path", frame_idx: int) -> np.ndarray:
    """Decode exactly one frame (0-based index) via ffmpeg; returns (H, W, 3) uint8 RGB.

    Raises ValueError if frame_idx is past the end of the video.
    """
    video_path = str(video_path)
    # Probe dimensions with ffprobe so the raw pipe can be reshaped
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", video_path,
        ],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])

    # select filter decodes only the requested frame; rawvideo/rgb24 pipe avoids temp files
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", video_path,
            "-vf", f"select=eq(n\\,{frame_idx})", "-vframes", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    if len(out) != h * w * 3:
        raise ValueError(f"extract_frame: frame {frame_idx} not found in {video_path}")
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3).copy()
```

Export it: in `collab_splats/preproc/__init__.py`, add `extract_frame` to the existing import-from-`sampling` line and to `__all__` (match the file's existing style).

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_extract_frame.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_extract_frame.py
git commit -m "feat(preproc): extract_frame — ffmpeg single-frame decode by index"
```

---

### Task 2: Zarr provenance attrs in `CameraLocalizer`

Write build provenance on the extractor group at `save_index` time and per-frame source provenance at `add_localized_frame` time. Strictly additive — old stores load unchanged.

**Files:**
- Modify: `collab_splats/localization/localizer.py`
- Test: `tests/localization/test_provenance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/localization/test_provenance.py`:

```python
"""Provenance attrs round-trip through the zarr feature cache."""
import numpy as np
import pytest
import torch
import zarr

from collab_splats.localization.extractors import LocalFeatures
from collab_splats.localization.localizer import CameraLocalizer


class _FakeExtractor:
    """Deterministic extractor: 8 fixed keypoints, 4-dim descriptors."""

    def extract(self, rgb):
        k = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        d = torch.ones(8, 4)
        return LocalFeatures(keypoints=k, descriptors=d, scores=None)

    def match(self, a, b, hw):
        return torch.zeros((0, 2), dtype=torch.int64)


def _make_localizer(tmp_path, n_frames=2):
    """Build a localizer from synthetic images on disk (no GPU)."""
    import cv2

    paths = []
    for i in range(n_frames):
        p = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 128, dtype=np.uint8))
        paths.append(p)
    pts3d = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    extr = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
    intr = np.tile(np.array([[60, 0, 32], [0, 60, 24], [0, 0, 1]], np.float32), (n_frames, 1, 1))
    return CameraLocalizer(pts3d, extr, intr, paths, extractor=_FakeExtractor()), pts3d, extr, intr


def test_save_index_writes_build_attrs(tmp_path):
    loc, *_ = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk", attrs={"backbone": "vggtx", "ba": True, "lc": False,
                                      "built_at": "2026-07-14T00:00:00"})
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert group.attrs["backbone"] == "vggtx"
    assert group.attrs["ba"] is True
    assert group.attrs["extractor"] == "disk"


def test_save_index_without_attrs_still_stamps_extractor(tmp_path):
    loc, *_ = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")
    group = zarr.open(str(zp), mode="r")["local_features/disk"]
    assert group.attrs["extractor"] == "disk"


def test_add_localized_frame_records_provenance(tmp_path):
    loc, pts3d, extr, intr = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")

    feats = _FakeExtractor().extract(None)
    pose = np.eye(4, dtype=np.float32)
    prov = {"video_ref": "2024_02_06-session_0001/rgb_1/cam.mp4",
            "session": "2024_02_06-session_0001", "camera": "rgb_1", "frame_idx": 42}
    loc.add_localized_frame(tmp_path / "query.jpg", pose, intr[0], feats,
                            zarr_path=zp, extractor_name="disk", provenance=prov)

    lg = zarr.open(str(zp), mode="r")["local_features/disk/localized"]
    assert lg.attrs["provenance"][0]["frame_idx"] == 42
    assert lg.attrs["provenance"][0]["camera"] == "rgb_1"


def test_provenance_list_grows_per_frame(tmp_path):
    loc, pts3d, extr, intr = _make_localizer(tmp_path)
    zp = tmp_path / "feedforward.zarr"
    loc.save_index(zp, "disk")
    feats = _FakeExtractor().extract(None)
    pose = np.eye(4, dtype=np.float32)
    for i in range(2):
        loc.add_localized_frame(tmp_path / f"q{i}.jpg", pose, intr[0], feats,
                                zarr_path=zp, extractor_name="disk",
                                provenance={"frame_idx": i})
    lg = zarr.open(str(zp), mode="r")["local_features/disk/localized"]
    assert len(lg.attrs["provenance"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_provenance.py -v`
Expected: FAIL — `TypeError: save_index() got an unexpected keyword argument 'attrs'`

- [ ] **Step 3: Implement provenance attrs**

In `collab_splats/localization/localizer.py`:

(a) Change `save_index` signature (`localizer.py:239`) to:

```python
    def save_index(self, zarr_path: "str | Path", extractor_name: str,
                   attrs: "dict | None" = None) -> None:
```

and after `rec_group = store.require_group(rec_key)` (line ~255), add the group-attrs write on the **extractor-level** group (parent of `reconstruction/`):

```python
        # Build provenance on the extractor group: always stamp the extractor name;
        # merge any caller-supplied provenance (backbone, ba, lc, built_at, ...)
        ext_group = store.require_group(f"local_features/{extractor_name}")
        ext_group.attrs["extractor"] = extractor_name
        for k, v in (attrs or {}).items():
            ext_group.attrs[k] = v
```

(b) Change `add_localized_frame` signature (`localizer.py:497`) to add a final kwarg:

```python
        provenance: "dict | None" = None,
```

and pass it through to `_append_localized_to_zarr` (extend that call and its signature with `provenance: "dict | None" = None`). Inside `_append_localized_to_zarr`, in **both** the create-group branch and the append branch, maintain a `provenance` attrs list parallel to `image_paths`:

Create branch — after `loc_group.attrs["image_paths"] = [str(image_path)]`:

```python
            loc_group.attrs["provenance"] = [provenance or {}]
```

Append branch — after the `image_paths` attrs update:

```python
            prov_list = list(loc_group.attrs.get("provenance", []))
            prov_list.append(provenance or {})
            loc_group.attrs["provenance"] = prov_list
```

(c) Also pad `provenance` when an existing store predates this change: in the append branch, before appending, if `len(prov_list) < len(existing) - 1` (older frames without provenance), left-pad with `{}`:

```python
            while len(prov_list) < len(existing) - 1:
                prov_list.append({})
```

(`existing` is the already-updated `image_paths` list in that branch.)

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_provenance.py tests/localization/ -v`
Expected: new tests PASS; existing localization tests still PASS (signature changes are kwarg-only additive).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization/test_provenance.py
git commit -m "feat(localization): provenance attrs on feature-DB groups and localized frames"
```

---

### Task 3: viz — `plot_correspondences` refactor + `plot_inlier_distribution`

**Files:**
- Modify: `collab_splats/localization/viz.py`
- Modify: `collab_splats/localization/__init__.py` (export `plot_inlier_distribution`)
- Test: `tests/localization/test_viz_distribution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/localization/test_viz_distribution.py`:

```python
"""Figure smoke tests for localization visualisations."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from collab_splats.localization.localizer import LocalizationResult
from collab_splats.localization.viz import plot_correspondences, plot_inlier_distribution


def _fake_result(n_frames=4, per_frame=10):
    """Synthetic result: frame i contributes per_frame correspondences, i+1 inliers."""
    m = n_frames * per_frame
    ref_idx = np.repeat(np.arange(n_frames, dtype=np.int32), per_frame)
    inlier = np.zeros(m, dtype=bool)
    for i in range(n_frames):
        inlier[i * per_frame : i * per_frame + i + 1] = True
    rng = np.random.default_rng(0)
    return LocalizationResult(
        pose=np.eye(4, dtype=np.float32),
        n_correspondences=m,
        n_inliers=int(inlier.sum()),
        pts2d=rng.uniform(0, 64, (m, 2)).astype(np.float32),
        pts3d_matched=rng.normal(size=(m, 3)).astype(np.float32),
        inlier_mask=inlier,
        pts2d_ref=rng.uniform(0, 64, (m, 2)).astype(np.float32),
        ref_frame_indices=ref_idx,
    )


def test_distribution_returns_figure_uniform_totals():
    loc = _fake_result()
    fig = plot_inlier_distribution(loc, n_frames=4)
    assert fig is not None
    # Uniform totals → one dashed hline, no per-bar ticks
    ax = fig.axes[0]
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    plt.close(fig)


def test_distribution_per_bar_ticks_when_totals_vary():
    loc = _fake_result()
    # Drop 3 correspondences from frame 0 → totals no longer uniform
    keep = np.ones(len(loc.ref_frame_indices), dtype=bool)
    keep[:3] = False
    loc = LocalizationResult(
        pose=loc.pose, n_correspondences=int(keep.sum()), n_inliers=loc.n_inliers,
        pts2d=loc.pts2d[keep], pts3d_matched=loc.pts3d_matched[keep],
        inlier_mask=loc.inlier_mask[keep], pts2d_ref=loc.pts2d_ref[keep],
        ref_frame_indices=loc.ref_frame_indices[keep],
    )
    fig = plot_inlier_distribution(loc, n_frames=4)
    ax = fig.axes[0]
    # No global dashed line; per-bar ticks drawn as solid short hlines
    assert not any(line.get_linestyle() == "--" for line in ax.get_lines())
    plt.close(fig)


def test_distribution_marks_localized_frames():
    loc = _fake_result()
    fig = plot_inlier_distribution(loc, n_frames=4,
                                   frame_sources=["reconstruction"] * 3 + ["localized"])
    assert fig is not None
    plt.close(fig)


def test_correspondences_returns_figure_and_accepts_ref_idx(tmp_path):
    import cv2

    loc = _fake_result()
    query = np.full((48, 64, 3), 100, dtype=np.uint8)
    paths = []
    for i in range(4):
        p = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 60, dtype=np.uint8))
        paths.append(p)
    fig = plot_correspondences(loc, query, paths, ref_idx=2, show=False)
    assert fig is not None
    assert "frame 2" in fig.axes[0].get_title()
    plt.close(fig)


def test_correspondences_default_picks_best_frame(tmp_path):
    import cv2

    loc = _fake_result()  # frame 3 has most inliers (4)
    query = np.full((48, 64, 3), 100, dtype=np.uint8)
    paths = []
    for i in range(4):
        p = tmp_path / f"{i:05d}.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), 60, dtype=np.uint8))
        paths.append(p)
    fig = plot_correspondences(loc, query, paths, show=False)
    assert "frame 3" in fig.axes[0].get_title()
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_viz_distribution.py -v`
Expected: FAIL — `ImportError: cannot import name 'plot_inlier_distribution'`

- [ ] **Step 3: Implement**

In `collab_splats/localization/viz.py`:

(a) `plot_correspondences` — change the signature to:

```python
def plot_correspondences(
    loc: LocalizationResult,
    query_image: np.ndarray,
    image_paths: list,
    max_pairs: int = 200,
    warp_corners: bool = False,
    ref_idx: "int | None" = None,
    show: bool = True,
):
```

Update the docstring Args with:

```
        ref_idx:      Reference frame to plot; None selects the frame with most inliers.
        show:         Call plt.show() (notebook behaviour). Dashboard passes False.

    Returns:
        The matplotlib Figure, or None when there is nothing to plot.
```

Replace the best-frame block (`viz.py:43-50`) with:

```python
    # Reference frame: caller override, else the frame with most inlier correspondences
    if ref_idx is not None:
        best_ref_idx = int(ref_idx)
    else:
        inlier_frames = loc.ref_frame_indices[loc.inlier_mask]
        if len(inlier_frames) == 0:
            logger.warning("plot_correspondences: zero inliers — nothing to plot")
            return None
        best_ref_idx = int(np.bincount(inlier_frames.astype(np.intp)).argmax())
    frame_mask = loc.ref_frame_indices == best_ref_idx
```

Guard the empty-per-frame case right after (`kpts0` may be empty for an explicit `ref_idx`):

```python
    if not frame_mask.any():
        logger.warning("plot_correspondences: no correspondences for frame %d", best_ref_idx)
        return None
```

Replace the tail (`viz.py:128-129`, currently `plt.tight_layout(); plt.show()`) with:

```python
    plt.tight_layout()
    if show:
        plt.show()
    return fig
```

The early-return branches at the top of the function change from bare `return` to `return None` (no behavioural change, explicit for the new return contract).

(b) Add `plot_inlier_distribution` at the end of `viz.py`:

```python
def plot_inlier_distribution(
    loc: LocalizationResult,
    n_frames: "int | None" = None,
    frame_sources: "list[str] | None" = None,
) -> "plt.Figure | None":
    """Per-reference-image inlier bars with total-correspondence markers.

    Bars are coloured viridis by frame index (frame order == time) so this plot
    cross-reads with the 3D camera view. Each bar gets a black tick at that
    image's total correspondence count; when totals are uniform across images
    the ticks collapse to a single dashed horizontal line. Frames whose source
    is 'localized' get a red bar edge.

    Args:
        loc:           LocalizationResult from CameraLocalizer.localize().
        n_frames:      Total reference frames (bars include zero-match frames);
                       defaults to max(ref_frame_indices) + 1.
        frame_sources: Per-frame provenance list ('reconstruction' | 'localized').
    """
    if loc.ref_frame_indices is None or loc.inlier_mask is None:
        logger.warning("plot_inlier_distribution: no correspondence data to plot")
        return None

    n = int(n_frames) if n_frames is not None else int(loc.ref_frame_indices.max()) + 1
    idx = loc.ref_frame_indices.astype(np.intp)
    totals = np.bincount(idx, minlength=n)
    inliers = np.bincount(idx[loc.inlier_mask], minlength=n)

    # Viridis by frame index — matches the time colouring of the 3D camera plot
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, max(n, 2)))[:n]
    edge = ["red" if frame_sources is not None and i < len(frame_sources)
            and frame_sources[i] == "localized" else "none" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    x = np.arange(n)
    ax.bar(x, inliers, color=colors, edgecolor=edge, linewidth=1.5)

    # Totals: single dashed line when uniform, per-bar ticks otherwise
    nonzero = totals[totals > 0]
    if len(nonzero) and (nonzero == nonzero[0]).all():
        ax.axhline(int(nonzero[0]), linestyle="--", color="0.4", linewidth=1)
    else:
        for xi, t in zip(x, totals):
            if t > 0:
                ax.plot([xi - 0.4, xi + 0.4], [t, t], color="0.2", linewidth=1)

    ax.set_xlabel("reference image (time →)")
    ax.set_ylabel("inliers")
    ax.set_title(
        f"{loc.n_inliers}/{loc.n_correspondences} inliers "
        f"({100 * loc.n_inliers / max(loc.n_correspondences, 1):.0f}%)",
        fontsize=10,
    )
    plt.tight_layout()
    return fig
```

(c) In `collab_splats/localization/__init__.py`, add `plot_inlier_distribution` to the existing `viz` import/`__all__` (match file style).

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_viz_distribution.py tests/localization/ -v`
Expected: new tests PASS, existing suite green. Also run the docs notebook's usage pattern mentally: existing callers pass no new kwargs → `show=True` preserves behaviour.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/viz.py collab_splats/localization/__init__.py tests/localization/test_viz_distribution.py
git commit -m "feat(localization): inlier-distribution plot; plot_correspondences gains ref_idx/show/Figure return"
```

---

### Task 4: `estimate_intrinsics` (experimental)

**Files:**
- Create: `collab_splats/localization/intrinsics.py`
- Modify: `collab_splats/localization/__init__.py`
- Test: `tests/localization/test_intrinsics.py`

- [ ] **Step 1: Verify FeedforwardResult images layout (one command, no code change)**

Run:

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import inspect
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
print([f for f in FeedforwardResult.__dataclass_fields__])
print(inspect.getsource(FeedforwardResult.save_zarr)[:800])
EOF
```

Expected: fields include `images`, `intrinsics`; `save_zarr` shows `images` handled as `(N, H, W, 3)` (see `base.py:174`). If the in-memory layout is channel-first `(N, 3, H, W)` instead, adapt the two `shape` lines in Step 4 accordingly (the test in Step 2 pins the contract via a fake, so the unit test passes either way — this check is for the real-path smoke in Task 10).

- [ ] **Step 2: Write the failing test**

Create `tests/localization/test_intrinsics.py`:

```python
"""estimate_intrinsics: single-frame feedforward inference + rescale to query resolution."""
import numpy as np

from collab_splats.localization.intrinsics import estimate_intrinsics


class _FakeCreator:
    """Mimics the BaseFeedforwardCreator run surface at inference resolution 96x128."""

    def __init__(self):
        self.outputs = None
        self.seen_dir = None

    def load_model(self):
        pass

    def setup_inference(self, image_dir):
        self.seen_dir = image_dir

    def run_inference(self):
        pass

    def postprocess(self):
        class _R:
            intrinsics = np.array([[[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]]], np.float32)
            images = np.zeros((1, 96, 128, 3), np.uint8)

        self.outputs = _R()


def test_estimate_intrinsics_rescales_to_query_resolution():
    frame = np.zeros((192, 256, 3), dtype=np.uint8)  # 2x the fake inference res
    K = estimate_intrinsics(frame, creator=_FakeCreator())
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K[0, 0], 200.0)  # fx * (256/128)
    np.testing.assert_allclose(K[1, 1], 200.0)  # fy * (192/96)
    np.testing.assert_allclose(K[0, 2], 128.0)  # cx scaled
    np.testing.assert_allclose(K[2], [0, 0, 1])


def test_estimate_intrinsics_writes_frame_for_creator(tmp_path):
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    creator = _FakeCreator()
    estimate_intrinsics(frame, creator=creator)
    assert creator.seen_dir is not None  # creator consumed a staged image dir
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_intrinsics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.localization.intrinsics'`

- [ ] **Step 4: Implement**

Create `collab_splats/localization/intrinsics.py`:

```python
"""EXPERIMENTAL — query-camera intrinsics estimation via single-frame feedforward inference.

Feedforward backbones (VGGT-X, MapAnything) predict per-frame intrinsics; running one on a
single query frame yields an approximate pinhole K when the query camera is uncalibrated.
This path is under validation (see spec 2026-07-14): errors of a few percent in focal are
typical and are partially absorbed by pycolmap's focal refinement during PnP (enabled by
default in CameraLocalizer). Prefer a real calibration when one exists.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def estimate_intrinsics(frame: np.ndarray, creator=None) -> np.ndarray:
    """Estimate a (3, 3) pinhole K for one RGB frame, rescaled to the frame's resolution.

    Args:
        frame:   (H, W, 3) uint8 RGB query frame.
        creator: Feedforward creator instance (load_model/setup_inference/run_inference/
                 postprocess/outputs surface). Defaults to VGGTXCreator — imported lazily
                 because the feedforward stack is a heavy optional dependency.
    """
    if creator is None:
        # Heavy import kept inside the function: pulls the full reconstruction stack
        from collab_splats.pointcloud.feedforward import VGGTXCreator

        creator = VGGTXCreator()

    # Stage the frame as a one-image directory — the creator API consumes image dirs
    with tempfile.TemporaryDirectory() as td:
        Image.fromarray(frame).save(Path(td) / "00000.jpg")
        creator.load_model()
        creator.setup_inference(Path(td))
        creator.run_inference()
        creator.postprocess()

    result = creator.outputs
    K = np.asarray(result.intrinsics[0], dtype=np.float64).copy()

    # Rescale from the model's inference resolution to the query frame's resolution
    proc = np.asarray(result.images)
    h_proc, w_proc = int(proc.shape[1]), int(proc.shape[2])  # (N, H, W, 3)
    h_q, w_q = frame.shape[:2]
    K[0, :] *= w_q / w_proc
    K[1, :] *= h_q / h_proc

    logger.info(
        "estimate_intrinsics (EXPERIMENTAL): fx=%.1f fy=%.1f cx=%.1f cy=%.1f @ %dx%d",
        K[0, 0], K[1, 1], K[0, 2], K[1, 2], w_q, h_q,
    )
    return K.astype(np.float32)
```

In `collab_splats/localization/__init__.py`, export `estimate_intrinsics` (match file style).

- [ ] **Step 5: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_intrinsics.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/localization/intrinsics.py collab_splats/localization/__init__.py tests/localization/test_intrinsics.py
git commit -m "feat(localization): experimental single-frame intrinsics estimation"
```

---

### Task 5: `SessionSource` — field sessions + remote DB discovery

**Files:**
- Modify: `collab_splats/dashboard/sources.py`
- Test: `tests/dashboard/test_sources_field.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_sources_field.py`:

```python
"""Field-session browsing and remote localization-DB discovery."""
from collab_splats.dashboard.sources import SessionSource


class _FakeClient:
    """Returns canned rclone listings keyed by (bucket, path)."""

    remote_name = "collab-data"

    def __init__(self, listings):
        self._listings = listings

    def list_directory(self, bucket, path):
        key = (bucket, path)
        if key not in self._listings:
            raise RuntimeError(f"no such path: {key}")
        return self._listings[key]

    def _cmd(self, *args):
        return ["true"]


def _dirs(*names):
    return [{"Name": n, "IsDir": True} for n in names]


def _files(*names):
    return [{"Name": n, "IsDir": False} for n in names]


def test_list_field_sessions_filters_pattern():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_curated", ""): _dirs(
            "2024_02_06-session_0001", "2024_02_07-session_0012", "reconstruction", "misc"
        ),
    }))
    assert src.list_field_sessions() == ["2024_02_06-session_0001", "2024_02_07-session_0012"]


def test_list_rgb_cameras_only():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_curated", "2024_02_06-session_0001"): _dirs("rgb_1", "rgb_2", "thermal_1"),
    }))
    assert src.list_rgb_cameras("2024_02_06-session_0001") == ["rgb_1", "rgb_2"]


def test_list_camera_videos_filters_extensions():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_curated", "2024_02_06-session_0001/rgb_1"):
            _files("a.mp4", "b.MOV", "notes.txt"),
    }))
    assert src.list_camera_videos("2024_02_06-session_0001", "rgb_1") == ["a.mp4", "b.MOV"]


def test_list_localization_dbs_returns_extractor_names():
    src = SessionSource(client=_FakeClient({
        ("fieldwork_processed",
         "reconstruction/2024_02_06/vid/feedforward.zarr/local_features"):
            _dirs("loma-g", "disk"),
    }))
    assert src.list_localization_dbs("2024_02_06", "vid") == ["disk", "loma-g"]


def test_list_localization_dbs_missing_path_returns_empty():
    src = SessionSource(client=_FakeClient({}))
    assert src.list_localization_dbs("2024_02_06", "vid") == []


def test_pull_processed_passes_exclude_flags(tmp_path, monkeypatch):
    import collab_splats.dashboard.sources as sources

    captured = {}

    class _CmdClient(_FakeClient):
        def _cmd(self, *args):
            captured["args"] = args
            return ["true"]

    monkeypatch.setattr(sources.subprocess, "run", lambda cmd, check: None)
    src = SessionSource(client=_CmdClient({}))
    src.pull_processed("2024_02_06", "vid", tmp_path, excludes=("frames.zarr/**",))
    assert "--exclude" in captured["args"]
    assert "frames.zarr/**" in captured["args"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources_field.py -v`
Expected: FAIL — `AttributeError: 'SessionSource' object has no attribute 'list_field_sessions'`

- [ ] **Step 3: Implement**

In `collab_splats/dashboard/sources.py`, add `import re` to the top imports, a module constant next to the others:

```python
# YYYY_MM_DD-session_XXXX field-session folders at the fieldwork_curated root
_FIELD_SESSION_RE = re.compile(r"^\d{4}_\d{2}_\d{2}-session_\d{4}$")
```

and these methods on `SessionSource` (after `list_videos`):

```python
    def list_field_sessions(self) -> list[str]:
        """Return sorted YYYY_MM_DD-session_XXXX folders at the curated bucket root."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, "")
        return sorted(
            i["Name"] for i in items
            if i.get("IsDir") and _FIELD_SESSION_RE.match(i["Name"])
        )

    def list_rgb_cameras(self, field_session: str) -> list[str]:
        """Return sorted rgb_X camera folders in a field session (thermal_X deferred)."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, field_session)
        return sorted(i["Name"] for i in items if i.get("IsDir") and i["Name"].startswith("rgb_"))

    def list_camera_videos(self, field_session: str, camera: str) -> list[str]:
        """Return video filenames under a field session's camera folder."""
        client = self._require_client()
        items = client.list_directory(CURATED_BUCKET, f"{field_session}/{camera}")
        return [i["Name"] for i in items
                if not i.get("IsDir") and i["Name"].lower().endswith(_VIDEO_EXTS)]

    def fetch_field_video(self, field_session: str, camera: str, name: str, dest_dir: Path) -> Path:
        """rclone-copy a field-camera video to dest_dir; return the local path."""
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local = dest_dir / name
        remote = f"{client.remote_name}:{CURATED_BUCKET}/{field_session}/{camera}/{name}"
        subprocess.run(client._cmd("copyto", remote, str(local)), check=True)
        return local

    def list_localization_dbs(self, session: str, stem: str) -> list[str]:
        """Extractor names with a feature DB in the remote zarr (cheap directory listing)."""
        try:
            client = self._require_client()
            items = client.list_directory(
                PROCESSED_BUCKET, f"{ROOT}/{session}/{stem}/feedforward.zarr/local_features"
            )
        except Exception:  # path absent (no DB yet) or rclone unavailable
            return []
        return sorted(i["Name"] for i in items if i.get("IsDir"))
```

Also make `pull_processed` (`sources.py:77`) accept optional rclone excludes — additive, existing callers unchanged:

```python
    def pull_processed(self, session: str, stem: str, dest_dir: Path,
                       excludes: tuple = ()) -> Path:
        """rclone-copy processed outputs to dest_dir; return the local dir.

        excludes: rclone --exclude patterns (e.g. "frames.zarr/**") to skip
        artifacts a consumer does not need — keeps pulls minimal.
        """
        client = self._require_client()
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{client.remote_name}:{PROCESSED_BUCKET}/{ROOT}/{session}/{stem}"
        flags: list[str] = []
        for pattern in excludes:
            flags += ["--exclude", pattern]
        # no public remote->local API on RcloneClient; use _cmd directly
        subprocess.run(client._cmd("copy", *flags, remote, str(dest_dir)), check=True)
        return dest_dir
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_sources_field.py tests/dashboard/ -v`
Expected: new tests PASS, existing dashboard tests green.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/sources.py tests/dashboard/test_sources_field.py
git commit -m "feat(dashboard): field-session browsing + remote localization-DB discovery"
```

---

### Task 6: `LocalizationConfig`

**Files:**
- Modify: `collab_splats/dashboard/config.py`
- Test: `tests/dashboard/test_localization_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_localization_config.py`:

```python
"""LocalizationConfig defaults and construction."""
from collab_splats.dashboard.config import LocalizationConfig


def test_defaults():
    cfg = LocalizationConfig()
    assert cfg.extractor == "loma-g"
    assert cfg.append_to_db is True
    assert cfg.top_k_viz == 3
    assert cfg.calibration_path is None


def test_override():
    cfg = LocalizationConfig(extractor="disk", append_to_db=False)
    assert cfg.extractor == "disk"
    assert cfg.append_to_db is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localization_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalizationConfig'`

- [ ] **Step 3: Implement**

Append to `collab_splats/dashboard/config.py`:

```python
@dataclass
class LocalizationConfig:
    """Knobs for one localization run. UI/call state only — provenance for persisted
    localized frames lives in zarr attrs, so this is never serialised to gcloud."""

    extractor: str = "loma-g"          # feature-DB / matcher registry key
    top_k_viz: int = 3                 # match-pair figures shown, best-first
    append_to_db: bool = True          # persist successful poses to localized/
    calibration_path: "str | None" = None  # per-camera K yaml override; None → estimate
    max_pairs: int = 200               # line cap per match-pair figure
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localization_config.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/config.py tests/dashboard/test_localization_config.py
git commit -m "feat(dashboard): LocalizationConfig dataclass"
```

---

### Task 7: `run_localization` orchestrator

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py`
- Test: `tests/dashboard/test_run_localization.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_run_localization.py`:

```python
"""run_localization orchestration with all heavy pieces faked."""
from pathlib import Path

import numpy as np
import pytest

import collab_splats.dashboard.pipeline as pipeline
from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.localization.localizer import LocalizationResult


class _FakeSource:
    def __init__(self):
        self.pulled = False
        self.pushed = False
        self.excludes = None

    def pull_processed(self, session, stem, dest, excludes=()):
        self.pulled = True
        self.excludes = excludes
        (Path(dest) / "feedforward.zarr").mkdir(parents=True, exist_ok=True)

    def push_outputs(self, out_dir, session, stem, on_line=None):
        self.pushed = True


class _FakeLocalizer:
    def __init__(self, pose):
        self._pose = pose
        self.appended = None
        self._image_paths = [Path("/orig/00000.jpg"), Path("/orig/00001.jpg")]
        self._extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))

    @property
    def frame_sources(self):
        return ["reconstruction", "reconstruction"]

    def localize(self, image, K):
        m = 8
        return LocalizationResult(
            pose=self._pose, n_correspondences=m, n_inliers=6,
            pts2d=np.zeros((m, 2), np.float32), pts3d_matched=np.zeros((m, 3), np.float32),
            inlier_mask=np.ones(m, bool), pts2d_ref=np.zeros((m, 2), np.float32),
            ref_frame_indices=np.zeros(m, np.int32), query_features=object(),
        )

    def add_localized_frame(self, image_path, pose, intrinsics, features,
                            zarr_path=None, extractor_name=None, provenance=None):
        self.appended = provenance


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Patch every heavy dependency; return handles for assertions."""
    fake_localizer = _FakeLocalizer(pose=np.eye(4, dtype=np.float32))

    class _FakeResult:
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        intrinsics = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        image_paths = [Path("/orig/00000.jpg"), Path("/orig/00001.jpg")]

    monkeypatch.setattr(pipeline, "_load_feedforward_result", lambda out_dir: _FakeResult())
    monkeypatch.setattr(pipeline, "_build_localizer",
                        lambda result, cfg, zarr_path, op_log, cache=None, scene_key=None:
                        fake_localizer)
    monkeypatch.setattr(pipeline, "_stamp_db_provenance",
                        lambda zarr_path, extractor, out_dir: None)
    monkeypatch.setattr(pipeline, "extract_frame",
                        lambda video, idx: np.zeros((48, 64, 3), np.uint8))
    monkeypatch.setattr(pipeline, "_resolve_query_intrinsics",
                        lambda frame, cfg, op_log: np.eye(3, dtype=np.float32))
    # Push runs inline (no thread) so the flag is set before assertions
    monkeypatch.setattr(pipeline, "_push_async",
                        lambda source, out_dir, session, stem, op_log: source.push_outputs(
                            out_dir, session, stem))
    return fake_localizer


def _run(tmp_path, wired, append=True, source=None):
    source = source or _FakeSource()
    out = pipeline.run_localization(
        query_video=tmp_path / "cam.mp4",
        frame_idx=42,
        session="2024_02_06",
        stem="vid",
        config=LocalizationConfig(append_to_db=append),
        op_log=OperationLog(),
        source=source,
        base_dir=tmp_path,
        provenance={"camera": "rgb_1", "frame_idx": 42},
    )
    return out, source


def test_returns_result_and_scene_context(tmp_path, wired):
    out, source = _run(tmp_path, wired)
    assert out.result.pose is not None
    assert out.result.n_inliers == 6
    assert len(out.ref_image_paths) == 2
    assert out.ref_extrinsics.shape == (2, 4, 4)
    assert source.pulled  # zarr absent locally → pulled


def test_append_and_push_on_success(tmp_path, wired):
    out, source = _run(tmp_path, wired, append=True)
    assert wired.appended == {"camera": "rgb_1", "frame_idx": 42}
    assert source.pushed


def test_no_append_when_disabled(tmp_path, wired):
    out, source = _run(tmp_path, wired, append=False)
    assert wired.appended is None
    assert not source.pushed


def test_no_append_on_failed_pose(tmp_path, wired):
    wired._pose = None
    out, source = _run(tmp_path, wired, append=True)
    assert out.result.pose is None
    assert wired.appended is None
    assert not source.pushed


def test_ref_paths_remapped_to_local_frames_dir(tmp_path, wired):
    out, _ = _run(tmp_path, wired)
    assert out.ref_image_paths[0] == tmp_path / "2024_02_06" / "vid" / "frames" / "00000.jpg"


def test_pull_uses_minimal_excludes(tmp_path, wired):
    _, source = _run(tmp_path, wired)
    assert source.excludes == pipeline._PULL_EXCLUDES


def test_stamp_db_provenance_writes_attrs(tmp_path):
    """Real (unpatched) _stamp_db_provenance: run_config provenance lands on the group."""
    import yaml
    import zarr

    out_dir = tmp_path / "s" / "v"
    out_dir.mkdir(parents=True)
    (out_dir / "run_config.yaml").write_text(yaml.safe_dump({
        "env_model": "vggtx", "frame_indices": [0, 5, 10],
        "video_ref": "reconstruction/s/v/v.mp4",
    }))
    zp = out_dir / "feedforward.zarr"
    store = zarr.open(str(zp), mode="a")
    store.require_group("local_features/loma-g/reconstruction")

    pipeline._stamp_db_provenance(zp, "loma-g", out_dir)
    g = zarr.open(str(zp), mode="r")["local_features/loma-g"]
    assert g.attrs["backbone"] == "vggtx"
    assert g.attrs["frame_indices"] == [0, 5, 10]
    assert g.attrs["extractor"] == "loma-g"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_run_localization.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run_localization'`

- [ ] **Step 3: Implement**

In `collab_splats/dashboard/pipeline.py`:

(a) Top-of-file imports — add:

```python
from dataclasses import dataclass

from PIL import Image  # already imported — keep single import

from collab_splats.dashboard.config import LocalizationConfig, RunConfig
from collab_splats.preproc import extract_frame, sample_frames
```

(merge into the existing import lines; `RunConfig` and `sample_frames` are already imported — extend those lines rather than duplicating).

(b) Add a `Localization` section after the `run_pipeline` orchestrator:

```python
########
# Localization
########


@dataclass
class LocalizationRunOutput:
    """Everything the localize page needs to render one run."""

    result: "object"                 # LocalizationResult
    query_frame: np.ndarray          # (H, W, 3) uint8 RGB
    query_intrinsics: np.ndarray     # (3, 3) — estimated or calibrated
    intrinsics_source: str           # "estimated (experimental)" | "calibration file"
    ref_image_paths: list            # local paths, index-aligned with ref_frame_indices
    ref_extrinsics: np.ndarray       # (N, 4, 4) world-to-camera
    frame_sources: list              # per-frame 'reconstruction' | 'localized'


# Never pulled: frames.zarr duplicates the frames/ jpg dir the viz reads.
# NOTE(min-pull): dense arrays (depth/world_points/confidence) stay in the pull until
# FeedforwardResult.load_zarr is audited for tolerance to missing members — see Deferred.
_PULL_EXCLUDES = ("frames.zarr/**",)


def _load_feedforward_result(out_dir: Path):
    """Load the reconstruction result from the local zarr (lazy heavy import)."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    return FeedforwardResult.load_zarr(out_dir / "feedforward.zarr")


def _stamp_db_provenance(zarr_path: Path, extractor_name: str, out_dir: Path) -> None:
    """Write build provenance from run_config.yaml onto the extractor's zarr group.

    Idempotent — safe to call on every run; older stores gain attrs on first touch."""
    import zarr

    cfg_path = Path(out_dir) / "run_config.yaml"
    attrs: dict = {"extractor": extractor_name}
    if cfg_path.exists():
        run_cfg = RunConfig.from_yaml(cfg_path)
        attrs.update({
            "backbone": run_cfg.env_model,
            "frame_indices": list(run_cfg.frame_indices),
            "video_ref": run_cfg.video_ref,
        })
    store = zarr.open(str(zarr_path), mode="a")
    group = store.require_group(f"local_features/{extractor_name}")
    for k, v in attrs.items():
        group.attrs[k] = v


def _build_localizer(result, config: LocalizationConfig, zarr_path: Path,
                     op_log: OperationLog, cache=None, scene_key=None):
    """Load (or build, with progress) the feature DB; keep the localizer warm in the
    SceneCache so consecutive runs skip index reload and extractor model load."""
    from collab_splats.localization import CameraLocalizer
    from collab_splats.localization.extractors import BaseLocalExtractor

    if cache is not None and scene_key is not None:
        cached = cache.get(scene_key, f"localizer:{config.extractor}")
        if cached is not None:
            return cached

    extractor = BaseLocalExtractor.get(config.extractor)()

    def on_progress(done: int, total: int) -> None:
        # Only fires on a cache miss (DB build); scale into the 25→55% band
        op_log.update_progress(int(25 + 30 * (done + 1) / max(total, 1)),
                               f"localize: building DB {done + 1}/{total}", log=False)

    localizer = CameraLocalizer.from_feedforward(
        result,
        extractor=extractor,
        extractor_name=config.extractor,
        zarr_path=zarr_path,
        progress_callback=on_progress,
    )
    if cache is not None and scene_key is not None:
        cache.put(scene_key, f"localizer:{config.extractor}", localizer)
    return localizer


def _resolve_query_intrinsics(frame: np.ndarray, config: LocalizationConfig,
                              op_log: OperationLog) -> np.ndarray:
    """Calibration file when configured; else experimental feedforward estimate."""
    if config.calibration_path:
        import yaml

        data = yaml.safe_load(Path(config.calibration_path).read_text())
        return np.asarray(data["K"], dtype=np.float32).reshape(3, 3)

    from collab_splats.localization.intrinsics import estimate_intrinsics
    from collab_splats.utils.torch_utils import pytorch_gc

    op_log.append_line("localize: intrinsics are ESTIMATED (experimental) — validate before trusting poses")
    K = estimate_intrinsics(frame)
    pytorch_gc()  # free the feedforward model before matching
    return K


def _local_ref_paths(localizer, out_dir: Path) -> list:
    """Remap DB image paths (recorded on the machine that built the DB) to local files."""
    paths = []
    for p, src in zip(localizer._image_paths, localizer.frame_sources):
        sub = "frames" if src == "reconstruction" else "localized_frames"
        paths.append(Path(out_dir) / sub / Path(p).name)
    return paths


def run_localization(
    *,
    query_video: Path,
    frame_idx: int,
    session: str,
    stem: str,
    config: LocalizationConfig,
    op_log: OperationLog,
    source: SessionSource,
    base_dir: Path,
    provenance: "dict | None" = None,
    cache=None,
) -> LocalizationRunOutput:
    """Localize one query-video frame against an existing reconstruction; optionally
    append the result to the localized/ DB group and push incrementally."""
    out_dir = Path(base_dir) / session / stem
    op_log.start_op(f"localize {Path(query_video).name}#{frame_idx}")
    try:
        with op_log.attach_logging("collab_splats"):
            # Reconstruction data: pull once (minimal set), then load from local zarr
            op_log.update_progress(5, "localize: pulling reconstruction")
            if not (out_dir / "feedforward.zarr").exists():
                source.pull_processed(session, stem, out_dir, excludes=_PULL_EXCLUDES)
            op_log.update_progress(15, "localize: loading reconstruction")
            result = _load_feedforward_result(out_dir)

            # Feature DB: warm-cache hit skips reload; zarr hit is fast; miss builds on GPU
            op_log.update_progress(25, f"localize: loading DB ({config.extractor})")
            localizer = _build_localizer(result, config, out_dir / "feedforward.zarr",
                                         op_log, cache=cache, scene_key=(session, stem))
            _stamp_db_provenance(out_dir / "feedforward.zarr", config.extractor, out_dir)

            # Query frame + intrinsics
            op_log.update_progress(55, f"localize: extracting frame {frame_idx}")
            frame = extract_frame(query_video, frame_idx)
            op_log.update_progress(60, "localize: resolving query intrinsics")
            K = _resolve_query_intrinsics(frame, config, op_log)
            intr_source = "calibration file" if config.calibration_path else "estimated (experimental)"

            # Pose: single-pose PnP + refinement — the DB is never modified here
            op_log.update_progress(70, "localize: matching + solving pose")
            loc = localizer.localize(frame, K)
            op_log.append_line(
                f"localize: {loc.n_inliers}/{loc.n_correspondences} inliers"
                + ("" if loc.pose is not None else " — POSE FAILED")
            )

            # Persist: save the query frame locally, append to localized/, push new chunks
            if loc.pose is not None and config.append_to_db:
                op_log.update_progress(85, "localize: appending to DB")
                img_dir = out_dir / "localized_frames"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"{Path(query_video).stem}_f{frame_idx:06d}.jpg"
                Image.fromarray(frame).save(img_path)
                localizer.add_localized_frame(
                    img_path, loc.pose, K, loc.query_features,
                    zarr_path=out_dir / "feedforward.zarr",
                    extractor_name=config.extractor,
                    provenance=provenance,
                )
                op_log.update_progress(92, "localize: pushing to fieldwork_processed (background)")
                _push_async(source, out_dir, session, stem, op_log)

            output = LocalizationRunOutput(
                result=loc,
                query_frame=frame,
                query_intrinsics=K,
                intrinsics_source=intr_source,
                ref_image_paths=_local_ref_paths(localizer, out_dir),
                ref_extrinsics=np.asarray(localizer._extrinsics),
                frame_sources=localizer.frame_sources,
            )
        op_log.finish_op()
        return output
    except Exception as exc:
        logger.exception("localization failed")
        op_log.error_op(str(exc))
        raise
```

Note: `ref_extrinsics` uses the localizer's stored reconstruction extrinsics (`localizer.py:172`) — the same world frame `loc.pose` is solved in, so the 3D view plots both directly.

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_run_localization.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/pipeline.py tests/dashboard/test_run_localization.py
git commit -m "feat(dashboard): run_localization orchestrator with OperationLog progress"
```

---

### Task 8: Split `SplatsApp` into sidebar/main parts (mechanical)

Shell needs the page's sidebar and main content separately. Keep `view()` working (tests + standalone use).

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_shell.py` (first two tests)

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_shell.py` (shell tests land in Task 9's second half; start with the split):

```python
"""Shell composition and page-split contracts."""
import panel as pn

from collab_splats.dashboard.app import SplatsApp
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource


class _NoopSource(SessionSource):
    def __init__(self):
        self._client = None  # degrade: listings fail soft, nothing remote


def _app(tmp_path):
    return SplatsApp(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())


def test_splats_page_exposes_sidebar_and_main(tmp_path):
    app = _app(tmp_path)
    assert isinstance(app.sidebar(), pn.Column)
    main = app.main()
    assert isinstance(main, pn.Column)


def test_splats_page_view_still_returns_template(tmp_path):
    app = _app(tmp_path)
    tpl = app.view()
    assert isinstance(tpl, pn.template.MaterialTemplate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_shell.py -v`
Expected: FAIL — `AttributeError: 'SplatsApp' object has no attribute 'sidebar'`

- [ ] **Step 3: Implement the split**

In `collab_splats/dashboard/app.py`, replace `view()` (`app.py:531-559`) with three methods:

```python
    def sidebar(self) -> pn.Column:
        """Sidebar contents — composed by view() or by the tabbed shell."""
        return self._sidebar

    def main(self) -> pn.Column:
        """Main-area contents: split viewer + live progress strip."""
        # Live operations strip: stage label + progress bar + scrolling per-step log.
        # Poll the shared op_log on THIS session's IOLoop (op_log is mutated from the GpuWorker
        # thread; pushing Bokeh updates cross-thread glitches). Polling reads a locked snapshot and
        # updates the pane on the IOLoop → flicker-free, and a refreshed page re-attaches live.
        progress = pn.pane.HTML(self._op_log.render_html(), sizing_mode="stretch_width")

        def _tick() -> None:
            progress.object = self._op_log.render_html()

        try:
            pn.state.add_periodic_callback(_tick, period=300, start=True)
        except Exception:
            # No live server (tests) — leave the static snapshot.
            logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)

        return pn.Column(self._viewer.layout, progress, sizing_mode="stretch_both")

    def view(self) -> pn.template.MaterialTemplate:
        """Standalone single-page layout (kept for tests and direct serving).

        The 'vtk' extension is loaded once in run_app (main thread, before serving) —
        loading it here per-session fails to inject the VTK JS and the panes hang.
        """
        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self.sidebar()],
            main=[self.main()],
            header_background="#2596be",
            sidebar_width=340,
        )
```

Add the spec's rename as an alias after the class definition (cosmetic; tests and callers keep working):

```python
# Spec 2026-07-14 names this class SplatsPage; alias until callers migrate.
SplatsPage = SplatsApp
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -v`
Expected: new tests PASS; every pre-existing dashboard test green (the split is behaviour-preserving).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_shell.py
git commit -m "refactor(dashboard): split SplatsApp.view into sidebar()/main() for the tabbed shell"
```

---

### Task 9: `LocalizePage` + `SceneCache`

The one substantial new file. Pure logic (method preselect, camera subsampling, camera centers) is unit-tested; Panel wiring follows `SplatsApp` patterns and is manually verified in Task 10.

**Files:**
- Create: `collab_splats/dashboard/localize.py`
- Test: `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_localize_page.py`:

```python
"""Pure-logic tests for the localize page helpers."""
import numpy as np

from collab_splats.dashboard.localize import (
    SceneCache,
    camera_centers,
    preselect_method,
    subsample_step,
)


def test_camera_centers_inverts_world_to_camera():
    # Camera at world (1, 2, 3), identity rotation: extrinsic t = -R @ C = -C
    ext = np.eye(4, dtype=np.float32)[None]
    ext[0, :3, 3] = [-1.0, -2.0, -3.0]
    centers = camera_centers(ext)
    np.testing.assert_allclose(centers[0], [1.0, 2.0, 3.0], atol=1e-6)


def test_subsample_step_thresholds():
    assert subsample_step(30) == 1
    assert subsample_step(60) == 1
    assert subsample_step(61) == 3
    assert subsample_step(300) == 3


def test_preselect_existing_db_wins():
    options, value = preselect_method(["disk"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "disk"
    assert options == ["disk", "xfeat", "loma", "loma-g"]


def test_preselect_defaults_to_loma_g_when_no_db():
    _, value = preselect_method([], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma-g"


def test_preselect_prefers_loma_g_among_multiple_dbs():
    _, value = preselect_method(["disk", "loma-g"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma-g"


def test_scene_cache_roundtrip():
    cache = SceneCache()
    assert cache.get(("s", "v"), "mesh") is None
    cache.put(("s", "v"), "mesh", object())
    assert cache.get(("s", "v"), "mesh") is not None
    cache.clear()
    assert cache.get(("s", "v"), "mesh") is None


def test_scene_cache_drop_kind_prefix():
    cache = SceneCache()
    cache.put(("s", "v"), "mesh", object())
    cache.put(("s", "v"), "localizer:loma-g", object())
    cache.put(("s", "w"), "localizer:disk", object())
    cache.drop_kind("localizer")
    assert cache.get(("s", "v"), "localizer:loma-g") is None
    assert cache.get(("s", "w"), "localizer:disk") is None
    assert cache.get(("s", "v"), "mesh") is not None  # CPU loads survive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.dashboard.localize'`

- [ ] **Step 3: Implement `localize.py`**

Create `collab_splats/dashboard/localize.py`:

```python
"""Localization page: localize an rgb_X field-camera frame against a reconstruction."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import panel as pn
import param
import pyvista as pv

from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource

# NB: pipeline / localization viz imports are lazy (inside run/render paths) — they pull
# the heavy reconstruction stack, and the page must render immediately on launch.

logger = logging.getLogger(__name__)

########
# Constants + pure helpers (unit-tested)
########

_METHODS = ["disk", "xfeat", "loma", "loma-g"]
_DEFAULT_METHOD = "loma-g"
_SUBSAMPLE_ABOVE = 60  # plot every 3rd camera beyond this many reconstruction frames


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """World-space camera centers C = -R^T t from (N, 4, 4) world-to-camera transforms."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def subsample_step(n_cameras: int) -> int:
    """1 (all cameras) up to the threshold, 3 (every 3rd) beyond it."""
    return 1 if n_cameras <= _SUBSAMPLE_ABOVE else 3


def preselect_method(available_dbs: list[str], registered: list[str],
                     default: str = _DEFAULT_METHOD) -> tuple[list[str], str]:
    """Dropdown (options, value): prefer the default method's DB, then any existing DB,
    else the default (which will build on demand)."""
    if default in available_dbs:
        return registered, default
    if available_dbs:
        return registered, available_dbs[0]
    return registered, default


class SceneCache:
    """Session-level cache of expensive loads, keyed (scene_key, kind).

    CPU loads (mesh, arrays) persist across tabs; GPU-holding entries use a
    'localizer:*' kind prefix so drop_kind('localizer') can evict them on tab switch."""

    def __init__(self) -> None:
        self._store: dict = {}

    def get(self, scene_key, kind: str):
        return self._store.get((scene_key, kind))

    def put(self, scene_key, kind: str, value) -> None:
        self._store[(scene_key, kind)] = value

    def drop_kind(self, prefix: str) -> None:
        """Evict every entry whose kind starts with prefix (e.g. GPU-holding localizers)."""
        for key in [k for k in self._store if k[1].startswith(prefix)]:
            del self._store[key]

    def clear(self) -> None:
        self._store.clear()


########
# Page
########


class LocalizePage(param.Parameterized):
    """Sidebar (scene + query + method) and three-panel result layout."""

    def __init__(
        self,
        base_dir: Path,
        source: SessionSource,
        gpu_worker: GpuWorker,
        op_log: OperationLog,
        cache: SceneCache | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source
        self._gpu = gpu_worker
        self._op_log = op_log
        self._cache = cache if cache is not None else SceneCache()
        self._build_sidebar()
        self._build_main()
        self._refresh_listings()

    # ---- sidebar -------------------------------------------------------

    def _build_sidebar(self) -> None:
        """Scene (reconstruction) + query (field camera) + method widgets."""
        self.scene_session = pn.widgets.Select(name="Scene session", options=[])
        self.scene_video = pn.widgets.Select(name="Scene video", options=[])
        self.field_session = pn.widgets.Select(name="Field session", options=[])
        self.camera = pn.widgets.Select(name="Camera (rgb only)", options=[])
        self.query_video = pn.widgets.Select(name="Query video", options=[])
        self.frame_slider = pn.widgets.IntSlider(name="Frame", start=0, end=0, value=0)
        self.method = pn.widgets.Select(name="Method", options=_METHODS, value=_DEFAULT_METHOD)
        self.db_note = pn.pane.HTML("", sizing_mode="stretch_width")
        self.append_db = pn.widgets.Checkbox(name="Append localized frame to DB", value=True)
        self.run_btn = pn.widgets.Button(label="Run", button_type="primary")

        self.scene_session.param.watch(self._on_scene_session, "value")
        self.scene_video.param.watch(self._on_scene_video, "value")
        self.field_session.param.watch(self._on_field_session, "value")
        self.camera.param.watch(self._on_camera, "value")
        self.query_video.param.watch(self._on_query_video, "value")
        self.method.param.watch(self._on_method, "value")
        self.run_btn.on_click(self._on_run)

        self._sidebar = pn.Column(
            "## Scene",
            self.scene_session,
            self.scene_video,
            "## Query",
            self.field_session,
            self.camera,
            self.query_video,
            self.frame_slider,
            "## Localization",
            self.method,
            self.db_note,
            self.append_db,
            self.run_btn,
        )

    def sidebar(self) -> pn.Column:
        return self._sidebar

    # ---- main layout ---------------------------------------------------

    def _build_main(self) -> None:
        """Left frame/matches column, right 3D pane, bottom distribution + stats."""
        self._frame_pane = pn.pane.Image(None, sizing_mode="scale_width")
        self._matches_col = pn.Column(self._frame_pane, sizing_mode="stretch_width",
                                      scroll=True, max_height=700)
        self._plotter = pv.Plotter(off_screen=True)
        self._vtk_pane = pn.pane.VTK(self._plotter.ren_win, sizing_mode="stretch_both",
                                     min_height=500)
        self._dist_pane = pn.pane.Matplotlib(None, sizing_mode="stretch_width", tight=True)
        self._stats = pn.pane.HTML("", sizing_mode="stretch_width")

        # Progress strip: identical polling pattern to SplatsApp.main()
        self._progress = pn.pane.HTML(self._op_log.render_html(), sizing_mode="stretch_width")

    def main(self) -> pn.Column:
        def _tick() -> None:
            self._progress.object = self._op_log.render_html()

        try:
            pn.state.add_periodic_callback(_tick, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)

        top = pn.Row(self._matches_col, self._vtk_pane, sizing_mode="stretch_both")
        bottom = pn.Column(self._dist_pane, self._stats, sizing_mode="stretch_width")
        return pn.Column(top, bottom, self._progress, sizing_mode="stretch_both")

    def release_gpu(self) -> None:
        """Free GPU memory when the user leaves this tab (models reload on next run).

        Warm localizers hold the extractor model — evict them first or pytorch_gc
        cannot actually release the VRAM they reference."""
        from collab_splats.utils.torch_utils import pytorch_gc

        self._cache.drop_kind("localizer")
        pytorch_gc()

    # ---- listings (background threads, options set on the IOLoop) -------

    def _refresh_listings(self) -> None:
        """Populate scene sessions and field sessions off the IOLoop (rclone is blocking)."""
        doc = pn.state.curdoc

        def work():
            try:
                scenes = self._source.list_sessions()
            except Exception as exc:
                logger.warning("scene session listing failed: %s", exc)
                scenes = []
            try:
                fields = self._source.list_field_sessions()
            except Exception as exc:
                logger.warning("field session listing failed: %s", exc)
                fields = []

            def setter():
                self.scene_session.options = scenes
                self.field_session.options = fields

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="localize-list", daemon=True).start()

    def _on_scene_session(self, event) -> None:
        if not event.new:
            return
        self.scene_video.options = [Path(v).stem for v in self._source.list_videos(event.new)]

    def _on_scene_video(self, event) -> None:
        """Scene chosen → discover remote feature DBs and preselect the method."""
        if not event.new:
            return
        session, stem = self.scene_session.value, event.new
        doc = pn.state.curdoc

        def work():
            dbs = self._source.list_localization_dbs(session, stem)

            def setter():
                options, value = preselect_method(dbs, _METHODS)
                self.method.options = options
                self.method.value = value
                self._dbs = dbs
                self._update_db_note()

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="db-list", daemon=True).start()

    def _on_method(self, event) -> None:
        self._update_db_note()

    def _update_db_note(self) -> None:
        """Warn when the selected method has no DB yet (run will build it on GPU)."""
        dbs = getattr(self, "_dbs", [])
        if self.method.value in dbs:
            self.db_note.object = "<span style='color:#50c050;font-size:11px'>DB exists — will reuse</span>"
        else:
            self.db_note.object = (
                "<span style='color:#e0a050;font-size:11px'>no DB for this method — "
                "Run will build it (GPU, minutes)</span>"
            )

    def _on_field_session(self, event) -> None:
        if not event.new:
            return
        self.camera.options = self._source.list_rgb_cameras(event.new)

    def _on_camera(self, event) -> None:
        if not event.new:
            return
        self.query_video.options = self._source.list_camera_videos(
            self.field_session.value, event.new)

    def _on_query_video(self, event) -> None:
        """Fetch the video in the background; set slider bound + preview frame 0."""
        if not event.new:
            return
        fs, cam, name = self.field_session.value, self.camera.value, event.new
        doc = pn.state.curdoc

        def work():
            try:
                video = self._ensure_local_query_video(fs, cam, name)
                from collab_splats.preproc import extract_frame, get_video_info

                total = int(get_video_info(str(video)).get("total_frames") or 1)
                frame = extract_frame(video, 0)
            except Exception:
                logger.warning("query video fetch/preview failed", exc_info=True)
                return

            def setter():
                self.frame_slider.end = max(total - 1, 0)
                self.frame_slider.value = 0
                self._show_frame(frame)

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="query-video", daemon=True).start()

    def _ensure_local_query_video(self, field_session: str, camera: str, name: str) -> Path:
        local = self._base_dir / "queries" / field_session / camera / name
        if not local.exists():
            local = self._source.fetch_field_video(field_session, camera, name, local.parent)
        return local

    def _show_frame(self, frame: np.ndarray) -> None:
        """Show the selected query frame in the left panel (pre-run state)."""
        from PIL import Image as PILImage

        # pn.pane.Image renders PIL images directly; raw bytes are not accepted
        self._frame_pane.object = PILImage.fromarray(frame)
        self._matches_col[:] = [self._frame_pane]

    # ---- run -----------------------------------------------------------

    def _current_config(self) -> LocalizationConfig:
        return LocalizationConfig(extractor=self.method.value, append_to_db=self.append_db.value)

    def _on_run(self, event) -> None:
        scene_session = self.scene_session.value
        stem = self.scene_video.value
        fs, cam, name = self.field_session.value, self.camera.value, self.query_video.value
        frame_idx = self.frame_slider.value
        if not (scene_session and stem and fs and cam and name):
            self._op_log.error_op("select a scene and a query video first")
            return
        config = self._current_config()
        provenance = {
            "video_ref": f"{fs}/{cam}/{name}",
            "session": fs,
            "camera": cam,
            "frame_idx": int(frame_idx),
        }
        doc = pn.state.curdoc

        def job():
            # Lazy import: pulls the heavy stack only when a run starts (mirrors SplatsApp)
            from collab_splats.dashboard.pipeline import run_localization

            video = self._ensure_local_query_video(fs, cam, name)
            return run_localization(
                query_video=video,
                frame_idx=frame_idx,
                session=scene_session,
                stem=stem,
                config=config,
                op_log=self._op_log,
                source=self._source,
                base_dir=self._base_dir,
                provenance=provenance,
                cache=self._cache,  # keeps the localizer (and its extractor) warm across runs
            )

        def on_done(res):
            self.run_btn.disabled = False
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._render_result(res, config)

        self.run_btn.disabled = True
        self._gpu.submit(job, on_done, doc)

    # ---- rendering -----------------------------------------------------

    def _render_result(self, out, config: LocalizationConfig) -> None:
        """Fill all three panels from a LocalizationRunOutput (IOLoop thread)."""
        from collab_splats.localization.viz import plot_correspondences, plot_inlier_distribution

        loc = out.result
        n_frames = len(out.ref_image_paths)

        # Bottom: inlier distribution + summary stats
        fig = plot_inlier_distribution(loc, n_frames=n_frames, frame_sources=out.frame_sources)
        self._dist_pane.object = fig
        ratio = 100 * loc.n_inliers / max(loc.n_correspondences, 1)
        pose_msg = "" if loc.pose is not None else " — <b style='color:#e05050'>POSE FAILED</b>"
        self._stats.object = (
            f"<div style='font-size:12px'>inliers {loc.n_inliers}/{loc.n_correspondences} "
            f"({ratio:.0f}%) · intrinsics: {out.intrinsics_source} "
            f"(fx={out.query_intrinsics[0, 0]:.0f}){pose_msg}</div>"
        )

        # Left: top-k match-pair figures, best-first (replaces the frame preview)
        if loc.ref_frame_indices is not None and loc.inlier_mask is not None:
            counts = np.bincount(
                loc.ref_frame_indices[loc.inlier_mask].astype(np.intp), minlength=n_frames)
            top = np.argsort(counts)[::-1][: config.top_k_viz]
            panes = []
            for ref in top:
                if counts[ref] == 0 or not Path(out.ref_image_paths[ref]).exists():
                    continue
                mfig = plot_correspondences(
                    loc, out.query_frame, out.ref_image_paths,
                    max_pairs=config.max_pairs, ref_idx=int(ref), show=False)
                if mfig is not None:
                    panes.append(pn.pane.Matplotlib(mfig, sizing_mode="stretch_width", tight=True))
            if panes:
                self._matches_col[:] = panes

        # Right: mesh + viridis reconstruction cameras + red localized camera
        scene_key = (self.scene_session.value, self.scene_video.value)
        mesh_path = self._base_dir / scene_key[0] / scene_key[1] / "mesh" / "mesh_tsdf.ply"
        self._render_scene(scene_key, mesh_path, out.ref_extrinsics, loc.pose)

    def _render_scene(self, scene_key, mesh_path: Path, extrinsics: np.ndarray,
                      localized_pose: "np.ndarray | None") -> None:
        """Rebuild the 3D pane: mesh, time-coloured cameras, red localized camera."""
        self._plotter.clear()

        # Mesh (cached across runs and tabs — expensive read)
        mesh = self._cache.get(scene_key, "mesh")
        if mesh is None and mesh_path.exists():
            mesh = pv.read(str(mesh_path))
            self._cache.put(scene_key, "mesh", mesh)
        if mesh is not None:
            self._plotter.add_mesh(mesh, rgb="RGB" in mesh.array_names, opacity=0.9)

        # Reconstruction cameras: viridis by time; subsampled with an on-plot note
        centers = camera_centers(np.asarray(extrinsics))
        step = subsample_step(len(centers))
        sub = centers[::step]
        poly = pv.PolyData(sub)
        poly["time"] = np.arange(len(sub), dtype=np.float32)
        self._plotter.add_mesh(poly, scalars="time", cmap="viridis", point_size=14,
                               render_points_as_spheres=True, show_scalar_bar=False)
        if step > 1:
            self._plotter.add_text(f"showing every {step}rd camera", font_size=8,
                                   position="lower_left")

        # Localized camera in red, drawn larger
        if localized_pose is not None:
            loc_center = camera_centers(localized_pose[np.newaxis])
            self._plotter.add_mesh(pv.PolyData(loc_center), color="red", point_size=22,
                                   render_points_as_spheres=True)

        self._plotter.reset_camera()
        self._vtk_pane.synchronize()
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_localize_page.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/localize.py tests/dashboard/test_localize_page.py
git commit -m "feat(dashboard): LocalizePage — three-panel localization UI + SceneCache"
```

---

### Task 10: Tabbed shell + entry point

**Files:**
- Create: `collab_splats/dashboard/shell.py`
- Modify: `collab_splats/dashboard/__main__.py`
- Test: `tests/dashboard/test_shell.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/dashboard/test_shell.py`:

```python
def test_shell_builds_tabs_with_two_pages(tmp_path):
    from collab_splats.dashboard.shell import DashboardShell

    shell = DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())
    tpl = shell.view()
    assert isinstance(tpl, pn.template.MaterialTemplate)
    assert len(shell._tabs) == 2
    assert [t for t in shell._tabs._names] == ["Splats", "Localize"]


def test_shell_sidebar_swaps_on_tab_change(tmp_path):
    from collab_splats.dashboard.shell import DashboardShell

    shell = DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())
    shell.view()
    splats_sidebar = shell._splats.sidebar()
    localize_sidebar = shell._localize.sidebar()
    assert shell._sidebar_holder[0] is splats_sidebar
    shell._tabs.active = 1
    assert shell._sidebar_holder[0] is localize_sidebar
    shell._tabs.active = 0
    assert shell._sidebar_holder[0] is splats_sidebar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_shell.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.dashboard.shell'`

- [ ] **Step 3: Implement `shell.py`**

Create `collab_splats/dashboard/shell.py`:

```python
"""Tabbed shell: splats + localize pages in one Panel session (no-reload switching)."""

from __future__ import annotations

import logging
from pathlib import Path

import panel as pn

from collab_splats.dashboard.app import SplatsApp, _ensure_display
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.localize import LocalizePage, SceneCache
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource

logger = logging.getLogger(__name__)


class DashboardShell:
    """One MaterialTemplate hosting both pages under dynamic tabs; sidebar follows the tab."""

    def __init__(
        self,
        base_dir: Path,
        source: SessionSource | None = None,
        gpu_worker: GpuWorker | None = None,
        op_log: OperationLog | None = None,
    ) -> None:
        source = source if source is not None else SessionSource()
        gpu_worker = gpu_worker if gpu_worker is not None else GpuWorker()
        op_log = op_log if op_log is not None else OperationLog()
        self._cache = SceneCache()
        self._splats = SplatsApp(base_dir=Path(base_dir), source=source,
                                 gpu_worker=gpu_worker, op_log=op_log)
        self._localize = LocalizePage(base_dir=Path(base_dir), source=source,
                                      gpu_worker=gpu_worker, op_log=op_log, cache=self._cache)
        self._tabs: pn.Tabs | None = None
        self._sidebar_holder: pn.Column | None = None

    def _on_tab(self, event) -> None:
        """Swap sidebar contents to match the active tab; free GPU when leaving localize."""
        page = self._splats if event.new == 0 else self._localize
        self._sidebar_holder[:] = [page.sidebar()]
        if event.old == 1:
            self._localize.release_gpu()

    def view(self) -> pn.template.MaterialTemplate:
        """Assemble tabs + swapping sidebar. dynamic=True defers the localize page's
        VTK build until first visit; both pages stay alive after that."""
        self._tabs = pn.Tabs(
            ("Splats", self._splats.main()),
            ("Localize", self._localize.main()),
            dynamic=True,
            sizing_mode="stretch_both",
        )
        self._sidebar_holder = pn.Column(self._splats.sidebar(), sizing_mode="stretch_width")
        self._tabs.param.watch(self._on_tab, "active")
        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self._sidebar_holder],
            main=[self._tabs],
            header_background="#2596be",
            sidebar_width=340,
        )
```

Note for the implementer: `pn.Tabs` stores tab titles in `_names`; if the installed Panel version doesn't expose it, assert on `len(shell._tabs)` only and drop the names assertion — do not add a public-API workaround for a test-only need.

- [ ] **Step 4: Wire the entry point**

In `collab_splats/dashboard/app.py`, update `run_app`'s factory to serve the shell (keep everything else — Xvfb, `pn.extension("vtk", inline=True)`, shared worker/log — unchanged):

```python
    def factory() -> pn.template.MaterialTemplate:
        # Tabbed shell: splats + localize pages share one session, worker, and op_log.
        from collab_splats.dashboard.shell import DashboardShell

        return DashboardShell(base_dir=Path(base_dir), gpu_worker=gpu_worker, op_log=op_log).view()
```

(The import stays local to `factory` to avoid an `app.py` ↔ `shell.py` circular import at module load; `shell.py` imports `SplatsApp` from `app.py` at the top.)

Check `collab_splats/dashboard/__main__.py`: it calls `run_app` — no change needed if so; if it constructs `SplatsApp` directly, switch it to `run_app`.

- [ ] **Step 5: Run the full dashboard test module + suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -v`
Expected: all PASS.
Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: suite green (compare against `docs/known-test-failures.md` for pre-existing xfails).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/shell.py collab_splats/dashboard/app.py collab_splats/dashboard/__main__.py tests/dashboard/test_shell.py
git commit -m "feat(dashboard): tabbed shell — splats + localize pages, swapping sidebar"
```

---

### Task 11: Manual verification + format pass

- [ ] **Step 1: Format and lint**

```bash
black collab_splats/ tests/ && isort collab_splats/ tests/
git diff --stat  # commit any formatting changes with the previous task's scope
```

- [ ] **Step 2: Launch the dashboard**

```bash
source /opt/venv/reconstruction/bin/activate
python -m collab_splats.dashboard
```

Verify checklist (browser):
1. Two tabs render; switching does not reload the page; sidebar follows the tab.
2. Localize tab: scene session/video dropdowns populate; picking a scene fills the method dropdown note (green "DB exists" when a DB was previously built, orange warning otherwise).
3. Field session → camera (rgb_X only) → video dropdowns cascade; selecting a video shows frame 0 on the left and sets the slider bound.
4. Run with an existing DB: progress strip walks pull → load DB → extract → intrinsics → solve → append → push; three panels fill (match pairs left, mesh + cameras right with red localized camera, distribution + stats bottom).
5. Run with a method that has no DB: orange warning shown beforehand; progress strip shows the DB build band (25→55%).
6. `estimate_intrinsics` real-path smoke: stats line shows a plausible fx (same order of magnitude as the reconstruction camera's fx).
7. Failure path: pick a frame with no overlap (e.g., camera pointing away) — stats line shows POSE FAILED, no append, distribution still renders.

- [ ] **Step 3: Verify the zarr after an append run**

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import zarr, sys
store = zarr.open("/workspace/outputs/<session>/<stem>/feedforward.zarr", mode="r")
g = store["local_features/loma-g/localized"]
print(g.attrs["image_paths"]); print(g.attrs["provenance"])
EOF
```

Expected: appended frame path + provenance dict with `video_ref/session/camera/frame_idx`.

- [ ] **Step 4: Commit any fixes; update docs index**

Add the page to the module docs if `docs/` has a dashboard page (mirror existing style, brief). Commit:

```bash
git commit -am "docs(dashboard): localize page notes"
```

---

## Deferred (tracked in spec, not in this plan)

- Intrinsics-validation eval harness (known-intrinsics cameras).
- thermal_X cameras; per-camera calibration workflow (only `calibration_path` hook ships).
- SplatsPage consuming SceneCache (cross-tab mesh reuse for the splats viewer itself).
- Per-session localized-frames manifest for cross-reconstruction lookup.
- Excluding dense zarr arrays (`depth`/`world_points`/`confidence`) from the localization
  pull: requires auditing `FeedforwardResult.load_zarr` for tolerance to missing members
  before extending `_PULL_EXCLUDES`. v1 excludes only `frames.zarr/**` (safe duplicate).
