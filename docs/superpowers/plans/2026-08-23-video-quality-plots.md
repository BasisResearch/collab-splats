# Video Quality Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render `video_quality_report.json` to five static PNGs beside `frames.zarr`, written by `extract_frames`, reusable from notebooks.

**Architecture:** Five standalone matplotlib plotters in `collab_splats/preproc/viz.py` — one per measurement family, one PNG each, every block (unpack, time axis, panels, overlay, title, save) inlined; no helpers, no registry. `extract_frames` (video branch) calls the five explicitly after `FrameStore.create`, passing the selected frame indices as an overlay.

**Tech Stack:** matplotlib (Agg, already a dependency), numpy. Python via `/opt/venv/reconstruction/bin/python`. seaborn deliberately not used (not installed, not a dependency, adds nothing to raw line plots).

**Spec:** `docs/superpowers/specs/2026-08-23-video-quality-plots-design.md`

**Conventions that apply (CLAUDE.md):** imports at top, block comments per logical block, `########` section dividers, one-line docstrings, `logging` not `print`, flat test functions, conventional commits with scope. Commit with `git commit --only <files>` (shared index across concurrent sessions — see memory). Format touched files with `black <file> && isort <file>` only (never repo-wide).

---

## File map

| File | Change |
|---|---|
| `collab_splats/preproc/viz.py` | New section `# Video quality report plots` with `plot_photometric_blur`, `plot_photometric_exposure`, `plot_motion_translation`, `plot_motion_parallax`, `plot_motion_matches`. Three module constants (`_FIG_WIDTH_IN`, `_PANEL_HEIGHT_IN`, `_PNG_DPI`). |
| `tests/preproc/test_viz.py` | New section: `_fake_report()` + `_assert_png()` + one test per plotter + empty-pairs test. |
| `collab_splats/wrapper/reconstructor.py:24-27,181` | Import `viz as preproc_viz`; five calls after `FrameStore.create` in the video branch. |
| `tests/wrapper/test_reconstructor.py:137` | Stub `preproc_viz` in the dispatch test; new test asserting five PNGs from a full synthetic report. |
| `configs/README.md:51,286` | List the five PNGs in the output layout and the measure/select section. |

Every plotter is the same shape — read Task 1's implementation once, the rest differ only in columns and panels.

---

### Task 1: Synthetic report fixture + `plot_photometric_blur`

**Files:**
- Modify: `collab_splats/preproc/viz.py` (imports at top; append new section at end)
- Test: `tests/preproc/test_viz.py` (append at end)

- [ ] **Step 1: Write the failing test**

Extend the import block at the top of `tests/preproc/test_viz.py`:

```python
from collab_splats.preproc.viz import (
    plot_disparity_sensitivity,
    plot_frame_grid,
    plot_frame_scores,
    plot_photometric_blur,
    plot_quality_examples,
    plot_selection,
)
```

Append at the end of the file:

```python
########################################################################
# Video quality report plots
########################################################################


def _fake_report(n_frames=20, fps=10.0, stride=2, n_pairs=10):
    """
    Report shaped like qa.compute_video_quality output; pair 3 failed to match.
    """
    rng = np.random.default_rng(0)
    frames = {
        "frame_idx": list(range(n_frames)),
        "blur": rng.uniform(0.1, 0.9, n_frames).tolist(),
        "laplacian": rng.uniform(50, 500, n_frames).tolist(),
        "exposure_mean": rng.uniform(80, 160, n_frames).tolist(),
        "exposure_median": rng.uniform(80, 160, n_frames).tolist(),
        "exposure_std": rng.uniform(20, 60, n_frames).tolist(),
        "clipped_low_frac": rng.uniform(0, 0.05, n_frames).tolist(),
        "clipped_high_frac": rng.uniform(0, 0.05, n_frames).tolist(),
    }
    a = list(range(0, n_pairs * stride, stride))
    translation = rng.uniform(0, 30, n_pairs).tolist()
    parallax = rng.uniform(0, 1, n_pairs).tolist()
    # A failed pair is None in the dict (nan -> null in the JSON)
    if n_pairs > 3:
        translation[3] = None
        parallax[3] = None
    pairs = {
        "frame_idx_a": a,
        "frame_idx_b": [i + stride for i in a],
        "n_matches": rng.integers(0, 500, n_pairs).tolist(),
        "translation_px": translation,
        "parallax": parallax,
    }
    return {
        "available": True,
        "video": {"path": "/data/clip.mp4", "fps": fps, "total_frames": n_frames, "width": 64, "height": 48},
        "params": {"motion_stride": stride},
        "frames": frames,
        "pairs": pairs,
    }


def _assert_png(path, expected_name):
    assert path.name == expected_name
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_photometric_blur_writes_png(tmp_path):
    _assert_png(plot_photometric_blur(_fake_report(), tmp_path), "photometric-blur.png")
    # Overlay path, into a directory that does not exist yet
    _assert_png(plot_photometric_blur(_fake_report(), tmp_path / "new", selected=[3, 7]), "photometric-blur.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: `ImportError: cannot import name 'plot_photometric_blur'`

- [ ] **Step 3: Implement**

Add to the import block at the top of `collab_splats/preproc/viz.py` (after `from __future__ import annotations`):

```python
from pathlib import Path
```

Append at the end of the file:

```python
########################################################################
# Video quality report plots
#
# One PNG per measurement family of qa.compute_video_quality's report. Every
# plotter draws EVERY frame / pair in the report as shipped — raw columns, no
# thresholds, no verdicts. `selected` only marks which frames made it into
# frames.zarr. Headless: save and close, never plt.show(). Title, overlay and
# save are inlined in each plotter on purpose — five short duplicates beat a
# helper layer.
########################################################################

_FIG_WIDTH_IN = 12
_PANEL_HEIGHT_IN = 2.8
_PNG_DPI = 90  # matches collab-data/track_reprojection/report.py


def plot_photometric_blur(report: dict, out_dir: str | Path, *, selected=None) -> Path:
    """
    blur (↑ blurrier) over laplacian variance (↑ sharper), per frame, vs seconds.
    """
    # Unpack columns
    frames, video = report["frames"], report["video"]
    fps = video["fps"]
    blur = np.asarray(frames["blur"], dtype=float)
    laplacian = np.asarray(frames["laplacian"], dtype=float)

    # Time axis: wall-clock seconds
    t = np.asarray(frames["frame_idx"], dtype=float) / fps

    # Panels: opposite directions on purpose — a saturated blur reads against laplacian
    fig, axes = plt.subplots(2, 1, figsize=(_FIG_WIDTH_IN, _PANEL_HEIGHT_IN * 2), sharex=True)
    axes[0].plot(t, blur, linewidth=0.8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("blur (↑ blurrier)")
    axes[1].plot(t, laplacian, linewidth=0.8)
    axes[1].set_ylabel("laplacian var (↑ sharper)")
    axes[1].set_xlabel("time (s)")

    # Selected-frame overlay: 1 px lines, never axvspan — a one-frame span is
    # under a pixel on a long video and vanishes
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        for ax in axes:
            ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    fig.suptitle(
        f"{Path(video['path']).name} — {len(t)} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "photometric-blur.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/viz.py tests/preproc/test_viz.py && isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py -m "feat(preproc): plot_photometric_blur — first video quality report PNG"
```

---

### Task 2: `plot_photometric_exposure`

**Files:**
- Modify: `collab_splats/preproc/viz.py` (append after `plot_photometric_blur`)
- Test: `tests/preproc/test_viz.py`

- [ ] **Step 1: Write the failing test**

Add `plot_photometric_exposure` to the import list; append:

```python
def test_plot_photometric_exposure_writes_png(tmp_path):
    _assert_png(plot_photometric_exposure(_fake_report(), tmp_path), "photometric-exposure.png")
    _assert_png(plot_photometric_exposure(_fake_report(), tmp_path, selected=[3, 7]), "photometric-exposure.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: `ImportError: cannot import name 'plot_photometric_exposure'`

- [ ] **Step 3: Implement**

```python
def plot_photometric_exposure(report: dict, out_dir: str | Path, *, selected=None) -> Path:
    """
    Exposure mean ± std band with median, over stacked clipping fractions, per frame.
    """
    # Unpack columns
    frames, video = report["frames"], report["video"]
    fps = video["fps"]
    mean = np.asarray(frames["exposure_mean"], dtype=float)
    median = np.asarray(frames["exposure_median"], dtype=float)
    std = np.asarray(frames["exposure_std"], dtype=float)
    clip_lo = np.asarray(frames["clipped_low_frac"], dtype=float)
    clip_hi = np.asarray(frames["clipped_high_frac"], dtype=float)

    # Time axis: wall-clock seconds
    t = np.asarray(frames["frame_idx"], dtype=float) / fps

    # Panels: brightness on the 8-bit scale, then the destroyed-pixel fractions
    fig, axes = plt.subplots(2, 1, figsize=(_FIG_WIDTH_IN, _PANEL_HEIGHT_IN * 2), sharex=True)
    axes[0].fill_between(t, mean - std, mean + std, alpha=0.2, label="mean ± std")
    axes[0].plot(t, mean, linewidth=0.8, label="mean")
    axes[0].plot(t, median, linewidth=0.8, linestyle="--", label="median")
    axes[0].set_ylim(0, 255)
    axes[0].set_ylabel("exposure (gray level)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].stackplot(t, clip_lo, clip_hi, labels=["clipped low", "clipped high"], alpha=0.7)
    axes[1].set_ylabel("clipped pixel fraction")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=8)

    # Selected-frame overlay: 1 px lines, never axvspan
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        for ax in axes:
            ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    fig.suptitle(
        f"{Path(video['path']).name} — {len(t)} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "photometric-exposure.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/viz.py tests/preproc/test_viz.py && isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py -m "feat(preproc): plot_photometric_exposure PNG"
```

---

### Task 3: `plot_motion_translation`

**Files:**
- Modify: `collab_splats/preproc/viz.py`
- Test: `tests/preproc/test_viz.py`

- [ ] **Step 1: Write the failing test**

Add `plot_motion_translation` to the import list; append:

```python
def test_plot_motion_translation_writes_png(tmp_path):
    _assert_png(plot_motion_translation(_fake_report(), tmp_path), "motion-translation.png")
    _assert_png(plot_motion_translation(_fake_report(), tmp_path, selected=[3, 7]), "motion-translation.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: `ImportError: cannot import name 'plot_motion_translation'`

- [ ] **Step 3: Implement**

```python
def plot_motion_translation(report: dict, out_dir: str | Path, *, selected=None) -> Path:
    """
    Per-pair translation in pixels over its cumulative sum (camera path-length proxy).
    """
    # Unpack columns: None (failed pair) -> nan in one place
    pairs, video = report["pairs"], report["video"]
    fps = video["fps"]
    translation = np.asarray(pairs["translation_px"], dtype=float)

    # Time axis: a pair sits at its first member
    t = np.asarray(pairs["frame_idx_a"], dtype=float) / fps

    # Panels: raw per-pair value (nan leaves a gap), then the path proxy (nan counts as 0)
    fig, axes = plt.subplots(2, 1, figsize=(_FIG_WIDTH_IN, _PANEL_HEIGHT_IN * 2), sharex=True)
    axes[0].plot(t, translation, linewidth=0.8)
    axes[0].set_ylabel("translation (px / pair)")
    axes[1].plot(t, np.cumsum(np.nan_to_num(translation, nan=0.0)), linewidth=0.8)
    axes[1].set_ylabel("path-length proxy (px)")
    axes[1].set_xlabel("time (s)")

    # Selected-frame overlay: 1 px lines, never axvspan
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        for ax in axes:
            ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    n_frames = len(report["frames"]["frame_idx"])
    fig.suptitle(
        f"{Path(video['path']).name} — {n_frames} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "motion-translation.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/viz.py tests/preproc/test_viz.py && isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py -m "feat(preproc): plot_motion_translation PNG"
```

---

### Task 4: `plot_motion_parallax` — failed pairs stay visible

**Files:**
- Modify: `collab_splats/preproc/viz.py`
- Test: `tests/preproc/test_viz.py`

- [ ] **Step 1: Write the failing test**

Add `plot_motion_parallax` to the import list; append:

```python
def test_plot_motion_parallax_writes_png(tmp_path):
    _assert_png(plot_motion_parallax(_fake_report(), tmp_path), "motion-parallax.png")
    _assert_png(plot_motion_parallax(_fake_report(), tmp_path, selected=[3, 7]), "motion-parallax.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: `ImportError: cannot import name 'plot_motion_parallax'`

- [ ] **Step 3: Implement**

```python
def plot_motion_parallax(report: dict, out_dir: str | Path, *, selected=None) -> Path:
    """
    Per-pair parallax vs seconds; pairs that failed to match drawn as red ticks at 0.
    """
    # Unpack columns: None (failed pair) -> nan in one place
    pairs, video = report["pairs"], report["video"]
    fps = video["fps"]
    parallax = np.asarray(pairs["parallax"], dtype=float)

    # Time axis: a pair sits at its first member
    t = np.asarray(pairs["frame_idx_a"], dtype=float) / fps

    # Panel: failed pairs are the worst pairs — marked, never dropped
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_WIDTH_IN, _PANEL_HEIGHT_IN))
    ax.plot(t, parallax, linewidth=0.8)
    failed = np.isnan(parallax)
    if failed.any():
        ax.plot(t[failed], np.zeros(int(failed.sum())), "|", color="red", markersize=10, label="failed to match")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("parallax")
    ax.set_xlabel("time (s)")

    # Selected-frame overlay: 1 px lines, never axvspan
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    n_frames = len(report["frames"]["frame_idx"])
    fig.suptitle(
        f"{Path(video['path']).name} — {n_frames} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "motion-parallax.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/viz.py tests/preproc/test_viz.py && isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py -m "feat(preproc): plot_motion_parallax PNG, failed pairs marked not dropped"
```

---

### Task 5: `plot_motion_matches` + empty-pairs test

**Files:**
- Modify: `collab_splats/preproc/viz.py`
- Test: `tests/preproc/test_viz.py`

- [ ] **Step 1: Write the failing tests**

Add `plot_motion_matches` to the import list; append:

```python
def test_plot_motion_matches_writes_png(tmp_path):
    _assert_png(plot_motion_matches(_fake_report(), tmp_path), "motion-matches.png")
    _assert_png(plot_motion_matches(_fake_report(), tmp_path, selected=[3, 7]), "motion-matches.png")


@pytest.mark.parametrize("plotter", (plot_motion_translation, plot_motion_parallax, plot_motion_matches))
def test_motion_plotters_tolerate_empty_pairs(plotter, tmp_path):
    """
    A video shorter than the stride has zero pairs; an empty plot is a true statement.
    """
    out = plotter(_fake_report(n_pairs=0), tmp_path)
    assert out.read_bytes()[:4] == b"\x89PNG"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: `ImportError: cannot import name 'plot_motion_matches'`

- [ ] **Step 3: Implement**

```python
def plot_motion_matches(report: dict, out_dir: str | Path, *, selected=None) -> Path:
    """
    ORB match count per pair vs seconds.
    """
    # Unpack columns
    pairs, video = report["pairs"], report["video"]
    fps = video["fps"]
    n_matches = np.asarray(pairs["n_matches"], dtype=float)

    # Time axis: a pair sits at its first member
    t = np.asarray(pairs["frame_idx_a"], dtype=float) / fps

    # Panel
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_WIDTH_IN, _PANEL_HEIGHT_IN))
    ax.plot(t, n_matches, linewidth=0.8)
    ax.set_ylabel("matches / pair")
    ax.set_xlabel("time (s)")

    # Selected-frame overlay: 1 px lines, never axvspan
    if selected is not None:
        t_sel = np.asarray(list(selected), dtype=float) / fps
        ax.vlines(t_sel, 0, 1, transform=ax.get_xaxis_transform(), color="green", alpha=0.15, linewidth=1)

    # Title, save and close
    n_frames = len(report["frames"]["frame_idx"])
    fig.suptitle(
        f"{Path(video['path']).name} — {n_frames} frames @ {fps:.2f} fps, stride {report['params']['motion_stride']}"
    )
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "motion-matches.png"
    fig.savefig(path, dpi=_PNG_DPI)
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/viz.py tests/preproc/test_viz.py && isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py -m "feat(preproc): plot_motion_matches PNG; motion plotters tolerate empty pairs"
```

---

### Task 6: Wire into `extract_frames`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:24-27` (imports) and `:181-183` (video-branch tail)
- Test: `tests/wrapper/test_reconstructor.py:137` (dispatch test) + new test at end of file

- [ ] **Step 1: Write the failing test**

In `test_extract_frames_dispatches_per_frame_selection` the stubbed report `{"available": True, "frames": {}}` has no columns, so the plotters must be stubbed there. Add `import types` at the top of the file if absent, and after the `get_video_info` monkeypatch line add:

```python
    # Plots are not the dispatch under test, and the stub report has no columns
    monkeypatch.setattr(
        R,
        "preproc_viz",
        types.SimpleNamespace(
            **{
                name: lambda *a, **k: None
                for name in (
                    "plot_photometric_blur",
                    "plot_photometric_exposure",
                    "plot_motion_translation",
                    "plot_motion_parallax",
                    "plot_motion_matches",
                )
            }
        ),
    )
```

Append at the end of `tests/wrapper/test_reconstructor.py` (check the top of the file: if `cv2` is not imported there, add `import cv2` to its import block):

```python
def test_extract_frames_writes_video_quality_pngs(tmp_path, monkeypatch):
    """
    The video branch renders all five report PNGs beside frames.zarr; the dir branch none.
    """
    from collab_splats.wrapper import reconstructor as R

    rng = np.random.default_rng(0)
    n = 20
    columns = ("blur", "laplacian", "exposure_mean", "exposure_median", "exposure_std", "clipped_low_frac", "clipped_high_frac")
    report = {
        "available": True,
        "video": {"path": "/data/clip.mp4", "fps": 10.0, "total_frames": n, "width": 64, "height": 48},
        "params": {"motion_stride": 2},
        "frames": {"frame_idx": list(range(n)), **{k: rng.uniform(0, 1, n).tolist() for k in columns}},
        "pairs": {
            "frame_idx_a": list(range(0, 20, 2)),
            "frame_idx_b": list(range(2, 22, 2)),
            "n_matches": [10] * 10,
            "translation_px": [1.0] * 9 + [None],
            "parallax": [0.5] * 9 + [None],
        },
    }
    two_frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 2
    two_records = [{"frame_idx": 3, "blur_score": 1.0}, {"frame_idx": 7, "blur_score": 1.0}]
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: report)
    monkeypatch.setattr(R, "get_video_info", lambda path: {"total_frames": n})
    monkeypatch.setattr(R, "sample_fps", lambda path, **kw: (two_frames, two_records))

    out = tmp_path / "scene"
    video = tmp_path / "v.mp4"
    video.touch()
    R.extract_frames(video, out / "frames.zarr", "fps", 1.0, None, 50)

    expected = {
        "photometric-blur.png",
        "photometric-exposure.png",
        "motion-translation.png",
        "motion-parallax.png",
        "motion-matches.png",
    }
    assert {p.name for p in out.glob("*.png")} == expected
    for name in expected:
        assert (out / name).read_bytes()[:4] == b"\x89PNG"

    # Image directory: no report, no plots
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    out2 = tmp_path / "scene2"
    R.extract_frames(img_dir, out2 / "frames.zarr", "fps", 1.0, None, 50)
    assert list(out2.glob("*.png")) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -q -p no:randomly -k extract_frames`
Expected: dispatch test fails with `AttributeError: ... has no attribute 'preproc_viz'`; the new test fails on the png-set assertion (empty set).

- [ ] **Step 3: Implement**

Import block in `collab_splats/wrapper/reconstructor.py` (lines 24–27 become):

```python
from collab_splats.preproc import get_video_info
from collab_splats.preproc import viz as preproc_viz
from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.qa import load_video_quality
```

Video-branch tail of `extract_frames` (replace the current `FrameStore.create(...)` + `return len(frame_arrays)` at `:181-183`):

```python
    FrameStore.create(frames_zarr, frame_arrays, records, provenance=prov)

    # Render the report beside frames.zarr with the kept frames marked. Written
    # whenever frames are, so the PNGs never go stale against the store.
    out_dir = frames_zarr.parent
    selected = [r["frame_idx"] for r in records]
    preproc_viz.plot_photometric_blur(report, out_dir, selected=selected)
    preproc_viz.plot_photometric_exposure(report, out_dir, selected=selected)
    preproc_viz.plot_motion_translation(report, out_dir, selected=selected)
    preproc_viz.plot_motion_parallax(report, out_dir, selected=selected)
    preproc_viz.plot_motion_matches(report, out_dir, selected=selected)
    logger.info("video quality: wrote 5 plots to %s", out_dir)

    return len(frame_arrays)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py tests/preproc/test_viz.py -q -p no:randomly`
Expected: all pass.

- [ ] **Step 5: Confirm the dashboard fast-bind path stayed matplotlib-free**

Run: `/opt/venv/reconstruction/bin/python -c "import sys; import collab_splats.remote; print('matplotlib' in sys.modules)"`
Expected: `False`

- [ ] **Step 6: Format and commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py && isort collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py -m "feat(wrapper): extract_frames writes the five video quality PNGs beside frames.zarr"
```

---

### Task 7: Docs + live render on the tutorial video

**Files:**
- Modify: `configs/README.md:51` (output layout block), `:286` (measure step)

- [ ] **Step 1: Output layout**

In the layout block, after the `frames.zarr` line at `configs/README.md:51`, add:

```
  video_quality_report.json    ← per-frame photometry + per-pair motion, report-only
  photometric-blur.png         ← the report rendered, one PNG per measurement family;
  photometric-exposure.png     ←   green ticks mark the frames kept in frames.zarr
  motion-translation.png
  motion-parallax.png
  motion-matches.png
```

- [ ] **Step 2: Measure step**

At the end of the **Measure** bullet (`configs/README.md:286`, after "so delete it to re-measure."), add:

```
   The report is also rendered to five PNGs beside it (`photometric-blur`,
   `photometric-exposure`, `motion-translation`, `motion-parallax`,
   `motion-matches`), written whenever frames are extracted, with the kept
   frames marked — raw columns only, no thresholds. Scenes processed before
   these existed get them on the next `preprocess(overwrite=True)`; there is
   no backfill.
```

- [ ] **Step 3: Live render**

Measures the tutorial video once if no report is cached (~100 s at 4 workers), then renders with every 24th frame marked:

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
from pathlib import Path
from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc import viz
video = next(Path("data/tutorial").glob("*.mp4"))
out = Path("/tmp/claude-0/-workspace-collab-splats/d88f5d6c-76ba-4c07-8ab2-3dca857eea13/scratchpad/vq")
report = load_video_quality(video, out / "video_quality_report.json", workers=4)
n = len(report["frames"]["frame_idx"])
sel = range(0, n, 24)
for fn in (viz.plot_photometric_blur, viz.plot_photometric_exposure, viz.plot_motion_translation, viz.plot_motion_parallax, viz.plot_motion_matches):
    print(fn(report, out, selected=sel))
EOF
```

Open the five PNGs (the Read tool renders images) and confirm: title present, seconds on x, green ticks visible across the full width, red `|` markers on `motion-parallax.png` if any pair failed.

- [ ] **Step 4: Graph + commit**

```bash
graphify update .
git commit --only configs/README.md -m "docs(configs): list the five video quality PNGs in the processed-scene contract"
```

---

## Self-review

- **Spec coverage:** five plotters (T1–T5) with fixed signature/return; raw columns only; seconds on x; `sharex`; `figsize`/`dpi=90`; inline suptitle; `vlines` overlay (T1–T5); `None → nan` via `np.asarray(dtype=float)` (T3–T5); parallax failed markers (T4); empty pairs tolerated (T5); wiring video-branch only, dir branch none (T6); fast-bind path check (T6 step 5); README contract + no-backfill wording (T7). Unavailable report: natural `KeyError`, nothing built (spec, revised).
- **Placeholder scan:** none. Every code step is complete.
- **Type consistency:** `(report: dict, out_dir: str | Path, *, selected=None) -> Path` on all five; file names identical in T1–T5 code, T6 test set, T7 README.
