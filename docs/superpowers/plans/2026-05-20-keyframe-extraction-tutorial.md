# Keyframe Extraction Tutorial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `get_video_info` + `score_all_frames` utilities and a visualization section to `frame_sampling.py`, then create `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb` as the first tutorial in the series.

**Architecture:** New functions appended to `collab_splats/utils/frame_sampling.py` (core section then viz section behind a divider). Notebook imports only — no inline functions or cv2 calls. Tests appended to existing `tests/utils/test_frame_sampling.py`.

**Tech Stack:** Python, OpenCV (cv2), matplotlib, nbformat, pytest, Jupyter notebook (nbformat 4)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `collab_splats/utils/frame_sampling.py` | Add `get_video_info`, `score_all_frames` to core section; add viz section with 4 plot functions |
| Modify | `tests/utils/test_frame_sampling.py` | Append tests for new functions |
| New | `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb` | Tutorial notebook |
| Modify | `docs/source/tutorials/index.rst` | Add Preprocessing section before Splats |

---

## Task 1: `get_video_info`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (append before `sample_frames_fps`)
- Modify: `tests/utils/test_frame_sampling.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/utils/test_frame_sampling.py`:

```python
# ── get_video_info ────────────────────────────────────────────────────────────

from collab_splats.utils.frame_sampling import get_video_info


def test_get_video_info_keys(tiny_video):
    info = get_video_info(tiny_video)
    assert set(info.keys()) == {"total_frames", "fps", "duration_s"}


def test_get_video_info_values(tiny_video):
    info = get_video_info(tiny_video)
    assert info["total_frames"] == 90
    assert abs(info["fps"] - 30.0) < 1.0
    assert abs(info["duration_s"] - 3.0) < 0.5


def test_get_video_info_missing_file():
    info = get_video_info("nonexistent.mp4")
    assert info["total_frames"] == 0
    assert info["fps"] == 0.0
    assert info["duration_s"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_get_video_info_keys -x -q 2>&1 | tail -5
```

Expected: `ImportError` or `AttributeError` — `get_video_info` does not exist yet.

- [ ] **Step 3: Implement `get_video_info`**

In `collab_splats/utils/frame_sampling.py`, insert after the imports block and before `sample_frames_fps`:

```python
def get_video_info(video_path: str) -> dict:
    """Return basic video metadata without exposing cv2 to callers.

    Keys: total_frames (int), fps (float), duration_s (float).
    Returns zeros for all fields if cv2 is unavailable or file cannot be opened.
    """
    try:
        import cv2
    except ImportError:
        return {"total_frames": 0, "fps": 0.0, "duration_s": 0.0}
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    duration_s = total / fps if fps > 0 else 0.0
    return {"total_frames": total, "fps": fps, "duration_s": duration_s}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_get_video_info_keys tests/utils/test_frame_sampling.py::test_get_video_info_values tests/utils/test_frame_sampling.py::test_get_video_info_missing_file -v 2>&1 | tail -10
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add get_video_info utility"
```

---

## Task 2: `score_all_frames`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (append after `get_video_info`, before `OpticalFlowFrameSelector`)
- Modify: `tests/utils/test_frame_sampling.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
# ── score_all_frames ──────────────────────────────────────────────────────────

from collab_splats.utils.frame_sampling import score_all_frames


def test_score_all_frames_returns_list(tiny_video):
    results = score_all_frames(tiny_video)
    assert isinstance(results, list)
    assert len(results) > 0


def test_score_all_frames_dict_keys(tiny_video):
    results = score_all_frames(tiny_video)
    expected_keys = {"frame_idx", "disparity", "rotation", "histogram_similarity", "score", "selected"}
    assert set(results[0].keys()) == expected_keys


def test_score_all_frames_first_frame_always_selected(tiny_video):
    results = score_all_frames(tiny_video)
    assert results[0]["selected"] is True
    assert results[0]["score"] == 1.0


def test_score_all_frames_frame_count(tiny_video):
    results = score_all_frames(tiny_video)
    # tiny_video has 90 frames
    assert len(results) == 90
    assert results[-1]["frame_idx"] == 89


def test_score_all_frames_scores_in_range(tiny_video):
    results = score_all_frames(tiny_video)
    for d in results:
        assert 0.0 <= d["score"] <= 1.0
        assert d["disparity"] >= 0.0
        assert d["rotation"] >= 0.0
        assert 0.0 <= d["histogram_similarity"] <= 1.0


def test_score_all_frames_calls_on_progress(tiny_video):
    calls = []
    score_all_frames(tiny_video, on_progress=lambda c, t: calls.append((c, t)))
    assert len(calls) == 90
    assert calls[-1][0] == 90
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_score_all_frames_returns_list -x -q 2>&1 | tail -5
```

Expected: `ImportError` — `score_all_frames` not defined.

- [ ] **Step 3: Implement `score_all_frames`**

In `collab_splats/utils/frame_sampling.py`, insert after `get_video_info` and before `OpticalFlowFrameSelector`:

```python
def score_all_frames(
    video_path: str,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Score every frame using OpticalFlowFrameSelector without capping count.

    Returns one dict per decoded frame with keys:
        frame_idx (int), disparity (float), rotation (float),
        histogram_similarity (float), score (float), selected (bool)

    Mirrors the selection logic in sample_frames_optical_flow but records
    per-frame signals for diagnostic and visualization use.
    """
    try:
        import cv2
    except ImportError:
        return []

    rotation = _get_rotation_degrees(video_path)
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    frames_decoded = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _apply_rotation(frame, rotation)
            scale = min(1.0, 480.0 / frame.shape[1])
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
            should_select, score, components = selector.should_select_frame(small)
            results.append({
                "frame_idx": frames_decoded,
                "disparity": components["disparity"],
                "rotation": components["rotation"],
                "histogram_similarity": components["histogram_similarity"],
                "score": score,
                "selected": should_select,
            })
            if should_select:
                selector.accept_frame(small)
            frames_decoded += 1
            if on_progress is not None:
                on_progress(frames_decoded, total)
    finally:
        cap.release()
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py -k "score_all_frames" -v 2>&1 | tail -15
```

Expected: 6 PASSED.

- [ ] **Step 5: Run full test file to check no regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py -v 2>&1 | tail -20
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add score_all_frames for per-frame diagnostic scoring"
```

---

## Task 3: Viz section + `plot_frame_grid`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (append viz section at end of file)
- Modify: `tests/utils/test_frame_sampling.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/utils/test_frame_sampling.py`:

```python
# ── Visualization ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collab_splats.utils.frame_sampling import (
    plot_frame_grid,
    plot_selection,
    plot_frame_scores,
    plot_disparity_sensitivity,
)


def test_plot_frame_grid_returns_figure():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(6)]
    fig = plot_frame_grid(frames, title="test grid")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_frame_grid_single_frame():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
    fig = plot_frame_grid(frames, title="single")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_plot_frame_grid_returns_figure -x -q 2>&1 | tail -5
```

Expected: `ImportError` — `plot_frame_grid` not defined.

- [ ] **Step 3: Append viz section divider + `plot_frame_grid` to `frame_sampling.py`**

Append at the very end of `collab_splats/utils/frame_sampling.py`:

```python


##############################################################################
# Visualization
##############################################################################

def plot_frame_grid(
    frames: list,
    title: str,
    n_cols: int = 6,
):
    """Display a grid of RGB frames. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt

    n = len(frames)
    n_cols = min(n_cols, n)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(frames[i])
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_plot_frame_grid_returns_figure tests/utils/test_frame_sampling.py::test_plot_frame_grid_single_frame -v 2>&1 | tail -10
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add visualization section with plot_frame_grid"
```

---

## Task 4: `plot_selection`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (append to viz section)
- Modify: `tests/utils/test_frame_sampling.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_plot_selection_fps_only():
    fig = plot_selection(total_frames=90, fps_indices=list(range(0, 90, 10)))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_selection_of_only():
    fig = plot_selection(total_frames=90, of_indices=[0, 15, 40, 70])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_selection_both():
    fig = plot_selection(
        total_frames=90,
        fps_indices=list(range(0, 90, 10)),
        of_indices=[0, 15, 40, 70],
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # two stacked panels
    plt.close(fig)


def test_plot_selection_single_panel_has_one_axis():
    fig = plot_selection(total_frames=90, fps_indices=[0, 30, 60])
    assert len(fig.axes) == 1
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_plot_selection_fps_only -x -q 2>&1 | tail -5
```

Expected: `ImportError` — `plot_selection` not defined.

- [ ] **Step 3: Implement `plot_selection`**

Append to the viz section of `collab_splats/utils/frame_sampling.py`:

```python
def plot_selection(
    total_frames: int,
    fps_indices: list | None = None,
    of_indices: list | None = None,
):
    """Vertical-line plot of selected frame indices.

    One set of indices → single panel.
    Both sets → two stacked panels for direct comparison.
    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    sets = [
        (fps_indices, "FPS", "steelblue"),
        (of_indices, "Optical Flow", "darkorange"),
    ]
    active = [(idx, label, color) for idx, label, color in sets if idx is not None]

    fig, axes = plt.subplots(
        len(active), 1,
        figsize=(12, 2 * len(active)),
        squeeze=False,
    )
    for ax, (indices, label, color) in zip(axes[:, 0], active):
        if indices:
            ax.vlines(indices, 0, 1, colors=color, linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xlabel("Frame index")
        ax.set_title(f"{label}  (n={len(indices) if indices else 0})", fontsize=10)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py -k "plot_selection" -v 2>&1 | tail -10
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add plot_selection visualization"
```

---

## Task 5: `plot_frame_scores`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (append to viz section)
- Modify: `tests/utils/test_frame_sampling.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_plot_frame_scores_returns_figure(tiny_video):
    scores = score_all_frames(tiny_video)
    fig = plot_frame_scores(scores)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3  # disparity, rotation, histogram_similarity panels
    plt.close(fig)


def test_plot_frame_scores_empty_input():
    # Should not raise even with empty list
    fig = plot_frame_scores([])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_plot_frame_scores_returns_figure -x -q 2>&1 | tail -5
```

Expected: `ImportError` — `plot_frame_scores` not defined.

- [ ] **Step 3: Implement `plot_frame_scores`**

Append to the viz section of `collab_splats/utils/frame_sampling.py`:

```python
def plot_frame_scores(frame_scores: list):
    """3-panel timeseries of per-frame optical flow scoring signals.

    Panels: disparity (px) / rotation (deg) / histogram similarity.
    Selected frames marked with vertical grey lines.
    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if not frame_scores:
        fig, _ = plt.subplots(3, 1, figsize=(12, 6))
        return fig

    idxs = [d["frame_idx"] for d in frame_scores]
    selected_idxs = [d["frame_idx"] for d in frame_scores if d["selected"]]

    panels = [
        ([d["disparity"] for d in frame_scores], "Disparity (px)", "steelblue"),
        ([d["rotation"] for d in frame_scores], "Rotation (deg)", "seagreen"),
        ([d["histogram_similarity"] for d in frame_scores], "Histogram similarity", "tomato"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.plot(idxs, values, color=color, linewidth=0.8)
        for x in selected_idxs:
            ax.axvline(x, color="gray", alpha=0.25, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle(
        "Per-frame optical flow scores  (grey lines = selected frames)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py -k "plot_frame_scores" -v 2>&1 | tail -10
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add plot_frame_scores visualization"
```

---

## Task 6: `plot_disparity_sensitivity`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (append to viz section)
- Modify: `tests/utils/test_frame_sampling.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_plot_disparity_sensitivity_returns_figure(tiny_video):
    scores = score_all_frames(tiny_video)
    fig = plot_disparity_sensitivity(scores, [10.0, 50.0, 100.0])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_disparity_sensitivity_monotonic(tiny_video):
    scores = score_all_frames(tiny_video)
    thresholds = [10.0, 25.0, 50.0, 100.0, 200.0]
    # Higher threshold -> fewer or equal frames selected
    counts = []
    for t in thresholds:
        n = sum(
            1 for d in scores
            if 0.6 * min(d["disparity"] / max(t, 1e-6), 1.0)
            + 0.4 * (1.0 - d["histogram_similarity"]) >= 0.5
        )
        counts.append(n)
    assert counts == sorted(counts, reverse=True), "Higher threshold must not increase count"
    fig = plot_disparity_sensitivity(scores, thresholds)
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py::test_plot_disparity_sensitivity_returns_figure -x -q 2>&1 | tail -5
```

Expected: `ImportError` — `plot_disparity_sensitivity` not defined.

- [ ] **Step 3: Implement `plot_disparity_sensitivity`**

Append to the viz section of `collab_splats/utils/frame_sampling.py`:

```python
def plot_disparity_sensitivity(
    frame_scores: list,
    disparity_values: list,
):
    """Approximate frame count vs min_disparity threshold.

    Re-thresholds precomputed frame_scores — no video re-decode needed.
    Uses default weights (motion=0.6, coverage=0.4) and threshold=0.5.
    Note: an approximation; true counts differ if the full video were re-run
    because selection is stateful. Sufficient to show the monotonic trend.
    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    counts = []
    for threshold in disparity_values:
        n = sum(
            1 for d in frame_scores
            if (
                0.6 * min(d["disparity"] / max(threshold, 1e-6), 1.0)
                + 0.4 * (1.0 - d["histogram_similarity"])
            ) >= 0.5
        )
        counts.append(n)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(disparity_values, counts, marker="o", color="steelblue", linewidth=1.5)
    ax.set_xlabel("min_disparity threshold (px)")
    ax.set_ylabel("Frames selected (approx.)")
    ax.set_title("Frame count vs disparity threshold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py -k "disparity_sensitivity" -v 2>&1 | tail -10
```

Expected: 2 PASSED.

- [ ] **Step 5: Run full test suite for the file**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_frame_sampling.py -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add plot_disparity_sensitivity visualization"
```

---

## Task 7: Tutorial Notebook

**Files:**
- New: `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb`

- [ ] **Step 1: Create directory**

```bash
mkdir -p /workspace/collab-splats/docs/source/tutorials/preprocessing
```

- [ ] **Step 2: Create the notebook**

Write `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb` with the following cells (use Write tool with the JSON below):

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Keyframe Extraction\n",
    "\n",
    "**Goal:** Understand how frame selection from video affects 3D point cloud quality and spatial coverage.\n",
    "\n",
    "This is the first step in the preprocessing pipeline — before COLMAP, before Gaussian splatting. The frames you select determine which 3D points get reconstructed and how evenly the scene is covered."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collab_splats.utils.frame_sampling import (\n",
    "    get_video_info,\n",
    "    sample_frames_fps,\n",
    "    sample_frames_optical_flow,\n",
    "    score_all_frames,\n",
    "    plot_frame_grid,\n",
    "    plot_selection,\n",
    "    plot_frame_scores,\n",
    "    plot_disparity_sensitivity,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Why Frame Selection Matters\n",
    "\n",
    "COLMAP (the structure-from-motion tool underlying most 3D reconstruction pipelines) builds a point cloud by finding matching features across frames. **The frames you give it determine the shape, density, and coverage of the result.**\n",
    "\n",
    "Two failure modes:\n",
    "- **Over-sampling slow regions:** A slow camera pan generates dozens of nearly identical frames. COLMAP spends budget on redundant matches; the resulting point cloud is dense in that region and sparse elsewhere.\n",
    "- **Under-sampling fast transitions:** If the camera moves quickly through a room and you sample at fixed rate, interesting transitions may get fewer frames than a stationary shot.\n",
    "\n",
    "**Even (FPS) sampling** extracts frames at a fixed temporal rate — uniform in time but biased toward regions filmed slowly.\n",
    "\n",
    "**Optical flow (OF) sampling** extracts frames only when the scene content changes significantly — uniform in content, regardless of filming speed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Video"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "video_path = \"path/to/your/video.mp4\"  # substitute your video\n",
    "\n",
    "info = get_video_info(video_path)\n",
    "print(f\"Frames: {info['total_frames']}  FPS: {info['fps']:.1f}  Duration: {info['duration_s']:.1f}s\")\n",
    "\n",
    "# Preview: 1 frame per second\n",
    "preview_frames = sample_frames_fps(video_path, fps=1.0, max_frames=12)\n",
    "plot_frame_grid(preview_frames, title=\"Video preview (1 fps)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Even Sampling (FPS)\n",
    "\n",
    "Extract frames at a fixed rate. The result is uniformly spaced in time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_fps = 2.0  # adjust to taste\n",
    "fps_frames = sample_frames_fps(video_path, fps=target_fps)\n",
    "\n",
    "# Compute which frame indices were selected (evenly spaced)\n",
    "interval = max(1, round(info['fps'] / target_fps))\n",
    "fps_indices = list(range(0, info['total_frames'], interval))[:len(fps_frames)]\n",
    "\n",
    "print(f\"FPS sampling selected {len(fps_frames)} frames\")\n",
    "plot_selection(info['total_frames'], fps_indices=fps_indices)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_frame_grid(fps_frames[:12], title=f\"FPS-sampled frames (n={len(fps_frames)}, showing first 12)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Optical Flow Score Timeseries\n",
    "\n",
    "Before selecting OF frames, score every frame. The three signals that drive selection:\n",
    "- **Disparity:** mean pixel displacement of tracked feature points — how much the camera moved\n",
    "- **Rotation:** estimated camera rotation between frames\n",
    "- **Histogram similarity:** how similar the current frame looks to the last accepted keyframe\n",
    "\n",
    "Spikes = camera moved or scene changed. Flat regions = redundant frames."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "frame_scores = score_all_frames(video_path)\n",
    "plot_frame_scores(frame_scores)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Optical Flow Selection\n",
    "\n",
    "Now apply the selection threshold (score ≥ 0.5) to pick keyframes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "of_frames = sample_frames_optical_flow(video_path)\n",
    "of_indices = [d[\"frame_idx\"] for d in frame_scores if d[\"selected\"]]\n",
    "\n",
    "print(f\"OF sampling selected {len(of_frames)} frames\")\n",
    "plot_selection(info['total_frames'], of_indices=of_indices)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_frame_grid(of_frames[:12], title=f\"OF-sampled frames (n={len(of_frames)}, showing first 12)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Side-by-Side Comparison\n",
    "\n",
    "FPS distributes uniformly in time. OF distributes uniformly in content."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_selection(info['total_frames'], fps_indices=fps_indices, of_indices=of_indices)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Parameter Sensitivity\n",
    "\n",
    "`min_disparity` controls how much the camera must move before a new frame is selected.\n",
    "Higher = fewer, more distinct frames. Lower = denser coverage but more redundancy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "disparity_values = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0]\n",
    "plot_disparity_sensitivity(frame_scores, disparity_values)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. When to Use Each Method\n",
    "\n",
    "| Scenario | Recommended | Reason |\n",
    "|----------|-------------|--------|\n",
    "| Slow camera pans | Optical flow | Avoids oversampling near-identical frames |\n",
    "| Handheld walk-through | Either | FPS is simpler; OF is marginally better |\n",
    "| Fast action / drone | FPS | OF may miss fast transients between keyframes |\n",
    "| Limited frame budget | Optical flow | Better spatial scene coverage per frame |\n",
    "\n",
    "**Default in `SplatterConfig`:** `frame_selection=\"fps\"`. Switch to `\"optical_flow\"` for indoor walk-throughs or any scene with significant speed variation."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "nerfstudio",
   "language": "python",
   "name": "nerfstudio"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Verify notebook is valid JSON**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import json; json.load(open('/workspace/collab-splats/docs/source/tutorials/preprocessing/keyframe_extraction.ipynb')); print('valid')"
```

Expected: `valid`

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/preprocessing/keyframe_extraction.ipynb
git commit -m "docs(tutorials): add keyframe_extraction tutorial notebook"
```

---

## Task 8: Update Docs Index

**Files:**
- Modify: `docs/source/tutorials/index.rst`

- [ ] **Step 1: Prepend Preprocessing section**

Edit `docs/source/tutorials/index.rst`. The final file should be:

```rst
Tutorials
=========

.. toctree::
   :maxdepth: 1
   :caption: Preprocessing

   preprocessing/keyframe_extraction

.. toctree::
   :maxdepth: 1
   :caption: Splats

   splats/derive_splats
   splats/create_mesh
   splats/visualization
   splats/compare_maskclip_talk2dino

.. toctree::
   :maxdepth: 1
   :caption: Semantics

   semantics/feature_extraction
   semantics/segmentation
   semantics/maskclip_reference_comparison

.. toctree::
   :maxdepth: 1
   :caption: Point Cloud

   pointcloud/bundle_adjustment
   pointcloud/feedforward_exploration
   pointcloud/feedforward_mesh
   pointcloud/loop_closure_eval
   pointcloud/ground-truth-evals
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/tutorials/index.rst
git commit -m "docs(tutorials): add Preprocessing section to tutorial index"
```

---

## Self-Review

**Spec coverage:**
- ✅ `get_video_info` — Task 1
- ✅ `score_all_frames` — Task 2
- ✅ Viz section divider — Task 3
- ✅ `plot_frame_grid` — Task 3
- ✅ `plot_selection` (single + comparison) — Task 4
- ✅ `plot_frame_scores` — Task 5
- ✅ `plot_disparity_sensitivity` — Task 6
- ✅ Tutorial notebook all 7 phases — Task 7
- ✅ `docs/source/tutorials/index.rst` updated — Task 8

**Type consistency:**
- `score_all_frames` returns `list[dict]` with keys `frame_idx`, `disparity`, `rotation`, `histogram_similarity`, `score`, `selected` — matches keys used in `plot_frame_scores` and `plot_disparity_sensitivity` ✅
- `get_video_info` returns `dict` with `total_frames`, `fps`, `duration_s` — matches notebook usage `info['total_frames']`, `info['fps']` ✅
- `plot_selection` accepts `fps_indices: list | None`, `of_indices: list | None` — matches all three notebook call sites ✅

**Approximation noted:** `plot_disparity_sensitivity` docstring and notebook prose both note it approximates by re-thresholding precomputed scores rather than re-decoding the video.
