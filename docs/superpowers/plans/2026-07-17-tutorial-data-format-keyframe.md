# Tutorial Data-Format Alignment + Keyframe Tutorial Rework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `tutorial_config.py` with the canonical `BASE/<date>/<stem>/` storage layout and rework the keyframe-extraction tutorial to showcase the blur/exposure quality gate with real example frames.

**Architecture:** Promote the private quality gate to a public `check_frame_quality` returning `(ok, metrics)` with a `reject_reason`; thread metrics through `_iter_scored_frames` so `score_frames` records carry exposure data; add `plot_quality_examples` to `preproc/viz.py`; rewrite `tutorial_config.py` (FRAMES, TUTORIAL_CACHE, `video_ref` inference); edit the notebook to write only to scratch and add a quality-gate section.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, matplotlib, ffmpeg-backed `collab_splats.preproc`, Jupyter notebook (NotebookEdit tool + `jupyter nbconvert --execute`).

**Spec:** `docs/superpowers/specs/2026-07-16-tutorial-data-format-keyframe-design.md`

**Conventions:** flat test functions; imports at top; block-level comments; `black . && isort .` before final commit. Run all tests with `/opt/venv/reconstruction/bin/python -m pytest`.

---

### Task 1: Public `check_frame_quality` with reject reasons

**Files:**
- Modify: `collab_splats/preproc/sampling.py:171-187` (replace `_check_frame_quality`), `:437`, `:480` (call sites)
- Modify: `collab_splats/preproc/__init__.py`
- Test: `tests/preproc/test_sampling.py:76-119` (Quality gate section), `:326-338` (`test_public_api_surface`)

- [ ] **Step 1: Rewrite the quality-gate tests to the new API**

In `tests/preproc/test_sampling.py`, replace the import at line 80 and the five gate tests (lines 95-119) with:

```python
from collab_splats.preproc.sampling import check_frame_quality, compute_blur_score
```

```python
def test_check_frame_quality_accepts_sharp_frame():
    ok, metrics = check_frame_quality(_sharp_gray())
    assert ok is True
    assert metrics["reject_reason"] is None


def test_check_frame_quality_rejects_blurred_frame():
    sharp = _sharp_gray()
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    # Threshold between the two measured scores makes the test threshold-robust
    threshold = (compute_blur_score(sharp) + compute_blur_score(blurred)) / 2
    ok, metrics = check_frame_quality(blurred, blur_threshold=threshold)
    assert ok is False and metrics["reject_reason"] == "blur"
    ok, _ = check_frame_quality(sharp, blur_threshold=threshold)
    assert ok is True


def test_check_frame_quality_rejects_bad_exposure():
    # Near-black and near-white frames fail regardless of sharpness
    dark = np.zeros((240, 320), dtype=np.uint8)
    bright = np.full((240, 320), 255, dtype=np.uint8)
    for gray in (dark, bright):
        ok, metrics = check_frame_quality(gray, blur_threshold=0.0)
        assert ok is False and metrics["reject_reason"] == "exposure"


def test_check_frame_quality_metrics_fields():
    _, metrics = check_frame_quality(_sharp_gray())
    assert set(metrics) == {"blur_score", "exposure_mean", "exposure_std", "reject_reason"}


def test_check_frame_quality_uses_precomputed_blur_score():
    # Passing blur_score short-circuits the Laplacian recompute
    ok, metrics = check_frame_quality(_sharp_gray(), blur_threshold=100.0, blur_score=50.0)
    assert ok is False and metrics["blur_score"] == 50.0
```

Also update `test_public_api_surface` (line ~330): the expected `__all__` set gains `"check_frame_quality"` (8 names total).

- [ ] **Step 2: Run the gate tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k check_frame_quality -v`
Expected: FAIL — `ImportError: cannot import name 'check_frame_quality'`

- [ ] **Step 3: Replace `_check_frame_quality` in `sampling.py`**

Replace lines 171-187 (`_check_frame_quality`) with:

```python
def check_frame_quality(
    gray: np.ndarray,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    blur_score: float | None = None,
) -> tuple[bool, dict]:
    """Quality gate: is the frame sharp enough and reasonably exposed?

    Returns (ok, metrics) where metrics holds blur_score, exposure_mean,
    exposure_std, and reject_reason (None | "blur" | "exposure").
    blur_score: pass a precomputed value to skip the Laplacian recompute.
    """
    if blur_score is None:
        blur_score = compute_blur_score(gray)
    mean, std = float(gray.mean()), float(gray.std())
    lo, hi = _EXPOSURE_MEAN_RANGE
    # Blur checked first — the first failing check names the reason
    reason = None
    if blur_score < blur_threshold:
        reason = "blur"
    elif not (lo <= mean <= hi) or std < _EXPOSURE_MIN_STD:
        reason = "exposure"
    metrics = {
        "blur_score": blur_score,
        "exposure_mean": mean,
        "exposure_std": std,
        "reject_reason": reason,
    }
    return reason is None, metrics
```

Update the two call sites (boolean semantics only — metrics threading is Task 2):

Line 437 (uniform sampler):
```python
            usable, _ = check_frame_quality(gray, blur_threshold, blur_score=blur)
```

Line 480 (`_iter_scored_frames`):
```python
            ok, _ = check_frame_quality(gray, blur_threshold, blur_score=blur)
            if not ok:
```
(keep the existing `yield idx, frame, False, blur, 0.0, {}` / `continue` body unchanged in this task)

- [ ] **Step 4: Export from `collab_splats/preproc/__init__.py`**

Add `check_frame_quality` to both the import block and `__all__`:

```python
from collab_splats.preproc.sampling import (
    check_frame_quality,
    compute_blur_score,
    extract_frame,
    extract_frames,
    get_video_info,
    load_frames,
    sample_frames,
    score_frames,
)

__all__ = [
    "sample_frames",
    "score_frames",
    "get_video_info",
    "load_frames",
    "extract_frame",
    "extract_frames",
    "compute_blur_score",
    "check_frame_quality",
]
```

- [ ] **Step 5: Run the full preproc sampling tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: all PASS (gate tests + public-API test + unchanged sampler tests)

- [ ] **Step 6: Commit**

```bash
git add collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): public check_frame_quality with per-frame metrics and reject_reason"
```

---

### Task 2: `score_frames` records carry exposure metrics + reject reason

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (`_iter_scored_frames` ~L457-489, `_sample_optical_flow` ~L510-524, `score_frames` ~L536-565)
- Test: `tests/preproc/test_sampling.py` (`test_score_frames_one_record_per_frame` ~L279, new reject-reason test)

- [ ] **Step 1: Update/add failing tests**

In `test_score_frames_one_record_per_frame`, extend the expected key set:

```python
    assert set(records[0]) == {
        "frame_idx",
        "blur_score",
        "exposure_mean",
        "exposure_std",
        "reject_reason",
        "score",
        "selected",
        "disparity",
        "rotation",
        "histogram_similarity",
    }
```

Add after it:

```python
def test_score_frames_reject_reason_blur(tiny_video):
    # Threshold above any real Laplacian variance → every frame blur-rejected
    records = score_frames(tiny_video, blur_threshold=1e12)
    assert len(records) == 60
    assert all(r["selected"] is False and r["reject_reason"] == "blur" for r in records)


def test_score_frames_accepted_have_no_reject_reason(tiny_video):
    records = score_frames(tiny_video, blur_threshold=0.0)
    assert all(r["reject_reason"] is None for r in records)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k score_frames -v`
Expected: FAIL — KeyError/assert on missing `exposure_mean` / `reject_reason`

- [ ] **Step 3: Thread metrics through `_iter_scored_frames`**

Change `_iter_scored_frames` to yield the quality-metrics dict instead of the bare blur float. New signature/docstring/body (replaces ~L457-489):

```python
def _iter_scored_frames(
    video_path: str,
    selector: OpticalFlowFrameSelector,
    *,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
    desc: str,
) -> Iterator[tuple[int, np.ndarray, bool, dict, float, dict]]:
    """Yield (frame_idx, frame_bgr, selected, quality, score, components) per frame.

    quality is check_frame_quality's metrics dict. Single scoring loop shared by
    _sample_optical_flow and score_frames. Gate-rejected frames yield
    selected=False with score 0.0 and never reach the selector. Selected frames
    (score >= _SELECT_THRESHOLD) become the selector's new reference keyframe.
    """
    info = get_video_info(str(video_path))
    report, close = _progress_reporter(info["total_frames"], desc, on_progress)
    try:
        for idx, frame in enumerate(_iter_frames(video_path)):
            report(idx + 1)
            gray = _analysis_gray(frame)
            # Quality gate first: unusable frames never reach the selector
            ok, quality = check_frame_quality(gray, blur_threshold)
            if not ok:
                yield idx, frame, False, quality, 0.0, {}
                continue
            score, components = selector.score_frame(gray)
            selected = score >= _SELECT_THRESHOLD
            if selected:
                selector.accept_frame(gray)
            yield idx, frame, selected, quality, score, components
    finally:
        close()
```

(The `blur = compute_blur_score(gray)` line is dropped — `check_frame_quality` computes it.)

- [ ] **Step 4: Update the two consumers**

`_sample_optical_flow` loop (record keys unchanged — sampler records keep their existing schema):

```python
    for idx, frame, selected, quality, score, comp in _iter_scored_frames(
        video_path,
        selector,
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="Optical flow selection",
    ):
        if not selected:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # frame_idx is the SOURCE video index (fixes old positional-index bug)
        records.append({"frame_idx": idx, "blur_score": quality["blur_score"], "score": score, "selected": True, **comp})
        if max_frames is not None and len(frames) >= max_frames:
            break
```

`score_frames` loop body + docstring:

```python
    """Score every frame without keeping pixel data — analysis/viz workflow.

    Returns one record per frame: frame_idx, blur_score, exposure_mean,
    exposure_std, reject_reason, disparity, rotation, histogram_similarity,
    score, selected.
    """
```

```python
    for idx, _frame, selected, quality, score, comp in _iter_scored_frames(
        video_path,
        selector,
        blur_threshold=blur_threshold,
        on_progress=on_progress,
        desc="Scoring frames",
    ):
        records.append(
            {
                "frame_idx": idx,
                "blur_score": quality["blur_score"],
                "exposure_mean": quality["exposure_mean"],
                "exposure_std": quality["exposure_std"],
                "reject_reason": quality["reject_reason"],
                "score": score,
                "selected": selected,
                "disparity": comp.get("disparity", 0.0),
                "rotation": comp.get("rotation", 0.0),
                "histogram_similarity": comp.get("histogram_similarity", 1.0),
            }
        )
```

- [ ] **Step 5: Run the sampling tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: all PASS (including unchanged `test_optical_flow_records_have_source_indices` — sampler record schema untouched)

- [ ] **Step 6: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): score_frames records carry exposure metrics and reject_reason"
```

---

### Task 3: `plot_quality_examples` in `preproc/viz.py`

**Files:**
- Modify: `collab_splats/preproc/viz.py`
- Test: `tests/preproc/test_viz.py`

- [ ] **Step 1: Write failing tests**

In `tests/preproc/test_viz.py`: extend `_fake_records` so each record also carries the new fields, and add the new tests.

Replace `_fake_records` with:

```python
def _fake_records(n=30):
    """Score records shaped like score_frames output."""
    rng = np.random.default_rng(0)
    reasons = [None, "blur", "exposure"]
    return [
        {
            "frame_idx": i,
            "blur_score": 200.0,
            "exposure_mean": 120.0,
            "exposure_std": 40.0,
            "reject_reason": reasons[i % 3],
            "disparity": float(rng.random() * 80),
            "rotation": float(rng.random() * 3),
            "histogram_similarity": float(rng.random()),
            "score": float(rng.random()),
            "selected": i % 5 == 0,
        }
        for i in range(n)
    ]
```

Add to the viz import block: `plot_quality_examples`. Add tests:

```python
def test_plot_quality_examples_three_rows(monkeypatch):
    # Stub decode: viz must only ask for the frames it displays
    fake = lambda _path, idxs: [np.zeros((24, 32, 3), dtype=np.uint8) for _ in idxs]
    monkeypatch.setattr("collab_splats.preproc.viz.load_frames", fake)
    plot_quality_examples("unused.mp4", _fake_records(), n_examples=3)
    # One row per non-empty category (accepted / blur / exposure), n_examples cols
    assert len(plt.gcf().axes) == 9


def test_plot_quality_examples_skips_empty_categories(monkeypatch):
    fake = lambda _path, idxs: [np.zeros((24, 32, 3), dtype=np.uint8) for _ in idxs]
    monkeypatch.setattr("collab_splats.preproc.viz.load_frames", fake)
    records = [d for d in _fake_records() if d["reject_reason"] != "exposure"]
    plot_quality_examples("unused.mp4", records, n_examples=3)
    assert len(plt.gcf().axes) == 6  # accepted + blur rows only


def test_plot_quality_examples_empty_input():
    plot_quality_examples("unused.mp4", [])  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -v`
Expected: FAIL — `ImportError: cannot import name 'plot_quality_examples'`

- [ ] **Step 3: Implement `plot_quality_examples`**

In `collab_splats/preproc/viz.py`, change the sampling import (line 13) to:

```python
from collab_splats.preproc.sampling import _combine_scores, load_frames
```

Append at end of file:

```python
def plot_quality_examples(video_path: str, frame_scores: list, n_examples: int = 4) -> None:
    """Example frames per quality-gate outcome: accepted / blur- / exposure-rejected.

    Takes score_frames() records; decodes only the displayed frames. Empty
    categories are dropped from the grid (counts still shown in the title).
    """
    categories = [
        ("Accepted", [d for d in frame_scores if d.get("reject_reason") is None]),
        ("Rejected: blur", [d for d in frame_scores if d.get("reject_reason") == "blur"]),
        ("Rejected: exposure", [d for d in frame_scores if d.get("reject_reason") == "exposure"]),
    ]
    counts = " · ".join(f"{label.lower()}: {len(recs)}" for label, recs in categories)
    # Spread picks evenly across each non-empty category rather than taking the first n
    rows = []
    for label, recs in categories:
        if recs:
            step = max(1, len(recs) // n_examples)
            rows.append((label, recs[::step][:n_examples]))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No records ({counts})", ha="center", va="center")
        plt.show()
        return
    fig, axes = plt.subplots(len(rows), n_examples, figsize=(n_examples * 2.6, len(rows) * 2.4), squeeze=False)
    for row_axes, (label, recs) in zip(axes, rows):
        # Records are in stream order, so load_frames returns aligned frames
        frames = load_frames(video_path, [d["frame_idx"] for d in recs])
        for ax, d, frame in zip(row_axes, recs, frames):
            ax.imshow(frame)
            ax.set_title(f"#{d['frame_idx']}  blur {d['blur_score']:.0f} · mean {d['exposure_mean']:.0f}", fontsize=8)
        for ax in row_axes:
            ax.axis("off")
        # Row label survives axis("off") because it's a plain text artist
        row_axes[0].text(-0.06, 0.5, label, transform=row_axes[0].transAxes, rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle(f"Quality gate examples  ({counts})", fontsize=11)
    fig.tight_layout()
    plt.show()
```

- [ ] **Step 4: Run the viz tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -v`
Expected: all PASS. Also re-run `tests/preproc/test_sampling.py::test_importing_preproc_does_not_import_matplotlib` — must still PASS (viz stays out of `__init__`).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit -m "feat(preproc): plot_quality_examples viz for blur/exposure gate outcomes"
```

---

### Task 4: Rewrite `tutorial_config.py`

**Files:**
- Modify: `docs/source/tutorials/tutorial_config.py` (full rewrite)

No unit tests (docs helper script, mirrors existing untested file); verified by import.

- [ ] **Step 1: Replace file content**

```python
from pathlib import Path

import yaml

# ── Edit these two variables to point at your data ───────────────────────────
BASE_DIR   = Path("/workspace/outputs")
DATASET    = "2024_02_06/C0043"  # <session-date>/<video-stem>
MAX_FRAMES = 30

# ── Derived paths (do not edit) ───────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / DATASET
CACHE_DIR  = OUTPUT_DIR                    # alias — notebooks read canonical artifacts here
FRAMES     = OUTPUT_DIR / "frames"         # pipeline-written keyframes (read-only for tutorials)
TUTORIAL_CACHE = BASE_DIR / "tutorial_cache" / DATASET  # notebook scratch output (never synced)


def _infer_video_path(output_dir: Path) -> Path | None:
    """Resolve the scene's source video: run_config.yaml keys, then *.mp4 glob."""
    config_file = output_dir / "run_config.yaml"
    if config_file.exists():
        try:
            cfg = yaml.safe_load(config_file.read_text())
            # video_ref is rclone-relative — resolve its basename against the scene dir
            raw = cfg.get("video_path") or cfg.get("input_path") or cfg.get("video_ref")
            if raw:
                for candidate in (Path(raw), output_dir / Path(raw).name):
                    if candidate.exists():
                        return candidate
        except Exception:
            pass
    # Fallback: a video file sitting directly in the scene dir
    matches = sorted(output_dir.glob("*.MP4")) + sorted(output_dir.glob("*.mp4"))
    return matches[0] if matches else None


VIDEO_PATH = _infer_video_path(OUTPUT_DIR)
```

- [ ] **Step 2: Verify resolution against real data**

Run:
```bash
cd /workspace/collab-splats/docs/source/tutorials && /opt/venv/reconstruction/bin/python -c "
exec(open('tutorial_config.py').read())
print('FRAMES:', FRAMES, FRAMES.is_dir())
print('TUTORIAL_CACHE:', TUTORIAL_CACHE)
print('VIDEO_PATH:', VIDEO_PATH)
assert VIDEO_PATH is not None and VIDEO_PATH.name == 'C0043.MP4', VIDEO_PATH
assert FRAMES.is_dir()
"
```
Expected: prints paths; `VIDEO_PATH = /workspace/outputs/2024_02_06/C0043/C0043.MP4`; no assertion error. Also remove the stale bytecode: `rm -f docs/source/tutorials/__pycache__/tutorial_config.cpython-311.pyc`.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/tutorial_config.py
git commit -m "docs(tutorials): tutorial_config aligned to <date>/<stem> layout (FRAMES, TUTORIAL_CACHE, video_ref)"
```

---

### Task 5: Rework `keyframe_extraction.ipynb`

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` (use NotebookEdit; read the notebook first to get cell IDs — locate cells by the leading content quoted below)

- [ ] **Step 1: Update the imports cell** (currently starts `import json`)

```python
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from collab_splats.preproc import (
    extract_frames,
    get_video_info,
    load_frames,
    sample_frames,
    score_frames,
)
from collab_splats.preproc.viz import (
    plot_disparity_sensitivity,
    plot_frame_grid,
    plot_frame_scores,
    plot_quality_examples,
    plot_selection,
)
```

- [ ] **Step 2: Update the config cell** (currently `%run ../tutorial_config.py` + `FRAME_SCORES = CACHE_DIR / "frame_scores.json"`)

```python
%run ../tutorial_config.py

# ── Configuration ─────────────────────────────────────────────────────────────
# All notebook output goes under TUTORIAL_CACHE — the canonical scene dir
# (OUTPUT_DIR, incl. FRAMES) is pipeline-owned and read-only here.
FRAME_SCORES = TUTORIAL_CACHE / "frame_scores.json"
KEYFRAMES = TUTORIAL_CACHE / "keyframes"
```

- [ ] **Step 3: Update the §1 setup cell** (currently `# Ensure cache dirs exist` / `CACHE_DIR.mkdir...` / `IMAGES.mkdir...`)

```python
# Tutorial scratch dir; the canonical scene dir is never written to
TUTORIAL_CACHE.mkdir(parents=True, exist_ok=True)

# Load video metadata
info = get_video_info(VIDEO_PATH)
print(f"Video: {VIDEO_PATH}")
print(f"Scratch: {TUTORIAL_CACHE.resolve()}")
print(f"Frames: {info['total_frames']}  FPS: {info['fps']:.1f}  Duration: {info['duration_s']:.1f}s")
```

Also update the §1 markdown cell sentence "All subsequent cells read from `VIDEO_PATH` and write outputs under `CACHE_DIR`." → "All subsequent cells read from `VIDEO_PATH` and write outputs under `TUTORIAL_CACHE`."

- [ ] **Step 4: Update the §3 scoring cell** (currently `if FRAME_SCORES.exists():`) — adds stale-schema cache guard

```python
frame_scores = None
if FRAME_SCORES.exists():
    cached = json.loads(FRAME_SCORES.read_text())
    # Invalidate caches written before reject_reason existed
    if cached and "reject_reason" in cached[0]:
        frame_scores = cached
        print(f"Loaded frame scores from cache ({len(frame_scores)} frames)")
if frame_scores is None:
    frame_scores = score_frames(VIDEO_PATH)
    FRAME_SCORES.write_text(json.dumps(frame_scores))
    print(f"Scored frames → saved to {FRAME_SCORES}")

# Plot the three scoring signals over the video timeline
plot_frame_scores(frame_scores)
```

- [ ] **Step 5: Insert the quality-gate section after the §3 scoring cell** (three new cells, in order)

New markdown cell:

```markdown
### Quality Gate: Blur & Exposure

`score_frames` gates every frame before motion scoring: frames whose Laplacian variance falls below `blur_threshold` (default 50) are rejected as blurred; frames with mean intensity outside [20, 235] or standard deviation below 10 are rejected as badly exposed. Each record carries `reject_reason` (`None`, `"blur"`, or `"exposure"`) — rejected frames never become keyframes.

The histogram shows where the blur threshold cuts this video; the grid below it shows real examples of accepted and rejected frames.
```

New code cell:

```python
# Blur-score distribution; frames left of the threshold line are gate-rejected
blur_scores = [d["blur_score"] for d in frame_scores]
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(blur_scores, bins=60, color="steelblue", alpha=0.85)
ax.axvline(50.0, color="tomato", linestyle="--", label="blur threshold (50)")
ax.set_xlabel("Laplacian variance (blur score)")
ax.set_ylabel("Frames")
ax.legend()
plt.show()

# Gate outcome counts across the whole video
print(Counter(d["reject_reason"] or "accepted" for d in frame_scores))
```

New code cell:

```python
# Example frames per gate outcome — decodes only the displayed frames
plot_quality_examples(VIDEO_PATH, frame_scores)
```

- [ ] **Step 6: Rewrite the §4 selection cell** (currently `# Derive selected indices from pre-scored frames`)

```python
# Derive selected indices from pre-scored frames
of_indices = [d["frame_idx"] for d in frame_scores if d["selected"]][:MAX_FRAMES]

# Extract selected keyframes as frame_{idx:06d}.jpg into the tutorial scratch dir
extract_frames(VIDEO_PATH, of_indices, KEYFRAMES)
print(f"Extracted {len(of_indices)} OF keyframes → {KEYFRAMES}")

# Load OF frames by index from video (rotation applied automatically by the ffmpeg decode)
of_frames = load_frames(VIDEO_PATH, of_indices)
plot_selection(info['total_frames'], of_indices=of_indices)
```

Update the §4 markdown cell above it: replace "Any frames not already cached in `IMAGES/` are extracted and saved as `frame_{idx:06d}.jpg`, preserving the original video frame index." → "Selected keyframes are extracted to `KEYFRAMES/` as `frame_{idx:06d}.jpg`, preserving the original video frame index."

- [ ] **Step 7: Delete the duplicate extraction cell and its markdown**

Delete the markdown cell starting "Extract the selected keyframes from the video and write them to `IMAGES/`..." and the code cell starting `if not any(IMAGES.glob("*.jpg")):`. (Stale: writes FPS frames into the pipeline-owned dir and its cache guard silently no-ops when pipeline frames exist.)

- [ ] **Step 8: Fix the §7 footer markdown** (currently ends `**Default in `SplatterConfig`:** `frame_selection="fps"`...`)

Replace that final paragraph with:

```markdown
**Pipeline knob:** the dashboard pipeline exposes `RunConfig.sampling_method` (`collab_splats/dashboard/config.py`) with values `"balanced"` (uniform sharpest-in-window) and `"optical_flow"`. This notebook's `sample_frames` accepts `method="uniform"` or `"optical_flow"`.
```

(Before committing, confirm the "balanced"→uniform mapping with `grep -n "sampling_method" collab_splats/dashboard/pipeline.py`; adjust the parenthetical if it maps differently.)

- [ ] **Step 9: Execute the notebook end-to-end against C0043**

Full run decodes the source video ~3× (preview, FPS pass, score_frames) — expect minutes, run with a generous timeout (background if needed, per repo memory-cap rules):

```bash
touch /tmp/claude-0/-workspace-collab-splats/866d5112-2bcf-45f0-8918-602651e62a5f/scratchpad/nb_stamp
cd /workspace/collab-splats && /opt/venv/reconstruction/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
```

Expected: exits 0, all cells executed. The quality-gate cells must show a non-empty `Counter` and the example grid (at minimum the Accepted row; rejected rows appear only if C0043 contains gated frames — if it has none, that is a valid outcome, the plot title still reports counts).

- [ ] **Step 10: Verify the canonical scene dir is untouched**

```bash
find /workspace/outputs/2024_02_06/C0043 -newer /tmp/claude-0/-workspace-collab-splats/866d5112-2bcf-45f0-8918-602651e62a5f/scratchpad/nb_stamp | head
```
Expected: empty output (no file in the scene dir created/modified by the run). Scratch outputs exist instead: `ls /workspace/outputs/tutorial_cache/2024_02_06/C0043` → `frame_scores.json`, `keyframes/`.

- [ ] **Step 11: Commit**

```bash
git add docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit -m "docs(tutorials): keyframe tutorial — quality-gate section, scratch-dir writes, stale cells removed"
```

---

### Task 6: Format, full-suite check, graph update

- [ ] **Step 1: Format touched files**

```bash
cd /workspace/collab-splats && black collab_splats/preproc tests/preproc docs/source/tutorials/tutorial_config.py && isort collab_splats/preproc tests/preproc docs/source/tutorials/tutorial_config.py
```
If black/isort changed anything, re-run `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q` and commit as `style(preproc): black/isort pass`.

- [ ] **Step 2: Full preproc + adjacent suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/dashboard/ -q`
Expected: PASS (dashboard pipeline consumes `sample_frames`/`score_frames` — record-schema change must not break it). Any failure outside known-failures list (`docs/known-test-failures.md`) blocks completion.

- [ ] **Step 3: Update the knowledge graph**

```bash
cd /workspace && graphify update .
```

---

## Self-review notes

- Spec coverage: config rewrite (Task 4), preproc gate API (Task 1), record enrichment (Task 2), viz (Task 3), notebook rework incl. stale-cell removals + §7 fix + cache guard (Task 5), tests (Tasks 1-3), verification (Task 5 steps 9-10, Task 6). `load_frames` rotation claim verified true during planning — comment kept, no task needed. VGGT-Omega-primary + remaining notebooks: deferred to sweep (spec §Deferred), not in this plan.
- Sampler record schema deliberately unchanged (`test_optical_flow_records_have_source_indices` stays green); only `score_frames` records gain fields.
