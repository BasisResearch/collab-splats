# Seek-Based Uniform Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make uniform frame sampling seek to evenly-spaced positions (O(1), no full decode) and return exactly `min(max_frames, total)` frames.

**Architecture:** Split `extract_frame` into a probe-once `_seek_frame` helper + thin `extract_frame` wrapper. Rewrite `_sample_uniform` to build evenly-spaced target indices, seek each one, and keep every seeked frame (best-effort) so the count is exact. `_sample_optical_flow` and the dashboard are untouched.

**Tech Stack:** Python 3.11, numpy, cv2 (in-memory only), ffmpeg via subprocess, pytest.

Spec: `docs/superpowers/specs/2026-07-22-uniform-sampling-seek-design.md`

---

### Task 1: Probe-once seek helper

Extract the ffmpeg input-seek body of `extract_frame` into `_seek_frame` so per-seek callers don't re-spawn ffprobe. `extract_frame`'s public behavior is unchanged.

**Files:**
- Modify: `collab_splats/preproc/sampling.py:561-600` (`extract_frame`)
- Test: `tests/preproc/test_sampling.py` (existing `extract_frame` tests must still pass)

- [ ] **Step 1: Replace `extract_frame` with `_seek_frame` + wrapper**

Replace the whole current `extract_frame` (lines 561-600) with:

```python
def _seek_frame(
    video_path: str | Path, frame_idx: int, *, fps: float, w: int, h: int, total: int = 0
) -> np.ndarray:
    """Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    Caller supplies pre-probed fps/w/h/total so a batch of seeks probes the
    video only once. Seeks by timestamp (O(1) in frame depth).
    """
    _require_ffmpeg()
    if not fps or not w or not h:
        raise ValueError(f"cannot seek {video_path}: missing fps/width/height")
    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"_seek_frame: frame {frame_idx} out of range for {video_path}")
    # Seek to the frame midpoint, not its start: PTS float rounding can otherwise land
    # the demuxer just past the target timestamp and decode frame N+1 instead of N.
    seek_s = max(frame_idx - 0.5, 0) / fps
    # -ss before -i = input seek (demuxer-level); rawvideo pipe avoids a temp file.
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{seek_s:.6f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    raw = proc.stdout
    if len(raw) < w * h * 3:
        err = proc.stderr.decode(errors="replace")[-500:]
        raise ValueError(f"_seek_frame: frame {frame_idx} not found in {video_path}: {err}")
    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()


def extract_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    """Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    Exact on constant-frame-rate video; may land one frame off near keyframes
    on VFR sources.
    """
    info = get_video_info(str(video_path))
    return _seek_frame(
        video_path,
        frame_idx,
        fps=info["fps"],
        w=info["width"],
        h=info["height"],
        total=info["total_frames"],
    )
```

- [ ] **Step 2: Run existing extract_frame tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "extract_frame or seek" -q`
Expected: PASS (public `extract_frame` contract unchanged).

- [ ] **Step 3: Commit**

```bash
git add collab_splats/preproc/sampling.py
git commit -m "refactor(preproc): split _seek_frame probe-once helper from extract_frame"
```

---

### Task 2: Seek-based `_sample_uniform` with exact count

Rewrite `_sample_uniform` to seek evenly-spaced indices and keep every seeked frame (best-effort), giving an exact count.

**Files:**
- Modify: `collab_splats/preproc/sampling.py:400-448` (`_sample_uniform`)
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Update the failing/obsolete tests first**

In `tests/preproc/test_sampling.py`:

**Remove** `test_uniform_picks_sharpest_in_window` (lines 231-235) — sharpest-within-window selection is an intentional non-goal of seek-based sampling.

**Add** these tests (place near the other uniform tests):

```python
def test_uniform_derives_count_from_max_frames_exact(tiny_video):
    # fps omitted: 60-frame video, request 6 → exactly 6 evenly-spaced frames
    frames, records = sample_frames(tiny_video, method="uniform", max_frames=6)
    assert len(frames) == len(records) == 6
    assert set(records[0]) == {"frame_idx", "blur_score"}
    # indices are non-decreasing source-video indices within range
    idxs = [r["frame_idx"] for r in records]
    assert idxs == sorted(idxs) and idxs[0] >= 0 and idxs[-1] < 60


def test_uniform_best_effort_keeps_count_when_gate_rejects_all(tiny_video):
    # Impossible blur threshold → every frame fails the gate, but best-effort
    # keeps one per position, so the count still hits the target.
    frames, _ = sample_frames(
        tiny_video, method="uniform", max_frames=8, blur_threshold=1e12
    )
    assert len(frames) == 8


def test_uniform_seeks_target_count_not_whole_video(tiny_video, monkeypatch):
    # Seek-based: one _seek_frame call per requested frame, NOT one per source frame.
    import collab_splats.preproc.sampling as s

    calls = {"n": 0}
    real_seek = s._seek_frame

    def counting_seek(*args, **kwargs):
        calls["n"] += 1
        return real_seek(*args, **kwargs)

    monkeypatch.setattr(s, "_seek_frame", counting_seek)
    frames, _ = sample_frames(tiny_video, method="uniform", max_frames=10)
    assert len(frames) == 10
    assert calls["n"] == 10  # not 60 (the source frame count)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "exact or best_effort or seeks_target" -q`
Expected: FAIL (old `_sample_uniform` full-decodes; `_seek_frame` not called; count may differ).

- [ ] **Step 3: Rewrite `_sample_uniform`**

Replace the whole current `_sample_uniform` (lines 400-448) with:

```python
def _sample_uniform(
    video_path: str,
    *,
    fps: float | None,
    max_frames: int | None,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Seek evenly-spaced frames; keep one per position (best-effort, exact count)."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    native_fps, w, h = info["fps"] or 30.0, info["width"], info["height"]
    # Target indices: from fps (samples/second) if given, else spread max_frames
    # evenly over the video; fall back to 2.0 fps when neither is set.
    if fps is not None:
        step = max(1, int(round(native_fps / fps)))
        indices = list(range(0, total, step))
    elif max_frames:
        n = min(max_frames, total)
        indices = np.unique(np.linspace(0, total - 1, n).round().astype(int)).tolist()
    else:
        step = max(1, int(round(native_fps / 2.0)))
        indices = list(range(0, total, step))
    # max_frames is a hard cap in every mode (e.g. fps + max_frames together)
    if max_frames is not None:
        indices = indices[:max_frames]
    report, close = _progress_reporter(len(indices), "Uniform sampling", on_progress)
    frames: list[np.ndarray] = []
    records: list[dict] = []
    try:
        # Seek each target index (O(1) input-seek) and keep it — the quality gate
        # only annotates blur_score; best-effort keeps the frame so count is exact.
        for done, idx in enumerate(indices):
            frame = _seek_frame(video_path, idx, fps=native_fps, w=w, h=h, total=total)
            gray = _analysis_gray(frame[:, :, ::-1])  # _analysis_gray expects BGR
            blur = compute_blur_score(gray)
            frames.append(frame)
            records.append({"frame_idx": idx, "blur_score": blur})
            report(done + 1)
    finally:
        close()
    return frames, records
```

Note: `_seek_frame` returns RGB, so append directly (no cvtColor). `_analysis_gray` expects BGR, so pass `frame[:, :, ::-1]`. `check_frame_quality` is no longer called in uniform mode — best-effort keeps every frame; `blur_score` is still recorded. Verify `check_frame_quality` remains used by `_iter_scored_frames` (optical_flow) so it is not left unused — it is, so keep the import/def.

- [ ] **Step 4: Run all uniform tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k uniform -q`
Expected: PASS. `test_uniform_returns_frames_and_records` (fps=10 → 20), `test_uniform_respects_max_frames` (fps=10 + max_frames=5 → 5), `test_uniform_frames_are_rgb`, and the three new tests all pass.

- [ ] **Step 5: Run the full sampling suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -q`
Expected: PASS (optical_flow tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "fix(preproc): seek-based uniform sampling with exact frame count"
```

---

### Task 3: Update docstring + verify end-to-end

**Files:**
- Modify: `collab_splats/preproc/sampling.py:364-380` (`sample_frames` docstring)

- [ ] **Step 1: Update the `sample_frames` "uniform" docstring**

The current text says "one frame per fixed window — the sharpest usable frame in each". Replace the `"uniform"` bullet (lines 367-369) with:

```
        "uniform": one frame per evenly-spaced position via O(1) seeks —
            returns exactly max_frames frames (or fewer only when the video is
            shorter). Spacing comes from `fps` (samples/second), or from
            `max_frames` when fps is None (falls back to 2.0 fps). Frames are
            kept best-effort; blur_score is recorded but does not drop frames.
```

- [ ] **Step 2: Format**

Run: `black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py && isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py`
Expected: reformatted / no changes.

- [ ] **Step 3: Dashboard smoke gate (mandatory pre-commit)**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: prints `SMOKE PASS`.

- [ ] **Step 4: Full preproc suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py
git commit -m "docs(preproc): document seek-based uniform sampling behavior"
```

---

## Self-Review Notes

- **Spec coverage:** probe-once helper (Task 1), seek-based indices + best-effort exact count (Task 2), fps-mode cap preserved (Task 2 Step 3 `indices[:max_frames]`), docstring + smoke gate + format (Task 3). All spec sections covered.
- **Obsolete test:** `test_uniform_picks_sharpest_in_window` removed — sharpest-within-window is a declared non-goal.
- **Kept behaviors:** record keys `frame_idx`/`blur_score` unchanged (FrameStore + pipeline consumers); `check_frame_quality` still used by optical_flow path.
- **RGB/BGR:** `_seek_frame` returns RGB → appended directly; `_analysis_gray` fed `frame[:, :, ::-1]` (BGR).
